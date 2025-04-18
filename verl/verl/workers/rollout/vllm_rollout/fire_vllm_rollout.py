# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import Optional, List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length, pad_2d_list_to_length_left
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.vllm_rollout.vllm_rollout import vLLMRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _pre_process_responses(pad_token_id, response_token_ids: torch.Tensor) -> List[int]:
    non_pad_indices = torch.nonzero(response_token_ids != pad_token_id, as_tuple=False)
    last_non_pad_index = non_pad_indices[-1][0] 
    token_ids = response_token_ids[:last_non_pad_index+1].tolist()
    return token_ids

def find_token_positions_in_response(
    response: List[int],
    exploration_token_ids: List[List[torch.Tensor]]
) -> List[int]:
    token_positions = []

    for token_tensor_list in exploration_token_ids:
        token_ids = token_tensor_list[0].tolist()
        token_len = len(token_ids)
        found = False
        for i in range(len(response) - token_len + 1):
            if response[i:i + token_len] == token_ids:
                token_positions.append(i) 
                found = True
                break
        if not found:
            pass

    return token_positions

class FirstTokenForbiddenProcessor:
    def __init__(self, exploration_token_ids: List[torch.Tensor]):
        self.forbidden_token_ids = {
            t.view(-1)[0].item() for t in exploration_token_ids if t.numel() > 0
        }

    def __call__(self, past_tokens, logits: torch.Tensor) -> torch.Tensor:
        is_first_step = (
            isinstance(past_tokens, tuple) and
            len(past_tokens) > 0 and
            isinstance(past_tokens[0], torch.Tensor) and
            past_tokens[0].numel() == 0
        ) or (
            isinstance(past_tokens, tuple) and len(past_tokens) == 0
        )

        if is_first_step:
            for token_id in self.forbidden_token_ids:
                logits[token_id] = -float("inf")

        return logits

    def __repr__(self):
        return f"FirstTokenForbiddenProcessor(forbidden_token_ids={list(self.forbidden_token_ids)})"

    def clone(self):
        return FirstTokenForbiddenProcessor([
            torch.tensor([token_id]) for token_id in self.forbidden_token_ids
        ])
    
class FIREvLLMRollout(vLLMRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__(actor_module, config, tokenizer, model_hf_config, **kwargs)

        self.use_fire_sampling = config.get('use_fire_sampling', False)
        if self.use_fire_sampling:
            kwargs_0 = kwargs.copy()
            kwargs_0['temperature'] = 30
            kwargs_0['max_tokens'] = 1
            if 'top_k' not in kwargs_0 or kwargs_0['top_k'] <= 0:
                kwargs_0['top_k'] = 16
            self.sampling_params.max_tokens = config.response_length - 1
            for k in config.keys():
                if hasattr(SamplingParams(), str(k)):
                    kwargs_0[k] = config.get(k)
            self.sampling_params_0 = SamplingParams(**kwargs_0)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        if self.use_fire_sampling:
            old_sampling_params_args_0 = {}
            if kwargs:
                for key, value in kwargs.items():
                    if hasattr(self.sampling_params_0, key):
                        old_value = getattr(self.sampling_params_0, key)
                        old_sampling_params_args_0[key] = old_value
                        setattr(self.sampling_params_0, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
        if self.use_fire_sampling:
            for key, value in old_sampling_params_args_0.items():
                setattr(self.sampling_params_0, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        if not self.use_fire_sampling:
            # users can customize different sampling_params at different run
            with self.update_sampling_params(**kwargs):
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=idx_list,
                    use_tqdm=False)

            response = output[0].to(idx.device)  # (bs, response_length)
            log_probs = output[1].to(idx.device)  # (bs, response_length)
        else:
            with self.update_sampling_params(**kwargs):
                output_0 = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params_0,
                    prompt_token_ids=idx_list,
                    use_tqdm=False)
                new_idx_list = []
                for i in range(batch_size):
                    new_idx_list.append(idx_list[i] + output_0[0][i].tolist())
                output = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=self.sampling_params,
                    prompt_token_ids=new_idx_list,
                    use_tqdm=False)

            response = torch.cat([output_0[0], output[0]], dim=1).to(idx.device)  # (bs, response_length)
            # log_probs = torch.cat([output_0[1], output[1]], dim=1).to(idx.device)  # (bs, response_length)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

    @torch.no_grad()
    def generate_branch(self, gen_batch: DataProto, exploration_token: List[torch.tensor], **kwargs) -> Optional[DataProto]:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        prompts = gen_batch.batch['prompts']  # (bs, prompt_length)
        responses = gen_batch.batch['responses']  # (bs, response_length)
        attention_mask = gen_batch.batch['attention_mask']
        position_ids = gen_batch.batch['position_ids']
        batch_size = responses.size(0)
        device = responses.device
        # used to construct attention_mask
        eos_token_id = gen_batch.meta_info['eos_token_id']

        forbidden_processor = FirstTokenForbiddenProcessor(exploration_token_ids=exploration_token)

        kwargs = {
                'top_k': -1,
                'top_p': 0.95,
                'temperature': 0.6,
                'n': 3,
                'logits_processors': [forbidden_processor]
            }
        
        idx_list = []
        exploration_position_mask = []
        exploration_sample_index = []
        for i in range(batch_size):
            response = _pre_process_responses(self.pad_token_id,responses[i])
            raw_idx = _pre_process_inputs(self.pad_token_id,prompts[i])
    
            all_token_positions = find_token_positions_in_response(
                response,
                exploration_token
            )
            all_token_positions = sorted(set(all_token_positions))
            exploration_positions = []
            for local_j, pos in enumerate(all_token_positions):
                exploration_position_line = torch.zeros(1, responses.size(1), device=device)
                exploration_position_line[0, pos] = 1
                exploration_positions.append(exploration_position_line)
                new_prompt = raw_idx + response[:pos]
                idx_list.append(new_prompt)
                exploration_sample_index.append([i, local_j])
    
            if exploration_positions:
                exploration_position_mask_data = torch.cat(exploration_positions, dim=0)
                exploration_position_mask.append(exploration_position_mask_data)

        if exploration_position_mask:
            exploration_position_mask = torch.cat(exploration_position_mask, dim=0)
        
        exploration_sample_index = torch.tensor(exploration_sample_index, dtype=torch.float32, device=device)

        if not idx_list:
            return None
        
        batch_size = len(idx_list)

        with self.update_sampling_params(**kwargs):
            new_outputs = self.inference_engine.generate(
                prompts=None,
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False
            )

            new_response = new_outputs[0].to(device)  # (bs, response_length)

            if new_response.shape[1] < self.config.response_length:
                new_response = pad_sequence_to_length(new_response, self.pad_token_id,
                                            max_length=self.config.response_length).to(device)
            
            # TODO: align the new prompt & attention mask & position ids
            new_prompt = pad_2d_list_to_length_left(idx_list, self.pad_token_id,
                                            max_length=self.config.prompt_length).to(device)
            new_attention_mask = (new_prompt != self.pad_token_id).to(dtype=attention_mask.dtype, device=device)
            new_position_ids = (new_attention_mask.cumsum(dim=1) - 1).clamp(min=0).to(device)

            if self.sampling_params.n > 1:
                new_prompt = new_prompt.repeat_interleave(self.sampling_params.n, dim=0)
                new_attention_mask = new_attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                new_position_ids = new_position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
                exploration_position_mask = exploration_position_mask.repeat_interleave(self.sampling_params.n, dim=0) 
                exploration_sample_index = exploration_sample_index.repeat_interleave(self.sampling_params.n, dim=0)
            seq = torch.cat([new_prompt, new_response], dim=-1)
        
        response_length = new_response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=new_position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = new_position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([new_position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=new_response, eos_token=eos_token_id, dtype=new_attention_mask.dtype)
        attention_mask = torch.cat((new_attention_mask, response_attention_mask), dim=-1)

        # build new batch
        batch = TensorDict(
            {   
                'branch_prompts': new_prompt,
                'branch_responses': new_response,
                'branch_input_ids': seq, 
                'exploration_position_mask': exploration_position_mask,
                'branch_attention_mask': attention_mask,
                'branch_position_ids': position_ids,
                'exploration_sample_index': exploration_sample_index
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)