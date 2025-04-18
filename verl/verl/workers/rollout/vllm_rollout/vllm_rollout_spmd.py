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
import numpy as np
from typing import Optional, List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length, pad_2d_list_to_length_left
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

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


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

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


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

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
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

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
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

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

        non_tensor_batch = gen_batch.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, prompts[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')
        
        if 'multi_modal_data' in non_tensor_batch:
            raise NotImplementedError('multi_modal_data is not supported in generate_branch')
        
        idx_list = []
        exploration_position_mask = []
        exploration_sample_index = []
        for i in range(batch_size):
            if 'raw_prompt_ids' not in non_tensor_batch:
                raw_idx = _pre_process_inputs(self.pad_token_id, prompts[i])
            else:
                raw_idx = list(non_tensor_batch['raw_prompt_ids'][i])
            response = _pre_process_responses(self.pad_token_id, responses[i])
    
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
        
        vllm_inputs = [{
            'prompt_token_ids': raw_prompt_ids
        } for raw_prompt_ids in idx_list]

        batch_size = len(idx_list)
    
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            new_response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(device)
            
            new_prompt = pad_2d_list_to_length_left(idx_list, self.pad_token_id,
                                            max_length=self.config.prompt_length).to(device)
            new_attention_mask = (new_prompt != self.pad_token_id).to(dtype=attention_mask.dtype, device=device)
            new_position_ids = (new_attention_mask.cumsum(dim=1) - 1).clamp(min=0).to(device)

            if self.sampling_params.n > 1:
                new_prompt = _repeat_interleave(new_prompt, self.sampling_params.n)
                new_attention_mask = _repeat_interleave(new_attention_mask, self.sampling_params.n)
                new_position_ids = _repeat_interleave(new_position_ids, self.sampling_params.n)
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
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)