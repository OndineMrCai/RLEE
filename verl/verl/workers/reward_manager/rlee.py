from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
from rlee.rewards.math_reward import deepscaler_reward_fn
from rlee.rewards import RLEE
from verl.utils.reward_score import gsm8k, math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

def _select_rm_score_fn(data_source,config):
    if config.trainer.reward_type == 'default':
        if data_source == 'openai/gsm8k':
            return gsm8k.compute_score
        elif data_source == 'lighteval/MATH':
            return math.compute_score
        else:
            return deepscaler_reward_fn
    elif config.trainer.reward_type == 'rlee':
        return RLEE.compute_score
    else:
        raise NotImplementedError

class RleeRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.config=config

    def __call__(self, data: DataProto, gen_branch: Optional[DataProto] = None, return_dict: bool = True):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']
            

        already_print_data_sources = {}
        data_source = data[0].non_tensor_batch['data_source']
        compute_score_fn = _select_rm_score_fn(data_source,self.config)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        exploration_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        if gen_branch is None:
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)
                ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                data_source = data_item.non_tensor_batch['data_source']

                answer_reward, format_reward, exploration_reward = compute_score_fn(solution_str=sequences_str,
                                                                                solution_length=valid_response_length,
                                                                                branch=None,
                                                                                branch_length=None,
                                                                                ground_truth=ground_truth)
                
                reward_tensor[i, valid_response_length - 1] = answer_reward + format_reward

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[sequences]:", sequences_str)
                    print("[ground_truth]:", ground_truth)
                    print("[exploration_reward]: 0")
                    print("[score]:", str(answer_reward + format_reward))

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "exploration_reward_tensor": exploration_reward_tensor,
                }
            else:
                return reward_tensor



        exploration_sample_index = gen_branch.batch['exploration_sample_index']
        sample_ids = exploration_sample_index[:, 0]
        counts = torch.bincount(sample_ids, minlength=len(data))
        branch_idx = 0

        for idx in range(len(data)):
            data_item = data[idx]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            branch_num = counts[idx]
            branch = []
            branch_length = []
            for j in range(branch_num):
                branch_data_item = gen_branch[branch_idx + j]
                branch_prompt_ids = branch_data_item.batch['branch_prompts']
                branch_prompt_length = branch_prompt_ids.shape[-1]
                valid_branch_prompt_length = branch_data_item.batch['attention_mask'][:branch_prompt_length].sum()
                valid_branch_prompt_ids = branch_prompt_ids[-valid_branch_prompt_length:]
                branch_response_ids = branch_data_item.batch['responses']
                valid_branch_response_length = branch_data_item.batch['attention_mask'][branch_prompt_length:].sum()
                valid_branch_response_ids = branch_response_ids[:valid_branch_response_length]
                branch_sequences = torch.cat((valid_branch_prompt_ids, valid_branch_response_ids))
                branch_sequences_str = self.tokenizer.decode(branch_sequences)
                branch_length.append(valid_branch_response_length)
                branch.append(branch_sequences_str)

            grouped_branch = [branch[i:i + 3] for i in range(0, len(branch), 3)]
            grouped_branch_length = [branch_length[i:i + 3] for i in range(0, len(branch_length), 3)]

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']
            answer_reward, format_reward, exploration_reward = compute_score_fn(solution_str=sequences_str,
                                                                                solution_length=valid_response_length,
                                                                                branch=grouped_branch,
                                                                                branch_length=grouped_branch_length,
                                                                                ground_truth=ground_truth)
            
            reward_tensor[idx, valid_response_length - 1] = answer_reward + format_reward
            for j in range(len(exploration_reward)):
                reward_tensor[idx, exploration_sample_index[branch_idx + j,1]] = exploration_reward[j]
                exploration_reward_tensor[idx, exploration_sample_index[branch_idx + j,1]] = exploration_reward[j]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[sequences]", sequences_str)
                print("[ground_truth]", ground_truth)
                print("[exploration_reward]", str(exploration_reward))
                print("[score]", str(answer_reward + format_reward))
                
            branch_idx = branch_idx + counts[idx]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "exploration_reward_tensor": exploration_reward_tensor,
            }
        else:
            return reward_tensor
