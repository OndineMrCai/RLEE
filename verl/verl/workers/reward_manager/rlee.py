from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict
from rlee.rewards.math_reward import deepscaler_reward_fn
from rlee.rewards import RLEE
from verl.utils.reward_score import gsm8k, math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

def _select_rm_score_fn(data_source,config):
    if config.trainer.reward_type == 'default':
        if data_source == 'openai/gsm8k':
            return gsm8k.compute_score
        elif data_source == 'lighteval/MATH':
            return math.compute_score
        else:
            return deepscaler_reward_fn
    elif config.trainer.reward_type == 'ShortRL':
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

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        

        already_print_data_sources = {}
        data_source = data[0].non_tensor_batch['data_source']
        compute_score_fn = _select_rm_score_fn(data_source,self.config)
        exploration_sample_index = data.batch['exploration_sample_index']
        sample_ids = exploration_sample_index[:, 0]
        counts = torch.bincount(sample_ids, minlength=len(data))
        reward_tensor = torch.zeros_like((len(counts), data.batch['responses'].shape[-1]), dtype=torch.float32)
        idx = 0
        sample_index = 0
        while idx < len(data):
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

            branch_num = counts[sample_index]
            branch = []
            branch_length = []
            for i in range(branch_num):
                branch_data_item = data[idx + i]
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
            
            reward_tensor[sample_index, valid_response_length - 1] = answer_reward + format_reward
            for j in range(len(exploration_reward)):
                reward_tensor[sample_index, exploration_sample_index[idx + j,1]] = exploration_reward[j]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

            idx = idx + counts[sample_index]
            sample_index = sample_index + 1

        return reward_tensor
