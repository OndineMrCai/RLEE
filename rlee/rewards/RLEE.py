from typing import List, Union
from rlee.rewards.rewards_type import RewardType, RLEEConfig, RLEERewardInput, RLEERewardOutput, RLEERewardFn
from rlee.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from rlee.rewards.math_reward import deepscaler_reward_fn
import re
from typing import Dict, Tuple, Optional
from collections import defaultdict
import torch

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True
    
    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

class RLEERewardMathFn(RLEERewardFn):
    """
    Reward function for evaluating mathematical answers in RLEE project.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RLEERewardInput) -> RLEERewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        response_length = input.model_response_length
        branch = input.branch
        branch_length = input.branch_length
        zero_exploration_reward = [0 for _ in branch]
        print(f"Model Response: {model_response}")
        for i in range(len(branch)):
            print(f"Branch {i}: {branch[i]}")

        model_solution = model_response
        model_answer,processed_str = extract_solution(model_solution)
        format_correct = validate_response_structure(processed_str)
        if not format_correct:
            return RLEERewardOutput(answer_reward=-2, format_reward=-1, exploration_reward=zero_exploration_reward, is_correct=False)
        if model_answer is None:
            return RLEERewardOutput(answer_reward=-2, format_reward=-1, exploration_reward=zero_exploration_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        #print("ground_truths: ", ground_truths)
        if ground_truths is None:
            return RLEERewardOutput(answer_reward=-2, format_reward=-1, exploration_reward=zero_exploration_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        #print("processed_ground_truths: ", processed_ground_truths)
        if not processed_ground_truths:
            return RLEERewardOutput(answer_reward=-2, format_reward=-1, exploration_reward=zero_exploration_reward, is_correct=False)
        
        branch_correctness = []
        for i in range(len(branch)):
            sub_branch_correctness = []
            for j in range(len(branch[i])):
                branch_answer, branch_reponse = extract_solution(branch[i][j])
                is_correct_branch = False
                if branch_answer is None:
                    sub_branch_correctness.append(False)
                    continue
                # Check against all possible correct answers
                for ground_truth in processed_ground_truths:
                    is_correct = grade_answer_mathd(branch_answer, ground_truth) or grade_answer_sympy(branch_answer, ground_truth)
                    if is_correct:
                        is_correct_branch = True
                        sub_branch_correctness.append(True)
                        break
                if not is_correct_branch:
                    sub_branch_correctness.append(False)
            branch_correctness.append(sub_branch_correctness)
            
        branch_averages_length = [sum(sublist) / len(sublist) for sublist in branch_length]
        answer_correct = False
        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                answer_correct = True
                break
        
        exploration_reward_list = []
        correct_counts = [sum(value for value in sub_branch) for sub_branch in branch_correctness]
        if answer_correct:
            for i in range(len(branch_correctness)):
                if correct_counts[i] == len(branch_correctness[i]):
                    if branch_averages_length[i] < response_length:
                        exploration_reward = -0.3
                else:
                    exploration_reward = (len(branch_correctness[i]) - correct_counts[i]) / len(branch_correctness[i]) * 0.3
                exploration_reward_list.append(exploration_reward)
        else:
            for i in range(len(branch_correctness)):
                if correct_counts[i] > 0:
                    exploration_reward = (len(branch_correctness[i]) - correct_counts[i]) / len(branch_correctness[i]) * -0.3
                else:
                    if branch_averages_length[i] < response_length:
                        exploration_reward = -0.1
                    else:
                        exploration_reward = 0.1
                exploration_reward_list.append(exploration_reward)
        if answer_correct:
            return RLEERewardOutput(answer_reward=2, format_reward=1, exploration_reward=exploration_reward_list, is_correct=True)
        else:
            return RLEERewardOutput(answer_reward=-1.5, format_reward=1, exploration_reward=exploration_reward_list, is_correct=False)
        
def compute_score(solution_str, solution_length, branch, branch_length, ground_truth, enable_llm=False, rlee = True):        
    if branch is None or branch_length is None:
        answer_score, format_score = deepscaler_reward_fn(solution_str, ground_truth, enable_llm)
        return answer_score, format_score, [0]

    reward_config = RLEEConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RLEERewardMathFn(reward_config)
    RewardOutput = reward_fn(RLEERewardInput(problem=solution_str, model_response=solution_str, model_response_length=solution_length, 
                                                 branch=branch, branch_length=branch_length, problem_type=RewardType.MATH, ground_truth={"answer": ground_truth}))
    return RewardOutput.answer_reward, RewardOutput.format_reward, RewardOutput.exploration_reward