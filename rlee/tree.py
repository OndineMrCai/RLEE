from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

class Node:
    def __init__(self, parent=None, text="", token_usage=0):
        """
        Represents a node in the generation tree.
        
        :param parent: The parent node.
        :param text: The text generated up to this node.
        """
        self.parent = parent
        self.text = text
        self.children = []
        self.token_usage = token_usage

    def add_child(self, child_text, token_num):
        """
        Creates a new child node with the given text and appends it to the children list.
        
        :param child_text: The text for the new child node.
        :return: The created child node.
        """
        child = Node(parent=self, text=child_text, token_usage= token_num)
        self.children.append(child)
        return child

class BinaryTree:
    def __init__(self, 
                 root_prompt: str, 
                 model: LLM, 
                 tokenizer: AutoTokenizer, 
                 max_tokens: int = 8192,
                 exploration_tokens: list[str] = []):
        """
        Represents the complete generation tree for exploration-exploitation branching.
        The stop tokens used for sampling include both the exploration tokens and the EOS token.
        
        :param root_prompt: The initial prompt for the model.
        :param model: The vLLM model instance.
        :param tokenizer: The associated tokenizer.
        :param exploration_tokens: A list of tokens that trigger exploration (e.g., "wait", "alternatively").
        """
        self.tokenizer = tokenizer
        self.model = model
        self.root = Node(text=root_prompt)
        # Combine exploration tokens with the EOS token for stopping
        all_stop_tokens = exploration_tokens + [tokenizer.eos_token]
        # Convert the stop tokens to token IDs
        self.stop_token_ids = self.tokenizer(all_stop_tokens, return_tensors='pt')["input_ids"]
        
        self.exploration_tokens = exploration_tokens  # Keep the list for later string matching
        self.eos_token = tokenizer.eos_token
        self.max_tokens = max_tokens
        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,  
            stop_token_ids=self.stop_token_ids,  # Stop when any exploration token or EOS is generated
            temperature=0.6,
            skip_special_tokens=False
        )

    def generate(self, node: Node):
        """
        Generates text from the LLM starting from the given node's text.
        The generation stops when any stop token is encountered.
        If the stop token is an exploration token (and not the EOS token), two branches are created:
          - One branch keeps the exploration token (exploration branch).
          - One branch removes the exploration token (exploitation branch).
        Generation continues recursively until the branch terminates with the EOS token.
        
        :param node: The node from which to continue text generation.
        """
        # Encode the current node text
        # Generate output using the vLLM model; note that vLLM's generate returns an object
        # where the generated string is found in o[0].outputs[0].text.
        if node.text == self.root.text:
            prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + node.text + "<|im_end|>\n<|im_start|>assistant\n<|im_start|>"
            output = self.model.generate(prompt, self.sampling_params)
        else:
            output = self.model.generate(node.text, self.sampling_params)
        generated_text = output[0].outputs[0].text  # Get the generated string
        
        # Remove any extra whitespace
        generated_text = generated_text.strip()
        
        # Check if the generated text ends with the EOS token
        if generated_text.endswith(self.eos_token):
            # Terminal branch: append the generated text and stop recursion.
            node.add_child(generated_text)
            return
        
        # Check for any exploration token at the end of the generated text
        exploration_found = None
        for token in self.exploration_tokens:
            # Here we simply check if the generated text ends with the exploration token.
            if generated_text.endswith(token):
                exploration_found = token
                break
        
        if exploration_found is not None:
            # Create two branches at this exploration stop point.
            # Exploration branch: keep the exploration token.
            exploration_branch = node.add_child(generated_text)
            # Exploitation branch: remove the exploration token from the end.
            exploitation_text = generated_text[:-len(exploration_found)].strip()
            exploitation_branch = node.add_child(exploitation_text)
            
            # Recursively generate for both branches.
            self.generate(exploration_branch)
            self.generate(exploitation_branch)
            return
        
        # If no stop token is detected (which ideally should not happen because sampling stops at one),
        # continue generation by appending the generated text and recursively generating further.
        next_branch = node.add_child(generated_text)
        self.generate(next_branch)

    def get_all_branches(self, node=None, path=None):
        """
        Recursively traverses the generation tree to collect all complete reasoning paths.
        
        :param node: The current node being traversed.
        :param path: The accumulated text along the current path.
        :return: A list of complete generated paths.
        """
        if node is None:
            node = self.root
        if path is None:
            path = []
        
        path.append(node.text)
        
        # If the node has no children, it's a leaf: return the complete path.
        if not node.children:
            return [" ".join(path)]
        
        # Otherwise, traverse each child.
        paths = []
        for child in node.children:
            paths.extend(self.get_all_branches(child, path.copy()))
        return paths

"""unit test"""
if __name__ == "__main__":
    # Load vLLM model and tokenizer (ensure you have the correct model name)
    model = LLM("/scratch/sccai/reasoning_model/DeepScaleR-1.5B-Preview")
    tokenizer = AutoTokenizer.from_pretrained("/scratch/sccai/reasoning_model/DeepScaleR-1.5B-Preview")
    
    # Define exploration tokens
    exploration_words = ["wait", "alternatively", "but", "let's check"]
    
    # Create a BinaryTree instance with the initial prompt
    tree = BinaryTree(root_prompt="Solve x^2 - 4 = 0", model=model, tokenizer=tokenizer, exploration_tokens=exploration_words)
    
    # Start text generation from the root prompt
    tree.generate(tree.root)
    
    # Retrieve and print all generated reasoning paths
    all_paths = tree.get_all_branches()
    for path in all_paths:
        print("Generated Path:", path)
