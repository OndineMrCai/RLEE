<div align="center">

# RLEE

**RLEE**: Reinforcement Learaning with Exploration and Exploitation

</div>

## Overview
We introduce **RLEE**, which attempts to mitigate the overthinking phenomenon in large reasoning models by rewarding or penalizing the model's **exploration** and **exploitation** actions during the reasoning process. Here, **exploration** actions refer to reasoning steps involving reflection, verification, or trying new ideas, while **exploitation** actions refer to continuing the reasoning using existing methods and the current intermediate results.

## Getting Started 🚀

### Installation
To begin working with **RLEE** for the ORZMath dataset, just run:
```
git clone https://github.com/OndineMrCai/RLEE.git
cd RLEE

# Install verl and rlee
pip3 install -e . verl/
pip3 install -e .

# Install the latest stable version of vLLM
pip3 install vllm==0.8.2

# Install flash-attn and tensor dict
pip3 install flash-attn --no-build-isolation
pip3 install tensordict==0.6.2
```
Note: please remove deepspeed, otherwise the verl will crash

### Training
We directly use the training data from [Open-Zero-Reasoner](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) at rlee/data/train directory. Please remember to change the save path in the bash shell.

Train RLEE
```
cd RLEE
bash scrpits/train/train.sh
```


## Acknowledgements
- Our training framework is built on [Short-RL](https://github.com/lblankl/Short-RL)，[deepscaler](https://github.com/agentica-project/deepscaler), [verl](https://github.com/volcengine/verl) and [ray](https://github.com/ray-project/ray).
- Our model is based on [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- Our math data is from [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)