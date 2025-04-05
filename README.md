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
cd RLEE
pip3 install -e. /verl
pip3 install -e.
```


### Training
We directly use the training data from [Open-Zero-Reasoner](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) at rlee/data/train directory.

Train RLEE
```
cd RLEE
bash scrpits/train/train.sh
```


## Acknowledge
- Our training framework is built on [Short-RL](https://github.com/lblankl/Short-RL)，[deepscaler](https://github.com/agentica-project/deepscaler), [verl](https://github.com/volcengine/verl) and [ray](https://github.com/ray-project/ray).
- Our model is based on [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- Our math data is from [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero)