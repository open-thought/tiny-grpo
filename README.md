# Minimal GRPO implementation
Since I had a smaller 12 GB GPU, I tested this with smaller number of samples and an even smaller model of LLM instruct than originally proposed.
Goal: Working toy implementation of HuggingFaceTB/SmolLM-135M-Instruct locally RL training with GRPO. Understanding the algorithm & hyper parameters. Just running everything locally on a single node.

### Setup

1. Create conda env

```
conda create --name grpo python=3.12 -y
conda activate grpo
```

2. Install dependencies

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

#May need to upgrade nvcc--version to higher for flash-attn to work
```

3. Play with the source in `train_ds2.py`
Since I had only one 12 GB 3060 GPU, I modified the code to run on single GPU instead of distributed 
```
python train_ds2.py

```

with multiple gpu

```
torchrun --nproc_per_node=8 train.py
```

### Inspiration
https://github.com/open-thought/tiny-grpo
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)


### References

- [DeepSeek-R1 tech report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
