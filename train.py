from collections.abc import Callable
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


from transformers.models.llama.modeling_llama import LlamaDecoderLayer
def setup_dist():
    """Initialize process group and create device mesh"""
    dist.init_process_group("nccl")
    
    # Get local world size and rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
def load_model_fsdp(
    model_name_or_path: str,
    reduce_fp32: bool = False,
    reshard_after_forward: bool = True,
    trust_remote_code: bool = False,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    """Load model and apply composable FSDP"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        # torch_dtype=torch.bfloat16,
    ).to(dist.get_rank())
    
    # Define mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32 if reduce_fp32 else None
    )

    # Apply FSDP to transformer layers
    for layer_id, transformer_block in enumerate(model.model.layers):
        # Reshard all layers except the last one if enabled
        should_reshard = reshard_after_forward and layer_id < len(model.model.layers) - 1
        
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            reshard_after_forward=should_reshard,
        )
    
    # Apply FSDP to the whole model
    fully_shard(
        model,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    
    return model, tokenizer

def main():
    seed = 42
    wandb_project = None  # "tiny_grpo"
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    # FSDP specific configs
    reduce_fp32 = False
    reshard_after_forward = False

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    # Initialize distributed setup
    setup_dist()
    init_rng(seed)

    # Load models with composable FSDP
    reference_model, _ = load_model_fsdp(
        model_name, 
        reduce_fp32=reduce_fp32,
        reshard_after_forward=reshard_after_forward
    )
    model, tokenizer = load_model_fsdp(
        model_name,
        reduce_fp32=reduce_fp32,
        reshard_after_forward=reshard_after_forward
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    reference_model.eval()
    model.train()
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    
    if dist.get_rank() == 0:
        print(f"found {len(prompts)} matching prompts")
    
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    # Initialize wandb only on rank 0
    if dist.get_rank() == 0:
        if wandb_project is None:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                if dist.get_rank() == 0:
                    print(
                        f"rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}, "
                        f"replay_buffer_size={len(replay_buffer)}, sequence_ids={sequence_ids.shape}"
                    )
                
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != pad_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                log_probs_ref = sequences_log_probs(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                kl = approx_kl_divergence(
                    log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    action_mask=action_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=kl,
                )
                replay_buffer.append(experience.to("cpu"))

        if dist.get_rank() == 0:
            episode_return_sum = torch.stack(rollout_returns).sum()
            print(f"returns of step {k}: {episode_return_sum:.4f}")
            wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp = exp.to(dist.get_rank())
                
                optimizer.zero_grad()

                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, kl = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    if dist.get_rank() == 0:
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"experience.advantages={experience.advantages}")
                        print(f"Loss not finite, skipping backward, loss={loss}")
                        print(f"experience.advantages={experience.advantages}")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                
                if dist.get_rank() == 0:
                    grad_norm_full = grad_norm.full_tensor()
                    print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm_full: .4f}")
                    wandb.log({"kl": kl, "grad_norm": grad_norm_full})

                optimizer.step()

        # Save checkpoint only on rank 0
        if (
            dist.get_rank() == 0
            and checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            torch.save(model.state_dict(), checkpoint_path / f"step_{k}.pt")

    # Final save on rank 0
    if checkpoint_path is not None and dist.get_rank() == 0:
        torch.save(model.state_dict(), checkpoint_path / f"step_{k}.pt")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
