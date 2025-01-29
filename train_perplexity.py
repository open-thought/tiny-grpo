import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ckpt_utils import save_checkpoint
from loss import GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
        model_name_or_path: str,
        trust_remote_code: bool = False,
        bf16: bool = True,
) -> tuple:
    """Load model and tokenizer on a single GPU."""
    from transformers import AutoTokenizer, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
    ).to("cuda")  # Ensure model is loaded on GPU
    return model, tokenizer


def main():
    # Configuration
    seed = 42
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = "./output"
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    clip_eps = 0.2

    # Rollout parameters
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Load model and tokenizer (single GPU)
    model, tokenizer = load_model(model_name)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare data loader (single process)
    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128 and x["num_terms"] <= 3 and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )

    print(f"Found {len(prompts)} matching prompts")

    prompt_loader = DataLoader(
        prompts,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps)

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
                    num_rollouts=1,  # Single GPU, process one rollout at a time
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )
                print(f"Rollout q='{q}', a='{a}', returns={returns.sum().item():.2f}")
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                attention_mask = sequence_ids != tokenizer.eos_token_id

                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask.to("cuda"),
                )
                replay_buffer.append(experience.to("cpu"))

        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"Returns of step {k}: {episode_return_sum:.4f}")

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(1):  # Single epoch per step for simplicity
            model.train()

            for i, exp in enumerate(experience_sampler):
                exp = exp.to("cuda")

                optimizer.zero_grad()
                log_probs = sequences_log_probs(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, _ = objective.forward(log_probs=log_probs, experience=exp)
                if not loss.isfinite():
                    print(f"Loss not finite, skipping backward: {loss}")
                    continue

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                print(f"Step {i}: Loss={loss.item():.4f}, Grad Norm={grad_norm:.4f}")
                optimizer.step()

        # Save checkpoint periodically
        if checkpoint_path is not None and checkpoint_interval is not None and (k + 1) % checkpoint_interval == 0:
            print(f"Saving checkpoint to {checkpoint_path}/step_{k}.pt")
            save_checkpoint(model, optimizer, f"{checkpoint_path}/step_{k}.pt")


if __name__ == "__main__":
    main()
