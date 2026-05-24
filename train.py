import asyncio
import re
import warnings

warnings.filterwarnings("ignore", message="IProgress not found")

import datasets
import matplotlib.pyplot as plt
import tinker
import torch
from tinker import TensorData
from tinker_cookbook.renderers import get_renderer, get_text_content


def extract_boxed(text: str) -> str | None:
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match[-1].strip()
    return None


def grade_answer(response: str, ground_truth: str) -> float:
    answer = extract_boxed(response)
    if answer is None:
        return 0.0
    answer = answer.replace(",", "").strip()
    ground_truth = ground_truth.replace(",", "").strip()
    return 1.0 if answer == ground_truth else 0.0


def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    raise ValueError("No #### answer found")


async def main() -> None:
    # Setup
    base_model = "Qwen/Qwen3.5-4B"
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model, rank=32
    )
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_5_disable_thinking", tokenizer)

    sampling_params = tinker.SamplingParams(
        max_tokens=256,
        stop=renderer.get_stop_sequences(),
    )
    adam_params = tinker.AdamParams(learning_rate=4e-5, beta1=0.9, beta2=0.95)

    # Dataset
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    train_data = dataset["train"]

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."
    fewshot_prefix = [
        {"role": "user", "content": "How many r's are in strawberry?" + question_suffix},
        {
            "role": "assistant",
            "content": (
                "Let's spell the word out and number all the letters: "
                "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
                "We have r's at positions 3, 8, and 9. \\boxed{3}"
            ),
        },
    ]

    print(f"Loaded {len(train_data)} GSM8K training problems")

    # Training hyperparameters
    n_steps = 20
    batch_size = 16  # problems per step
    group_size = 8  # completions per problem

    # Tracking metrics
    metrics_history = []

    for step in range(n_steps):
        # 1. Get the batch of problems for this step
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        batch_rows = train_data.select(range(batch_start, batch_end))

        # 2. Save current weights and create a sampling client
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # 3. Submit all sampling requests concurrently
        prompts_P: list[tinker.ModelInput] = []
        _coros = []
        for question in batch_rows["question"]:
            convo = [*fewshot_prefix, {"role": "user", "content": question + question_suffix}]
            prompt = renderer.build_generation_prompt(convo)
            _coros.append(
                sampling_client.sample_async(
                    prompt=prompt, num_samples=group_size, sampling_params=sampling_params
                )
            )
            prompts_P.append(prompt)

        sample_results_P = await asyncio.gather(*_coros)

        # 4. Collect results, grade, compute advantages, build datums
        datums_D: list[tinker.Datum] = []
        rewards_P: list[float] = []
        n_degenerate = 0

        for sample_result, prompt, answer_text in zip(sample_results_P, prompts_P, batch_rows["answer"]):
            ground_truth = extract_gsm8k_answer(answer_text)

            # Grade each completion in the group
            rewards_G: list[float] = []
            tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []

            for sequence in sample_result.sequences:
                tokens_G_T.append(sequence.tokens)
                logprobs_G_T.append(sequence.logprobs)
                parsed_message, _ = renderer.parse_response(sequence.tokens)
                content = get_text_content(parsed_message)
                reward = grade_answer(content, ground_truth)
                rewards_G.append(reward)

            # Group-relative advantages
            mean_reward = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_reward for r in rewards_G]
            rewards_P.append(mean_reward)

            # Skip degenerate groups (all same reward -> zero advantage -> no signal)
            if all(a == 0.0 for a in advantages_G):
                n_degenerate += 1
                continue

            # Build a Datum for each completion
            ob_len = prompt.length - 1
            for tokens, logprobs, advantage in zip(tokens_G_T, logprobs_G_T, advantages_G):
                model_input = prompt.append(tinker.EncodedTextChunk(tokens=tokens[:-1]))
                target_tokens = [0] * ob_len + tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)

                datum = tinker.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                    },
                )
                datums_D.append(datum)

        # 5. Training step
        if len(datums_D) > 0:
            fwd_bwd_future = await training_client.forward_backward_async(
                datums_D, loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_bwd_future.result_async()
            await optim_future.result_async()

        mean_reward = sum(rewards_P) / len(rewards_P)
        frac_degenerate = n_degenerate / len(rewards_P)
        metrics_history.append(
            {"step": step, "reward": mean_reward, "frac_degenerate": frac_degenerate}
        )

        print(
            f"Step {step:2d} | reward: {mean_reward:.3f} | "
            f"degenerate: {frac_degenerate:.0%} | datums: {len(datums_D)}"
        )

    # Plot
    steps = [m["step"] for m in metrics_history]
    rewards = [m["reward"] for m in metrics_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rewards, marker="o", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean reward (fraction correct)")
    ax.set_title("RL Training: GSM8K Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig("rl_training_curve.png", dpi=150)
    print("Saved plot to rl_training_curve.png")


if __name__ == "__main__":
    asyncio.run(main())
