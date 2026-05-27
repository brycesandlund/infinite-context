"""RL training: GSM8K with a calculator tool, via the tinker_cookbook orchestrator.

The model is given four calculator tools (add/sub/mul/div) and grades on the
final \\boxed{} answer. Multi-turn rollouts, GRPO advantages, fwd/bwd, wandb,
and checkpointing are all handled by tinker_cookbook.rl.train.main.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

import chz
import datasets

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import (
    ToolResult,
    build_agent_tool_env,
    simple_tool_result,
    tool,
)


# ---------------------------------------------------------------------------
# Calculator tools
# ---------------------------------------------------------------------------


class CalculatorTools:
    """Four-function calculator. Each call returns the numeric result as a string."""

    @tool
    async def add(
        self,
        a: Annotated[float, "First operand"],
        b: Annotated[float, "Second operand"],
    ) -> ToolResult:
        """Add two numbers and return a + b."""
        return simple_tool_result(str(a + b))

    @tool
    async def sub(
        self,
        a: Annotated[float, "Minuend"],
        b: Annotated[float, "Subtrahend"],
    ) -> ToolResult:
        """Subtract and return a - b."""
        return simple_tool_result(str(a - b))

    @tool
    async def mul(
        self,
        a: Annotated[float, "First operand"],
        b: Annotated[float, "Second operand"],
    ) -> ToolResult:
        """Multiply and return a * b."""
        return simple_tool_result(str(a * b))

    @tool
    async def div(
        self,
        a: Annotated[float, "Numerator"],
        b: Annotated[float, "Denominator"],
    ) -> ToolResult:
        """Divide and return a / b. Returns an error string if b is zero."""
        if b == 0:
            return simple_tool_result("Error: division by zero")
        return simple_tool_result(str(a / b))


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_number(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip()


@dataclass
class GSM8KReward:
    """Format bonus + correctness check on the final \\boxed{} answer.

    reward = format_coef * (correct_format - 1) + correct_answer
    """

    gold_answer: str
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
        if final is None:
            return 0.0, {"format": 0.0, "correct": 0.0}

        content = get_text_content(final) or ""
        extracted = _extract_boxed(content)
        correct_format = float(extracted is not None)
        correct_answer = 0.0
        if extracted is not None and _normalize_number(extracted) == _normalize_number(
            self.gold_answer
        ):
            correct_answer = 1.0

        reward = self.format_coef * (correct_format - 1) + correct_answer
        return reward, {"format": correct_format, "correct": correct_answer}


# ---------------------------------------------------------------------------
# Env group builder
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a math problem solver. You have access to a four-function calculator "
    "(add, sub, mul, div). For multi-step arithmetic, call the calculator as many "
    "times as you need; for simple steps you may do them in your head. When you are "
    "confident in the final numerical answer, write it inside \\boxed{...} with no "
    "units and stop."
)


_ANSWER_RE = re.compile(r"####\s*(.+)")


def _extract_gsm8k_gold(answer_text: str) -> str:
    match = _ANSWER_RE.search(answer_text)
    if not match:
        raise ValueError(f"No #### answer in: {answer_text!r}")
    return match.group(1).replace(",", "").strip()


class GSM8KEnvGroupBuilder(EnvGroupBuilder):
    """One problem -> group_size parallel rollouts sharing the same prompt."""

    def __init__(
        self,
        question: str,
        gold_answer: str,
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        max_turns: int,
        max_trajectory_tokens: int,
        max_generation_tokens: int,
    ):
        self.question = question
        self.gold_answer = gold_answer
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.max_turns = max_turns
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens

    def _build_initial_messages(
        self, renderer: Renderer, calc: CalculatorTools
    ) -> list[Message]:
        tool_specs = [
            calc.add.to_spec(),
            calc.sub.to_spec(),
            calc.mul.to_spec(),
            calc.div.to_spec(),
        ]
        prefix = renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=SYSTEM_PROMPT
        )
        return prefix + [{"role": "user", "content": self.question}]

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)

        calc = CalculatorTools()
        tools = [calc.add, calc.sub, calc.mul, calc.div]
        initial_messages = self._build_initial_messages(renderer, calc)
        reward_fn = GSM8KReward(gold_answer=self.gold_answer)

        return [
            build_agent_tool_env(
                renderer=renderer,
                tools=tools,
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self.max_turns,
                max_trajectory_tokens=self.max_trajectory_tokens,
                max_generation_tokens=self.max_generation_tokens,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["gsm8k", "calculator"]


# ---------------------------------------------------------------------------
# Dataset + builder
# ---------------------------------------------------------------------------


class GSM8KRLDataset(RLDataset):
    def __init__(self, builders: list[EnvGroupBuilder], batch_size: int):
        self.builders = builders
        self.batch_size = batch_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        return self.builders[start : start + self.batch_size]

    def __len__(self) -> int:
        return len(self.builders) // self.batch_size


@chz.chz
class GSM8KDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    max_turns: int = 5
    max_trajectory_tokens: int = 4096
    max_generation_tokens: int = 256
    seed: int = 0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        ds = datasets.load_dataset("openai/gsm8k", "main")
        train_rows = ds["train"]

        indices = list(range(len(train_rows)))
        random.Random(self.seed).shuffle(indices)

        builders: list[EnvGroupBuilder] = []
        for idx in indices:
            row = train_rows[idx]
            try:
                gold = _extract_gsm8k_gold(row["answer"])
            except ValueError:
                continue
            builders.append(
                GSM8KEnvGroupBuilder(
                    question=row["question"],
                    gold_answer=gold,
                    model_name=self.model_name_for_tokenizer,
                    renderer_name=self.renderer_name,
                    group_size=self.group_size,
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                )
            )

        return GSM8KRLDataset(builders, batch_size=self.batch_size), None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str | None = "qwen3_5_disable_thinking"
    lora_rank: int = 32
    learning_rate: float = 4e-5

    batch_size: int = 4
    group_size: int = 4
    max_tokens: int = 256
    max_turns: int = 5
    max_trajectory_tokens: int = 4096
    max_generation_tokens: int = 256
    seed: int = 0

    log_path: str = "/tmp/gsm8k_calc_rl"
    eval_every: int = 0
    save_every: int = 20

    wandb_project: str | None = None
    wandb_name: str | None = None
    max_steps: int | None = None


async def cli_main(cli_config: CLIConfig) -> None:
    builder = GSM8KDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        batch_size=cli_config.batch_size,
        group_size=cli_config.group_size,
        renderer_name=cli_config.renderer_name,
        max_turns=cli_config.max_turns,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_generation_tokens=cli_config.max_generation_tokens,
        seed=cli_config.seed,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_log_path = f"{cli_config.log_path}/{timestamp}"

    config = train.Config(
        model_name=cli_config.model_name,
        renderer_name=cli_config.renderer_name,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        lora_rank=cli_config.lora_rank,
        log_path=run_log_path,
        dataset_builder=builder,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        wandb_project=cli_config.wandb_project,
        wandb_name=cli_config.wandb_name,
        max_steps=cli_config.max_steps,
    )
    await train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
