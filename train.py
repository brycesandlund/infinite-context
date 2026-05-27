"""Recursive-agent RL on GSM8K (Phase 1).

The parent agent has two tools:
- A four-function calculator (add/sub/mul/div).
- `spawn_subagent(subtask)` which runs a fresh-context child agent (same policy)
  and returns the child's \\boxed{} answer.

Children have the same tools and can recurse up to max_depth. Each parent
problem produces a *tree* of trajectories: 1 parent + N descendants. The
parent's group-relative advantage is applied uniformly to every descendant.

We hand-roll the outer training loop (cookbook's `train.main` assumes one
trajectory per env, which doesn't fit recursive agents). We still lean on
the cookbook for: tool dispatch (`AgentToolMessageEnv`), single-env rollout
(`do_single_rollout`), trajectory→Datum conversion (`trajectory_to_data`),
and the Qwen3.5 renderer.

Set DEBUG_ONE_ROLLOUT=True to run a single rollout, print the full tree of
trajectories + datum shapes, and exit without training.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Annotated

import datasets
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.tool_use import (
    ToolResult,
    build_agent_tool_env,
    simple_tool_result,
    tool,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"
LORA_RANK = 32
LEARNING_RATE = 4e-5

N_STEPS = 20
BATCH_SIZE = 4          # problems per training step
GROUP_SIZE = 4          # parent rollouts per problem
MAX_DEPTH = 2           # 0 = root only; 2 = root may spawn children that may spawn grandchildren
MAX_TURNS = 5           # per-agent multi-turn cap
MAX_TOKENS_PER_GEN = 256
MAX_TRAJ_TOKENS = 4096

# Debug: when True, do DEBUG_N_ROLLOUTS rollouts, print every tree + datum
# shapes, and exit before any training step.
DEBUG_ONE_ROLLOUT = False
DEBUG_N_ROLLOUTS = 8
# Debug: when True, print the full rollout tree after every training step.
DEBUG_PRINT_TREE_EACH_STEP = False


# ---------------------------------------------------------------------------
# Calculator tools (shared by every agent in the tree)
# ---------------------------------------------------------------------------


class CalculatorTools:
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
        """Divide and return a / b. Errors on zero denominator."""
        if b == 0:
            return simple_tool_result("Error: division by zero")
        return simple_tool_result(str(a / b))


# ---------------------------------------------------------------------------
# Tree of rollouts
# ---------------------------------------------------------------------------


@dataclass
class RolloutNode:
    """One agent's trajectory plus its direct children spawned via
    `spawn_subagent`. A `ParentRollout` is rooted at a depth-0 node."""

    trajectory: Trajectory
    depth: int
    subtask: str         # text the parent asked the child to solve; "" for root
    answer: str | None   # extracted \boxed{}, or None if the agent didn't emit one
    children: list[RolloutNode] = field(default_factory=list)


def flatten_tree(node: RolloutNode) -> list[RolloutNode]:
    """Depth-first flatten — returns root first, then descendants."""
    out = [node]
    for c in node.children:
        out.extend(flatten_tree(c))
    return out


# ---------------------------------------------------------------------------
# Subagent tool: recursive policy call
# ---------------------------------------------------------------------------


SUBAGENT_SYSTEM_PROMPT = (
    "You are a math problem solver. You can use a four-function calculator "
    "(add, sub, mul, div) for arithmetic, and you can call `spawn_subagent(subtask)` "
    "to delegate a sub-problem to a fresh-context copy of yourself, which will "
    "return its final \\boxed{} answer to you. When you are confident in the "
    "final numerical answer, write it inside \\boxed{...} with no units and stop."
)


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _last_assistant_text(traj: Trajectory, tokenizer) -> str:
    """Decode the model's final action tokens to text."""
    if not traj.transitions:
        return ""
    return tokenizer.decode(traj.transitions[-1].ac.tokens)


async def _trivial_reward(history: list[Message]) -> tuple[float, dict[str, float]]:
    """Child rollouts don't get individual rewards — they inherit the parent's."""
    return 0.0, {}


@dataclass
class SubagentTool:
    """A tool that, when called, recursively rolls out the same policy on a
    fresh context and returns the child's \\boxed{} answer.

    Each call appends a `RolloutNode` to `child_nodes` containing the child's
    trajectory plus any grandchildren it spawned (recursive tree). Each node
    is referenced from exactly one place — its parent's `children` list — so
    total memory is O(N) in the number of descendants.
    """

    sampling_client: tinker.SamplingClient
    tokenizer: object
    renderer: Renderer
    max_depth: int
    max_turns: int
    max_tokens: int
    max_trajectory_tokens: int
    current_depth: int = 0
    child_nodes: list[RolloutNode] = field(default_factory=list)

    @tool
    async def spawn_subagent(
        self,
        subtask: Annotated[str, "The sub-problem statement for the child agent to solve"],
    ) -> ToolResult:
        """Spawn a fresh-context copy of yourself to solve `subtask`. Returns the child's
        final \\boxed{} answer."""
        if self.current_depth >= self.max_depth:
            return simple_tool_result("Error: max recursion depth reached. Solve directly.")

        child_subagent = SubagentTool(
            sampling_client=self.sampling_client,
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            max_depth=self.max_depth,
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_trajectory_tokens=self.max_trajectory_tokens,
            current_depth=self.current_depth + 1,
        )
        child_calc = CalculatorTools()
        tool_list = [
            child_calc.add,
            child_calc.sub,
            child_calc.mul,
            child_calc.div,
            child_subagent.spawn_subagent,
        ]
        tool_specs = [t.to_spec() for t in tool_list]
        prefix = self.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=SUBAGENT_SYSTEM_PROMPT
        )
        initial_messages = prefix + [{"role": "user", "content": subtask}]

        child_env = build_agent_tool_env(
            renderer=self.renderer,
            tools=tool_list,
            initial_messages=initial_messages,
            reward_fn=_trivial_reward,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_tokens,
        )
        policy = TinkerTokenCompleter(
            sampling_client=self.sampling_client,
            max_tokens=self.max_tokens,
            temperature=1.0,
        )
        child_trajectory = await do_single_rollout(policy, child_env)

        final_text = _last_assistant_text(child_trajectory, self.tokenizer)
        answer = _extract_boxed(final_text)

        # Build this child's node. Its `children` field holds *its* descendants
        # (already populated as the child rolled out and called spawn_subagent
        # on its own SubagentTool instance).
        child_node = RolloutNode(
            trajectory=child_trajectory,
            depth=self.current_depth + 1,
            subtask=subtask,
            answer=answer,
            children=child_subagent.child_nodes,
        )
        self.child_nodes.append(child_node)

        return simple_tool_result(answer if answer is not None else "No answer found")


# ---------------------------------------------------------------------------
# GSM8K reward (on the parent's final answer only)
# ---------------------------------------------------------------------------


_ANSWER_RE = re.compile(r"####\s*(.+)")


def _extract_gsm8k_gold(answer_text: str) -> str:
    m = _ANSWER_RE.search(answer_text)
    if not m:
        raise ValueError(f"No #### answer in: {answer_text!r}")
    return m.group(1).replace(",", "").strip()


def _normalize_number(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip()


@dataclass
class GSM8KReward:
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
# Per-problem rollout
# ---------------------------------------------------------------------------


@dataclass
class ParentRollout:
    """One parent rollout = a tree of trajectories rooted at depth 0."""

    root: RolloutNode
    gold_answer: str

    def all_nodes(self) -> list[RolloutNode]:
        return flatten_tree(self.root)


async def _rollout_one_parent(
    question: str,
    gold_answer: str,
    sampling_client: tinker.SamplingClient,
    tokenizer,
    renderer: Renderer,
) -> ParentRollout:
    parent_subagent = SubagentTool(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        renderer=renderer,
        max_depth=MAX_DEPTH,
        max_turns=MAX_TURNS,
        max_tokens=MAX_TOKENS_PER_GEN,
        max_trajectory_tokens=MAX_TRAJ_TOKENS,
        current_depth=0,
    )
    parent_calc = CalculatorTools()
    tool_list = [
        parent_calc.add,
        parent_calc.sub,
        parent_calc.mul,
        parent_calc.div,
        parent_subagent.spawn_subagent,
    ]
    tool_specs = [t.to_spec() for t in tool_list]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_specs, system_prompt=SUBAGENT_SYSTEM_PROMPT
    )
    initial_messages = prefix + [{"role": "user", "content": question}]

    parent_env = build_agent_tool_env(
        renderer=renderer,
        tools=tool_list,
        initial_messages=initial_messages,
        reward_fn=GSM8KReward(gold_answer=gold_answer),
        max_turns=MAX_TURNS,
        max_trajectory_tokens=MAX_TRAJ_TOKENS,
        max_generation_tokens=MAX_TOKENS_PER_GEN,
    )
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=MAX_TOKENS_PER_GEN,
        temperature=1.0,
    )
    parent_traj = await do_single_rollout(policy, parent_env)

    parent_text = _last_assistant_text(parent_traj, tokenizer)
    parent_answer = _extract_boxed(parent_text)
    root = RolloutNode(
        trajectory=parent_traj,
        depth=0,
        subtask="",
        answer=parent_answer,
        children=parent_subagent.child_nodes,
    )
    return ParentRollout(root=root, gold_answer=gold_answer)


def _trajectory_total_reward(traj: Trajectory) -> float:
    return sum(t.reward for t in traj.transitions)


def _strip_mask(datum: tinker.Datum) -> tinker.Datum:
    """`trajectory_to_data` writes a `mask` entry into loss_fn_inputs that
    `importance_sampling` doesn't accept. Drop it before fwd_bwd."""
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


async def main() -> None:
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    adam_params = tinker.AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95)

    print(f"Loaded model {MODEL_NAME}, renderer {RENDERER_NAME}, max_depth {MAX_DEPTH}")
    ds = datasets.load_dataset("openai/gsm8k", "main")
    train_rows = ds["train"]
    print(f"Loaded {len(train_rows)} GSM8K problems")

    if DEBUG_ONE_ROLLOUT:
        from debug import debug_run_rollouts

        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        await debug_run_rollouts(
            n_rollouts=DEBUG_N_ROLLOUTS,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            train_rows=train_rows,
            rollout_fn=_rollout_one_parent,
            gold_extractor=_extract_gsm8k_gold,
        )
        return

    for step in range(N_STEPS):
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        batch_start = step * BATCH_SIZE
        batch_rows = train_rows.select(range(batch_start, batch_start + BATCH_SIZE))
        problems = [
            (row["question"], _extract_gsm8k_gold(row["answer"])) for row in batch_rows
        ]

        # Fire all parent rollouts concurrently (BATCH_SIZE * GROUP_SIZE of them).
        rollout_coros = []
        problem_idx_for_rollout: list[int] = []
        for pi, (question, gold) in enumerate(problems):
            for _ in range(GROUP_SIZE):
                rollout_coros.append(
                    _rollout_one_parent(question, gold, sampling_client, tokenizer, renderer)
                )
                problem_idx_for_rollout.append(pi)
        parent_results: list[ParentRollout] = await asyncio.gather(*rollout_coros)

        # Per-problem group-relative advantages.
        rewards: list[float] = [
            _trajectory_total_reward(r.root.trajectory) for r in parent_results
        ]
        per_problem_rewards: list[list[float]] = [[] for _ in problems]
        per_problem_indices: list[list[int]] = [[] for _ in problems]
        for ri, pi in enumerate(problem_idx_for_rollout):
            per_problem_rewards[pi].append(rewards[ri])
            per_problem_indices[pi].append(ri)

        advantages: list[float] = [0.0] * len(parent_results)
        for pi in range(len(problems)):
            group_rewards = per_problem_rewards[pi]
            group_mean = sum(group_rewards) / len(group_rewards)
            for ri in per_problem_indices[pi]:
                advantages[ri] = rewards[ri] - group_mean

        # Build Datums for the full tree of each rollout (parent + descendants).
        all_datums: list[tinker.Datum] = []
        n_children_total = 0
        for ri, parent_result in enumerate(parent_results):
            adv = advantages[ri]
            for node in parent_result.all_nodes():
                if not node.trajectory.transitions:
                    continue
                all_datums.extend(trajectory_to_data(node.trajectory, adv))
                if node.depth > 0:
                    n_children_total += 1

        nonzero_adv = any(abs(a) > 1e-9 for a in advantages)
        if all_datums and nonzero_adv:
            fwd_bwd_future = await training_client.forward_backward_async(
                [_strip_mask(d) for d in all_datums], loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_bwd_future.result_async()
            await optim_future.result_async()

        n_correct = sum(
            1
            for r in parent_results
            if r.root.answer is not None
            and _normalize_number(r.root.answer) == _normalize_number(r.gold_answer)
        )
        mean_reward = sum(rewards) / len(rewards)
        children_per_parent = n_children_total / len(parent_results)
        print(
            f"Step {step:2d} | mean_reward: {mean_reward:.3f} | "
            f"parent_correct: {n_correct}/{len(parent_results)} | "
            f"children/parent: {children_per_parent:.2f} | "
            f"datums: {len(all_datums)} | "
            f"trained: {bool(all_datums and nonzero_adv)}"
        )

        if DEBUG_PRINT_TREE_EACH_STEP:
            from debug import print_rollout_tree

            for ri, parent_result in enumerate(parent_results):
                print(f"--- rollout {ri} (advantage={advantages[ri]:+.3f}) ---")
                print_rollout_tree(parent_result.root)


if __name__ == "__main__":
    asyncio.run(main())
