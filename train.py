"""Recursive-agent RL on Needle-in-a-Haystack (Phase 2).

Each problem is a synthesized long document (Paul Graham essays + a single
"The magic number is X" needle). Each agent has a tight `AGENT_CONTEXT`
context, so the parent literally cannot fit the doc and must delegate.

Tools available to every agent in the tree:
- `read_chunk(start, end)`: read up to MAX_CHUNK_TOKENS of doc tokens.
- `spawn_subagent(subtask)`: fresh-context copy of the same policy. Returns
  the child's \\boxed{} answer.

Children share the parent's group-relative advantage uniformly. The outer
training loop, datum stitching, and tree bookkeeping match phase 1. The
cookbook still supplies: tool dispatch (`AgentToolMessageEnv`), single-env
rollout (`do_single_rollout`), trajectory→Datum (`trajectory_to_data`),
and the Qwen3.5 renderer.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

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

import harness
from tasks import (
    GradingMode,
    Problem,
    eval_grading_mode,
    grade_answer,
    grading_mode,
    list_tasks,
    load_pg_essays_text,
    make_problem,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen3.5-9B"
RENDERER_NAME = "qwen3_5"      # thinking enabled
LORA_RANK = 32
LEARNING_RATE = 4e-5

N_STEPS = 3                # short validation run from the SFT warm-start; scale up once the refactored loop is confirmed
BATCH_SIZE = 4              # problems per training step
GROUP_SIZE = 4              # parent rollouts per problem
MAX_DEPTH = 2               # 0 = root only; 2 = root may spawn children that may spawn grandchildren
MAX_TURNS = 8               # per-agent multi-turn cap

# A single knob: the per-agent context budget. Both the trajectory cap and the
# per-turn generation cap derive from this — TinkerTokenCompleter dynamically
# caps max_tokens = AGENT_CONTEXT - prompt.length, and the env terminates when
# the trajectory would exceed AGENT_CONTEXT. The model can think as much as it
# wants per turn, limited only by remaining budget.
AGENT_CONTEXT = 10_000

# Task knobs.
#
# TASK_MIXTURE is a weighted sampler over task names (must all be in list_tasks()).
# Each problem slot in a batch independently samples one task per these weights;
# all GROUP_SIZE rollouts of that slot share the same task+seed so GRPO's
# group-mean baseline still works (compares apples to apples within a problem).
#
# Set to a single-task dict to lock training to one task family — handy for
# debugging or ablations. Weights need not sum to 1; they're renormalized.
TASK_MIXTURE: dict[str, float] = {
    # 11 RULER training tasks (canonical names per NVIDIA/RULER/scripts/synthetic.yaml).
    # qa_1/qa_2 are held out for eval (require SQuAD+HotpotQA downloads and
    # we want a clean train/eval split).
    "niah_single_1": 1.0, "niah_single_2": 1.0, "niah_single_3": 1.0,
    "niah_multikey_1": 1.0, "niah_multikey_2": 1.0, "niah_multikey_3": 1.0,
    "niah_multivalue": 1.0, "niah_multiquery": 1.0,
    "vt": 1.0, "cwe": 1.0, "fwe": 1.0,
}
DOC_SIZE_TOKENS = 15_000    # haystack length per problem
MAX_CHUNK_TOKENS = 8_000    # cap on a single read_chunk return. Sized close to AGENT_CONTEXT so a single read can fill most of an agent's window — forces a clean "read one range, then answer or delegate" cycle rather than nibbling.

DATA_SEED = 0               # base seed for problem generation; per-problem seed = DATA_SEED + step*BATCH_SIZE + idx

# Checkpointing. After training, save under this name (overwrite-safe). Set to None to skip saving.
# To resume: paste a tinker:// path into LOAD_CHECKPOINT_PATH below.
SAVE_CHECKPOINT_NAME: str | None = "final"
# CKPT env var overrides — e.g. CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) to warm-start RL from SFT.
LOAD_CHECKPOINT_PATH: str | None = os.environ.get("CKPT") or None  # tinker:// path; None = base model
RESUME_OPTIMIZER = False    # restore Adam momentum too. False when starting a fresh fine-tune from an SFT/base ckpt (the SFT optimizer state is for cross_entropy, not RL).
EVAL_ONLY = False           # skip training, go straight to eval (requires LOAD_CHECKPOINT_PATH to be useful)
LAST_CHECKPOINT_FILE = Path.home() / ".cache" / "infinite-context" / "last_checkpoint.txt"

# After training, run this many rollouts on held-out problem seeds (no fwd/bwd)
# and print each full tree. 0 = skip.
EVAL_N_ROLLOUTS = 4
EVAL_SEED_OFFSET = 1_000_000  # held-out seeds start here
# Optional: force the eval rollouts onto specific task families instead of
# sampling from TASK_MIXTURE. Cycles through the list if EVAL_N_ROLLOUTS exceeds
# its length. Set to None to sample from TASK_MIXTURE like the training loop.
# Handy for a smoke test where you want to SEE one of each family render+grade.
EVAL_TASKS: list[str] | None = ["niah_single_2", "niah_multiquery", "vt", "cwe"]

# Debug: do DEBUG_N_ROLLOUTS rollouts, print every tree + datum shapes, exit.
DEBUG_ONE_ROLLOUT = False
DEBUG_N_ROLLOUTS = 4
DEBUG_PRINT_TREE_EACH_STEP = False


# ---------------------------------------------------------------------------
# Read-chunk tool: bounded token-range view onto the per-problem document.
# Same instance is shared across an entire rollout tree (parent + descendants).
# ---------------------------------------------------------------------------


@dataclass
class ReadChunkTool:
    """Bounded view onto a fixed document. Shared across an entire rollout tree."""

    document_tokens: list[int]
    tokenizer: object
    max_chunk_tokens: int

    @tool
    async def read_chunk(
        self,
        start: Annotated[int, "First token position to read (inclusive)."],
        end: Annotated[int, "Last token position to read (exclusive)."],
    ) -> ToolResult:
        """Read a slice of the document and return the decoded text of tokens [start, end). The document has a fixed length (stated in the system prompt). Each call is capped at the chunk limit; for larger ranges, issue multiple reads or delegate to a subagent."""
        # Slicing/decode/cap semantics live in harness.read_chunk_impl so the
        # eval driver hits the exact same behavior.
        return simple_tool_result(
            harness.read_chunk_impl(
                self.document_tokens, self.tokenizer, start, end, self.max_chunk_tokens
            )
        )


# ---------------------------------------------------------------------------
# Rollout-tree types
# ---------------------------------------------------------------------------


@dataclass
class RolloutNode:
    """One agent's trajectory plus its direct children via `spawn_subagent`."""

    trajectory: Trajectory
    depth: int
    subtask: str         # text the parent asked the child to solve; "" for root
    answer: str | None   # extracted \boxed{}, or None
    children: list[RolloutNode] = field(default_factory=list)


def flatten_tree(node: RolloutNode) -> list[RolloutNode]:
    out = [node]
    for c in node.children:
        out.extend(flatten_tree(c))
    return out


# ---------------------------------------------------------------------------
# Subagent tool: recursive policy call
# ---------------------------------------------------------------------------


# System prompt + boxed extraction live in harness.py (shared with the eval
# driver). Thin aliases keep call sites below unchanged.
_make_system_prompt = harness.make_system_prompt
_extract_boxed = harness.extract_boxed


def _last_assistant_text(traj: Trajectory, tokenizer) -> str:
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

    The same `ReadChunkTool` instance is threaded to every agent in the tree —
    they all read from the same document, but each has its own context budget.
    """

    sampling_client: tinker.SamplingClient
    tokenizer: object
    renderer: Renderer
    read_chunk_tool: ReadChunkTool
    system_prompt: str
    max_depth: int
    max_turns: int
    context_budget: int           # per-agent total context (trajectory + remaining gen room)
    current_depth: int = 0
    child_nodes: list[RolloutNode] = field(default_factory=list)

    @tool
    async def spawn_subagent(
        self,
        subtask: Annotated[str, "Sub-problem statement (free text) for the child to solve"],
    ) -> ToolResult:
        """Spawn a fresh-context copy of yourself to solve `subtask`; returns the child's \\boxed{} answer."""
        if self.current_depth >= self.max_depth:
            return simple_tool_result("Error: max recursion depth reached. Solve directly.")

        child_subagent = SubagentTool(
            sampling_client=self.sampling_client,
            tokenizer=self.tokenizer,
            renderer=self.renderer,
            read_chunk_tool=self.read_chunk_tool,
            system_prompt=self.system_prompt,
            max_depth=self.max_depth,
            max_turns=self.max_turns,
            context_budget=self.context_budget,
            current_depth=self.current_depth + 1,
        )
        tool_list = [
            self.read_chunk_tool.read_chunk,
            child_subagent.spawn_subagent,
        ]
        tool_specs = [t.to_spec() for t in tool_list]
        prefix = self.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=self.system_prompt
        )
        initial_messages = prefix + [{"role": "user", "content": subtask}]

        child_env = build_agent_tool_env(
            renderer=self.renderer,
            tools=tool_list,
            initial_messages=initial_messages,
            reward_fn=_trivial_reward,
            max_turns=self.max_turns,
            max_trajectory_tokens=self.context_budget,
            # 1-token reserve so the env terminates cleanly at obs == context_budget
            # instead of letting the completer raise "No room for generation". Not a
            # generation cap — TinkerTokenCompleter dynamically caps max_tokens.
            max_generation_tokens=1,
            # Neutral (0.0), NOT the default -0.1: a negative overflow reward makes
            # "read thoroughly and risk overflow" score WORSE than "box a fast wrong
            # guess (0.0)", which collapses the policy into not reading. Keep all
            # failure modes at 0 so the only positive signal is a correct answer.
            context_overflow_reward=0.0,
        )
        policy = TinkerTokenCompleter(
            sampling_client=self.sampling_client,
            max_tokens=self.context_budget,
            context_window=self.context_budget,
            temperature=1.0,
        )
        child_trajectory = await do_single_rollout(policy, child_env)

        final_text = _last_assistant_text(child_trajectory, self.tokenizer)
        answer = _extract_boxed(final_text)
        child_node = RolloutNode(
            trajectory=child_trajectory,
            depth=self.current_depth + 1,
            subtask=subtask,
            answer=answer,
            children=child_subagent.child_nodes,
        )
        self.child_nodes.append(child_node)
        return simple_tool_result(answer if answer is not None else "No answer found")


# Drift guard: the model must see the same tool descriptions in training and in
# the eval driver. These cookbook @tool docstrings are the training-side spec;
# harness.* are the eval-side canonical strings. Fail loudly at import if they
# diverge. (FunctionTool exposes `.description` = the captured docstring.)
assert ReadChunkTool.read_chunk.description == harness.READ_CHUNK_DESCRIPTION, (
    "read_chunk docstring drifted from harness.READ_CHUNK_DESCRIPTION"
)
assert SubagentTool.spawn_subagent.description == harness.SPAWN_SUBAGENT_DESCRIPTION, (
    "spawn_subagent docstring drifted from harness.SPAWN_SUBAGENT_DESCRIPTION"
)


# ---------------------------------------------------------------------------
# Reward (task-agnostic): graded on the parent's final \boxed{} answer
# ---------------------------------------------------------------------------


@dataclass
class LongContextReward:
    """Format bonus + score on the parent's final \\boxed{} answer.

    `mode` is one of:
    - "exact":   single value, normalized equality;            score ∈ {0, 1}
    - "set":     any-order list of values in \\boxed{};        score ∈ {0, 1}
    - "numeric": float-valued answer, 0.75**|y-y_hat| partial credit; score ∈ [0, 1]

    Reward = format_coef * (format - 1) + score. The format term is a small
    negative shaping bonus for failing to emit \\boxed{}; the score is the
    grader's float output.
    """

    gold_answers: list[str]
    mode: GradingMode = "exact"
    format_coef: float = 0.1

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        final = next((m for m in reversed(history) if m.get("role") == "assistant"), None)
        if final is None:
            return 0.0, {"format": 0.0, "score": 0.0}
        content = get_text_content(final) or ""
        extracted = _extract_boxed(content)
        correct_format = float(extracted is not None)
        score = grade_answer(extracted, self.gold_answers, self.mode)
        reward = self.format_coef * (correct_format - 1) + score
        return reward, {"format": correct_format, "score": score}


# ---------------------------------------------------------------------------
# Per-problem rollout
# ---------------------------------------------------------------------------


@dataclass
class ParentRollout:
    """One parent rollout = a tree of trajectories rooted at depth 0."""

    root: RolloutNode
    problem: Problem

    @property
    def gold_answers(self) -> list[str]:
        return self.problem.gold_answers

    @property
    def task(self) -> str:
        return self.problem.task

    def all_nodes(self) -> list[RolloutNode]:
        return flatten_tree(self.root)


async def _rollout_one_parent(
    problem: Problem,
    sampling_client: tinker.SamplingClient,
    tokenizer,
    renderer: Renderer,
) -> ParentRollout:
    read_chunk_tool = ReadChunkTool(
        document_tokens=problem.document_tokens,
        tokenizer=tokenizer,
        max_chunk_tokens=MAX_CHUNK_TOKENS,
    )
    system_prompt = _make_system_prompt(
        doc_length=len(problem.document_tokens),
        context_budget=AGENT_CONTEXT,
        max_chunk_tokens=MAX_CHUNK_TOKENS,
        task_context=problem.task_context,
    )
    parent_subagent = SubagentTool(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        renderer=renderer,
        read_chunk_tool=read_chunk_tool,
        system_prompt=system_prompt,
        max_depth=MAX_DEPTH,
        max_turns=MAX_TURNS,
        context_budget=AGENT_CONTEXT,
        current_depth=0,
    )
    tool_list = [read_chunk_tool.read_chunk, parent_subagent.spawn_subagent]
    tool_specs = [t.to_spec() for t in tool_list]
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_specs, system_prompt=system_prompt
    )
    user_message = (
        f"The document is {len(problem.document_tokens)} tokens long. {problem.question}"
    )
    initial_messages = prefix + [{"role": "user", "content": user_message}]

    parent_env = build_agent_tool_env(
        renderer=renderer,
        tools=tool_list,
        initial_messages=initial_messages,
        reward_fn=LongContextReward(
            gold_answers=problem.gold_answers, mode=grading_mode(problem.task)
        ),
        max_turns=MAX_TURNS,
        max_trajectory_tokens=AGENT_CONTEXT,
        max_generation_tokens=1,  # see SubagentTool.spawn_subagent for rationale
        context_overflow_reward=0.0,  # neutral failure — see child env note above
    )
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=AGENT_CONTEXT,
        context_window=AGENT_CONTEXT,
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
    return ParentRollout(root=root, problem=problem)


def _trajectory_total_reward(traj: Trajectory) -> float:
    return sum(t.reward for t in traj.transitions)


def _strip_mask(datum: tinker.Datum) -> tinker.Datum:
    """`trajectory_to_data` writes a `mask` entry that `importance_sampling`
    doesn't accept. Drop it before fwd_bwd."""
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
    print("Loading Paul Graham essays + tokenizing corpus...")
    corpus_text = load_pg_essays_text()
    corpus_tokens = tokenizer.encode(corpus_text, add_special_tokens=False)
    print(
        f"Corpus: {len(corpus_text)} chars -> {len(corpus_tokens)} tokens. "
        f"Doc size per problem: {DOC_SIZE_TOKENS} tokens; agent context cap: "
        f"{AGENT_CONTEXT}; chunk cap: {MAX_CHUNK_TOKENS}."
    )
    if len(corpus_tokens) < DOC_SIZE_TOKENS:
        raise SystemExit(
            f"Corpus too small ({len(corpus_tokens)} tokens) for DOC_SIZE_TOKENS={DOC_SIZE_TOKENS}."
        )

    unknown_tasks = [t for t in TASK_MIXTURE if t not in list_tasks()]
    if unknown_tasks:
        raise SystemExit(f"Unknown tasks in TASK_MIXTURE: {unknown_tasks}. Available: {list_tasks()}")
    if not TASK_MIXTURE:
        raise SystemExit("TASK_MIXTURE is empty.")
    _mixture_names = list(TASK_MIXTURE.keys())
    _mixture_weights = [TASK_MIXTURE[t] for t in _mixture_names]

    def gen_problem(seed: int) -> Problem:
        """Sample one task from TASK_MIXTURE using `seed`, then generate it.

        The task choice is a function of `seed` alone, so a given seed always
        produces the same problem — needed for GRPO (all GROUP_SIZE rollouts of
        a slot share the same problem/task) and for held-out eval reproducibility.
        """
        rng = random.Random(seed)
        task = rng.choices(_mixture_names, weights=_mixture_weights, k=1)[0]
        # Use a derived seed for the generator so its sampling is independent
        # of which task we chose (otherwise the same `seed` would deterministically
        # produce the same haystack across different tasks).
        gen_seed = rng.randrange(2**32)
        return make_problem(
            task=task,
            corpus_tokens=corpus_tokens,
            tokenizer=tokenizer,
            doc_size_tokens=DOC_SIZE_TOKENS,
            seed=gen_seed,
        )

    def gen_problem_for_task(task: str, seed: int) -> Problem:
        """Generate a problem for a SPECIFIC task (bypasses TASK_MIXTURE).
        Used by EVAL_TASKS to force coverage of chosen families."""
        return make_problem(
            task=task,
            corpus_tokens=corpus_tokens,
            tokenizer=tokenizer,
            doc_size_tokens=DOC_SIZE_TOKENS,
            seed=random.Random(seed).randrange(2**32),
        )

    if EVAL_TASKS:
        unknown_eval = [t for t in EVAL_TASKS if t not in list_tasks()]
        if unknown_eval:
            raise SystemExit(f"Unknown tasks in EVAL_TASKS: {unknown_eval}. Available: {list_tasks()}")

    # Optional: load a previously saved checkpoint before doing anything else.
    if LOAD_CHECKPOINT_PATH:
        print(f"Loading checkpoint: {LOAD_CHECKPOINT_PATH}")
        loader = (
            training_client.load_state_with_optimizer_async
            if RESUME_OPTIMIZER
            else training_client.load_state_async
        )
        load_future = await loader(LOAD_CHECKPOINT_PATH)
        await load_future.result_async()
        print("  loaded.")

    if DEBUG_ONE_ROLLOUT:
        from debug import debug_run_rollouts_niah

        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        await debug_run_rollouts_niah(
            n_rollouts=DEBUG_N_ROLLOUTS,
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            problems=[gen_problem(DATA_SEED + i) for i in range(DEBUG_N_ROLLOUTS)],
            rollout_fn=_rollout_one_parent,
        )
        return

    if EVAL_ONLY:
        print("EVAL_ONLY: skipping training loop.")

    for step in range(0 if EVAL_ONLY else N_STEPS):
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # One problem per slot in the batch (task sampled from TASK_MIXTURE);
        # GROUP_SIZE rollouts each, all sharing that problem so the GRPO group-mean
        # baseline compares like-vs-like.
        problems = [gen_problem(DATA_SEED + step * BATCH_SIZE + pi) for pi in range(BATCH_SIZE)]

        rollout_coros = []
        problem_idx_for_rollout: list[int] = []
        for pi, problem in enumerate(problems):
            for _ in range(GROUP_SIZE):
                rollout_coros.append(
                    _rollout_one_parent(problem, sampling_client, tokenizer, renderer)
                )
                problem_idx_for_rollout.append(pi)
        parent_results: list[ParentRollout] = await asyncio.gather(*rollout_coros)

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

        all_datums: list[tinker.Datum] = []
        trajectories_per_depth: dict[int, int] = {}
        for ri, parent_result in enumerate(parent_results):
            adv = advantages[ri]
            for node in parent_result.all_nodes():
                if not node.trajectory.transitions:
                    continue
                all_datums.extend(trajectory_to_data(node.trajectory, adv))
                trajectories_per_depth[node.depth] = trajectories_per_depth.get(node.depth, 0) + 1

        nonzero_adv = any(abs(a) > 1e-9 for a in advantages)
        if all_datums and nonzero_adv:
            fwd_bwd_future = await training_client.forward_backward_async(
                [_strip_mask(d) for d in all_datums], loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_bwd_future.result_async()
            await optim_future.result_async()

        # Per-task aggregation: mean score (the grader's float output) per task
        # name. Lets us see at a glance whether the mixture is balanced and
        # which families the policy is improving on.
        per_task_scores: dict[str, list[float]] = {}
        for r in parent_results:
            score = grade_answer(r.root.answer, r.gold_answers, grading_mode(r.task))
            per_task_scores.setdefault(r.task, []).append(score)
        per_task_summary = ", ".join(
            f"{t}={sum(s) / len(s):.2f}({len(s)})"
            for t, s in sorted(per_task_scores.items())
        )
        mean_reward = sum(rewards) / len(rewards)
        mean_score = sum(
            s for scores in per_task_scores.values() for s in scores
        ) / len(parent_results)
        depth_counts = ", ".join(
            f"d{d}={trajectories_per_depth.get(d, 0)}"
            for d in sorted(trajectories_per_depth)
        )
        print(
            f"Step {step:2d} | mean_reward: {mean_reward:.3f} | "
            f"mean_score: {mean_score:.3f} | "
            f"by_task: {per_task_summary} | "
            f"trajectories: {depth_counts} | "
            f"datums: {len(all_datums)} | "
            f"trained: {bool(all_datums and nonzero_adv)}"
        )

        if DEBUG_PRINT_TREE_EACH_STEP:
            from debug import print_rollout_tree

            for ri, parent_result in enumerate(parent_results):
                print(f"--- rollout {ri} (advantage={advantages[ri]:+.3f}) ---")
                print_rollout_tree(parent_result.root)

    # ------------------------------------------------------------------ #
    # Save checkpoint (skipped on EVAL_ONLY to avoid overwriting with    #
    # a model we didn't train).                                          #
    # ------------------------------------------------------------------ #
    if SAVE_CHECKPOINT_NAME and not EVAL_ONLY:
        print(f"Saving checkpoint '{SAVE_CHECKPOINT_NAME}'...")
        save_future = await training_client.save_state_async(
            SAVE_CHECKPOINT_NAME, overwrite=True
        )
        save_resp = await save_future.result_async()
        print(f"  saved: {save_resp.path}")
        LAST_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_CHECKPOINT_FILE.write_text(save_resp.path)
        print(f"  (path written to {LAST_CHECKPOINT_FILE})")

    # ------------------------------------------------------------------ #
    # Post-training eval: K rollouts on held-out problems. Each rollout  #
    # gets a full per-agent transcript dump (system + user + every       #
    # assistant turn + every tool result).                               #
    # ------------------------------------------------------------------ #
    if EVAL_N_ROLLOUTS > 0:
        from debug import print_rollout_tree_verbose

        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        if EVAL_TASKS:
            # Force one problem per listed task (cycling if N > len), so the
            # smoke test exercises a known spread of families.
            eval_problems = [
                gen_problem_for_task(EVAL_TASKS[i % len(EVAL_TASKS)], EVAL_SEED_OFFSET + i)
                for i in range(EVAL_N_ROLLOUTS)
            ]
        else:
            eval_problems = [gen_problem(EVAL_SEED_OFFSET + i) for i in range(EVAL_N_ROLLOUTS)]
        print()
        print("=" * 72)
        print(f"POST-TRAINING EVAL: {EVAL_N_ROLLOUTS} held-out rollouts (verbose)")
        print("=" * 72)
        eval_coros = [
            _rollout_one_parent(p, sampling_client, tokenizer, renderer)
            for p in eval_problems
        ]
        eval_results = await asyncio.gather(*eval_coros)
        # Two scores per eval rollout:
        # - `ruler_score`: RULER's official substring matcher (string_match_all /
        #   string_match_part). Comparable to NVIDIA's published leaderboard.
        # - `train_score`: strict grader matching the training reward — useful
        #   for diagnosing whether the agent learned the strict format or only
        #   the looser substring criterion.
        eval_ruler_by_task: dict[str, list[float]] = {}
        eval_train_by_task: dict[str, list[float]] = {}
        for ri, (problem, result) in enumerate(zip(eval_problems, eval_results)):
            nodes = result.all_nodes()
            parent_reward = _trajectory_total_reward(result.root.trajectory)
            ruler_score = grade_answer(
                result.root.answer, result.gold_answers, eval_grading_mode(result.task)
            )
            train_score = grade_answer(
                result.root.answer, result.gold_answers, grading_mode(result.task)
            )
            eval_ruler_by_task.setdefault(result.task, []).append(ruler_score)
            eval_train_by_task.setdefault(result.task, []).append(train_score)
            print()
            print("#" * 72)
            print(
                f"# Eval rollout {ri}  task={result.task}  "
                f"gold={result.gold_answers}  "
                f"doc_tokens={len(problem.document_tokens)}  "
                f"nodes={len(nodes)}  parent_reward={parent_reward:.3f}  "
                f"ruler_score={ruler_score:.3f}  train_score={train_score:.3f}"
            )
            print("#" * 72)
            print_rollout_tree_verbose(result.root, tokenizer)
        print()
        for t in sorted(eval_ruler_by_task):
            rs = eval_ruler_by_task[t]
            ts = eval_train_by_task[t]
            print(
                f"Eval {t}:  ruler {sum(rs)/len(rs):.3f}  "
                f"strict {sum(ts)/len(ts):.3f}  ({len(rs)} rollouts)"
            )
        all_ruler = [s for ss in eval_ruler_by_task.values() for s in ss]
        all_train = [s for ss in eval_train_by_task.values() for s in ss]
        print(
            f"Eval overall:  ruler {sum(all_ruler)/len(all_ruler):.3f}  "
            f"strict {sum(all_train)/len(all_train):.3f}  ({len(all_ruler)} rollouts)"
        )


if __name__ == "__main__":
    asyncio.run(main())
