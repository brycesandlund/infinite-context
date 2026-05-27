"""Recursive-agent RL on Needle-in-a-Haystack (Phase 2).

Each problem is a synthesized long document (Paul Graham essays + a single
"The magic number is X" needle). Each agent has a tight `MAX_TRAJ_TOKENS`
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
import re
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

from niah_data import NIAHProblem, load_pg_essays_text, make_niah_problem


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"
LORA_RANK = 32
LEARNING_RATE = 4e-5

N_STEPS = 20
BATCH_SIZE = 4              # problems per training step
GROUP_SIZE = 4              # parent rollouts per problem
MAX_DEPTH = 2               # 0 = root only; 2 = root may spawn children that may spawn grandchildren
MAX_TURNS = 8               # per-agent multi-turn cap
MAX_TOKENS_PER_GEN = 256    # max tokens generated per assistant turn
MAX_TRAJ_TOKENS = 10_000    # per-agent context budget; if a turn would overflow, episode ends

# NIAH task knobs
DOC_SIZE_TOKENS = 15_000    # haystack length per problem; v0 keeps this just slightly over the context budget so the model can sometimes solve it without delegation, giving GRPO some reward variance to bootstrap from. Ramp up once the loop trains.
MAX_CHUNK_TOKENS = 8_000    # cap on a single read_chunk return. Sized close to MAX_TRAJ_TOKENS so a single read can fill most of an agent's window — forces a clean "read one range, then answer or delegate" cycle rather than nibbling.

DATA_SEED = 0               # base seed for problem generation; per-problem seed = DATA_SEED + step*BATCH_SIZE + idx

# Checkpointing. After training, save under this name (overwrite-safe). Set to None to skip saving.
# To resume: paste a tinker:// path into LOAD_CHECKPOINT_PATH below.
SAVE_CHECKPOINT_NAME: str | None = "final"
LOAD_CHECKPOINT_PATH: str | None = None  # e.g. "tinker://<run-id>/weights/final"; cat ~/.cache/infinite-context/last_checkpoint.txt to recall the last save
RESUME_OPTIMIZER = True     # if loading, also restore Adam momentum (use False when starting a fresh fine-tune from a base ckpt)
EVAL_ONLY = False           # skip training, go straight to eval (requires LOAD_CHECKPOINT_PATH to be useful)
LAST_CHECKPOINT_FILE = Path.home() / ".cache" / "infinite-context" / "last_checkpoint.txt"

# After training, run this many rollouts on held-out problem seeds (no fwd/bwd)
# and print each full tree. 0 = skip.
EVAL_N_ROLLOUTS = 4
EVAL_SEED_OFFSET = 1_000_000  # held-out seeds start here

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
        """Read a slice of the document. Returns the decoded text of tokens [start, end).
        The total document has a fixed length (stated in the system prompt). Each call
        is capped at the chunk limit; for larger ranges, issue multiple reads or delegate
        to a subagent."""
        n = len(self.document_tokens)
        if start < 0:
            start = 0
        if end > n:
            end = n
        if end <= start:
            return simple_tool_result("Empty range.")
        if end - start > self.max_chunk_tokens:
            return simple_tool_result(
                f"Range too large ({end - start} tokens > {self.max_chunk_tokens} cap). "
                f"Issue smaller reads or delegate to a subagent."
            )
        text = self.tokenizer.decode(self.document_tokens[start:end])
        return simple_tool_result(text)


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


def _make_system_prompt(doc_length: int, context_budget: int, max_chunk_tokens: int) -> str:
    return (
        f"You are a long-document search assistant. The document is {doc_length} tokens "
        f"long. Your own context window is {context_budget} tokens — the conversation "
        f"(system prompt, user message, your responses, and all tool results) must fit "
        f"in this budget or the episode ends.\n\n"
        f"You have two tools:\n"
        f"- `read_chunk(start, end)`: read the document tokens in [start, end). A "
        f"single read returns at most {max_chunk_tokens} tokens — most of your context "
        f"window — so plan accordingly: typically you can do one read of any range and "
        f"then must either answer or delegate further.\n"
        f"- `spawn_subagent(subtask)`: delegate to a fresh-context copy of yourself "
        f"(also {context_budget} tokens) with the same tools. Pass an explicit subtask "
        f"string that names a token range, e.g. \"Read tokens 5000..7000 and return the "
        f"magic number, or 'not found'\". The subagent returns its final \\boxed{{}} "
        f"answer as a string.\n\n"
        f"You may call spawn_subagent multiple times in parallel within a single turn "
        f"to scan disjoint ranges concurrently. When you are confident in the final "
        f"answer, emit it as \\boxed{{value}} and stop."
    )


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


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
    max_tokens: int
    max_trajectory_tokens: int
    current_depth: int = 0
    child_nodes: list[RolloutNode] = field(default_factory=list)

    @tool
    async def spawn_subagent(
        self,
        subtask: Annotated[str, "Sub-problem statement (free text) for the child to solve"],
    ) -> ToolResult:
        """Spawn a fresh-context copy of yourself to solve `subtask`. Returns the child's
        \\boxed{} answer."""
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
            max_tokens=self.max_tokens,
            max_trajectory_tokens=self.max_trajectory_tokens,
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
# NIAH reward
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip()


@dataclass
class NIAHReward:
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
        if extracted is not None and _normalize_answer(extracted) == _normalize_answer(self.gold_answer):
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
    problem: NIAHProblem

    def all_nodes(self) -> list[RolloutNode]:
        return flatten_tree(self.root)


async def _rollout_one_parent(
    problem: NIAHProblem,
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
        context_budget=MAX_TRAJ_TOKENS,
        max_chunk_tokens=MAX_CHUNK_TOKENS,
    )
    parent_subagent = SubagentTool(
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        renderer=renderer,
        read_chunk_tool=read_chunk_tool,
        system_prompt=system_prompt,
        max_depth=MAX_DEPTH,
        max_turns=MAX_TURNS,
        max_tokens=MAX_TOKENS_PER_GEN,
        max_trajectory_tokens=MAX_TRAJ_TOKENS,
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
        reward_fn=NIAHReward(gold_answer=problem.gold_answer),
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
    return ParentRollout(root=root, gold_answer=problem.gold_answer, problem=problem)


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
        f"{MAX_TRAJ_TOKENS}; chunk cap: {MAX_CHUNK_TOKENS}."
    )
    if len(corpus_tokens) < DOC_SIZE_TOKENS:
        raise SystemExit(
            f"Corpus too small ({len(corpus_tokens)} tokens) for DOC_SIZE_TOKENS={DOC_SIZE_TOKENS}."
        )

    def gen_problem(seed: int) -> NIAHProblem:
        return make_niah_problem(
            corpus_tokens=corpus_tokens,
            tokenizer=tokenizer,
            doc_size_tokens=DOC_SIZE_TOKENS,
            seed=seed,
        )

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

        # One NIAH problem per slot in the batch; GROUP_SIZE rollouts each.
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

        n_correct = sum(
            1
            for r in parent_results
            if r.root.answer is not None
            and _normalize_answer(r.root.answer) == _normalize_answer(r.gold_answer)
        )
        mean_reward = sum(rewards) / len(rewards)
        depth_counts = ", ".join(
            f"d{d}={trajectories_per_depth.get(d, 0)}"
            for d in sorted(trajectories_per_depth)
        )
        print(
            f"Step {step:2d} | mean_reward: {mean_reward:.3f} | "
            f"parent_correct: {n_correct}/{len(parent_results)} | "
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
        n_eval_correct = 0
        for ri, (problem, result) in enumerate(zip(eval_problems, eval_results)):
            nodes = result.all_nodes()
            parent_reward = _trajectory_total_reward(result.root.trajectory)
            is_correct = (
                result.root.answer is not None
                and _normalize_answer(result.root.answer)
                == _normalize_answer(result.gold_answer)
            )
            if is_correct:
                n_eval_correct += 1
            print()
            print("#" * 72)
            print(
                f"# Eval rollout {ri}  gold={result.gold_answer}  "
                f"needle_pos={problem.needle_position}/{len(problem.document_tokens)}  "
                f"nodes={len(nodes)}  parent_reward={parent_reward:.3f}  "
                f"correct={is_correct}"
            )
            print("#" * 72)
            print_rollout_tree_verbose(result.root, tokenizer)
        print()
        print(f"Eval: {n_eval_correct}/{EVAL_N_ROLLOUTS} correct")


if __name__ == "__main__":
    asyncio.run(main())
