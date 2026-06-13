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
from tinker_cookbook.renderers import get_renderer
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
import metrics  # optional W&B logging (no-op unless WANDB=1)
from eval.judge import make_judge  # LLM-as-a-judge for subagent credit assignment
from tasks import (
    GradingMode,
    Problem,
    grade_answer,
    list_tasks,
    load_pg_essays_text,
    make_problem,
    resolve_grading_mode,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
# disable_thinking: generation prefills an EMPTY <think> block — byte-identical to
# the prefix SFT datums condition on (oracle traces have no thinking), so train and
# sample distributions match. Free thinking would also eat the tight 8k budget.
RENDERER_NAME = "qwen3_5_disable_thinking"
LORA_RANK = 32
# 1.5e-5: run 1 went from healthy delegation to full collapse in ~3 steps at 4e-5 —
# the policy was moving too far per update for GRPO's noisy small-group advantages.
LEARNING_RATE = 1.5e-5

N_STEPS = 25               # full RL run from the SFT warm-start (overflow-reward collapse fixed)
BATCH_SIZE = 4             # problems per training step
GROUP_SIZE = 4              # parent rollouts per problem

# LLM-as-a-judge DENSE credit assignment for SUBAGENTS. The root keeps its exact
# gold reward (compute_reward); each non-root node is graded by the judge on its OWN
# subtask (verifying its answer against what it actually read), and its training
# advantage blends own-quality with the tree outcome:
#   subagent_adv = JUDGE_BETA*(judge_score - batch_mean_judge) + (1-JUDGE_BETA)*root_adv
# This fixes the shared-advantage problem: a subagent that did its job inside a
# failing tree no longer eats the tree's full negative advantage. JUDGE=0 disables
# it (every node falls back to the root advantage, i.e. prior behavior).
JUDGE_SUBAGENTS = os.environ.get("JUDGE", "1") == "1"
JUDGE_BETA = float(os.environ.get("JUDGE_BETA", "0.5"))   # 0 = pure outcome, 1 = pure own-quality
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "openai/gpt-5.4-nano")
JUDGE_SOURCE_CHARS = 24_000   # cap on the source text (chunk reads / child reports) shown to the judge
# Reasoning models burn hidden reasoning tokens against this budget; too small a cap
# truncates the verdict to empty (no SCORE -> parse fail). 16k gives ample headroom.
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "16000"))
MAX_DEPTH = 2               # 0 = root only; 2 = root may spawn children that may spawn grandchildren
MAX_TURNS = 8               # per-agent multi-turn cap

# A single knob: the per-agent context budget. Both the trajectory cap and the
# per-turn generation cap derive from this — TinkerTokenCompleter dynamically
# caps max_tokens = AGENT_CONTEXT - prompt.length, and the env terminates when
# the trajectory would exceed AGENT_CONTEXT. The model can think as much as it
# wants per turn, limited only by remaining budget.
AGENT_CONTEXT = 8_000

# Task knobs.
#
# TASK_MIXTURE is a weighted sampler over task names (must all be in list_tasks()).
# Each problem slot in a batch independently samples one task per these weights;
# all GROUP_SIZE rollouts of that slot share the same task+seed so GRPO's
# group-mean baseline still works (compares apples to apples within a problem).
#
# Set to a single-task dict to lock training to one task family — handy for
# debugging or ablations. Weights need not sum to 1; they're renormalized.
# OOLONG-only for this phase: the counting-only SFT primer made delegation
# samplable; RL's job is to make it dominant and to GENERALIZE it to the user and
# temporal families (which SFT deliberately skipped). Uncomment the RULER block
# to return to the full mixture once the recursive-aggregation loop is stable.
TASK_MIXTURE: dict[str, float] = {
    # 11 RULER training tasks (canonical names per NVIDIA/RULER/scripts/synthetic.yaml).
    # qa_1/qa_2 are held out for eval (require SQuAD+HotpotQA downloads and
    # we want a clean train/eval split).
    # "niah_single_1": 1.0, "niah_single_2": 1.0, "niah_single_3": 1.0,
    # "niah_multikey_1": 1.0, "niah_multikey_2": 1.0, "niah_multikey_3": 1.0,
    # "niah_multivalue": 1.0, "niah_multiquery": 1.0,
    # "vt": 1.0, "cwe": 1.0, "fwe": 1.0,
    "oolong_counting": 1.0, "oolong_user": 1.0,
    # temporal deferred: at ~0.005 success its groups are all-failure -> zero
    # variance -> zero gradient under flat failure rewards (pure compute cost).
    # Re-add once counting/user delegation is stable and transfer lifts it off 0.
    # "oolong_temporal": 1.0,
}
DOC_SIZE_TOKENS = 10_000    # haystack length per problem (> AGENT_CONTEXT, so read-it-all can't fit)
MAX_CHUNK_TOKENS = 6_000    # cap on a single read_chunk return; < DOC_SIZE so no single read covers the doc, and < AGENT_CONTEXT so a max read still leaves headroom to act on it.

DATA_SEED = 0               # base seed for problem generation; per-problem seed = DATA_SEED + step*BATCH_SIZE + idx

# Checkpointing. After training, save under this name (overwrite-safe). Set to None to skip saving.
# To resume: paste a tinker:// path into LOAD_CHECKPOINT_PATH below.
SAVE_CHECKPOINT_NAME: str | None = "final"
# Also save every SAVE_EVERY steps (0 = only at the end), so a multi-hour run that
# dies mid-way isn't a total loss — the last periodic save is eval-able and
# resumable. Periodic saves use name "step{N}" so they don't clobber each other or
# the final "final"; the latest path is always mirrored to LAST_CHECKPOINT_FILE.
SAVE_EVERY = 5
# CKPT env var overrides — e.g. CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) to warm-start RL from SFT.
LOAD_CHECKPOINT_PATH: str | None = os.environ.get("CKPT") or None  # tinker:// path; None = base model
RESUME_OPTIMIZER = False    # restore Adam momentum too. False when starting a fresh fine-tune from an SFT/base ckpt (the SFT optimizer state is for cross_entropy, not RL).
LAST_CHECKPOINT_FILE = Path.home() / ".cache" / "infinite-context" / "last_checkpoint.txt"
# Post-training eval lives in eval/run.py (run it on the saved checkpoint).

# Every training rollout (full tree) is appended here, rendered by the SAME
# eval/render.py printer as eval + SFT traces. "" disables.
ROLLOUT_DUMP = os.environ.get("ROLLOUT_DUMP", "/tmp/rl_rollouts.txt")

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
    # Tree-wide read counter (the instance is shared by every agent in the tree, and
    # children complete before the parent's final turn) — lets the reward check that
    # an answer is GROUNDED in at least one actual document read, not a blind guess.
    n_reads: int = 0

    @tool
    async def read_chunk(
        self,
        start: Annotated[int, "First token position to read (inclusive)."],
        end: Annotated[int, "Last token position to read (exclusive)."],
    ) -> ToolResult:
        """Read a slice of the document and return the decoded text of tokens [start, end). The document has a fixed length (stated in the system prompt). Each call is capped at the chunk limit; for larger ranges, issue multiple reads or delegate to a subagent."""
        self.n_reads += 1
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
            # Trajectory rewards are inert (training reward = compute_reward,
            # post-hoc); no overflow/parse constants to keep aligned.
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
        # Tell the parent WHY a subagent failed, so it can react (shrink the range
        # on overflow, retry on a non-answer) — matches eval/agent.py semantics.
        if answer is not None:
            result = answer
        elif any(
            (tr.metrics or {}).get("context_overflow") or (tr.metrics or {}).get("max_tokens_reached")
            for tr in child_trajectory.transitions
        ):
            result = "agent overflowed context"
        else:
            result = "agent did not box an answer"
        return simple_tool_result(result)


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


# Training reward is computed POST-HOC from the completed ParentRollout (see
# compute_reward below) — never from trajectory-embedded transition rewards.
#
# Why: the env assigns per-transition rewards through FOUR different channels
# depending on how the episode ends (reward_fn at a clean episode end; the
# context_overflow_reward constant on generation-length stop AND on next-prompt
# overflow; the failed_parse_reward constant on a renderer parse failure — see
# tinker_cookbook/rl/message_env.py:EnvFromMessageEnv.step). Keeping a reward
# POLICY consistent across four scattered assignment sites is how runs 1-3 each
# died to a different privileged failure mode. Post-hoc computation makes the
# env's reward plumbing inert: every termination path (overflow, parse error,
# stall, max_turns) simply yields answer=None and lands in the failure tier by
# construction. One function owns the policy.

FAILURE_REWARD = -0.1   # the single flat failure tier (see compute_reward)
MIN_CREDIT = 0.05       # numeric 0.75**|err| never reaches exactly 0; without a
                        # floor a FABRICATED number nicks epsilon credit and
                        # floats above failures. 0.05 <=> |err| <= ~10.


def compute_reward(parent: "ParentRollout") -> float:
    """Single source of truth for the training reward. Exactly TWO tiers:

      meaningful credit:  boxed answer + tree read the doc + score >= MIN_CREDIT
                          -> score (in [MIN_CREDIT, 1])
      everything else:    -> FAILURE_REWARD (flat)

    "Everything else" deliberately includes every failure mode — overflow, stall,
    no box, parse error, ungrounded guess, and a grounded-but-WRONG answer (run-3
    lesson: when subagents all failed, the root FABRICATED an answer; at 0.0 it
    out-ranked honest -0.1 failures and GRPO taught "make something up"). A wrong
    answer must be worth no more than no answer.

    Failure FLATNESS is the load-bearing property: an all-fail group has zero
    reward variance -> contributes no gradient (clean skip) instead of teaching a
    preferred failure style. (GRPO advantages are invariant to uniform shifts, so
    the -0.1 level itself is cosmetic; flatness is what matters.)

    Eval grading stays paper-faithful and ungated; eval/run.py reports raw and
    grounded scores separately.
    """
    if parent.root.answer is None or parent.n_reads == 0:
        return FAILURE_REWARD
    score = grade_answer(
        parent.root.answer, parent.gold_answers, resolve_grading_mode(parent.problem)
    )
    return score if score >= MIN_CREDIT else FAILURE_REWARD


async def judged_node_advantages(parent_results, root_advs, judge, tokenizer, renderer):
    """Per-node training advantages for a batch, with the judge scoring subagents.

    Returns (advs_per_parent, stats) where advs_per_parent[ri] is a list aligned
    with parent_results[ri].all_nodes() (DFS order):
      - root node (depth 0): the root's gold GRPO advantage (unchanged)
      - subagent node:        JUDGE_BETA*(judge_score - batch_mean) + (1-JUDGE_BETA)*root_adv
    The judge grades each subagent on its OWN subtask, verifying against the tool
    results in its own trajectory (the chunk it read and/or its children's reports).
    """
    from eval.agent import flatten as flatten_agent
    from eval.render import rollout_to_agent_node

    rollout_nodes = [pr.all_nodes() for pr in parent_results]
    agent_nodes = [
        flatten_agent(rollout_to_agent_node(pr.root, tokenizer, renderer)) for pr in parent_results
    ]

    # Collect every non-root node to grade, remembering where it came from.
    items: list[dict] = []
    refs: list[tuple[int, int]] = []
    for ri, (rns, ans) in enumerate(zip(rollout_nodes, agent_nodes)):
        for ni, (rn, an) in enumerate(zip(rns, ans)):
            if rn.depth == 0:
                continue
            source = "\n".join(m["content"] for m in an.messages if m.get("role") == "tool")
            items.append({
                "task": an.subtask or rn.subtask or "(no subtask)",
                "answer": rn.answer or "(no answer given)",
                "source": source[:JUDGE_SOURCE_CHARS] or None,
            })
            refs.append((ri, ni))

    verdicts = await judge.score_batch(items) if items else []
    # A judge failure (empty/truncated/no SCORE -> parsed=False) yields None, which
    # gets ADVANTAGE ZERO below (no gradient) — never a 0.0 reward that would punish
    # the subagent for the judge's infra hiccup. Failed nodes are also excluded from
    # the baseline so they don't skew it.
    jscore = {ref: (v.score if v.parsed else None) for ref, v in zip(refs, verdicts)}
    valid = [s for s in jscore.values() if s is not None]
    baseline = sum(valid) / len(valid) if valid else 0.0

    # advs aligned with all_nodes(); judges aligned too (None for the gold-graded root
    # AND for judge failures) so the rollout dump shows per-node accountability.
    advs_per_parent: list[list[float]] = []
    judge_per_parent: list[list[float | None]] = []
    for ri, rns in enumerate(rollout_nodes):
        advs, js = [], []
        for ni, rn in enumerate(rns):
            if rn.depth == 0:
                advs.append(root_advs[ri]); js.append(None)
            else:
                s = jscore.get((ri, ni))
                if s is None:                       # judge failed -> zero gradient
                    advs.append(0.0); js.append(None)
                else:
                    advs.append(JUDGE_BETA * (s - baseline) + (1.0 - JUDGE_BETA) * root_advs[ri])
                    js.append(s)
        advs_per_parent.append(advs)
        judge_per_parent.append(js)

    stats = {
        "judge_mean": baseline,
        "judge_parsed": len(valid),
        "judge_n": len(verdicts),
    }
    return advs_per_parent, judge_per_parent, stats


# ---------------------------------------------------------------------------
# Per-problem rollout
# ---------------------------------------------------------------------------


@dataclass
class ParentRollout:
    """One parent rollout = a tree of trajectories rooted at depth 0."""

    root: RolloutNode
    problem: Problem
    n_reads: int = 0  # read_chunk calls across the WHOLE tree (0 = ungrounded)

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
        # Trajectory-embedded rewards are INERT: the training reward is computed
        # post-hoc by compute_reward(ParentRollout) from (answer, n_reads, problem),
        # so every env termination channel (overflow constants, parse-failure
        # constant, reward_fn) is out of the gradient path by construction.
        reward_fn=_trivial_reward,
        max_turns=MAX_TURNS,
        max_trajectory_tokens=AGENT_CONTEXT,
        max_generation_tokens=1,  # see SubagentTool.spawn_subagent for rationale
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
    return ParentRollout(root=root, problem=problem, n_reads=read_chunk_tool.n_reads)




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

    judge = None
    if JUDGE_SUBAGENTS:
        if JUDGE_MODEL.startswith("openai/") and not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("JUDGE=1 needs OPENAI_API_KEY in the env (or set JUDGE=0).")
        judge = make_judge(JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS)
        print(f"Judge: {JUDGE_MODEL} | beta={JUDGE_BETA} | max_tokens={JUDGE_MAX_TOKENS} "
              f"(subagent dense credit assignment ON; judge-fail -> advantage 0)")

    print(f"Loaded model {MODEL_NAME}, renderer {RENDERER_NAME}, max_depth {MAX_DEPTH}")
    metrics.init(
        project="infinite-context",
        config={
            "phase": "rl", "model": MODEL_NAME, "renderer": RENDERER_NAME,
            "lora_rank": LORA_RANK, "lr": LEARNING_RATE, "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE, "group_size": GROUP_SIZE,
            "max_depth": MAX_DEPTH, "max_turns": MAX_TURNS,
            "agent_context": AGENT_CONTEXT, "doc_size": DOC_SIZE_TOKENS,
            "max_chunk": MAX_CHUNK_TOKENS, "task_mixture": TASK_MIXTURE,
            "warmstart": LOAD_CHECKPOINT_PATH, "resume_optimizer": RESUME_OPTIMIZER,
        },
    )
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
        problem = make_problem(
            task=task,
            corpus_tokens=corpus_tokens,
            tokenizer=tokenizer,
            doc_size_tokens=DOC_SIZE_TOKENS,
            seed=gen_seed,
        )
        problem.metadata["seed"] = seed  # the OUTER seed: reproduce via gen_problem(seed)
        return problem

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

    async def save_checkpoint(name: str) -> None:
        """Save training state under `name` and mirror the path to LAST_CHECKPOINT_FILE
        (so eval/resume can `cat` the latest). overwrite=True keeps Tinker storage from
        accumulating one blob per save when a name is reused."""
        print(f"Saving checkpoint '{name}'...")
        save_future = await training_client.save_state_async(name, overwrite=True)
        save_resp = await save_future.result_async()
        LAST_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_CHECKPOINT_FILE.write_text(save_resp.path)
        print(f"  saved: {save_resp.path}  (path -> {LAST_CHECKPOINT_FILE})")

    for step in range(N_STEPS):
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

        # SINGLE SOURCE OF TRUTH: reward computed post-hoc from the completed
        # rollout, never read out of trajectory transitions (the env writes those
        # through four different channels depending on termination path — see
        # compute_reward's docstring for the history of leaks that caused).
        rewards: list[float] = [compute_reward(r) for r in parent_results]
        per_problem_rewards: list[list[float]] = [[] for _ in problems]
        per_problem_indices: list[list[int]] = [[] for _ in problems]
        for ri, pi in enumerate(problem_idx_for_rollout):
            per_problem_rewards[pi].append(rewards[ri])
            per_problem_indices[pi].append(ri)

        # Root GRPO advantage (gold reward vs same-problem group mean).
        advantages: list[float] = [0.0] * len(parent_results)
        for pi in range(len(problems)):
            group_rewards = per_problem_rewards[pi]
            group_mean = sum(group_rewards) / len(group_rewards)
            for ri in per_problem_indices[pi]:
                advantages[ri] = rewards[ri] - group_mean

        # Per-node advantages. With the judge on, subagents get their own graded
        # advantage blended with the tree outcome; otherwise every node inherits the
        # root advantage (prior behavior).
        if JUDGE_SUBAGENTS and judge is not None:
            node_advs, node_judge, judge_stats = await judged_node_advantages(
                parent_results, advantages, judge, tokenizer, renderer
            )
        else:
            node_advs = [[advantages[ri]] * len(pr.all_nodes())
                         for ri, pr in enumerate(parent_results)]
            node_judge = [[None] * len(pr.all_nodes()) for pr in parent_results]
            judge_stats = {"judge_mean": 0.0, "judge_parsed": 0, "judge_n": 0}

        all_datums: list[tinker.Datum] = []
        trajectories_per_depth: dict[int, int] = {}
        for ri, parent_result in enumerate(parent_results):
            for ni, node in enumerate(parent_result.all_nodes()):
                if not node.trajectory.transitions:
                    continue
                all_datums.extend(trajectory_to_data(node.trajectory, node_advs[ri][ni]))
                trajectories_per_depth[node.depth] = trajectories_per_depth.get(node.depth, 0) + 1

        # Train if ANY per-node advantage is nonzero — with the judge on, a group
        # whose roots all tie (root_adv=0) can still have subagent signal.
        nonzero_adv = any(abs(a) > 1e-9 for advs in node_advs for a in advs)
        if all_datums and nonzero_adv:
            fwd_bwd_future = await training_client.forward_backward_async(
                [_strip_mask(d) for d in all_datums], loss_fn="importance_sampling"
            )
            optim_future = await training_client.optim_step_async(adam_params)
            await fwd_bwd_future.result_async()
            await optim_future.result_async()

        # Per-task aggregation of REWARD (the gated quantity the policy is actually
        # trained on — grounded success = score, any failure = -0.1), so by_task
        # tracks legitimate RL progress per family. Raw scores still appear as
        # mean_score; a score>>reward gap (or low grounded:) = guessing creep.
        # per_task_scores (raw grader) is kept for the W&B score/<task> curves.
        per_task_scores: dict[str, list[float]] = {}
        per_task_rewards: dict[str, list[float]] = {}
        for r, rew in zip(parent_results, rewards):
            score = grade_answer(r.root.answer, r.gold_answers, resolve_grading_mode(r.problem))
            per_task_scores.setdefault(r.task, []).append(score)
            per_task_rewards.setdefault(r.task, []).append(rew)
        per_task_summary = ", ".join(
            f"{t}={sum(s) / len(s):.2f}({len(s)})"
            for t, s in sorted(per_task_rewards.items())
        )
        mean_reward = sum(rewards) / len(rewards)
        mean_score = sum(
            s for scores in per_task_scores.values() for s in scores
        ) / len(parent_results)
        depth_counts = ", ".join(
            f"d{d}={trajectories_per_depth.get(d, 0)}"
            for d in sorted(trajectories_per_depth)
        )
        n_grounded = sum(1 for r in parent_results if r.n_reads > 0)
        judge_str = (
            f"judge: {judge_stats['judge_mean']:.2f}({judge_stats['judge_parsed']}/{judge_stats['judge_n']}) | "
            if judge_stats["judge_n"] else ""
        )
        print(
            f"Step {step:2d} | mean_reward: {mean_reward:.3f} | "
            f"mean_score: {mean_score:.3f} | "
            f"grounded: {n_grounded}/{len(parent_results)} | "
            f"{judge_str}"
            f"by_task: {per_task_summary} | "
            f"trajectories: {depth_counts} | "
            f"datums: {len(all_datums)} | "
            f"trained: {bool(all_datums and nonzero_adv)}"
        )

        # W&B: per-step curves. The two we care about most given our history are
        # score/<task> (does the new capability climb? is the mix balanced?) and
        # rollout/root_no_answer_rate (early-warning for the reward-collapse mode
        # where the policy stops reading and boxes fast guesses / overflows).
        root_no_answer = sum(
            1 for r in parent_results if r.root.answer is None
        ) / len(parent_results)
        mean_tree_size = sum(len(r.all_nodes()) for r in parent_results) / len(parent_results)
        mean_root_turns = sum(
            len(r.root.trajectory.transitions) for r in parent_results
        ) / len(parent_results)
        metrics.log(
            {
                "reward/mean": mean_reward,
                "score/mean": mean_score,
                **{f"score/{t}": sum(s) / len(s) for t, s in per_task_scores.items()},
                **{f"reward/{t}": sum(s) / len(s) for t, s in per_task_rewards.items()},
                "train/frac_nonzero_adv": sum(1 for a in advantages if abs(a) > 1e-9)
                / len(advantages),
                "train/datums": len(all_datums),
                "train/trained": int(bool(all_datums and nonzero_adv)),
                "rollout/grounded_rate": n_grounded / len(parent_results),
                "rollout/root_no_answer_rate": root_no_answer,
                "rollout/mean_tree_size": mean_tree_size,
                "rollout/mean_root_turns": mean_root_turns,
                **({"judge/mean_score": judge_stats["judge_mean"],
                    "judge/parse_rate": judge_stats["judge_parsed"] / judge_stats["judge_n"]}
                   if judge_stats["judge_n"] else {}),
                **{f"traj/d{d}": n for d, n in trajectories_per_depth.items()},
            },
            step=step,
        )

        # Persist EVERY training rollout (full tree, same renderer as eval/run.py
        # + sft.py) so the RL process is post-hoc inspectable — e.g. to see what
        # the policy actually did in the steps where behavior shifted. Set
        # ROLLOUT_DUMP="" to disable.
        if ROLLOUT_DUMP:
            from eval.render import rollout_header, rollout_to_agent_node, tree_to_text

            from eval.agent import flatten as flatten_agent

            with open(ROLLOUT_DUMP, "a") as f:
                for ri, (r, adv, rew) in enumerate(zip(parent_results, advantages, rewards)):
                    md = r.problem.metadata or {}
                    node = rollout_to_agent_node(r.root, tokenizer, renderer)
                    # Annotate each node with the credit it actually received, so
                    # print_tree shows per-trajectory adv + judge score inline.
                    for an, a_v, j_v in zip(flatten_agent(node), node_advs[ri], node_judge[ri]):
                        an.advantage, an.judge_score = a_v, j_v
                    f.write(f"\n@@@ STEP {step} rollout {ri} reward={rew:.3f} "
                            f"advantage={adv:+.3f}\n")
                    f.write(rollout_header(
                        r.task, md.get("seed", "?"), md.get("dataset"),
                        md.get("task_type"), r.problem.question, r.gold_answers,
                        r.root.answer, node.termination,
                        grade_answer(r.root.answer, r.gold_answers,
                                     resolve_grading_mode(r.problem)),
                    ))
                    f.write(tree_to_text(node))

        if DEBUG_PRINT_TREE_EACH_STEP:
            from eval.render import print_tree, rollout_to_agent_node

            for ri, parent_result in enumerate(parent_results):
                print(f"--- rollout {ri} (advantage={advantages[ri]:+.3f}) ---")
                print_tree(rollout_to_agent_node(parent_result.root, tokenizer, renderer))

        # Periodic checkpoint (insurance against a mid-run kill). Skip the last
        # step — the final save below covers it. Named per-step so a crashed run
        # leaves an eval-able artifact at the last completed multiple of SAVE_EVERY.
        if SAVE_EVERY and (step + 1) % SAVE_EVERY == 0 and step + 1 < N_STEPS:
            await save_checkpoint(f"step{step + 1}")

    # ------------------------------------------------------------------ #
    # Final checkpoint                                                   #
    # ------------------------------------------------------------------ #
    if SAVE_CHECKPOINT_NAME:
        await save_checkpoint(SAVE_CHECKPOINT_NAME)

    # Post-training eval lives in eval/run.py (the dedicated multi-backend eval
    # harness — grounded metric, per-dataset/qtype breakdowns, OUT.{jsonl,txt}):
    #   CKPT=$(cat ~/.cache/infinite-context/last_checkpoint.txt) uv run python -m eval.run
    metrics.finish()


if __name__ == "__main__":
    asyncio.run(main())
