"""Multi-backend eval entry point.

Runs the shared recursive-agent loop (eval/agent.py) against a chosen backend
over a set of RULER tasks, and reports RULER-official scores per task — so you
can drop in Qwen (via Tinker), Claude, or GPT and compare on identical problems
through identical harness code.

Budget/recursion/data constants are imported from train.py (single source of
truth) so eval and training can't silently diverge on them.

Usage:
    uv run python -m eval.run                      # uses BACKEND below
    BACKEND=anthropic/claude-sonnet-4-20250514 uv run python -m eval.run
"""

from __future__ import annotations

import asyncio
import os
import random

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

import train  # single source of truth for budget/recursion/data constants
from eval.agent import AgentNode, flatten, run_agent
from eval.backends import APIBackend, ModelBackend, TinkerBackend
from tasks import eval_grading_mode, grade_answer, list_tasks, load_pg_essays_text, make_problem


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "tinker" → the Tinker-hosted Qwen policy (base or LOAD_CHECKPOINT_PATH below).
# Anything else → a LiteLLM model string, e.g.:
#   "anthropic/claude-sonnet-4-20250514", "openai/gpt-5-mini"
BACKEND = os.environ.get("BACKEND", "tinker")

# Tinker-backend knobs (ignored for API backends).
TINKER_LOAD_CHECKPOINT_PATH: str | None = None  # None = base model
TEMPERATURE = 1.0

# Which tasks to eval, and how many problems each.
EVAL_TASKS = ["niah_single_2", "niah_multiquery", "vt", "cwe"]
N_PER_TASK = 1
SEED_OFFSET = 2_000_000          # held-out seeds, distinct from train/eval-in-train
CONCURRENCY = 4                  # max parent rollouts in flight (mind API rate limits)
VERBOSE = True                   # dump full transcripts


# Pull the shared harness constants from train.py so they can't drift.
AGENT_CONTEXT = train.AGENT_CONTEXT
MAX_DEPTH = train.MAX_DEPTH
MAX_TURNS = train.MAX_TURNS
MAX_CHUNK_TOKENS = train.MAX_CHUNK_TOKENS
DOC_SIZE_TOKENS = train.DOC_SIZE_TOKENS
MODEL_NAME = train.MODEL_NAME
RENDERER_NAME = train.RENDERER_NAME
LORA_RANK = train.LORA_RANK


# ---------------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------------


async def _build_backend(tokenizer) -> ModelBackend:
    if BACKEND == "tinker":
        import tinker

        service_client = tinker.ServiceClient()
        training_client = await service_client.create_lora_training_client_async(
            base_model=MODEL_NAME, rank=LORA_RANK
        )
        if TINKER_LOAD_CHECKPOINT_PATH:
            print(f"Loading checkpoint: {TINKER_LOAD_CHECKPOINT_PATH}")
            fut = await training_client.load_state_async(TINKER_LOAD_CHECKPOINT_PATH)
            await fut.result_async()
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
        renderer = get_renderer(RENDERER_NAME, tokenizer)
        return TinkerBackend(sampling_client, tokenizer, renderer, temperature=TEMPERATURE)

    return APIBackend(BACKEND, temperature=TEMPERATURE)


# ---------------------------------------------------------------------------
# Verbose tree printer
# ---------------------------------------------------------------------------


def _print_tree(node: AgentNode, indent: int = 0) -> None:
    prefix = "  " * indent
    bar = "=" * max(8, 76 - len(prefix))
    print(f"{prefix}{bar}")
    print(
        f"{prefix}[depth={node.depth}] turns={node.n_turns} "
        f"termination={node.termination} answer={node.answer!r}"
    )
    if node.subtask:
        print(f"{prefix}SUBTASK: {node.subtask}")
    print(f"{prefix}{bar}")
    for m in node.messages:
        role = m["role"]
        raw_content = m.get("content") or ""
        content = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip()
        if role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m["tool_calls"])
            print(f"{prefix}[assistant] {content[:500]}")
            print(f"{prefix}  -> CALLS: {calls}")
        elif role == "tool":
            snippet = content if len(content) <= 300 else content[:300] + " …"
            print(f"{prefix}[tool:{m.get('name')}] {snippet}")
        else:
            shown = content if len(content) <= 800 else content[:800] + " …"
            print(f"{prefix}[{role}] {shown}")
    print()
    for c in node.children:
        _print_tree(c, indent + 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    unknown = [t for t in EVAL_TASKS if t not in list_tasks()]
    if unknown:
        raise SystemExit(f"Unknown tasks in EVAL_TASKS: {unknown}. Available: {list_tasks()}")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    print(f"Backend: {BACKEND} | tasks: {EVAL_TASKS} | n/task: {N_PER_TASK} | "
          f"budget: {AGENT_CONTEXT} | doc: {DOC_SIZE_TOKENS} | depth: {MAX_DEPTH}")
    print("Loading + tokenizing PG-essay corpus...")
    corpus_tokens = tokenizer.encode(load_pg_essays_text(), add_special_tokens=False)

    backend = await _build_backend(tokenizer)

    # Build (task, problem) work items.
    work: list[tuple[str, int, object]] = []
    for task in EVAL_TASKS:
        for i in range(N_PER_TASK):
            seed = SEED_OFFSET + abs(hash(task)) % 10_000 + i
            problem = make_problem(task, corpus_tokens, tokenizer, DOC_SIZE_TOKENS, seed)
            work.append((task, seed, problem))

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(problem) -> AgentNode:
        async with sem:
            return await run_agent(
                backend,
                document_tokens=problem.document_tokens,
                tokenizer=tokenizer,
                task_context=problem.task_context,
                question=problem.question,
                budget=AGENT_CONTEXT,
                max_chunk_tokens=MAX_CHUNK_TOKENS,
                max_depth=MAX_DEPTH,
                max_turns=MAX_TURNS,
            )

    nodes = await asyncio.gather(*[_one(p) for (_, _, p) in work])

    # Score.
    scores_by_task: dict[str, list[float]] = {}
    print()
    print("=" * 72)
    print(f"EVAL: backend={BACKEND}")
    print("=" * 72)
    for (task, seed, problem), node in zip(work, nodes):
        score = grade_answer(node.answer, problem.gold_answers, eval_grading_mode(task))
        scores_by_task.setdefault(task, []).append(score)
        n_nodes = len(flatten(node))
        print(
            f"\n# task={task} seed={seed} gold={problem.gold_answers} "
            f"doc_tokens={len(problem.document_tokens)} nodes={n_nodes} "
            f"answer={node.answer!r} term={node.termination} score={score:.3f}"
        )
        if VERBOSE:
            _print_tree(node)

    print()
    print("-" * 72)
    for t in sorted(scores_by_task):
        s = scores_by_task[t]
        print(f"{BACKEND}  {t}: ruler {sum(s)/len(s):.3f}  ({len(s)} rollouts)")
    alls = [s for ss in scores_by_task.values() for s in ss]
    print(f"{BACKEND}  OVERALL: ruler {sum(alls)/len(alls):.3f}  ({len(alls)} rollouts)")


if __name__ == "__main__":
    asyncio.run(main())
