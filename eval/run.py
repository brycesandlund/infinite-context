"""Multi-backend eval entry point.

Runs the shared recursive-agent loop (eval/agent.py) against a chosen backend
over a set of RULER tasks, and reports RULER-official scores per task — so you
can drop in Qwen (via Tinker), Claude, or GPT and compare on identical problems
through identical harness code.

Budget/recursion/data constants are imported from train.py (single source of
truth) so eval and training can't silently diverge on them.

Every rollout (full tree) is saved to $OUT.jsonl (structured) + $OUT.txt
(readable). For OOLONG tasks, results are also broken down per source dataset
and per question type, with no-answer / mean-tree-size health metrics.

Usage:
    uv run python -m eval.run                                  # uses config below
    CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) uv run python -m eval.run
    EVAL_TASKS=oolong_counting N_PER_TASK=10 TEMP=0.2 OUT=/tmp/probe uv run python -m eval.run
    BACKEND=anthropic/claude-sonnet-4-20250514 uv run python -m eval.run
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

import train  # single source of truth for budget/recursion/data constants
from eval.agent import AgentNode, flatten, run_agent, run_single_shot
from eval.backends import APIBackend, ModelBackend, TinkerBackend
from tasks import grade_answer, list_tasks, load_pg_essays_text, make_problem, resolve_eval_grading_mode
from tasks.oolong import make_oolong_problem, oolong_spec


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# "tinker" → the Tinker-hosted Qwen policy (base or LOAD_CHECKPOINT_PATH below).
# Anything else → a LiteLLM model string, e.g.:
#   "anthropic/claude-sonnet-4-20250514", "openai/gpt-5-mini"
BACKEND = os.environ.get("BACKEND", "tinker")

# "decompose" (default): the recursive 8K-budget agent harness — the model must read
# the doc via read_chunk and delegate (run_agent). "single": the whole document is put
# directly in context and the model answers in ONE tool-free call (run_single_shot) —
# the raw-ability ceiling (frontier single-shot, or an un-finetuned base model). Same
# problems, same grading; only the protocol differs. MODE=single ignores the budget.
MODE = os.environ.get("MODE", "decompose")
# Single-shot output cap (room for reasoning models to think before \boxed{}).
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", "16384"))

# Tinker-backend knobs (ignored for API backends).
# CKPT env var overrides — e.g. CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt)
TINKER_LOAD_CHECKPOINT_PATH: str | None = os.environ.get("CKPT") or None  # None = base model
TEMPERATURE = float(os.environ.get("TEMP", "1.0"))

# Which tasks to eval, and how many problems each (both env-overridable).
EVAL_TASKS = os.environ.get(
    "EVAL_TASKS", "oolong_counting,oolong_user,oolong_temporal"
).split(",")
N_PER_TASK = int(os.environ.get("N_PER_TASK", "5"))
SEED_OFFSET = 2_000_000          # RULER held-out seeds (OOLONG uses OOLONG_BASE below)
# OOLONG problems are indexed by the SHARED oolong_spec(task, idx, base). Eval
# defaults to a held-out base; set OOLONG_BASE to SFT's DATA_SEED (500000) to run
# the EXACT same problems SFT trained on.
OOLONG_BASE = int(os.environ.get("OOLONG_BASE", "2000000"))
# Only keep problems whose question type is in this set (csv); empty = all. Lets
# us target the HARD exact-count questions, e.g. QTYPE=numeric_one_class,represented_n_times.
QTYPE = set(filter(None, os.environ.get("QTYPE", "").split(",")))
CONCURRENCY = 4                  # max parent rollouts in flight (mind API rate limits)
VERBOSE = os.environ.get("VERBOSE", "0") == "1"   # also dump trees to stdout (all are saved regardless)
# EVERY rollout (full tree) is persisted here — OUT.jsonl (structured) + OUT.txt.
OUT = os.environ.get("OUT", "/tmp/eval_rollouts")


# Pull the shared harness constants from train.py so they can't drift. AGENT_CONTEXT
# and MAX_CHUNK_TOKENS are env-overridable so we can probe a backend's RAW capability
# at a task by lifting the budget (e.g. give an API model 50k so it can hold the whole
# doc and just answer — isolating "can it do the task" from "can it operate the 10k harness").
AGENT_CONTEXT = int(os.environ.get("AGENT_CONTEXT", train.AGENT_CONTEXT))
MAX_CHUNK_TOKENS = int(os.environ.get("MAX_CHUNK_TOKENS", train.MAX_CHUNK_TOKENS))
# MAX_DEPTH=none lifts the depth cap entirely (left-fold is a depth-#chunks chain, so
# depth must NOT be the binding constraint); the total-node cap below is the real
# runaway/speed guard. Default falls back to train.py.
_mdpth = os.environ.get("MAX_DEPTH")
MAX_DEPTH = (None if _mdpth.lower() == "none" else int(_mdpth)) if _mdpth else train.MAX_DEPTH
_mt = os.environ.get("MAX_TURNS")
MAX_TURNS = int(_mt) if _mt else train.MAX_TURNS  # default None = uncapped (budget terminates)
# Hard cap on total agents per rollout tree — kills runaway chains/cascades early and
# bounds per-rollout work (a clean binary tree @80K is ~500 nodes; overflow cascades hit
# 6000+). Set generously above legit trees; lower it to speed sampling.
MAX_NODES = int(os.environ.get("MAX_NODES", "2000"))
DOC_SIZE_TOKENS = int(os.environ.get("DOC_SIZE_TOKENS", train.DOC_SIZE_TOKENS))
# Tinker base model + renderer are env-overridable so we can probe ANY Tinker-hosted
# model (e.g. MODEL_NAME=moonshotai/Kimi-K2.6 RENDERER_NAME=kimi_k26) through the same
# harness — no checkpoint = base-model behaviour.
MODEL_NAME = os.environ.get("MODEL_NAME", train.MODEL_NAME)
RENDERER_NAME = os.environ.get("RENDERER_NAME", train.RENDERER_NAME)
LORA_RANK = int(os.environ.get("LORA_RANK", train.LORA_RANK))


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
# Verbose tree printer — shared with sft.py and train.py via eval/render.py.
# Underscore aliases preserved for existing imports.
# ---------------------------------------------------------------------------

from eval.render import (  # noqa: E402
    node_to_dict as _node_to_dict,
    print_tree as _print_tree,
    rollout_header as _rollout_header,
    tree_to_text as _tree_to_text,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    unknown = [t for t in EVAL_TASKS if t not in list_tasks()]
    if unknown:
        raise SystemExit(f"Unknown tasks in EVAL_TASKS: {unknown}. Available: {list_tasks()}")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    budget_str = "n/a (whole doc in context)" if MODE == "single" else str(AGENT_CONTEXT)
    print(f"Backend: {BACKEND} | mode: {MODE} | tasks: {EVAL_TASKS} | n/task: {N_PER_TASK} | "
          f"budget: {budget_str} | doc: {DOC_SIZE_TOKENS} | depth: {MAX_DEPTH}")
    print("Loading + tokenizing PG-essay corpus...")
    corpus_tokens = tokenizer.encode(load_pg_essays_text(), add_special_tokens=False)

    backend = await _build_backend(tokenizer)
    print(f"temp={TEMPERATURE} | out={OUT}.{{jsonl,txt}}")

    # Build (task, seed, problem) work items deterministically. OOLONG problems
    # come from the SHARED oolong_spec(task, idx, OOLONG_BASE) — so a given
    # (base, task, idx) is the IDENTICAL problem in SFT and eval. RULER keeps its
    # own deterministic seed scheme. With a QTYPE filter we oversample idx and keep
    # only matching question types (e.g. the hard exact-count ones).
    def _make(task: str, ti: int, idx: int):
        if task.startswith("oolong"):
            seed, ds = oolong_spec(task, idx, OOLONG_BASE)
            return seed, make_oolong_problem(
                task, corpus_tokens, tokenizer, DOC_SIZE_TOKENS, seed, dataset=ds)
        seed = SEED_OFFSET + ti * 1000 + idx
        return seed, make_problem(task, corpus_tokens, tokenizer, DOC_SIZE_TOKENS, seed)

    work: list[tuple[str, int, object]] = []
    for ti, task in enumerate(EVAL_TASKS):
        collected, idx = 0, 0
        while collected < N_PER_TASK and idx < 100_000:
            seed, problem = _make(task, ti, idx)
            idx += 1
            if QTYPE and problem.metadata.get("task_type") not in QTYPE:
                continue
            work.append((task, seed, problem))
            collected += 1
        if collected < N_PER_TASK:
            print(f"WARNING: only found {collected}/{N_PER_TASK} {task} problems "
                  f"matching QTYPE={QTYPE} in {idx} tries")

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(problem) -> AgentNode:
        async with sem:
            if MODE == "single":
                return await run_single_shot(
                    backend,
                    document_tokens=problem.document_tokens,
                    tokenizer=tokenizer,
                    task_context=problem.task_context,
                    question=problem.question,
                    max_output_tokens=OUT_TOKENS,
                )
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
                max_nodes=MAX_NODES,
            )

    nodes = await asyncio.gather(*[_one(p) for (_, _, p) in work])

    def _grounded(node: AgentNode) -> bool:
        """Did any agent in the tree actually read the document? An answer produced
        without a single read_chunk can only be a guess (binary/comparison questions
        pay ~0.5 EV for free), so we report score split on this. In MODE=single the
        whole document is already in context, so every answer is grounded by
        construction (there is no read_chunk to look for)."""
        if MODE == "single":
            return True
        return any(
            tc.name == "read_chunk"
            for n in flatten(node)
            for m in n.messages
            for tc in (m.get("tool_calls") or [])
        )

    # Score + persist EVERY rollout (structured JSONL + readable full trees).
    scores_by_task: dict[str, list[float]] = defaultdict(list)
    grounded_by_task: dict[str, list[float]] = defaultdict(list)  # score if grounded else 0
    by_dataset: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_qtype: dict[tuple[str, str], list[float]] = defaultdict(list)
    health: dict[str, list] = defaultdict(list)  # task -> list of (no_answer, tree_size)
    jsonl = open(f"{OUT}.jsonl", "w")
    txt = open(f"{OUT}.txt", "w")
    print()
    print("=" * 72)
    print(f"EVAL: backend={BACKEND} ckpt={TINKER_LOAD_CHECKPOINT_PATH or 'BASE'}")
    print("=" * 72)
    for (task, seed, problem), node in zip(work, nodes):
        score = grade_answer(node.answer, problem.gold_answers, resolve_eval_grading_mode(problem))
        grounded = _grounded(node)
        scores_by_task[task].append(score)
        grounded_by_task[task].append(score if grounded else 0.0)
        ds = problem.metadata.get("dataset")
        qt = problem.metadata.get("task_type")
        if ds is not None:
            by_dataset[(task, ds)].append(score)
        if qt is not None:
            by_qtype[(task, qt)].append(score)
        n_nodes = len(flatten(node))
        health[task].append((node.answer is None, n_nodes))
        jsonl.write(json.dumps({
            "task": task, "seed": seed, "dataset": ds, "qtype": qt,
            "question": problem.question, "gold": problem.gold_answers,
            "answer": node.answer, "score": score, "grounded": grounded,
            "n_agents": n_nodes, "root_termination": node.termination,
            "tree": _node_to_dict(node),
        }) + "\n")
        txt.write(_rollout_header(task, seed, ds, qt, problem.question,
                                  problem.gold_answers, node.answer, node.termination, score))
        txt.write(_tree_to_text(node))
        print(
            f"# task={task} seed={seed} dataset={ds} gold={problem.gold_answers} "
            f"nodes={n_nodes} answer={node.answer!r} term={node.termination} score={score:.3f}"
            + ("" if grounded else " UNGROUNDED(no reads)")
        )
        if VERBOSE:
            _print_tree(node)
    jsonl.close()
    txt.close()

    print()
    print("-" * 72)
    for t in sorted(scores_by_task):
        s = scores_by_task[t]
        g = grounded_by_task[t]
        no_ans = sum(1 for na, _ in health[t] if na)
        mean_tree = sum(sz for _, sz in health[t]) / len(health[t])
        print(f"{t}: {sum(s)/len(s):.3f}  (grounded {sum(g)/len(g):.3f} | {len(s)} rollouts | "
              f"no_answer={no_ans}/{len(s)} | mean_tree={mean_tree:.1f})")
        for (tt, ds), ss in sorted(by_dataset.items()):
            if tt == t:
                print(f"    dataset {ds:14s} {sum(ss)/len(ss):.3f}  (n={len(ss)})")
        for (tt, qt), ss in sorted(by_qtype.items()):
            if tt == t:
                print(f"    qtype   {qt:18s} {sum(ss)/len(ss):.3f}  (n={len(ss)})")
    alls = [s for ss in scores_by_task.values() for s in ss]
    allg = [s for ss in grounded_by_task.values() for s in ss]
    print(f"\nOVERALL: {sum(alls)/len(alls):.3f}  (grounded {sum(allg)/len(allg):.3f} | "
          f"{len(alls)} rollouts)")
    print(f"All {len(alls)} rollouts saved -> {OUT}.jsonl + {OUT}.txt")


if __name__ == "__main__":
    asyncio.run(main())
