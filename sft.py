"""SFT warm-start from scripted-oracle delegation traces.

Base Qwen sits in the 0-reward regime on this harness (it over-reads into
context overflow, or spawns chaotic subagent swarms), so RL-from-scratch has no
reward variance to learn from. This script teaches the *skeleton* of good
delegation first, giving RL a non-zero base to improve on.

Pipeline:
1. Generate gold traces with `OracleBackend` through the exact `run_agent` loop
   (pure CPU — the oracle plays scripted-optimal moves). Each trace is a tree of
   agents: a root that splits the doc into ranges + spawns a subagent per range,
   and subagents that each read once and report their range's findings.
2. Convert every agent's conversation to per-assistant-turn cross-entropy Datums
   (the Qwen3 renderer strips thinking from history, so we build one example per
   assistant turn with TrainOnWhat.LAST_ASSISTANT_MESSAGE rather than training
   multiple assistant messages in one example).
3. SFT: forward_backward(loss_fn="cross_entropy") + optim_step over epochs.
4. Save a checkpoint to warm-start RL from (set rl.py's LOAD_CHECKPOINT_PATH
   to it, with RESUME_OPTIMIZER=False).

Run: uv run python sft.py
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import tinker
import torch
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import datum_from_model_input_weights

import metrics  # optional W&B logging (no-op unless WANDB=1)
import rl  # shared constants + cookbook tool specs
from eval.agent import flatten, run_agent
from eval.backends import neutral_to_cookbook
from oracle import make_oracle
from eval.run import _rollout_header, _tree_to_text  # shared rollout renderer
from tasks import (
    grade_answer, list_tasks, load_pg_essays_text, make_problem, resolve_eval_grading_mode,
)
from tasks.oolong import make_oolong_problem, oolong_spec  # shared deterministic spec


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# MULTI-FAMILY warm-start. The SFT primer teaches ONE binary scaffold (split ->
# spawn -> combine) across DIVERSE leaf-ops so the model learns to infer the leaf
# operation from the task instead of overfitting one. Counting-only SFT generalized
# the *scaffold* to user/temporal (it recursed/split correctly) but ran the counting
# leaf-op everywhere -> spurious answers. The fix is leaf-op diversity: all three
# OOLONG families now DERIVE correctly under the binary oracle (counting: sum-by-
# label; user: per-user argmax; temporal: date-filtered before/after compare; each
# 10/10 exact), so all three are genuine teaching data. User/temporal roots carry
# more bookkeeping (per-user / per-date keys) but the aggregation is mechanical.
SFT_TASKS = os.environ.get(
    "SFT_TASKS", "oolong_counting,oolong_user,oolong_temporal"
).split(",")
# 150 traces (~15/dataset): MoE LoRA gets gradient only from tokens routed to each
# expert, so the sparse 35B-A3B needs materially more data than a dense model to
# absorb the same behavior; also widens prefix coverage against exposure bias.
N_PER_TASK = int(os.environ.get("N_PER_TASK", "150"))


def _parse_per_task(s: str) -> dict[str, int]:
    """Per-task trace-count overrides, e.g. "realdoc_count:50,bookqa:20". bookqa self-caps at
    its corpus ceiling regardless of the number."""
    out: dict[str, int] = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = int(v)
    return out


# Default bumps realdoc (real-prose counting — closest in-distribution analog to the OOLONG-
# counting eval); synths stay on N_PER_TASK. Override the whole map via the env var.
N_PER_TASK_OVERRIDE: dict[str, int] = _parse_per_task(
    os.environ.get("N_PER_TASK_OVERRIDE", "realdoc_count:50")
)
DATA_SEED = 500_000             # distinct from train/eval seed ranges
# Decomposition strategy for the synth_* tasks (the TRAINING knob). "mixed" = each
# task's favored default (binary for bounded, left_fold for stateful); "binary" or
# "left_fold" forces ALL synth tasks onto one strategy; "both" renders every bounded task
# BINARY and LEFT-FOLD (fold-native tasks stay fold), teaching fold as a GENERAL technique
# instead of only on runreset/varchain.
SYNTH_STRATEGY = os.environ.get("SYNTH_STRATEGY", "mixed")

# Synth tasks with no binary oracle (non-associative sequential state) — always left_fold.
_FOLD_ONLY_SYNTH = {"synth_runreset", "synth_varchain"}


def _synth_renderings(task: str, n: int) -> list[tuple[str | None, int]]:
    """(strategy, count) renderings for `task` given SYNTH_STRATEGY. Non-synth tasks get one
    default rendering. "both" splits a bounded task's N across binary + fold so the model sees
    left-fold applied to many leaf-ops, not just the two fold-native ones."""
    if not task.startswith("synth_"):
        return [(None, n)]
    if SYNTH_STRATEGY == "both":
        if task in _FOLD_ONLY_SYNTH:
            return [("left_fold", n)]
        return [("binary", n // 2), ("left_fold", n - n // 2)]
    if SYNTH_STRATEGY == "mixed":
        return [(None, n)]                 # each task's own default
    return [(SYNTH_STRATEGY, n)]           # forced binary / left_fold


# BookQA's leaf judgment ("does this prose answer the question?") is reading comprehension,
# not a mechanical reduce, so its oracle delegates the leaf to a real model (ModelBackend)
# and we REJECTION-SAMPLE: keep a trace only if its boxed answer matches gold. This filters
# both wrong model reads and noisy gold. Set BOOKQA_LEAF_MODEL to a litellm id to enable;
# the model is swappable precisely because it's just a ModelBackend behind make_oracle.
BOOKQA_LEAF_MODEL = os.environ.get("BOOKQA_LEAF_MODEL", "").strip()
BOOKQA_LEAF_TEMP = float(os.environ.get("BOOKQA_LEAF_TEMP", "0"))
REJECT_ACCEPT_MIN = float(os.environ.get("REJECT_ACCEPT_MIN", "1.0"))   # keep iff score >= this
BOOKQA_GEN_CONCURRENCY = int(os.environ.get("BOOKQA_GEN_CONCURRENCY", "8"))  # in-flight traces
_REJECT_SAMPLE_TASKS = {"bookqa", "narrativeqa"}   # need the model leaf + gold rejection
# Tasks with answer/none leaf VERDICTS to rebalance (the class-imbalance fix). niah_novel is
# scripted (not reject-sampled) but has the same 1-answer : many-none shape, and the user
# wants the needle (answer) leaf oversampled — so it's here but not in the reject set.
_QA_VERDICT_TASKS = {"bookqa", "narrativeqa", "niah_novel"}


def _make_leaf_model():
    """The bookqa leaf model — any litellm-supported chat model, behind the ModelBackend ABC."""
    if not BOOKQA_LEAF_MODEL:
        return None
    from eval.backends import APIBackend
    return APIBackend(BOOKQA_LEAF_MODEL, temperature=BOOKQA_LEAF_TEMP)


# Class-imbalance fix for the QA leaf signal. Each KEPT QA trace has ~1 answer-bearing verdict
# on the path to the root but MANY no-answer ("none") verdicts (the chunks without the answer),
# so uniform SFT teaches the model to over-abstain — which then tanks retrieval (RULER
# niah/cwe/fwe collapse to "None"). Resample the VERDICT turns toward parity: keep only a
# fraction of no-answer verdicts, duplicate answer verdicts. Scoped to QA tasks only (the
# read/split/combine turns and all synth/realdoc turns stay uniform).
# Base ratio in TRAINING traces is ~1:2 pos:neg (the answer often recurs across chunks, so
# several leaves fire) — not the ~19:1 the eval abstention implied. So a MODERATE pos-lean,
# not a heavy one. This ratio is the key knob to tune against the next eval's abstention rate:
# too high over-fires (hurts precision via the first-answer-wins combine), too low keeps the
# over-abstention. Default ~2:1 pos:neg after resampling.
QA_NONE_KEEP = float(os.environ.get("QA_NONE_KEEP", "0.5"))   # fraction of no-answer verdicts kept
QA_ANSWER_DUP = int(os.environ.get("QA_ANSWER_DUP", "2"))     # copies of each answer verdict


def _qa_verdict_class(text: str) -> str:
    """Classify a QA leaf/combine turn: 'neg' = abstain/no-answer, 'pos' = answer-bearing,
    'normal' = a read/split/plumbing turn (left uniform). Only meaningful for QA tasks."""
    t = text.lower()
    if ("no relevant information in this range" in t
            or "neither half had relevant information" in t
            or "\\boxed{none}" in t):
        return "neg"
    if ("\\boxed{answer" in t or "found the answer" in t
            or "located the answer" in t or "the answer is" in t):
        return "pos"
    return "normal"


EPOCHS = 1                      # 1 epoch over MORE data beats 2 over little: the 2nd
                                # epoch on 30 traces bought NLL via surface memorization
                                # (e.g. degenerate '\'-loops at sampling). With 150
                                # traces a single pass captures the pattern.
SFT_BATCH_SIZE = 16             # datums per optim step
LEARNING_RATE = 1e-5

SAVE_CHECKPOINT_NAME = os.environ.get("SAVE_NAME", "sft_general")   # output checkpoint name
# (just the save-state label — SFT always trains a FRESH LoRA from base_model, no warm-start)
LAST_SFT_CHECKPOINT_FILE = Path.home() / ".cache" / "infinite-context" / "last_sft_checkpoint.txt"

# Debug toggles (env-overridable):
#   TRAIN=0          -> generate (+optionally print) oracle traces, but SKIP the
#                       Tinker training/save. Pure-CPU dry run to inspect the data.
#   PRINT_TRACES=1   -> print every oracle trace we build SFT datums FROM, using
#                       the eval tree renderer (full text, read_chunk clipped),
#                       and save them to $TRACE_OUT.txt. This is the ground-truth
#                       behaviour we're teaching — check it's actually clean.
TRAIN = os.environ.get("TRAIN", "1") == "1"
PRINT_TRACES = os.environ.get("PRINT_TRACES", "0") == "1"
TRACE_OUT = os.environ.get("TRACE_OUT", "/tmp/sft_traces")

# Shared with training / eval (single source of truth).
MODEL_NAME = rl.MODEL_NAME
RENDERER_NAME = rl.RENDERER_NAME
LORA_RANK = rl.LORA_RANK
AGENT_CONTEXT = rl.AGENT_CONTEXT
MAX_CHUNK_TOKENS = rl.MAX_CHUNK_TOKENS
DOC_SIZE_TOKENS = rl.DOC_SIZE_TOKENS
MAX_DEPTH = rl.MAX_DEPTH
MAX_TURNS = rl.MAX_TURNS


# ---------------------------------------------------------------------------
# Trace generation (CPU — no sampling client needed; the oracle is scripted)
# ---------------------------------------------------------------------------


# Temporal subtypes still excluded from SFT: date_most/date_2nd ("which date is
# represented most often") render gold as a dateobj we can't reliably format-match,
# so the oracle would gold-leak. (date_ntimes was previously here too, but we fixed
# its root cause — the upstream [:50] cap — in the generator, so it now derives the
# literal answer and is trainable.) We skip these and draw the next index instead.
_SKIP_TMODES = {"date_most", "date_2nd"}


def _make_sft_problem(task, ti, i, corpus_tokens, tokenizer):
    """Deterministic (task, idx) -> problem. OOLONG uses the shared oolong_spec (same
    problem as eval by seed); synth/ruler use make_problem with a per-task seed range."""
    if task.startswith("oolong"):
        seed, dataset = oolong_spec(task, i, DATA_SEED)
        return seed, make_oolong_problem(
            task, corpus_tokens, tokenizer, DOC_SIZE_TOKENS, seed, dataset=dataset
        )
    seed = DATA_SEED + ti * 100_000 + i
    return seed, make_problem(task, corpus_tokens, tokenizer, DOC_SIZE_TOKENS, seed)


async def _one_trace(oracle, problem, tokenizer):
    return await run_agent(
        oracle,
        document_tokens=problem.document_tokens,
        tokenizer=tokenizer,
        task_context=problem.task_context,
        question=problem.question,
        budget=AGENT_CONTEXT,
        max_chunk_tokens=MAX_CHUNK_TOKENS,
        max_depth=MAX_DEPTH,
        max_turns=MAX_TURNS,
    )


async def _collect_scripted(task, ti, strategy, want, corpus_tokens, tokenizer, start_idx=0):
    """Scripted oracles (synth / oolong / realdoc): every trace solves by construction, so
    build `want` and run them concurrently — no grading/rejection needed. `start_idx` lets the
    two "both" renderings of one task draw disjoint problems."""
    coros, meta, i, collected = [], [], start_idx, 0
    while collected < want:
        seed, problem = _make_sft_problem(task, ti, i, corpus_tokens, tokenizer)
        i += 1
        oracle = make_oracle(
            problem, tokenizer,
            budget=AGENT_CONTEXT, max_chunk_tokens=MAX_CHUNK_TOKENS,
            strategy=strategy,
        )
        if getattr(oracle, "tmode", None) in _SKIP_TMODES:
            continue  # un-trainable gold — skip, try the next index
        collected += 1
        coros.append(_one_trace(oracle, problem, tokenizer))
        meta.append((task, seed, problem))
    nodes = await asyncio.gather(*coros)
    return list(zip(meta, nodes))


async def _collect_rejection(task, ti, want, corpus_tokens, tokenizer, leaf_model):
    """Model-driven oracle (bookqa): the leaf answer can be wrong (or the gold noisy), so grade
    each trace against gold and KEEP only the matches. Selection is 1:1 seed->question, so we
    walk DISTINCT questions once each (no duplicates, no wraparound) up to the corpus size, in
    concurrency-capped waves, harvesting every accepted trace."""
    from tasks.bookqa.generators import bookqa_corpus_size
    size = bookqa_corpus_size(task)
    out, i, tries = [], 0, 0
    while len(out) < want and i < size:
        wave = min(BOOKQA_GEN_CONCURRENCY, size - i)
        batch = []   # (meta, oracle, problem)
        for _ in range(wave):
            seed, problem = _make_sft_problem(task, ti, i, corpus_tokens, tokenizer)
            i += 1
            tries += 1
            oracle = make_oracle(
                problem, tokenizer,
                budget=AGENT_CONTEXT, max_chunk_tokens=MAX_CHUNK_TOKENS,
                leaf_model=leaf_model,
            )
            batch.append(((task, seed, problem), oracle, problem))
        nodes = await asyncio.gather(*[_one_trace(o, p, tokenizer) for _, o, p in batch])
        for (meta, oracle, problem), node in zip(batch, nodes):
            if len(out) >= want:
                break
            # Explicitly drop the oracle's abstention (belt-and-suspenders vs a gold word that
            # happens to occur in the NO_ANSWER sentinel); then require the gold gate.
            if not node.answer or node.answer.strip() == oracle.NO_ANSWER:
                continue
            sc = grade_answer(node.answer, problem.gold_answers, resolve_eval_grading_mode(problem))
            if sc >= REJECT_ACCEPT_MIN:
                out.append((meta, node))
    rate = len(out) / max(tries, 1)
    short = "" if len(out) >= want else f"  (exhausted {size} distinct questions)"
    print(f"  [{task}] kept {len(out)}/{want} distinct, accept {rate:.0%}{short}")
    return out


async def _gen_traces(corpus_tokens, tokenizer):
    leaf_model = _make_leaf_model()
    if any(t in _REJECT_SAMPLE_TASKS for t in SFT_TASKS) and leaf_model is None:
        raise SystemExit(
            "bookqa/narrativeqa SFT needs a leaf model — set BOOKQA_LEAF_MODEL=<litellm id> "
            "(e.g. anthropic/claude-haiku-4-5-20251001). The scripted span leaf is NOT "
            "faithful for free-form QA, so we refuse to train on it."
        )
    out = []
    for ti, task in enumerate(SFT_TASKS):
        want = N_PER_TASK_OVERRIDE.get(task, N_PER_TASK)
        if task in _REJECT_SAMPLE_TASKS:
            out += await _collect_rejection(task, ti, want, corpus_tokens, tokenizer, leaf_model)
            continue
        # Expand synth tasks into strategy renderings (binary / fold / both); disjoint start
        # indices keep the two "both" renderings on different problems.
        for k, (strat, cnt) in enumerate(_synth_renderings(task, want)):
            out += await _collect_scripted(
                task, ti, strat, cnt, corpus_tokens, tokenizer, start_idx=k * 100_000
            )
    return out


# ---------------------------------------------------------------------------
# Trace -> SFT Datums
# ---------------------------------------------------------------------------


def _node_to_datums(node, renderer, tool_specs, is_qa: bool = False) -> list[tuple]:
    """One (Datum, klass) per assistant turn in this agent's conversation. `klass` is the QA
    verdict class (pos/neg/normal) used by main() to resample the imbalanced QA leaf signal;
    it's always 'normal' for non-QA tasks.

    Per-turn (LAST_ASSISTANT_MESSAGE) rather than ALL_ASSISTANT_MESSAGES because
    the Qwen3 renderer lacks the extension property (it strips thinking from
    history), so each assistant turn must be rendered with its own real prefix.
    """
    cb = neutral_to_cookbook(node.messages, renderer, tool_specs)
    out: list[tuple] = []   # (datum, klass) — klass drives QA verdict resampling in main()
    for i, m in enumerate(cb):
        if m.get("role") != "assistant":
            continue
        model_input, weights = renderer.build_supervised_example(
            cb[: i + 1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        if float(weights.sum()) == 0.0:  # nothing trainable (shouldn't happen)
            continue
        datum = datum_from_model_input_weights(
            model_input, weights, max_length=AGENT_CONTEXT, reduction="mean"
        )
        klass = _qa_verdict_class(m.get("content") or "") if is_qa else "normal"
        out.append((datum, klass))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    unknown = [t for t in SFT_TASKS if t not in list_tasks()]
    if unknown:
        raise SystemExit(f"Unknown SFT_TASKS: {unknown}. Available: {list_tasks()}")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    tool_specs = [
        rl.ReadChunkTool.read_chunk.to_spec(),
        rl.SubagentTool.spawn_subagent.to_spec(),
    ]

    print(f"SFT warm-start | tasks={len(SFT_TASKS)} x {N_PER_TASK} | "
          f"doc={DOC_SIZE_TOKENS} budget={AGENT_CONTEXT} epochs={EPOCHS}")
    print("Loading + tokenizing PG-essay corpus...")
    corpus_tokens = tokenizer.encode(load_pg_essays_text(), add_special_tokens=False)

    print("Generating oracle traces (scripted, CPU)...")
    traces = await _gen_traces(corpus_tokens, tokenizer)

    # Optionally dump every oracle trace we train FROM — the ground-truth
    # behaviour being taught. Uses the eval tree renderer (full text, read_chunk
    # clipped) and grades the oracle's answer as a sanity check that it solved.
    if PRINT_TRACES:
        with open(f"{TRACE_OUT}.txt", "w") as tf:
            for (task, seed, problem), node in traces:
                sc = grade_answer(node.answer, problem.gold_answers, resolve_eval_grading_mode(problem))
                ds = problem.metadata.get("dataset")
                qt = problem.metadata.get("task_type")
                hdr = _rollout_header(task, seed, ds, qt, problem.question,
                                      problem.gold_answers, node.answer, node.termination, sc)
                tf.write(hdr)
                tf.write(_tree_to_text(node))
        print(f"Printed {len(traces)} oracle traces -> {TRACE_OUT}.txt")

    # Convert to (datum, klass); also sanity-check oracle traces actually solved the task.
    tagged: list[tuple] = []
    n_agents = 0
    for (_task, _seed, _problem), node in traces:
        is_qa = _task in _QA_VERDICT_TASKS
        for agent in flatten(node):
            n_agents += 1
            tagged.extend(_node_to_datums(agent, renderer, tool_specs, is_qa=is_qa))

    # Resample the QA verdict imbalance: keep only a stride-fraction of no-answer verdicts,
    # duplicate each answer verdict. Deterministic (stride) so it's reproducible.
    datums: list[tinker.Datum] = []
    neg_stride = max(1, round(1.0 / QA_NONE_KEEP)) if QA_NONE_KEEP > 0 else 1
    n_pos = n_neg_keep = n_neg_drop = neg_i = 0
    for datum, klass in tagged:
        if klass == "pos":
            datums.extend([datum] * QA_ANSWER_DUP); n_pos += 1
        elif klass == "neg":
            if neg_i % neg_stride == 0:
                datums.append(datum); n_neg_keep += 1
            else:
                n_neg_drop += 1
            neg_i += 1
        else:
            datums.append(datum)
    print(f"Traces: {len(traces)} | agents: {n_agents} | datums: {len(datums)} "
          f"(QA verdicts: pos={n_pos}x{QA_ANSWER_DUP}, neg kept {n_neg_keep}/{n_neg_keep + n_neg_drop})")
    if not datums:
        raise SystemExit("No datums produced — check oracle trace generation.")

    if not TRAIN:
        print("TRAIN=0: skipping Tinker training/save (dry run). "
              f"{'Traces at ' + TRACE_OUT + '.txt' if PRINT_TRACES else 'Set PRINT_TRACES=1 to inspect traces.'}")
        return

    # Training client.
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )
    adam_params = tinker.AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95)

    metrics.init(
        project="infinite-context",
        config={
            "phase": "sft", "model": MODEL_NAME, "renderer": RENDERER_NAME,
            "lora_rank": LORA_RANK, "lr": LEARNING_RATE, "epochs": EPOCHS,
            "sft_batch_size": SFT_BATCH_SIZE, "n_datums": len(datums),
            "tasks": SFT_TASKS, "n_per_task": N_PER_TASK,
            "n_per_task_override": N_PER_TASK_OVERRIDE,
        },
    )

    rng = random.Random(0)
    global_batch = 0
    for epoch in range(EPOCHS):
        rng.shuffle(datums)
        n_batches = (len(datums) + SFT_BATCH_SIZE - 1) // SFT_BATCH_SIZE
        epoch_nll = 0.0
        n_logged = 0
        for b in range(n_batches):
            batch = datums[b * SFT_BATCH_SIZE : (b + 1) * SFT_BATCH_SIZE]
            fwd_bwd = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
            optim = await training_client.optim_step_async(adam_params)
            fb_result = await fwd_bwd.result_async()
            await optim.result_async()
            # weighted-mean NLL for logging (best-effort; never block training on it)
            try:
                lp = [o["logprobs"] for o in fb_result.loss_fn_outputs]
                w = [d.loss_fn_inputs["weights"] for d in batch]
                num = sum(float(l.to_torch().dot(wi.to_torch())) for l, wi in zip(lp, w))
                den = sum(float(wi.to_torch().sum()) for wi in w)
                if den:
                    batch_nll = -(num / den)
                    epoch_nll += batch_nll
                    n_logged += 1
                    metrics.log({"sft/batch_nll": batch_nll, "sft/epoch": epoch}, step=global_batch)
            except Exception:
                pass
            global_batch += 1
        nll_str = f"{epoch_nll / n_logged:.4f}" if n_logged else "n/a"
        print(f"Epoch {epoch}: batches {n_batches} | mean batch NLL {nll_str}")
        if n_logged:
            metrics.log({"sft/epoch_nll": epoch_nll / n_logged}, step=global_batch)

        # Checkpoint after EVERY epoch (overwrite). Tinker runs are slow (~15
        # min/epoch) and occasionally drop connection, so a kill/crash should
        # never cost a full redo — last_sft_checkpoint.txt always points at the
        # latest completed epoch's weights, which are usable for RL warm-start.
        print(f"  saving checkpoint '{SAVE_CHECKPOINT_NAME}' (after epoch {epoch})...")
        save_future = await training_client.save_state_async(SAVE_CHECKPOINT_NAME, overwrite=True)
        save_resp = await save_future.result_async()
        LAST_SFT_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_SFT_CHECKPOINT_FILE.write_text(save_resp.path)
        print(f"  saved: {save_resp.path}  (path -> {LAST_SFT_CHECKPOINT_FILE})")
    print("\nTo warm-start RL: set rl.py LOAD_CHECKPOINT_PATH to the path above "
          "and RESUME_OPTIMIZER=False.")
    metrics.finish()


if __name__ == "__main__":
    asyncio.run(main())
