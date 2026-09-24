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
import hashlib
import json
import dataclasses
import os
import random
import time
from pathlib import Path

import tinker
import torch
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import datum_from_model_input_weights

import logging
logging.getLogger("tinker_cookbook.renderers.base").setLevel(logging.ERROR)   # ALL_ASSISTANT_MESSAGES warning; see DATUM_MODE
import metrics  # optional W&B logging (no-op unless WANDB=1)
import rl  # shared constants + cookbook tool specs
from eval.agent import AgentNode, flatten, run_agent
from eval.backends import ToolCall, neutral_to_cookbook
from oracle import make_oracle
from eval.run import _rollout_header, _tree_to_text  # shared rollout renderer
from tasks import (
    grade_answer, list_tasks, load_pg_essays_text, make_problem, resolve_eval_grading_mode,
)
from tasks.oolong import OOLONG_TASKS, make_oolong_problem, oolong_spec  # shared deterministic spec


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
# "left_fold" forces ALL synth tasks onto one strategy; "both" = binary by default, fold where
# the op REQUIRES it, plus a small fold share on scalar-state ops (see below).
SYNTH_STRATEGY = os.environ.get("SYNTH_STRATEGY", "mixed")


def _synth_renderings(task: str, n: int) -> list[tuple[str | None, int]]:
    """(strategy, count) renderings for `task`. The strategy is a PROPERTY OF THE TASK, not a
    training knob: order-dependent (sequential) tasks left-fold, everything else splits binary
    (run 5 post-mortem — rendering order-independent ops as fold taught "fold and binary are
    interchangeable", and the root then drifted both ways: fold on a big tally (cwe, run 4) and
    binary-collect on variable tracking (vt, run 5)). Each oracle's `strategy_default` carries the
    task's strategy and the root preamble states the reason. SYNTH_STRATEGY=binary|left_fold
    still force-renders synth tasks for experiments; "mixed" (default) = the task's own strategy."""
    if task.startswith("synth_") and SYNTH_STRATEGY in ("binary", "left_fold"):
        return [(SYNTH_STRATEGY, n)]
    return [(None, n)]


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
_QA_VERDICT_TASKS = {"bookqa", "narrativeqa", "niah_novel", "niah_multi"}


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
# The ROOT's first turn — question → strategy preamble → per-node subtask — is the one step
# that is never self-similar (every other turn is a copy of a taught template) and the one the
# run-4 RULER failures broke at (a retrieval question rendered as "compute the hidden number";
# a subtask with literal START..END placeholders). It is also the LEAST-trained turn: one per
# trace vs ~30 leaf/split turns, i.e. ~1.5% of datums. Upweight it.
ROOT_DUP = int(os.environ.get("ROOT_DUP", "4"))                # copies of each root (agent or first turn)
# Cost knobs (2026-09-15, runs cost ~$240 from base):
# DATUM_MODE=agent — ONE datum per agent conversation with every assistant turn weighted, instead of
#   one datum per turn each repaying the system prompt + tool schema + earlier turns. Verified exactly
#   equivalent on our (thinking-free) traces: every per-turn datum is a strict prefix of the full
#   render and the weight masks coincide (275/275 turns, 123/123 agents); 1.9x fewer tokens.
# INTERNAL_KEEP — fraction of pure split/combine agents (depth>0, no read_chunk) kept. They are
#   near-identical to each other ("Range a..b > 500; splitting at midpoint m" / "Sum my children");
#   roots and every reading agent (leaves, fold hops) are always kept.
DATUM_MODE = os.environ.get("DATUM_MODE", "agent")
# Run 7 (from base) showed INTERNAL_KEEP=0.3 is NOT free: the split-only agents carry the boundary
# rule ("Range 1000..1500 > 500; splitting" vs "Range 1000..1500 fits") and with 70% of them gone
# the model read 500-token leaves at eval (run 6: split at exactly-500 ranges 144:15; run 7: 20:153)
# -> dense leaves overflowed (fwe/vt/multikey). Default back to keep everything.
INTERNAL_KEEP = float(os.environ.get("INTERNAL_KEEP", "1.0"))


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
# Datums per optim step. With DATUM_MODE=agent a datum holds a whole conversation, so at the same
# batch size the run takes ~2.5x fewer optimizer steps than per-turn mode (run 7: 3,600 vs 7,973) and
# ended less fit (NLL 0.0140 vs 0.0088). Tokens per step, not steps, drive cost, so use a smaller batch
# in agent mode to restore the step count.
SFT_BATCH_SIZE = int(os.environ.get("SFT_BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LR", "1e-5"))
# Warm start: load these weights (not the optimizer) into the fresh LoRA client before training —
# for iterating on NEW tasks train on them + a small replay of the rest from the last full run,
# instead of everything from base (~4x cheaper). Full from-base runs stay the reference.
INIT_CHECKPOINT = os.environ.get("INIT_CHECKPOINT", "").strip() or None

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


# Document-size MIX: train a FRACTION of problems at LONGER docs -> deeper trees -> more
# mid-range split turns (the OOLONG runaway is the model mis-emitting a split on a range
# magnitude it saw too rarely), and it teaches length generalization directly. Spec is
# "size:weight,size:weight" — e.g. "6000:4,12000:1" ≈ 80% @6000, 20% @12000. Empty -> every
# problem at DOC_SIZE_TOKENS. Sizes are assigned deterministically by problem index.
def _parse_doc_mix(spec: str, default: int) -> list[int]:
    if not spec.strip():
        return [default]
    cyc: list[int] = []
    for part in spec.split(","):
        size, _, w = part.partition(":")
        cyc += [int(size)] * (int(w) if w.strip() else 1)
    return cyc or [default]


_DOC_CYCLE = _parse_doc_mix(os.environ.get("DOC_MIX", ""), DOC_SIZE_TOKENS)


# Per-task doc mix: "vt_novel=6000:1,14000:1;long_records=..." — a task whose eval regime is long
# (RULER vt at 10K+ is a 25-hop fold chain; run-8w roots abandoned fold at 10K) gets more long docs.
_DOC_MIX_OVERRIDE = {
    kv.split("=")[0].strip(): _parse_doc_mix(kv.split("=")[1], DOC_SIZE_TOKENS)
    for kv in os.environ.get("DOC_MIX_OVERRIDE", "").split(";") if "=" in kv
}


# BUDGET x LENGTH JITTER (run 16). The split rule is FIXED — midpoint, "less than 500 tokens" — but the
# corpus used to show it at ONE budget (3000) and three doc sizes (6000/8400/14000 -> a handful of range-size
# families per depth), so the model could predict split-vs-read from memorized size families and surface cues
# instead of the rule. Run 15w read "Range 2000..4000" as a leaf with P=1.0 (round range ending at the stated
# doc length; trace_snippets/run15_split_decision_probe.txt). Here each problem draws a context BUDGET and a
# CONTINUOUS doc length, larger budgets with longer docs; the oracle's behaviour does not change with the
# budget, so the model sees "your context window is 12000 tokens" and still splits at 500.
# Spec "3000:50,5000:15,8000:13,10000:11,12000:11" = budget:weight. Empty -> the old fixed AGENT_CONTEXT + DOC_MIX behaviour.
# Doc ranges never exceed the pre-run-16 ceiling (14000-token target). Each row's MEAN doc exceeds its budget.
# Lengths are TRIANGULAR with the mode at the row's LOW end: node count doubles at 8000 tokens (31 -> 63 agents),
# so a uniform draw would shift gradient mass from roots to internal/leaf agents; this keeps ~54% of traces at
# >= 8000 (run 15: 50%) and the root share of datums about where it was.
_BUDGET_DOC_RANGE = {3000: (4000, 14000), 5000: (5000, 14000), 8000: (6000, 14000),
                     10000: (9000, 14000), 12000: (11500, 14000)}
_BUDGET_MIX = [(int(b), float(w)) for b, _, w in
               (kv.partition(":") for kv in os.environ.get("BUDGET_MIX", "").split(",") if kv.strip())]
# A share of doc targets rounded to a multiple of 1000 (exact-slice generators then produce round lengths).
ROUND_DOC_FRAC = float(os.environ.get("ROUND_DOC_FRAC", "0.2"))
MAX_BUDGET = max([AGENT_CONTEXT] + [b for b, _ in _BUDGET_MIX])


def _budget_doc_for(i: int, task: str) -> tuple[int, int]:
    """(agent budget, doc size) for problem `i` of `task`. Deterministic (string-seeded RNG)."""
    if not _BUDGET_MIX:
        return AGENT_CONTEXT, _doc_size_for(i, task)
    rng = random.Random(f"budgetdoc|{task}|{i}")
    b = rng.choices([b for b, _ in _BUDGET_MIX], weights=[w for _, w in _BUDGET_MIX])[0]
    lo, hi = _BUDGET_DOC_RANGE[b]
    d = int(rng.triangular(lo, hi, lo))
    if rng.random() < ROUND_DOC_FRAC:
        d = round(d / 1000) * 1000
    return b, d


def _doc_size_for(i: int, task: str | None = None) -> int:
    """Doc size for problem index `i` — cycles DOC_MIX (or the task's DOC_MIX_OVERRIDE) so the long
    fraction spreads evenly and reproducibly across each task's problems."""
    cyc = _DOC_MIX_OVERRIDE.get(task, _DOC_CYCLE)
    return cyc[i % len(cyc)]


# ---------------------------------------------------------------------------
# Trace generation (CPU — no sampling client needed; the oracle is scripted)
# ---------------------------------------------------------------------------


# Temporal subtypes still excluded from SFT: date_most/date_2nd ("which date is
# represented most often") render gold as a dateobj we can't reliably format-match,
# so the oracle would gold-leak. (date_ntimes was previously here too, but we fixed
# its root cause — the upstream [:50] cap — in the generator, so it now derives the
# literal answer and is trainable.) We skip these and draw the next index instead.
_SKIP_TMODES = {"date_most", "date_2nd"}

# Tasks whose QUESTION alone specifies the job, so the system-prompt description can be dropped
# without making the problem ill-posed (see _make_sft_problem). Their eval counterparts (RULER
# vt / niah_*, and the in-dist prose tasks) carry no description at all.
# vt_novel is NOT here: its generator owns the bare condition (_VT_BARE_FRAC), dropping the
# description, the copy-semantics gloss and the order preface TOGETHER so the RULER-matched case is a
# controlled share rather than the product of three independent coins. One mechanism, not two.
_CONTEXT_DROP_TASKS = {"niah_novel", "niah_multi", "narrativeqa", "realdoc_count"}
CONTEXT_DROP = float(os.environ.get("CONTEXT_DROP", "0"))

# Our OWN harness boilerplate: tasks/ruler/_common.py splices this where RULER's template had
# {context}, so it is on 100% of RULER eval questions (vt, cwe, fwe, all niah — measured 25/25) and
# was on 0% of the 2,500 training questions. Not RULER's content; our document-presentation
# convention. Train a fraction of questions with it so the convention itself is familiar.
_QUESTION_PLACEHOLDER = (
    "[The relevant text is in a separate document accessible via the read_chunk tool — see the "
    "system prompt for usage.] (Document length: {n} tokens.)"
)
QUESTION_PLACEHOLDER_FRAC = float(os.environ.get("QUESTION_PLACEHOLDER_FRAC", "0"))


def _make_sft_problem(task, ti, i, corpus_tokens, tokenizer):
    """Deterministic (task, idx) -> problem. OOLONG uses the shared oolong_spec (same
    problem as eval by seed); synth/ruler use make_problem with a per-task seed range."""
    # DOC_MIX (long-doc tier) applies only to SCRIPTED tasks — for a model-leaf task
    # (narrativeqa/bookqa) a longer doc means a bigger tree = many more paid leaf calls for no
    # QA benefit, so pin those to the base size.
    # Model-leaf tasks keep the base size AND the base budget (their cost is paid leaf calls).
    budget, doc = (AGENT_CONTEXT, DOC_SIZE_TOKENS) if task in _REJECT_SAMPLE_TASKS else _budget_doc_for(i, task)
    if task in OOLONG_TASKS:   # OOLONG-synth only
        seed, dataset = oolong_spec(task, i, DATA_SEED)
        return seed, make_oolong_problem(
            task, corpus_tokens, tokenizer, doc, seed, dataset=dataset
        )
    seed = DATA_SEED + ti * 100_000 + i
    problem = make_problem(task, corpus_tokens, tokenizer, doc, seed)
    problem = dataclasses.replace(problem, metadata={**problem.metadata, "agent_budget": budget})
    # FORMAT DIVERSITY: drop the system-prompt document description on a fraction of the
    # PROSE tasks, so the root must infer the task's shape (order-dependent? one stated fact?
    # a tally?) from the QUESTION and the text alone. Every RULER eval task ships with an empty
    # task_context (RULER fidelity: vt, cwe, fwe, niah_*), while every training task has a
    # description — run 13 (from base) folded 2/2 on vt_novel (description present, "...in
    # order") and went BINARY 5/5 on RULER vt (no description). Only tasks whose QUESTION is
    # self-sufficient are eligible; synth/labeled/long/rule_label keep theirs (the record
    # layout and label set exist nowhere else).
    if task in _CONTEXT_DROP_TASKS and CONTEXT_DROP > 0:
        if random.Random(("ctxdrop", task, seed).__hash__() & 0xFFFFFFFF).random() < CONTEXT_DROP:
            problem = dataclasses.replace(problem, task_context="")
    if QUESTION_PLACEHOLDER_FRAC > 0:
        if random.Random(("qph", task, seed).__hash__() & 0xFFFFFFFF).random() < QUESTION_PLACEHOLDER_FRAC:
            line = _QUESTION_PLACEHOLDER.format(n=len(problem.document_tokens))
            problem = dataclasses.replace(problem, question=f"{line}\n{problem.question}")
    return seed, problem


def _budget(problem) -> int:
    return problem.metadata.get("agent_budget", AGENT_CONTEXT)


async def _one_trace(oracle, problem, tokenizer):
    return await run_agent(
        oracle,
        document_tokens=problem.document_tokens,
        tokenizer=tokenizer,
        task_context=problem.task_context,
        question=problem.question,
        budget=_budget(problem),
        max_chunk_tokens=MAX_CHUNK_TOKENS,
        max_depth=MAX_DEPTH,
        max_turns=MAX_TURNS,
    )


# --- trace cache: reuse generated traces across runs (mainly to avoid re-paying the haiku
# leaf calls for narrativeqa). Keyed by everything that affects a trace; bump CACHE_VERSION when
# the QA oracle / trace format changes so stale traces are ignored. -------------------------
TRACE_CACHE = os.environ.get("SFT_TRACE_CACHE", "1") == "1"
_TRACE_CACHE_DIR = os.path.expanduser(
    os.environ.get("SFT_TRACE_CACHE_DIR", "~/.cache/infinite-context/sft_traces")
)
_CACHE_VERSION = "v11"  # v11: minimal-state audit — labeled relative/sections_cmp/first_month_cmp/section_most keep only the named labels, author_count/author_top subset variants; rule_label section_most per-section count; synth_2d months_cmp/month_for_grp/grp_in_month minimal (2026-09-24); v10: v10: labeled author_cmp filters to the two named authors AND the label (2026-09-24); v9: niah filtered leaves list skipped keys (convention) + needle distractors are records; labeled author_label_count + multi-author subsets + filtered share ~30% (2026-09-23); v8: niah_multi filters to asked keys (explicit/multiquery/multivalue) + needle haystack; niah_novel uuid keys/values + needle haystack (2026-09-23); v7: vt_novel bare-gloss variant + labeled_records schema-lite description (question/task_context changed for existing seeds) (2026-09-22); v6: retrieval root names the question; v5: pure 8w preamble


def _trace_key(task, seed, doc_len, strategy, leaf_model_name, nodesc=False, qph=False, ctx=None) -> str:
    from oracle.base import ScaffoldOracle
    payload = dict(
        v=_CACHE_VERSION, task=task, seed=seed, doc=doc_len, strategy=strategy or "default",
        ctx=ctx or AGENT_CONTEXT, leaf=ScaffoldOracle.LEAF_TOKENS, fold=ScaffoldOracle.FOLD_LEAF_TOKENS,
        chunk=MAX_CHUNK_TOKENS, model=leaf_model_name,
        # the system prompt is baked into the cached trace, so a description-dropped trace is a
        # DIFFERENT artifact for the same (task, seed, doc) — key it, or CONTEXT_DROP silently
        # reuses described traces.
        **({"nodesc": True} if nodesc else {}),
        **({"qph": True} if qph else {}),
    )
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _ser_node(n) -> dict:
    def tc(t): return {"id": t.id, "name": t.name, "arguments": t.arguments}
    def msg(m):
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [tc(t) for t in m["tool_calls"]]
        return m
    return dict(depth=n.depth, subtask=n.subtask, answer=n.answer, n_turns=n.n_turns,
                termination=n.termination, messages=[msg(m) for m in n.messages],
                children=[_ser_node(c) for c in n.children])


def _deser_node(d) -> AgentNode:
    def msg(m):
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [ToolCall(t["id"], t["name"], t["arguments"]) for t in m["tool_calls"]]
        return m
    return AgentNode(depth=d["depth"], subtask=d["subtask"], answer=d["answer"],
                     n_turns=d["n_turns"], termination=d["termination"],
                     messages=[msg(m) for m in d["messages"]],
                     children=[_deser_node(c) for c in d["children"]])


async def _cached_trace(oracle, problem, tokenizer, *, task, seed, strategy, leaf_model_name):
    """Like _one_trace but memoized to disk. Caches EVERY generated trace (accepted or not) so a
    re-run's rejection walk reuses them without re-calling the leaf model."""
    if not TRACE_CACHE:
        return await _one_trace(oracle, problem, tokenizer)
    os.makedirs(_TRACE_CACHE_DIR, exist_ok=True)
    path = os.path.join(_TRACE_CACHE_DIR, _trace_key(
        task, seed, len(problem.document_tokens), strategy, leaf_model_name,
        nodesc=not problem.task_context,
        qph=problem.question.startswith("[The relevant text is in a separate document"),
        ctx=_budget(problem)) + ".json")
    if os.path.exists(path):
        try:
            return _deser_node(json.load(open(path)))
        except Exception:
            pass   # corrupt/incompatible -> regenerate
    node = await _one_trace(oracle, problem, tokenizer)
    try:
        with open(path, "w") as f:
            json.dump(_ser_node(node), f)
    except Exception:
        pass
    return node


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
            budget=_budget(problem), max_chunk_tokens=MAX_CHUNK_TOKENS,
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
                budget=_budget(problem), max_chunk_tokens=MAX_CHUNK_TOKENS,
                leaf_model=leaf_model,
            )
            batch.append(((task, seed, problem), oracle, problem))
        nodes = await asyncio.gather(*[
            _cached_trace(o, p, tokenizer, task=task, seed=m[1], strategy=None,
                          leaf_model_name=(BOOKQA_LEAF_MODEL or "scripted"))
            for m, o, p in batch
        ])
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
    verdict class (pos/neg/normal) used by main() to resample the imbalanced QA leaf signal
    ('normal' for non-QA tasks), or 'root' for the depth-0 agent's FIRST turn (see ROOT_DUP).

    Per-turn (LAST_ASSISTANT_MESSAGE) rather than ALL_ASSISTANT_MESSAGES because
    the Qwen3 renderer lacks the extension property (it strips thinking from
    history), so each assistant turn must be rendered with its own real prefix.
    """
    cb = neutral_to_cookbook(node.messages, renderer, tool_specs)
    out: list[tuple] = []   # (datum, klass) — klass drives resampling in main()
    is_root = getattr(node, "depth", 1) == 0
    reads = any(m.get("role") == "tool" and m.get("name") == "read_chunk" for m in node.messages)
    if DATUM_MODE == "agent":
        model_input, weights = renderer.build_supervised_example(
            cb, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        if float(weights.sum()) == 0.0:
            return out
        datum = datum_from_model_input_weights(model_input, weights, max_length=MAX_BUDGET, reduction="mean")
        last = next((m.get("content") or "" for m in reversed(cb) if m.get("role") == "assistant"), "")
        # Order matters: QA verdicts FIRST. Most pos/neg verdicts are ANCESTORS of QA leaves
        # ("a subagent located the answer" / "neither half had relevant information") — spawn-only
        # agents that would otherwise fall into `internal` and lose the rebalancing (narrativeqa:
        # 56 of 76 pos verdicts are ancestors). Only verdict-less spawn-only agents are `internal`.
        if is_root:
            klass = "root"
        else:
            klass = _qa_verdict_class(last) if is_qa else "normal"
            if klass == "normal" and not reads:
                klass = "internal"      # pure split/combine node
        return [(datum, klass)]
    first_assistant = True
    for i, m in enumerate(cb):
        if m.get("role") != "assistant":
            continue
        is_root_turn = first_assistant and is_root
        first_assistant = False
        model_input, weights = renderer.build_supervised_example(
            cb[: i + 1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        if float(weights.sum()) == 0.0:  # nothing trainable (shouldn't happen)
            continue
        datum = datum_from_model_input_weights(
            model_input, weights, max_length=MAX_BUDGET, reduction="mean"
        )
        klass = _qa_verdict_class(m.get("content") or "") if is_qa else "normal"
        if is_root_turn:
            klass = "root"
        out.append((datum, klass))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


RETRY_MINUTES = float(os.environ.get("SFT_RETRY_MINUTES", "30"))   # in-process retry window per call


async def _retry(label: str, fn):
    """Re-run `fn()` (an awaitable factory) across transient Tinker/network failures for up to
    RETRY_MINUTES (backoff 15s→2min). Each call is retried on its own (never the
    forward_backward+optim_step pair together, which would double-apply a batch); result_async on
    an already-submitted future is idempotent server-side. Outages longer than the window raise;
    the launch script then RESTARTS sft.py, which resumes from the last periodic checkpoint (see
    SAVE_EVERY / RESUME) — that is the path for multi-hour outages."""
    delay, t0, i = 15.0, time.monotonic(), 0
    while True:
        try:
            return await fn()
        except (tinker.APIConnectionError, TimeoutError, asyncio.TimeoutError, OSError) as e:
            i += 1
            if time.monotonic() - t0 > RETRY_MINUTES * 60:
                print(f"  [retry] {label}: giving up after {i} attempts / {RETRY_MINUTES:.0f} min", flush=True)
                raise
            print(f"  [retry] {label}: {type(e).__name__} — attempt {i}, sleeping {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 1.6, 120.0)


# Periodic checkpoint + resume. Every SAVE_EVERY batches the trainer state (weights + Adam) is
# saved to Tinker under "<SAVE_NAME>_resume" and the batch cursor is written locally; on restart
# with the same config, sft.py reloads that state into a FRESH training client and skips the
# completed batches. Data is regenerated deterministically (fixed seeds, disk trace cache, seeded
# shuffle), so the datum order is identical — checked via n_datums.
SAVE_EVERY = int(os.environ.get("SFT_SAVE_EVERY", "300"))          # ~40 min at run-5 pace
# Intermediate checkpoints cost storage: they carry a TTL (long enough to ride out a multi-hour
# outage and the restart) and the last one is deleted when the run completes. The FINAL
# checkpoint never expires.
RESUME_TTL_HOURS = float(os.environ.get("SFT_RESUME_TTL_HOURS", "48"))
RESUME = os.environ.get("SFT_RESUME", "1") == "1"
RESUME_FILE = Path.home() / ".cache" / "infinite-context" / f"sft_resume_{SAVE_CHECKPOINT_NAME}.json"


LOG_EVERY = int(os.environ.get("SFT_LOG_EVERY", "100"))   # batches between progress lines


async def main() -> None:
    unknown = [t for t in SFT_TASKS if t not in list_tasks()]
    if unknown:
        raise SystemExit(f"Unknown SFT_TASKS: {unknown}. Available: {list_tasks()}")
    if "oolong_real" in SFT_TASKS:
        raise SystemExit("oolong_real is EVAL ONLY (held-out benchmark) — never train on it.")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    tool_specs = [
        rl.ReadChunkTool.read_chunk.to_spec(),
        rl.SubagentTool.spawn_subagent.to_spec(),
    ]

    _mix = ",".join(f"{s}x{_DOC_CYCLE.count(s)}" for s in sorted(set(_DOC_CYCLE)))
    print(f"SFT warm-start | tasks={len(SFT_TASKS)} x {N_PER_TASK} | "
          f"doc_mix=[{_mix}] budget={AGENT_CONTEXT} epochs={EPOCHS}"
          + (f" | BUDGET_MIX={_BUDGET_MIX} (doc ranges {_BUDGET_DOC_RANGE}, round share {ROUND_DOC_FRAC})" if _BUDGET_MIX else ""))
    print("Loading + tokenizing PG-essay corpus...")
    corpus_tokens = tokenizer.encode(load_pg_essays_text(), add_special_tokens=False)

    print("Generating oracle traces (scripted, CPU)...")
    traces = await _gen_traces(corpus_tokens, tokenizer)
    if _BUDGET_MIX:
        from collections import Counter as _C
        _bd = _C(_budget(m[2]) for m, _ in traces)
        _dl = [len(m[2].document_tokens) for m, _ in traces]
        print(f"Budget mix (traces): {dict(sorted(_bd.items()))} | doc tokens min/median/max "
              f"{min(_dl)}/{sorted(_dl)[len(_dl) // 2]}/{max(_dl)} | exact multiples of 1000: "
              f"{sum(d % 1000 == 0 for d in _dl)}")

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
    n_pos = n_neg_keep = n_neg_drop = neg_i = n_root = n_int_keep = n_int_drop = int_i = 0
    int_stride = max(1, round(1.0 / INTERNAL_KEEP)) if INTERNAL_KEEP > 0 else 1
    for datum, klass in tagged:
        if klass == "root":
            datums.extend([datum] * ROOT_DUP); n_root += 1
        elif klass == "internal":
            if int_i % int_stride == 0:
                datums.append(datum); n_int_keep += 1
            else:
                n_int_drop += 1
            int_i += 1
        elif klass == "pos":
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
          f"(QA verdicts: pos={n_pos}x{QA_ANSWER_DUP}, neg kept {n_neg_keep}/{n_neg_keep + n_neg_drop}; "
          f"root turns {n_root}x{ROOT_DUP}; internal split agents kept {n_int_keep}/{n_int_keep + n_int_drop})")
    n_tok = sum(len(d.model_input.to_ints()) for d in datums)
    print(f"Datum mode: {DATUM_MODE} | training tokens: {n_tok:,} ({n_tok / max(1, len(datums)):.0f}/datum)")
    if not datums:
        raise SystemExit("No datums produced — check oracle trace generation.")

    if not TRAIN:
        print("TRAIN=0: skipping Tinker training/save (dry run). "
              f"{'Traces at ' + TRACE_OUT + '.txt' if PRINT_TRACES else 'Set PRINT_TRACES=1 to inspect traces.'}")
        return

    # Training client (+ resume from a periodic checkpoint if one matches this config).
    start_batch = 0
    resume = None
    if RESUME and RESUME_FILE.exists():
        try:
            resume = json.loads(RESUME_FILE.read_text())
        except Exception:
            resume = None
        if resume and resume.get("n_datums") != len(datums):
            print(f"  resume file {RESUME_FILE} has n_datums={resume.get('n_datums')} != {len(datums)}; "
                  f"ignoring it (config changed?)")
            resume = None
    service_client = await _retry("ServiceClient", lambda: asyncio.to_thread(tinker.ServiceClient))
    training_client = await _retry("create_training_client", lambda: service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    ))
    if INIT_CHECKPOINT and not resume:
        print(f"  WARM START: loading weights from {INIT_CHECKPOINT} (fresh optimizer, LR {LEARNING_RATE:g})")
        fut = await _retry("load_state(init)", lambda: training_client.load_state_async(INIT_CHECKPOINT))
        await _retry("load_state(init).result", fut.result_async)
    if resume:
        print(f"  RESUMING from {resume['path']} at batch {resume['next_batch']}")
        fut = await _retry("load_state_with_optimizer", lambda: training_client.load_state_with_optimizer_async(resume["path"]))
        await _retry("load_state_with_optimizer.result", fut.result_async)
        start_batch = int(resume["next_batch"])
    adam_params = tinker.AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95)

    async def _save_resume(next_batch: int):
        fut = await _retry("save_state(resume)", lambda: training_client.save_state_async(
            f"{SAVE_CHECKPOINT_NAME}_resume", overwrite=True, ttl_seconds=int(RESUME_TTL_HOURS * 3600)))
        resp = await _retry("save_state(resume).result", fut.result_async)
        RESUME_FILE.parent.mkdir(parents=True, exist_ok=True)
        RESUME_FILE.write_text(json.dumps({"path": resp.path, "next_batch": next_batch, "n_datums": len(datums)}))
        print(f"  [ckpt] batch {next_batch}: {resp.path}", flush=True)

    metrics.init(
        project="infinite-context",
        config={
            "phase": "sft", "model": MODEL_NAME, "renderer": RENDERER_NAME,
            "lora_rank": LORA_RANK, "lr": LEARNING_RATE, "epochs": EPOCHS,
            "sft_batch_size": SFT_BATCH_SIZE, "n_datums": len(datums),
            "tasks": SFT_TASKS, "n_per_task": N_PER_TASK,
            "n_per_task_override": N_PER_TASK_OVERRIDE,
            "init_checkpoint": INIT_CHECKPOINT, "datum_mode": DATUM_MODE, "internal_keep": INTERNAL_KEEP,
        },
    )

    rng = random.Random(0)
    global_batch = 0
    for epoch in range(EPOCHS):
        rng.shuffle(datums)
        n_batches = (len(datums) + SFT_BATCH_SIZE - 1) // SFT_BATCH_SIZE
        epoch_nll = 0.0
        n_logged = 0
        first = start_batch if epoch == 0 else 0
        if first:
            print(f"  skipping {first} completed batches")
        for b in range(first, n_batches):
            batch = datums[b * SFT_BATCH_SIZE : (b + 1) * SFT_BATCH_SIZE]
            fwd_bwd = await _retry("forward_backward", lambda: training_client.forward_backward_async(batch, loss_fn="cross_entropy"))
            optim = await _retry("optim_step", lambda: training_client.optim_step_async(adam_params))
            fb_result = await _retry("forward_backward.result", fwd_bwd.result_async)
            await _retry("optim_step.result", optim.result_async)
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
                    if LOG_EVERY and (b + 1) % LOG_EVERY == 0:
                        print(f"  batch {b + 1}/{n_batches} | running mean NLL {epoch_nll / n_logged:.4f}", flush=True)
            except Exception:
                pass
            global_batch += 1
            if SAVE_EVERY and (b + 1) % SAVE_EVERY == 0 and (b + 1) < n_batches:
                await _save_resume(b + 1)
        nll_str = f"{epoch_nll / n_logged:.4f}" if n_logged else "n/a"
        print(f"Epoch {epoch}: batches {n_batches} | mean batch NLL {nll_str}")
        if n_logged:
            metrics.log({"sft/epoch_nll": epoch_nll / n_logged}, step=global_batch)

        # Checkpoint after EVERY epoch (overwrite). Tinker runs are slow (~15
        # min/epoch) and occasionally drop connection, so a kill/crash should
        # never cost a full redo — last_sft_checkpoint.txt always points at the
        # latest completed epoch's weights, which are usable for RL warm-start.
        print(f"  saving checkpoint '{SAVE_CHECKPOINT_NAME}' (after epoch {epoch})...")
        save_future = await _retry("save_state(final)", lambda: training_client.save_state_async(SAVE_CHECKPOINT_NAME, overwrite=True))
        save_resp = await _retry("save_state(final).result", save_future.result_async)
        LAST_SFT_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_SFT_CHECKPOINT_FILE.write_text(save_resp.path)
        print(f"  saved: {save_resp.path}  (path -> {LAST_SFT_CHECKPOINT_FILE})")
    if RESUME_FILE.exists():
        # finished: delete the intermediate checkpoint (it costs storage) and the local cursor so
        # a later run with the same SAVE_NAME starts fresh. Best-effort — the TTL is the backstop.
        try:
            rpath = json.loads(RESUME_FILE.read_text()).get("path")
            if rpath:
                await service_client.create_rest_client().delete_checkpoint_from_tinker_path_async(rpath)
                print(f"  deleted intermediate checkpoint {rpath}")
        except Exception as e:
            print(f"  (could not delete intermediate checkpoint: {type(e).__name__}: {e}; TTL will expire it)")
        RESUME_FILE.unlink()
    print("\nTo warm-start RL: set rl.py LOAD_CHECKPOINT_PATH to the path above "
          "and RESUME_OPTIMIZER=False.")
    metrics.finish()


if __name__ == "__main__":
    asyncio.run(main())
