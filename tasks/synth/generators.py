"""Abstract synthetic decomposition tasks.

A document is a list of trivial structured records, one per line:

    [0000] id=7314 grp=K3 amt=+12 flag=Y
    [0001] id=2241 grp=RST amt=-5 flag=N

The leaf-op (parse a field off a line) is deliberately trivial, so the ONLY thing
a model must learn is the DECOMPOSITION — split + combine + which strategy — in
isolation from leaf-op difficulty. The thesis: train the scaffold here on abstract
data, then transfer it zero-shot to RULER / OOLONG (where the leaf-op is a real
classification the base model can already do in a small chunk).

Tasks span the two SINGLE-PASS strategy regimes (each token read ~once):
  - bounded-associative  (sum / count / max): O(1) combine -> favors BINARY tree-reduce.
  - stateful-sequential  (runreset): non-associative running state -> favors LEFT-FOLD.

Gold is exactly computable, so the reward is free and exact (clean for RFT later).
`metadata["record_spans"]` gives each record's token span + parsed fields, so the
oracle can map a token range to the records that start in it (same trick OOLONG uses).
"""

from __future__ import annotations

import random

from tasks.base import Problem

# task -> (family, favored single-pass strategy). The favored strategy is the
# "mixed" training default; sft.py can override per task to run all-binary /
# all-left-fold experiments (the strategy is a TRAINING knob).
SYNTH_TASKS: dict[str, dict] = {
    "synth_sum":      {"family": "bounded",  "strategy": "binary"},
    "synth_count":    {"family": "bounded",  "strategy": "binary"},
    "synth_max":      {"family": "bounded",  "strategy": "binary"},
    "synth_runreset": {"family": "stateful", "strategy": "left_fold"},
}

_GROUPS = ["K1", "K2", "K3", "K4"]
_RST = "RST"
_RST_RATE = 0.08   # ~8% of records are reset markers (only matter for runreset)


def _one_record(rng: random.Random, idx: int) -> dict:
    is_rst = rng.random() < _RST_RATE
    return {
        "idx": idx,
        "id": rng.randint(1000, 9999),
        "grp": _RST if is_rst else rng.choice(_GROUPS),
        "amt": rng.randint(-20, 20),
        "flag": "Y" if rng.random() < 0.5 else "N",
    }


def _render(r: dict) -> str:
    sign = f"+{r['amt']}" if r["amt"] >= 0 else str(r["amt"])
    return f"[{r['idx']:04d}] id={r['id']} grp={r['grp']} amt={sign} flag={r['flag']}"


def gold_for(task: str, recs: list[dict]) -> int:
    """Exact ground truth from the full record list (the oracle's tree must match)."""
    if task == "synth_sum":
        return sum(r["amt"] for r in recs)
    if task == "synth_count":
        return sum(1 for r in recs if r["flag"] == "Y")
    if task == "synth_max":
        return max((r["amt"] for r in recs), default=0)
    if task == "synth_runreset":
        tot = 0
        for r in recs:
            tot = 0 if r["grp"] == _RST else tot + r["amt"]
        return tot
    raise ValueError(f"Unknown synth task: {task!r}")


_QUESTION = {
    "synth_sum": (
        "Each record has a signed integer 'amt' field. What is the SUM of 'amt' "
        "over ALL records in the document? Give the single integer in \\boxed{}."
    ),
    "synth_count": (
        "Each record has a 'flag' field that is Y or N. How many records have "
        "flag=Y? Give the single integer in \\boxed{}."
    ),
    "synth_max": (
        "Each record has a signed integer 'amt' field. What is the MAXIMUM 'amt' "
        "over all records in the document? Give the single integer in \\boxed{}."
    ),
    "synth_runreset": (
        "Process the records in order, keeping a running total of 'amt'. Whenever "
        "a record has grp=RST, reset the running total to 0 (that record's amt is "
        "NOT added). What is the final running total after the last record? Give "
        "the single integer in \\boxed{}."
    ),
}

_CONTEXT = (
    "The document is a list of records, one per line, each formatted as:\n"
    "  [<index>] id=<int> grp=<K1|K2|K3|K4|RST> amt=<signed int> flag=<Y|N>\n"
    "Records are 0-indexed and appear in order."
)


def make_synth_problem(
    task: str,
    corpus_tokens,          # unused — synth builds its own document
    tokenizer,
    doc_size_tokens: int,
    seed: int,
) -> Problem:
    if task not in SYNTH_TASKS:
        raise ValueError(f"Unknown synth task: {task!r}")
    rng = random.Random(seed)

    # Generate records one at a time, packing until the next record would exceed
    # the doc budget. Track each record's token span so the oracle can map a token
    # range -> the records starting in it.
    doc_tokens: list[int] = []
    recs: list[dict] = []
    spans: list[tuple] = []   # (tok_start, tok_end, idx, amt, flag, grp)
    idx = 0
    while True:
        r = _one_record(rng, idx)
        toks = tokenizer.encode(_render(r) + "\n", add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        spans.append((start, len(doc_tokens), r["idx"], r["amt"], r["flag"], r["grp"]))
        recs.append(r)
        idx += 1

    gold = gold_for(task, recs)
    return Problem(
        document_tokens=doc_tokens,
        question=_QUESTION[task],
        gold_answers=[str(gold)],
        task=task,
        task_context=_CONTEXT,
        metadata={
            "family": SYNTH_TASKS[task]["family"],
            "strategy_default": SYNTH_TASKS[task]["strategy"],
            "task": task,
            "n_records": len(recs),
            "record_spans": spans,
            "gold_int": gold,
        },
    )
