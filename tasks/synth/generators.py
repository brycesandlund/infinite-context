"""Abstract synthetic decomposition tasks.

A document is a list of trivial structured records, one per line:

    [0000] id=7314 grp=K3 amt=+12 flag=Y
    [0001] id=2241 grp=RST amt=-5 flag=N

(or, for the variable-tracking task, assignment records:)

    [0000] set A = 42
    [0001] set B = A

The leaf-op (parse a field) is deliberately trivial, so the ONLY thing a model must
learn is the DECOMPOSITION — split + combine + which strategy — in isolation from
leaf-op difficulty. Train the scaffold here on abstract data, then transfer it
zero-shot to RULER / OOLONG.

Tasks span both SINGLE-PASS strategies and several combine archetypes:
  bounded-associative (binary tree-reduce):
    sum / min / max / count / sumwhere  -> scalar reduce
    mode                                -> per-key tally then argmax  (≈ OOLONG "most common")
    distinct                            -> set-union then count       (≈ dedup / multivalue)
  stateful-sequential (left-fold):
    runreset                            -> running total w/ resets
    varchain                            -> thread variable bindings   (≈ RULER variable tracking)

Gold is exactly computable, so the reward is free and exact. `metadata["record_spans"]`
gives each record's token span + parsed fields, so the oracle maps a token range to
the records that start in it.
"""

from __future__ import annotations

import random
from collections import Counter

from tasks.base import Problem

# task -> (family, favored single-pass strategy). The favored strategy is the
# "mixed" training default; sft.py can override per task (the strategy training knob).
SYNTH_TASKS: dict[str, dict] = {
    "synth_sum":      {"family": "bounded",  "strategy": "binary"},
    "synth_count":    {"family": "bounded",  "strategy": "binary"},
    "synth_max":      {"family": "bounded",  "strategy": "binary"},
    "synth_min":      {"family": "bounded",  "strategy": "binary"},
    "synth_sumwhere": {"family": "bounded",  "strategy": "binary"},
    "synth_mode":     {"family": "bounded",  "strategy": "binary"},
    "synth_distinct": {"family": "bounded",  "strategy": "binary"},
    # combining tasks (use >=2 fields):
    "synth_sumby":    {"family": "bounded",  "strategy": "binary"},   # grp(key)+amt(value) -> argmax
    "synth_count2":   {"family": "bounded",  "strategy": "binary"},   # flag AND grp (compound predicate)
    "synth_diff":     {"family": "bounded",  "strategy": "binary"},   # sum(flag=Y) - sum(flag=N)
    "synth_maxwhere": {"family": "bounded",  "strategy": "binary"},   # max amt where flag=?
    # temporal-esque (threshold / range on amt):
    "synth_count_cmp":   {"family": "bounded", "strategy": "binary"},  # how many amt > / < T
    "synth_count_range": {"family": "bounded", "strategy": "binary"},  # how many amt in [L, H]
    "synth_runreset": {"family": "stateful", "strategy": "left_fold"},
    "synth_varchain": {"family": "stateful", "strategy": "left_fold"},
}

# Tasks whose question/gold are PARAMETERIZED per problem (the predicate is randomized
# and stored in metadata so the oracle can apply it). The rest use gold_for + _QUESTION.
_PARAM_TASKS = {"synth_count2", "synth_maxwhere", "synth_count_cmp", "synth_count_range"}

_GROUPS = ["K1", "K2", "K3", "K4"]
_RST = "RST"
_RST_RATE = 0.08          # ~8% reset markers (only matter for runreset)
_VARS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Standard records: [idx] id=.. grp=.. amt=±n flag=Y/N
# ---------------------------------------------------------------------------


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


def _mode(recs: list[dict]) -> str:
    """Most common grp among non-RST records; ties broken lexicographically (so the
    oracle's argmax and this gold agree exactly)."""
    c = Counter(r["grp"] for r in recs if r["grp"] != _RST)
    if not c:
        return _GROUPS[0]
    mx = max(c.values())
    return min(g for g, n in c.items() if n == mx)


def gold_for(task: str, recs: list[dict]):
    """Exact ground truth from the full record list (the oracle's tree must match)."""
    if task == "synth_sum":
        return sum(r["amt"] for r in recs)
    if task == "synth_count":
        return sum(1 for r in recs if r["flag"] == "Y")
    if task == "synth_max":
        return max((r["amt"] for r in recs), default=0)
    if task == "synth_min":
        return min((r["amt"] for r in recs), default=0)
    if task == "synth_sumwhere":
        return sum(r["amt"] for r in recs if r["flag"] == "Y")
    if task == "synth_mode":
        return _mode(recs)
    if task == "synth_distinct":
        return len({r["grp"] for r in recs if r["grp"] != _RST})
    if task == "synth_sumby":
        tot = {}
        for r in recs:
            if r["grp"] != _RST:
                tot[r["grp"]] = tot.get(r["grp"], 0) + r["amt"]
        if not tot:
            return _GROUPS[0]
        mx = max(tot.values())
        return min(g for g, v in tot.items() if v == mx)   # tie -> lexicographic
    if task == "synth_diff":
        return (sum(r["amt"] for r in recs if r["flag"] == "Y")
                - sum(r["amt"] for r in recs if r["flag"] == "N"))
    if task == "synth_runreset":
        tot = 0
        for r in recs:
            tot = 0 if r["grp"] == _RST else tot + r["amt"]
        return tot
    raise ValueError(f"Unknown synth task: {task!r}")


def _param_question_gold(task: str, recs: list[dict], rng: random.Random):
    """Parameterized tasks: pick the predicate, build the question, compute exact gold,
    and return (question, gold, metadata-params)."""
    if task == "synth_count2":
        qflag, qgrp = rng.choice(["Y", "N"]), rng.choice(_GROUPS)
        gold = sum(1 for r in recs if r["flag"] == qflag and r["grp"] == qgrp)
        q = (f"How many records have BOTH flag={qflag} AND grp={qgrp}? Give the single "
             f"integer in \\boxed{{}}.")
        return q, gold, {"qflag": qflag, "qgrp": qgrp}
    if task == "synth_maxwhere":
        qflag = rng.choice(["Y", "N"])
        m = [r["amt"] for r in recs if r["flag"] == qflag]
        gold = max(m) if m else 0
        q = (f"What is the MAXIMUM 'amt' among records with flag={qflag}? Give the single "
             f"integer in \\boxed{{}}.")
        return q, gold, {"qflag": qflag}
    if task == "synth_count_cmp":
        op, t = rng.choice([">", "<"]), rng.randint(-12, 12)
        gold = sum(1 for r in recs if (r["amt"] > t if op == ">" else r["amt"] < t))
        word = "greater than" if op == ">" else "less than"
        q = f"How many records have 'amt' {word} {t}? Give the single integer in \\boxed{{}}."
        return q, gold, {"op": op, "thresh": t}
    if task == "synth_count_range":
        lo, hi = rng.randint(-15, -2), rng.randint(2, 15)
        gold = sum(1 for r in recs if lo <= r["amt"] <= hi)
        q = (f"How many records have 'amt' between {lo} and {hi}, inclusive? Give the single "
             f"integer in \\boxed{{}}.")
        return q, gold, {"lo": lo, "hi": hi}
    raise ValueError(task)


# ---------------------------------------------------------------------------
# Variable-tracking records: [idx] set VAR = (int | VAR)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Questions / context
# ---------------------------------------------------------------------------

_QUESTION = {
    "synth_sum": "Each record has a signed integer 'amt' field. What is the SUM of 'amt' "
                 "over ALL records in the document? Give the single integer in \\boxed{}.",
    "synth_count": "Each record has a 'flag' field that is Y or N. How many records have "
                   "flag=Y? Give the single integer in \\boxed{}.",
    "synth_max": "Each record has a signed integer 'amt' field. What is the MAXIMUM 'amt' "
                 "over all records? Give the single integer in \\boxed{}.",
    "synth_min": "Each record has a signed integer 'amt' field. What is the MINIMUM 'amt' "
                 "over all records? Give the single integer in \\boxed{}.",
    "synth_sumwhere": "Each record has an 'amt' integer and a 'flag' (Y/N). What is the SUM "
                      "of 'amt' over ONLY the records with flag=Y? Give the integer in \\boxed{}.",
    "synth_mode": "Each record has a 'grp' field (K1, K2, K3, K4, or RST). Ignoring RST, "
                  "which grp value appears MOST often? Give the grp (e.g. K2) in \\boxed{}.",
    "synth_distinct": "Each record has a 'grp' field (K1, K2, K3, K4, or RST). Ignoring RST, "
                      "how many DISTINCT grp values appear? Give the integer in \\boxed{}.",
    "synth_sumby": "Each record has a 'grp' field (K1, K2, K3, K4, or RST) and a signed integer "
                   "'amt'. Ignoring RST, which grp has the LARGEST total 'amt'? Give the grp "
                   "(e.g. K2) in \\boxed{}.",
    "synth_diff": "Each record has a signed integer 'amt' and a 'flag' (Y or N). What is the SUM "
                  "of 'amt' over flag=Y records MINUS the SUM over flag=N records? Give the single "
                  "integer in \\boxed{}.",
    "synth_runreset": "Process the records in order, keeping a running total of 'amt'. "
                      "Whenever a record has grp=RST, reset the running total to 0 (that "
                      "record's amt is NOT added). What is the final running total after the "
                      "last record? Give the single integer in \\boxed{}.",
    # varchain question is filled per-problem (needs the queried variable).
}

_CONTEXT = (
    "The document is a list of records, one per line, each formatted as:\n"
    "  [<index>] id=<int> grp=<K1|K2|K3|K4|RST> amt=<signed int> flag=<Y|N>\n"
    "Records are 0-indexed and appear in order."
)
_CONTEXT_VC = (
    "The document is a list of variable-assignment records, one per line, formatted as:\n"
    "  [<index>] set <VAR> = <integer or another VAR>\n"
    "Records execute in order; 'set B = A' copies A's CURRENT value into B."
)


def _pack(rng, render, one_record, doc_size_tokens, tokenizer):
    """Pack records until the next would exceed the budget; return (records, doc_tokens, spans)."""
    doc_tokens, recs, spans, idx = [], [], [], 0
    while True:
        r = one_record(idx)
        toks = tokenizer.encode(render(r) + "\n", add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        spans.append((start, len(doc_tokens), r))
        recs.append(r)
        idx += 1
    return recs, doc_tokens, spans


def make_synth_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in SYNTH_TASKS:
        raise ValueError(f"Unknown synth task: {task!r}")
    rng = random.Random(seed)

    if task == "synth_varchain":
        # Pack assignment records, threading the bindings as we go (a `= VAR` copies the
        # source's CURRENT value), then query a variable bound at the end.
        doc_tokens, spans, binding, idx = [], [], {}, 0
        while True:
            v = rng.choice(_VARS)
            set_vars = [k for k in _VARS if k in binding]
            if set_vars and rng.random() < 0.4:
                rhs, is_ref, val = (src := rng.choice(set_vars)), True, binding[src]
            else:
                val = rng.randint(1, 99)
                rhs, is_ref = str(val), False
            toks = tokenizer.encode(f"[{idx:04d}] set {v} = {rhs}\n", add_special_tokens=False)
            if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
                break
            start = len(doc_tokens)
            doc_tokens.extend(toks)
            spans.append((start, len(doc_tokens), idx, v, rhs, is_ref))
            binding[v] = val
            idx += 1
        bound = [v for v in _VARS if v in binding]
        qvar = rng.choice(bound) if bound else _VARS[0]
        gold = binding.get(qvar, 0)
        question = (f"Track the variable assignments in order ('set B = A' copies A's current "
                    f"value). What is the FINAL value of variable {qvar}? Give the integer in \\boxed{{}}.")
        return Problem(
            document_tokens=doc_tokens, question=question, gold_answers=[str(gold)], task=task,
            task_context=_CONTEXT_VC, grading_mode="numeric",
            metadata={"family": "stateful", "strategy_default": "left_fold", "task": task,
                      "query_var": qvar, "n_records": len(spans), "record_spans": spans, "gold_int": gold},
        )

    # Standard-record tasks.
    recs, doc_tokens, raw_spans = _pack(rng, _render, lambda i: _one_record(rng, i), doc_size_tokens, tokenizer)
    spans = [(s, e, r["idx"], r["amt"], r["flag"], r["grp"]) for (s, e, r) in raw_spans]
    if task in _PARAM_TASKS:
        question, gold, qparams = _param_question_gold(task, recs, rng)
    else:
        question, gold, qparams = _QUESTION[task], gold_for(task, recs), {}
    grading = "exact" if task in ("synth_mode", "synth_sumby") else "numeric"  # grp answers
    return Problem(
        document_tokens=doc_tokens, question=question, gold_answers=[str(gold)], task=task,
        task_context=_CONTEXT, grading_mode=grading,
        metadata={"family": SYNTH_TASKS[task]["family"], "strategy_default": SYNTH_TASKS[task]["strategy"],
                  "task": task, "n_records": len(recs), "record_spans": spans, "gold_int": gold, **qparams},
    )
