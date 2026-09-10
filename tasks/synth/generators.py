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
from tasks.phrasing import filtered_question

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
    # more ORDER-DEPENDENT ops (left_fold is a property of these tasks, not a training knob):
    "synth_peak":     {"family": "stateful", "strategy": "left_fold"},   # max value the running total ever reaches
    "synth_streak":   {"family": "stateful", "strategy": "left_fold"},   # longest run of consecutive flag=Y
    "synth_adjacent": {"family": "stateful", "strategy": "left_fold"},   # records whose amt > previous record's amt
    "synth_first_exceed": {"family": "stateful", "strategy": "left_fold"},  # first index where running total > T
    # month-records family (records carry a `mon` field; question variant drawn per problem):
    "synth_filter_argmax": {"family": "bounded", "strategy": "binary"},  # filter -> per-grp tally -> argmax/argmin
    "synth_2d":            {"family": "bounded", "strategy": "binary"},  # month x grp tally -> reduce along one axis
}

# Tasks whose question/gold are PARAMETERIZED per problem (the predicate is randomized
# and stored in metadata so the oracle can apply it). The rest use gold_for + _QUESTION.
_PARAM_TASKS = {"synth_count2", "synth_maxwhere", "synth_count_cmp", "synth_count_range", "synth_first_exceed"}

_GROUPS = ["K1", "K2", "K3", "K4"]
_RST = "RST"
_RST_RATE = 0.08          # ~8% reset markers (only matter for runreset)
_VARS = ["A", "B", "C", "D", "E"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_TASKS = {"synth_filter_argmax", "synth_2d"}


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
    if task == "synth_peak":
        tot, peak = 0, 0
        for r in recs:
            tot += r["amt"]; peak = max(peak, tot)
        return peak
    if task == "synth_streak":
        cur = best = 0
        for r in recs:
            cur = cur + 1 if r["flag"] == "Y" else 0
            best = max(best, cur)
        return best
    if task == "synth_adjacent":
        n, prev = 0, None
        for r in recs:
            if prev is not None and r["amt"] > prev: n += 1
            prev = r["amt"]
        return n
    raise ValueError(f"Unknown synth task: {task!r}")


def _param_question_gold(task: str, recs: list[dict], rng: random.Random):
    """Parameterized tasks: pick the predicate, build the question, compute exact gold,
    and return (question, gold, metadata-params)."""
    if task == "synth_count2":
        qflag, qgrp = rng.choice(["Y", "N"]), rng.choice(_GROUPS)
        gold = sum(1 for r in recs if r["flag"] == qflag and r["grp"] == qgrp)
        q = rng.choice([
            f"How many records have BOTH flag={qflag} AND grp={qgrp}? Give the single integer in \\boxed{{}}.",
            filtered_question(rng, "records", f"flag={qflag}", f"how many have grp={qgrp}",
                              "Give the single integer in \\boxed{}."),
            filtered_question(rng, "records", f"grp={qgrp}", f"how many have flag={qflag}",
                              "Give the single integer in \\boxed{}."),
        ])
        return q, gold, {"qflag": qflag, "qgrp": qgrp}
    if task == "synth_maxwhere":
        qflag = rng.choice(["Y", "N"])
        m = [r["amt"] for r in recs if r["flag"] == qflag]
        gold = max(m) if m else 0
        q = filtered_question(rng, "records", f"flag={qflag}", "what is the MAXIMUM 'amt'",
                              "Give the single integer in \\boxed{}.")
        return q, gold, {"qflag": qflag}
    if task == "synth_count_cmp":
        op, t = rng.choice([">", "<"]), rng.randint(-12, 12)
        gold = sum(1 for r in recs if (r["amt"] > t if op == ">" else r["amt"] < t))
        word = "greater than" if op == ">" else "less than"
        q = f"How many records have 'amt' {word} {t}? Give the single integer in \\boxed{{}}."
        return q, gold, {"op": op, "thresh": t}
    if task == "synth_first_exceed":
        tot, run = 0, []
        for r in recs:
            tot += r["amt"]; run.append(tot)
        peak = max(run) if run else 0
        if peak >= 3 and rng.random() < 0.85:
            t = rng.randint(1, peak - 1)          # reachable
            gold = next(i for i, v in enumerate(run) if v > t)
            grading = "numeric"
        else:                                     # occasionally unreachable -> 'none'
            t = peak + rng.randint(1, 10); gold, grading = "none", "exact"
        q = (f"Process the records in order, keeping a running total of 'amt'. What is the INDEX of "
             f"the FIRST record at which the running total exceeds {t} (is strictly greater than {t})? "
             f"If it never does, answer none. Give the index (or none) in \\boxed{{}}.")
        return q, gold, {"thresh": t, "_grading": grading}
    if task == "synth_count_range":
        lo, hi = rng.randint(-15, -2), rng.randint(2, 15)
        gold = sum(1 for r in recs if lo <= r["amt"] <= hi)
        q = (f"How many records have 'amt' between {lo} and {hi}, inclusive? Give the single "
             f"integer in \\boxed{{}}.")
        return q, gold, {"lo": lo, "hi": hi}
    raise ValueError(task)


# ---------------------------------------------------------------------------
# Month records: [idx] id=.. mon=Aug grp=.. amt=±n flag=Y/N   (no RST)
#
# A second categorical field so the aggregation can be COMPOUND: filter on one field then
# tally another (synth_filter_argmax), or keep a joint month x grp tally and reduce it
# along one axis (synth_2d). Both are variant families — the question form is drawn per
# problem and carried in metadata for the oracle.
# ---------------------------------------------------------------------------


# 2-D field SCHEMES (outer field, ordered outer vocabulary, inner field, sorted inner vocabulary).
# Drawn per problem so the joint-tally shape is learned across names, vocab sizes, multi-token
# values ("Mar 2024") and values containing the key separator ("Sci/Tech"): run 5's OOLONG
# temporal children invented three key formats for month×label because training had exactly one
# (mon/grp over a 4x4 vocabulary). Inner vocabularies are alphabetical so "alphabetically first"
# is the tie rule; outer vocabularies have a natural order (calendar / listed order) that the
# task context states, and ties on the outer axis break by that order.
_SCHEMES = [
    ("mon", _MONTHS, "grp", ["K1", "K2", "K3", "K4"]),
    ("period", [f"{m} {y}" for y in (2023, 2024) for m in _MONTHS], "label",
     ["Business", "Sci/Tech", "Sports", "World"]),
    ("region", ["north", "south", "east", "west", "central"], "product", ["gadget", "gizmo", "widget"]),
    ("dept", ["eng", "finance", "hr", "legal", "ops", "sales"], "status", ["closed", "escalated", "open", "pending"]),
    ("site", ["BER", "LON", "NYC", "SFO", "SYD", "TYO"], "code", ["A1", "B2", "C3", "D4", "E5"]),
]


def _pick_scheme(rng: random.Random):
    """-> (f_out, out_vals (3-8, natural order), f_in, in_vals)."""
    f_out, ovocab, f_in, ivocab = rng.choice(_SCHEMES)
    # Cap the joint key space at ~18: the binary root holds two children's tallies plus the merge
    # plus the per-outer reduction table; 40 keys measured 3.4k real tokens and 24 keys 2.96k
    # (multi-token values like "Mar 2024" / "Sci/Tech" make each key ~9 tokens). 18 -> ~2.6k.
    k = rng.randint(3, min(8, len(ovocab), max(3, 18 // len(ivocab))))
    out_vals = sorted(rng.sample(ovocab, k), key=ovocab.index)
    return f_out, out_vals, f_in, list(ivocab)


def _one_record_m(rng: random.Random, idx: int, out_vals: list[str], in_vals: list[str]) -> dict:
    return {
        "idx": idx,
        "id": rng.randint(1000, 9999),
        "out": rng.choice(out_vals),
        "in": rng.choice(in_vals),
        "amt": rng.randint(-20, 20),
        "flag": "Y" if rng.random() < 0.5 else "N",
    }


def _render_m(r: dict, f_out: str, f_in: str) -> str:
    sign = f"+{r['amt']}" if r["amt"] >= 0 else str(r["amt"])
    return f"[{r['idx']:04d}] id={r['id']} {f_out}={r['out']} {f_in}={r['in']} amt={sign} flag={r['flag']}"


def _context_m(f_out, out_vals, f_in, in_vals) -> str:
    return (
        "The document is a list of records, one per line, each formatted as:\n"
        f"  [<index>] id=<int> {f_out}=<{'|'.join(out_vals)}> {f_in}=<{'|'.join(in_vals)}> "
        "amt=<signed int> flag=<Y|N>\n"
        f"Records are 0-indexed and appear in order. The {f_out} values are listed above in their "
        "natural order."
    )


def _argbest(counts: dict, best, order):
    """Key with the best count (max or min) among keys PRESENT in `counts`; ties -> first in
    `order`. Matches the oracle's finalize exactly."""
    target = best(counts.values())
    return min((k for k, n in counts.items() if n == target), key=order.index)


def _month_question_gold(task: str, recs: list[dict], sch, rng: random.Random):
    """-> (question, gold, grading_mode, metadata-params) for the 2-D record variant families.
    `sch` = (f_out, out_vals, f_in, in_vals)."""
    f_out, out_vals, f_in, in_vals = sch
    ex_in = in_vals[1]
    if task == "synth_filter_argmax":
        if rng.random() < 0.5:
            ffield, fval = "flag", rng.choice(["Y", "N"])
        else:
            ffield, fval = f_out, rng.choice(out_vals)
        agg = rng.choice(["most", "least"])
        c = Counter(r["in"] for r in recs if (r["flag"] if ffield == "flag" else r["out"]) == fval)
        gold = _argbest(c, max if agg == "most" else min, in_vals) if c else in_vals[0]
        q = filtered_question(
            rng, "records", f"{ffield}={fval}", f"which {f_in} value is the {agg.upper()} common",
            f"Consider only {f_in} values that appear at least once among those records; break ties "
            f"by the alphabetically first {f_in}. Give the {f_in} (e.g. {ex_in}) in \\boxed{{}}.")
        return q, gold, "exact", {"qtype": "filter_argmax", "ffield": ffield, "fval": fval, "agg": agg}

    # synth_2d: joint tally cnt[outer][inner]
    cnt = {m: Counter() for m in out_vals}
    for r in recs:
        cnt[r["out"]][r["in"]] += 1
    qtype = rng.choice(["months_argmax", "months_argmax", "months_cmp", "grp_in_month", "month_for_grp"])
    if qtype == "months_argmax":
        g = rng.choice(in_vals)
        gold = sum(1 for m in out_vals
                   if cnt[m][g] > max((n for k, n in cnt[m].items() if k != g), default=0))
        q = (f"For how many {f_out} values is {f_in}={g} the single most common {f_in} — i.e. that "
             f"{f_out} has STRICTLY more {f_in}={g} records than records of any other one {f_in} "
             f"value? Give the single integer in \\boxed{{}}.")
        return q, gold, "numeric", {"qtype": qtype, "qgrp": g}
    if qtype == "months_cmp":
        g1, g2 = rng.sample(in_vals, 2)
        gold = sum(1 for m in out_vals if cnt[m][g1] > cnt[m][g2])
        q = (f"For how many {f_out} values are there STRICTLY more {f_in}={g1} records than "
             f"{f_in}={g2} records? Give the single integer in \\boxed{{}}.")
        return q, gold, "numeric", {"qtype": qtype, "qgrp": g1, "qgrp2": g2}
    if qtype == "grp_in_month":
        m = rng.choice(out_vals)
        gold = _argbest(cnt[m], max, in_vals) if cnt[m] else in_vals[0]
        q = filtered_question(
            rng, "records", f"{f_out}={m}", f"which {f_in} value is the MOST common",
            f"Break ties by the alphabetically first {f_in}. Give the {f_in} (e.g. {ex_in}) in \\boxed{{}}.")
        return q, gold, "exact", {"qtype": qtype, "qmon": m}
    g = rng.choice(in_vals)   # month_for_grp
    per = {m: cnt[m][g] for m in out_vals}
    gold = _argbest(per, max, out_vals)
    q = (f"Which {f_out} has the MOST {f_in}={g} records? Break ties by the {f_out} that comes "
         f"earliest in the natural order given in the document description. Give the {f_out} value "
         f"(e.g. {out_vals[0]}) in \\boxed{{}}.")
    return q, gold, "exact", {"qtype": qtype, "qgrp": g}


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
    "synth_peak": "Process the records in order, keeping a running total of 'amt' (starting at 0). "
                  "What is the HIGHEST value the running total ever reaches (0 if it never goes "
                  "positive)? Give the single integer in \\boxed{}.",
    "synth_streak": "Process the records in order. What is the length of the LONGEST run of "
                    "CONSECUTIVE records with flag=Y? Give the single integer in \\boxed{}.",
    "synth_adjacent": "Process the records in order. How many records have an 'amt' STRICTLY "
                      "GREATER than the 'amt' of the record immediately before them? (The first "
                      "record has no predecessor.) Give the single integer in \\boxed{}.",
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

    if task in _MONTH_TASKS:
        sch = _pick_scheme(rng)
        f_out, out_vals, f_in, in_vals = sch
        recs, doc_tokens, raw_spans = _pack(rng, lambda r: _render_m(r, f_out, f_in),
                                            lambda i: _one_record_m(rng, i, out_vals, in_vals),
                                            doc_size_tokens, tokenizer)
        spans = [(s_, e, r["idx"], r["amt"], r["flag"], r["in"], r["out"]) for (s_, e, r) in raw_spans]
        question, gold, grading, qparams = _month_question_gold(task, recs, sch, rng)
        return Problem(
            document_tokens=doc_tokens, question=question, gold_answers=[str(gold)], task=task,
            task_context=_context_m(f_out, out_vals, f_in, in_vals), grading_mode=grading,
            metadata={"family": "bounded", "strategy_default": "binary", "task": task,
                      "n_records": len(recs), "record_spans": spans, "gold_int": gold,
                      "f_out": f_out, "out_vals": out_vals, "f_in": f_in, "in_vals": in_vals,
                      "months": out_vals, **qparams},
        )

    # Standard-record tasks.
    recs, doc_tokens, raw_spans = _pack(rng, _render, lambda i: _one_record(rng, i), doc_size_tokens, tokenizer)
    spans = [(s, e, r["idx"], r["amt"], r["flag"], r["grp"]) for (s, e, r) in raw_spans]
    if task in _PARAM_TASKS:
        question, gold, qparams = _param_question_gold(task, recs, rng)
    else:
        question, gold, qparams = _QUESTION[task], gold_for(task, recs), {}
    grading = qparams.pop("_grading", "exact" if task in ("synth_mode", "synth_sumby") else "numeric")
    return Problem(
        document_tokens=doc_tokens, question=question, gold_answers=[str(gold)], task=task,
        task_context=_CONTEXT, grading_mode=grading,
        metadata={"family": SYNTH_TASKS[task]["family"], "strategy_default": SYNTH_TASKS[task]["strategy"],
                  "task": task, "n_records": len(recs), "record_spans": spans, "gold_int": gold, **qparams},
    )
