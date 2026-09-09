"""Synthetic "hidden sentences in REAL novel filler" tasks — teach decomposition of a prose
HAYSTACK (the corpus type our SFT was missing) with MECHANICAL, faithful leaf-ops.

Why: the model only decomposes documents that look like its training data; on unfamiliar prose
haystacks it single-shots a partial read (or reads its whole range at once) and overflows/abstains.
Inserting mechanical facts into real novel prose and teaching the scaffold on them transfers to
RULER (niah_novel took cwe 0.00 -> 0.97). All leaves here are scripted matches — no model, no
rejection sampling — like realdoc, but retrieval/tracking instead of counting.

- `niah_novel`  : ONE needle "magic number for KEY is VALUE"; question asks for KEY. Uses
                  BookQAOracle's scripted fallback (single answer propagates up).
- `niah_multi`  : K value-needles + ONE hidden QUERY needle ("the key you must look up is KEY_j").
                  Collect-then-resolve: leaves fold facts into a dict {key: value, QUERY: key_j},
                  combine merges, root resolves state[state["QUERY"]]. The RULER multikey/
                  multiquery analog, with a real multi-hop step. Oracle: NiahMultiOracle.
- `vt_novel`    : RULER-shaped variable chains "VAR A = 12345", "VAR B = VAR A" ... in prose.
                  Left-fold (order matters); answer = every variable holding the target value
                  (set-graded). The RULER vt analog. Oracle: VtNovelOracle.

Span layouts (`metadata["record_spans"]`, one per inserted sentence, doc order):
  niah_novel : (tok_start, tok_end, idx, relevance=1, snippet, has_answer=True)
  niah_multi : (tok_start, tok_end, idx, key, value_or_None, is_query)
  vt_novel   : (tok_start, tok_end, idx, name, rhs, is_ref)
"""

from __future__ import annotations

import random
import re
import string

from tasks.base import Problem
from tasks.realdoc.books import BOOKS
from tasks.realdoc.generators import _book_tokens

NIAH_TASKS: dict[str, dict] = {
    "niah_novel": {"family": "niah", "strategy": "binary"},
    "niah_multi": {"family": "niah", "strategy": "binary"},
    "vt_novel": {"family": "niah", "strategy": "left_fold"},
}

# Key pool — plain nouns, à la RULER's magic-number keys (a word the question asks about).
_KEYS = [
    "harbor", "lantern", "compass", "meadow", "cascade", "ember", "thicket", "quarry",
    "beacon", "trellis", "cavern", "orchard", "granite", "willow", "marble", "cinder",
    "vellum", "saffron", "obsidian", "juniper", "cobalt", "bramble", "hollow", "pewter",
    "mariner", "citadel", "gallows", "tempest", "furrow", "lattice", "verdigris", "gable",
    "sextant", "brindle", "cistern", "palisade", "reliquary", "escarpment", "byway", "kiln",
]

_NEEDLE = "One of the special magic numbers for {key} is: {value}."
_QUERY_NEEDLE = "For this task, the key you must look up is {key}."
_QUESTION = (
    "What is the special magic number for {key}? A sentence somewhere in the document states "
    "it. Give the number in \\boxed{{}}."
)
# NOTE: used raw (no .format), so single braces here.
_MULTI_QUESTION = (
    "This document hides several sentences, each stating the special magic number for some "
    "key, and ONE sentence saying which key you must look up. Find that instruction, then report "
    "the magic number for that key. Give the number in \\boxed{}."
)
_CONTEXT = (
    "The document is a long passage of prose with a single factual sentence hidden inside it. "
    "Answer the question using ONLY this passage."
)
_MULTI_CONTEXT = (
    "The document is a long passage of prose with a few short factual sentences hidden inside "
    "it. Answer the question using ONLY this passage."
)
_VT_LIT = "VAR {name} = {value}."
_VT_REF = "VAR {name} = VAR {rhs}."
_VT_QUESTION = (
    "Memorize and track the chains of variable assignment hidden in the text (a line "
    "'VAR A = VAR B' copies B's current value into A). Which variables end up assigned the "
    "value {value}? List every such variable name, comma-separated, in \\boxed{{}}."
)
_VT_CONTEXT = (
    "The document is a long passage of prose with short 'VAR ... = ...' assignment lines "
    "hidden inside it, in order. Answer the question using ONLY this passage."
)


# ---------------------------------------------------------------------------
# shared: novel filler + ordered multi-sentence insertion + token spans
# ---------------------------------------------------------------------------


def _filler(rng, tokenizer, doc_size_tokens) -> str:
    name = rng.choice(sorted(BOOKS))
    bt = _book_tokens(name, tokenizer)
    start = rng.randint(0, max(0, len(bt) - doc_size_tokens))
    return tokenizer.decode(bt[start:start + doc_size_tokens])


def _boundaries(filler: str, k: int, rng) -> list[int]:
    """k distinct sentence-boundary char positions well inside the filler, ascending."""
    bounds = [m.end() for m in re.finditer(r"[.!?]\s", filler)]
    bounds = [p for p in bounds if 0.05 * len(filler) < p < 0.95 * len(filler)]
    if len(bounds) < k:   # degenerate filler — spread evenly instead
        bounds = [int(len(filler) * (i + 1) / (k + 1)) for i in range(k)]
    return sorted(rng.sample(bounds, k))


def _insert(filler: str, positions: list[int], sentences: list[str]):
    """Insert sentences[i] at ascending positions[i] (so DOCUMENT ORDER == list order).
    Returns (doc_text, [(char_start, char_end)]) for each inserted sentence."""
    out, last, spans = [], 0, []
    cur = 0
    for p, s in zip(positions, sentences):
        out.append(filler[last:p]); cur += p - last
        spans.append((cur, cur + len(s)))
        out.append(s + " "); cur += len(s) + 1
        last = p
    out.append(filler[last:])
    return "".join(out), spans


def _tok_spans(offs, char_spans):
    """Map char spans -> (tok_start, tok_end) over an offsets mapping (single monotone pass)."""
    res, i = [], 0
    for cs, ce in char_spans:
        while i < len(offs) and offs[i][1] <= cs:
            i += 1
        ts = i
        j = i
        while j < len(offs) and offs[j][0] < ce:
            j += 1
        res.append((ts, j))
    return res


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def make_niah_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in NIAH_TASKS:
        raise ValueError(f"Unknown niah task: {task!r}")
    if task == "niah_multi":
        return _make_niah_multi(tokenizer, doc_size_tokens, seed)
    if task == "vt_novel":
        return _make_vt_novel(tokenizer, doc_size_tokens, seed)

    rng = random.Random(seed)
    filler = _filler(rng, tokenizer, doc_size_tokens)
    key = rng.choice(_KEYS)
    value = str(rng.randint(1_000_000, 9_999_999))     # 7-digit magic number, RULER-style
    needle = _NEEDLE.format(key=key, value=value)
    doc_text, cspans = _insert(filler, _boundaries(filler, 1, rng), [needle])
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    (ts, te), = _tok_spans(enc["offset_mapping"], cspans)
    spans = [(ts, te, 0, 1, needle.strip(), True)]   # (start,end,idx,rel,snippet,has_answer)
    return Problem(
        document_tokens=enc["input_ids"],
        question=_QUESTION.format(key=key),
        gold_answers=[value],
        task=task,
        task_context=_CONTEXT,
        grading_mode="qa_part",   # word-boundary match on the magic number
        metadata={
            "family": "niah", "strategy_default": "binary", "task": task,
            "answer": value, "key": key, "record_spans": spans, "k": 12,
        },
    )


def _make_niah_multi(tokenizer, doc_size_tokens, seed) -> Problem:
    rng = random.Random(seed)
    filler = _filler(rng, tokenizer, doc_size_tokens)
    k = rng.randint(3, 5)
    keys = rng.sample(_KEYS, k)
    values = rng.sample(range(1_000_000, 9_999_999), k)
    target = rng.randrange(k)
    facts = [(key, str(v), False) for key, v in zip(keys, values)]
    facts.append((keys[target], None, True))          # the hidden lookup instruction
    rng.shuffle(facts)                                # query can land anywhere
    sentences = [
        (_QUERY_NEEDLE if is_q else _NEEDLE).format(key=key, value=val)
        for key, val, is_q in facts
    ]
    doc_text, cspans = _insert(filler, _boundaries(filler, len(sentences), rng), sentences)
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [
        (ts, te, i, key, val, is_q)
        for i, ((ts, te), (key, val, is_q)) in enumerate(zip(_tok_spans(enc["offset_mapping"], cspans), facts))
    ]
    answer = str(values[target])
    return Problem(
        document_tokens=enc["input_ids"],
        question=_MULTI_QUESTION,
        gold_answers=[answer],
        task="niah_multi",
        task_context=_MULTI_CONTEXT,
        grading_mode="qa_part",
        metadata={
            "family": "niah", "strategy_default": "binary", "task": "niah_multi",
            "answer": answer, "target_key": keys[target], "n_needles": k,
            "record_spans": spans,
        },
    )


def _var_names(rng, n) -> list[str]:
    names: set[str] = set()
    while len(names) < n:
        names.add("".join(rng.choice(string.ascii_uppercase) for _ in range(5)))
    return sorted(names, key=lambda _: rng.random())


def _make_vt_novel(tokenizer, doc_size_tokens, seed) -> Problem:
    """RULER-vt shape: a TARGET chain (root literal + hops that copy the previous var) plus a
    distractor chain with a different value, sentences interleaved but each chain in order."""
    rng = random.Random(seed)
    filler = _filler(rng, tokenizer, doc_size_tokens)
    n_target = rng.randint(4, 6)          # vars in the target chain (RULER default is 5)
    n_distract = rng.randint(2, 3)
    names = _var_names(rng, n_target + n_distract)
    t_names, d_names = names[:n_target], names[n_target:]
    t_val, d_val = rng.sample(range(10_000, 99_999), 2)

    def chain(ns, val):
        recs = [(ns[0], str(val), False)]
        recs += [(ns[i], ns[i - 1], True) for i in range(1, len(ns))]
        return recs

    # Interleave the two chains randomly while preserving each chain's internal order.
    t, d = chain(t_names, t_val), chain(d_names, d_val)
    recs, ti, di = [], 0, 0
    while ti < len(t) or di < len(d):
        take_t = di >= len(d) or (ti < len(t) and rng.random() < len(t) / (len(t) + len(d)))
        if take_t:
            recs.append(t[ti]); ti += 1
        else:
            recs.append(d[di]); di += 1
    sentences = [
        (_VT_REF.format(name=n, rhs=r) if is_ref else _VT_LIT.format(name=n, value=r))
        for n, r, is_ref in recs
    ]
    doc_text, cspans = _insert(filler, _boundaries(filler, len(sentences), rng), sentences)
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [
        (ts, te, i, n, r, is_ref)
        for i, ((ts, te), (n, r, is_ref)) in enumerate(zip(_tok_spans(enc["offset_mapping"], cspans), recs))
    ]
    return Problem(
        document_tokens=enc["input_ids"],
        question=_VT_QUESTION.format(value=t_val),
        gold_answers=sorted(t_names),
        task="vt_novel",
        task_context=_VT_CONTEXT,
        grading_mode="set",       # RULER vt's training grader: set F1 over variable names
        metadata={
            "family": "niah", "strategy_default": "left_fold", "task": "vt_novel",
            "target_value": t_val, "record_spans": spans,
        },
    )
