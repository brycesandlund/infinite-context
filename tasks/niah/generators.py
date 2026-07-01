"""Synthetic needle-in-a-haystack over REAL novel filler — teaches decomposition of a
prose HAYSTACK (the missing corpus type in our SFT).

Why: the model learned to decompose the synthetic record-list format but single-shots real
prose haystacks (RULER niah/vt/cwe/fwe → reads a partial, abstains). The hypothesis is that
it's the CORPUS/format, not the question — so we insert a mechanical needle into real novel
prose (the same filler distribution as bookqa/narrativeqa) and teach the binary retrieval
decomposition on it. The leaf-op is a faithful SCRIPTED match (find the needle for the queried
key), so no model / rejection is needed — like realdoc, but retrieval instead of counting.

Reuses BookQAOracle: a NIAH problem is just its scripted fallback (leaf_model=None) over a
single needle span — the leaf that owns the needle returns the value, others return "none",
the combine propagates it, the root reads it off. `record_spans` = the one needle span
`(tok_start, tok_end, idx, relevance, snippet, has_answer=True)`.
"""

from __future__ import annotations

import random
import re

from tasks.base import Problem
from tasks.realdoc.books import BOOKS
from tasks.realdoc.generators import _book_tokens

NIAH_TASKS: dict[str, dict] = {
    "niah_novel": {"family": "niah", "strategy": "binary"},
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
_QUESTION = (
    "What is the special magic number for {key}? A sentence somewhere in the document states "
    "it. Give the number in \\boxed{{}}."
)
_CONTEXT = (
    "The document is a long passage of prose with a single factual sentence hidden inside it. "
    "Answer the question using ONLY this passage."
)


def make_niah_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in NIAH_TASKS:
        raise ValueError(f"Unknown niah task: {task!r}")
    rng = random.Random(seed)
    # Novel filler (same distribution as bookqa/narrativeqa).
    name = rng.choice(sorted(BOOKS))
    bt = _book_tokens(name, tokenizer)
    start = rng.randint(0, max(0, len(bt) - doc_size_tokens))
    filler = tokenizer.decode(bt[start:start + doc_size_tokens])

    key = rng.choice(_KEYS)
    value = str(rng.randint(1_000_000, 9_999_999))     # 7-digit magic number, RULER-style
    needle = _NEEDLE.format(key=key, value=value)

    # Insert the needle at a sentence boundary well inside the filler (not the very edges).
    bounds = [m.end() for m in re.finditer(r"[.!?]\s", filler)]
    bounds = [p for p in bounds if 0.05 * len(filler) < p < 0.95 * len(filler)] or [len(filler) // 2]
    pos = rng.choice(bounds)
    doc_text = filler[:pos] + needle + " " + filler[pos:]

    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ncs, nce = pos, pos + len(needle)               # needle char span
    tstart = tend = None
    for i, (cs, ce) in enumerate(offs):
        if ce <= ncs:
            continue
        if cs >= nce:
            break
        if tstart is None:
            tstart = i
        tend = i + 1
    spans = [(tstart, tend, 0, 1, needle.strip(), True)]   # (start,end,idx,rel,snippet,has_answer)

    return Problem(
        document_tokens=ids,
        question=_QUESTION.format(key=key),
        gold_answers=[value],
        task=task,
        task_context=_CONTEXT,
        grading_mode="qa_part",   # word-boundary match on the magic number
        metadata={
            "family": "niah",
            "strategy_default": "binary",
            "task": task,
            "answer": value,
            "key": key,
            "record_spans": spans,
            "k": 12,
        },
    )
