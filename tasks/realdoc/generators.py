"""Realdoc tasks — REAL prose (Gutenberg novels), questions with EXACT computable gold.

The bridge from synthetic records to real text: the document is a contiguous slice
of an actual novel, and the question is one whose answer we can compute exactly by
matching the text — so the oracle trace is clean and faithful while the leaf-op is
"parse real prose" (the transfer target), not "parse a synthetic record".

v1 task: realdoc_count — "how many times does the word 'X' appear?" Leaf counts the
occurrences in its chunk; root sums (the same bounded-associative combine as
synth_count/sum, but over real prose). `metadata["record_spans"]` is one entry per
occurrence (its start token + a context snippet), so the oracle maps a token range to
the occurrences that start in it — exactly like the synthetic tasks.
"""

from __future__ import annotations

import random
import re

from tasks.base import Problem
from tasks.realdoc.books import BOOKS, load_book_text

REALDOC_TASKS: dict[str, dict] = {
    "realdoc_count": {"family": "realdoc", "strategy": "binary"},
}

# Module-level tokenized-book cache: tokenizing a 180k-token novel per problem is slow.
_TOK_CACHE: dict[str, list[int]] = {}

# Candidate countable entities: capitalized words (proper nouns), 3+ letters.
_WORD = re.compile(r"\b[A-Z][a-z]{2,}\b")
# Capitalized-but-not-a-name words (sentence-initial function words, titles, etc.) —
# excluded so the question lands on an actual proper noun.
_STOP = {
    "The", "And", "But", "For", "Nor", "Yet", "His", "Her", "She", "Was", "Had",
    "Has", "Have", "You", "Your", "They", "Their", "Them", "This", "That", "These",
    "Those", "There", "Then", "When", "Where", "What", "Who", "Whom", "Why", "How",
    "Which", "With", "Will", "Would", "Could", "Should", "Are", "Were", "Not", "Now",
    "One", "All", "Any", "Some", "Such", "Said", "Did", "Don", "Yes", "Mr", "Mrs",
    "Miss", "Sir", "Lady", "Chapter", "But", "It", "He", "We", "As", "At", "In", "On",
    "Of", "To", "By", "Up", "So", "If", "Or", "Do", "Is", "Be", "An", "My", "Me",
}


def _book_tokens(name: str, tokenizer) -> list[int]:
    if name not in _TOK_CACHE:
        _TOK_CACHE[name] = tokenizer.encode(load_book_text(name), add_special_tokens=False)
    return _TOK_CACHE[name]


_QUESTION = (
    "How many times does the word '{entity}' appear in the document below? Count "
    "whole-word, case-sensitive occurrences. Give the single integer in \\boxed{{}}."
)
_CONTEXT = (
    "The document is a contiguous passage from a public-domain novel (real prose, "
    "running text). Answer the question about this passage only."
)


def make_realdoc_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in REALDOC_TASKS:
        raise ValueError(f"Unknown realdoc task: {task!r}")
    rng = random.Random(seed)
    name = rng.choice(sorted(BOOKS))
    bt = _book_tokens(name, tokenizer)
    if len(bt) <= doc_size_tokens:
        sl = bt
    else:
        start = rng.randint(0, len(bt) - doc_size_tokens)
        sl = bt[start:start + doc_size_tokens]
    text = tokenizer.decode(sl)
    # Re-encode the slice WITH char offsets, so the token ids we hand out as the
    # document line up exactly with the occurrence token positions we compute below.
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    # Pick an entity that occurs a moderate number of times (non-trivial but bounded).
    counts: dict[str, int] = {}
    for m in _WORD.finditer(text):
        counts[m.group()] = counts.get(m.group(), 0) + 1
    cands = sorted(w for w, c in counts.items() if 4 <= c <= 80 and w not in _STOP)
    if not cands:  # rare (tiny slice) — fall back to the most common NON-stopword
        nonstop = {w: c for w, c in counts.items() if w not in _STOP}
        cands = sorted(nonstop or counts, key=(nonstop or counts).get)[-1:] or ["the"]
    entity = rng.choice(cands)

    # Occurrence spans: each whole-word match's START token index (+ a context snippet),
    # in document order. Pointer walks the offsets monotonically since matches are sorted.
    spans: list[tuple] = []
    ti = 0
    for m in re.finditer(r"\b" + re.escape(entity) + r"\b", text):
        cs = m.start()
        while ti < len(offs) and not (offs[ti][0] <= cs < offs[ti][1]):
            ti += 1
        if ti >= len(offs):
            break
        snip = text[max(0, cs - 18):cs + len(entity) + 18].replace("\n", " ").strip()
        spans.append((ti, ti + 1, len(spans), snip))
    gold = len(spans)

    return Problem(
        document_tokens=ids,
        question=_QUESTION.format(entity=entity),
        gold_answers=[str(gold)],
        task=task,
        task_context=_CONTEXT,
        metadata={
            "family": "realdoc",
            "strategy_default": "binary",
            "task": task,
            "book": name,
            "entity": entity,
            "n_records": len(spans),
            "record_spans": spans,
            "gold_int": gold,
        },
    )
