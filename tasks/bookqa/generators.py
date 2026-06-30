"""Book-QA tasks — REAL questions over real novels (InfiniteBench En.QA).

Real human questions ("who accompanied X to the lighthouse", "what is the family's
summer home") over full novels whose character names are ANONYMIZED, so the answer
must come from reading the text, not pretraining memory — the genuine long-document
semantic-understanding target.

We keep the extractive subset (gold answer appears in the context), so the oracle is
faithful. The subtask names ONLY the question (never the entities): each **leaf** reads
its range and tries to answer — if the answer sentence is in range it returns the answer
with that sentence; otherwise it returns any relevant context, or "No relevant information
in this range." The **combine** propagates a child's answer up (or merges the bounded
top-K relevant context); the **root** reads the answer off the surfaced sentence. Strategy:
binary scan.

The question entities are used only HERE, to build the candidate-evidence spans and mark
which sentence carries the answer — they never appear in any prompt the model sees.
`metadata["record_spans"]` is one entry per candidate evidence sentence:
`(start_tok, end_tok, idx, relevance, snippet, has_answer)`.
"""

from __future__ import annotations

import json
import os
import random
import re

from tasks.base import Problem

_CACHE = os.path.expanduser("~/.cache/infinite-context/bookqa/enqa.jsonl")
_ROWS: list[dict] | None = None

BOOKQA_TASKS: dict[str, dict] = {"bookqa": {"family": "bookqa", "strategy": "binary"}}

K_EVIDENCE = 12   # bounded combine: keep at most this many evidence sentences up the tree
_STOPQ = {"Which", "What", "Who", "Whom", "When", "Where", "Why", "How", "The", "Mr",
          "Mrs", "Miss", "For", "And", "Did", "Does", "Is", "Are", "Was", "Were",
          # common titles / function words that get capitalized in title-cased questions
          "Ser", "Lord", "Lady", "King", "Queen", "Prince", "Princess", "Sir", "Saint",
          "His", "Her", "Their", "Love", "Youth", "Life", "Death", "Name", "Home", "In",
          "Of", "On", "At", "To", "By", "With", "From", "About", "Into", "Over"}
_SENT = re.compile(r"[^.!?]*[.!?]")   # crude sentence splitter (good enough for evidence units)
# Questions our retrieval+collect oracle CANNOT answer faithfully: comparison /
# superlative / aggregation needs reasoning over many mentions, not a single retrieved
# sentence, so the root would have to assert the gold. Keep only direct factoid retrieval.
_SKIP_Q = re.compile(r"\bamong\b|\bhow many\b|\bhow much\b|\bcompare\b|\brank\b|\boldest\b"
                     r"|\byoungest\b|\bmost\b|\bleast\b| or .+\?", re.I)


def _rows() -> list[dict]:
    global _ROWS
    if _ROWS is None:
        if not os.path.exists(_CACHE):
            raise FileNotFoundError(
                f"{_CACHE} missing — cache InfiniteBench En.QA first (see scripts/cache_bookqa)."
            )
        _ROWS = [json.loads(l) for l in open(_CACHE, encoding="utf-8")]
    return _ROWS


def _entities(question: str, context: str) -> list[str]:
    """Proper nouns named in the question. Title-cased questions capitalize every word,
    so keep only candidates that appear PREDOMINANTLY CAPITALIZED in the prose (real names
    like 'Barristan' are always capitalized; 'love'/'youth' are mostly lowercase)."""
    keep = []
    for e in sorted(set(re.findall(r"\b[A-Z][a-z]{2,}\b", question)) - _STOPQ):
        cap = len(re.findall(r"\b" + re.escape(e) + r"\b", context))
        low = len(re.findall(r"\b" + re.escape(e.lower()) + r"\b", context))
        if cap >= 2 and cap >= low:   # a genuine proper noun in this text
            keep.append(e)
    return keep


def _build_spans(window, offs, ents, ans):
    """Evidence = sentences mentioning a question ENTITY (relevance = #entities). NO
    answer-based selection — the leaf must retrieve on the question alone, or the oracle
    leaks the gold. Returns (spans, grounded) where grounded means an entity sentence
    also contains the answer (so entity-retrieval genuinely surfaces it)."""
    spans, grounded, ti = [], False, 0
    for m in _SENT.finditer(window):
        sent = m.group()
        mentioned = [e for e in ents if re.search(r"\b" + re.escape(e) + r"\b", sent)]
        if not mentioned:
            continue
        sc = m.start()
        while ti < len(offs) and not (offs[ti][0] <= sc < offs[ti][1]):
            ti += 1
        if ti >= len(offs):
            break
        has_ans = ans in sent
        grounded = grounded or has_ans
        flat = re.sub(r"\s+", " ", sent).strip()
        if has_ans:
            # center the 140-char snippet on the answer so the displayed evidence actually
            # SHOWS the answer — the oracle reads the answer off this snippet, so it must be
            # visible (otherwise the answer would appear conjured from a sentence not showing it)
            j = flat.find(ans); lo = max(0, j - 60)
            snip = ("…" if lo > 0 else "") + flat[lo:lo + 140]
        else:
            snip = flat[:140]
        snip = snip.replace("}", ")")
        spans.append((ti, ti + 1, len(spans), len(mentioned), snip, has_ans))
    return spans, grounded


def _best_answer_pos(context: str, ans: str, ents: list[str]) -> tuple[int, int]:
    """The answer occurrence whose surrounding sentence mentions the MOST question entities —
    the best guess at the actual evidence sentence (a bare name can recur many times; the one
    co-located with the question's subjects is the one that answers). Returns (char_pos, score)."""
    best_pos, best_score, start = context.find(ans), -1, 0
    while True:
        j = context.find(ans, start)
        if j < 0:
            break
        ls = context.rfind(".", 0, j) + 1
        le = context.find(".", j)
        sent = context[ls: le if le >= 0 else len(context)]
        score = sum(1 for e in ents if re.search(r"\b" + re.escape(e) + r"\b", sent))
        if score > best_score:
            best_score, best_pos = score, j
        start = j + 1
    return best_pos, best_score


_ELIGIBLE: list[tuple] | None = None


def _eligible_rows() -> list[tuple]:
    """Usable bookqa rows as (row, answer_pos): non-comparison (`_SKIP_Q`), extractive (gold
    verbatim), and ANSWERABLE-by-local-retrieval (some answer occurrence sits in a sentence
    that also names a question entity — strongly correlated with the model finding it). Kept in
    a FIXED shuffled order so indexing by seed is 1:1: consecutive SFT seeds draw DISTINCT,
    answerable questions (no duplicate funnelling, high accept rate). Computed once."""
    global _ELIGIBLE
    if _ELIGIBLE is None:
        elig = []
        for r in _rows():
            if _SKIP_Q.search(r["question"]) or r["answer"] not in r["context"]:
                continue
            ents = _entities(r["question"], r["context"])
            if not ents:
                continue
            pos, score = _best_answer_pos(r["context"], r["answer"], ents)
            if score >= 1:                      # the answer co-occurs with a question entity
                elig.append((r, pos))
        random.Random(20240607).shuffle(elig)   # fixed permutation — deterministic across runs
        _ELIGIBLE = elig
    return _ELIGIBLE


def bookqa_corpus_size() -> int:
    """Distinct usable questions — the ceiling on duplicate-free bookqa traces."""
    return len(_eligible_rows())


def make_bookqa_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    eligible = _eligible_rows()
    row, ai = eligible[seed % len(eligible)]   # 1:1 seed -> distinct, answerable question
    span_chars = doc_size_tokens * 4
    q, ans = row["question"], row["answer"]
    ents = _entities(q, row["context"])
    # Center the window on the BEST answer occurrence (its sentence names a question entity),
    # so the evidence the model needs is actually in-window. Spans built for the scripted fallback.
    cs = max(0, ai - span_chars // 2)
    window = row["context"][cs:cs + span_chars]
    enc = tokenizer(window, return_offsets_mapping=True, add_special_tokens=False)
    spans, _ = _build_spans(window, enc["offset_mapping"], ents, ans)
    ids = enc["input_ids"]

    return Problem(
        document_tokens=ids,
        question=q,
        gold_answers=[ans],
        task=task,
        task_context=(
            "The document is a passage from a novel (character names are anonymized). "
            "Answer the question using ONLY this passage."
        ),
        grading_mode="ruler_part",   # substring match — QA answers are short free-form
        metadata={
            "family": "bookqa",
            "strategy_default": "binary",
            "task": task,
            "entities": ents,
            "answer": ans,
            "k": K_EVIDENCE,
            "record_spans": spans,
        },
    )
