"""Book-QA tasks — REAL questions over real novels (InfiniteBench En.QA).

Real human questions ("who accompanied X to the lighthouse", "what is the family's
summer home") over full novels whose character names are ANONYMIZED, so the answer
must come from reading the text, not pretraining memory — the genuine long-document
semantic-understanding target.

We keep the extractive subset (gold answer appears in the context), so the oracle is
faithful: the **leaf-op** retrieves sentences mentioning the question's entities (a
legit operation from the question alone); the **combine** keeps the top-K most-relevant
evidence sentences (bounded → no overflow); the **root** reads the collected evidence
and emits the gold answer (which an entity sentence contains). Strategy: binary scan.

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
        snip = re.sub(r"\s+", " ", sent).strip()[:140].replace("}", ")")
        spans.append((ti, ti + 1, len(spans), len(mentioned), snip, has_ans))
    return spans, grounded


def make_bookqa_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    rng = random.Random(seed)
    rows = _rows()
    span_chars = doc_size_tokens * 4
    chosen = None
    # Resample until we find an entity-retrievable question: one whose answer sits in a
    # sentence that ALSO mentions a question entity, so the entity-retrieval leaf surfaces
    # it without the oracle ever consulting the gold. (Fallback to the last try.)
    for _ in range(80):
        row = rows[rng.randrange(len(rows))]
        if _SKIP_Q.search(row["question"]):
            continue   # comparison/aggregation — our oracle can't answer it faithfully
        ents = _entities(row["question"], row["context"])
        if not ents:
            continue
        ai = max(0, row["context"].find(row["answer"]))
        cs = max(0, ai - span_chars // 2)
        window = row["context"][cs:cs + span_chars]
        enc = tokenizer(window, return_offsets_mapping=True, add_special_tokens=False)
        spans, _ = _build_spans(window, enc["offset_mapping"], ents, row["answer"])
        # STRICT grounding: the answer-bearing sentence must actually survive the oracle's
        # global top-K (merge keeps top-K by rel, ties broken by doc order), or the root
        # would have to conjure the answer from evidence that doesn't include it.
        kept = sorted(spans, key=lambda s: (-s[3], s[2]))[:K_EVIDENCE]
        grounded = any(s[5] for s in kept)
        chosen = (row, ents, window, enc, spans)
        if grounded:
            break

    row, ents, window, enc, spans = chosen
    q, ans = row["question"], row["answer"]
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
