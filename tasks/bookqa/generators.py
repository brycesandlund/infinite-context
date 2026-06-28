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
          "Mrs", "Miss", "For", "And", "Did", "Does", "Is", "Are", "Was", "Were"}
_SENT = re.compile(r"[^.!?]*[.!?]")   # crude sentence splitter (good enough for evidence units)


def _rows() -> list[dict]:
    global _ROWS
    if _ROWS is None:
        if not os.path.exists(_CACHE):
            raise FileNotFoundError(
                f"{_CACHE} missing — cache InfiniteBench En.QA first (see scripts/cache_bookqa)."
            )
        _ROWS = [json.loads(l) for l in open(_CACHE, encoding="utf-8")]
    return _ROWS


def _entities(question: str) -> list[str]:
    return sorted(set(re.findall(r"\b[A-Z][a-z]{2,}\b", question)) - _STOPQ)


def make_bookqa_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    rng = random.Random(seed)
    row = _rows()[rng.randrange(len(_rows()))]
    q, ans, ctx = row["question"], row["answer"], row["context"]
    ents = _entities(q)

    # Window the (huge) novel context around a gold occurrence so the answer sentence is
    # present, then re-encode with offsets so token positions match document_tokens.
    ai = max(0, ctx.find(ans))
    span_chars = doc_size_tokens * 4
    cs = max(0, ai - span_chars // 2)
    window = ctx[cs:cs + span_chars]
    enc = tokenizer(window, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]

    # Candidate evidence sentences: mention a question entity (relevance = #entities), with
    # the answer-bearing sentence boosted so it survives the top-K combine.
    spans: list[tuple] = []
    ti = 0
    for m in _SENT.finditer(window):
        sent = m.group()
        mentioned = [e for e in ents if re.search(r"\b" + re.escape(e) + r"\b", sent)]
        has_ans = ans in sent
        if not mentioned and not has_ans:
            continue
        sc = m.start()
        while ti < len(offs) and not (offs[ti][0] <= sc < offs[ti][1]):
            ti += 1
        if ti >= len(offs):
            break
        relevance = len(mentioned) + (3 if has_ans else 0)
        snip = re.sub(r"\s+", " ", sent).strip()[:140].replace("}", ")")
        spans.append((ti, ti + 1, len(spans), relevance, snip, has_ans))

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
