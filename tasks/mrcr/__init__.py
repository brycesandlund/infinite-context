"""OpenAI MRCR (multi-round co-reference resolution; openai/mrcr, after Vodrahalli et al. 2024, arXiv 2409.12640).
EVAL ONLY: never used for training.

A long synthetic chat in which the user asks for the same kind of text about the same topic several times (2, 4 or 8
"needles") among other requests; the final message asks for the i-th one, reproduced verbatim behind a 10-character
prefix.

Data: `scripts/prepare_mrcr.py` (one-time) turns the HF snapshot into per-problem token arrays + an index.

Mapping onto our harness:
  document      = every message but the last, written out mechanically as "User: …" / "Assistant: …" lines in the
                  format the dataset's own worked examples use (prepare_mrcr.py); nothing dropped or rewritten, so the
                  worked-examples message the dataset starts with is the start of the document;
  task_context  = one sentence saying the document is a chat transcript in that format (the dataset gives a chat, our
                  harness gives a document: this is the only thing we add);
  question      = the dataset's final user message, verbatim ("Prepend <prefix> to the <i>th (1 indexed) <format>
                  about <topic>. Do not include any other text in your response.");
  grading       = "mrcr", the card's grader: 0 unless the response starts with the prefix, else difflib
                  SequenceMatcher ratio between response and answer (prefix removed from both). The official response
                  is the model's whole reply; ours is the content of the root's \\boxed{}, so an answer that is not
                  boxed scores 0 (and extract_boxed trims surrounding whitespace).

Selection (env, read per call):
  MRCR_NEEDLES     comma list of 2,4,8; default all
  MRCR_BINS        comma list of bin UPPER edges in o200k tokens (8192, 16384, … 1048576); default every prepared bin
  MRCR_MIN_TOKENS / MRCR_MAX_TOKENS   document-length window in Qwen tokens (default 0 / inf)
A seed indexes a fixed shuffle of the filtered pool, as in the other dataset-backed tasks.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache

import numpy as np

from tasks.base import Problem

MRCR_TASKS = {"mrcr": {"family": "mrcr"}}
_ROOT = os.path.expanduser("~/.cache/infinite-context/mrcr/prepared")
TASK_CONTEXT = ("The document is the transcript of a conversation between a user and an assistant: each turn starts "
                "on a new line with \"User: \" or \"Assistant: \".")


@lru_cache(maxsize=1)
def _index() -> tuple[dict, ...]:
    path = os.path.join(_ROOT, "index.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"MRCR not prepared: run `PYTHONPATH=. uv run python scripts/prepare_mrcr.py` ({path})")
    return tuple(json.loads(l) for l in open(path))


@lru_cache(maxsize=8)
def _pool(needles: str, bins: str, lo: int, hi: int) -> tuple[dict, ...]:
    want_n = {int(x) for x in needles.split(",") if x}
    want_b = {int(x) for x in bins.split(",") if x}
    rows = [r for r in _index()
            if r["doc_tokens"] is not None and lo <= r["doc_tokens"] <= hi
            and (not want_n or r["n_needles"] in want_n) and (not want_b or r["bin"] in want_b)]
    rows.sort(key=lambda r: r["id"])
    random.Random(20251205).shuffle(rows)
    if not rows:
        raise SystemExit(f"MRCR: empty pool (needles={needles or 'all'}, bins={bins or 'all'}, tokens {lo}..{hi}); "
                         f"longer bins need `scripts/prepare_mrcr.py --max-bin <edge>`")
    return tuple(rows)


def make_mrcr_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    pool = _pool(os.environ.get("MRCR_NEEDLES", ""), os.environ.get("MRCR_BINS", ""),
                 int(os.environ.get("MRCR_MIN_TOKENS", "0")), int(os.environ.get("MRCR_MAX_TOKENS", str(10**9))))
    r = pool[seed % len(pool)]
    doc = np.load(os.path.join(_ROOT, "docs", f"{r['id']}.npy")).tolist()
    return Problem(
        document_tokens=doc, question=r["question"], gold_answers=[r["answer"], r["prefix"]], task=task,
        task_context=TASK_CONTEXT, grading_mode="mrcr",
        metadata={"family": "mrcr", "task": task, "dataset": f"{r['n_needles']}needle", "task_type": f"bin_{r['bin']}",
                  "id": r["id"], "desired_msg_index": r["desired_msg_index"], "total_messages": r["total_messages"],
                  "doc_tokens": r["doc_tokens"], "date_added": r["date_added"]},
    )
