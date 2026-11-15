"""LongBench v2 (Bai et al. 2024, arXiv 2412.15204) — 503 four-way multiple-choice questions on real long documents.
EVAL ONLY: never used for training.

Data: `scripts/prepare_longbench_v2.py` (one-time) turns the HF snapshot (THUDM/LongBench-v2) into per-question token
arrays + an index.

Mapping onto our harness:
  document      = the dataset's `context`, verbatim;
  task_context  = empty (the official prompt has no task description beyond "read the text and answer");
  question      = the official 0-shot template's question block, verbatim ("What is the correct answer to this
                  question: … Choices: (A) … (D) …"), with its last line — 'Format your response as follows: "The
                  correct answer is (insert answer here)"' — replaced by a request for the letter, because our harness
                  takes the answer from \\boxed{};
  grading       = "choice": the official extraction (their two regexes on "The correct answer is (X)") or a bare
                  letter, then exact match with the gold letter. Chance is 0.25.
The official protocol puts the whole context in the prompt and cuts the MIDDLE out of prompts longer than the model's
window (pred.py); MODE=single here does not truncate, so use it only where the document fits.

Selection (env, read per call):
  LONGBENCH_LENGTH        comma list of the dataset's length labels (short < 32K words, medium 32K-128K, long 128K-2M)
  LONGBENCH_DOMAIN        comma list of `domain` values; default all
  LONGBENCH_DIFFICULTY    easy | hard; default both
  LONGBENCH_MIN_TOKENS / LONGBENCH_MAX_TOKENS   document-length window in Qwen tokens (default 0 / inf)
A seed indexes a fixed shuffle of the filtered pool: consecutive seeds give distinct questions (until the pool wraps)
and the same seed gives the same question everywhere.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache

import numpy as np

from tasks.base import Problem

LONGBENCH_V2_TASKS = {"longbench_v2": {"family": "longbench_v2"}}
_ROOT = os.path.expanduser("~/.cache/infinite-context/longbench_v2/prepared")


@lru_cache(maxsize=1)
def _index() -> tuple[dict, ...]:
    path = os.path.join(_ROOT, "index.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"LongBench v2 not prepared: run `PYTHONPATH=. uv run python scripts/prepare_longbench_v2.py` ({path})")
    return tuple(json.loads(l) for l in open(path))


@lru_cache(maxsize=8)
def _pool(length: str, domain: str, difficulty: str, lo: int, hi: int) -> tuple[dict, ...]:
    want_len, want_dom = set(filter(None, length.split(","))), set(filter(None, domain.split(",")))
    rows = [r for r in _index()
            if lo <= r["doc_tokens"] <= hi and (not want_len or r["length"] in want_len)
            and (not want_dom or r["domain"] in want_dom) and (not difficulty or r["difficulty"] == difficulty)]
    rows.sort(key=lambda r: r["id"])
    random.Random(20241219).shuffle(rows)
    if not rows:
        raise SystemExit(f"LongBench v2: empty pool (length={length or 'all'}, domain={domain or 'all'}, tokens {lo}..{hi})")
    return tuple(rows)


def format_question(r: dict) -> str:
    c = r["choices"]
    return (f"What is the correct answer to this question: {r['question'].strip()}\n"
            f"Choices:\n(A) {c['A'].strip()}\n(B) {c['B'].strip()}\n(C) {c['C'].strip()}\n(D) {c['D'].strip()}\n\n"
            f"Answer with the letter of the correct choice (A, B, C or D).")


def make_longbench_v2_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    pool = _pool(os.environ.get("LONGBENCH_LENGTH", ""), os.environ.get("LONGBENCH_DOMAIN", ""),
                 os.environ.get("LONGBENCH_DIFFICULTY", ""),
                 int(os.environ.get("LONGBENCH_MIN_TOKENS", "0")), int(os.environ.get("LONGBENCH_MAX_TOKENS", str(10**9))))
    r = pool[seed % len(pool)]
    doc = np.load(os.path.join(_ROOT, "docs", f"{r['id']}.npy")).tolist()
    return Problem(
        document_tokens=doc, question=format_question(r), gold_answers=[r["answer"]], task=task,
        task_context="", grading_mode="choice",
        metadata={"family": "longbench_v2", "task": task, "dataset": r["domain"], "task_type": r["length"],
                  "id": r["id"], "sub_domain": r["sub_domain"], "difficulty": r["difficulty"],
                  "doc_tokens": r["doc_tokens"]},
    )
