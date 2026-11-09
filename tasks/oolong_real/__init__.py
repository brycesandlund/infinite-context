"""OOLONG-real (Bertsch et al. 2025, arXiv 2511.02817) — Critical Role D&D transcripts, EVAL ONLY.

Never used for training. Data: `scripts/prepare_oolong_real.py` (one-time) turns the HF snapshot
(oolongbench/oolong-real, config `dnd`) into per-window token arrays + a question index. The official harness
evaluates the `test` split (campaign 1); `validation` is campaign 2.

Mapping onto our harness (nothing rewritten):
  task_context  = everything before the first [START OF EPISODE], verbatim: the instruction paragraph AND the
                  player->character mapping (the official harness sends the whole window as the system message;
                  the mapping is task-level context every subagent needs, as OOLONG-synth's description is);
  document      = the transcripts verbatim ([START OF EPISODE] … [END OF EPISODE] blocks);
  question      = the dataset question verbatim;
  grading       = the authors' scorer, vendored verbatim (vendored_eval.py): 0.75^|err| for integers, exact
                  (case-insensitive) for strings, |gold ∩ pred| / |gold| for comma lists.

Selection (env, read per call so one process can run several slices):
  OOLONG_REAL_SPLIT       test (default, the official split) | validation
  OOLONG_REAL_MIN_TOKENS  / OOLONG_REAL_MAX_TOKENS   document-length window in Qwen tokens (default 0 / inf)
  OOLONG_REAL_TYPES       comma list of question_type (singledoc_rolls, singledoc_spells, multidoc_rolls,
                          multidoc_spells); default all
A seed indexes a fixed shuffle of the filtered pool, so consecutive seeds give distinct questions and the same
seed gives the same question everywhere.
"""
from __future__ import annotations

import json
import os
import random
from functools import lru_cache

import numpy as np

from tasks.base import Problem

OOLONG_REAL_TASKS = {"oolong_real": {"family": "oolong_real"}}
_ROOT = os.path.expanduser("~/.cache/infinite-context/oolong_real/prepared")


@lru_cache(maxsize=4)
def _index(split: str) -> tuple[dict, ...]:
    path = os.path.join(_ROOT, split, "index.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"OOLONG-real not prepared: run `PYTHONPATH=. uv run python scripts/prepare_oolong_real.py` ({path})")
    return tuple(json.loads(l) for l in open(path))


@lru_cache(maxsize=8)
def _pool(split: str, lo: int, hi: int, types: str) -> tuple[dict, ...]:
    want = set(filter(None, types.split(",")))
    rows = [r for r in _index(split)
            if lo <= r["doc_tokens"] <= hi and (not want or r["question_type"] in want)]
    rows.sort(key=lambda r: r["id"])
    random.Random(20251103).shuffle(rows)
    if not rows:
        raise SystemExit(f"OOLONG-real: empty pool (split={split}, tokens {lo}..{hi}, types={types or 'all'})")
    return tuple(rows)


def make_oolong_real_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    split = os.environ.get("OOLONG_REAL_SPLIT", "test")
    lo = int(os.environ.get("OOLONG_REAL_MIN_TOKENS", "0"))
    hi = int(os.environ.get("OOLONG_REAL_MAX_TOKENS", str(10**9)))
    pool = _pool(split, lo, hi, os.environ.get("OOLONG_REAL_TYPES", ""))
    r = pool[seed % len(pool)]
    wdir = os.path.join(_ROOT, split, "windows")
    doc = np.load(os.path.join(wdir, f"{r['context_window_id']}.npy")).tolist()
    instr = open(os.path.join(wdir, f"{r['context_window_id']}.instr.txt")).read()
    return Problem(
        document_tokens=doc, question=r["question"], gold_answers=[r["answer"]], task=task,
        task_context=instr, grading_mode="oolong_real",
        metadata={"family": "oolong_real", "task": task, "dataset": r["question_type"], "id": r["id"],
                  "context_window_id": r["context_window_id"], "episodes": r["episodes"],
                  "campaign": r["campaign"], "split": split, "doc_tokens": r["doc_tokens"]},
    )
