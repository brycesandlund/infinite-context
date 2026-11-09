"""Cache the InfiniteBench En.QA extractive subset for the bookqa tasks.

Streams `longbook_qa_eng`, keeps examples whose short gold answer appears verbatim in
the (anonymized) novel context — so the bookqa oracle's collected evidence genuinely
contains the answer — and writes them to ~/.cache/infinite-context/bookqa/enqa.jsonl.

Run:  N=150 uv run python scripts/cache_bookqa.py
"""

import json
import os

from datasets import load_dataset

N = int(os.environ.get("N", "150"))
MAX_SEEN = int(os.environ.get("MAX_SEEN", "1000"))
cache = os.path.expanduser("~/.cache/infinite-context/bookqa")
os.makedirs(cache, exist_ok=True)
out = os.path.join(cache, "enqa.jsonl")

ds = load_dataset("xinrongzhang2022/InfiniteBench", streaming=True, split="longbook_qa_eng")
kept = seen = 0
with open(out, "w", encoding="utf-8") as f:
    for ex in ds:
        seen += 1
        ans = ex["answer"][0].strip().strip('"') if ex["answer"] else ""
        ctx = ex["context"]
        if ans and len(ans) < 40 and ans in ctx:   # extractive: gold present in context
            f.write(json.dumps({"id": ex["id"], "question": ex["input"],
                                "answer": ans, "context": ctx}) + "\n")
            kept += 1
        if kept >= N or seen >= MAX_SEEN:
            break
print(f"seen={seen} kept={kept} -> {out} ({os.path.getsize(out) // 1024 // 1024}MB)")
