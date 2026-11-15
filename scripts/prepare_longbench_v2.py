"""One-time preparation of LongBench v2 (THUDM/LongBench-v2, 503 multiple-choice questions) for our harness. EVAL ONLY.

Input:  ~/.cache/infinite-context/longbench_v2/raw/data.json   (HF snapshot, ~465 MB)
Output: ~/.cache/infinite-context/longbench_v2/prepared/
          docs/<_id>.npy   the `context` field, verbatim, as Qwen tokens (int32)
          index.jsonl      one row per question: id, domain, sub_domain, difficulty, length (the dataset's own
                           short / medium / long label, defined in WORDS), question, choices, answer, doc_tokens

    PYTHONPATH=. uv run python scripts/prepare_longbench_v2.py
"""
import json
import os

import numpy as np
from tinker_cookbook import tokenizer_utils

import rl

ROOT = os.path.expanduser("~/.cache/infinite-context/longbench_v2")


def main():
    tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
    data = json.load(open(os.path.join(ROOT, "raw", "data.json")))
    out = os.path.join(ROOT, "prepared")
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    ids = [r["_id"] for r in data]
    assert len(set(ids)) == len(ids), "duplicate _id"
    sizes = []
    with open(os.path.join(out, "index.jsonl"), "w") as idx:
        for i, r in enumerate(data):
            assert r["answer"] in "ABCD" and len(r["answer"]) == 1, r["answer"]
            path = os.path.join(out, "docs", f"{r['_id']}.npy")
            if os.path.exists(path):
                n = int(np.load(path, mmap_mode="r").shape[0])
            else:
                toks = tok.encode(r["context"], add_special_tokens=False)
                np.save(path, np.asarray(toks, dtype=np.int32))
                n = len(toks)
            sizes.append(n)
            idx.write(json.dumps({
                "id": r["_id"], "domain": r["domain"], "sub_domain": r["sub_domain"], "difficulty": r["difficulty"],
                "length": r["length"], "question": r["question"],
                "choices": {c: r[f"choice_{c}"] for c in "ABCD"}, "answer": r["answer"], "doc_tokens": n,
            }) + "\n")
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(data)}", flush=True)
    s = sorted(sizes)
    print(f"{len(data)} questions; doc tokens min/median/p90/max {s[0]}/{s[len(s) // 2]}/{s[int(len(s) * .9)]}/{s[-1]}; "
          f"total {sum(s) / 1e6:.1f}M")


if __name__ == "__main__":
    main()
