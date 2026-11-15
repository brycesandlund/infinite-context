"""One-time preparation of OpenAI MRCR (openai/mrcr; 2 / 4 / 8 needles x 8 length bins x 100) for our harness. EVAL ONLY.

Input:  ~/.cache/infinite-context/mrcr/raw/{2,4,8}needle/*.parquet   (HF snapshot with the 2025-12-05 fixes, ~1.4 GB)
Output: ~/.cache/infinite-context/mrcr/prepared/
          docs/<id>.npy   the conversation as ONE document (Qwen tokens, int32) — only for rows up to --max-bin
          index.jsonl     one row per problem: id (<n>needle_<row>), n_needles, question (the final user message),
                          answer, prefix, desired_msg_index, total_messages, n_chars, date_added,
                          o200k_tokens + bin (the dataset card's rule), doc_tokens (null if not tokenized yet)

The dataset's `prompt` is a chat: messages[0] is a user message of worked examples, then alternating user requests and
assistant replies, and the LAST user message is the question. Our harness takes a document and a question, so every
message but the last is written out mechanically in the format the dataset's own examples use — one line break between
turns, nothing dropped, reordered or rewritten:

    User: <content>
    Assistant: <content>
    User: <content>
    ...

Bins follow the card: o200k_base tokens of all message contents plus the answer, in (lo, hi] with edges 4096 * 2^k
(the first bin is closed at 4096).

    PYTHONPATH=. uv run python scripts/prepare_mrcr.py [--max-bin 131072]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import tiktoken
from tinker_cookbook import tokenizer_utils

import rl

ROOT = os.path.expanduser("~/.cache/infinite-context/mrcr")
EDGES = [4096 * 2 ** k for k in range(9)]   # 4096 … 1048576
ROLE = {"user": "User", "assistant": "Assistant"}


def flatten(messages: list[dict]) -> str:
    return "".join(f"{ROLE[m['role']]}: {m['content']}\n" for m in messages)


def bin_hi(n: int) -> int | None:
    """Upper edge of the card's bin for a token count (None outside 4096..1048576)."""
    if n < EDGES[0] or n > EDGES[-1]:
        return None
    return next(hi for hi in EDGES[1:] if n <= hi)


def main():
    max_bin = int(sys.argv[sys.argv.index("--max-bin") + 1]) if "--max-bin" in sys.argv else 131072
    tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
    enc = tiktoken.get_encoding("o200k_base")
    out = os.path.join(ROOT, "prepared")
    os.makedirs(os.path.join(out, "docs"), exist_ok=True)
    rows = []
    for n in (2, 4, 8):
        files = sorted(glob.glob(os.path.join(ROOT, "raw", f"{n}needle", "*.parquet")))
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        for i, r in enumerate(df.itertuples(index=False)):
            msgs = json.loads(r.prompt)
            assert msgs[-1]["role"] == "user" and int(r.n_needles) == n
            assert r.answer.startswith(r.random_string_to_prepend) and r.random_string_to_prepend in msgs[-1]["content"]
            o200k = sum(len(x) for x in enc.encode_ordinary_batch([m["content"] for m in msgs] + [r.answer]))
            pid, hi = f"{n}needle_{i:03d}", bin_hi(o200k)
            path, doc_tokens = os.path.join(out, "docs", f"{pid}.npy"), None
            if hi is not None and hi <= max_bin:
                if os.path.exists(path):
                    doc_tokens = int(np.load(path, mmap_mode="r").shape[0])
                else:
                    ids = tok.encode(flatten(msgs[:-1]), add_special_tokens=False)
                    np.save(path, np.asarray(ids, dtype=np.int32))
                    doc_tokens = len(ids)
            rows.append({
                "id": pid, "n_needles": n, "question": msgs[-1]["content"], "answer": r.answer,
                "prefix": r.random_string_to_prepend, "desired_msg_index": int(r.desired_msg_index),
                "total_messages": int(r.total_messages), "n_chars": int(r.n_chars), "date_added": str(r.date_added),
                "o200k_tokens": o200k, "bin": hi, "doc_tokens": doc_tokens,
            })
            if (i + 1) % 200 == 0:
                print(f"  {n}needle {i + 1}/{len(df)}", flush=True)
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} problems; tokenized {sum(r['doc_tokens'] is not None for r in rows)} (bins <= {max_bin})")
    for n in (2, 4, 8):
        c = {}
        for r in rows:
            if r["n_needles"] == n:
                c[r["bin"]] = c.get(r["bin"], 0) + 1
        print(f"  {n} needles, problems per bin: {dict(sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)))}")


if __name__ == "__main__":
    main()
