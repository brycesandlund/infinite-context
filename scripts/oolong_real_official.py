"""Reproduce the OFFICIAL OOLONG-real protocol on a frontier API model, to validate our data + scorer against
the paper's Table 4 (e.g. Claude-Sonnet-4: 50.58 at 55K = single-episode windows, test split).

Exactly as abertsch72/oolong src/eval/eval_script_batched.py: system = ["You are a helpful assistant.",
context_window_text], user = question; scored with the vendored `dnd_score` (== dnd_process_response). Rows are
read RAW from the HF snapshot (not our prepared index), so this also checks that the prepared adapter (which
reads the same rows) inherits a correct gold/question pairing.

    N=30 EPISODES=1 MODEL=anthropic/claude-sonnet-4-20250514 PYTHONPATH=. uv run python scripts/oolong_real_official.py
"""
import asyncio
import json
import os
import random

import litellm

from tasks.oolong_real.vendored_eval import dnd_parse_response, dnd_score

ROOT = os.path.expanduser("~/.cache/infinite-context/oolong_real")
N = int(os.environ.get("N", "30"))
EPISODES = int(os.environ.get("EPISODES", "1"))
MODEL = os.environ.get("MODEL", "anthropic/claude-sonnet-4-20250514")
SPLIT = os.environ.get("SPLIT", "test")
OUT = os.environ.get("OUT", f"eval_results/raw/oolong_real_official_{MODEL.split('/')[-1]}_{EPISODES}ep.jsonl")


async def one(r, sem):
    async with sem:
        resp = await litellm.acompletion(
            model=MODEL, max_tokens=int(os.environ.get("MAX_OUT", "32000")),
            messages=[
                {"role": "system", "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "text", "text": r["context_window_text"]},
                ]},
                {"role": "user", "content": r["question"]},
            ],
        )
        out = resp.choices[0].message.content or ""
        parsed, conf = dnd_parse_response(out)
        return {"id": r["id"], "question_type": r["question_type"], "question": r["question"],
                "answer": r["answer"], "attempted_parse": parsed if isinstance(parsed, (int, str)) else ", ".join(parsed),
                "parse_confidence": conf, "score": dnd_score(r["answer"], out),
                "usage_in": resp.usage.prompt_tokens, "full_answer": out}


async def main():
    rows = []
    with open(os.path.join(ROOT, "dnd", f"{SPLIT}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if len(r["episodes"]) == EPISODES:
                rows.append(r)
    rows.sort(key=lambda r: r["id"])
    random.Random(20251103).shuffle(rows)
    rows = rows[:N]
    sem = asyncio.Semaphore(4)
    res = await asyncio.gather(*[one(r, sem) for r in rows])
    with open(OUT, "w") as g:
        for x in res:
            g.write(json.dumps(x) + "\n")
    by = {}
    for x in res:
        by.setdefault(x["question_type"], []).append(x["score"])
    mean = sum(x["score"] for x in res) / len(res)
    print(f"{MODEL} | {SPLIT} | {EPISODES}-episode windows | n={len(res)} | score {100 * mean:.2f}")
    for k, v in sorted(by.items()):
        print(f"   {k:18s} {100 * sum(v) / len(v):6.2f}  (n={len(v)})")
    print(f"   low-confidence parses: {sum(x['parse_confidence'] == 'low' for x in res)} | mean input tokens "
          f"{sum(x['usage_in'] for x in res) // len(res)} | -> {OUT}")


asyncio.run(main())
