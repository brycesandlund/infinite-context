"""Throughput of our vLLM server on agent-shaped requests:
    PYTHONPATH=. uv run python scripts/vllm_load_test.py --url https://... --data <export dir> [--n 600 --conc 128]

Each request is one training conversation cut at the start of its LAST assistant turn (the longest prompt that agent
ever sends), asking for up to --max-tokens. Conversations are distinct, so only the system prompt is shared between
requests: real evals re-send growing prefixes and benefit more from vLLM's prefix cache, i.e. this is a lower bound."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from offtinker.core import Dataset  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=600); ap.add_argument("--conc", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=200); ap.add_argument("--start", type=int, default=20000)
    a = ap.parse_args()
    ds = Dataset(a.data)
    key = Path("~/.cache/infinite-context/vllm_api_key.txt").expanduser().read_text().strip()
    http = httpx.AsyncClient(base_url=a.url.rstrip("/"), headers={"Authorization": f"Bearer {key}"}, timeout=900,
                             limits=httpx.Limits(max_connections=a.conc + 8))
    prompts = []
    for u in dict.fromkeys(int(x) for x in ds.order[a.start:a.start + 4 * a.n]):
        seq, mask = ds.datum(u)
        on = np.nonzero(mask)[0]
        starts = on[np.insert(np.diff(on) > 1, 0, True)]            # first token of every assistant turn
        prompts.append([int(t) for t in seq[:int(starts[-1])]])
        if len(prompts) == a.n:
            break
    sem = asyncio.Semaphore(a.conc)
    lat, pt, ot = [], [], []

    async def one(p):
        async with sem:
            t = time.time()
            for _ in range(60):
                r = await http.post("/v1/completions", json={
                    "model": "ic", "prompt": p, "temperature": 0.2, "top_p": 1.0, "max_tokens": a.max_tokens,
                    "stop_token_ids": [248046], "skip_special_tokens": False, "add_special_tokens": False})
                if r.status_code == 200:
                    break
                await asyncio.sleep(10)
            r.raise_for_status()
            u = r.json()["usage"]
            lat.append(time.time() - t); pt.append(u["prompt_tokens"]); ot.append(u["completion_tokens"])

    await one(prompts[0])                       # wake the server / warm up before timing
    lat.clear(); pt.clear(); ot.clear()
    t0 = time.time()
    await asyncio.gather(*[one(p) for p in prompts])
    dt = time.time() - t0
    print(f"{len(prompts)} requests, concurrency {a.conc}, {dt:.0f}s | prompt {sum(pt):,} tok ({np.mean(pt):.0f}/req) -> {sum(pt) / dt:,.0f} tok/s | "
          f"output {sum(ot):,} tok ({np.mean(ot):.0f}/req) -> {sum(ot) / dt:,.0f} tok/s | total {(sum(pt) + sum(ot)) / dt:,.0f} tok/s | "
          f"latency p50 {np.median(lat):.1f}s p95 {np.quantile(lat, .95):.1f}s")


asyncio.run(main())
