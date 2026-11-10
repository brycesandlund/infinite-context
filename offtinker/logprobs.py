"""Per-token log-probs of exported datums under the Hugging Face model (+ optional adapter) — the off-Tinker side of
the parity checks (scripts/offtinker_parity.py holds the Tinker side and the comparison).

    python -m offtinker.logprobs --data <export dir> --ids 0,5,9 --out lp.json [--adapter <dir>] [--merged <model dir>]

Writes {"<unique datum id>": [logprob of token 1, token 2, ...]}.
"""

from __future__ import annotations

import argparse
import json

from offtinker.core import Dataset, load_model, token_logprobs
from offtinker.lora import LoRASet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ids", required=True, help="comma list of unique-datum ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="base model, or a merged model directory")
    ap.add_argument("--adapter", default="", help="adapter directory to apply (unmerged) on top of --model")
    ap.add_argument("--head-fp32", type=int, default=1)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"], help="fp32 = high-precision reference")
    a = ap.parse_args()

    ds = Dataset(a.data)
    text, owner = load_model(a.model, head_fp32=bool(a.head_fp32), dtype=a.dtype)
    if a.adapter:
        lora = LoRASet(text, owner)
        lora.load(a.adapter)
        print(f"applied adapter {a.adapter} ({lora.n_params() / 1e6:.1f}M params)")
    out = {}
    for u in [int(x) for x in a.ids.split(",") if x.strip()]:
        seq, _ = ds.datum(u)
        out[str(u)] = [round(float(v), 5) for v in token_logprobs(text, owner, seq)]
    json.dump(out, open(a.out, "w"))
    print(f"wrote log-probs for {len(out)} datums -> {a.out}")


if __name__ == "__main__":
    main()
