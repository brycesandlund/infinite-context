"""Merge a LoRA adapter into the base checkpoint -> a plain Hugging Face model directory that vLLM (or anything) serves.

    python -m offtinker.merge --adapter <adapter dir> --out <model dir> [--model Qwen/Qwen3.5-9B]

Patches the checkpoint's safetensors shards directly (W += scale * B @ A, computed in float32, stored back in the
shard's dtype), so it does not depend on a model class and keeps every other tensor (vision tower included) and the
tokenizer / config files byte-identical. Works for adapters trained here and adapters downloaded from Tinker
(same key names). CPU is enough (~40 GB RAM for the 9B); a GPU just makes the matmuls faster.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from offtinker.lora import PREFIX, adapter_scale, hf_weight_for, read_adapter


def stochastic_bf16(wf: torch.Tensor, gen: torch.Generator, rows: int = 8192) -> torch.Tensor:
    """float32 -> bf16 with UNBIASED rounding: each value goes to one of its two bf16 neighbours with probability
    proportional to proximity, so E[result] = value.

    Why: a LoRA update trained at lr 1e-5 is mostly SMALLER than half a bf16 step of the weight it is added to
    (step ~1.2e-4 at |w| ~0.02; update entries ~1e-5..1e-4). Round-to-nearest then returns the original weight for
    most entries — the update is silently shrunk (measured: merged NLL 0.081 vs 0.060 unmerged on a 150-step
    adapter). Unbiased rounding keeps the update in expectation and turns the error into zero-mean noise."""
    out = torch.empty(wf.shape, dtype=torch.bfloat16, device=wf.device)
    for a in range(0, wf.shape[0], rows):
        x = wf[a:a + rows]
        n = x.to(torch.bfloat16)                       # nearest neighbour
        nf = n.float()
        bits = n.view(torch.int16).to(torch.int32)
        # the OTHER neighbour: one step from n towards x. In the int16 bit pattern +1 grows the magnitude.
        toward_up = x > nf
        grow = torch.where(toward_up, nf > 0, nf < 0)
        step = torch.where(x == nf, 0, torch.where(grow, 1, -1))
        step = torch.where(nf == 0, 0, step)           # exact zeros: keep nearest
        other = (bits + step).to(torch.int16).view(torch.bfloat16)
        gap = other.float() - nf
        p = torch.where(gap != 0, (x - nf) / gap, torch.zeros_like(x)).clamp_(0, 1)
        take = torch.rand(x.shape, generator=gen, device=x.device) < p
        out[a:a + rows] = torch.where(take, other, n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="hub id or local directory of the BASE checkpoint")
    ap.add_argument("--rounding", default="stochastic", choices=["stochastic", "nearest", "fp32"],
                    help="how merged weights go back to the checkpoint dtype: stochastic = unbiased bf16 (default); "
                         "nearest = plain cast (LOSES small updates); fp32 = store merged matrices in float32 "
                         "(exact, but the server must then run with --dtype float32)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(a.model) if Path(a.model).is_dir() else Path(snapshot_download(a.model))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = read_adapter(a.adapter)
    scale = adapter_scale(a.adapter)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    # weight name -> list of (row-slice name or None, A, B)
    patches: dict[str, list] = defaultdict(list)
    for k in sd:
        if not k.endswith(".lora_A.weight"):
            continue
        key = k[len(PREFIX):-len(".lora_A.weight")]
        wname, part = hf_weight_for(key)
        patches[wname].append((part, sd[k], sd[k.replace(".lora_A.", ".lora_B.")]))
    n_done = 0
    for f in sorted(base.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy2(f, out / f.name)
            continue
        tensors = {}
        with safe_open(str(f), "pt") as sf:
            meta = sf.metadata()
            for name in sf.keys():
                w = sf.get_tensor(name)
                if name in patches:
                    wf = w.to(dev, torch.float32)
                    row = 0
                    parts = patches[name]
                    if parts[0][0] is not None:                     # fused in_proj_qkv: rows are [q | k | v]
                        parts = sorted(parts, key=lambda p: "qkv".index(p[0]))
                    for _, A, B in parts:
                        d = scale * (B.to(dev, torch.float32) @ A.to(dev, torch.float32))
                        if d.shape == wf.shape:
                            wf += d
                        else:
                            wf[row:row + d.shape[0]] += d
                            row += d.shape[0]
                    assert parts[0][0] is None or row == wf.shape[0], (name, row, wf.shape)
                    if a.rounding == "fp32":
                        w = wf.cpu()
                    elif a.rounding == "stochastic" and w.dtype == torch.bfloat16:
                        w = stochastic_bf16(wf, gen).cpu()
                    else:
                        w = wf.to(w.dtype).cpu()
                    n_done += len(parts)
                tensors[name] = w
        save_file(tensors, str(out / f.name), metadata=meta)
        print(f"  wrote {f.name} ({len(tensors)} tensors)", flush=True)
    n_adapters = sum(k.endswith(".lora_A.weight") for k in sd)
    assert n_done == n_adapters, f"merged {n_done} of {n_adapters} adapters — some target weights were not found"
    print(f"merged {n_done} adapters (scale {scale:g}, rounding {a.rounding}) from {a.adapter} into {out}")


if __name__ == "__main__":
    main()
