"""Throughput benchmark on real batches:  python -m offtinker.bench --data <export dir> [--configs 1:16000,0:16000,0:32000]

For each config (grad_ckpt:micro_tokens) runs --steps optimizer steps (after a warm-up that compiles the Triton
kernels) and prints tokens/s and peak memory; then profiles two steps of the fastest config (top ops by GPU time).
Weights are trained during the benchmark and thrown away."""

from __future__ import annotations

import argparse
import time

import torch

from offtinker.core import Dataset, load_model, loss_backward, pack
from offtinker.lora import LoRASet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--configs", default="1:16000,0:16000,0:32000,1:32000")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--start", type=int, default=100, help="first batch index used")
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--trace", type=int, default=0, help="print every micro-batch's shape and time")
    ap.add_argument("--pad-mult", type=int, default=1)
    ap.add_argument("--cudnn-sdp", type=int, default=1, help="0 = use torch's own flash-attention kernel instead of cuDNN's")
    a = ap.parse_args()
    from offtinker import core
    core.PAD_MULT = a.pad_mult
    torch.backends.cuda.enable_cudnn_sdp(bool(a.cudnn_sdp))
    seen = set()

    torch.backends.cuda.matmul.allow_tf32 = True
    ds = Dataset(a.data)
    text, owner = load_model(a.model)
    lora = LoRASet(text, owner)
    opt = torch.optim.Adam(lora.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-12)
    B = ds.batch_size
    cursor = [a.start]

    def step(micro: int) -> tuple[int, float]:
        data = [ds.datum(u) for u in ds.batch(cursor[0], B)]
        cursor[0] += 1
        torch.cuda.synchronize()
        t = time.time()
        opt.zero_grad(set_to_none=True)
        for grp in pack([len(s) for s, _ in data], micro):
            tm = time.time()
            loss_backward(text, owner, [data[i][0] for i in grp], [data[i][1] for i in grp])
            if a.trace:
                torch.cuda.synchronize()
                L = -(-(len(data[grp[0]][0]) - 1) // a.pad_mult) * a.pad_mult
                shp = (len(grp), L)
                print(f"    micro-batch {shp} {'NEW ' if shp not in seen else 'seen'} {time.time() - tm:.2f}s "
                      f"({len(grp) * L / (time.time() - tm):,.0f} padded tok/s)", flush=True)
                seen.add(shp)
        opt.step()
        torch.cuda.synchronize()
        return sum(len(s) - 1 for s, _ in data), time.time() - t

    def mode(ckpt: int) -> None:
        if ckpt:
            text.gradient_checkpointing_enable(); text.train()
        else:
            text.gradient_checkpointing_disable(); text.eval()

    mode(1)
    for i in range(a.warmup):
        n, dt = step(16000)
        print(f"warmup {i + 1}: {n / dt:,.0f} tok/s ({dt:.1f}s)", flush=True)
    results = []
    for cfg in a.configs.split(","):
        ck, micro = (int(x) for x in cfg.split(":"))
        mode(ck)
        try:
            step(micro)                                  # settle (new shapes / allocator)
            torch.cuda.reset_peak_memory_stats()
            tot_n = tot_t = 0.0
            times = []
            for _ in range(a.steps):
                n, dt = step(micro)
                tot_n += n; tot_t += dt; times.append(n / dt)
            results.append((tot_n / tot_t, ck, micro))
            print(f"grad_ckpt={ck} micro_tokens={micro}: {tot_n / tot_t:,.0f} tok/s (per-step median {sorted(times)[len(times) // 2]:,.0f} min {min(times):,.0f} max {max(times):,.0f}) | "
                  f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        except torch.OutOfMemoryError:
            print(f"grad_ckpt={ck} micro_tokens={micro}: OUT OF MEMORY", flush=True)
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    if a.profile and results:
        _, ck, micro = max(results)
        mode(ck)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step(micro); step(micro)
        print(f"\nprofile of 2 steps at grad_ckpt={ck} micro_tokens={micro}:")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22, max_name_column_width=60))


if __name__ == "__main__":
    main()
