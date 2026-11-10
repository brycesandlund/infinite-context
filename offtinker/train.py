"""LoRA SFT on one GPU, reproducing the Tinker loop in sft.py on an exported dataset.

    python -m offtinker.train --data <export dir> --out <run dir> [--model Qwen/Qwen3.5-9B] [--max-steps N]

Same optimization problem as sft.py on Tinker:
  * batch b = the datums order[b*B:(b+1)*B] of the export (Tinker's Random(0) shuffle), B = 16 by default;
  * loss = SUM over the batch of each datum's MEAN NLL over its assistant tokens;
  * Adam(lr 1e-5 constant, betas 0.9/0.95, eps 1e-12, no weight decay, no clipping), one step per batch, 1 epoch;
  * LoRA rank 32 / alpha 32 on the modules Tinker adapts (offtinker/lora.py).
A batch is run as several length-sorted micro-batches (gradients accumulate, one optimizer step), so nothing is
padded to the longest conversation; that changes memory use, not the math.

Checkpoints (adapter + optimizer + step) go to <out>/ckpt every --save-every steps; --resume continues from the
last one, so a preempted or crashed job loses at most that many steps. The final adapter is <out>/final.
Per-step metrics: <out>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from offtinker.core import Dataset, load_model, loss_backward, pack
from offtinker.lora import LoRASet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--lr", type=float, default=None, help="default: the export's (sft.py LR)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: the export's (SFT_BATCH_SIZE)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = the full epoch)")
    ap.add_argument("--micro-tokens", type=int, default=32000,
                    help="max padded tokens per forward/backward (32000 peaks near 40 GB with --grad-ckpt 1)")
    ap.add_argument("--grad-ckpt", type=int, default=1, help="gradient checkpointing (less memory, ~30%% more compute)")
    ap.add_argument("--head-chunk", type=int, default=1024)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--init-adapter", default="", help="start from this adapter's weights (fresh optimizer)")
    ap.add_argument("--snapshot-steps", default="", help="comma list: also keep the adapter after these step counts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset(a.data)
    B = a.batch_size or ds.batch_size
    lr = a.lr if a.lr is not None else float(ds.meta["lr"])
    n_batches = ds.n_batches(B)
    last = min(n_batches, a.max_steps) if a.max_steps else n_batches
    snaps = {int(x) for x in a.snapshot_steps.split(",") if x.strip()}

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    text, owner = load_model(a.model)
    lora = LoRASet(text, owner, r=a.rank, alpha=a.alpha, seed=a.seed)
    params = lora.parameters()
    if a.grad_ckpt:
        text.gradient_checkpointing_enable()
        text.train()                      # transformers only checkpoints in train mode (all dropouts are 0)
    opt = torch.optim.Adam(params, lr=lr, betas=(a.beta1, a.beta2), eps=a.eps, weight_decay=0.0)

    step = 0
    ck = out / "ckpt"
    if a.resume and (ck / "state.json").exists():
        st = json.loads((ck / "state.json").read_text())
        assert st["n_batches"] == n_batches and st["batch_size"] == B, f"checkpoint is for another config: {st}"
        lora.load(ck)
        opt.load_state_dict(torch.load(ck / "opt.pt", map_location="cuda"))
        step = int(st["step"])
        print(f"RESUMED from {ck} at step {step}", flush=True)
    elif a.init_adapter:
        lora.load(a.init_adapter)
        print(f"INIT from adapter {a.init_adapter} (fresh optimizer)", flush=True)
    print(f"model {a.model} loaded in {time.time() - t0:.0f}s | LoRA r={a.rank} alpha={a.alpha:g}: "
          f"{lora.n_params() / 1e6:.1f}M params in {len(lora.index)} adapters | data {a.data}: {len(ds):,} datums, "
          f"{ds.meta['train_tokens']:,} tokens | batch {B} x {n_batches} steps (running to {last}) | lr {lr:g} | "
          f"micro_tokens {a.micro_tokens} grad_ckpt {a.grad_ckpt} | gpu {torch.cuda.get_device_name(0)}", flush=True)

    def save(dst: Path, with_opt: bool) -> None:
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        lora.save(tmp, extra={"base_model_name_or_path": a.model, "step": step})
        if with_opt:
            torch.save(opt.state_dict(), tmp / "opt.pt")
            (tmp / "state.json").write_text(json.dumps({"step": step, "n_batches": n_batches, "batch_size": B, "lr": lr}))
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(tmp, dst)

    mf = open(out / "metrics.jsonl", "a")
    run_nll = run_n = 0.0
    tok_done = 0
    t_start = t_log = time.time()
    tok_log = 0
    torch.cuda.reset_peak_memory_stats()
    while step < last:
        ids = ds.batch(step, B)
        data = [ds.datum(u) for u in ids]
        ts = time.time()
        opt.zero_grad(set_to_none=True)
        nlls: list[float] = [0.0] * len(ids)
        for grp in pack([len(s) for s, _ in data], a.micro_tokens):
            per = loss_backward(text, owner, [data[i][0] for i in grp], [data[i][1] for i in grp], a.head_chunk)
            for i, v in zip(grp, per):
                nlls[i] = v
        opt.step()
        step += 1
        ntok = sum(len(s) - 1 for s, _ in data)
        nll = sum(nlls) / len(nlls)
        if not (nll == nll):
            raise SystemExit(f"NaN loss at step {step}")
        tok_done += ntok
        tok_log += ntok
        run_nll += nll
        run_n += 1
        mf.write(json.dumps({"step": step, "nll": nll, "tokens": ntok, "sec": round(time.time() - ts, 3),
                             "datum_nll": [round(v, 5) for v in nlls], "ids": ids}) + "\n")
        mf.flush()
        if step % a.log_every == 0 or step == last:
            now = time.time()
            rate = tok_log / max(now - t_log, 1e-9)
            eta = (last - step) * (ds.meta["train_tokens"] / n_batches) / max(rate, 1)
            print(f"step {step}/{last} | nll {nll:.4f} | running mean {run_nll / run_n:.4f} | {rate:,.0f} tok/s | "
                  f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB | elapsed {(now - t_start) / 60:.1f} min"
                  f" | eta {eta / 60:.0f} min", flush=True)
            t_log, tok_log, run_nll, run_n = now, 0, 0.0, 0.0
        if step in snaps:
            save(out / f"step{step}", with_opt=False)
        if a.save_every and step % a.save_every == 0 and step < last:
            save(ck, with_opt=True)
            print(f"  [ckpt] step {step} -> {ck}", flush=True)
    save(out / "final", with_opt=False)
    save(ck, with_opt=True)
    print(f"DONE {step} steps in {(time.time() - t_start) / 60:.1f} min -> {out / 'final'}", flush=True)


if __name__ == "__main__":
    main()
