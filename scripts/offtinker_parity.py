"""Tinker side of the off-Tinker parity checks, and the comparisons. All read an sft.py export (SFT_EXPORT=<dir>).

  tinker-logprobs  per-token log-probs of chosen datums from Tinker (base model, or --ckpt tinker://.../sampler_weights/x)
  tinker-ref       a SHORT reference training run on Tinker over the export's first --steps batches (same datums, same
                   order, same Adam as sft.py), logging per-step NLL and saving + downloading the adapter at --snap steps
                   (0 = the fresh init). The off-Tinker run starts from that init and is compared step by step.
  cmp-logprobs     compare two log-prob files (all positions, and trained positions only)
  cmp-curves       compare two per-step NLL logs (Tinker ref vs offtinker metrics.jsonl)
  cmp-adapters     compare two adapters' weight updates relative to a common init

    PYTHONPATH=. uv run python scripts/offtinker_parity.py tinker-ref --data <dir> --steps 150 --snap 0,20,150 --out <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from offtinker.core import Dataset  # noqa: E402  (numpy only at import time for Dataset)


def _datum(ds: Dataset, u: int):
    import tinker
    seq, mask = ds.datum(u)
    w = mask[1:].astype(np.float32)
    w = w / w.sum()
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([int(t) for t in seq[:-1]]),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=w.tolist(), dtype="float32", shape=[len(w)]),
            "target_tokens": tinker.TensorData(data=[int(t) for t in seq[1:]], dtype="int64", shape=[len(w)]),
        },
    )


async def tinker_logprobs(a) -> None:
    import tinker
    ds = Dataset(a.data)
    sc = tinker.ServiceClient()
    client = (await sc.create_sampling_client_async(model_path=a.ckpt) if a.ckpt
              else await sc.create_sampling_client_async(base_model=a.model))
    ids = [int(x) for x in a.ids.split(",") if x.strip()]

    async def one(u):
        seq, _ = ds.datum(u)
        lp = await client.compute_logprobs_async(tinker.ModelInput.from_ints([int(t) for t in seq]))
        return [float(v) for v in lp[1:]]

    out = dict(zip(map(str, ids), await asyncio.gather(*[one(u) for u in ids])))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print(f"wrote Tinker log-probs for {len(out)} datums ({sum(len(v) for v in out.values()):,} tokens) -> {a.out}")


async def tinker_ref(a) -> None:
    import tinker
    ds = Dataset(a.data)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = a.batch_size or ds.batch_size
    snaps = sorted({int(x) for x in a.snap.split(",") if x.strip()})
    lr = a.lr if a.lr is not None else float(ds.meta["lr"])
    n_tok = sum(ds.length(u) - 1 for s in range(a.steps) for u in ds.batch(s, B))
    print(f"Tinker reference run: {a.model} rank {a.rank} | {a.steps} steps x {B} | {n_tok:,} training tokens | lr {lr:g}")
    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(base_model=a.model, rank=a.rank)
    adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95)
    paths = {}

    async def snap(step: int) -> None:
        r = await (await tc.save_weights_for_sampler_async(f"{a.name}_s{step}", ttl_seconds=a.ttl_hours * 3600)).result_async()
        paths[step] = r.path
        (out / "paths.json").write_text(json.dumps(paths, indent=1))
        print(f"  [snap] step {step}: {r.path}", flush=True)

    if 0 in snaps:
        await snap(0)
    with open(out / "metrics.jsonl", "w") as mf:
        for s in range(a.steps):
            ids = ds.batch(s, B)
            batch = [_datum(ds, u) for u in ids]
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            op = await tc.optim_step_async(adam)
            res = await fb.result_async()
            await op.result_async()
            per = [-float(np.dot(np.asarray(o["logprobs"].data, dtype=np.float64), np.asarray(d.loss_fn_inputs["weights"].data)))
                   for o, d in zip(res.loss_fn_outputs, batch)]
            nll = sum(per) / len(per)
            mf.write(json.dumps({"step": s + 1, "nll": nll, "datum_nll": [round(v, 5) for v in per], "ids": ids}) + "\n")
            mf.flush()
            if (s + 1) % 10 == 0 or s == 0:
                print(f"  step {s + 1}/{a.steps} | nll {nll:.4f}", flush=True)
            if (s + 1) in snaps:
                await snap(s + 1)
    if a.download:
        for step, p in paths.items():
            dst = out / f"adapter_s{step}"
            if not dst.exists():
                subprocess.run(["tinker", "checkpoint", "download", p, "--output", str(out / "dl"), "--force"], check=True)
                (out / "dl" / p.replace("tinker://", "").replace("/", "_")).rename(dst)
            print(f"  downloaded step {step} -> {dst}")
    print("DONE")


async def sample_cmp(a) -> None:
    """Greedy-decode the first assistant turn of N datums on Tinker (a sampler checkpoint or the base model) and on
    our vLLM server; report how often the two continuations are token-identical."""
    import httpx
    import tinker
    ds = Dataset(a.data)
    ids = [int(x) for x in a.ids.split(",") if x.strip()]
    sc = tinker.ServiceClient()
    client = (await sc.create_sampling_client_async(model_path=a.ckpt) if a.ckpt
              else await sc.create_sampling_client_async(base_model=a.model))
    key = Path("~/.cache/infinite-context/vllm_api_key.txt").expanduser().read_text().strip()
    http = httpx.AsyncClient(base_url=a.url.rstrip("/"), headers={"Authorization": f"Bearer {key}"}, timeout=900)
    STOP = 248046   # <|im_end|>

    async def one(u):
        seq, mask = ds.datum(u)
        cut = int(np.nonzero(mask)[0][0])                      # first trained token = start of the first assistant turn
        prompt = [int(t) for t in seq[:cut]]
        t = await client.sample_async(tinker.ModelInput.from_ints(prompt), num_samples=1,
                                      sampling_params=tinker.SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=[STOP]))
        tk = [int(x) for x in t.sequences[0].tokens if int(x) != STOP]
        for attempt in range(60):
            r = await http.post("/v1/completions", json={
                "model": "ic", "prompt": prompt, "temperature": 0.0, "top_p": 1.0, "max_tokens": a.max_tokens,
                "stop_token_ids": [STOP], "skip_special_tokens": False, "add_special_tokens": False, "return_token_ids": True,
                "logprobs": 5, "return_tokens_as_token_ids": True})
            if r.status_code == 200:
                break
            await asyncio.sleep(10)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        vk = [int(x) for x in ch["token_ids"] if int(x) != STOP]
        gold = [int(x) for x in seq[cut:cut + a.max_tokens]]
        common = next((i for i, (x, y) in enumerate(zip(tk, vk)) if x != y), min(len(tk), len(vk)))
        # At the first differing token: how close was the call under vLLM's own distribution?
        gap = None
        if tk != vk and common < len(vk) and common < len(tk):
            top = {int(k.split(":")[1]): v for k, v in ch["logprobs"]["top_logprobs"][common].items()}
            gap = (top[vk[common]] - top[tk[common]]) if tk[common] in top else float("inf")
        return {"u": u, "same": tk == vk, "common": common, "len_t": len(tk), "len_v": len(vk), "gap": gap,
                "t_gold": tk[:len(gold)] == gold[:len(tk)], "v_gold": vk[:len(gold)] == gold[:len(vk)]}

    res = await asyncio.gather(*[one(u) for u in ids])
    gaps = [r["gap"] for r in res if r["gap"] is not None]
    print(f"{len(res)} prompts, greedy, up to {a.max_tokens} tokens | identical continuations {sum(r['same'] for r in res)}/{len(res)} | "
          f"mean common prefix {np.mean([r['common'] for r in res]):.1f} of mean length {np.mean([r['len_t'] for r in res]):.1f} | "
          f"follows the training trace: Tinker {sum(r['t_gold'] for r in res)}, vLLM {sum(r['v_gold'] for r in res)}")
    if gaps:
        g = np.array([x for x in gaps if x != float("inf")])
        print(f"at the first differing token, vLLM's log-prob margin for its own token over Tinker's: median {np.median(g):.3f}, "
              f"max {g.max():.3f} nats; margins under 0.1: {int((g < 0.1).sum())}/{len(gaps)}, under 0.5: {int((g < 0.5).sum())}/{len(gaps)}; "
              f"Tinker's token outside vLLM's top-5: {sum(x == float('inf') for x in gaps)}")


def cmp_logprobs(a) -> None:
    ds = Dataset(a.data)
    x, y = json.load(open(a.a)), json.load(open(a.b))
    alld, trd, na, nb = [], [], [], []
    for u in sorted(set(x) & set(y), key=int):
        _, mask = ds.datum(int(u))
        p, q, m = np.asarray(x[u]), np.asarray(y[u]), mask[1:].astype(bool)
        assert len(p) == len(q) == len(m), (u, len(p), len(q), len(m))
        alld.append(p - q); trd.append((p - q)[m]); na.append(-p[m].mean()); nb.append(-q[m].mean())
    d, t = np.concatenate(alld), np.concatenate(trd)
    print(f"{len(alld)} datums | all positions n={len(d):,}: mean |d| {np.abs(d).mean():.4f}, median {np.median(np.abs(d)):.4f}, "
          f"p99 {np.quantile(np.abs(d), .99):.3f}, max {np.abs(d).max():.3f}, mean signed {d.mean():+.4f}")
    print(f"           trained positions n={len(t):,}: mean |d| {np.abs(t).mean():.4f}, p99 {np.quantile(np.abs(t), .99):.3f}, "
          f"max {np.abs(t).max():.3f} | mean per-datum NLL  A {np.mean(na):.4f}  B {np.mean(nb):.4f}  "
          f"(per-datum |dNLL| mean {np.mean(np.abs(np.array(na) - np.array(nb))):.4f}, max {np.max(np.abs(np.array(na) - np.array(nb))):.4f})")


def cmp_curves(a) -> None:
    A = {r["step"]: r for r in map(json.loads, open(a.a))}
    Bm = {r["step"]: r for r in map(json.loads, open(a.b))}
    steps = sorted(set(A) & set(Bm))
    x, y = np.array([A[s]["nll"] for s in steps]), np.array([Bm[s]["nll"] for s in steps])
    print(f"{len(steps)} common steps | corr {np.corrcoef(x, y)[0, 1]:.4f} | mean |dNLL| {np.abs(x - y).mean():.4f} | "
          f"mean A {x.mean():.4f}  B {y.mean():.4f}")
    w = max(1, len(steps) // 10)
    for i in range(0, len(steps), w):
        print(f"  steps {steps[i]:>4}-{steps[min(i + w, len(steps)) - 1]:<4}  A {x[i:i + w].mean():.4f}   B {y[i:i + w].mean():.4f}   "
              f"ratio {y[i:i + w].mean() / max(x[i:i + w].mean(), 1e-9):.3f}")


def cmp_adapters(a) -> None:
    from safetensors.torch import load_file
    f = lambda p: load_file(str(Path(p) / "adapter_model.safetensors"))
    init, A, Bm = f(a.init), f(a.a), f(a.b)
    rows = []
    for k in sorted(init):
        if k not in A or k not in Bm:
            continue
        da, db = (A[k].double() - init[k].double()).flatten().numpy(), (Bm[k].double() - init[k].double()).flatten().numpy()
        cos = float(np.dot(da, db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-30))
        rows.append((k, cos, float(np.linalg.norm(da)), float(np.linalg.norm(db)), float(np.mean(np.sign(da) == np.sign(db)))))
    for kind in ("lora_A", "lora_B"):
        r = [x for x in rows if kind in x[0]]
        if r:
            print(f"{kind}: {len(r)} tensors | cosine(update A, update B) mean {np.mean([x[1] for x in r]):.4f} min {np.min([x[1] for x in r]):.4f} | "
                  f"norm ratio B/A mean {np.mean([x[3] / max(x[2], 1e-30) for x in r]):.4f} | sign agreement {np.mean([x[4] for x in r]):.3f}")
    worst = sorted(rows, key=lambda x: x[1])[:5]
    for k, cos, n1, n2, sg in worst:
        print(f"   lowest cosine: {k.replace('base_model.model.model.', '')}: cos {cos:.3f} |dA| {n1:.4g} |dB| {n2:.4g}")
    missing = sorted(set(A) ^ set(Bm))
    if missing:
        print(f"   keys in only one adapter: {len(missing)}, e.g. {missing[:6]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tinker-logprobs")
    p.add_argument("--data", required=True); p.add_argument("--ids", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B"); p.add_argument("--ckpt", default="")
    p = sub.add_parser("tinker-ref")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=150); p.add_argument("--snap", default="0,20,150")
    p.add_argument("--model", default="Qwen/Qwen3.5-9B"); p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=None); p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--name", default="offtinker_ref"); p.add_argument("--ttl-hours", type=int, default=72)
    p.add_argument("--download", type=int, default=1)
    p = sub.add_parser("sample-cmp")
    p.add_argument("--data", required=True); p.add_argument("--ids", required=True); p.add_argument("--url", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.5-9B"); p.add_argument("--ckpt", default=""); p.add_argument("--max-tokens", type=int, default=96)
    p = sub.add_parser("cmp-logprobs")
    p.add_argument("--data", required=True); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("cmp-curves"); p.add_argument("a"); p.add_argument("b")
    p = sub.add_parser("cmp-adapters"); p.add_argument("--init", required=True); p.add_argument("a"); p.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "tinker-logprobs":
        asyncio.run(tinker_logprobs(a))
    elif a.cmd == "tinker-ref":
        asyncio.run(tinker_ref(a))
    elif a.cmd == "sample-cmp":
        asyncio.run(sample_cmp(a))
    else:
        {"cmp-logprobs": cmp_logprobs, "cmp-curves": cmp_curves, "cmp-adapters": cmp_adapters}[a.cmd](a)


if __name__ == "__main__":
    main()
