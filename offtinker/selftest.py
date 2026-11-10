"""CPU self-test on a tiny random Qwen3.5 (no downloads, no GPU):  python -m offtinker.selftest

1. LoRA wrapping leaves the model unchanged while B = 0, and adapts exactly the modules Tinker adapts.
2. loss_backward (micro-batched, head run in chunks at trained positions only) gives the same loss and the same
   gradients as the textbook computation (full logits, sum over the batch of per-datum mean NLL).
3. adapter save -> load round-trips.
4. merge.py on a saved checkpoint reproduces the unmerged LoRA model's log-probs.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from offtinker import core
from offtinker.core import loss_backward, pack, token_logprobs
from offtinker.lora import LoRASet


def tiny():
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    cfg = Qwen3_5Config(
        text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                         linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                         layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
    )
    torch.manual_seed(0)
    m = Qwen3_5ForConditionalGeneration(cfg).float()
    with torch.no_grad():
        for p in m.parameters():        # non-degenerate norms / gates
            if p.dim() == 1:
                p.add_(0.1 * torch.randn_like(p))
    return m


def main() -> None:
    assert core.DEV == "cpu" or True
    m = tiny().to(core.DEV)
    m.requires_grad_(False)
    text = m.model.language_model
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 300, size=n).astype(np.int32) for n in (37, 12, 25, 9, 31)]
    masks = []
    for s in seqs:
        mk = (rng.random(len(s)) < 0.4).astype(np.uint8)
        mk[0] = 0
        mk[-1] = 1
        masks.append(mk)
    base_lp = [token_logprobs(text, m, s) for s in seqs]

    lora = LoRASet(text, m, r=4, alpha=4)
    names = sorted(lora.index)
    assert sum("in_proj_q" in n for n in names) == 3 and sum("in_proj_k" in n for n in names) == 3
    assert sum("self_attn" in n for n in names) == 4 and "unembed_tokens" in names
    assert not any("in_proj_a" in n or "in_proj_b" in n or "embed_tokens" == n for n in names)
    for s, b in zip(seqs, base_lp):
        assert np.allclose(token_logprobs(text, m, s), b, atol=1e-5), "B=0 LoRA changed the model"
    print(f"1 ok: {len(names)} adapters, B=0 is a no-op")

    with torch.no_grad():
        for p in lora.parameters():
            if p.abs().sum() == 0:
                p.copy_(0.05 * torch.randn_like(p))
    params = lora.parameters()

    # textbook: full logits, sum_i mean NLL_i
    for p in params:
        p.grad = None
    total = 0.0
    ref_per = []
    for s, mk in zip(seqs, masks):
        h = text(input_ids=torch.from_numpy(s[:-1].astype(np.int64))[None].to(core.DEV)).last_hidden_state[0]
        nll = F.cross_entropy(m.lm_head(h).float(), torch.from_numpy(s[1:].astype(np.int64)).to(core.DEV), reduction="none")
        w = torch.from_numpy(mk[1:].astype(np.float32)).to(core.DEV)
        li = (nll * w).sum() / w.sum()
        ref_per.append(float(li))
        total = total + li
    total.backward()
    ref = [p.grad.clone() for p in params]

    for micro, chunk in ((10_000, 1024), (40, 7), (1, 3)):
        for p in params:
            p.grad = None
        per = [0.0] * len(seqs)
        groups = pack([len(s) for s in seqs], micro)
        for g in groups:
            out = loss_backward(text, m, [seqs[i] for i in g], [masks[i] for i in g], head_chunk=chunk)
            for i, v in zip(g, out):
                per[i] = v
        assert np.allclose(per, ref_per, rtol=1e-4, atol=1e-5), (per, ref_per)
        worst = max(float((p.grad - r).abs().max() / (r.abs().max() + 1e-12)) for p, r in zip(params, ref))
        assert worst < 2e-3, f"gradient mismatch {worst}"
        print(f"2 ok: micro_tokens={micro} ({len(groups)} micro-batches) head_chunk={chunk}: loss + grads match "
              f"(worst rel err {worst:.1e})")

    with tempfile.TemporaryDirectory() as td:
        lora.save(f"{td}/adapter")
        lp_lora = [token_logprobs(text, m, s) for s in seqs]
        m2 = tiny().to(core.DEV)
        m2.requires_grad_(False)
        l2 = LoRASet(m2.model.language_model, m2, r=4, alpha=4, seed=123)
        l2.load(f"{td}/adapter")
        for s, b in zip(seqs, lp_lora):
            assert np.allclose(token_logprobs(m2.model.language_model, m2, s), b, atol=1e-5)
        print("3 ok: adapter save/load round-trip")

        tiny().save_pretrained(f"{td}/base")
        subprocess.run([sys.executable, "-m", "offtinker.merge", "--adapter", f"{td}/adapter", "--out", f"{td}/merged",
                        "--model", f"{td}/base"], check=True, capture_output=True)
        from transformers import Qwen3_5ForConditionalGeneration
        m3 = Qwen3_5ForConditionalGeneration.from_pretrained(f"{td}/merged").float().to(core.DEV)
        m3.requires_grad_(False)
        d = max(float(np.abs(token_logprobs(m3.model.language_model, m3, s) - b).max()) for s, b in zip(seqs, lp_lora))
        d0 = max(float(np.abs(a - b).max()) for a, b in zip(base_lp, lp_lora))
        assert d < 1e-4 < d0, (d, d0)
        print(f"4 ok: merged checkpoint == unmerged LoRA (max |dlogprob| {d:.1e}; LoRA vs base {d0:.2f})")
    print("ALL OK")


if __name__ == "__main__":
    main()
