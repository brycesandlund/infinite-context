"""Shared pieces of the off-Tinker stack: dataset reader, model loader, and the loss / log-prob forward passes.

Runs anywhere with one CUDA GPU (Modal, Vast, a workstation). Needs torch, transformers>=5.5 (Qwen3.5),
flash-linear-attention (the Gated DeltaNet kernels; without it transformers falls back to a slow torch loop),
safetensors, numpy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"      # cpu only for offtinker/selftest.py (tiny random model)


class Dataset:
    """The export written by sft.py (SFT_EXPORT=<dir>): unique datums + Tinker's training order. See sft._export_dataset."""

    def __init__(self, data_dir: str | Path):
        d = Path(data_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self.mask = np.load(d / "mask.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")
        self.order = np.load(d / "order.npy")
        self.task_ids = np.load(d / "task_ids.npy") if (d / "task_ids.npy").exists() else None
        self.batch_size = int(self.meta["batch_size"])

    def __len__(self) -> int:
        return len(self.order)

    def n_batches(self, batch_size: int | None = None) -> int:
        b = batch_size or self.batch_size
        return (len(self.order) + b - 1) // b

    def datum(self, u: int) -> tuple[np.ndarray, np.ndarray]:
        """(tokens, mask) of unique datum u. Model input = tokens[:-1], targets = tokens[1:], trained where mask[1:] == 1."""
        a, b = int(self.offsets[u]), int(self.offsets[u + 1])
        return np.asarray(self.tokens[a:b]), np.asarray(self.mask[a:b])

    def batch(self, step: int, batch_size: int | None = None) -> list[int]:
        b = batch_size or self.batch_size
        return [int(u) for u in self.order[step * b:(step + 1) * b]]

    def length(self, u: int) -> int:
        return int(self.offsets[u + 1] - self.offsets[u])


def load_model(model_name: str, head_fp32: bool = True, dtype: str = "bf16"):
    """-> (text_model, head_owner). The checkpoint is the vision-language model; we keep the text backbone + lm_head.
    Weights stay bf16 (the checkpoint dtype). With head_fp32 the output layer is held in float32 (≈4 GB for a 248K
    vocab) so losses / log-probs are not limited by bf16 logits."""
    from transformers import AutoModelForImageTextToText

    # dtype="fp32" is a high-precision reference for numerical checks (twice the memory, several times slower).
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, dtype=torch.float32 if dtype == "fp32" else torch.bfloat16, device_map={"": "cuda"})
    model.requires_grad_(False)
    text = model.model.language_model
    model.model.visual = None          # not used for text; frees ~1 GB
    if head_fp32:
        model.lm_head.float()
    model.eval()
    return text, model


def pack(lengths: list[int], micro_tokens: int, overhead_tokens: int = 1300) -> list[list[int]]:
    """Split positions 0..n-1 into micro-batches (each = a run of the length-sorted order, padded to its longest
    sequence) minimising  sum over micro-batches of (padded tokens + overhead_tokens), subject to padded size
    <= micro_tokens (a sequence longer than that gets a micro-batch of its own).

    overhead_tokens is the fixed cost of one forward/backward in token-equivalents: measured ~0.2 s on an H100 at
    ~6,500 padded tokens/s. Greedy "fill until full" wastes ~13-25% of the compute on padding; this does not."""
    idx = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    n = len(idx)
    best = [0.0] + [float("inf")] * n          # best[j] = min cost of packing the first j sorted sequences
    cut = [0] * (n + 1)
    for j in range(1, n + 1):
        for i in range(j):                     # last micro-batch = sorted[i:j], padded to lengths[idx[i]]
            padded = (j - i) * lengths[idx[i]]
            if padded > micro_tokens and j - i > 1:
                continue
            c = best[i] + padded + overhead_tokens
            if c < best[j]:
                best[j], cut[j] = c, i
    groups, j = [], n
    while j > 0:
        groups.append(idx[cut[j]:j])
        j = cut[j]
    return groups[::-1]


PAD_MULT = 1      # round padded lengths up to a multiple of this (fewer distinct shapes; see train.py --pad-mult)


def _hidden(text, seqs: list[np.ndarray]) -> torch.Tensor:
    """Final hidden states [b, L, H] for right-padded inputs seq[:-1]. Padding is on the right and every layer is
    causal, so real positions are unaffected by it."""
    L = max(len(s) - 1 for s in seqs)
    L = -(-L // PAD_MULT) * PAD_MULT
    inp = torch.zeros(len(seqs), L, dtype=torch.long)
    att = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, :len(s) - 1] = torch.from_numpy(s[:-1].astype(np.int64))
        att[i, :len(s) - 1] = 1
    inp, att = inp.to(DEV, non_blocking=True), att.to(DEV, non_blocking=True)
    return text(input_ids=inp, attention_mask=None if bool(att.all()) else att).last_hidden_state


def loss_backward(text, head_owner, seqs: list[np.ndarray], masks: list[np.ndarray], head_chunk: int = 1024) -> list[float]:
    """Accumulate d(sum_i mean_NLL_i)/d(params) for one micro-batch and return each datum's mean NLL over its trained
    tokens. This is Tinker's cross_entropy on datums built with reduction="mean": weight 1/n_i on each trained token,
    summed over the batch.

    The output layer runs only at trained positions, in chunks: logits for a 248K vocab are ~1 MB per position in
    float32, so the head is back-propagated chunk by chunk into the hidden states' gradient, then the backbone once."""
    h = _hidden(text, seqs)
    bi, pi, tg, wt = [], [], [], []
    for i, (s, m) in enumerate(zip(seqs, masks)):
        pos = np.nonzero(m[1:])[0]
        bi.append(np.full(len(pos), i)); pi.append(pos); tg.append(s[1:][pos]); wt.append(np.full(len(pos), 1.0 / len(pos)))
    bi_t = torch.from_numpy(np.concatenate(bi)).to(DEV)
    pi_t = torch.from_numpy(np.concatenate(pi)).to(DEV)
    tg_t = torch.from_numpy(np.concatenate(tg).astype(np.int64)).to(DEV)
    wt_t = torch.from_numpy(np.concatenate(wt)).float().to(DEV)
    hs = h[bi_t, pi_t]
    head = head_owner.lm_head
    hdtype = next(p for p in head.parameters() if not p.requires_grad).dtype     # the frozen output weight's dtype
    hs_leaf = hs.detach().to(hdtype).requires_grad_(True)
    nll_all = torch.empty(len(tg_t), device=DEV, dtype=torch.float32)
    for a in range(0, len(tg_t), head_chunk):
        sl = slice(a, a + head_chunk)
        nll = F.cross_entropy(head(hs_leaf[sl]).float(), tg_t[sl], reduction="none")
        (nll * wt_t[sl]).sum().backward()
        nll_all[sl] = nll.detach()
    hs.backward(hs_leaf.grad.to(hs.dtype))
    per = torch.zeros(len(seqs), device=DEV).index_add_(0, bi_t, nll_all * wt_t)
    return per.tolist()


@torch.no_grad()
def token_logprobs(text, head_owner, seq: np.ndarray, head_chunk: int = 2048) -> np.ndarray:
    """log P(seq[t] | seq[:t]) for t = 1..len-1 (float32, length len-1) — what Tinker's compute_logprobs returns
    after its leading None."""
    h = _hidden(text, [seq])[0]
    head = head_owner.lm_head
    hdtype = next(p for p in head.parameters() if not p.requires_grad).dtype
    tg = torch.from_numpy(seq[1:].astype(np.int64)).to(DEV)
    out = torch.empty(len(tg), device=DEV, dtype=torch.float32)
    for a in range(0, len(tg), head_chunk):
        sl = slice(a, a + head_chunk)
        out[sl] = -F.cross_entropy(head(h[sl].to(hdtype)).float(), tg[sl], reduction="none")
    return out.cpu().numpy()
