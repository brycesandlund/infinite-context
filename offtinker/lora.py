"""LoRA for the Qwen3.5 / 3.6 models (dense and MoE), laid out the way Tinker lays out its adapters.

What Tinker adapts (read off a downloaded Tinker adapter, see training_runs.md "off-Tinker stack"):
  * every attention projection: full-attention q/k/v/o, linear-attention (Gated DeltaNet) q, k, v, z and out_proj —
    with q, k and v as THREE separate rank-r adapters even though Hugging Face stores them fused as `in_proj_qkv`;
  * the MLP gate/up/down projections (MoE models: the shared expert's, plus the routed experts — see ExpertLoRA);
  * the output layer (`unembed_tokens` = `lm_head`).
  Not adapted: the DeltaNet a/b gates, conv, norms, the input embedding, the vision tower.
  rank 32, alpha 32 (scale 1.0), A ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (the PEFT default), B = 0, float32.

The adapter file uses Tinker's key names (`base_model.model.model.layers.N.<module>.lora_{A,B}.weight`), so the same
loader / merger handles adapters trained here and adapters downloaded from Tinker.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PREFIX = "base_model.model.model."


class LoRALinear(nn.Module):
    """y = base(x) + scale * cat_i(B_i A_i x). `parts` = [(name, out_size)], one independent (A, B) per output slice
    (a single part covers the whole output). The base weight stays frozen in its own dtype (bf16); A and B are float32
    parameters cast to the activation dtype on the fly, so updates accumulate at full precision."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, parts: list[tuple[str, int]]):
        super().__init__()
        assert sum(s for _, s in parts) == base.out_features, (parts, base.out_features)
        self.base = base
        self.r, self.scale = r, alpha / r
        self.part_names = [n for n, _ in parts]
        dev = base.weight.device
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32)) for _ in parts])
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(s, r, device=dev, dtype=torch.float32)) for _, s in parts])
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))      # U(-1/sqrt(fan_in), 1/sqrt(fan_in))

    def forward(self, x):
        y = self.base(x)
        d = [F.linear(F.linear(x, a.to(x.dtype)), b.to(x.dtype)) for a, b in zip(self.lora_A, self.lora_B)]
        d = d[0] if len(d) == 1 else torch.cat(d, dim=-1)
        return y + (d if self.scale == 1.0 else self.scale * d)

    def delta(self) -> torch.Tensor:
        """The full-rank update to add to base.weight (float32), for merging."""
        return self.scale * torch.cat([b.float() @ a.float() for a, b in zip(self.lora_A, self.lora_B)], dim=0)


def grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Rows of x [S, in] are sorted by group; group g = rows offs[g-1]:offs[g] and is multiplied by w[g] [in, out].
    On CUDA in bf16 this is torch's native grouped matmul (one kernel for all experts); elsewhere a plain loop."""
    if x.is_cuda and x.dtype == torch.bfloat16 and hasattr(torch, "_grouped_mm"):
        return torch._grouped_mm(x, w, offs=offs)
    out, a = [], 0
    for g, b in enumerate(offs.tolist()):
        if b > a:
            out.append(x[a:b] @ w[g])
        a = b
    return torch.cat(out, dim=0) if out else x.new_zeros(0, w.shape[2])


import os as _os
# Sum each token's K expert outputs with index_add_ (default; ~10% faster end to end on the 35B-A3B) or, with =0, by
# un-sorting and summing in a fixed order (deterministic, one more [S, H] gather).
# Tried and dropped (H200, 2026-11-10): a padded per-expert bmm for the LoRA factors (slower: expert loads are uneven,
# max ~3.5x the mean) and a custom batched backward for grouped_mm (no gain: the step is GPU-bound, not launch-bound).
ACC_INDEX_ADD = _os.environ.get("OFFTINKER_MOE_INDEX_ADD", "1") == "1"


class ExpertLoRA(nn.Module):
    """The routed experts of one MoE block (Hugging Face `Qwen3_5MoeExperts`: gate_up_proj [E, 2I, H] = [gate; up],
    down_proj [E, H, I]) with Tinker's expert adapters, and a forward pass that handles all experts at once.

    Tinker's layout (read off a downloaded adapter; w1 = gate, w3 = up, w2 = down as in  down(silu(w1 x) * w3 x)):
        w1, w3:  A shared by all experts [1, r, H],  B per expert [E, I, r];  w1's A and w3's A are the SAME tensor
                 (identical in every layer of Tinker adapters, at init and after training) — i.e. one LoRA on the
                 fused gate_up projection with A [r, H] and B [E, 2I, r]
        w2:      A per expert [E, r, I],             B shared     [1, H, r]
    i.e. every expert gets its own rank-r update, with the H-sided factor shared across the block's experts.

    Forward: the (token, expert) pairs are sorted by expert and each projection is ONE grouped matmul over the sorted
    rows, instead of Hugging Face's Python loop over 256 experts. Frozen weights are kept transposed ([E, in, out], what
    grouped_mm wants) and replace the original parameters (no second copy)."""

    part_names = ["w1", "w2", "w3"]

    def __init__(self, experts: nn.Module, r: int, alpha: float):
        super().__init__()
        gu, dn = experts.gate_up_proj.data, experts.down_proj.data
        E, I2, H = gu.shape
        I = I2 // 2
        assert dn.shape == (E, H, I), (gu.shape, dn.shape)
        self.E, self.I, self.H, self.r, self.scale = E, I, H, r, alpha / r
        self.act_fn = experts.act_fn
        self.register_buffer("gu_t", gu.transpose(1, 2).contiguous(), persistent=False)     # [E, H, 2I]
        self.register_buffer("dn_t", dn.transpose(1, 2).contiguous(), persistent=False)     # [E, I, H]
        experts.gate_up_proj = experts.down_proj = None
        dev, f32 = gu.device, torch.float32
        A13 = nn.Parameter(torch.empty(1, r, H, device=dev, dtype=f32))      # ONE tensor, stored as both w1.A and w3.A
        A2 = nn.Parameter(torch.empty(E, r, I, device=dev, dtype=f32))
        for a, bound in ((A13, 1 / math.sqrt(H)), (A2, 1 / math.sqrt(r * I))):
            nn.init.uniform_(a, -bound, bound)        # Tinker's init scales (read off a step-0 Tinker adapter)
        A1 = A3 = A13
        B1, B3 = (nn.Parameter(torch.zeros(E, I, r, device=dev, dtype=f32)) for _ in range(2))
        B2 = nn.Parameter(torch.zeros(1, H, r, device=dev, dtype=f32))
        self.lora_A = nn.ParameterList([A1, A2, A3])       # order = part_names
        self.lora_B = nn.ParameterList([B1, B2, B3])

    def forward(self, x: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        N, K = top_k_index.shape
        dt, s = x.dtype, self.scale
        (A1, A2, A3), (B1, B2, B3) = self.lora_A, self.lora_B
        with torch.no_grad():
            ids = top_k_index.reshape(-1)
            order = torch.argsort(ids, stable=True)                    # pairs sorted by expert
            tok = torch.div(order, K, rounding_mode="floor")
            offs = torch.bincount(ids, minlength=self.E).cumsum(0).to(torch.int32)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=order.device)
        xs = x[tok]                                                    # [S, H]
        gu = grouped_mm(xs, self.gu_t, offs)                         # [S, 2I] = [gate | up]
        z = (x @ A1[0].to(dt).T)[tok]                                  # shared (and gate/up-tied) A: once per token, [S, r]
        d = grouped_mm(z, torch.cat([B1, B3], dim=1).to(dt).transpose(1, 2), offs)     # [S, 2I]
        gate, up = (gu + d if s == 1.0 else gu + s * d).chunk(2, dim=-1)
        h = self.act_fn(gate) * up
        out = grouped_mm(h, self.dn_t, offs)                         # [S, H]
        q = grouped_mm(h, A2.to(dt).transpose(1, 2), offs)
        q = q @ B2[0].to(dt).T
        out = out + (q if s == 1.0 else s * q)
        out = out * top_k_weights.reshape(-1)[order].unsqueeze(-1).to(dt)
        if ACC_INDEX_ADD:
            return torch.zeros(N, self.H, dtype=dt, device=x.device).index_add_(0, tok, out)
        return out[inv].view(N, K, self.H).sum(dim=1).to(dt)          # back to token order; sum the K experts

    def delta(self) -> dict[str, torch.Tensor]:
        (A1, A2, A3), (B1, B2, B3) = self.lora_A, self.lora_B
        return {"gate_up": self.scale * torch.cat([B1 @ A1, B3 @ A3], dim=1), "down": self.scale * (B2 @ A2)}


def _targets(text_model, lm_head) -> list[tuple[object, str, str, list[tuple[str, int]]]]:
    """(parent module, attribute, adapter key stem, parts) for every adapted Linear."""
    out = []
    for i, layer in enumerate(text_model.layers):
        if hasattr(layer, "linear_attn"):
            m = layer.linear_attn
            st = f"layers.{i}.linear_attn."
            out.append((m, "in_proj_qkv", st, [("in_proj_q", m.key_dim), ("in_proj_k", m.key_dim), ("in_proj_v", m.value_dim)]))
            out.append((m, "in_proj_z", st, [("in_proj_z", m.in_proj_z.out_features)]))
            out.append((m, "out_proj", st, [("out_proj", m.out_proj.out_features)]))
        else:
            m = layer.self_attn
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                out.append((m, n, f"layers.{i}.self_attn.", [(n, getattr(m, n).out_features)]))
        moe = hasattr(layer.mlp, "experts")                 # Qwen3.5 MoE: routed experts + one always-on shared expert
        mlp, st = (layer.mlp.shared_expert, f"layers.{i}.mlp.shared_expert.") if moe else (layer.mlp, f"layers.{i}.mlp.")
        for n in ("gate_proj", "up_proj", "down_proj"):
            out.append((mlp, n, st, [(n, getattr(mlp, n).out_features)]))
        if moe:
            out.append((layer.mlp, "experts", f"layers.{i}.mlp.experts.", None))      # ExpertLoRA: parts w1, w2, w3
    return out


class LoRASet:
    """All LoRA modules of one model. `text_model` = Qwen3_5TextModel, `lm_head` holder = the object owning .lm_head."""

    def __init__(self, text_model, head_owner, r: int = 32, alpha: float = 32.0, seed: int = 0):
        self.r, self.alpha = r, alpha
        self.modules: dict[str, LoRALinear] = {}      # key stem + part name -> (module, part index) via self.index
        self.index: dict[str, tuple[LoRALinear, int]] = {}
        g = torch.random.get_rng_state()
        torch.manual_seed(seed)
        for p in list(text_model.parameters()) + list(head_owner.lm_head.parameters()):
            p.requires_grad_(False)
        for parent, attr, stem, parts in _targets(text_model, head_owner.lm_head):
            if parts is None:
                mod = ExpertLoRA(getattr(parent, attr), r, alpha)
                parts = [(pn, 0) for pn in mod.part_names]
            else:
                mod = LoRALinear(getattr(parent, attr), r, alpha, parts)
            setattr(parent, attr, mod)
            for j, (pn, _) in enumerate(parts):
                self.index[stem + pn] = (mod, j)
        head = LoRALinear(head_owner.lm_head, r, alpha, [("unembed_tokens", head_owner.lm_head.out_features)])
        head_owner.lm_head = head
        self.index["unembed_tokens"] = (head, 0)
        torch.random.set_rng_state(g)

    def parameters(self) -> list[nn.Parameter]:
        seen, out = set(), []
        for mod, _ in self.index.values():
            if id(mod) not in seen:
                seen.add(id(mod))
                out += list(mod.lora_A) + list(mod.lora_B)
        uniq = {id(p): p for p in out}                # the experts' tied A appears under two names
        return list(uniq.values())

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- adapter files (Tinker / PEFT key names) ---------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd = {}
        for key, (mod, j) in self.index.items():
            sd[f"{PREFIX}{key}.lora_A.weight"] = mod.lora_A[j].detach().float().cpu().clone()
            sd[f"{PREFIX}{key}.lora_B.weight"] = mod.lora_B[j].detach().float().cpu().clone()
        return sd

    def save(self, out_dir: str | Path, extra: dict | None = None) -> None:
        from safetensors.torch import save_file
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(self.state_dict(), str(out / "adapter_model.safetensors"))
        (out / "adapter_config.json").write_text(json.dumps(
            {"peft_type": "LORA", "r": self.r, "lora_alpha": self.alpha, "target_modules": "all-linear",
             "lora_dropout": 0.0, "bias": "none", "use_rslora": False, "layout": "tinker", **(extra or {})}, indent=1))

    def load(self, adapter_dir: str | Path, strict: bool = True) -> None:
        sd = read_adapter(adapter_dir)
        used = set()
        for key, (mod, j) in self.index.items():
            for ab, plist in (("lora_A", mod.lora_A), ("lora_B", mod.lora_B)):
                k = f"{PREFIX}{key}.{ab}.weight"
                if k not in sd:
                    if strict:
                        raise KeyError(f"adapter is missing {k}")
                    continue
                t = sd[k]
                assert tuple(t.shape) == tuple(plist[j].shape), (k, tuple(t.shape), tuple(plist[j].shape))
                with torch.no_grad():
                    plist[j].copy_(t.to(plist[j].device, torch.float32))
                used.add(k)
        for k in sd:                                  # tied factors must agree in the file
            if k.endswith(".mlp.experts.w1.lora_A.weight"):
                assert torch.equal(sd[k], sd[k.replace(".w1.", ".w3.")]), f"{k}: w1 and w3 A differ — not a Tinker-layout adapter"
        extra = sorted(set(sd) - used)
        if extra and strict:
            raise KeyError(f"adapter has {len(extra)} tensors this layout does not adapt, e.g. {extra[:4]}")


def read_adapter(adapter_dir: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    p = Path(adapter_dir)
    return load_file(str(p / "adapter_model.safetensors" if p.is_dir() else p))


def adapter_scale(adapter_dir: str | Path) -> float:
    p = Path(adapter_dir)
    cfg = json.loads((p / "adapter_config.json").read_text()) if (p / "adapter_config.json").exists() else {}
    return float(cfg.get("lora_alpha", 32)) / float(cfg.get("r", 32))


# Adapter key stem -> (Hugging Face checkpoint weight name, row-slice name or None). Used by merge.py, which patches
# the checkpoint's safetensors directly (no model class involved).
def hf_weight_for(key: str) -> tuple[str, str | None]:
    """'layers.3.self_attn.q_proj' -> ('model.language_model.layers.3.self_attn.q_proj.weight', None);
    'layers.0.linear_attn.in_proj_k' -> ('model.language_model.layers.0.linear_attn.in_proj_qkv.weight', 'k')."""
    if key == "unembed_tokens":
        return "lm_head.weight", None
    stem, leaf = key.rsplit(".", 1)
    if stem.endswith(".mlp.experts"):         # MoE experts: 3-D weights without a ".weight" suffix
        return (f"model.language_model.{stem}.gate_up_proj", {"w1": "gate", "w3": "up"}[leaf]) if leaf != "w2" else \
               (f"model.language_model.{stem}.down_proj", None)
    if leaf in ("in_proj_q", "in_proj_k", "in_proj_v"):
        return f"model.language_model.{stem}.in_proj_qkv.weight", leaf[-1]
    return f"model.language_model.{key}.weight", None
