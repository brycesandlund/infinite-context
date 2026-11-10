"""LoRA for the dense Qwen3.5 models, laid out the way Tinker lays out its adapters.

What Tinker adapts (read off a downloaded Tinker adapter, see training_runs.md "off-Tinker stack"):
  * every attention projection: full-attention q/k/v/o, linear-attention (Gated DeltaNet) q, k, v, z and out_proj —
    with q, k and v as THREE separate rank-r adapters even though Hugging Face stores them fused as `in_proj_qkv`;
  * the MLP gate/up/down projections;
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
        for n in ("gate_proj", "up_proj", "down_proj"):
            out.append((layer.mlp, n, f"layers.{i}.mlp.", [(n, getattr(layer.mlp, n).out_features)]))
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
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # -- adapter files (Tinker / PEFT key names) ---------------------------------------------------------------

    def state_dict(self) -> dict[str, torch.Tensor]:
        sd = {}
        for key, (mod, j) in self.index.items():
            sd[f"{PREFIX}{key}.lora_A.weight"] = mod.lora_A[j].detach().float().cpu().contiguous()
            sd[f"{PREFIX}{key}.lora_B.weight"] = mod.lora_B[j].detach().float().cpu().contiguous()
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
    if leaf in ("in_proj_q", "in_proj_k", "in_proj_v"):
        return f"model.language_model.{stem}.in_proj_qkv.weight", leaf[-1]
    return f"model.language_model.{key}.weight", None
