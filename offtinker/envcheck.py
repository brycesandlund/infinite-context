"""Environment check for a GPU box:  python -m offtinker.envcheck
Prints library versions and verifies that the Gated DeltaNet fast path (flash-linear-attention's Triton kernels)
loads and runs forward + backward in bf16 — the piece most sensitive to torch / triton / fla version drift."""

import importlib.metadata as md
import time

import torch

for p in ("torch", "triton", "transformers", "flash-linear-attention", "safetensors", "numpy"):
    try:
        print(f"{p:24s} {md.version(p)}")
    except Exception as e:      # noqa: BLE001
        print(f"{p:24s} MISSING ({e})")
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB" if torch.cuda.is_available() else "")

from transformers.utils.import_utils import is_flash_linear_attention_available  # noqa: E402

print("transformers sees fla:", is_flash_linear_attention_available())
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: E402

from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule  # noqa: E402

torch.manual_seed(0)
B, T, H, K, V = 2, 1000, 4, 64, 64
mk = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float32)
q, k, v = mk(B, T, H, K), mk(B, T, H, K), mk(B, T, H, V)
g = -torch.rand(B, T, H, device="cuda") * 0.1
beta = torch.rand(B, T, H, device="cuda")
ref, _ = torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
print(f"fla kernel vs transformers' torch reference (fp32): max |diff| {float((out - ref).abs().max()):.2e} (scale {float(ref.abs().mean()):.2e})")
qb, kb, vb = (x.to(torch.bfloat16).requires_grad_(True) for x in (q, k, v))
t0 = time.time()
o, _ = chunk_gated_delta_rule(qb, kb, vb, g=g, beta=beta.to(torch.bfloat16), initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
o.float().pow(2).sum().backward()
print(f"bf16 forward+backward ok ({time.time() - t0:.1f}s incl. kernel compile); grad finite: {bool(torch.isfinite(qb.grad).all())}; "
      f"bf16 vs fp32 forward max |diff| {float((o.float() - ref).abs().max()):.2e}")
print("ENVCHECK OK")
