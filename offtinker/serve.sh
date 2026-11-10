#!/usr/bin/env bash
# vLLM server for evals on a rented box (the Modal equivalent is offtinker/modal_app.py::serve).
#     IC_VLLM_KEY=<key> bash offtinker/serve.sh <merged model dir or hub id> [max model len, default 16384]
# Then from the laptop:  ssh -N -L 8000:localhost:8000 <box>   and   BACKEND=vllm VLLM_URL=http://localhost:8000 ...
set -euo pipefail
exec ~/venv-vllm/bin/vllm serve "$1" --served-model-name ic --host 0.0.0.0 --port 8000 \
  --api-key "${IC_VLLM_KEY:?set IC_VLLM_KEY}" --max-model-len "${2:-16384}" \
  --generation-config vllm --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --gpu-memory-utilization 0.92 --max-num-seqs 256
