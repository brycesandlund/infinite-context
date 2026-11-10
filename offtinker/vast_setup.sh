#!/usr/bin/env bash
# One-time setup of a rented CUDA box (Vast.ai or any other) for the off-Tinker stack. Run ON the box, in the directory
# holding the offtinker/ package:
#     bash offtinker/vast_setup.sh            # training / merge / log-probs
#     bash offtinker/vast_setup.sh serve      # also vLLM, in its own virtualenv (it pins its own torch)
# Pick an image with a recent NVIDIA driver + CUDA 12 (e.g. Vast's "PyTorch (cuDNN devel)" template); Python >= 3.10.
set -euo pipefail
python3 -m venv ~/venv-train
~/venv-train/bin/pip install -q --upgrade pip
~/venv-train/bin/pip install -q "torch==2.15.0" "transformers==5.22.2" "flash-linear-attention==0.6.0" safetensors numpy huggingface_hub accelerate
~/venv-train/bin/python -m offtinker.envcheck
if [ "${1:-}" = "serve" ]; then
  python3 -m venv ~/venv-vllm
  ~/venv-vllm/bin/pip install -q --upgrade pip
  ~/venv-vllm/bin/pip install -q "vllm==0.33.0"
  ~/venv-vllm/bin/vllm --version
fi
echo "ready: ~/venv-train/bin/python -m offtinker.selftest"
