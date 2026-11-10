"""Modal wrapper for the off-Tinker stack: the same CLIs (offtinker.train / logprobs / merge) in a GPU container with
persistent volumes, plus a vLLM server for evals. Run from the repo root (see offtinker/README.md):

    uvx modal volume put ic-data ~/.cache/infinite-context/offtinker/data/<name> /<name>          # upload a dataset
    OFFTINKER_GPU=H100 uvx modal run --detach offtinker/modal_app.py::run --module train \\
        --args "--data /data/<name> --out /runs/<run>"                                            # train
    uvx modal run offtinker/modal_app.py::run --module merge --args "--adapter /runs/<run>/final --out /runs/<run>/merged"
    IC_SERVE_MODEL=/runs/<run>/merged IC_SERVE_NAME=<run> uvx modal deploy offtinker/modal_app.py  # serve (scales to 0)

Volumes: ic-hf-cache (/hf, model downloads), ic-data (/data, exported datasets), ic-runs (/runs, adapters, merged
models, metrics). Nothing here is Modal-specific beyond this file: on Vast or any CUDA box run the same
`python -m offtinker.<module>` commands (offtinker/vast_setup.sh).
"""

from __future__ import annotations

import os
import shlex
import subprocess

import modal

GPU = os.environ.get("OFFTINKER_GPU", "H100")
SERVE_NAME = os.environ.get("IC_SERVE_NAME", "")            # set at deploy time -> its own app + URL per served model
app = modal.App("ic-serve-" + SERVE_NAME if SERVE_NAME else "ic-offtinker")

VOLS = {
    "/hf": modal.Volume.from_name("ic-hf-cache", create_if_missing=True),
    "/data": modal.Volume.from_name("ic-data", create_if_missing=True),
    "/runs": modal.Volume.from_name("ic-runs", create_if_missing=True),
}
# TRITON_CACHE_DIR on the volume: the DeltaNet kernels take minutes to compile on a cold container otherwise.
ENV = {"HF_HOME": "/hf", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "TRITON_CACHE_DIR": "/hf/triton_cache"}

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    # Pinned: the Gated DeltaNet Triton kernels are sensitive to the torch/triton/fla combination (fla 0.5.2 fails to
    # import under triton 3.9). `python -m offtinker.envcheck` verifies a box.
    .pip_install("torch==2.15.0", "transformers==5.22.2", "flash-linear-attention==0.6.0", "safetensors", "numpy",
                 "huggingface_hub", "accelerate")
    .env(ENV)
    .add_local_python_source("offtinker")
)
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.33.0", "huggingface_hub")
    # FlashInfer's sampler JIT-compiles a CUDA extension and needs nvcc, which this slim image lacks; vLLM's own
    # torch sampler is used instead.
    .env({**ENV, "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)


def _sh(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=train_image, gpu=GPU, volumes=VOLS, timeout=24 * 3600)
def run(module: str, args: str = "") -> None:
    """python -m offtinker.<module> <args> on the GPU (module = train | logprobs | merge | selftest)."""
    _sh(["python", "-m", f"offtinker.{module}", *shlex.split(args)])
    VOLS["/runs"].commit()


@app.function(image=train_image, gpu=GPU, volumes=VOLS, timeout=24 * 3600,
              retries=modal.Retries(max_retries=4, initial_delay=30.0, backoff_coefficient=1.0))
def train_long(args: str) -> None:
    """A full training run: retried on preemption / crash, and train.py resumes from <out>/ckpt each time."""
    _sh(["python", "-m", "offtinker.train", *shlex.split(args)])
    VOLS["/runs"].commit()


@app.function(image=train_image, gpu=GPU, volumes=VOLS, timeout=3600)
def shell(cmd: str) -> None:
    """Ad-hoc command in the training image (e.g. --cmd 'nvidia-smi; pip list')."""
    _sh(["bash", "-lc", cmd])


@app.local_entrypoint()
def sh(file: str) -> None:
    """Run a local bash script file in the training image on the GPU:  modal run offtinker/modal_app.py::sh --file x.sh"""
    shell.remote(open(file).read())


@app.function(image=vllm_image, gpu=GPU, volumes=VOLS, timeout=3600)
def vshell(cmd: str) -> None:
    """Ad-hoc command in the vLLM image."""
    _sh(["bash", "-lc", cmd])


# ---------------------------------------------------------------------------------------------------------------
# vLLM server (OpenAI-compatible). The eval harness renders prompts itself and sends TOKEN IDS to /v1/completions,
# so the chat template is never used. Scales to zero after SERVE_IDLE seconds without requests.
# ---------------------------------------------------------------------------------------------------------------

SERVE_MODEL = os.environ.get("IC_SERVE_MODEL", "Qwen/Qwen3.5-9B")   # hub id or a merged model dir under /runs
SERVE_MAX_LEN = os.environ.get("IC_MAX_LEN", "16384")               # >= the per-agent context budget
SERVE_IDLE = int(os.environ.get("IC_SERVE_IDLE", "300"))
_KEY_FILE = os.path.expanduser("~/.cache/infinite-context/vllm_api_key.txt")
_serve_secret = modal.Secret.from_dict({
    "IC_SERVE_MODEL": SERVE_MODEL, "IC_MAX_LEN": SERVE_MAX_LEN,
    "IC_VLLM_KEY": open(_KEY_FILE).read().strip() if os.path.exists(_KEY_FILE) else os.environ.get("IC_VLLM_KEY", ""),
})


@app.function(image=vllm_image, gpu=GPU, volumes=VOLS, timeout=24 * 3600, scaledown_window=SERVE_IDLE,
              max_containers=1, secrets=[_serve_secret])
@modal.concurrent(max_inputs=1000)
@modal.web_server(port=8000, startup_timeout=1200)
def serve() -> None:
    cmd = [
        "vllm", "serve", os.environ["IC_SERVE_MODEL"], "--served-model-name", "ic", "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["IC_VLLM_KEY"], "--max-model-len", os.environ["IC_MAX_LEN"],
        "--generation-config", "vllm",            # do NOT inherit the checkpoint's top_k/top_p/temperature defaults
        "--limit-mm-per-prompt", '{"image": 0, "video": 0}',
        "--gpu-memory-utilization", "0.92", "--max-num-seqs", "256",
    ]
    print("$ vllm serve", os.environ["IC_SERVE_MODEL"], "(max len", os.environ["IC_MAX_LEN"] + ")", flush=True)
    subprocess.Popen(cmd)
