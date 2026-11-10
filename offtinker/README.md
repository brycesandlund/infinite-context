# Off-Tinker training and evals

LoRA fine-tuning and eval serving for the dense Qwen3.5 models on a rented GPU (Modal or Vast), reproducing the
Tinker loop in `sft.py` and the Tinker eval backend. The trace generator, renderer, tools and eval harness are
unchanged; only the forward/backward passes move. Validation against Tinker is in `training_runs.md`
("OFF-TINKER STACK").

| File | What it does |
|---|---|
| `sft.py` (`SFT_EXPORT=<dir>`) | Writes the exact training set (token ids, trained-token mask) and Tinker's batch order. |
| `offtinker/train.py` | The training loop: same batches, loss, Adam and LoRA layout as Tinker. Checkpoints and resumes. |
| `offtinker/lora.py` | LoRA modules and adapter files in Tinker's layout and key names (adapters load in either direction). |
| `offtinker/merge.py` | Adapter + base checkpoint -> a plain model directory for vLLM. Also takes adapters downloaded from Tinker. |
| `offtinker/logprobs.py` | Per-token log-probs, for parity checks. |
| `offtinker/bench.py` | Throughput benchmark and profile on real batches. |
| `offtinker/envcheck.py`, `selftest.py` | Version / kernel check for a GPU box; CPU self-test on a tiny random model. |
| `offtinker/modal_app.py` | Modal wrapper: GPU functions, volumes, and the vLLM server. |
| `offtinker/vast_setup.sh`, `serve.sh` | The same on a rented box (written, NOT yet run on Vast). |
| `eval/backends.py::VLLMBackend` | `BACKEND=vllm`: the eval harness against our own server, sending token ids. |
| `scripts/offtinker_parity.py` | Tinker-side reference runs and the comparisons. |

Pinned versions (the linear-attention Triton kernels break across versions): torch 2.15.0, transformers 5.22.2,
flash-linear-attention 0.6.0, vLLM 0.33.0. `python -m offtinker.envcheck` verifies a box.

## 1. Export a dataset (laptop)

The Qwen3.5/3.6 family shares one tokenizer, so traces and the trace cache are the same for every model in it.

```bash
TRAIN=0 SFT_EXPORT=~/.cache/infinite-context/offtinker/data/<name> <the usual sft.py recipe env> PYTHONPATH=. uv run python sft.py
```

## 2. Train

Modal (from the repo root; `OFFTINKER_GPU` picks the GPU type):

```bash
uvx modal volume put ic-data ~/.cache/infinite-context/offtinker/data/<name> /<name>
OFFTINKER_GPU=H200 uvx modal run --detach offtinker/modal_app.py::train_long \
  --args "--data /data/<name> --out /runs/<run> --grad-ckpt 0 --micro-tokens 11000"
uvx modal app logs ic-offtinker            # progress lines: step, NLL, tokens/s, ETA
uvx modal volume get ic-runs /<run>/final/adapter_model.safetensors .
```

Measured on the 18b-recipe export (199.5M tokens, 7,330 steps of 16):

| GPU | Flags | Tokens/s | Full run | Modal price |
|---|---|---|---|---|
| H200 (141 GB) | `--grad-ckpt 0 --micro-tokens 11000` (peak 109 GB) | ~8,300 | ~6.7 h | ~$30 |
| H100 (80 GB) | defaults: `--grad-ckpt 1 --micro-tokens 32000` (peak 32 GB) | ~6,200 | ~8.9 h | ~$35 |

Without gradient checkpointing the saved activations cost about 6 MB per token, so it does not fit an H100. The
first minutes of a fresh container are slow while the Triton kernels compile (cached on the `ic-hf-cache` volume
afterwards). `train.py` checkpoints every 300 steps and `--resume 1` (the default) continues from the last one;
`train_long` retries on preemption. Other flags: `--max-steps`, `--init-adapter <dir>` (warm start),
`--snapshot-steps`.

Vast or any CUDA box (copy `offtinker/` and the export there, then on the box):

```bash
bash offtinker/vast_setup.sh
nohup ~/venv-train/bin/python -m offtinker.train --data data/<name> --out runs/<run> > train.log 2>&1 &
```

## 3. Merge and serve

```bash
uvx modal run offtinker/modal_app.py::run --module merge --args "--adapter /runs/<run>/final --out /runs/<run>/merged"
OFFTINKER_GPU=H100 IC_SERVE_MODEL=/runs/<run>/merged IC_SERVE_NAME=<run> uvx modal deploy offtinker/modal_app.py   # prints the URL
```

`merge.py` rounds merged weights to bf16 WITHOUT bias (`--rounding stochastic`, the default). Do not use a plain
cast: at our learning rate most LoRA updates are smaller than half a bf16 step and get rounded away (loss on a
150-step adapter: 0.0605 unmerged, 0.0807 after a plain cast, 0.0603 with unbiased rounding).

Measured capacity on agent-shaped requests: H100 ~33,000 tokens/s (~$0.033 per 1M on Modal), L40S ~9,000
(~$0.060 per 1M) — the H100 is faster and cheaper per token (`scripts/vllm_load_test.py`).

The server needs the key in `~/.cache/infinite-context/vllm_api_key.txt` (created once, private). It scales to
zero after 5 idle minutes (`IC_SERVE_IDLE`), so it only bills while an eval is running; remove it with
`uvx modal app stop ic-serve-<run>`. On a rented box: `IC_VLLM_KEY=... bash offtinker/serve.sh runs/<run>/merged`
and tunnel port 8000.

## 4. Evaluate

Everything in `eval/run.py` works as before with two extra variables:

```bash
BACKEND=vllm VLLM_URL=https://<workspace>--ic-serve-<run>-serve.us-east.modal.run <usual eval env> PYTHONPATH=. uv run python -m eval.run
```

`AGENT_CONTEXT` must not exceed the server's `IC_MAX_LEN` (default 16384). `MODEL_NAME` only selects the tokenizer
and renderer, which are shared across the family, so the default is fine.

## Ad-hoc GPU commands

```bash
OFFTINKER_GPU=H100 uvx modal run offtinker/modal_app.py::sh --file my_script.sh     # a local bash script, in the training image
uvx modal run offtinker/modal_app.py::run --module envcheck
```
