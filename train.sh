#!/usr/bin/env bash
# Launch an RL run, warm-starting from the latest SFT checkpoint, with the log
# filtered of one benign macOS stderr line: wandb-core emits
#   "MallocStackLogging: can't turn off malloc stack logging because it was not enabled."
# once per spawned subprocess. It's triggered by MallocNanoZone=0 in the env (a
# PyTorch fork-safety workaround) interacting with macOS libmalloc — cosmetic, NOT
# an error. Do NOT unset MallocNanoZone to silence it; that re-exposes the fork crash.
# Python can't intercept the line (OS-level stderr from a grandchild process), so we
# filter it at the redirect here.
#
# Usage:  ./train.sh [logfile]          (defaults to /tmp/rl_run.log)
#   TINKER_API_KEY / WANDB_API_KEY must already be in the environment.
#   OPENAI_API_KEY too when JUDGE=1 (the default — gpt-5.4-nano grades subagents);
#   set JUDGE=0 to disable the judge and run on the root gold reward alone.
#   CKPT defaults to the latest SFT checkpoint; override by exporting CKPT yourself.
set -euo pipefail

if [ "${JUDGE:-1}" != "0" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "JUDGE is on but OPENAI_API_KEY is unset. Export it, or run with JUDGE=0." >&2
  exit 1
fi

LOG="${1:-/tmp/rl_run.log}"
: "${CKPT:=$(cat "$HOME/.cache/infinite-context/last_sft_checkpoint.txt")}"
export CKPT
export WANDB="${WANDB:-1}"

echo "RL run -> $LOG   (warm-start CKPT=$CKPT, WANDB=$WANDB)"
uv run python -u train.py 2>&1 \
  | grep --line-buffered -vF "MallocStackLogging:" \
  > "$LOG"
