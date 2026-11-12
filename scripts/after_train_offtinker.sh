#!/usr/bin/env bash
# Everything after an off-Tinker training run (offtinker/README.md) — one watcher, so nothing depends on a session being
# alive: wait for the Modal training app to stop, check the final adapter exists, merge it, serve it with vLLM, run the
# 18b eval suite (scripts/after_run9b.sh), stop the server.
#   usage: scripts/after_train_offtinker.sh <modal app id> <run dir under /runs> <eval name> [base model] [serve gpu]
#   nohup caffeinate -is zsh -ic "bash scripts/after_train_offtinker.sh ap-... sft35b_18b_v15 35b_18b" > /tmp/after_train.log 2>&1 &
# (zsh -ic: the API keys live in the zsh profile.)
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
APP=$1; RUN=$2; NAME=$3; MODEL=${4:-Qwen/Qwen3.6-35B-A3B}; SERVE_GPU=${5:-H200}
say() { echo "[$(date +%H:%M:%S)] $*"; }
state() {
  uvx modal app list --json 2>/dev/null | python3 -c "
import json, sys
try:
    print(next((a['State'] for a in json.load(sys.stdin) if a['App ID'] == '$APP'), 'missing'))
except Exception:
    print('unknown')"
}

say "waiting for training app $APP (run $RUN) to stop"
while true; do
  s=$(state)
  case "$s" in stopped*|missing) break;; esac     # 'unknown' (CLI / network error): keep waiting
  sleep 300
done
say "training app state: $s"
if ! uvx modal volume ls ic-runs "/$RUN/final" 2>/dev/null | grep -q adapter_model.safetensors; then
  say "ABORT: no /runs/$RUN/final/adapter_model.safetensors — the run did not finish (uvx modal app logs $APP)"; exit 1
fi
uvx modal volume get --force ic-runs "/$RUN/metrics.jsonl" "eval_results/offtinker/${RUN}_metrics.jsonl" > /dev/null 2>&1
say "final training lines: $(tail -1 "eval_results/offtinker/${RUN}_metrics.jsonl" 2>/dev/null | cut -c1-200)"

say "merging -> /runs/$RUN/merged"
OFFTINKER_GPU=H100 uvx modal run offtinker/modal_app.py::run --module merge \
  --args "--adapter /runs/$RUN/final --out /runs/$RUN/merged --model $MODEL" > "/tmp/merge_$NAME.log" 2>&1 \
  || { say "ABORT: merge failed (/tmp/merge_$NAME.log)"; exit 1; }
say "$(grep '^merged' "/tmp/merge_$NAME.log" | tail -1)"

SERVE=$(echo "$NAME" | tr -d '_')
OFFTINKER_GPU=$SERVE_GPU IC_SERVE_MODEL=/runs/$RUN/merged IC_SERVE_NAME=$SERVE IC_SERVE_IDLE=600 \
  uvx modal deploy offtinker/modal_app.py > "/tmp/deploy_$NAME.log" 2>&1 || { say "ABORT: deploy failed (/tmp/deploy_$NAME.log)"; exit 1; }
URL=$(grep -o "https://[a-z0-9.-]*ic-serve-$SERVE[a-z0-9.-]*modal.run" "/tmp/deploy_$NAME.log" | head -1)
[ -n "$URL" ] || { say "ABORT: no server URL in /tmp/deploy_$NAME.log"; exit 1; }
say "serving at $URL ($SERVE_GPU); running the eval suite"

CONCURRENCY=${CONCURRENCY:-16} bash scripts/after_run9b.sh "$NAME" "$URL"

uvx modal app stop "ic-serve-$SERVE" > /dev/null 2>&1
say "server ic-serve-$SERVE stopped. ALL DONE"
