#!/usr/bin/env bash
# OOLONG length points matching the June chart (eval_results/RESULTS.md): same problems (OOLONG_BASE=2000000, idx 0-9
# per family = one per dataset), 8K agent budget, MAX_CHUNK_TOKENS=6000, MAX_DEPTH=10, TEMP 0.2.
#   usage: scripts/eval_oolong_chart.sh <tag> <ckpt> <doc_tokens>
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
TAG=$1; CKPT=$2; DOC=$3
for attempt in $(seq 1 20); do
  CKPT=$CKPT OOLONG_BASE=2000000 EVAL_TASKS=oolong_counting,oolong_user,oolong_temporal \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal N_PER_TASK=10 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=$DOC AGENT_CONTEXT=8000 MAX_CHUNK_TOKENS=6000 MAX_DEPTH=10 \
TEMP=0.2 MAX_NODES=4000 OUT=/tmp/eval_$TAG PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval $TAG attempt $attempt failed at $(date +%H:%M:%S); retrying in 3 min ====="
  sleep 180
done
echo "===== $TAG DONE $(date +%H:%M:%S) ====="
