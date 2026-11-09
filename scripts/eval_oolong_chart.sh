#!/usr/bin/env bash
# OOLONG length points on the June chart's PROBLEMS (eval_results/RESULTS.md: OOLONG_BASE=2000000, idx 0-9 per family =
# one per dataset), 8K agent budget, TEMP 0.2. Standard protocol limits, NOT the June ones: no depth cap (MAX_DEPTH=10
# would cut long folds) and a non-binding chunk limit; neither limit bound the June trees.
#   usage: scripts/eval_oolong_chart.sh <tag> <ckpt> <doc_tokens>
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
TAG=$1; CKPT=$2; DOC=$3
# MAX_NODES = 4 x the nominal binary tree (2 x next-pow2(DOC/500) - 1): healthy trees stay <= 2x nominal (p95 exactly
# 2x; largest successful trees ~2x), every tree past ~3.5x in the 16w long evals failed (435-2000 agents) — so 4x never
# cuts a healthy tree and caps a runaway's cost. Override with MAX_NODES=... if needed.
_n=$(( (DOC + 499) / 500 )); _p=1; while [ $_p -lt $_n ]; do _p=$(( _p * 2 )); done
MAX_NODES=${MAX_NODES:-$(( 4 * (2 * _p - 1) ))}
for attempt in $(seq 1 20); do
  CKPT=$CKPT OOLONG_BASE=2000000 EVAL_TASKS=oolong_counting,oolong_user,oolong_temporal \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal N_PER_TASK=10 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=$DOC AGENT_CONTEXT=8000 MAX_CHUNK_TOKENS=200000 MAX_DEPTH=none \
TEMP=${TEMP:-0.2} MAX_NODES=$MAX_NODES OUT=/tmp/eval_$TAG PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval $TAG attempt $attempt failed at $(date +%H:%M:%S); retrying in 3 min ====="
  sleep 180
done
echo "===== $TAG DONE $(date +%H:%M:%S) ====="
