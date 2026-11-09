#!/usr/bin/env bash
# Held-out eval on a FRESH seed block at a chosen doc length / budget.
#   usage: scripts/eval_len.sh <tag> <ckpt> <doc_tokens> <budget> [n_per_task]
#   e.g.:  scripts/eval_len.sh A7 tinker://.../sft_general7 10000 3000
# Runs the 9 held-out tasks (same order as the scoreboard) on seeds 3_000_000+ (never read).
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
TAG=$1; CKPT=$2; DOC=$3; BUDGET=$4; N=${5:-5}
for attempt in $(seq 1 20); do
  CKPT=$CKPT SEED_OFFSET=3000000 OOLONG_BASE=3000000 \
EVAL_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=$N BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=$DOC AGENT_CONTEXT=$BUDGET MAX_CHUNK_TOKENS=200000 \
TEMP=0.2 MAX_NODES=800 MAX_DEPTH=none OUT=/tmp/eval_$TAG \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval $TAG attempt $attempt failed at $(date +%H:%M:%S); retrying in 3 min ====="
  sleep 180
done
echo "===== $TAG DONE $(date +%H:%M:%S) ====="
