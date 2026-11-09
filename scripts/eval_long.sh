#!/usr/bin/env bash
# FULL held-out eval at a chosen doc length / per-agent budget, on a FRESH seed block (3_000_000+, never
# trained on or read): all 13 RULER tasks (5 seeds each) + all 3 OOLONG-synth families over ALL 10 validated
# datasets (N=20 per family -> 2 per dataset; oolong_spec round-robins datasets by index).
#   usage: scripts/eval_long.sh <tag> <ckpt> <doc_tokens> <budget> [ruler_n] [oolong_n]
#   e.g.:  scripts/eval_long.sh L16w_8k tinker://.../sft_general16w 8000 5000
# EVAL_TASKS order fixes the seeds — keep it stable across lengths/checkpoints so results are comparable.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
TAG=$1; CKPT=$2; DOC=$3; BUDGET=$4; RN=${5:-5}; ON=${6:-20}
TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2
# MAX_NODES = 4 x the nominal binary tree (2 x next-pow2(DOC/500) - 1): healthy trees stay <= 2x nominal (p95 exactly
# 2x; largest successful trees ~2x), every tree past ~3.5x in the 16w long evals failed (435-2000 agents) — so 4x never
# cuts a healthy tree and caps a runaway's cost. Override with MAX_NODES=... if needed.
_n=$(( (DOC + 499) / 500 )); _p=1; while [ $_p -lt $_n ]; do _p=$(( _p * 2 )); done
MAX_NODES=${MAX_NODES:-$(( 4 * (2 * _p - 1) ))}
for attempt in $(seq 1 20); do
  CKPT=$CKPT SEED_OFFSET=3000000 OOLONG_BASE=3000000 EVAL_TASKS=$TASKS SCORE_TASKS=$TASKS \
N_PER_TASK=$RN N_PER_TASK_OVERRIDE=oolong_counting:$ON,oolong_user:$ON,oolong_temporal:$ON \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=$DOC AGENT_CONTEXT=$BUDGET MAX_CHUNK_TOKENS=200000 \
TEMP=${TEMP:-0.2} MAX_NODES=$MAX_NODES MAX_DEPTH=none OUT=/tmp/eval_$TAG \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval $TAG attempt $attempt failed at $(date +%H:%M:%S); retrying in 3 min ====="
  sleep 180
done
echo "===== $TAG DONE $(date +%H:%M:%S) ====="
