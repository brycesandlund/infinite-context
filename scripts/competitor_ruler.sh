#!/usr/bin/env bash
# Competitor single-shot on the RULER-13 set our fine-tunes are scored on (scripts/eval_long.sh): SAME task order (the
# three OOLONG tasks kept at 0 problems so every RULER task keeps its seed index), SEED_OFFSET=3000000, 5 per task.
# Whole document in context, one tool-free call.
#   usage: [TEMP=0] [SINGLE_CONTEXT=65536] [OUT_TOKENS=...] [REASONING_EFFORT=...] \
#          scripts/competitor_ruler.sh <model> <tag> <doc lengths...>
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODEL=$1; TAG=$2; shift 2
TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2
SCORE=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2,niah_multikey_3,niah_multivalue,niah_multiquery,vt,cwe,fwe,qa_1,qa_2
mkdir -p eval_results/competitor
for DOC in "$@"; do
  K=$((DOC / 1000))k
  BACKEND=$MODEL MODE=single TEMP=${TEMP:-0} SEED_OFFSET=3000000 EVAL_TASKS=$TASKS SCORE_TASKS=$SCORE \
  N_PER_TASK=5 N_PER_TASK_OVERRIDE=oolong_counting:0,oolong_user:0,oolong_temporal:0 DOC_SIZE_TOKENS=$DOC \
  OUT=eval_results/competitor/${TAG}_single_ruler_$K \
  PYTHONPATH=. uv run python -m eval.run > eval_results/competitor/${TAG}_single_ruler_$K.log 2>&1
  echo "$TAG $K: $(grep -o '^SCORE.*' eval_results/competitor/${TAG}_single_ruler_$K.log | tail -1)"
done
echo "DONE $(date +%H:%M:%S)"
