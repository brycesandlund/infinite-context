#!/usr/bin/env bash
# Competitor single-shot on the June OOLONG chart problems (OOLONG_BASE=2000000, 10/family = 1 per dataset), whole
# document in context, temperature 0, current eval/run.py MODE=single path.
#   usage: [TEMP=none] scripts/competitor_oolong_chart.sh <litellm-model> <tag> <doc lengths...>   (TEMP=none: models that reject temperature, e.g. claude-fable-5-1)
#   e.g.:  scripts/competitor_oolong_chart.sh openai/gpt-5.4-2026-03-05 gpt5_4 10000 20000 40000 80000
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
MODEL=$1; TAG=$2; shift 2
mkdir -p eval_results/competitor
for DOC in "$@"; do
  K=$((DOC / 1000))k
  BACKEND=$MODEL MODE=single TEMP=${TEMP:-0} OOLONG_BASE=2000000 \
  EVAL_TASKS=oolong_counting,oolong_user,oolong_temporal SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal \
  N_PER_TASK=10 DOC_SIZE_TOKENS=$DOC OUT=eval_results/competitor/${TAG}_single_chart_$K \
  PYTHONPATH=. uv run python -m eval.run > eval_results/competitor/${TAG}_single_chart_$K.log 2>&1
  echo "$TAG $K: $(grep -o '^SCORE.*' eval_results/competitor/${TAG}_single_chart_$K.log | tail -1)"
done
echo "DONE $(date +%H:%M:%S)"
