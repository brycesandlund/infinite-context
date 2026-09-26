#!/usr/bin/env bash
# Progress of running evals: rollouts done / total, elapsed, mean score so far (from eval.run's [progress] lines).
#   bash scripts/eval_status.sh C18w_320k R18w_160k_b8k ...
for t in "$@"; do
  L=/tmp/eval_$t.log
  if grep -q "^SCORE" "$L" 2>/dev/null; then echo "$t: FINISHED — $(grep -o '^SCORE.*' $L | tail -1 | grep -o '[0-9.]*  (.*')"; continue; fi
  last=$(grep "^\[progress\]" "$L" 2>/dev/null | tail -1)
  if [ -z "$last" ]; then echo "$t: running, no rollout finished yet (log $(wc -l < $L) lines)"; continue; fi
  n=$(grep -c "^\[progress\]" "$L"); tot=$(echo "$last" | sed -E 's/.* ([0-9]+)\/([0-9]+) done.*/\2/'); min=$(echo "$last" | sed -E 's/.*\| ([0-9.]+) min$/\1/')
  mean=$(grep "^\[progress\]" "$L" | sed -E 's/.*score ([0-9.]+).*/\1/' | awk '{s+=$1} END{printf "%.3f", s/NR}')
  echo "$t: $n/$tot done, ${min} min elapsed, mean score so far $mean"
done
