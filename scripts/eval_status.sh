#!/usr/bin/env bash
# Progress of running evals: rollouts done / total, elapsed, mean score so far (from eval.run's [progress] lines).
#   bash scripts/eval_status.sh C18w_320k R18w_160k_b8k ...
for t in "$@"; do
  L=/tmp/eval_$t.log
  if grep -q "^SCORE" "$L" 2>/dev/null; then echo "$t: FINISHED — $(grep -o '^SCORE.*' $L | tail -1 | grep -o '[0-9.]*  (.*')"; continue; fi
  last=$(grep "^\[progress\]" "$L" 2>/dev/null | tail -1)
  if [ -z "$last" ]; then echo "$t: running, no rollout finished yet (log $(wc -l < $L) lines)"; continue; fi
  n=$(grep -c "^\[progress\]" "$L"); tot=$(echo "$last" | sed -E 's/.* ([0-9]+)\/([0-9]+) done.*/\2/'); min=$(echo "$last" | sed -E 's/.*\| ([0-9.]+) min$/\1/')
  # TASK mean (average of per-task means — how RULER-13 / chart scores are reported), not the per-rollout mean
  mean=$(grep "^\[progress\]" "$L" | sed -E 's/.*done \| ([a-z_0-9]+) seed .* score ([0-9.]+).*/\1 \2/' \
         | awk '{s[$1]+=$2; c[$1]++} END{for (k in s){m+=s[k]/c[k]; n++}; printf "%.3f over %d tasks", m/n, n}')
  echo "$t: $n/$tot done, ${min} min elapsed, task-mean score so far $mean"
done
