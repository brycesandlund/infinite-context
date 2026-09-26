#!/usr/bin/env bash
# Everything that runs after run 19 (scripts/run_sft19_warm.sh) finishes — one watcher, so nothing depends on a session.
#   1. wait for run 19's "ALL DONE"; pin the 19w checkpoint (last_sft_checkpoint.txt is shared with other runs, e.g. 18b)
#   2. archive the 4K scoreboard eval
#   3. plain leaf-classification probe
#   4. OOLONG chart @10K/@40K/@80K + RULER-13 @10K/@40K/@80K (8K budget; eval_long OOLONG count 0 — RULER seeds unchanged)
#   5. LLM-judge the open-ended QA rollouts
#   nohup caffeinate -is bash scripts/after_run19.sh > /tmp/after_run19.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=/tmp/sft19_eval.log
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for run 19 to finish ($LOG)"
until grep -q "===== ALL DONE" "$LOG" 2>/dev/null; do sleep 120; done
CK=$(grep -o "tinker://[^ ]*sft_general19w[^ ]*" "$LOG" | tail -1)
[ -z "$CK" ] && CK=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt)
case "$CK" in *sft_general19w*) ;; *) say "ABORT: checkpoint is not 19w: $CK"; exit 1;; esac
echo "$CK" > ~/.cache/infinite-context/ckpt_sft_general19w.txt
say "19w checkpoint: $CK"
cp /tmp/eval_general19w.jsonl eval_results/raw/sft_general19w.jsonl 2>/dev/null
cp /tmp/eval_general19w.txt eval_results/raw/sft_general19w.txt 2>/dev/null
cp "$LOG" eval_results/raw/sft_general19w.log
say "19w 4K SCORE-9: $(grep -o '^SCORE (held-out.*' "$LOG" | tail -1)"

say "plain leaf-classification probe"
CKPT=$CK TAG=19w PYTHONPATH=. uv run python scripts/leaf_classify_probe.py 6 > eval_results/probe_19w.log 2>&1

run_set() {   # $1 = doc length tag (10k/40k/80k), $2 = doc tokens
  bash scripts/eval_oolong_chart.sh C19w_$1 "$CK" $2 > /tmp/eval_C19w_$1.log 2>&1 &
  bash scripts/eval_long.sh R19w_$1_b8k "$CK" $2 8000 5 0 > /tmp/eval_R19w_$1_b8k.log 2>&1 &
}
say "evals @10K and @40K"
run_set 10k 10000; run_set 40k 40000; wait
say "evals @80K"
run_set 80k 80000; wait
for t in C19w_10k C19w_40k C19w_80k R19w_10k_b8k R19w_40k_b8k R19w_80k_b8k; do
  cp /tmp/eval_$t.jsonl eval_results/raw/sft_general19w_$t.jsonl 2>/dev/null
  cp /tmp/eval_$t.txt eval_results/raw/sft_general19w_$t.txt 2>/dev/null
  cp /tmp/eval_$t.log eval_results/raw/sft_general19w_$t.log 2>/dev/null
  say "$t: $(grep -o '^SCORE.*' /tmp/eval_$t.log | tail -1)"
done

say "LLM-judging open-ended QA"
PYTHONPATH=. uv run python scripts/judge_open_qa.py eval_results/raw/sft_general19w.jsonl \
  eval_results/raw/sft_general19w_R19w_10k_b8k.jsonl eval_results/raw/sft_general19w_R19w_40k_b8k.jsonl \
  eval_results/raw/sft_general19w_R19w_80k_b8k.jsonl > eval_results/qa_judge_19w.txt 2>&1
say "ALL 19w EVALS DONE"
