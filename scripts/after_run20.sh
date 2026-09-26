#!/usr/bin/env bash
# Everything that runs after run 20 (scripts/run_sft20_warm.sh) finishes — one watcher, so nothing depends on a session.
#   1. wait for run 20's "ALL DONE"; pin the 20w checkpoint (last_sft_checkpoint.txt is shared with other runs, e.g. 18b)
#   2. archive the 4K scoreboard eval
#   3. plain leaf-classification probe
#   4. root-contract probe (19w vs 20w); OOLONG chart + RULER-13 @10K/@40K (8K budget; no 80K)
#   5. LLM-judge the open-ended QA rollouts
#   nohup caffeinate -is bash scripts/after_run20.sh > /tmp/after_run20.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG=/tmp/sft20_eval.log
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for run 20 to finish ($LOG)"
until grep -q "===== ALL DONE" "$LOG" 2>/dev/null; do sleep 120; done
CK=$(grep -o "tinker://[^ ]*sft_general20w[^ ]*" "$LOG" | tail -1)
[ -z "$CK" ] && CK=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt)
case "$CK" in *sft_general20w*) ;; *) say "ABORT: checkpoint is not 20w: $CK"; exit 1;; esac
echo "$CK" > ~/.cache/infinite-context/ckpt_sft_general20w.txt
say "20w checkpoint: $CK"
cp /tmp/eval_general20w.jsonl eval_results/raw/sft_general20w.jsonl 2>/dev/null
cp /tmp/eval_general20w.txt eval_results/raw/sft_general20w.txt 2>/dev/null
cp "$LOG" eval_results/raw/sft_general20w.log
say "20w 4K SCORE-9: $(grep -o '^SCORE (held-out.*' "$LOG" | tail -1)"

say "root-contract probe (OOLONG user @10K, 200 questions) — the v18 target"
CKPTS="19w=$(cat ~/.cache/infinite-context/ckpt_sft_general19w.txt),20w=$CK" PYTHONPATH=. uv run python scripts/root_contract_probe.py 200 10000 > eval_results/root_contract_probe_20w.log 2>&1
say "$(grep -E '^(19w|20w):' eval_results/root_contract_probe_20w.log)"
say "plain leaf-classification probe"
CKPT=$CK TAG=20w PYTHONPATH=. uv run python scripts/leaf_classify_probe.py 6 > eval_results/probe_20w.log 2>&1

run_set() {   # $1 = doc length tag (10k/40k/80k), $2 = doc tokens
  bash scripts/eval_oolong_chart.sh C20w_$1 "$CK" $2 > /tmp/eval_C20w_$1.log 2>&1 &
  bash scripts/eval_long.sh R20w_$1_b8k "$CK" $2 8000 5 0 > /tmp/eval_R20w_$1_b8k.log 2>&1 &
}
say "OOLONG chart + RULER-13 @10K and @40K (8K budget; no 80K)"
run_set 10k 10000; run_set 40k 40000; wait
for t in C20w_10k C20w_40k R20w_10k_b8k R20w_40k_b8k; do
  cp /tmp/eval_$t.jsonl eval_results/raw/sft_general20w_$t.jsonl 2>/dev/null
  cp /tmp/eval_$t.txt eval_results/raw/sft_general20w_$t.txt 2>/dev/null
  cp /tmp/eval_$t.log eval_results/raw/sft_general20w_$t.log 2>/dev/null
  say "$t: $(grep -o '^SCORE.*' /tmp/eval_$t.log | tail -1)"
done

say "LLM-judging open-ended QA"
PYTHONPATH=. uv run python scripts/judge_open_qa.py eval_results/raw/sft_general20w.jsonl \
  eval_results/raw/sft_general20w_R20w_10k_b8k.jsonl eval_results/raw/sft_general20w_R20w_40k_b8k.jsonl > eval_results/qa_judge_20w.txt 2>&1
say "ALL 20w EVALS DONE"
