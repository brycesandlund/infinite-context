#!/usr/bin/env bash
# Everything that runs after run 18 (scripts/run_sft18_base.sh, from base) finishes — one watcher, so nothing depends on a
# session being alive.
#   1. wait for run 18's "ALL DONE"; pin the 18b checkpoint (last_sft_checkpoint.txt is overwritten by any later run)
#   2. archive the 4K eval; read SCORE-9
#   4. 18b evals (): plain leaf-classification probe; OOLONG chart @10K/@40K (8K budget);
#      RULER-13 + OOLONG @10K/@40K (8K budget) — the same protocols 17w was measured with.
#   nohup caffeinate -is bash scripts/after_run18b.sh > /tmp/after_run18.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
LOG18=/tmp/sft18b_eval.log
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for run 18 to finish ($LOG18)"
until grep -q "===== ALL DONE" "$LOG18" 2>/dev/null; do sleep 120; done
CK=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt)
case "$CK" in *sft_general18b*) ;; *) say "ABORT: last checkpoint is not 18b: $CK"; exit 1;; esac
echo "$CK" > ~/.cache/infinite-context/ckpt_sft_general18b.txt
say "18b checkpoint: $CK"
cp /tmp/eval_general18b.jsonl eval_results/raw/sft_general18b.jsonl 2>/dev/null
cp /tmp/eval_general18b.txt eval_results/raw/sft_general18b.txt 2>/dev/null
cp "$LOG18" eval_results/raw/sft_general18b.log

SCORE=$(grep -o "^SCORE (held-out.*): [0-9.]*" "$LOG18" | tail -1 | grep -o "[0-9.]*$")
say "18b 4K SCORE-9 = ${SCORE:-MISSING}"

say "18b plain leaf-classification probe"
CKPT=$CK TAG=18b PYTHONPATH=. uv run python scripts/leaf_classify_probe.py 6 > eval_results/probe_18b.log 2>&1

say "18b OOLONG chart @10K / @40K and RULER-13+OOLONG @10K / @40K (8K budget)"
bash scripts/eval_oolong_chart.sh C18b_10k "$CK" 10000 > /tmp/eval_C18b_10k.log 2>&1 &
bash scripts/eval_oolong_chart.sh C18b_40k "$CK" 40000 > /tmp/eval_C18b_40k.log 2>&1 &
bash scripts/eval_long.sh R18b_10k_b8k "$CK" 10000 8000 5 0 > /tmp/eval_R18b_10k_b8k.log 2>&1 &
bash scripts/eval_long.sh R18b_40k_b8k "$CK" 40000 8000 5 0 > /tmp/eval_R18b_40k_b8k.log 2>&1 &
wait
for t in C18b_10k C18b_40k R18b_10k_b8k R18b_40k_b8k; do
  cp /tmp/eval_$t.jsonl eval_results/raw/sft_general18b_$t.jsonl 2>/dev/null
  cp /tmp/eval_$t.txt eval_results/raw/sft_general18b_$t.txt 2>/dev/null
  cp /tmp/eval_$t.log eval_results/raw/sft_general18b_$t.log 2>/dev/null
  say "$t: $(grep -o '^SCORE.*' /tmp/eval_$t.log | tail -1)"
done
say "ALL 18b EVALS DONE"
