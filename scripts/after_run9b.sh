#!/usr/bin/env bash
# The 18b eval suite (scripts/run_sft18_base.sh eval block + scripts/after_run18b.sh) against a model served by our own
# vLLM server (offtinker/README.md), for the off-Tinker Qwen3.5-9B run on the 18b corpus.
#   usage: scripts/after_run9b.sh <name> <vllm url>       e.g. after_run9b.sh 9b_18b https://...modal.run
#   nohup caffeinate -is bash scripts/after_run9b.sh 9b_18b <url> > /tmp/after_run9b.log 2>&1 &
# Same tasks, seeds, budgets, temperature and MAX_NODES as 18b; only BACKEND differs. (The plain leaf-classification
# probe is Tinker-only and is not run here.)
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
NAME=$1
export BACKEND=vllm VLLM_URL=$2 CONCURRENCY=${CONCURRENCY:-8}
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "$NAME 4K eval (3K budget)"
for attempt in $(seq 1 20); do
  EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk,niah_single_2,niah_single_3,niah_multikey_2,niah_multikey_3,niah_multivalue,qa_1,qa_2 \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0,niah_single_2:5,niah_single_3:5,niah_multikey_2:5,niah_multikey_3:5,niah_multivalue:5,qa_1:5,qa_2:5 \
MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general$NAME \
PYTHONPATH=. uv run python -m eval.run > /tmp/eval_general$NAME.log 2>&1 && break
  say "4K eval attempt $attempt failed; retrying in 3 min"
  sleep 180
done
cp /tmp/eval_general$NAME.jsonl eval_results/offtinker/sft_general$NAME.jsonl 2>/dev/null
cp /tmp/eval_general$NAME.txt eval_results/offtinker/sft_general$NAME.txt 2>/dev/null
cp /tmp/eval_general$NAME.log eval_results/offtinker/sft_general$NAME.log 2>/dev/null
say "$NAME 4K: $(grep -o '^SCORE.*' /tmp/eval_general$NAME.log | tail -1)"

say "$NAME OOLONG chart @10K / @40K and RULER-13 @10K / @40K (8K budget)"
bash scripts/eval_oolong_chart.sh C${NAME}_10k "" 10000 > /tmp/eval_C${NAME}_10k.log 2>&1 &
bash scripts/eval_oolong_chart.sh C${NAME}_40k "" 40000 > /tmp/eval_C${NAME}_40k.log 2>&1 &
bash scripts/eval_long.sh R${NAME}_10k_b8k "" 10000 8000 5 0 > /tmp/eval_R${NAME}_10k_b8k.log 2>&1 &
bash scripts/eval_long.sh R${NAME}_40k_b8k "" 40000 8000 5 0 > /tmp/eval_R${NAME}_40k_b8k.log 2>&1 &
wait
for t in C${NAME}_10k C${NAME}_40k R${NAME}_10k_b8k R${NAME}_40k_b8k; do
  cp /tmp/eval_$t.jsonl eval_results/offtinker/sft_general${NAME}_$t.jsonl 2>/dev/null
  cp /tmp/eval_$t.txt eval_results/offtinker/sft_general${NAME}_$t.txt 2>/dev/null
  cp /tmp/eval_$t.log eval_results/offtinker/sft_general${NAME}_$t.log 2>/dev/null
  say "$t: $(grep -o '^SCORE.*' /tmp/eval_$t.log | tail -1)"
done
say "LLM-judging open-ended QA"
PYTHONPATH=. uv run python scripts/judge_open_qa.py eval_results/offtinker/sft_general$NAME.jsonl \
  eval_results/offtinker/sft_general${NAME}_R${NAME}_10k_b8k.jsonl eval_results/offtinker/sft_general${NAME}_R${NAME}_40k_b8k.jsonl \
  > eval_results/offtinker/qa_judge_$NAME.txt 2>&1
say "ALL $NAME EVALS DONE"
