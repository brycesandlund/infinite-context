#!/usr/bin/env bash
# Run 6 SFT -> eval chain. Launch from zsh with API keys in env (source ~/.zshrc first):
#   nohup caffeinate -is bash scripts/run_sft6_eval.sh > /tmp/sft6_eval.log 2>&1 &
# Strategy is a property of the task now (sequential -> left_fold, else binary); no SYNTH_STRATEGY knob.
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "===== SFT (sft_general6) STARTING $(date +%H:%M:%S) ====="
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,long_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=40 N_PER_TASK_OVERRIDE=realdoc_count:160,niah_novel:100,niah_multi:120,vt_novel:200,narrativeqa:80,long_records:160,synth_2d:80,synth_filter_argmax:60 \
CTX=3000 DOC=6000 DOC_MIX=6000:3,14000:1 CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general6 \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general6_traces \
PYTHONPATH=. uv run python sft.py

echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
# Task ORDER is load-bearing: eval/run.py seeds a task by its position in EVAL_TASKS, so the first 19
# entries keep the run-3/4/5 order (identical problems for seeds 0-2); dropped in-dist tasks get N=0.
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general6 \
PYTHONPATH=. uv run python -m eval.run
echo "===== ALL DONE $(date +%H:%M:%S) ====="
