#!/usr/bin/env bash
# Run 5 SFT -> eval chain. Launch from zsh with API keys in env (source ~/.zshrc first):
#   nohup caffeinate -is bash scripts/run_sft5_eval.sh > /tmp/sft5_eval.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "===== SFT (sft_general5) STARTING $(date +%H:%M:%S) ====="
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_filter_argmax,synth_2d,long_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=20 N_PER_TASK_OVERRIDE=realdoc_count:80,niah_novel:80,niah_multi:80,vt_novel:80,narrativeqa:80,long_records:80 SYNTH_STRATEGY=both \
CTX=3000 DOC=6000 DOC_MIX=6000:3,14000:1 CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 SCALAR_FOLD_FRAC=0.25 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general5 \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general5_traces \
PYTHONPATH=. uv run python sft.py

echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
BACKEND=tinker MODE=decompose N_PER_TASK=3 DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general5 \
PYTHONPATH=. uv run python -m eval.run
echo "===== ALL DONE $(date +%H:%M:%S) ====="
