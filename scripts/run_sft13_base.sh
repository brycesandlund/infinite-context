#!/usr/bin/env bash
# Run 13 FROM BASE on the run-12 oracle text at run-7 scale: the single-stage reference for the paper and the
# corpus ablation against corrected run 7 (0.662; same recipe, pre-audit corpus, batch 8). Batch 16, constant LR
# 1e-5 (sft.py default), 1 epoch. ~2,500 traces / ~170M tokens / ~13 h.
#   nohup caffeinate -is bash scripts/run_sft13_base.sh > /tmp/sft13_eval.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "===== SFT (sft_general13, FROM BASE) STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 200); do
  if SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,synth_topk,long_records,rule_label,labeled_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=40 N_PER_TASK_OVERRIDE=labeled_records:400,vt_novel:300,synth_topk:120,long_records:160,rule_label:160,realdoc_count:160,niah_multi:120,niah_novel:100,narrativeqa:80,synth_2d:80,synth_filter_argmax:60 \
CTX=3000 DOC=6000 DOC_MIX=6000:2,8400:1,14000:1 DOC_MIX_OVERRIDE="vt_novel=6000:1,14000:1" CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 DATUM_MODE=agent INTERNAL_KEEP=1.0 SFT_BATCH_SIZE=16 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general13 \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general13_traces \
PYTHONPATH=. uv run python sft.py; then break; fi
  echo "===== SFT attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min (resumes from checkpoint) ====="
  sleep 300
done

# Task ORDER is load-bearing: eval/run.py seeds a task by its position in EVAL_TASKS (same list as runs 6-12).
echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 20); do
  CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general13 \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min ====="
  sleep 300
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
