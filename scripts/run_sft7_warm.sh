#!/usr/bin/env bash
# Run 7 as a WARM START from sft_general6: train on the new/changed tasks (rule_label, vt_novel with
# empty-slice fold, key-space contracts) plus a ~20% replay of everything else, then run the eval.
#   nohup caffeinate -is bash scripts/run_sft7_warm.sh > /tmp/sft7_eval.log 2>&1 &
# Full from-base runs remain the reference (scripts/run_sft7_eval.sh); this is the fast iteration path.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
INIT=${INIT_CHECKPOINT:-tinker://7161d592-9d8a-5583-9ff1-13a273154e96:train:0/weights/sft_general6}

echo "===== SFT (sft_general7w2, warm start from $INIT) STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 200); do
  if INIT_CHECKPOINT=$INIT LR=5e-6 \
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,long_records,rule_label,labeled_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=8 N_PER_TASK_OVERRIDE=rule_label:160,labeled_records:160,vt_novel:120,synth_2d:16,synth_filter_argmax:12,long_records:32,realdoc_count:32,niah_novel:20,niah_multi:24,narrativeqa:30 \
CTX=3000 DOC=6000 DOC_MIX=6000:2,8400:1,14000:1 CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 DATUM_MODE=agent INTERNAL_KEEP=1.0 SFT_BATCH_SIZE=8 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general7w2 \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general7w2_traces \
PYTHONPATH=. uv run python sft.py; then break; fi
  echo "===== SFT attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min (resumes from checkpoint) ====="
  sleep 300
done

# Task ORDER is load-bearing: eval/run.py seeds a task by its position in EVAL_TASKS, so the first 19
# entries keep the run-3..6 order (identical problems for seeds 0-4); dropped in-dist tasks get N=0.
echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 20); do
  CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general7w2 \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min ====="
  sleep 300
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
