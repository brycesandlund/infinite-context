#!/usr/bin/env bash
# Run 20 (QUICK): warm from 19w on cache v18 — label-free user questions (author_count x2, rank most/second/third, noun
# author|user, label-free leaves). HALF the run-19 mix (base 5/task, every override halved, labeled_records 240 -> 120 so it
# stays ~20% of the corpus), batch 16 -> 64 and LR 5e-6 -> 1e-5 (sqrt batch scaling) to fit ~2 h. First warm run off the
# batch-16 / 5e-6 recipe of runs 14-19.
#   Dry run:   INIT_CHECKPOINT=x TRAIN=0 bash scripts/run_sft20_warm.sh
#   Real run:  INIT_CHECKPOINT=<19w> nohup caffeinate -is bash scripts/run_sft20_warm.sh > /tmp/sft20_eval.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
INIT=${INIT_CHECKPOINT:?set INIT_CHECKPOINT (19w)}
TRAIN=${TRAIN:-1}

echo "===== SFT (sft_general20w, warm start from $INIT, TRAIN=$TRAIN) STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 200); do
  if TRAIN=$TRAIN INIT_CHECKPOINT=$INIT LR=1e-5 CONTEXT_DROP=0.5 QUESTION_PLACEHOLDER_FRAC=0.3 \
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,synth_topk,long_records,rule_label,labeled_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,niah_bridge \
N_PER_TASK=5 N_PER_TASK_OVERRIDE=vt_novel:75,niah_bridge:40,synth_runreset:10,synth_varchain:10,synth_peak:10,synth_streak:10,synth_adjacent:10,synth_first_exceed:10,labeled_records:120,long_records:60,synth_topk:60,rule_label:20,realdoc_count:20,niah_novel:30,niah_multi:30,narrativeqa:15,synth_2d:12,synth_filter_argmax:8 \
CTX=3000 DOC=6000 BUDGET_MIX=3000:50,5000:15,8000:13,10000:11,12000:11 ROUND_DOC_FRAC=0.2 CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 DATUM_MODE=agent INTERNAL_KEEP=1.0 SFT_BATCH_SIZE=64 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general20w \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general20w_traces \
PYTHONPATH=. uv run python sft.py; then break; fi
  echo "===== SFT attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min (resumes from checkpoint) ====="
  sleep 300
done
if [ "$TRAIN" != "1" ]; then echo "===== DRY RUN DONE $(date +%H:%M:%S) (traces in /tmp/sft_general20w_traces) ====="; exit 0; fi

echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 20); do
  CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk,niah_single_2,niah_single_3,niah_multikey_2,niah_multikey_3,niah_multivalue,qa_1,qa_2 \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0,niah_single_2:5,niah_single_3:5,niah_multikey_2:5,niah_multikey_3:5,niah_multivalue:5,qa_1:5,qa_2:5 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general20w \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min ====="
  sleep 300
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
