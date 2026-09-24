#!/usr/bin/env bash
# Run 16: run-15 corpus (cache v9 fixes) + BUDGET x LENGTH JITTER. The split rule itself is unchanged (midpoint,
# "less than 500 tokens"). Run 15w read "Range 2000..4000" as a leaf with P=1.0 — the one-token split decision had
# been learned from a corpus with ONE budget (3000) and three doc sizes (6000/8400/14000), i.e. a few range-size
# families, not the rule (trace_snippets/run15_split_decision_probe.txt). Each problem now draws:
#   budget (share)  doc length, triangular with the mode at the LOW end (mean doc > budget; ceiling 14000 = old max)
#   3000  (50%)     4000-14000   mean ~7300
#   5000  (15%)     5000-14000   mean ~8000
#   8000  (13%)     6000-14000   mean ~8700
#   10000 (11%)     9000-14000   mean ~10700
#   12000 (11%)    11500-14000   mean ~12300
# Overall mean ~8.5K and ~54% of docs >= 8000 (run 15: 50%), so agents/trace and the root share of datums stay
# about where they were. 20% of doc targets rounded to a multiple of 1000 (ROUND_DOC_FRAC). narrativeqa keeps
# doc 6000 / budget 3000 (paid leaf calls). EVAL unchanged: budget 3000, doc 4000, same task list and seeds.
#
#   Dry run (traces only, no Tinker):  TRAIN=0 bash scripts/run_sft16_warm.sh
#   Real run:  INIT_CHECKPOINT=<ckpt> nohup caffeinate -is bash scripts/run_sft16_warm.sh > /tmp/sft16_eval.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
INIT=${INIT_CHECKPOINT:?set INIT_CHECKPOINT (14w or 15w — undecided)}
TRAIN=${TRAIN:-1}

echo "===== SFT (sft_general16w, warm start from $INIT, TRAIN=$TRAIN) STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 200); do
  if TRAIN=$TRAIN INIT_CHECKPOINT=$INIT LR=5e-6 CONTEXT_DROP=0.5 QUESTION_PLACEHOLDER_FRAC=0.3 \
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,synth_topk,long_records,rule_label,labeled_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=10 N_PER_TASK_OVERRIDE=vt_novel:150,labeled_records:240,long_records:120,synth_topk:40,rule_label:40,realdoc_count:40,niah_novel:60,niah_multi:60,narrativeqa:30,synth_2d:24,synth_filter_argmax:15 \
CTX=3000 DOC=6000 BUDGET_MIX=3000:50,5000:15,8000:13,10000:11,12000:11 ROUND_DOC_FRAC=0.2 CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 DATUM_MODE=agent INTERNAL_KEEP=1.0 SFT_BATCH_SIZE=16 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general16w \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general16w_traces \
PYTHONPATH=. uv run python sft.py; then break; fi
  echo "===== SFT attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min (resumes from checkpoint) ====="
  sleep 300
done
if [ "$TRAIN" != "1" ]; then echo "===== DRY RUN DONE $(date +%H:%M:%S) (traces in /tmp/sft_general16w_traces) ====="; exit 0; fi

echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 20); do
  CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk,niah_single_2,niah_single_3,niah_multikey_2,niah_multikey_3,niah_multivalue,qa_1,qa_2 \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0,niah_single_2:5,niah_single_3:5,niah_multikey_2:5,niah_multikey_3:5,niah_multivalue:5,qa_1:5,qa_2:5 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general16w \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min ====="
  sleep 300
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
