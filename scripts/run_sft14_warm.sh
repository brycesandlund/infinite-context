#!/usr/bin/env bash
# Run 14: WARM START FROM RUN 13 (the from-base checkpoint) to install the fold/search prior that
# one pass from base does not. Run 13 folds 2/2 in-distribution (vt_novel) but went BINARY 5/5 on
# RULER vt — a RECOGNITION failure, not a capability one: every training task ships a system-prompt
# document description ("...assignment lines hidden inside it, in order") and every RULER eval task
# ships NONE. Two levers:
#   1. CONTEXT_DROP=0.5 — half the PROSE traces (vt_novel, niah_novel, niah_multi, narrativeqa,
#      realdoc_count) train with NO description, so the root must read order-dependence off the
#      QUESTION. Format diversity in TRAINING, not an eval-time guard.
#   1b. QUESTION_PLACEHOLDER_FRAC=0.3 — our OWN harness boilerplate ("[The relevant text is in a
#      separate document accessible via the read_chunk tool...] (Document length: N tokens.)") is on
#      100% of RULER eval questions (25/25) and was on 0% of 2,500 training questions.
#   2. Fold mass 24% -> ~31% (vt_novel 150->200, synth fold 10->20 each) — the user's lever, kept
#      because it is cheap and complementary. NOTE: fold SHARE was never the difference (run 12
#      corpus 24.4% fold, run 13 corpus 22.7%).
#   nohup caffeinate -is bash scripts/run_sft14_warm.sh > /tmp/sft14_eval.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
INIT=${INIT_CHECKPOINT:-tinker://bb67e67e-7d0f-5bf1-adaf-7022ee1f0d8f:train:0/weights/sft_general13}

echo "===== SFT (sft_general14w, warm start from $INIT) STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 200); do
  if INIT_CHECKPOINT=$INIT LR=5e-6 CONTEXT_DROP=0.5 QUESTION_PLACEHOLDER_FRAC=0.3 \
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,synth_topk,long_records,rule_label,labeled_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=10 N_PER_TASK_OVERRIDE=vt_novel:200,synth_runreset:20,synth_varchain:20,synth_peak:20,synth_streak:20,synth_adjacent:20,synth_first_exceed:20,labeled_records:240,long_records:120,synth_topk:40,rule_label:40,realdoc_count:40,niah_novel:40,niah_multi:40,narrativeqa:30,synth_2d:24,synth_filter_argmax:15 \
CTX=3000 DOC=6000 DOC_MIX=6000:2,8400:1,14000:1 DOC_MIX_OVERRIDE="vt_novel=6000:1,14000:1" CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 DATUM_MODE=agent INTERNAL_KEEP=1.0 SFT_BATCH_SIZE=16 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general14w \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general14w_traces \
PYTHONPATH=. uv run python sft.py; then break; fi
  echo "===== SFT attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min (resumes from checkpoint) ====="
  sleep 300
done

# Task ORDER is load-bearing: eval/run.py seeds a task by its position in EVAL_TASKS (same list as runs 6-13).
echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 20); do
  CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general14w \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min ====="
  sleep 300
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
