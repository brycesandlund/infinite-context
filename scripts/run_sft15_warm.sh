#!/usr/bin/env bash
# Run 15: WARM START FROM RUN 14w, run-14 recipe unchanged except the corpus fixes from the full-RULER
# audit (cache v9, commit a9b208d):
#   1. niah_multi FILTERS when the question names the key(s) (explicit/multiquery/multivalue): leaves keep
#      only the asked keys and list every other key's fact as `(other key, skip)`. The old
#      "(collecting every stated key=… fact in the range)" hedge made RULER niah_multikey_3 (every sentence
#      a uuid=uuid needle) overflow 4/5 on 14w — including the leaf that held the answer. Hidden mode
#      (lookup instruction in the document) still collects everything.
#   2. niah_novel gets uuid keys/values: uuid keys existed ONLY under niah_multi's collect template, so the
#      14w root picked "collect" for multikey_3 and "search" for the word-keyed multikey_2.
#   3. NEEDLE haystack (RULER's type_haystack: needle) on 25% of eligible niah problems: the haystack is
#      nothing but distractor needles; no training range had ever held more than a few.
#   4. labeled_records: author_label_count (filtered count — the OOLONG-user agnews shape), 2-author subsets
#      (30%), author-filtered share ~14% -> ~29%. 14w's OOLONG-user leaves dropped the user filter and
#      tallied every line (0.95 -> 0.65). Leaf text follows the existing filtered-leaf convention
#      (filter in the op-phrase parenthetical, every line with a verdict) — no new header shape.
# Counts: niah_novel/niah_multi 40 -> 60 (the needle/uuid variants are a fraction of each). All else = run 14.
#
# EVAL: the run-6..14 list in its EXACT order (seeds derive from position), then the 7 RULER tasks never
# scored before APPENDED at the end so no existing seed moves. SCORE_TASKS stays the 9-task set (comparable
# to 14w's 0.832); RULER-13 and the 16-task mean are computed from the jsonl.
#
#   Dry run (traces only, no Tinker):  TRAIN=0 bash scripts/run_sft15_warm.sh
#   Real run:  nohup caffeinate -is bash scripts/run_sft15_warm.sh > /tmp/sft15_eval.log 2>&1 &
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
INIT=${INIT_CHECKPOINT:-tinker://de730742-0918-5a2d-80fe-48f3d41ed0c3:train:0/weights/sft_general14w}
TRAIN=${TRAIN:-1}

echo "===== SFT (sft_general15w, warm start from $INIT, TRAIN=$TRAIN) STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 200); do
  if TRAIN=$TRAIN INIT_CHECKPOINT=$INIT LR=5e-6 CONTEXT_DROP=0.5 QUESTION_PLACEHOLDER_FRAC=0.3 \
SFT_TASKS=synth_sum,synth_count,synth_max,synth_min,synth_sumwhere,synth_mode,synth_distinct,synth_sumby,synth_count2,synth_diff,synth_maxwhere,synth_count_cmp,synth_count_range,synth_runreset,synth_varchain,synth_peak,synth_streak,synth_adjacent,synth_first_exceed,synth_filter_argmax,synth_2d,synth_topk,long_records,rule_label,labeled_records,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa \
N_PER_TASK=10 N_PER_TASK_OVERRIDE=vt_novel:150,labeled_records:240,long_records:120,synth_topk:40,rule_label:40,realdoc_count:40,niah_novel:60,niah_multi:60,narrativeqa:30,synth_2d:24,synth_filter_argmax:15 \
CTX=3000 DOC=6000 DOC_MIX=6000:2,8400:1,14000:1 DOC_MIX_OVERRIDE="vt_novel=6000:1,14000:1" CHUNK=200000 FOLD_LEAF_TOKENS=400 ROOT_DUP=4 DATUM_MODE=agent INTERNAL_KEEP=1.0 SFT_BATCH_SIZE=16 \
BOOKQA_LEAF_MODEL=anthropic/claude-haiku-4-5-20251001 SAVE_NAME=sft_general15w \
PRINT_TRACES=1 TRACE_OUT=/tmp/sft_general15w_traces \
PYTHONPATH=. uv run python sft.py; then break; fi
  echo "===== SFT attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min (resumes from checkpoint) ====="
  sleep 300
done
if [ "$TRAIN" != "1" ]; then echo "===== DRY RUN DONE $(date +%H:%M:%S) (traces in /tmp/sft_general15w_traces) ====="; exit 0; fi

echo "===== SFT DONE, EVAL STARTING $(date +%H:%M:%S) ====="
for attempt in $(seq 1 20); do
  CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk,niah_single_2,niah_single_3,niah_multikey_2,niah_multikey_3,niah_multivalue,qa_1,qa_2 \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0,niah_single_2:5,niah_single_3:5,niah_multikey_2:5,niah_multikey_3:5,niah_multivalue:5,qa_1:5,qa_2:5 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_general15w \
PYTHONPATH=. uv run python -m eval.run && break
  echo "===== eval attempt $attempt failed at $(date +%H:%M:%S); retrying in 5 min ====="
  sleep 300
done
echo "===== ALL DONE $(date +%H:%M:%S) ====="
