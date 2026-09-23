# Handoff — state as of 2026-09-23 03:00

Written at session compaction. Everything below is verifiable from `training_runs.md`,
`eval_results/raw/`, and `trace_snippets/`. Nothing is running.

## 1. Checkpoints

Held-out SCORE = mean over 9 RULER+OOLONG tasks x 5 seeds, doc 4000, budget 3000, temp 0.2.
Diagnostics = 12 in-distribution tasks x 2, excluded from SCORE.

| run | lineage | SCORE | diag | checkpoint |
|---|---|---|---|---|
| **14w** | warm from 13 | **0.832** | 1.000 | `tinker://de730742-0918-5a2d-80fe-48f3d41ed0c3:train:0/weights/sft_general14w` |
| 12w | warm from 8w | 0.760 | 1.000 | `tinker://a8b5c74c-a05e-52cc-9cbc-797739ee23c5:train:0/weights/sft_general12w` |
| 8w | warm from 7 | 0.758 | 0.958 | `tinker://75b29e89-caf6-5e79-8415-5f4e64af39a3:train:0/weights/sft_general8w` |
| 13 | **from base** | 0.718 | 0.958 | `tinker://bb67e67e-7d0f-5bf1-adaf-7022ee1f0d8f:train:0/weights/sft_general13` |
| 11w | warm from 8w | 0.709 | 0.958 | `tinker://a316bf8b-51dd-586b-b3a9-09c47b08c531:train:0/weights/sft_general11w` |
| 10w | warm from 8w | 0.711 | 1.000 | `tinker://cbf4f333-a798-5a49-92ca-7f1d89a4b8a0:train:0/weights/sft_general10w` |
| 9w | warm from 8w | 0.646 | 1.000 | `tinker://d7385f1e-f958-53e4-a776-c642d5d51844:train:0/weights/sft_general9w` |
| 7 | **from base** | 0.662 | 0.911 | `tinker://8ec58e87-7310-55dd-a31d-9f9caf3c0382:train:0/weights/sft_general7` |
| 6 | from base | 0.675 | 0.946 | `tinker://7161d592-9d8a-5583-9ff1-13a273154e96:train:0/weights/sft_general6` |

Per-task for the four that matter:

| task | 8w | 12w | 13 (base) | **14w** |
|---|---|---|---|---|
| vt | 0.88 | 0.96 | 0.28 | **1.00** |
| niah_single_1 / multikey_1 / multiquery | 1.00 / 0.80 / 1.00 | 1.00 / 0.80 / 0.70 | 0.80 / 0.80 / 1.00 | **1.00 / 1.00 / 1.00** |
| cwe / fwe | 1.00 / 0.93 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.87 |
| oolong_counting | 0.36 | 0.42 | 0.35 | **0.63** |
| oolong_user | 0.60 | 0.75 | **0.95** | 0.65 |
| oolong_temporal | 0.25 | 0.21 | 0.28 | **0.35** |

## 2. The two results the paper rests on

1. **Corpus ablation, both from base:** run 7 (0.662) vs run 13 (0.718). Six weeks of corpus work is
   worth ~+0.06 from base. Confound: batch 8 vs 16 (accepted deliberately; NLL has never tracked eval
   here — 7w2 batch 8 NLL 0.0027 scored 0.666, 8w batch 16 NLL 0.0136 scored 0.758).
2. **Train/serve prompt skew** (`memory/train-serve-prompt-skew.md`): training always described the
   document schema, RULER never does and OOLONG describes only part. Run 13 folded **2/2
   in-distribution** on vt_novel and **0/5** on RULER vt — recognition, not capability. Four prompt
   diversity knobs (`CONTEXT_DROP`, `_VT_BARE_FRAC`, `_SCHEMA_LITE_FRAC`, `QUESTION_PLACEHOLDER_FRAC`)
   took 0.718 → 0.832, with OOLONG aggregation moving too despite never being trained on.

## 3. Running an eval

Standard scoreboard (69 rollouts, ~9 min, few $):
```bash
CKPT=<tinker://...> \
EVAL_TASKS=synth_sum,synth_count,synth_mode,synth_varchain,synth_count_cmp,realdoc_count,niah_novel,niah_multi,vt_novel,narrativeqa,oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe,synth_2d,synth_peak,long_records,rule_label,labeled_records,synth_topk \
SCORE_TASKS=oolong_counting,oolong_user,oolong_temporal,niah_single_1,niah_multikey_1,niah_multiquery,vt,cwe,fwe \
N_PER_TASK=2 N_PER_TASK_OVERRIDE=oolong_counting:5,oolong_user:5,oolong_temporal:5,niah_single_1:5,niah_multikey_1:5,niah_multiquery:5,vt:5,cwe:5,fwe:5,synth_count:0,synth_varchain:0,synth_count_cmp:0,niah_novel:0 \
BACKEND=tinker MODE=decompose DOC_SIZE_TOKENS=4000 AGENT_CONTEXT=3000 MAX_CHUNK_TOKENS=200000 TEMP=0.2 MAX_NODES=150 \
OUT=/tmp/eval_X PYTHONPATH=. uv run python -m eval.run
```
EVAL_TASKS **order is load-bearing** (seeds derive from task position) — never reorder.
Run it under `zsh -ic` so the Anthropic key is in the environment.

Fresh-seed / length / budget probes (seeds 3,000,000+, never trained or read):
```bash
scripts/eval_len.sh <tag> <ckpt> <doc_tokens> <budget> [n_per_task]
```

## 4. Candidate evals & ablations, ranked

| # | what | cost | why |
|---|---|---|---|
| 1 | **Description-ablation probe**: run 13 and 12w on in-dist `vt_novel` with `task_context=""` | ~10 min, few $ | The ONE open confound. Run 14 is warm-from-13 so staging and prompt diversity are entangled; +0.07 > the +0.04 staging has ever bought, but that's inference. If 13 flips to binary and 12w holds, the mechanism is proven on an in-dist task. Needs a diagnostic-only eval flag (an ablation that REMOVES information, not a harness guard). |
| 2 | **Fresh seeds on 14w** at doc 4K | ~10 min | Every number above is one seed block. 14w's 0.832 needs a second block before it goes in a paper. `scripts/eval_len.sh fresh14 <14w> 4000 3000` |
| 3 | **Length generalization**: 14w at 10K/20K doc | ~30 min each | Prior finding (A7/A8 at 10K): length generalization holds, counting/temporal are length-sensitive. Redo on the best checkpoint. |
| 4 | **Budget probe**: 14w at 5K budget | ~10 min | `memory/eval-budget-state-ceiling.md` — at 3K a faithful 2-D tally >~60 keys can't merge at the root; the 5K probe recovered trec/negation temporal. Quantifies the ceiling for the paper. |
| 5 | **oolong_user regression** 0.95 → 0.65 | investigation | Only task where run 13 still leads. Both failing roots wrote NO key contract; `_SCHEMA_LITE_FRAC=0.4` likely over-corrected. Try keeping the author tag described while dropping the date rule. |
| 6 | Frontier single-shot baseline refresh | $ | `eval_results/raw/gpt5_4_oolong_20k.summary.txt` has gpt-5.4 at 20K: counting 0.552, user 0.742, temporal 0.455. |

## 5. Open items

- **oolong_temporal ceiling is real and should be reported, not fixed**: trec (38 months x 6 labels) and
  negation (72 dates) overflow the root at 3K with a faithful 2-D tally. Verified by the 5K probe.
  Historical "better" scores on those seeds came from lossy states (1-D tallies, month keys without years).
- **Leaf label accuracy** on 6-class trec is the counting ceiling — base-model territory, not ours.
- **Trace cache**: `_CACHE_VERSION` is `v7`. Bump it whenever question text or oracle text changes;
  `nodesc`/`qph` are in `_trace_key` but the QUESTION body is not.
- **Paper**: `paper/iclr2027/paper.tex`, `latexmk -pdf paper.tex`. Bib has 76 entries incl. the
  long-context claims and the harness lit review (`paper/lit_review_long_context_harnesses.md`).

## 6. Key files

- `training_runs.md` — every run, per-task tables, diagnoses. The primary record.
- `trace_snippets/` — `run14_rollouts.txt`, `run13_base_rollouts.txt`, `run12_failures.txt`,
  `root_audit_v11.txt`, `preamble_comparison_8w_10w_11.txt`, `new_qtypes_run10.txt`.
- `scripts/` — `run_sft14_warm.sh` (best recipe), `run_sft13_base.sh` (from-base reference),
  `verify_tasks.py` (gold-grade + budget check before any launch), `audit_roots.py`, `extract_roots.py`.
- `eval_results/raw/sft_general*.{jsonl,txt}` — every rollout, with full trees.
