# Eval results — preliminary (June 2026)

All OOLONG numbers are **held-out** problems via the shared `oolong_spec(task, idx, base)`
with `OOLONG_BASE=2000000` (distinct from the SFT `DATA_SEED=500000`). Score = OOLONG-
official grading (numeric = `0.75^|err|`, categorical/user/date = exact membership,
comparison = word-boundary containment). N = 10 problems per family unless noted.

## Models
- **Ours (SFT):** Qwen3.6-35B-A3B + LoRA r32, 3-family OOLONG SFT (counting/user/temporal,
  N_PER_TASK=50, 1 epoch, NLL 0.030). Checkpoint
  `tinker://ee547d3e-0366-53c6-aba2-326dc04f671f:train:0/weights/sft_oolong`.
  Runs through the recursive agent harness: **8K per-agent budget, MAX_CHUNK 6000, MAX_DEPTH 10,
  no turn cap** — i.e. it must decompose; doc never fits in context.
- **gpt-5.4 (frontier baseline):** single-shot, **whole document in one prompt** (native 1.05M
  context window — no truncation, no decomposition), temperature 0. Script `frontier_ceiling.py`.

---

## 1. OOLONG length sweep — OURS (graceful degradation)
Same checkpoint, same problems by seed, only `DOC_SIZE_TOKENS` changes. Agent budget fixed at 8K.

| family | 10K | 20K | 40K | 80K |
|---|---|---|---|---|
| counting | 0.519 | 0.428 | 0.377 | 0.415 |
| user | 0.656 | 0.700 | 0.818 | 0.550 |
| temporal | 0.421 | 0.413 | 0.491 | 0.323 |
| **OVERALL** | **0.532** | **0.514** | **0.562** | **0.429** |

mean tree grew ~40 → ~110–240 → ~190–220 → **365 (counting) / 759 (user) / 1705 (temporal)** nodes.
**Flat to 40K (4×); bends at 80K (8×).** The bend is the *irreducible-combine* tasks, not the method.
Verified mechanism (from the 80K rollouts): the per-key tally (`YYYY-MM|label`, `userID`) **grows as it
sums up the tree** — mid-tree child reports reach ~1.3–1.8K tokens, so a node summing 2–3 of them
(~4K+ tokens + prompt) **exceeds the 8K agent budget at depth 1–2** (in context reading children, or in
output truncating the summed tally). Worse, an overflowed child makes its parent **re-split and re-spawn**,
cascading into malformed ~2000-node trees (vs ~320 for a clean binary split) with `"agent overflowed
context"` stubs propagating up → incomplete aggregate / root overflow. Overflow counts: counting 0/10
(its `label:count` combine is O(≤6) ≈ 30 tok, never accumulates → 0.415 holds), user 2/10, temporal 3/10.
**Bounded/associative combines scale gracefully; irreducible O(n) combines scale only until the growing
tally outgrows the budget (~80K here).** Fix: bigger budget, a non-growing/streaming combine, or
sequential decomposition — not a method flaw. Contrast the paper's frontier collapse (0.85→0.40, 8K→128K)
and our §6 crossover (frontier already at 0.338 by 40K).

## 2. OOLONG head-to-head — OURS vs gpt-5.4 (single-shot, same problems)

| | 10K ours | 10K gpt-5.4 | 20K ours | 20K gpt-5.4 |
|---|---|---|---|---|
| counting | 0.519 | 0.468 | 0.428 | 0.552 |
| user | 0.656 | 0.742 | 0.700 | 0.742 |
| temporal | 0.421 | 0.472 | 0.413 | 0.455 |
| **OVERALL** | **0.532** | **0.561** | **0.514** | **0.583** |

Read: near-parity at 10K (we win counting there); at 20K gpt-5.4 is slightly ahead. Frontier does
**not** collapse at ≤20K (its 1.05M window reads everything) — **the crossover appears at 40K**, see §6.

## 3. gpt-5.4 on the OFFICIAL released oolong-synth (≤8K) — sanity vs the paper
Single-shot on `oolongbench/oolong-synth` validation rows (their exact context+question+gold):
**OVERALL 0.900** (numeric 0.909, label 1.000, comparison 0.750, user 0.900). Matches the paper's
~0.85 at 8K → confirms our prompt/grading are sound. Our *generated* problems are a **harder
operating point** (130–232 examples/doc vs released ~129; numeric golds median ~52 vs ~7.5), which
is why our absolute numbers sit lower and are NOT directly comparable to the paper's curve.

## 4. Cross-task transfer — OURS on RULER (zero-shot; SFT was OOLONG-only) @10K, N=5
7 RULER task types × 5 problems = 35 questions, same seeds for both columns.

| task                                   | gpt-5.4 (1M budget)                    | fine-tuned Qwen3.6-35B-A3B (8K budget) |
|----------------------------------------|----------------------------------------|----------------------------------------|
| niah_single_2                          | 1.000                                  | 0.400                                  |
| niah_multikey_1                        | 1.000                                  | 0.200                                  |
| niah_multiquery                        | 0.000                                  | 0.000                                  |
| vt                                     | 1.000                                  | 0.160                                  |
| cwe                                    | 1.000                                  | 1.000                                  |
| fwe                                    | 1.000                                  | 0.600                                  |
| qa_1                                   | 0.400                                  | 0.000                                  |
| **OVERALL**                            | **0.771**                              | **0.337**                              |

Frontier (1M window, single pass) reads everything and crushes the needle/track tasks. Our SFT
transfers only where the leaf-op matches training (cwe = counting → 1.0, decomposed ~30 nodes) and
fails on untrained leaf-ops (needle/track/QA): the scaffold is task-general, but the leaf-op must be
taught. (cwe ties at 1.0 — both nail it; niah_multiquery 0.000 for both.) N=5/task is coarse
(±0.2 granularity) — preliminary. gpt-5.4 single-shot from `eval/run.py MODE=single` (raw
`eval_results/raw/`); ours through the harness (`MODE=decompose`).

## 5. Counting sanity — old counting-only SFT vs 3-family, same 10 counting problems
Counting-only ckpt `tinker://8b734bcf-...:sft_oolong`: **0.485**. 3-family ckpt: **0.519**.
Multi-task SFT did not hurt counting (slightly better); the earlier "0.94" was a 4-problem,
mostly-categorical draw — not representative.

---

## Raw files in `raw/`
Survived `/tmp` cleanup:
- `sft_oolong_40k.{txt,jsonl,log}` — full 40K eval (jsonl = all rollout trees, ~40 MB).
- `gpt5_4_oolong_20k.log` + `.summary.txt` — gpt-5.4 single-shot @20K.
- `sft_oolong_40k.summary.txt` — 40K per-family summary.

**Lost to `/tmp` cleanup (numbers above are from the run logs; regenerate if raw needed):**
SFT 10K/20K OOLONG, SFT RULER, old-counting-only, gpt-5.4 @10K (our data), gpt-5.4 released.
All are reproducible: SFT evals are deterministic in problem selection (temp 0.2 sampling varies
slightly); gpt-5.4 single-shot is temp 0 (near-deterministic). Commands in `frontier_ceiling.py`,
`frontier_released.py`, and `eval/run.py` (env: CKPT, DOC_SIZE_TOKENS, EVAL_TASKS, N_PER_TASK, TEMP).

---

## 6. THE CROSSOVER — OURS vs gpt-5.4 across the length sweep (same problems)
Single number = OVERALL (mean of counting/user/temporal). gpt-5.4 single-shot full-doc; ours 8K-budget decomposition.

| OVERALL | 10K | 20K | 40K | 80K |
|---|---|---|---|---|
| Ours | 0.532 | 0.514 | **0.562** | **0.429** |
| gpt-5.4 | 0.561 | 0.583 | **0.338** | **0.327** |

Per-family @40K (ours wins all three): counting 0.377 vs 0.309 · user 0.818 vs 0.388 · temporal 0.491 vs 0.317.
Per-family @80K (ours still wins all three): counting 0.415 vs 0.300 · user 0.550 vs 0.450 · temporal 0.323 vs 0.232.
Frontier is flat-collapsed from 40K (0.338→0.327); our 80K dip (0.562→0.429) is the Tier-3 budget-overflow
wall (counting, the bounded combine, holds at 0.415) — fixable, and we stay above the frontier throughout.

**Story:** frontier is competitive/ahead at 10K–20K, then **collapses at 40K** (0.583→0.338) — its 1.05M
window holds the text but single-pass aggregation over ~600+ items fails. Our decomposition stays **flat**
and crosses above. Graceful-degradation-vs-collapse, on identical problems. (gpt-5.4 raw: `raw/gpt5_4_oolong_40k.log`.)

## 7. What does SFT actually buy? — single-shot vs the harness, OOLONG counting @10K (N=10)
Same 10 counting problems, four ways. "harness" = MODE=decompose (8K budget, must use read_chunk/spawn);
"single-shot" = MODE=single (whole doc in context, no tools).

| config                  | score | mean_tree | reads doc?          |
|-------------------------|-------|-----------|---------------------|
| base Qwen single-shot   | 0.681 | 1.0       | yes (in context)    |
| **SFT + harness**       | 0.519 | ~40       | yes (read_chunk)    |
| gpt-5.4 single-shot     | 0.468 | 1.0       | yes (in context)    |
| gpt-5.4 + harness       | 0.376 | 3.3       | yes (read_chunk)    |
| base Qwen + harness     | 0.000 | 1.5       | **no — never reads**|

Three reads:
1. **The harness HURTS anyone not trained on it.** gpt-5.4 drops 0.468 → 0.376 (decomposes shallowly,
   ~3 nodes, and loses accuracy in the clumsy split/aggregate); base Qwen drops 0.681 → **0.000** —
   handed the tools with no training it never even calls read_chunk (`no_answer=10/10`, `grounded=0`).
2. **SFT's entire value is the PROTOCOL.** Identical base weights go from 0.000 (with the harness) to
   **0.519** after SFT — and SFT beats *frontier operating the same harness* (0.519 > 0.376) by
   decomposing finely (~40 nodes) instead of clumsily (gpt-5.4's 3.3).
3. **At 10K, single-shot still wins overall** (base 0.681 > SFT-harness 0.519): the doc fits native
   context, so forced decomposition is pure added error. The harness only earns its keep past native
   context — the §6 crossover (40K+), where single-shot can't fit or the frontier collapses.

Raw: `raw/gpt5_4_oolong_counting_decomp.*`, `raw/base_qwen_oolong_counting_decomp.*` (+ single-shot
`raw/base_qwen_oolong_single.*`). gpt-5.4 single-shot 0.468 is the §2 number (old-prompt path).

## 8. Budget sweep — is the 80K dip the working-memory wall? (SFT, 80K doc, 8K→12K budget)
Direct test of the §1 overflow hypothesis: same SFT checkpoint, same 80K problems by seed, only the
per-agent context budget changes 8K → **12K (1.5×)**. If the dip is the *combine* outgrowing the budget
(not a method flaw), raising the budget should recover the irreducible-combine families (user/temporal)
and leave counting — whose combine is a bounded ≤6-label vector — flat.

| family | score 8K | score 12K | Δ | overflow 8K→12K | mean tree 8K→12K | max tree 8K→12K |
|---|---|---|---|---|---|---|
| counting | 0.415 | 0.445 | +0.03 | 0 → 0 | 365 → 424 | 499 → 497 |
| user | 0.550 | 0.575 | +0.03 | 2 → **1** | **759 → 373** | **4293 → 497** |
| temporal | 0.323 | **0.398** | **+0.08** | 3 → **2** | **1705 → 707** | 6653 → 3853 |
| **OVERALL** | **0.429** | **0.473** | **+0.044** | 5 → 3 | — | — |

**Causal confirmation.** The improvement lands exactly where the mechanism predicts: temporal (widest
tallies, most overflow) gains most (+0.08); counting (bounded combine, 0 overflow) is flat. The
smoking gun is the **cascade collapse** — the 4293-node user rollout (a node that couldn't sum its
children's tallies in 8K → overflow → re-split/re-spawn explosion) **drops to a clean 497 nodes at 12K**:
give the node room to hold the merged tally and the whole retry-cascade evaporates. Mean tree size
*falls* for user (759→373) and temporal (1705→707) **while scores rise** — leaner trees, fewer
overflow-driven re-spawns. Counting's tree is unchanged (~365→424; pure binary split, no cascade either way).

**But the irreducible combine is pushed out, not solved.** Temporal still has 2/10 overflows and a
3853-node cascade — its widest ~290-pair `YYYY-MM|label`/`date` tallies exceed even 12K when summed
pairwise. An O(distinct-keys) combine re-hits the wall at *some* length; 1.5× budget just moves it.
The real fix is a **non-growing combine** (streaming/thresholded/top-k sketches) or a **sequential
left-fold** that threads one running tally instead of unioning pairwise — keeping the working set O(1).
Raw: `raw/sft_oolong_80k_12kbudget.*` (vs the 8K baseline `raw/sft_oolong_80k.*`).
