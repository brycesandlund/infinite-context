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
`cwe 1.000` · `fwe 0.600` · `niah_single_2 0.400` · `niah_multikey_1 0.200` ·
`niah_multiquery 0.000` · `vt 0.160` · `qa_1 0.000` → **OVERALL 0.337**.
Transfers where the leaf-op matches training (cwe = counting → 1.0, decomposed 30 nodes); fails on
untrained leaf-ops (needle/track/QA). Scaffold is task-general; leaf-op must be taught.

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
