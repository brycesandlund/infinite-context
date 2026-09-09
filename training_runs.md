# Training runs

Log of SFT runs: configuration → eval results → problems → fixes carried to the next run.
Eval mode is `decompose` (the tool harness) at `TEMP=0.2` unless noted.

---

## sft_general (run 1)

**Goal:** first general-decomposition SFT — teach the scaffold on abstract synth primitives +
real-prose leaves, measure zero-shot transfer to OOLONG/RULER.

**SFT config**
- Tasks: 15 synth + `realdoc_count` + `bookqa` (En.QA, model leaf = haiku + gold rejection)
- `N_PER_TASK=10`, `realdoc_count:50`; `SYNTH_STRATEGY=mixed`
- `DOC=3000`, `CTX=8000` (budget), `CHUNK=200000` (uncapped read)
- From base Qwen3.6-35B-A3B, LoRA rank 32, 1 epoch, LR 1e-5
- Data: **208 traces / 5,980 datums**, NLL 0.126 → ckpt `tinker://fa300fb8-…/weights/sft_general`

**Eval** (DOC=6000, budget=8000, N=3) — **OVERALL 0.323**
- In-dist (decomposing): synth_distinct 0.81, realdoc 0.71, sumby 0.67, count 0.58,
  count_cmp 0.19, **sum 0.11, mode 0.00**; bookqa 0.33 (**single-shot, tree=1**)
- OOD: niah_single_1 0.67, oolong_user 0.50, fwe 0.33, vt 0.20, oolong_temporal 0.08,
  oolong_counting 0.00 (tree 669 runaway), cwe 0.00

**Problems**
1. **DOC (6000) < budget (8000) → the model single-shots** (reads whole doc, no decomposition):
   bookqa/niah/vt/cwe/fwe/oolong_temporal all `tree=1`. The eval measured single-shot ability,
   not decomposition. Root cause: per-agent context is only ~1.1k, so the 8k budget is ~6× oversized.
2. **Low in-dist leaf accuracy** (sum 0.11, mode 0.00): under-fit at N=10, and error accumulates
   over the longer 6k eval doc (more records to aggregate).
3. `ruler_part` substring grader false-positive (gold "no" ⊂ "answer not found").

**Fixes → run 2**
- Budget `CTX=3000` (so DOC>budget *forces* decomposition; matches the ~1.1k real per-agent use)
- `DOC=6000` train, eval at DOC=4000/budget=3000
- `N_PER_TASK=30`, `SYNTH_STRATEGY=both` (teach fold on every bounded op, not just runreset/varchain)
- Swap `bookqa`→`narrativeqa` (65 → 307 answerable questions)
- New `qa_part` word-boundary grader for QA (fixes the "no" substring bug), set on the Problem

---

## sft_general2 (run 2)

**SFT config**
- Tasks: 15 synth + `realdoc_count` + `narrativeqa` (haiku leaf + gold rejection)
- `N_PER_TASK=30`, `realdoc_count:50` (narrativeqa self-capped at ~30 accepted, 34% accept)
- `SYNTH_STRATEGY=both`; `CTX=3000`, `DOC=6000`, `CHUNK=200000`, `FOLD_LEAF=500` (default)
- Data: **530 traces / 26,135 datums**, NLL 0.029 → ckpt `tinker://476d840d-…/weights/sft_general2`

**Eval** (DOC=4000, budget=3000, N=3) — **OVERALL 0.551** (up from 0.323)
- In-dist: synth sum/count/mode/distinct/sumby/count_cmp all **1.00** (decomposing, tree≈8);
  realdoc 0.92; **varchain 0.00**; **narrativeqa 0.00**
- OOD: fwe 0.67, niah_single_1 0.67, oolong_user 0.33, vt 0.13, oolong_temporal 0.08,
  oolong_counting 0.01, cwe 0.00
- Full RULER-13 (separate): **OVERALL 0.159** — pervasive abstention ("None")

**Problems**
1. **varchain 0.00 — context OVERFLOW, not reasoning.** The model tracks all 5 vars perfectly
   (deepest leaf computes the gold), but the verbose fold node is **3038 real tokens** > 3000
   budget → mid-chain agents overflow → parents fall back to partial accumulators → wrong answer.
   Root cause: oracle `count_tokens` under-counts by ~480 (omits the tool-schema prefix), so it
   silently generated over-budget traces.
2. **narrativeqa 0.00 — over-abstention.** Leaf verdicts are class-imbalanced (~19 "none" : 1
   "answer" per trace on rare-answer Qs) → model learns the majority class (abstain), returns
   "none" even on trained questions. Compounded by the Qwen leaf-comprehension gap (can't find
   answers haiku found at data-gen).
3. **OOLONG runaway — degenerate split.** Model states the right midpoint but emits the spawn
   subtask with its OWN range instead of the two halves → infinite recursion to the 2000-node cap
   → garbage. Only on the unfamiliar OOLONG format (numeric-noise distraction).
4. **RULER over-abstention (0.159).** Model single-shots haystacks (doesn't recognize the prose
   format as "decompose me"), reads a partial, abstains → "None". Amplified by narrativeqa's
   none-bias bleeding into retrieval.
5. OOLONG classification leaf wrong even when it decomposes cleanly (25 vs 14 — fuzzy TREC NLU).

**Fixes → run 3**
- `FOLD_LEAF_TOKENS=400` (varchain 3038 → 2775, fits budget). [oracle count-realism: TODO]
- **QA verdict rebalancing** (`QA_NONE_KEEP=0.5`, `QA_ANSWER_DUP=2`): downsample "none" verdicts,
  upweight "answer" verdicts (scoped to QA tasks by content)
- **`niah_novel`** task: synthetic needle-in-a-novel, mechanical scripted leaf — teaches
  decomposing a **prose haystack** (the missing corpus type; RULER-transfer target)
- **`DOC_MIX=6000:3,14000:1`**: a fraction at longer docs → deeper trees → more mid-range split
  turns → robust split arithmetic (bet on the OOLONG runaway) + length generalization
- Rebalance the mix toward prose: `narrativeqa`/`niah_novel`/`realdoc_count` to 80, synth to 20
- **Eval `MAX_NODES` cap** to kill node explosion (runaway wastes time/money)
- Known NOT addressed this round: OOLONG classification accuracy (fuzzy NLU — needs OOLONG-in-SFT
  or RL); candidate follow-ups: multi-needle NIAH w/ hidden query, "OOLONG-like" oracle on a
  different labeled dataset.

---

## sft_general3 (run 3) — 2026-09-08

**SFT config**
- Tasks: 15 synth + `realdoc_count` + `niah_novel` + `narrativeqa`
- `N_PER_TASK=20`, `realdoc_count:80,niah_novel:80,narrativeqa:80`; `SYNTH_STRATEGY=both`
- `CTX=3000`, `DOC=6000`, `DOC_MIX=6000:3,14000:1` (model-leaf tasks exempt), `CHUNK=200000`,
  `FOLD_LEAF_TOKENS=400`
- QA verdict rebalancing (`QA_NONE_KEEP=0.5`, `QA_ANSWER_DUP=2`); haiku leaf; disk trace cache on
- Tinker SDK: local editable `~/GitHub/tinker-sdk` **0.27.1** (pinned via `uv add --editable`);
  fixed its `auth_token()` missing `timeout`/`max_retries` kwargs (crashed `ServiceClient` init)
- Data: **540 traces / 35,864 datums** (narrativeqa 80/80 @33% accept — all from cache, 0 API
  calls; QA verdicts pos 1166×2, neg 2235/4469 kept). NLL **0.0249**, 2,242 batches, ~4.7 h train
  → ckpt `tinker://2a5fa30c-…/weights/sft_general3`
- Artifacts: `eval_results/raw/sft_general3_doc4k_b3k.{jsonl,txt,summary.txt}`, `sft_general3_chain.log`

**Eval** (DOC=4000, budget=3000, N=3, `MAX_NODES=150`) — **OVERALL 0.686** (up from 0.551)

| task | g3 | g2 | tree | note |
|---|---|---|---|---|
| synth_varchain | **1.00** | 0.00 | 10 | overflow fix confirmed (FOLD_LEAF=400) |
| synth (count/mode/distinct/sumby/count_cmp) | **1.00** | 1.00 | 10 | held |
| synth_sum | 0.67 | 1.00 | 10 | one fold arithmetic miss (−57→−96), N=3 noise |
| realdoc_count | **1.00** | 0.92 | 30 | |
| niah_novel (new, in-dist) | **1.00** | — | 31 | |
| narrativeqa | **0.33** | 0.00 | 31 | rebalancing worked: no_answer 0/3 (was abstaining) |
| **cwe** (OOD) | **0.97** | 0.00 | 30 | **now decomposes** (was single-shot tree=1) |
| **fwe** (OOD) | **0.89** | 0.67 | 31 | now decomposes |
| niah_single_1 | 0.67 | 0.67 | 18 | now decomposes (was tree=1) |
| niah_multikey_1 | 0.33 | 0.00 | 4 | 2/3 **overflow** — see below |
| oolong_counting | **0.34** | 0.01 | 54 | **runaway gone** (was 677/2000-cap) |
| oolong_temporal | 0.17 | 0.08 | 4 | |
| oolong_user | 0.00 | 0.33 | 54 | 1 rollout hit the 150 cap; 1 degenerate gold ("numeric value") |
| vt | 0.00 | 0.13 | 9 | 3/3 **overflow** — see below |

**What the run proved**
1. **Format/corpus hypothesis confirmed.** `niah_novel` (prose haystack, mechanical leaf) made
   RULER `cwe`/`fwe`/`niah` start *decomposing* real-prose haystacks instead of single-shotting:
   cwe 0.00→0.97, fwe 0.67→0.89. Teaching the corpus type transfers the scaffold.
2. **DOC_MIX killed the degenerate-split runaway** (oolong_counting tree 677→54, no cap hits on
   counting). Deeper training trees → robust mid-range split arithmetic.
3. **Rebalancing fixed over-abstention** (narrativeqa 0→0.33, no_answer 0/3; cwe/fwe no_answer 0/3).
4. varchain was purely the overflow bug — 1.00 once the fold leaf fit the budget.

**Residual failure mode — OVER-READ → OVERFLOW** (replaces abstention as the RULER failure)
- All `vt` misses and 2/3 `niah_multikey_1` misses are `term=overflow`: the model issues a single
  huge read (e.g. `Read tokens 2000..4000` — half the doc into a 3000 budget) on these formats.
  Training only ever shows ≤~700-token reads; on vt/multikey layouts it doesn't map the range to
  a leaf-sized read. CHUNK is uncapped so nothing stops it (and per the no-eval-guards rule, the
  fix is training-side).
- `oolong_user` still has one runaway (hit MAX_NODES=150 — the cap did its job cheaply) and one
  degenerate OOLONG gold ("numeric value").
- OOLONG classification leaf accuracy (fuzzy NLU) still the ceiling on oolong_*.

**Fixes → run 4 (candidates)**
- **Multi-needle NIAH w/ hidden query** (mechanical) — the direct `niah_multikey`/`multiquery`
  analog; should teach leaf-sized reads + collect-then-resolve on haystacks.
- **vt-like synthetic** — variable-chain tracking over *prose* (varchain-on-haystack) to fix vt's
  over-read/overflow.
- **"OOLONG-like" classification+aggregation oracle** on a non-OOLONG labeled dataset (dbpedia)
  for the classification leaf.
- Monitoring: `WANDB=1` / `python -u` so batch NLL is visible mid-run (stdout is block-buffered).

---

## sft_general4 (run 4) — PLANNED

**Target:** the run-3 residual — **over-read → overflow** on unfamiliar haystack layouts
(vt 0.00 3/3 overflow; niah_multikey_1 2/3 overflow). Teach leaf-sized reads + the missing
leaf-ops on those layouts with two new MECHANICAL haystack tasks (tasks/niah, oracle/prose.py),
each a **broad VARIANT FAMILY drawn per seed** (goal: general capability, not fitting the eval):
- **`niah_multi`** — the RULER-niah family generalized. Mode ∈ {**hidden** (a "the key you must
  look up is K" sentence is itself in the doc — a real multi-hop; no RULER analog), **explicit**
  (1–6 needles, key in the question = multikey), **multiquery** (2–4 keys asked → set),
  **multivalue** (one key, 2–4 values → set)} × key {words, uuids} × value {numbers, uuids} ×
  filler {novel, essay, noise}. `dict` state: leaf folds facts, combine = per-key union, root
  resolves by mode. Binary and fold both work.
- **`vt_novel`** — the RULER-vt family generalized: 2–3 `VAR` chains × 3–7 hops (distractor chains
  carry other values), question ∈ {**which_vars** hold value V (set), **final_value** of VAR X
  (exact)} × the same three fillers. Left-fold bindings dict.

**Overflow fixes found while building these (all training-side):** the fold protocol restates the
accumulator at every hop, and big dict states (uuid keys/values ≈ 20 tok each; 5-letter random
var names ≈ 4 tok each) pushed fold nodes to 3.5–4k real tokens. Fixed by (1) per-record lines
show only the DELTA binding, (2) base fold node no longer reprints the accumulator in its header
/ delegation text / "The chain returned X" (saves ~3× state, benefits every fold task),
(3) niah_multi caps needles at 3 when uuids are involved, (4) vt_novel caps total vars ≤ ~17.

**SFT config** (run 3 + the two new tasks @80)
- Tasks: 15 synth + `realdoc_count` + `niah_novel` + `niah_multi` + `vt_novel` + `narrativeqa`
- `N_PER_TASK=20`, overrides `realdoc_count:80,niah_novel:80,niah_multi:80,vt_novel:80,narrativeqa:80`
- `SYNTH_STRATEGY=both`, `CTX=3000`, `DOC=6000`, `DOC_MIX=6000:3,14000:1`, `CHUNK=200000`,
  `FOLD_LEAF_TOKENS=400`, QA rebalancing on, trace cache on, `PYTHONUNBUFFERED=1` (mid-run log)
- Verification: 200/200 rollouts at 1.00 across all variants × binary/fold × doc 6k/14k (plus
  synth-fold regression), real max agent ctx 2721 (< 3000). Dry run: **700 traces / 46,052
  datums**, all 700 self-grade 1.00; narrativeqa 80/80 from cache.
- Inspection traces (one per variant): `niah_multi_traces.txt`, `vt_novel_traces.txt`

**Eval** (DOC=4000, budget=3000, N=3, `MAX_NODES=150`): in-dist spread + `niah_multi`, `vt_novel`;
OOD `oolong_*`, `niah_single_1`, `niah_multikey_1`, `niah_multiquery`, `vt`, `cwe`, `fwe`.

**Expected:** vt and niah_multikey_1 climb out of overflow (the direct analogs are now in-dist);
cwe/fwe/synth hold; OOLONG classification still capability-capped (unaddressed by design).
