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

## sft_general3 (run 3) — IN PROGRESS

**SFT config**
- Tasks: 15 synth + `realdoc_count` + `niah_novel` + `narrativeqa`
- `N_PER_TASK=20`, `realdoc_count:80,niah_novel:80,narrativeqa:80`; `SYNTH_STRATEGY=both`
- `CTX=3000`, `DOC=6000`, `DOC_MIX=6000:3,14000:1`, `CHUNK=200000`, `FOLD_LEAF_TOKENS=400`
- QA rebalancing on (`QA_NONE_KEEP=0.5`, `QA_ANSWER_DUP=2`); haiku leaf; `SAVE_NAME=sft_general3`

**Eval** (DOC=4000, budget=3000, N=3, `MAX_NODES=150`) — _results TBD_

**Expected** (pre-flight): varchain → ~1.0 (overflow fixed); RULER retrieval (niah/multikey) up
(niah_novel); narrativeqa modestly up (rebalancing, capability-capped); OOLONG likely still weak
(runaway fix is a bet, classification untouched).
