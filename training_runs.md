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

## sft_general4 (run 4) — 2026-09-08/09 — OVERALL 0.684 (57 rollouts)

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

**OOLONG post-mortem (run 3, 9 rollouts in `eval_results/raw/sft_general3_doc4k_b3k.txt`;
counting @24299/24743/24858, user @28980/29500/35243, temporal @35394/35591/35835).** Four
failure modes: (A) fuzzy per-item classification at the leaf — perfect tree, wrong TREC labels
(counting/trec 31 vs 14; imdb sentiment 1.00) — capability-capped; (B) **wrong accumulator
SHAPE**: temporal needs a joint month×label tally then "count months where label X wins", the
model tracked per-month OR per-label only (all 3 temporal, user/trec); (C) long multi-line
`Date || User || Instance` records + "in the above data" → line-number reasoning, over-reads
(95/159 reads >800 tok), raw `<tool_call>` text instead of a call → overflow / 150-node runaway
(user/agnews, user/imdb, counting/agnews); (D) one degenerate gold ("numeric value"). B and C are
curriculum gaps. Training has aggregation over PRE-LABELED fields (synth) and fuzzy judgment
WITHOUT aggregation (bookqa), never classification→tally, and never a 2-D state.

**Added for B — two synth VARIANT FAMILIES on a new month-record layout**
`[idx] id=.. mon=Aug grp=K2 amt=+7 flag=N` (tasks/synth/generators.py, oracle/synth.py):
- **`synth_filter_argmax`** — filter (flag=Y/N or mon=M) → per-grp Counter → MOST/LEAST common
  (exact; ties alphabetical). The oolong_user shape (filter by user → tally → argmin).
- **`synth_2d`** — joint `mon/grp` composite-key dict tally, root reduces along one axis: qtype ∈
  {**months_argmax** ×2 "for how many months is grp G strictly the most common", **months_cmp** "in
  how many months G1 > G2", **grp_in_month** (argmax within a month), **month_for_grp** (argmax
  month for a grp)}; 3–4 months per problem (5 months hit 2896 real tok in fold). Per-record lines
  are delta-only; the root prints the per-month table before boxing. The oolong_temporal shape.
- Verification: 320/320 at 1.00 (2 tasks × 4 variants × binary/fold × doc 6k/14k), real max ctx
  2756 (2d fold). Traces: `trace_snippets/month_traces.txt`.

**Added for C — `long_records`** (tasks/longrec, oracle/longrec.py): the document is a sequence of
MULTI-LINE entries — a short header + 60–250 tokens of real prose (novel or essay), blank-line
separated — and the question aggregates over entries. Layout ∈ {`=== entry 17 | src=B | tag=K3 ===`,
`Entry:/Source:/Tag:` key-value lines}; qtype ∈ {count_tag, src_most, tag_in_src (filter→argmax),
src_tag_2d (2-D reduce over src×tag), mention_count (entries whose body contains word W),
mention_most (entry with most whole-word occurrences of W; best-of monoid `best=17|n=3`)}. What it
teaches that nothing else did: ownership by where the HEADER starts, bodies straddling the leaf /
fold boundary (iterative reads extend across them — "The last entry's body is still cut off …"),
leaves that own 0–3 entries, and empty ranges that sit inside an entry owned by the previous range.
Base oracle got wording hooks (`_finish_phrase` / `_cutoff_phrase` / `_fold_header`) with defaults
identical to the old text, so every other task's traces are byte-for-byte unchanged (regression
checked). Follows SYNTH_STRATEGY=both. Verification: 240/240 at 1.00 (6 qtypes × 2 layouts × 2 prose
× binary/fold × doc 6k/14k), real max ctx 2590. Traces: `trace_snippets/longrec_traces.txt`.
A (fuzzy classification leaf) stays unaddressed — a later semantic per-entry variant ("how many
entries are about X") can drop into this same skeleton with the haiku leaf + rejection gate.

**SFT config** (run 3 + the five new tasks; niah/vt/long @80, month synths @20 like other synths)
- Tasks: 17 synth (incl. `synth_filter_argmax`, `synth_2d`) + `long_records` + `realdoc_count` +
  `niah_novel` + `niah_multi` + `vt_novel` + `narrativeqa`
- `N_PER_TASK=20`, overrides `realdoc_count:80,niah_novel:80,niah_multi:80,vt_novel:80,narrativeqa:80,long_records:80`
- `SYNTH_STRATEGY=both`, `CTX=3000`, `DOC=6000`, `DOC_MIX=6000:3,14000:1`, `CHUNK=200000`,
  `FOLD_LEAF_TOKENS=400`, QA rebalancing on, trace cache on, `PYTHONUNBUFFERED=1` (mid-run log)
- Verification: 200/200 rollouts at 1.00 across all variants × binary/fold × doc 6k/14k (plus
  synth-fold regression), real max agent ctx 2721 (< 3000). Dry run with all five new tasks:
  **820 traces / 25,271 agents / 54,353 datums**, all 820 self-grade 1.00; narrativeqa 80/80 from
  cache (QA verdicts pos 1201×2, neg kept 3358/6715).
- Inspection traces (one per variant): `trace_snippets/niah_multi_traces.txt`, `trace_snippets/vt_novel_traces.txt`

**Eval** (DOC=4000, budget=3000, N=3, `MAX_NODES=150`): in-dist spread + `niah_multi`, `vt_novel`;
OOD `oolong_*`, `niah_single_1`, `niah_multikey_1`, `niah_multiquery`, `vt`, `cwe`, `fwe`.

**Expected:** vt and niah_multikey_1 climb out of overflow (the direct analogs are now in-dist);
cwe/fwe/synth hold; oolong_temporal/user may pick the right accumulator shape now (B), but
OOLONG's classification leaf (A) and long-record layout (C) remain unaddressed in this run.

**Run 4 results** (ckpt `tinker://72155075-e0af-54b8-a1f8-e1068c35cb7e:train:0/weights/sft_general4`,
820 traces / 54,353 datums, 3398 batches, NLL 0.0190; SFT 19:30→03:06, eval →03:33; raw in
`eval_results/raw/sft_general4_doc4k_b3k.*`, chain log `sft_general4_chain.log`).

| task | run 3 | run 4 | note |
|---|---|---|---|
| vt | 0.00 | **1.00** | vt_novel did its job (was 3/3 overflow) |
| vt_novel / niah_multi (new, in-dist) | – | 0.95 / 1.00 | |
| niah_multiquery | 0.33 | 0.67 | |
| narrativeqa | 0.33 | 0.67 | |
| oolong_counting | 0.34 | **0.64** | trec still mislabels (20 vs 14); agnews 18 vs 17 |
| oolong_temporal | 0.17 | **0.48** | agnews relative_freq 1.00 with a real month/label 2-D state (`Apr 2022/World=2|…`) — the synth_2d shape transferred; trec most_common_label still 1-D (per-label only) |
| oolong_user | 0.00 | 0.33 | agnews user-subset count 1.00; imdb/trec still ignore the user filter |
| realdoc_count | 1.00 | 0.92 | 23 vs 22 |
| **cwe** | 0.97 | **0.00** | REGRESSION — see below |
| **fwe** | 0.89 | **0.00** | REGRESSION — see below |
| niah_single_1 | 0.67 | 0.33 | 2/3 overflow: root improvised a sequential 200-token read loop ("Folding the 1 results with no midpoint") |
| niah_multikey_1 | 0.33 | 0.00 | subtask drops the KEY ("compute the hidden number"), leaves return any number, root concatenates |
| synth (5) | 0.93 | 1.00 | |

**cwe/fwe regression = strategy choice, not capability.** All 6 rollouts picked LEFT-FOLD for a
per-word tally over a dense list (cwe: 77 records per 400 tokens; fwe: 40). The fold turn lists
every record (77 lines) and the accumulator has ~40 distinct keys, so the ROOT itself overflows
(fwe root turn = 88 lines, `turns=2 termination=overflow`); the model then improvises ("The chain
overflown — I'll restart…"). In run 3 the same tasks went binary (leaf partial tallies fit) →
0.97/0.89. Run 4 added three more tally-shaped tasks that render fold (`synth_filter_argmax`,
`synth_2d`, `long_records`), tipping the prior toward fold for "tally" questions. Every training
fold has a SMALL state (≤4 grps, ≤16 composite keys) — the model has never seen the rule "fold only
when the accumulator stays small; with an unbounded key space, split binary". Fix (training-side):
(1) teach state-size-aware strategy selection — the root preamble should state WHY it picks fold
(accumulator is a single number / a few keys) or binary (the tally can have many distinct keys, so
carry partials, not a chain); (2) add large-key-space tally variants (e.g. grp drawn from ~30 words)
rendered BINARY-only with that justification; (3) consider fold-rendering only int-state and
fold-native tasks. Also: dense lists (many records per 400 tokens) need a smaller fold slice or
binary — the per-record line cost dominates, not the state.

---

## sft_general5 (run 5) — PLANNED

**Change vs run 4 (one thing):** dial back fold. First principles (see run-4 notes): fold does NOT
reduce state pressure — it restates the FULL running accumulator ~4× per hop and every hop carries
it, whereas binary nodes carry only local partials except the top log₂ levels; fold's only hard
requirement is non-associative ops. `SYNTH_STRATEGY=both` now means: `runreset`/`varchain`/`vt_novel`
fold (required); scalar int-state ops (`sum`, `count`, `max`, `min`, `sumwhere`, `count2`,
`maxwhere`, `count_cmp`, `count_range`) get a **25 % fold share** (`SCALAR_FOLD_FRAC`) so fold stays
general where the state is one number; every counter/dict-state task (`mode`, `distinct`, `sumby`,
`diff`, `filter_argmax`, `2d`, `long_records`) renders **binary only**. Everything else identical
(tasks, N, DOC_MIX, budget, leaf model, cache).

**Expected:** cwe/fwe recover toward run-3 levels (binary lets leaves box a bounded top-k partial,
which the model invented on its own in run 3 — `trace_snippets/cwe_run3_vs_run4.txt`); OOLONG/vt/narrativeqa gains
from run 4 hold (they came from state/record shape, not from fold). Open decision: teach the top-k
sketch explicitly (large-vocabulary tally variant, binary-only) — pending inspection of the cwe traces.

**Also in run 5 (from the RULER-niah post-mortem):**
- **Per-task verb** (`ScaffoldOracle._verb`): reductions "compute", retrieval tasks (`niah_multi`,
  `vt_novel`) "collect". Run 4's multikey failure was a template collision — a retrieval question
  whose answer is a number was rendered "compute the hidden number" (key dropped) and the leaves ran
  the synth numeric template, literally ADDING the magic numbers (`25303820` = sum of other keys).
- **Key-naming goal phrases** in `niah_multi` explicit/multiquery/multivalue: "the magic number stated
  for {key} (collecting every stated key=magic number fact in the range)" — every subtask carries the
  key(s) down the tree.
- **Question paraphrase families** for `niah_novel`/`niah_multi`/`vt_novel` (`tasks/niah/generators.py`
  `_phrase`): optional instructional preface × 4–6 question forms × 4 answer-format tails, none copying
  RULER's wording. Why: on RULER `niah_single_1` the root failed at turn 1 — BEFORE reading anything —
  producing a subtask with literal `START..END` placeholders (run 4 seed 2013001) or a whole-doc read
  (run 3 seed 2013000); the "noise haystack" is not the cause, the root's question→subtask mapping is,
  and with one template per task it learned the template rather than the mapping.
- **`ROOT_DUP=4`**: the root's first turn (question → preamble → subtask) is the only non-self-similar
  step and the least-trained one (1 datum per trace ≈ 1.5% of datums); it is duplicated 4× at datum
  assembly, mirroring the QA-verdict rebalancing.
- **Filtered-question paraphrases** (`tasks/phrasing.py::filtered_question`, used by `synth_filter_argmax`,
  `synth_2d`/grp_in_month, `synth_count2`, `synth_maxwhere`, `long_records`/tag_in_src): the same
  condition stated in 7 positions/wordings (leading "consider only…" sentence, "restrict attention to…",
  inline "among ONLY…", trailing "considering only…", "filter … then answer", …). Why: OOLONG user
  seed 2100000 (`trace_snippets/oolong_user_run4.txt`) — the root applied the user filter in its own
  leaf but wrote "tallying each label" for the rest of the chain, so every downstream hop tallied all
  users. OOLONG states the filter as a separate leading sentence; every training question had it in one
  fixed inline position, so the root learned the position, not "restate the condition in the subtask".
  Gold and oracle op phrases unchanged (the oracle reads metadata, not question text).

**Run 5 results — 2026-09-09/10 — OVERALL 0.755 (57 rollouts)** (ckpt
`tinker://909847f4-bf8d-5406-b682-19d41f801f33:train:0/weights/sft_general5`, 820 traces / 59,939
datums, 3747 batches, NLL 0.0178; SFT 15:59→23:11 (survived a 17-min network drop via the new
per-call retry), eval →01:32; raw in `eval_results/raw/sft_general5_doc4k_b3k.*`. Training mix: 165
fold roots / 495 binary roots (was ~50/50); launch script now tracked: `scripts/run_sft5_eval.sh`.)

| task | run 3 | run 4 | run 5 | note |
|---|---|---|---|---|
| niah_single_1 | 0.67 | 0.33 | **1.00** | root now writes a proper retrieval subtask on RULER phrasing (was START..END) |
| niah_multikey_1 | 0.33 | 0.00 | **1.00** | key carried in every subtask (was summed) |
| niah_multiquery | 0.33 | 0.67 | 0.83 | |
| cwe | 0.97 | 0.00 | **0.93** | binary again (top-10 per leaf) |
| fwe | 0.89 | 0.00 | 0.78 | |
| niah_multi / vt_novel / niah_novel | – | 1.00/0.95/1.00 | 1.00/0.98/1.00 | |
| synth (5) | 0.93 | 1.00 | 1.00 | |
| **vt** | 0.00 | 1.00 | **0.20** | REGRESSION: root chose BINARY "collect every assignment statement" (the niah_multi template), merged the statements as a set, then resolved only the literal-assigned var. Run 4 folded (`vt_novel`-style) and got 1.00. The retrieval-shaped question + "collect" verb pulled vt toward the collect template; vt_novel is fold-only so "collect + binary" never appears with VAR data in training. |
| **realdoc_count** | 1.00 | 0.92 | **0.55** | one leaf (3250..3500, 4 occurrences) OVERFLOWED at turn 2 — a generation loop while enumerating occurrences — and its parent lost the partial (6 vs 11). Other leaves audited correct against the unclipped reads. Seed 2 19 vs 22 similar. |
| oolong_counting | 0.34 | 0.64 | 0.34 | agnews: children over-read → overflow → node cap; trec 28 vs 14 (labels) |
| oolong_temporal | 0.17 | 0.48 | 0.39 | trec: 2-D state with INCONSISTENT key formats (`Apr=…`, `Apr 2023:entity=1`, `Apr 2025/…`) → merge blew the budget; imdb 1.00; agnews 0 vs 6 |
| oolong_user | 0.00 | 0.33 | **0.00** | agnews: root improvised a 3-way split AND dropped the user filter again (tally over all 37 articles → 17 vs 1); imdb 7–7 tie → 'none'; trec wrong label |
| narrativeqa | 0.33 | 0.67 | 0.33 | both misses are `none`/`none` at the root (abstention back) |

**Reading:** the three targeted fixes did what they were for — every RULER retrieval task recovered
(single 0.33→1.00, multikey 0.00→1.00, cwe/fwe back to ~0.9/0.8). But two in-distribution-adjacent
tasks fell: vt (strategy choice flipped to binary-collect) and realdoc (a leaf generation loop),
and OOLONG gave back most of run 4's gains. Net +0.07 OVERALL, but the OOLONG/vt part says the
fixes shifted priors rather than adding capability: fewer folds + a strong "collect" template for
retrieval-looking questions moved vt off the fold it had learned. N=3 caveat applies to every row.

---

## sft_general6 (run 6) — 2026-09-10/11 — held-out SCORE 0.675 (45 rollouts, N=5), diagnostics 0.946

**Target:** the run-5 post-mortem — strategy drift (vt → binary-collect), a realdoc leaf loop, OOLONG
2-D key-format inconsistency and dropped filters — plus the user's read that paraphrased questions
leave the root without an exact template, so *more data / more examples* is the lever.

**Changes (all training-side)**
1. **Strategy is a PROPERTY OF THE TASK, not a knob.** Sequential (order-dependent) tasks left-fold,
   everything else splits binary; `SYNTH_STRATEGY=both`/`SCALAR_FOLD_FRAC` are gone (`sft.py`
   `_synth_renderings` just returns each task's `strategy_default`). The root preamble now STATES
   the reason: fold — *"This task is order-dependent: {reason} — so I process the document left to
   right with a running accumulator…"*; binary — *"The result over a range does not depend on the
   order of the records, so I split the range in half recursively…"* (`_sequential_reason` hook).
2. **Four new sequential synth tasks** (all dict state, fold): `synth_peak` (max the running total
   ever reaches), `synth_streak` (longest run of consecutive flag=Y), `synth_adjacent` (records whose
   amt > previous record's), `synth_first_exceed` (first index where the running total exceeds T;
   sometimes unreachable → `none`). Plus a sequential `long_records` variant **`first_reach`** (first
   entry at which the cumulative tag count reaches N) — fold on long prose records. Fold tasks: 3 → 8.
3. **`vt_novel` reassignments** (60% of problems re-bind 1–2 vars later): with each var assigned
   once, collect-then-resolve was technically valid (and is what the run-5 root did); re-binding makes
   fold REQUIRED, and gold is threaded over the final record order.
4. **realdoc frequent entities**: half the problems pick an entity with ≥ ~3 hits per 500-token leaf,
   so leaves practice enumerating many near-identical occurrences and stopping (run 5's leaf loop).
5. **Root dictates the state format** (`_state_format` hook): every binary subtask ends with
   *"Return the partial tally as `<mon>/<grp>=<count>` entries joined by `|` in \boxed{}"* (per kind:
   single integer / `<key>:<count>` / `|`-joined values / `<key>=<value>`; niah_multi and 2-D tasks
   override). Siblings serialize identically by construction; the merge is mechanical.
6. **2-D diversification.** `synth_2d`/`synth_filter_argmax` draw a field SCHEME per problem — mon×grp,
   period("Mar 2024")×label("Business|Sci/Tech|Sports|World"), region×product, dept×status,
   site×code — with 3–8 outer values (joint key space capped at 18: 40 keys measured 3.4k real
   tokens at the root, 24 → 2.96k, 18 → ≤2.5k). Keys split at the FIRST `/` so inner values may
   contain `/`. `long_records` header schemes likewise: src×tag (3×4, 5×3), region×topic,
   team×status, author×genre; the context lists the outer vocabulary in natural order (tie rule).
7. Data scale-up: synth 20→40, `synth_2d` 80, `synth_filter_argmax` 60, `vt_novel` 200,
   `realdoc_count` 160, `long_records` 160, `niah_multi` 120, `niah_novel` 100, narrativeqa 80 (cache).
   `ROOT_DUP=4` kept. Launch script: `scripts/run_sft6_eval.sh` (eval adds synth_peak, synth_2d,
   long_records to the in-dist list).

**Verification:** every task × its strategy × doc 6k/14k at 1.00 — sequential 7 tasks 280/280
(max ctx 2759, varchain), binary 8 tasks 320/320, diversified synth_2d 120/120 (max 2520),
synth_filter_argmax 80/80, long_records 120/120 (incl. first_reach 16, max 2302), regression on the
untouched synth tasks + niah_novel. Traces: `trace_snippets/longrec_first_reach_traces.txt`.

**Eval protocol (from run 6 on).** Decision criteria made explicit:
1. **Held-out rows are the SCORE** — `SCORE_TASKS` = oolong_counting/user/temporal, niah_single_1,
   niah_multikey_1, niah_multiquery, vt, cwe, fwe (never trained on, fixed seeds) at **N=5** each
   (45 rollouts). `eval/run.py` prints `SCORE (held-out …)` separately from OVERALL.
2. **In-dist rows are DIAGNOSTICS**, one per mechanism so a failure localizes, at N=2: synth_sum
   (scalar reduce), synth_mode (tally), synth_2d (2-D), synth_peak (sequential), long_records (long
   records), realdoc_count (prose count), vt_novel (fold on prose), niah_multi (retrieval),
   narrativeqa (model leaf). Reported as `DIAGNOSTIC (in-dist)`, excluded from the score.
3. A new task gets a diagnostic slot for the run it's introduced, then rotates out unless it IS the
   mechanism's diagnostic. 4. N=3 was a smoke test, not a measurement — run-5 decisions swung on
   single seeds. 5. Eval format stays general; never guarded or tuned to the model.
Run-6 eval: 63 rollouts (was 66) with a headline that excludes the in-dist 1.00s that inflated
OVERALL (run 5: 0.755 overall vs ~0.55 on the held-out rows).

**Outage robustness (added before run 6; Xfinity).** Two layers:
- In-process: every per-batch Tinker call retries with backoff for up to `SFT_RETRY_MINUTES` (30).
- Across processes: every `SFT_SAVE_EVERY` (300) batches sft.py saves trainer state (weights + Adam) to
  `<SAVE_NAME>_resume` with a **48 h TTL** (`SFT_RESUME_TTL_HOURS`) and writes a local cursor
  `~/.cache/infinite-context/sft_resume_<SAVE_NAME>.json`. On restart with the same config it
  reloads that state into a fresh training client (`load_state_with_optimizer`) and skips the
  completed batches; data regeneration is deterministic (seeds, trace cache, seeded shuffle), checked
  via `n_datums`. On completion the intermediate checkpoint is deleted (`delete_checkpoint_from_tinker_path`)
  and the cursor removed; the final checkpoint never expires. `scripts/run_sft6_eval.sh` loops the SFT
  stage (up to 200 attempts, 5 min apart → resumes) and the eval stage (20 attempts, rerun from scratch).
  Live-tested: killed a tiny run after its 2nd checkpoint, relaunched → "RESUMING … at batch 4 /
  skipping 4 completed batches" → completed, deleted the intermediate, cleared the cursor.
  Note: a restart creates a NEW Tinker training run id, so an intermediate checkpoint from a killed
  attempt is only expired by its TTL, not deleted (the final run deletes its own).

**Run 6 results** (ckpt `tinker://7161d592-9d8a-5583-9ff1-13a273154e96:train:0/weights/sft_general6`, 1,720
traces / 127,557 datums, 7,973 batches, NLL 0.0088; SFT 14:38→06:55 incl. a ~10-min network drop that
recovered in place; eval 06:55→07:15; raw in `eval_results/raw/sft_general6_doc4k_b3k.*`). First run with
the restructured eval: held-out 9 tasks × 5 seeds = SCORE; in-dist 9 × 2 = diagnostics. Seeds 3–4 are new
and add two OOLONG datasets (negation, yahoo), so SCORE is not comparable to run 5's 0.755; seeds 0–2 are.

| held-out task | run 5 (s0–2) | run 6 (s0–2) | run 6 (N=5) | note |
|---|---|---|---|---|
| niah_single_1 | 1.00 | 1.00 | **1.00** | |
| niah_multikey_1 | 1.00 | 1.00 | 0.80 | s4: children boxed garbage (`QUERY=1323691|1323691=1323691…`) → none |
| niah_multiquery | 0.67 | 0.67 | 0.80 | s0: both halves `none` (leaves missed all 4 needles) |
| cwe | 0.93 | ~0.93 | **0.96** | |
| fwe | 0.78 | ~0.89 | **0.93** | |
| vt | 0.20 | 0.40 | 0.44 | see below |
| oolong_counting | 0.34 | 0.14 | **0.09** | see below |
| oolong_temporal | 0.39 | 0.69 | 0.46 | agnews 1.00, imdb 0.75; negation "dates represented exactly once": children returned INCONSISTENT states (one `10`, one the full per-date tally) |
| oolong_user | 0.00 | 0.33 | **0.60** | the two NEW datasets (negation, yahoo — user argmax / relative-freq) both 1.00; imdb = sentiment, trec = degenerate gold |
| diagnostics (9 in-dist × 2) | — | — | 0.946 | all 1.00 except long_records `mention_most` (leaf undercounted 'have': best=1|n=3 vs gold entry 13) |

**vt (0.44): the RULE now fires, the fold EXECUTION on RULER's noise haystack is what fails.** 4/5 roots
chose fold with an (invented but sensible) order-dependence reason; 2 of those ran the chain correctly
(1.00). s0: the root folded its first slice, then instead of delegating kept folding the NEXT slice itself
("Folding the 0 records whose line STARTS in 400..800") → self-loop → overflow. s3: the root's first slice
was all noise → accumulator `none` → the depth-1 child folded and RETURNED `none` without delegating the
rest → chain truncated. s1: root mis-classified vt as order-independent ("collect every variable assigned
to 94850") → binary → 1 var. So: strategy prior fixed (run-5 problem), but the fold protocol is fragile
when slices contain 0 records — every training fold slice has records (synth dense; vt_novel/longrec
sparse but rarely empty at 400 tokens). Fix: train fold on documents where whole slices are empty (noise
filler share ↑ for vt_novel; explicit "(no VAR assignment starts here) → accumulator unchanged, delegating
the rest" hops), so "empty slice ⇒ still delegate" is demonstrated.

**oolong_counting (0.09): leaf classification + one format miss.** agnews: correct tree, tally `World=16`
(gold 17), root wrote "Answer: 16" but NO `\boxed{}` → `stopped_no_answer` (would have scored ~0.9).
imdb: tally negative 5 / positive 6 on ~11 reviews, gold says negative — per-item sentiment. negation: the
leaves INVENTED labels (`none`, `unrelated`, `not relevant`, `label=not in context`) instead of the
True/False label space → 4 vs 23. yahoo relative_freq: wrong direction; trec 17 vs 14. Every miss but one
is the fuzzy-classification leaf (failure mode A) — the only lever left there is a classification leaf task.

**Reading.** RULER retrieval/tally is now stable at ~0.9–1.0 across 5 seeds. The run-5 strategy drift is
fixed as a *rule* (fold chosen for vt), and the 2-D/filter machinery transfers (oolong_user 0.60, temporal
agnews 1.00). What remains: (1) fold robustness to EMPTY slices (vt s0/s3); (2) the OOLONG classification
leaf (counting 0.09 is almost entirely this); (3) a "print the boxed answer" slip at one OOLONG root;
(4) leaf enumeration accuracy on very common words (long_records mention_most).

**Empty-slice fold fix (after run 6, for run 7).** (1) Base fold node: an empty slice is narrated as a
normal step that continues the chain — *"No VAR assignment lines start in 2000..2400 — the accumulator is
unchanged. → accumulator = … Delegating the rest 2400..6155 with the unchanged accumulator."* (was
"Folding the 0 records … (no VAR assignment starts here)"). Applies to every fold task. (2) `vt_novel`:
filler ∈ {noise ×2, novel, essay} (RULER vt is noise-filled; was noise ×1 of 4) and chains ∈ {1 ×2, 2, 3}
(was 2–3), so most 400-token slices are empty, as in RULER vt at 4k. Verified vt_novel 60/60,
synth_runreset 60/60, long_records 60/60 (incl. first_reach) at 1.00, max ctx unchanged (2626).
Run-6 training already had 26% empty fold hops; the failures were specific to noise haystacks where
nearly every hop is empty and the model had to keep delegating anyway.

---

## sft_general7 (run 7) — 7w 0.657 · 7w2 0.666 · from-base (corrected, 09-17) held-out 0.662, diagnostics 0.911

**Target:** the run-6 post-mortem. (a) fold chains on nearly-empty haystacks (vt s0/s3) — DONE above
(empty-slice narration + sparse/noisy vt_novel). (b) Leaves that JUDGE items collapsing to one line
(`label=true count=10`) and INVENTING label names on OOLONG negation.

**Changes**
1. **Key space in the format contract** (`_key_space` hook): the root now dictates the allowed KEYS, not
   just the shape. Closed sets enumerate — *"…where `<key>` is exactly one of K1, K2, K3, K4"*, *"`<period>`
   is one of Nov 2023, Jan 2024, … and `<label>` is one of Business, Sci/Tech, Sports, World"* (synth mode /
   distinct / sumby / diff / filter_argmax / 2d, long_records src_most / tag_in_src / 2d, rule_label);
   open sets state what a key is — *"each `<key>` is the key name exactly as written in the text"*
   (niah_multi). Both forms are in training so the root learns to pick one.
2. **`rule_label`** (tasks/rulelabel, oracle/rulelabel.py): classification-SHAPED aggregation over real
   prose with an exactly checkable label. One sentence per line, tagged `[S<n>]` by section; the question
   states a rule the reader must apply per sentence — dialogue/narration (contains a quotation mark),
   question/statement (ends with ?), numeric/plain (contains a digit), named/unnamed (capitalized
   non-initial word) — and asks count / most_common / relative ("more common than…") / sections_cmp (2-D)
   / section_most (2-D). The leaf line SHOWS the judgment per item:
   `- S2 "…she said, 'come in.'" → contains a quotation mark? yes → dialogue → S2/dialogue=3`.
   Teaches the procedure classification needs — judge each item, write the check, tally — while the
   decision stays mechanical; the label is NOT in the text. Not OOLONG's data, labels, or layout.
   Verified 80/80 (4 rules × 5 qtypes, doc 6k/14k), max real ctx 2585.
3. Data: run-6 mix + `rule_label:160`; eval adds `rule_label` to diagnostics (N=2). Launch:
   `scripts/run_sft7_eval.sh`. Bet stated plainly: this teaches per-item enumeration under *mechanical*
   judgment and hopes it transfers to fuzzy judgment; it will not raise TREC/imdb label accuracy itself.

**Cost reduction (2026-09-15; from-base runs had reached ~$240).**
- **`DATUM_MODE=agent`** (default): one datum per agent conversation with every assistant turn weighted
  (`ALL_ASSISTANT_MESSAGES`), instead of one datum per turn each re-paying the system prompt + tool
  schema + earlier turns. Verified exactly equivalent on our thinking-free traces — every per-turn datum
  is a strict prefix of the full render and the weight masks coincide (275/275 turns, 123/123 agents) —
  so the supervision is identical at 1.9× fewer tokens. The cookbook's generic "extension property"
  warning is silenced for this renderer; `DATUM_MODE=turn` restores the old behaviour.
- **`INTERNAL_KEEP=0.3`**: pure split/combine agents (depth>0, no read_chunk) are near-identical to each
  other; keep 30%. Roots and every reading agent (leaves, fold hops) are always kept.
- **Warm start** (`INIT_CHECKPOINT=tinker://…`, `LR`): load a previous run's weights into the fresh LoRA
  client (fresh optimizer) and train on new/changed tasks + a small replay. Live-tested: loading
  sft_general6 gave NLL 0.0000 on synth_sum from the first batch (a fresh LoRA starts ~0.25).
  Policy: warm-start for iterating on new tasks (`scripts/run_sft7_warm.sh`: rule_label 160, vt_novel 120,
  ~20% replay of the rest, LR 5e-6); full from-base runs (`run_sft7_eval.sh`) remain the reference.

**Run 7w results — warm start from sft_general6** (ckpt `tinker://47c4be41-3212-5e02-8b9b-71f8ebe54d41:train:0/weights/sft_general7w`;
598 traces / 16,738 agent-datums / 30.5M tokens, 1,047 batches, LR 5e-6, NLL 0.0032; SFT 1 h 20 m, eval 8 min;
≈15% of a from-base run's cost. Raw: `eval_results/raw/sft_general7w_doc4k_b3k.*`.)

| held-out | run 6 | 7w | what the traces show |
|---|---|---|---|
| niah_single_1 | 1.00 | 1.00 | |
| niah_multikey_1 | 0.80 | **1.00** | |
| niah_multiquery | 0.80 | **0.90** | |
| oolong_user | 0.60 | **0.75** | imdb now correct (negative 5 / positive 3) |
| oolong_counting | 0.09 | **0.32** | negation: root now DICTATES the key space ("`<label>` is exactly one of `True, False`") and the leaves ENUMERATE per item ("line 0 … → label=False → False:1") — the procedure landed — but judged every claim False (67 vs 23): the classification ceiling, cleanly exposed |
| cwe | 0.96 | 0.68 | still top-10-per-leaf, but partials now admit junk keys (e-book:2, earth:1) that crowd out rarer common words; answer boxed as a tally |
| fwe | 0.93 | 0.73 | s3: children returned `0` / `none` to an open-key-space contract → format drift |
| oolong_temporal | 0.46 | 0.17 | agnews: root chose a 1-D per-label tally ("how many months have each label") — lost run 6's month×label 2-D state |
| vt | 0.44 | 0.36 | 4/5 roots went BINARY-collect again ("does not depend on the order of the assignments… collect every variable assigned to 94316"); run 6 had 4/5 fold. vt_novel diagnostic itself 1.00 (fold) |
| diagnostics (10 × 2) | 0.946 | 0.953 | rule_label 1.00; one synth_sum child arithmetic slip (62 + −109 vs gold) |

**Reading.** The two targeted changes did what they were built for — the root dictates label spaces
and leaves judge-then-enumerate (visible on OOLONG negation), and rule_label is learned. But the
warm start also DRIFTED on behaviours the 20% replay under-covered: 2-D shape selection (temporal),
the fold-for-vt rule, and the open-key-space partial format (cwe/fwe). Fold share in the warm mix was
the same ~30% as run 6, so this is optimisation dynamics (1,047 steps at 5e-6 on a 598-trace mix),
not curriculum. Consequence for the process: a warm-start iteration answers "did the new task/format
land?" reliably (yes) but its held-out SCORE is confounded by drift, so it should not be read as the
run-to-run scoreboard. Options: heavier replay (≥40%) / lower LR (2e-6) for warm starts, or treat only
from-base runs as the reference. The from-base run 7 now costs ≈$110 with agent-datums.

**`labeled_records` (added 2026-09-15, for run 7 proper).** The proven OOLONG-SFT recipe, generalized:
real labeled text with the DATASET's gold label per item (no model calls). Datasets with taxonomies
disjoint from OOLONG's: DBpedia-14 (4–6 classes sampled per problem), dair-ai/emotion (6), Yelp polarity
(2); cached by `scripts/cache_labeled.py`. Layout `[S<n>] [by <author>] <text>` (not OOLONG's
`Date || User || Instance`); the task context states the label set and what a label means; the
root's contract enumerates the labels. Leaf line = the judgment per item:
`- S1 [by Han] "i have stopped feeling surprised" → label: surprise (counts) → count=1`.
Questions: count / most_common / relative / sections_cmp / section_most (2-D) / author_most
(filter → argmax, the oolong_user shape). Verified 80/80 (6 qtypes × 3 datasets, doc 6k/14k), max
real ctx 2681. Traces: `trace_snippets/labeled_records_traces.txt`, `…_dbpedia.txt`. Added to both
run-7 scripts at 160 (from-base) / 160 (warm). RL is off the table for now (earlier attempts were
unstable and degraded the SFT policy); gold-label SFT is the path for leaf judgment.

**Run 7 (from base) results — 2026-09-16 — held-out 0.573, diagnostics 0.888 — NOT a valid read of the
curriculum.** (ckpt `…/weights/sft_general7`; 2,040 traces / 57,597 agent-datums / 103M tokens, 3,600
batches, NLL 0.0140; SFT 5 h 20 m, eval 9 min; raw `eval_results/raw/sft_general7_doc4k_b3k.*`.)
Held-out: single 0.80, multikey 0.60, multiquery 1.00, cwe 0.58, fwe 0.53, vt 0.24 (3 overflow),
counting 0.28, temporal 0.32, user 0.80. Diagnostics: narrativeqa 0.50, long_records 0.52, labeled 0.75.

**Root cause: the two "free" cost changes changed the optimisation, not just the token bill.**
1. `INTERNAL_KEEP=0.3` removed 70% of split-only agents — and those carry the LEAF-BOUNDARY rule.
   At exactly-500-wide ranges run 6 split 144× / read 15×; run 7 split 20× / read 153×. Leaves went from
   250 to 500 tokens (321 vs 153 first reads), so dense leaves (fwe word lists, multikey, vt fold slices
   read at 500 instead of 400) overflowed. Training docs (6k/14k) never halve to exactly 500, so the
   boundary case was learned only from the general split examples — the ones I thinned.
2. `DATUM_MODE=agent` at batch 16 = 2.2× fewer optimizer steps (3,600 vs 7,973). Per-step NLL matched
   run 6 (0.0140 at 3,600 vs run 6's 0.0179 there) but the run stopped half-fit (run 6 ended 0.0088).
   Token-level equivalence ≠ optimisation equivalence.
3. The vt roots also turned the new empty-slice wording into a self-loop ("No record starts in
   700..900 — the chain is unchanged" then read the NEXT slice itself instead of delegating) —
   plausibly a symptom of the under-fit fold protocol rather than the wording.

**Fixes (in): `INTERNAL_KEEP=1.0` default; `SFT_BATCH_SIZE` env (scripts use 8 in agent mode → step
count restored at the same token cost); `DOC_MIX=6000:2,8000:1,14000:1` so training halves land on
exactly 500 (8000→4000→2000→1000→500→250) and the boundary rule is demonstrated, not inferred.**
Agent-datum saving stands at ~1.7× (116M vs 196M tokens for this mix). The curriculum changes since
run 6 (rule_label, labeled_records, key-space contracts, empty-slice fold, vt reassignments) remain
untested by a clean from-base run.
- **Leaf-boundary rule made consistent and demonstrated.** The oracle split on `range > 500` while every
  subtask says "until the range is less than 500" — they disagreed at exactly 500, which is what a
  4000-token eval doc halves to (run 6's model split there 144:15 on its own; run 7's read 153:20).
  Now both oracles (base, BookQA) split on `range >= LEAF` with the text "Range a..b is not below 500;
  splitting…". An 8000 tier would have halved to 499-token LEAVES (packing stops just under the tier),
  teaching the opposite; the tier is **8400** (→ 525 → split → ~262-token leaves): verified 400
  just-above-500 splits and zero ≥500-token leaf reads across 8 task types. Trace cache bumped to v2
  (narrativeqa regenerates once, ~$10 of haiku).

**Run 7w2 results — 2026-09-16 — warm start from sft_general6 (same start as 7w) + labeled_records +
boundary fixes + batch 8.** (ckpt `tinker://b85e0baa-5cad-5441-b9c4-ed6f47c0d466:train:0/weights/sft_general7w2`;
758 traces / 32,509 agent-datums / 54.8M tokens, 4,064 batches, NLL 0.0027; SFT ~9 h wall (batch 8 doubles
steps without halving step time), eval 10 min; ≈$67 vs ≈$170 from base. Raw `eval_results/raw/sft_general7w2_*`.)
**Held-out 0.666, diagnostics 0.989.**

| held-out | run 6 | 7w | 7w2 | note |
|---|---|---|---|---|
| **oolong_counting** | 0.09 | 0.32 | **0.71** | imdb / agnews / yahoo 1.00; trec 12 vs 14 (0.56); negation still all-False (46 vs 23) |
| niah_single_1 | 1.00 | 1.00 | 1.00 | |
| niah_multikey_1 | 0.80 | 1.00 | 1.00 | |
| **cwe** | 0.96 | 0.68 | **0.92** | leaf boundary restored: at 500-wide ranges 211 splits : 1 read; leaves 250 again |
| niah_multiquery | 0.80 | 0.90 | 0.80 | |
| vt | 0.44 | 0.36 | 0.48 | 2 fold overflows, one partial |
| fwe | 0.93 | 0.73 | 0.53 | 3 partials (a junk 3rd word), one `none`, one `hcuuzt:255` |
| oolong_user | 0.60 | 0.75 | 0.40 | imdb `stopped_no_answer`; negation picked user 33202 (gold 49255); trec degenerate |
| oolong_temporal | 0.46 | 0.17 | 0.15 | trec 22 vs 8; agnews 1 vs 6; negation overflow |
| diagnostics (11 × 2) | 0.95 | 0.95 | **0.99** | labeled_records 0.875 (yelp 25 vs 26); everything else 1.00 |

**Reading.** The gold-label leaf task did what the OOLONG-only SFT did: counting 0.32 → 0.71 with the
leaves now labelling each item ("→ label: description and abstract concept → …: 2") against the dictated
label set — trec is within 2 of gold, imdb/agnews/yahoo exact. The boundary fixes verifiably restored
250-token leaves (cwe back to 0.92). Warm-start drift persists on the tasks that had little replay
(temporal 2-D shape, fwe partial format, user filter), same as 7w — expected, and why the from-base run
remains the reference. Negation is the one OOLONG set where the leaf's judgment itself is wrong
(every claim "False"); labeled_records has no claim-verification analogue.

**Three additions for OOLONG temporal/user shapes (2026-09-16, after 7w2), all in `labeled_records`:**
1. **Derived keys.** Half the problems tag items with a full DATE (`[Jul 28, 2022]`) instead of a section;
   the 2-D questions are per MONTH, so the outer key must be derived — the leaf line shows it:
   `- [Apr 15, 2025 → Apr 2025] "i am feeling generous and…" → label: joy → Apr 2025/joy=1`. Context and
   contract say "the month and year of the item's date, one of `Nov 2022`, `Sep 2023`, …". Every other
   training key was copied verbatim; OOLONG temporal's month key is derived from `Date: Jul 28, 2022`.
2. **`dates_rep_k`** — "considering only items dated in Nov 2025, how many distinct dates are represented
   exactly k times" — a per-date tally then COUNT-OF-KEYS-WITH-COUNT-k reduction (OOLONG temporal's
   `represented_n_times`, 0.00 in every run). Month-scoped so the dict stays ≤ ~14 keys.
3. **`author_top`** — "which author has the most `joy` items": argmax over the OUTER key of a label count
   (OOLONG user's "which user has the most X"); the existing `author_most` is the reverse (filter → label).
Budget work to fit the 14k tier: joint key space capped at ~18 (months × labels), combine turns no
longer re-list both children's tallies for counter/dict states (they are in the tool results above —
this halves internal-node state cost for EVERY tally task), author tag only on author questions,
snippets 32 chars, label list not repeated inside the goal phrase. Verified 120/120 (9 qtypes × 2
key modes × 3 datasets, doc 6k/14k). `labeled_records` count 160 → 200 in the from-base script.

**Run 7 (from base, corrected config) results — 2026-09-17 — held-out 0.662, diagnostics 0.911.**
(ckpt `tinker://8ec58e87-7310-55dd-a31d-9f9caf3c0382:train:0/weights/sft_general7`; 2,080 traces /
87,330 agent-datums / 143M tokens, 10,917 batches at batch 8, NLL **0.0058** (run 6: 0.0088); SFT 22:26→21:11
(~23 h — batch 8 doubles steps without halving step time), eval 19 min; ≈$175. Raw `eval_results/raw/sft_general7_*`.)

| held-out | run 6 | 7w2 (warm) | **run 7** | what happened |
|---|---|---|---|---|
| **oolong_counting** | 0.09 | 0.71 | **0.60** | trec **14 exact**, imdb 1.00, agnews 16/17, negation 28 vs 23 (no longer all-False), yahoo relative wrong (10-class leaf labels noisy) |
| oolong_user | 0.60 | 0.40 | 0.40 → **~0.6 real** | trec degenerate now correct; yahoo has the RIGHT user but boxed `User: User 30140` (doubled prefix → 0); imdb/negation wrong |
| oolong_temporal | 0.46 | 0.15 | 0.27 | the month×label 2-D state now appears 3/5 (trec, agnews, yahoo — `Apr/World=2`); answers off by leaf-label noise (agnews 3 vs 6); imdb 1-D; negation: document-wide per-DATE dict → overflow (training only scopes dates to a month) |
| niah_single_1 / multikey_1 | 1.00 / 0.80 | 1.00 / 1.00 | **1.00 / 1.00** | |
| niah_multiquery | 0.80 | 0.80 | **0.95** | |
| vt | 0.44 | 0.48 | 0.48 | no overflows now; chains return 1–3 of 5 vars; s1 still binary-collect |
| **cwe** | 0.96 | 0.92 | **0.52** | REGRESSION: the root now DICTATES a contract, and for this open key space it dictated "the 10 most common words as a comma-separated LIST" — no counts — so the merge is a set union; some leaves listed item numbers (605, 606) as words |
| fwe | 0.93 | 0.53 | 0.73 | one `c10, c11, c12`, one junk third word |
| diagnostics (11 × 2) | 0.95 | 0.99 | 0.911 | labeled_records / rule_label 1.00; narrativeqa 1 abstain; long_records mention_most ('have') 0.03 again |

Leaf boundary at eval: 156 splits vs 38 reads at ≥500-wide (run 6: 144:15; invalid run 7: 20:153) — restored,
not perfect (the 38 are mostly RULER niah/fwe leaves).

**Reading.** Third from-base run in a row at ~0.66–0.68 held-out with the composition shifting under it:
what we target moves (counting +0.51, retrieval to ~1.0, 2-D state now chosen on temporal), and a
behaviour we didn't guard regresses. This time it's cwe: the *root-dictates-format* habit is strong enough
that on an open vocabulary the root wrote a lossy contract (a word list, not `word:count`), and the model
obeyed it. The general fix is to TEACH the open-set tally contract — a top-k-frequent-words task over
non-RULER word lists whose partials are pruned `word:count` tallies — rather than hope the run-3 heuristic
survives. Also small: the `User: User X` double prefix is an answer-format slip worth a paraphrase of
"answer in the form …" instructions in training questions.

Recipe note for the next from-base run: batch 16 with peak LR 2e-5 + linear decay has the same total
movement as this run in half the wall time (see 09-17 discussion); do not stack it with curriculum changes.

---

## sft_general8w (run 8, warm start from sft_general7) — 2026-09-17/18 — regression recovery

**Goal:** claw back run 7's regressions with minimal new surface (~10 h window; a from-base run doesn't fit).
Warm start from `sft_general7` (keeps the counting gains), 25% replay, batch 16, LR 5e-6.

1. **`synth_topk`** (tasks/topk, oracle/topk.py) — the open-vocabulary tally contract the root wrote WRONG on
   cwe in run 7 ("a comma-separated list", no counts → set-union merge, 0.96 → 0.52). Word lists over a
   PG-essay vocabulary (not RULER's), two layouts (numbered `<n>. word` / plain stream); k ∈ {3,5,8} hot
   words at 3–5 occurrences per ~250-token leaf (cwe's regime), background one-off. Leaves narrate in 3-line
   blocks ("abandoned×2, function×2, …; +35 one-off words") and return a PRUNED tally — the most frequent
   words seen 2+ times, at most 2k (≥10), exact counts; the contract says so and adds "never a bare word
   list, never list numbers as words". Merge adds counts; root ranks. Verified 80/80, max ctx 2485.
   Built-in lessons from getting there: hot density must be per-LEAF (12–40× per doc left leaves with tied
   singletons — the range-mode no-guarantee case); uncapped "2+" tallies grow with range size (cap at m).
2. **Answer-form following** — 35% of exact-answer `labeled_records` questions now end "Give your final
   answer in the form 'Label: [X]' …" and the gold is the filled template (run 7 boxed `User: User 30140`).
3. Data: synth_topk 120, labeled_records 120, rule_label 40, vt_novel 50, replay ~10–40 elsewhere:
   **720 traces / 31,042 datums / 52M tokens** (≈$65), all 1.00 in the dry run. Eval adds synth_topk as a
   diagnostic. Script: `scripts/run_sft8_warm.sh`; output `/tmp/eval_general8w.*`.

**Run 8w results — 2026-09-18 — held-out 0.758 (best so far), diagnostics 0.958.**
(ckpt `tinker://75b29e89-caf6-5e79-8415-5f4e64af39a3:train:0/weights/sft_general8w`; 720 traces / 31,042
datums / 52M tokens, 1,941 batches at batch 16, NLL 0.0136; SFT 23:54→02:23 (2.5 h), eval 8 min; ≈$65.
Raw `eval_results/raw/sft_general8w_*`.)

| held-out | run 6 | run 7 | **8w** | note |
|---|---|---|---|---|
| **cwe** | 0.96 | 0.52 | **1.00** | root now dictates the `word:count` pruned-tally contract on every cwe/fwe rollout (272+136+84 subtasks) |
| fwe | 0.93 | 0.73 | **0.93** | one `none` third word |
| **vt** | 0.44 | 0.48 | **0.88** | all 5 fold; two chains dropped 1–2 vars — first time vt is high without a vt-specific change |
| niah_single_1 / multiquery | 1.00 / 0.80 | 1.00 / 0.95 | **1.00 / 1.00** | |
| niah_multikey_1 | 0.80 | 1.00 | 0.80 | s4 picked another key's number |
| oolong_user | 0.60 | 0.40 | 0.60 | negation/agnews/imdb right; yahoo STILL `User: User 30140` — the doc writes ids as "User 30140" and the model copies that token; our form templates only cover labels/authors/sections |
| oolong_counting | 0.09 | 0.60 | 0.36 | trec 19 vs 14, agnews 19 vs 17, negation 6 vs 23, yahoo answered `No` — warm-start drift on a task with only 30 replay traces of labeled_records |
| oolong_temporal | 0.46 | 0.27 | 0.25 | trec overflow; others partial |
| diagnostics (12 × 2) | 0.95 | 0.91 | 0.958 | synth_topk 1.00; narrativeqa 1 abstain |

**Reading.** Both regressions targeted were recovered exactly (cwe 0.52 → 1.00 via the contract the root now
writes; fwe back to 0.93), vt jumped to 0.88 as a side effect (fewer, cleaner priors?), and retrieval is at
ceiling. Counting gave back half of run 7's gain — the same warm-start drift as 7w/7w2, this time hitting the
task with the thinnest replay (labeled_records 30). The `User: User X` slip needs the template task to
include id-like answers ("User: [X]" where the doc writes `User 30140`), not just label names.
**Next from-base run** should carry: synth_topk, answer-form templates incl. ids, labeled_records at full
weight — and use batch 16 / peak 2e-5 with linear decay. Recipe note: with the curriculum now stable, that
run is the first candidate for a "freeze".

**Long-item regime (2026-09-20, after the 10K fresh-seed eval).** The per-item audit of A8 showed imdb
counting failing at 92% per-item label accuracy: reviews are 300–700 tokens, a 250-token leaf owns ≤1 and
SEES a fragment of a neighbour's, and 8 leaves labelled fragments they did not own (24 items → 32 votes).
Training had 336 single-extension leaves and exactly ONE leaf with 2+ consecutive "still cut off → read
the next 200" extensions (long_records bodies ≤250 tokens), so the repeated-extension loop and the
fragment-only leaf were essentially undemonstrated (they were routine in the OOLONG-only SFT, whose items
were ~450 tokens). Now: `long_records` — 40% of problems carry bodies of 400–700 tokens on 30% of entries;
`labeled_records` — 20% of problems use full-length yelp reviews (≥220 words, 677 rows; `text_long` added
by scripts/cache_labeled.py). Verified 160/160 at 1.00 (doc 6k/14k), max ctx 2851; in a 30-trace sample:
46 extension turns, 12 leaves with 2+ consecutive extensions, 15 fragment-only leaves. Leaf size stays a
constant 500 (leaf ≈ range + one item; items up to ~1,000 tokens fit a 3K budget).

---

## sft_general9w (run 9, warm start from sft_general8w) — 2026-09-20/21 — held-out 0.646, diagnostics 1.000

**From the 10K fresh-seed audit (A7/A8) — per-item leaf label accuracy vs OOLONG gold:** trec 0.87 (8w) / 0.34
(run 7); imdb 0.92 but WRONG answer (8 leaves labelled fragments they did not own → 24 items, 32 votes); agnews
0.70; yahoo 0.80; negation 0.46 with a strong True bias. Temporal: month key lost its YEAR on a "first month
where…" question; "more common before D than after" is a SHARE comparison in OOLONG (checked `get_frac` in the
vendored generator) and the model compared counts; yahoo root chose a 1-D tally for a per-month question. vt at
10K: 3/5 roots went binary-collect (fold prior weakens with length). Full audit in the 09-18 session notes.

**Changes**
1. **Long-item regime** (logged above): long_records bodies to 400–700 tokens; labeled_records full-length yelp.
2. **Claim verification** (`scripts/cache_claims.py` → labeled dataset `claims`, 6,244 balanced rows): DBpedia
   "Subject is/was predicate" sentences; FALSE = the predicate of a different class ("Carmeuse is a village in…").
   Negation's shape (one-sentence definitional claim, True/False) on non-OOLONG data; leaf lines quote ~100 chars.
3. **Temporal qtypes in labeled_records date mode:** `first_month_cmp` ("in which month did A FIRST occur more often
   than B" — chronological argmin over month×label, answer `Mon YYYY`, none if never) and `before_after` ("was L
   more/less/the same frequency before D vs on/after D" — SHARE semantics, stated in the question; state = 2 periods ×
   {L, other}; root shows both percentages). Month keys already carry the year across 2022–2025.
4. **Shape reason** (`_shape_reason` hook, all binary tally oracles): the root now says WHY the state has its shape —
   *"The question compares labels WITHIN each month and the items carry dates, so each item's month is derived from
   its date, and the state is a per-(month, label) tally, not a per-label one."* Same device that fixed fold/binary.
5. **`DOC_MIX_OVERRIDE`** (sft.py): per-task doc mix; `vt_novel=6000:1,14000:1` so half its fold chains are ~35 hops.
6. Warm from 8w, 25% replay, batch 16, LR 5e-6: labeled_records 200, long_records 120, vt_novel 100, synth_topk 40,
   rule_label 40, replay elsewhere. Verified: all tally tasks 60/60 each at doc 6k/14k after the shape-reason
   change; labeled_records 80/80 incl. claims / first_month_cmp / before_after / long items (max ctx 2842).
   Traces: `trace_snippets/labeled_records_v3.txt`. Script: `scripts/run_sft9_warm.sh`.
7. **Trace-text audit (2026-09-20, six subagents over ~600 dumped traces, one per task family).** Prompted by the
   "Give it." stub: same-class bugs found and fixed before run 9, all verified 12/12 at doc 6k/14k
   (`trace_snippets/audit_fixes_verify.txt`):
   - Shape reason moved BEFORE the strategy commitment in every binary preamble (reason → decision, not the reverse);
     fold connector reworded ("This task is order-dependent: {reason}. I therefore process…").
   - Subtask contract: "Put the result in \boxed{}: {fmt}; `none` if no <unit> starts in the range" — the empty-leaf
     `none` was contradicting the stated `key:count` format; `\boxed{}` no longer attaches to the wrong noun.
   - Plurals via `_n_units` ("the 1 entry", not "the 1 entries"; 843 hits in long_records alone); long_records kv
     layout says "HEADER STARTS" (it has a 3-line header, not a "HEADER line").
   - long_records: every step line carries exactly one marker (`(not better)` was an empty slot with a doubled
     space); tallies rendered in the document's NATURAL order (the stated tie rule) instead of alphabetically;
     evidence quotes use “ ” so an embedded `"` cannot break the list, and the cap says `+N more`; count roots
     close with a sentence like every other qtype.
   - rule_label: `section_most` reason said "WITHIN each section" for an ACROSS-sections question; evidence snippets
     keep head AND tail so the `?`/quote the rule inspects is visible (88% were `…`-truncated before the mark);
     "equally common as, in \boxed{}" comma (also in labeled_records) → "Answer in \boxed{} with exactly one of…".
   - labeled_records: author filter "where author Diaz" → "author = Diaz (the `[by Diaz]` tag)"; `on/after` two-slash
     period key → `on_or_after`; single-section docs → sections by document position; date-mode context clause.
   - synth: synth_distinct had the wrong shape reason; synth_2d month_for_grp reason; first_exceed fixed field list
     (`first=-1` = not yet); diff negative formatting; niah_multi hidden-mode reason; synth_topk binary-only guard.
   - Corpora: NarrativeQA cache had HTML in 1048/2000 rows → stripped and re-cached (0 rows with tags;
     `_CACHE_VERSION` v3, ~$10 haiku regen); yelp `\""` escapes unescaped; claims predicates >28 words dropped
     instead of `…`-truncated (5,450 rows).
   - BookQA/niah_novel: the subtask embeds the bare question (`q_core` metadata) instead of the full prompt with its
     preface and `\boxed{}` directive; empty-leaf sentinel says what to return; evidence/snippet clipping at word
     boundaries (`_clip`); passed-up context joined with ` ‖ ` so ` ⟐ ` means only the ANSWER record.
8. **Uniform root preamble (2026-09-20, after reading the dry run).** Binary roots opened with a SHAPE sentence
   ("The question compares labels WITHIN each month, so the state is a per-(month, label) tally…") and fold roots
   with an ORDER sentence — so the sentence type after "too long to read in one context" WAS the strategy choice,
   made through vocabulary that only appeared on one side (619 binary vs 180 fold roots). Fold roots also never
   reasoned about the accumulator's shape. Both strategies now make the same three moves in the same slots:
   (1) order relation, no state committed — "…does not depend on the order of the items, so disjoint ranges can be
   computed independently and merged" / "This task is order-dependent: {reason}, so the document has to be
   processed sequentially, in document order" (not "left to right" — a page-layout idiom, not a property of a token
   sequence; the plan sentence says "from the start"); (2) why the state has its shape, then the state — binary `_shape_reason` as before,
   and NEW accumulator reasons for every fold task (peak: "…so the accumulator carries the running total and the
   highest total so far"; vt_novel: "…the accumulator is the current value of every variable seen so far (a later
   assignment overwrites an earlier one)"); (3) the plan sentence, unchanged. realdoc_count gained a state sentence
   (it was the only tally task without one). Complete root episodes, one per task/qtype, straight from the dry run:
   `trace_snippets/root_turns_v9.txt` (`scripts/extract_roots.py`).
   Not fixed (cosmetic, in source prose): PG-essay scrape artifacts ("close.The way"), read_chunk clips mid-header.
   narrativeqa verify shows 9/12 with the SCRIPTED leaf — pre-existing (these are the gold-rejection-sampled cases;
   SFT uses the haiku leaf).

**Run 9w results — 2026-09-21 — held-out 0.646 (8w: 0.758), diagnostics 1.000 (24/24).**
(ckpt `tinker://d7385f1e-f958-53e4-a776-c642d5d51844:train:0/weights/sft_general9w`; 854 traces / 36,508 datums /
62.2M tokens, 2,282 batches at batch 16, LR 5e-6, NLL 0.0053; SFT 21:43→02:41 (5 h), eval 8 min. Raw
`eval_results/raw/sft_general9w.{jsonl,txt}`; failure rollouts `trace_snippets/run9_failures.txt`.)

| held-out | 8w | **9w** | note |
|---|---|---|---|
| oolong_counting | 0.36 | **0.63** | trec 17 vs 14, agnews 18 vs 17, negation answered 0 (vs 23) |
| oolong_temporal | 0.25 | **0.37** | imdb 3 vs 4, agnews 4 vs 6, yahoo 3 vs 6; trec 0 vs 8; negation dates overflow |
| oolong_user | 0.60 | 0.40 | negation "most common user": root INVENTED the key space `User A..E`, leaves returned `none:38`; trec: root read half itself + spawned half → overflow; yahoo `user 30140` (form slip) |
| **vt** | 0.88 | **0.16** | 3/5 roots went BINARY with a fabricated order sentence ("does not depend on the order of the variables assigned") and a filter-shaped state ("the set of variables assigned the value X") — chains lost, 1/5 vars each. 2/5 folded but with an invented op ("a variable can only be assigned once") and stored UNRESOLVED refs (`PYVCS=VAR MJHCS`) → none / 1 var |
| niah_multikey_1 | 0.80 | 0.60 | 2 × `none`: root used the generic tally opener ("does not depend on the order of the hidden number's occurrences… compute the magic number stated for sable-collector") instead of the retrieval preamble; the leaf that READ the needle said "0 occurrences whose line STARTS in the range → none" |
| niah_single_1 / multiquery | 1.00 / 1.00 | 1.00 / 0.85 | |
| cwe / fwe | 1.00 / 0.93 | 1.00 / 0.80 | fwe: one root's spawn call was emitted as truncated raw `<tool_call>` text → overflow at the root |
| diagnostics (12 × 2) | 0.958 | **1.000** | vt_novel 1.00, narrativeqa 1.00 |

**Reading.** The gold-label leaf work paid off (counting +0.27, temporal +0.12). The loss is concentrated in vt
(−0.72 → −0.08 on SCORE by itself) and comes from the uniform preamble: the new ORDER and STATE slots are
generated by the model for an unseen question, and on RULER vt ("Find all variables that are assigned the value
X") it inferred a filter-shaped state from the question's surface — the same slot that helps on OOLONG
(per-(month,label) tallies) hurts when the right state is NOT what the question names (the accumulator must carry
EVERY variable's value, not the ones holding X). 8w had no state slot for fold, so its 4/5 vt roots copied
vt_novel's accumulator op verbatim and scored 0.88. The niah regression has the same shape: the generic binary
opener ("does not depend on the order of the <unit>") now fires on retrieval questions too and displaces the
BookQA-style search preamble. With n=5/task the other moves (user, fwe, multiquery) are within noise.

**Options.** (A) Controlled re-run: 8w-form preamble + everything else from run 9 (audit fixes, long items,
claims, temporal qtypes, natural-order wording) — isolates the preamble effect for ≈$65/5 h. (B) Keep the uniform
form but make the fold state reasons CONTRASTIVE the way the binary ones are ("…so the accumulator carries every
variable's current value — not just the variables currently holding X, since which ones end with X is only known
at the end"), and give the retrieval tasks (niah_novel/narrativeqa) the same three-move opener so "search" is a
named alternative to "tally" in slot 2. (A) first is the cleaner experiment.

## sft_general10w (run 10, warm start from sft_general8w) — 2026-09-21 — held-out 0.711, diagnostics 1.000
Run 9's recipe and corpus, warm from **8w again** (not 9w — its vt/niah roots drifted), with the two fixes the run-9
post-mortem asked for. Verified 8/8 per task at doc 6k/14k (`scratchpad verify_G`), narrativeqa 6/8 = the known
scripted-leaf cases.
1. **Contrastive fold state reasons.** Every fold task's state sentence now names the tempting wrong state and rejects
   it, the way the binary ones do ("…not a per-label one"): vt_novel — *"The question asks which VARs finish with the
   value 91816, but that is only known at the end and a value reaches a VAR through copies, so the accumulator
   carries the current value of EVERY variable seen so far — not just the VARs currently holding 91816"*; varchain —
   "…not just the one asked about"; peak — "the final total alone would lose the peak"; streak — "a plain count of
   flag=Y records would not do"; adjacent — "the count alone cannot judge the next record"; first_exceed /
   first_reach — "the total/count alone forgets whether and where…"; runreset — "not a per-grp tally and not a count
   of resets". Rationale: the state slot is filled by the model on an unseen question; run 9 filled it with the state
   the QUESTION names (a filter). We want the training state regardless of the question's wording, so the training
   sentence must show the question-shaped state being rejected.
2. **Retrieval roots make the same three moves** (BookQAOracle → niah_novel, narrativeqa): *"The answer is stated in
   one place, and where it sits does not depend on the order of the sentences, so disjoint ranges can be searched
   independently and the one that finds it wins. The question asks for one fact, so the state is that fact once
   found (or none) — not a tally: nothing is counted, and a range with no relevant sentence contributes nothing. I
   therefore split…"*; the leaf contract adds "search it for a sentence that states the answer (there is nothing to
   count)". So "one fact once found" / "set of facts" (niah_multi) / "pruned tally" (topk) / "per-key tally" are
   four named alternatives in the same slot, instead of the 55 retrieval roots having a different-looking opener
   that loses to the 619 tally roots on frequency.
3. **Every binary root has a state sentence.** The question+root audit (`trace_snippets/root_audit_v10.txt`,
   `scripts/audit_roots.py`: one sample per (task, question shape), 129 groups over 300 traces) showed synth_sum /
   synth_max / synth_min / synth_maxwhere had none. Added, contrastive: "…a single running sum — no per-grp breakdown
   is needed"; "…just the best value seen so far (`none` until one is seen) — not a running total"; maxwhere "…other
   records are skipped, not counted".
4. **The binary order sentence is FIXED text**, byte-identical in all 227/300 binary roots incl. retrieval: *"The
   result over a range does not depend on the order in which the document is read, so disjoint ranges can be
   computed independently and merged."* It used to carry a per-task unit noun ("…order of the records / items /
   entries / sentences / words / hidden fact sentences / occurrences of the word 'X'") and retrieval had its own
   sentence; run 9w filled that slot from the question's surface ("the order of the hidden number's occurrences",
   "the order of the variables assigned") and the invented noun cascaded into the subtask and the leaf template.
   Now the root makes ONE decision at sentence two (the fixed binary sentence, or "This task is order-dependent:
   {reason}, so…") and all task specifics live in the state sentence; retrieval's plan sentence keeps "search each
   half … the one that finds it wins".
5. **vt_novel: filter-phrased question forms + more fold mass.** Every training ask said "finish / end up / final /
   after all assignments"; RULER's says "Find all variables that are assigned the value X" (no finality cue) — the
   phrasing family the run-9 root answered with a set filter. Added two forms that phrase it as a filter while
   staying a paraphrase: "Which variables are assigned the value X, directly or through a chain of copies? List
   them all, comma-separated." / "Find every variable that ends up with the value X (a copy passes the value
   along). List them, comma-separated." (6 forms for which_vars, 4 for final_value; RULER's sentence and its
   "Question:" layout are NOT copied.) vt_novel 100 → 150 roots, fold share ≈ 21% → 27%. Also: vt questions get their
   own prefaces about assignments ("The assignments in this passage form chains; a later line can overwrite an earlier
   one.") instead of the needle ones ("it contains a few planted facts"), and every LIST question (vt which_vars,
   niah_multi multivalue/multiquery) draws list-shaped answer tails ("Give them in \boxed{}") instead of "Reply with
   just the value".
6. **labeled_records coverage for the OOLONG-user / temporal shapes we lacked** (verified 80/80, all qtypes hit):
   `author_count` — "Which author has the MOST items overall (regardless of label)?" = OOLONG "which user is
   represented most often"; per-author count with an OPEN key space. `author_least` — "among author X's items,
   which label is the LEAST common (among labels that appear)" = OOLONG-user trec. `section_mode_count` — "For how
   many months/sections is `L` the single most common label (STRICTLY more than any other one label)?" = the
   OOLONG-temporal question that scored 0/8 on trec in 9w; per-(outer, label) tally, strict-mode finalize at the
   root. **Open key-space contract** for author-keyed tallies (author_top, author_count): "where `<author>` is the
   author exactly as written in the item's `[by …]` tag" instead of enumerating the authors (9w on OOLONG-user:
   root invented `User A`..`User E`, leaves returned `none:38`). **Numeric author ids** on 40% of documents
   (`[by 30140]`) so the open-set contract and the `Author: [X]` form see id-like keys (the `User: User 30140`
   slip, 3 runs). labeled_records 200 → 240 roots (13 qtypes now).
7. **`author_cmp` + form wording** (the OOLONG-user yahoo seed, lost the same way in 7/8w/9w with the TALLY correct):
   "which user has more instances with label L: User 30140 or User 92806? … form 'User: [X]', where [X] is the user
   ID" → 8w boxed `User: User 30140` (copied the question's mention), 9w `user 30140` (dropped the form); OOLONG's
   grader takes the text after the last ':' and needs an exact match. New qtype: "Which author has more `L` items:
   author 30140 or author 92806?" (per-author tally of L, two-way compare at the root; ties → alphabetical, stated);
   the Author form now says "[X] is the author exactly as written in the `[by …]` tag (just the name or id)" and the
   root's finalize says "The answer is the tag content itself, 30140 — not the word 'author'". Verified 80/80 with
   all 14 qtypes hit. Full traces for every new qtype: `trace_snippets/new_qtypes_run10.txt`.
Script: `scripts/run_sft10_warm.sh` (SAVE_NAME sft_general10w, eval → /tmp/eval_general10w).

**Run 10w results — 2026-09-21 — held-out 0.711 (8w 0.758, 9w 0.646), diagnostics 1.000 (24/24).**
(ckpt `tinker://cbf4f333-a798-5a49-92ca-7f1d89a4b8a0:train:0/weights/sft_general10w`; 944 traces / 39,978 datums /
68.8M tokens, 2,499 batches at batch 16, LR 5e-6, NLL 0.0055; SFT 12:40→15:55 (3.25 h), eval 7 min. Raw
`eval_results/raw/sft_general10w.{jsonl,txt}`; failure rollouts `trace_snippets/run10_failures.txt`.)

| held-out | 8w | 9w | **10w** | note |
|---|---|---|---|---|
| oolong_counting | 0.36 | 0.63 | 0.50 | trec 20 vs 14, agnews 13 vs 17, negation 5 vs 23 — leaf labelling noise, no root failure |
| oolong_user | 0.60 | 0.40 | 0.55 | yahoo `User: User 30140` AGAIN: the root ENUMERATED the key space "`<user>` is exactly one of `User 30140`, `User 92806`" (the question's mentions, descriptor included) so the tally keys carried "User 30140" and the box copied them — author_cmp's open-set contract did not transfer; trec: root emitted its spawn call as raw `<tool_call>` text → 1-agent overflow (same glitch as 9w fwe); agnews 0 vs 1 (leaf) |
| oolong_temporal | 0.25 | 0.37 | **0.21** | 3/5 roots chose a ONE-dimensional per-label tally for a per-month question ("the state is a per-label tally over an independent range"), then reasoned "no monthly information → 0" / "World=8 vs Sports=10 → No"; 9w had 2 of these 3 as 2-D. section_mode_count did not transfer |
| **vt** | 0.88 | 0.16 | **0.36** | 4/5 roots took the FIXED binary sentence then "the state is the set of variable=value facts … pruned to the queried value"; the 1 fold root scored 1.00 with a model-written reason and a correctly resolving accumulator. Fold EXECUTION is fine when chosen; the sentence-two CHOICE goes binary 4:1 |
| niah_multikey_1 | 0.80 | 0.60 | 0.80 | all 5 roots took the retrieval state sentence and the leaf found the needle; the miss is a leaf digit truncation (278069 vs 2780693) |
| niah_single_1 / multiquery | 1.00 / 1.00 | 1.00 / 0.85 | 1.00 / **1.00** | |
| cwe / fwe | 1.00 / 0.93 | 1.00 / 0.80 | 0.98 / **1.00** | |
| diagnostics (12 × 2) | 0.958 | 1.000 | 1.000 | |

**Reading.** The retrieval fix worked (niah_multikey back to 0.80 with the right root path on 5/5; the miss is a leaf
copy error). The fold fix did NOT: with sentence two fixed and byte-identical across 71% of roots, binary became the
zero-effort continuation and 4/5 RULER-vt roots took it — the downside case named before launch. The uniform
three-move preamble has now been tried twice (9w: unit-noun slot → invented filter noun; 10w: fixed sentence →
frequency prior) and lost vt both times; 8w's asymmetric form (binary opens with SHAPE, fold opens with ORDER and has
no state slot) is the only one that held fold on vt (0.88). Temporal regressed for a related reason: the state slot
is where the root decides 1-D vs 2-D, and with 14 labeled qtypes sharing the fixed opener the per-month shape lost to
per-label on 3/5 roots. OOLONG-user yahoo failed a 4th time on the form and now also on an ENUMERATED key space
built from the question's mentions — the open-set author contract in training did not carry to "User 30140".
**Best checkpoint remains 8w (0.758).** Best-of-per-task oracle across all runs is now 0.874 (counting 0.71 / user
0.75 / temporal 0.46 / vt 0.88 / fwe 1.00 / rest 1.00).

## sft_general11w (run 11, warm start from sft_general8w) — 2026-09-21 — held-out 0.709, diagnostics 0.958
Run 10's corpus (14 labeled qtypes incl. author_cmp, vt_novel 150 with filter forms, list tails, open-set author
contract) with ONE change: the root preamble goes back to 8w's ORDER OF COMMITMENT. An autoregressive root commits in
token order, so sentence two IS the fold-vs-binary decision:
- 8w: sentence two task-specific in both strategies (order relation fused with the plan and the GOAL noun), no
  separate state sentence → vt 0.88 (4/5 fold).
- 9w: state sentence BEFORE the plan, per-task unit noun in sentence two → root filled both from the question's
  surface on RULER vt (filter state) → 0.16.
- 10w: sentence two a FIXED generic sentence shared by 71% of roots → zero-effort continuation, 4/5 vt roots binary
  → 0.36 (the one fold root scored 1.00, so execution is fine; the CHOICE is what fails).
Run 11 form — the PURE 8w preamble, no state-reasoning sentence at all (decision 2026-09-21: keep the root simple to
learn; off-policy state-reasoning text has not shown a measurable gain — temporal 0.25 / 0.37 / 0.21 over 8w / 9w /
10w — and cost fold on vt whenever it preceded the commitment). Binary: "The result over a range does not depend on
the order of the {unit}, so I split the range in half recursively, having a subagent compute {goal} over each half
and combining the two partials. Once I have it…"; fold: "This task is order-dependent: {reason} — so I process the
document from the start with a running accumulator: … Once the final slice…"; retrieval: "The answer is stated in one
place, so I split … search each half … the one that finds it wins; once a half reports the answer, I read it off…".
The `_shape_reason` hooks stay in the oracles (unused) for a later experiment. Differences from 8w's actual text:
"from the start" instead of "left to right"; the reason strings and everything below the root (contracts, `none`
clause, plurals, natural-order tallies, audit fixes) are run 10's. Three-era side-by-side (8w regenerated from commit
fe62245, 10w, and the run-11 text):
`trace_snippets/preamble_comparison_8w_10w_11.txt`; audit `trace_snippets/root_audit_v11.txt` (0 of 240 roots carry a
state sentence). Verified 240/240 (30 tasks × 8) at doc 6k. Trace cache v5 (BookQA root text changed; ~$10 haiku regen).
Script: `scripts/run_sft11_warm.sh`.

**Run 11w results — 2026-09-21 — held-out 0.709 (8w 0.758, 9w 0.646, 10w 0.711), diagnostics 0.958 (23/24).**
(ckpt `tinker://a316bf8b-51dd-586b-b3a9-09c47b08c531:train:0/weights/sft_general11w`; 944 traces / 39,973 datums /
68.6M tokens, 2,499 batches at batch 16, LR 5e-6, NLL 0.0046; SFT 17:11→20:04 (2.9 h), eval 8 min. Raw
`eval_results/raw/sft_general11w.{jsonl,txt}`; rollouts of interest `trace_snippets/run11_failures.txt`.)

| held-out | 8w | 10w | **11w** | note |
|---|---|---|---|---|
| **vt** | 0.88 | 0.36 | **0.88** | **5/5 roots FOLD** (10w: 1/5) — the commitment-order hypothesis confirmed: with sentence two task-specific, fold wins. 4 perfect; seed 2016004 folded but with a malformed hand-off (2 agents) → 2/5 vars |
| oolong_user | 0.60 | 0.55 | **0.64** | **yahoo finally correct**: root wrote the OPEN-set contract "`<user>` is the user ID exactly as written in the data" and boxed `User: 30140` (author_cmp / open-set author contract transferred, on the 5th try); negation "most common user": root enumerated an invented `U0..U4` key space again (1-D per-user count — the open-set contract transferred for the 2-D tally, not the 1-D one); agnews: filter dropped |
| oolong_counting | 0.36 | 0.50 | 0.57 | negation: leaves labelled all 73 sentences True → 0; trec 22 vs 14; agnews 16 vs 17 |
| oolong_temporal | 0.25 | 0.21 | 0.20 | 3 overflows: trec and negation roots chose the right 2-D per-(month,label) state but the merged tally (many months × labels, `2023-10/numeric value=3|…`) blew the root budget; yahoo root emitted its spawn as raw `<tool_call>` text (1 agent) — that glitch has hit exactly one root per run since 9w |
| niah_single_1 | 1.00 | 1.00 | 1.00 | root subtask dropped the key ("Find the magic number in the document range") but a single needle is unambiguous |
| **niah_multikey_1** | 0.80 | 0.80 | **0.40** | **root subtask dropped the KEY on 4/5 roots** ("Find the magic number in the document range in tokens 0..2000") → leaves returned whichever key's number they saw (9042295 for overrated-merit, 8819265 for capable-cereal) and the leaf holding the true needle said "No relevant information". 10w carried the key on 5/5 (quoted question in the subtask); 8w on 4/5 |
| niah_multiquery | 1.00 | 1.00 | 0.70 | same drop on 1/5 ("Find the answer to the user's question"), two partial collections |
| cwe / fwe | 1.00 / 0.93 | 0.98 / 1.00 | 1.00 / 1.00 | |
| diagnostics | 0.958 | 1.000 | 0.958 | one labeled_records before_after miss |

**Reading.** Two clean results. (1) The fold decision is governed by sentence two: 8w-form sentence two → 5/5 fold and vt 0.88; 10w's generic
sentence → 1/5. That closes the vt question. (2) The retrieval regression has one mechanism, visible in the root subtasks: the run-11
retrieval opener is GENERIC ("The answer is stated in one place, so I split … search each half for the sentence that states the answer")
and the model instantiated "the answer" as "the magic number" and wrote a subtask with no key — the same slot-filling failure as 9w's
unit noun and 10w's fixed sentence, now on the retrieval side. 10w's retrieval opener named the question ("The question asks for one
fact…") and its subtasks carried the quoted question 5/5. So the design rule is confirmed in both directions: sentence two must carry the
task's specifics (order relation AND goal, with the key/question inside it); any generic slot gets filled from the question's surface.
The yahoo User-form fix landed via the open-set contract; the negation user seed did not — author_count in training already uses
the open-set wording for its 1-D per-author count, so that is a transfer failure on one seed, not a missing template. 8w remains the
best single checkpoint at 0.758; 11w has the best vt+user profile and the worst niah profile.

**Run 12 candidate (one line):** the retrieval root names the question in sentence two — "…having a subagent search each half for
the sentence that states the answer to the question \"{q_core}\"…" — so the key rides in the commitment sentence exactly as the tally
goal does. Expected: vt 0.88 held, niah_multikey back to ≥0.8, multiquery ~1.0 → SCORE ≈ 0.77–0.80.

## sft_general12w (run 12, warm start from sft_general8w) — 2026-09-22 — held-out **0.760** (best), diagnostics 1.000
Run 11 corpus and pure-8w preamble, plus one line: the retrieval root (niah_novel, narrativeqa) names the question in
sentence two — *"The answer to the question \"What is the special magic number for pewter?\" is stated in one place, so I
split the range in half recursively, having a subagent search each half for the sentence that answers it…"* — so the
key rides in the commitment sentence the way a tally root's goal noun does. Verified niah_novel 8/8 at doc 6k/14k
(narrativeqa 6/8 = the known scripted-leaf cases). Trace cache v6. Script: `scripts/run_sft12_warm.sh`.
Plus two labeled_records contract changes from the temporal audit (verified 80/80 at doc 6k and 14k):
- **Open-set month key.** Date-mode contract no longer enumerates the document's months ("one of `Mar 2022`, `Apr 2022`, …")
  — the root cannot know them before reading, and at eval it improvised ("the month name as written", "a number 1..12")
  and collapsed 38 month-years into 12 buckets. Now: "`<month>` is the month AND year of the item's date, written `Mon YYYY`
  exactly as in the date (e.g. `Jul 2022`)". Leaves unchanged ("[Apr 24, 2022 → Apr 2022]").
- **Whole-document dates_rep_k.** OOLONG asks "how many dates are represented exactly k times" over the WHOLE document;
  training only had the one-month scope, and the 11w root improvised a garbled contract (one leaf returned the scalar
  `dates=10`). Added the unscoped variant ("In the whole document: …") when the document has ≤ 40 distinct dates (50% of
  dates_rep_k draws at the 6k tier), contract "`<date>` is the item's date exactly as written in its tag (`Mon DD, YYYY`)".

## oolong_temporal across all runs (2026-09-21 audit)
| seed | dataset / question | run6 | run7 | 7w | 7w2 | 8w | 9w | 10w | 11w |
|---|---|---|---|---|---|---|---|---|---|
| trec — months where `location` is the single most common label (gold 8) | 6 labels × **38 months** | 0.32 (1-D, 4) | 0.10 (2-D, 0) | 0.13 (1-D) | 0.02 (1-D) | **OVF** (2-D) | 0.10 (1-D) | 0.10 (1-D) | **OVF** (2-D) |
| imdb — months where `positive` is most common (gold 4) | 2 labels × 7 months | 0.75 | 0.42 | 0.32 | 0.32 | 0.32 (1-D) | 0.75 (2-D, 3) | 0.75 (2-D, 3) | 0.42 (1-D) |
| agnews — months where World > Sports (gold 6) | 4 labels × 22 months | **1.00** (2-D) | 0.42 | 0.18 | 0.24 | 0.32 (2-D, 2) | 0.56 (4) | 0.00 (1-D, "No") | 0.56 (2-D, 4) |
| negation — dates represented exactly once (gold 75) | **72 distinct dates** | 0.00 (27) | OVF | 0.00 (27) | OVF | 0.04 (64) | OVF | OVF | OVF |
| yahoo — months where Politics > Computers (gold 6) | 10 labels × 28 months | 0.24 | 0.42 (2-D, 3) | 0.24 | 0.18 | 0.56 (2-D, 4) | 0.42 (2-D, 3) | 0.18 (1-D) | OVF (raw tool_call glitch) |
| **mean** | | **0.46** | 0.27 | 0.17 | 0.15 | 0.25 | 0.37 | 0.21 | 0.20 |

**Two failure classes, not one.** (a) *State choice*: 1-D per-label tally for a per-month question — the run-7 to 10w seesaw; the
pure-8w preamble in 11w picked 2-D on trec/agnews/yahoo. (b) *Budget*: when 2-D IS chosen, trec (38 months × 6 labels) and negation
(72 dates) overflow the ROOT every time — measured on 11w: children return 399+433 tokens (trec) / 493+352 (negation) and the root's
merge line RESTATES the merged tally (647 / 887 tokens) before the per-month rows, on top of ~900 tokens of fixed overhead → >3000.
The OOLONG documents are NOT date-sorted (adjacent-pair sortedness ≈ 0.5), so a date-ordered fold is not available; the 2-D tally is
the right state and it has to fit. Nothing consumes the root's restated merged tally — the finalize rows are derived from the two
children's boxes — so the fix is to drop the restatement AT THE ROOT ONLY (internal nodes still box the merged state for their parent).
A root-only "merge per key and read the answer off" line was tried and REVERTED the same day: every other root states its
operation and shows the result ("Sum my children [4, 6] -> 10", "Merge … -> merged tally"), and a root that says merge without
showing the merge breaks that pattern. The budget problem on trec/negation is therefore open; see the options below. imdb/agnews/yahoo are leaf-accuracy undercounts (3 vs 4, 4 vs 6, 4 vs 6), not structural.

**11w at a 5,000-token budget, same five temporal problems (2026-09-21):** mean 0.32 vs 0.20 at 3K. Raw
`eval_results/raw/sft_general11w_temporal_b5k.jsonl`.
| seed | gold | 3K | 5K | what changed |
|---|---|---|---|---|
| trec | 8 | overflow | 2 (0.18) | fits now, but the root keyed months by NAME (12 keys) instead of month-year (38) — lossy, so the count is wrong |
| imdb | 4 | 1 (1-D) | 0 (1-D) | state-choice flip, budget-independent |
| agnews | 6 | 4 | 3 | month-name keys again |
| negation | 75 | overflow | **68 (0.13)** | the per-date merge fits at 5K; 68 of 75 — budget was the whole problem here |
| yahoo | 6 | overflow (raw tool-call glitch) | 4 (0.56) | same as 8w's best on this seed |
Reading: negation was purely budget; trec is budget AND the open-set month key (with month-year keys a faithful
trec state is ≈2,250 tokens at the root — fits 5K, not 3K); imdb is the 1-D/2-D flip.

**Run 12w results — 2026-09-22 — held-out 0.760 (8w 0.758, 11w 0.709), diagnostics 1.000 (24/24).**
(ckpt `tinker://a8b5c74c-a05e-52cc-9cbc-797739ee23c5:train:0/weights/sft_general12w`; 944 traces / 39,954 datums /
68.6M tokens, 2,498 batches at batch 16, LR 5e-6, NLL 0.0044; SFT 23:58→02:54 (2.9 h), eval 8 min. Raw
`eval_results/raw/sft_general12w.{jsonl,txt}`; rollouts of interest `trace_snippets/run12_failures.txt`.)

| held-out | 8w | 11w | **12w** | note |
|---|---|---|---|---|
| **vt** | 0.88 | 0.88 | **0.96** | 5/5 fold; 4 perfect, seed 2016004 4/5 vars (one name garbled `UMZGAG`) — best vt of any run |
| **oolong_user** | 0.60 | 0.64 | **0.75** | yahoo `User: 30140` correct again (open-set user contract, 2 runs running); negation "most common user" still `none` (root invented a closed key space); agnews 2 vs 1 |
| oolong_counting | 0.36 | 0.57 | 0.42 | trec 23 vs 14, negation 0 vs 23 (leaves labelled everything True), yahoo compare wrong — leaf noise, same three seeds as always |
| oolong_temporal | 0.25 | 0.20 | 0.21 | **the month-year contract did NOT transfer**: roots wrote "the example's month (a number)", "the month name exactly as written" — same as 11w; trec overflow; negation root wrote the garbled `<count>` contract again (whole-document dates_rep_k was ~6 traces in the corpus — too few to override the month-scoped template) |
| **niah_multikey_1** | 0.80 | 0.40 | **0.80** | **key carried in the root subtask 5/5** (11w: 1/5) — the quoted-question opener did its job; the one miss is seed 2014003's digit truncation (278069 vs 2780693), the same miss in 10w and 11w |
| niah_multiquery | 1.00 | 0.70 | 0.70 | the retrieval opener now fires on 4-key questions ("The answer to the question "…4 keys…" is stated in one place") and the root writes a hybrid: one told its child to chain ("When done, call spawn_subagent … Continue") → 2-agent overflow; one dropped the STARTING ownership clause → 150-node runaway → half credit. 8w/10w multiquery roots used the niah_multi opener |
| niah_single_1 / cwe / fwe | 1.00 / 1.00 / 0.93 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | |
| diagnostics | 0.958 | 0.958 | **1.000** | |

**Reading.** First run above 8w, by a hair, with a clean mechanism behind each move: fold holds (5/5) with the 8w sentence two;
the quoted-question retrieval opener carries the key (5/5 multikey); the open-set user contract holds (yahoo 2 runs running).
Two things did not transfer: (1) the open-set MONTH contract — roots still improvise month keys; (2) the whole-document
dates contract — too rare in the corpus. Both are labeled_records changes that appeared in ≤240 roots against 8w's weights
that learned the enumerated form; a from-base run on this corpus is the clean test of whether they take. Multiquery's 0.70
is the retrieval opener over-firing on multi-key questions — a "one fact stated in one place" sentence applied to a four-key
question; the fix would be for the multi-key training questions (niah_multi multiquery mode) to be the ones whose opener the
model reaches for, i.e. give niah_multi's opener the same quoted-question form so the two families are distinguished by the
question text rather than by which opener the model happens to pick. **Best-of-per-task oracle across runs: 0.876** (recomputed from the raw jsonl; see `paper/results_across_runs.md`).

## sft_general13 (run 13, FROM BASE) — 2026-09-22 — held-out 0.718, diagnostics 0.958 — the single-stage reference
Corrected run 7's recipe on the run-12 oracle text, at run-7 scale with today's tasks: base count 40 for the 19 plain
synth tasks; labeled_records 400 (14 qtypes), vt_novel 300, synth_topk 120, long_records / rule_label / realdoc_count 160,
niah_multi 120, niah_novel 100, narrativeqa 80, synth_2d 80, synth_filter_argmax 60 → ≈2,500 traces, ≈170M tokens.
**Batch 16, constant LR 1e-5 (sft.py default), 1 epoch, LoRA rank 32, Adam β 0.9/0.95** — decided against batch 8 (NLL has
not tracked eval: 7w2 batch 8 NLL 0.0027 → 0.666; 8w batch 16 NLL 0.0136 → 0.758; the one from-base batch-16 data point,
run 7's first attempt at 0.573, was confounded by INTERNAL_KEEP 0.3) and against an LR schedule (keep the run-7 optimizer
so the difference from 0.662 is the corpus; the 5e-6 warm stage is the documented "settle" if wanted). Comparisons:
vs run 7 (0.662) = the corpus work, batch confound accepted; vs 12w (0.760) = single stage vs the three-stage chain
(run 7 → 8w → 12w, ≈3,700 traces of exposure in total). Script `scripts/run_sft13_base.sh`; log `/tmp/sft13_eval.log`;
eval → `/tmp/eval_general13`. Expected ≈13 h + 8 min eval, ≈$200 + ≈$30 haiku for narrativeqa at 80.

**Run 13 results — 2026-09-22 — held-out 0.718, diagnostics 0.958 (23/24).**
(ckpt `tinker://bb67e67e-7d0f-5bf1-adaf-7022ee1f0d8f:train:0/weights/sft_general13`; 2,500 traces / 106,639 datums /
181M tokens, 6,665 batches at batch 16, LR 1e-5 constant, NLL 0.0121; SFT 09:54→17:56 (8 h), eval 11 min; ≈$200.
Raw `eval_results/raw/sft_general13.{jsonl,txt}`.)

| held-out | run 7 (base, old corpus) | 8w | 12w | **run 13 (base, new corpus)** |
|---|---|---|---|---|
| oolong_counting | 0.60 | 0.36 | 0.42 | 0.35 |
| **oolong_user** | 0.40 | 0.60 | 0.75 | **0.95** |
| **oolong_temporal** | 0.27 | 0.25 | 0.21 | **0.28** |
| niah_single_1 | 1.00 | 1.00 | 1.00 | 0.80 |
| niah_multikey_1 | 1.00 | 0.80 | 0.80 | 0.80 |
| niah_multiquery | 0.95 | 1.00 | 0.70 | **1.00** |
| **vt** | 0.48 | 0.88 | 0.96 | **0.28** |
| cwe / fwe | 0.52 / 0.73 | 1.00 / 0.93 | 1.00 / 1.00 | **1.00 / 1.00** |
| **SCORE** | 0.662 | 0.758 | **0.760** | 0.718 |

**Corpus ablation (run 13 vs run 7, both from base): 0.662 → 0.718.** The six weeks of corpus work is worth ~+0.06
from base, concentrated exactly where it was aimed: user +0.55 (open-set contracts — BOTH user seeds correct, the
first run ever to get negation's "most common user", with the root writing "`<user>` is exactly the user's ID as
written in the sentences (never add a label)"), cwe +0.48 and fwe +0.27 (synth_topk), multiquery +0.05. Counting is
the one loss (0.60 → 0.35) and it is leaf-label noise on the same three seeds.

**Single-stage vs the warm chain (run 13 vs 12w): 0.718 vs 0.760.** The gap is ENTIRELY vt: 5/5 roots went BINARY
("collect every variable assigned the value 94316") → 0.28, against 12w's 5/5 fold → 0.96. Every other held-out task
is equal or better in run 13. So the fold prior is a two-stage artifact: 8w's weights carry it and the warm stages
preserve it, but one pass over this corpus from base does not install it — vt_novel is 300/2500 roots (12%) against
a 2,200-root binary majority, and from base the majority wins. That is the cleanest statement of what the staged
recipe buys, and it is a training-mix fact, not a prompt fact.

**Also new in run 13:** temporal's best score yet (0.28) with per-month roots on 3/5 seeds and negation at 67 vs 75
(the whole-document dates contract DID transfer from base — it was ≈15 traces here vs ≈6 in the warm runs); two
150-node runaways on niah (single_1 s3 and multikey s3) where the leaf hallucinated a needle sentence ("The magic
number for abundant-backpack is 42. This value is used in the calculation…") — a from-base leaf-grounding weakness
the warm runs do not show.

**Reading for the paper.** 12w (0.760) remains the headline checkpoint; run 13 is the single-stage reference and the
corpus ablation. The honest framing: the corpus work is worth +0.06 from base, and the staged recipe is worth a
further +0.04, all of it the fold/binary prior on vt.

## Why run 13 (from base) went binary on RULER vt — diagnosis 2026-09-22
**It is not fold mass.** Corpus fold share: run 12 (warm) **24.4%** (230/944), run 13 (base) **22.7%** (568/2500) —
essentially the same. **It is not capability.** Run 13 folds 2/2 on vt_novel and 2/2 on synth_peak in-distribution,
all at 1.00; it goes BINARY 5/5 on held-out RULER vt. It is a RECOGNITION failure, and the cue is in the prompt:

| | system-prompt document description |
|---|---|
| vt_novel (training) | "The document is a long passage of text with short 'VAR ... = ...' assignment lines hidden inside it, **in order**." |
| RULER vt (eval) | **(none — `task_context=""`, RULER fidelity)** |

Every RULER eval task ships an empty description (vt, cwe, fwe, niah_single/multikey/multiquery); every training task
ships one. Trained from base, the root learns *description → strategy*; with no description it falls back on the
question's surface, and "Find all variables that are assigned the value X" reads as a filter → binary. The warm chain
survives this because 8w's weights already carry the prior (three stages of fold exposure), which is exactly the
+0.04 that run 13 lacks. Same mechanism plausibly behind run 13's two 150-node niah runaways (also description-free).

**Fix (format diversity in TRAINING, per the standing principle — never an eval-time guard):** `CONTEXT_DROP` in
`sft.py` blanks `task_context` for a fraction of the traces of tasks whose QUESTION is self-sufficient —
`vt_novel, niah_novel, niah_multi, narrativeqa, realdoc_count`. synth/labeled/long/rule_label keep theirs (their
record layout and label set exist nowhere else; dropping those would teach hallucinated knowledge). Verified on a
dry run: 50% of prose traces have no description, the oracle text is unchanged (the scripted root still writes the
fold preamble), synth_peak keeps its description. **Cache-key bug found and fixed:** the system prompt is baked into
a cached trace but `_trace_key` ignored it, so a dropped trace would silently reuse a described one — `nodesc` is now
part of the key (described keys unchanged, so the paid narrativeqa haiku leaves stay valid).

## sft_general14w (run 14, warm start from RUN 13) — 2026-09-23 — held-out **0.832** (BEST), diagnostics 1.000
Warm from `sft_general13` (not 8w): run 13 has the better corpus-driven profile (user 0.95, temporal 0.28,
multiquery 1.00, cwe/fwe 1.00) and one deficit, vt. **ONE lever: prompt diversity** (the four changes above).
**The fold-mass boost was proposed, built, and then REMOVED** — the corpora measure 24.4% fold (run 12, warm) vs
22.7% (run 13, base), essentially identical, and run 13 folds 2/2 in-distribution, so mass is not the mechanism and
boosting it would only confound the run. Fold weights stay at run-12 warm levels (vt_novel 150, synth fold 10 each,
fold share 23%). niah_novel/niah_multi go 25/30 → 40/40 solely to size the CONTEXT_DROP treatment (~40
description-free traces rather than ~27), not as an independent lever. 969 traces, batch 16, LR 5e-6, 1 epoch
≈ 3 h / ≈$65. Treatment sizes: ~52 bare vt_novel, ~96 schema-lite labeled_records, ~40 description-free niah.
Script `scripts/run_sft14_warm.sh`. If vt recovers to ~0.9 on run 13's profile the SCORE lands ≈0.78–0.80; if the
two niah runaways also settle, ≈0.82.

## Train/serve prompt mismatch — full audit 2026-09-22 (extends the vt diagnosis)
The vt finding generalises: **our training always describes the document's schema; the evals often do not.**
The root therefore learns *read the description → know the schema*, and when the description is absent or partial it
FABRICATES one. Verbatim comparison:

| | RULER (vt, cwe, fwe, niah_*) | OOLONG (counting/user/temporal) | our training |
|---|---|---|---|
| system description | **none** (`task_context=""`, RULER fidelity) | item type + label set + count + "Do not guess… Calculate the exact answer" | always present |
| document schema in the description | — | **NEVER** — the `Date: … \|\| User: … \|\| Instance: …` columns are undescribed | **always** — `[S<n>]`, `[by <author>]`, `[Mon DD, YYYY]`, plus the month-derivation rule |
| format gloss in the QUESTION | none ("Memorize and track the chain(s) of variable assignment hidden in the following text.") | none | vt: `_VT_PREAMBLE` was on **100%** of questions — "(a line 'VAR A = VAR B' copies B's **current** value into A)" |

That single mechanism covers every open root-level failure: vt going binary (the order cue lived only in the
description + gloss), oolong_user inventing `U0`..`U4` / `User A`..`User E` (the author key form lived only in the
description), oolong_temporal writing "the month name as written" / "a number 1..12" (the month-key rule lived only
in the description), and plausibly run 13's description-free niah runaways.

**Three changes, all format diversity in TRAINING (`_CACHE_VERSION` → v7, questions/contexts changed for existing seeds):**
1. `CONTEXT_DROP` (sft.py) — blanks `task_context` for a fraction of the self-sufficient prose tasks
   (`vt_novel, niah_novel, niah_multi, narrativeqa, realdoc_count`). Cache-key bug fixed: `nodesc` is now in the key.
2. `_VT_PREAMBLE_BARE` (60/40) — vt_novel questions drop the copy-semantics gloss 40% of the time, so the root must
   derive order-dependence from the VAR lines it reads. Verified 30/30.
3. `_SCHEMA_LITE_FRAC = 0.4` (labeled_records) — 40% of descriptions state only item type + label set, matching
   OOLONG's information content; the `[S<n>]` / `[by <author>]` / `[Mon DD, YYYY]` tags and the month rule are left
   to be discovered. Verified 30/30 (15 full-schema / 15 schema-lite).
**Weighting (revised after review — the independent-coins version left the condition that matters at ~9%):**
vt_novel's three cues are now drawn TOGETHER — `_VT_BARE_FRAC = 0.35` makes 35% of its traces RULER-matched
(no description, no gloss, no preface) instead of 0.5 x 0.4 x 0.43 ~= 9%; the other 65% keep the description with
the independent gloss/preface mix. vt_novel was removed from sft.py's `_CONTEXT_DROP_TASKS` so exactly one
mechanism owns it. Measured on a 120-trace dry run: vt_novel 35% RULER-matched / 65% described; labeled_records
45% schema-lite / 55% full-schema; niah_novel 35% no-description. At run-14 counts that is ~70 RULER-matched
vt_novel traces (was ~17), ~96 schema-lite labeled_records, ~40 description-free niah — all above the ~40-trace
threshold at which synth_topk flipped cwe from 0.52 to 1.00.
4. `QUESTION_PLACEHOLDER_FRAC = 0.3` (sft.py) — our own harness boilerplate from
   `tasks/ruler/_common.py`, "[The relevant text is in a separate document accessible via the read_chunk
   tool — see the system prompt for usage.] (Document length: N tokens.)", measured on **25/25 RULER eval
   questions and 0/2500 training questions**. It is our document-presentation convention, not RULER content,
   so training a share of questions with it is skew reduction rather than eval fitting. Keyed in the trace
   cache (`qph`) so placeholder and plain traces are distinct artifacts.

**System prompt, for the record:** the harness FRAME ("You are a long-document assistant… two tools… emit it as
\boxed{value}") is emitted by `run_agent` and is byte-identical in training and at eval — not a mismatch. The
mismatch is entirely in the `task_context` slot, and OOLONG's differs from every description we train in three
structural ways: it is TWO paragraphs, it STATES THE ITEM COUNT ("86 general-knowledge questions… all 86 examples
in this dataset"), and it carries a task-level instruction we have never trained — "Do not try to guess, estimate,
or approximate the result. Calculate the exact answer given these datapoints." Our counting failures are precisely
what that instruction warns against, and a stated total is a coverage check the root has never learned to use
(children report 40+38=78 against a stated 86 → 8 items lost). NOT yet implemented; see the proposal below.

Remaining known leak (not changed): author questions still name the tag, "author = Okafor (the `[by Okafor]` tag)" —
that is task specification in the QUESTION, and OOLONG's question likewise names the user ID.

**Run 14w results — 2026-09-23 — held-out 0.832, diagnostics 1.000 (24/24). Best checkpoint by a wide margin.**
(ckpt `tinker://de730742-0918-5a2d-80fe-48f3d41ed0c3:train:0/weights/sft_general14w`; 969 traces / 40,805 datums /
69.5M tokens, 2,551 batches at batch 16, LR 5e-6, NLL 0.0025; SFT 23:37→02:45 (3.1 h), eval 9 min; ≈$65.
Raw `eval_results/raw/sft_general14w.{jsonl,txt}`.)

| held-out | 8w | 12w | run 13 (base) | **14w** |
|---|---|---|---|---|
| **vt** | 0.88 | 0.96 | 0.28 | **1.00** — 5/5 FOLD, all perfect (run 13: 5/5 binary) |
| **niah_multikey_1** | 0.80 | 0.80 | 0.80 | **1.00** — key carried 5/5 |
| niah_multiquery | 1.00 | 0.70 | 1.00 | **1.00** |
| niah_single_1 | 1.00 | 1.00 | 0.80 | **1.00** |
| **oolong_counting** | 0.36 | 0.42 | 0.35 | **0.63** |
| **oolong_temporal** | 0.25 | 0.21 | 0.28 | **0.35** (best ever) |
| oolong_user | 0.60 | 0.75 | **0.95** | 0.65 |
| cwe / fwe | 1.00 / 0.93 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.87 |
| **SCORE** | 0.758 | 0.760 | 0.718 | **0.832** |

**Reading.** Every RULER task is now at 1.00 except fwe (0.87, two third-word slips). vt went 0.28 → 1.00 with all
five roots folding, and niah_multikey carried the key 5/5 for the first time. The OOLONG *aggregation* tasks moved
too: counting 0.35 → 0.63 and temporal to its best-ever 0.35, with roots writing the open-set month contract
unprompted ("`<month>` is the month and year of the review's date, written `Mon YYYY`") — the schema-lite training
transferring exactly as intended. The one regression is oolong_user 0.95 → 0.65: trec picked the wrong label and
agnews answered 6 vs 1; both roots wrote NO key contract at all, so the schema-lite variant may have over-corrected
on that task. Temporal's three overflows are still the state-size ceiling (trec 38 months x 6 labels, negation 72
dates) — unchanged and expected at a 3K budget.

**Caveat, stated for the paper:** run 14 is warm from run 13, so the staged second pass and the prompt-diversity
corpus are confounded. 12w (staged, no diversity) scored 0.760 and run 13 (diversity-free, single stage) 0.718, so
+0.07 over 12w is larger than staging alone has ever bought (+0.04); that is suggestive but not a clean attribution.
The cheap disambiguator remains the description-ablation probe on in-distribution `vt_novel`.

## sft_general15w (run 15, warm start from 14w) — 2026-09-23 — held-out 0.786, RULER-13 0.882, diagnostics 0.903

(ckpt `tinker://fa3fd877-3c30-5b16-8f0d-5b1555ef6de3:train:0/weights/sft_general15w`; 1,009 traces / 42,044 datums,
2,628 batches at batch 16, LR 5e-6, NLL 0.0018; SFT 14:52→18:09, eval 9 min. Raw `eval_results/raw/sft_general15w.{jsonl,txt}`.
Script `scripts/run_sft15_warm.sh`, cache v9.) Run-14 recipe plus the full-RULER audit fixes: niah_multi filters to the
asked keys (other keys listed `(other key, skip)`), uuid keys/values in niah_novel, all-needle haystacks (25% of eligible
niah), labeled_records `author_label_count` + 2-author subsets + ~30% author-filtered; niah_novel/niah_multi 40 → 60.
Eval appends the 7 never-scored RULER tasks after the run-6..14 list (existing seeds unchanged; 14w's numbers for them
are from `eval_results/raw/sft_general14w_ruler7.*`, same seeds).

| task | 14w | **15w** |
|---|---|---|
| **niah_multikey_3** (all-needle, uuid→uuid) | 0.00 (4/5 overflow) | **1.00** — generalized with 0 exact-shape training traces |
| niah_multikey_2 (all-needle) | 0.80 | **1.00** |
| oolong_user | 0.65 | 0.75 (agnews subset count 0.24 → 0.75; trec still 0) |
| fwe | 0.87 | 1.00 |
| **niah_multikey_1** | 1.00 | **0.60** — see below |
| niah_single_2 / niah_multivalue | 1.00 / 1.00 | 0.80 / 0.70 — see below |
| oolong_counting / temporal | 0.63 / 0.35 | 0.46 / 0.30 (counting: one yahoo seed 1.0 → 0.0; temporal: known ceiling) |
| niah_single_1, niah_single_3, multiquery, cwe, qa_1 | 1.00 | 1.00 |
| vt / qa_2 | 1.00 / 0.40 | 0.96 / 0.40 |
| **SCORE-9 / RULER-13 / 16-task** | 0.832 / 0.851 / 0.793 | **0.786 / 0.882 / 0.811** |

**The regression is one artifact, not the new data.** 30 subagents given 1000/2000-token ranges said "Range
2000..4000 fits" and read the whole range. Every one is a ROUND range ending at the stated document length; RULER's
essay haystack is sized to exactly 4000 tokens (its noise/needle haystacks land at 3975-3999 and never trigger it).
A causal probe on the exact failing prompts (vary only the numbers; P(" fits") at the decision token) gives 15w
P = 1.000 for 2000..4000 / 3000..4000 at doc 4000 and 0.000 for every non-round or round-100 range of the same size;
not the budget, not the literal "4000" (doc 5000/8000 → 0.000), patchy across doc lengths (4000/8000 fire; 5000,
10000, 12000, 20000 don't). 12w and 14w are clean on every probe; run 13 has a weak version. Training has almost no
round ranges (only ~40 traces with 6000/14000 docs, count/sum subtasks) and the run-14 and run-15 corpora are identical
in this respect — so this is drift on an unsupported input across warm passes. It costs points only through the
never-taught overflow recovery: the parent writes "the answer is not in the range they covered → none" (15w: exactly
its three wrong niah answers; run 13: 25 times). multikey_1 alone is −0.044 of the −0.046 SCORE-9 drop.
Full tables: `trace_snippets/run15_split_decision_probe.txt`.

## sft_general16w (run 16, warm start from 14w) — 2026-09-23 — held-out 0.810, RULER-13 **0.937** (best), 16-task **0.853** (best)

(ckpt `tinker://97067667-0cfb-5879-9932-9ab90f9d06b2:train:0/weights/sft_general16w`; 1,009 traces / 42,943 datums /
71.8M tokens, root share of datums 9.4% (run 15: 9.6%); SFT 19:33→~22:55, eval 9 min. Raw `eval_results/raw/sft_general16w.*`.
Script `scripts/run_sft16_warm.sh`.) Same start (14w) and same v9 corpus as run 15; the ONE change is budget x length
jitter: per-problem budget {3K 50%, 5K 15%, 8K 13%, 10K 11%, 12K 11%}, doc length triangular (mode at the row's low end,
ceiling 14K, row mean > budget), 20% of targets rounded (only 8 docs came out exactly round). Split rule unchanged.

**Split-decision probe (the direct test): clean everywhere.** Every probed range — round and non-round, ending at the doc
or not, docs 2K-20K, budgets 1.5K-8K, search and collect subtasks — P(" fits") <= 0.011 (15w: 1.000 on 2000..4000).
0 oversized "fits" in the eval (15w: 30). 16w is also the first checkpoint to split an exact-500 round range as the rule
says ("less than 500"): 3500..4000 → 0.001 (all earlier checkpoints ~1.0). Round ranges were NOT directly trained (30
round splits vs run 15's 111), so the rule generalized from the jittered sizes — the user's hypothesis over mine.

| task | 14w | 15w | **16w** |
|---|---|---|---|
| niah_multikey_1 / single_2 / multivalue | 1.00 / 1.00 / 1.00 | 0.60 / 0.80 / 0.70 | **1.00 / 1.00 / 0.95** |
| niah_multikey_2 / multikey_3 | 0.80 / 0.00 | 1.00 / 1.00 | **1.00 / 1.00** |
| niah_single_1 / single_3 / multiquery | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.95 |
| vt | 1.00 | 0.96 | 0.88 — 4/5 fold; one root went binary (0.60), one name slip (0.80) |
| cwe / fwe / qa_1 / qa_2 | 1.00 / 0.87 / 1.00 / 0.40 | 1.00 / 1.00 / 1.00 / 0.40 | 1.00 / 1.00 / 1.00 / 0.40 |
| oolong_counting | 0.63 | 0.46 | 0.51 — agnews tally correct (17) but answered without \boxed{} |
| oolong_user | 0.65 | 0.75 | 0.68 — trec subset still 0 |
| oolong_temporal | 0.35 | 0.30 | 0.26 — same three overflows (the known ceiling) |
| **SCORE-9 / RULER-13 / 16-task** | **0.832** / 0.851 / 0.793 | 0.786 / 0.882 / 0.811 | 0.810 / **0.937** / **0.853** |

Reading: the jitter fixed the split rule and every NIAH regression while keeping run 15's needle/multikey gains.
SCORE-9 sits between 14w and 15w; the gap to 14w is vt (one binary root — the root-commitment prior again) plus
OOLONG seed-level moves (a missing \boxed on a correct tally; the known temporal overflows), each within 45-rollout noise.

## OOLONG-real — built 2026-09-24 (EVAL ONLY; never an SFT/RL task — sft.py refuses it)

Source: HF `oolongbench/oolong-real` (config `dnd`; 20 GB snapshot at `~/.cache/infinite-context/oolong_real`), code
+ scorer from `abertsch72/oolong` @ 0bb7eab (MIT). Critical Role transcripts, campaign 1 = `test` (the official split),
campaign 2 = `validation`. Each split: 110 context windows of 1-24 episodes, ~6.0-6.1K questions in 4 types
(singledoc_rolls/spells, multidoc_rolls/spells); answers are integers (0.75^|err|), strings (exact, case-insensitive)
or comma lists (|gold ∩ pred| / |gold|). Document length in Qwen tokens (after moving the instruction paragraph to
task_context): test 33.6K / 321K / 1.32M (min/median/max); 20 single-episode windows across both splits.

- Prep: `PYTHONPATH=. uv run python scripts/prepare_oolong_real.py` -> per-window token arrays + question index
  (`~/.cache/infinite-context/oolong_real/prepared/<split>/`). Aborts on any window without the mapping marker.
- Adapter: `tasks/oolong_real/` — task `oolong_real`; everything before the first `[START OF EPISODE]` (instruction
  paragraph AND player->character mapping) -> task_context (revised 2026-09-24: the official harness sends the whole
  window as the system message and the mapping is task-level context, as OOLONG-synth's description is; it was
  first put in the document), document = the transcripts verbatim, question verbatim,
  grading mode `oolong_real` = vendored `dnd_*` scorer (`tasks/oolong_real/vendored_eval.py`, verbatim).
  Env: `OOLONG_REAL_SPLIT` (test), `OOLONG_REAL_MIN_TOKENS` / `OOLONG_REAL_MAX_TOKENS`, `OOLONG_REAL_TYPES`.
  Per-type scores print as the eval's per-"dataset" breakdown.
- Fix needed on the way: `eval/run.py` and `sft.py` routed ANY task starting with "oolong" to the OOLONG-synth
  generator; now `task in OOLONG_TASKS`.

**Validation (base models only, as asked):**
| run | protocol | n | score | reference |
|---|---|---|---|---|
| GPT-5-mini | official (`scripts/oolong_real_official.py`: their system/user messages + vendored scorer, raw HF rows), test, 1-episode windows | 40 | **51.22** (rolls 45.4 / spells 57.7; 1 low-confidence parse) | paper Table 4, 55K: **49.86** |
| base Qwen3.6-35B-A3B | our harness, MODE=single (whole transcript in context), 1-episode | 10 | 0.570 | — (not in paper) |
| base Qwen3.6-35B-A3B | our harness, MODE=decompose, budget 5K, 1-episode | 10 | 0.000 — 10/10 roots overflow within 1-6 agents: reads the transcript sequentially into its own context, never decomposes | expected for an untrained model |

The official-protocol reproduction lands within sampling error of the paper (±~8 at n=40), so the raw data, the
question/gold pairing and the scorer are right; the two base-Qwen runs show our adapter end to end (loading,
task_context, tools over a 33-55K-token document, list/int/string grading). `claude-sonnet-4-20250514` (the paper's
Claude row) has been retired by the API, so GPT-5-mini is the reproduction. Raw: `eval_results/raw/oolong_real_*`,
`eval_results/raw/oolong_real_base_qwen_{single,decompose}_1ep.jsonl`. Fine-tuned checkpoints have NOT been evaluated on it (decision pending).

## 16w long-context eval — 2026-09-24 — 8K / 16K / 32K docs, 5K per-agent budget, FRESH seeds, all 10 OOLONG-synth datasets

`scripts/eval_long.sh <tag> <ckpt> <doc> <budget>`: all 13 RULER tasks x 5 seeds + the 3 OOLONG-synth families x 20
(= 2 per dataset over all 10 validated datasets; the 4K scoreboard only ever reached the first 5), seed block 3,000,000+
(never trained on or read), TEMP 0.2, MAX_NODES 2000, no depth cap. 5K is one of 16w's training budgets (15% of
traces); 8K/16K are inside the 14K training length ceiling (16K just past it), 32K is ~2.3x past it. Raw
`eval_results/raw/sft_general16w_long_{8k,16k,32k}.{jsonl,txt}`. Wall time 35 / 22 / 28 min.

| task | 8K | 16K | 32K |
|---|---|---|---|
| niah_single_1/2/3 | 1.00 / 0.80 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| niah_multikey_1/2/3 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| niah_multivalue / multiquery | 0.95 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| vt | 0.80 | 1.00 | 0.84 |
| cwe / fwe | 1.00 / 0.93 | 0.98 / 1.00 | 0.84 / 1.00 |
| qa_1 / qa_2 | 0.60 / 0.40 | 0.80 / 0.60 | 0.60 / 0.60 |
| **RULER-13** | **0.883** | **0.952** | **0.914** |
| oolong_counting | 0.48 | 0.48 | 0.42 |
| oolong_user | 0.55 | 0.25 | 0.36 |
| oolong_temporal | 0.45 | 0.51 | 0.30 |
| **OOLONG-synth-3** | **0.495** | **0.414** | **0.358** |
| root overflows (OOLONG, of 60) | 3 | 15 | 15 |
| agents / rollout (mean) | 55 | 89 | 188 |

OOLONG by dataset (mean of the 3 families): app_reviews 0.96 / 0.43 / 0.50, imdb 0.68 / 0.83 / 0.66, metaphors
0.63 / 0.34 / 0.33, agnews 0.61 / 0.35 / 0.50, trec_coarse 0.50 / 0.50 / 0.50, yahoo 0.33 / 0.50 / 0.51, negation
0.17 / 0.50 / 0.33, multinli 0.39 / 0.35 / 0.05, spam 0.34 / 0.33 / 0.16, formality 0.34 / 0.00 / 0.02
(n = 6 per cell — read as direction only).

**Reading.**
- RULER holds past the training length: every NIAH variant is at 1.00 at 16K and 32K (the 8K single_2 miss is the
  one runaway below). The misses are vt (one seed per length collapses — 32K: 1 of 5 variables), cwe counting slips at
  32K (0.6-0.9 partial credit: word frequencies off by one), and qa's ruler_part substring grader (e.g. "High risk
  preparations and other compounding functions" vs gold "…and SOME other…" scores 0).
- OOLONG degrades with length through the ROOT STATE CEILING, as predicted: overflows 3 → 15 → 15, now hitting
  oolong_user too (8/20 at 16K). Example: "which user has more `abbreviation` instances: User 73858 or User 19826?" —
  the root built a full per-(user, label) tally over every user instead of filtering to the two named users and one
  label; the same filter-vs-collect choice fixed for niah in run 15, and a training target (author_cmp exists but its
  state is the per-author count of one label, not a two-user filter).
- Harness caps work: one 8K niah_single_2 rollout wrote a subtask with NO range ("Find the magic number for
  obsequious-appellation (or any relevant key fact or number). Recursively split the range…"); every child restarted at
  0..8000 and the tree ran 460 levels deep until MAX_NODES=2000 stopped it. A real model failure (range dropped from a
  subtask), 1 of 125 rollouts; nothing like it at 16K/32K.

## Minimal-sufficient-state audit — 2026-09-24 (training corpus v10 vs what each question needs)

Trigger: 16w's 16K/32K OOLONG overflows. EVERY overflowing root wrote the same contract, a per-(X, label) tally over
ALL labels, whatever the question needed: months L1>L2 (8/14 overflow) and first-month L1>L2 carried months x all
labels (need months x {L1,L2}); user A vs B on L (7/16) and user-most-with-L (5/15) carried users x all labels (need 2
numbers / users x {L}); before/after share carried DATES x labels (needs 4 numbers). Only months-L-single-most and
dates-exactly-n need a large state inherently. The prior comes from the corpus: 105 of 1,009 traces (run-16 corpus)
root-dictate a joint tally, and 54 of those need less.

| task / qtype | carries | minimal sufficient | traces |
|---|---|---|---|
| labeled sections_cmp ("in how many sections/months is a > b") | outer x ALL labels | outer x {a, b} | 11 |
| labeled first_month_cmp | month x ALL labels | month x {a, b} | 8 |
| labeled section_most ("which section has most L") | outer x ALL labels | per-outer count of L | 10 |
| labeled relative ("is a more/less common than b") | all labels (1-D) | {a, b} | ~15 |
| rule_label section_most | section x both labels | per-section count of L | 10 |
| synth_2d months_cmp | outer x ALL inner | outer x {g1, g2} | 4 |
| synth_2d month_for_grp | outer x ALL inner | per-outer count of g | 8 |
| synth_2d grp_in_month | outer x ALL inner | 1-D tally within the one outer value (= filter_argmax) | 3 |
| labeled author_cmp | FIXED 2026-09-24 (v10): {a1, a2} x {L} | — | — |
| vt_novel which_vars (fold) | every variable's current value | the SET of variables currently holding X (copy-aware update) | ~100 |
| minimal already | synth scalar/mode/distinct/sumby/2-field folds, varchain (final value needs all vars forward), filter_argmax, months_argmax; long_records all 7 qtypes; labeled count/most_common/author_*/section_mode_count/dates_rep_k/before_after; rule_label count/most_common/relative/sections_cmp (2 labels); niah (hidden collects all by necessity); realdoc; topk (pruned by design) | | |

Gap (no training analogue): OOLONG-user "only consider the subset of users with IDs …; which user is represented most
often / has the most L" — our author_count/author_top are whole-document only (subsets exist only for author_most /
author_least / author_label_count).
Note on vt_novel: the run-10 contrastive sentence ("…not just the VARs currently holding X, since which ones end with X
is only known at the end") is wrong as a sufficiency claim; run 9's failure was BINARY + filter (insufficient because
copies are order-dependent), not fold + holder set, which was never tried.

**Implemented 2026-09-24 (cache v11)** — every over-carrying row above now keeps only what its question names, in the
existing filtered-leaf conventions (filter in the op-phrase parenthetical; every line shown with a verdict: `(other
label, skip)` / `(not `L`, skip)` / `(other author, skip)` / synth `(skip)` vs `(match)`); root answer-extraction updated
where the key shape changed (section_most -> `<outer>=<count>`, month_for_grp -> `<outer>=<count>`, grp_in_month -> 1-D).
Plus the gap: author_count / author_top draw a 2-3-author SUBSET half the time (`_AUTHOR_SUBSET_FRAC = 0.5`), state =
only the listed authors. Verified 1,200/1,200 gold (labeled_records, rule_label, synth_2d x 200 x doc 6K/14K; worst node
2,840, +37 tokens of longer subtask text on a 2-label yelp leaf) and a full run-16-recipe dry run: 1,009/1,009 gold,
42,918 datums, 71.6M tokens. Root-dictated joint tallies over ALL inner values: **105 -> 51**, exactly the qtypes that
need one (section_mode_count 14, src_tag_2d 19, rule_label sections_cmp 9, synth_2d months_argmax 9); 23 more are joint
tallies restricted to the two named values (sections_cmp / first_month_cmp / months_cmp); the rest became 1-D.
Rendered examples: `trace_snippets/audit_v11_minimal_state_leaves.txt`. vt_novel (holder set) deferred to its own change.

## 32K @ 8K budget + the vt fold-decision probe — 2026-09-24

**32K OOLONG-synth, 16w, 8K per-agent budget** (same fresh seeds / 10 datasets as the 5K-budget run; raw
`eval_results/raw/sft_general16w_long_32k_b8k.*`): counting 0.418 → 0.430, user 0.356 → **0.469**, temporal 0.299 → 0.291,
mean 0.358 → **0.397**; root overflows 15 → **7**. Budget buys back part of the loss (mostly oolong_user); temporal does
not move — its all-label states outgrow 8K too. The v11 minimal-state corpus targets the rest.

**vt: which checkpoints fold?** 5 of 20 RULER-vt roots across 16w's 4K-32K evals went binary, every one with niah_multi's
collect opener ("hidden fact sentences … collect every variable=value fact"). Probe: the exact 20 root prompts, P(" This"
= "This task is order-dependent…") vs P(" The" = "The result over a range does not depend…") after the shared first
sentence; mass on the two tokens >= 0.99 everywhere.
| checkpoint | 12w | 13 | 14w | 15w | 16w |
|---|---|---|---|---|---|
| mean P(fold), T=1 [T=0.2] | 1.00 [1.00] | 0.00 [0.00] | 1.00 [1.00] | 1.00 [1.00] | **0.68 [0.74]** (0.05-1.00 per prompt) |
15w (niah 60/60, v9 corpus) is 1.00 on all 20, so the "more niah_multi examples pulled vt toward collect" hypothesis is
REFUTED. 15w and 16w share start (14w) and corpus; the only difference is run 16's budget x length jitter. Varying only
the budget STATED in the eval prompt does not explain it (16w mean P(fold) 0.81 / 0.60 / 0.75 / 0.64 at 3K / 5K / 8K /
12K). Open: whether the jitter's vt_novel length distribution (the 6K/14K override was dropped for the triangular mix)
or plain run-to-run drift moved it. Probe script: scratchpad `fold_probe.py` (same method as the split-decision probe).

## qa_1/qa_2 on the unit-filtered pool + niah_bridge — 2026-09-24

**RULER QA question pool filtered** (`tasks/ruler/qa_data.py`): questions whose every gold is a number followed by
words ("35 people", "500-room", "3.5 million", "1 October 1998") are dropped — a correct bare number can never contain
the unit, so RULER's substring match scores it 0 (qa_2: `35` for "…killed how many people?"). SQuAD 5,928 -> 5,782,
HotpotQA 7,405 -> 7,143; paragraphs stay as distractors. Every qa seed now maps to a different question.
16w, filtered pool, 16K doc, 5K budget, 10 fresh seeds each (`eval_results/raw/sft_general16w_qa_filtered_16k.*`):
**qa_1 0.70, qa_2 0.30**. qa_2 misses: 3 alias / surface variants the substring grader cannot match (`Caligula` vs
"Gaius Julius Caesar Augustus Germanicus", `Bigg Boss 10` vs "the tenth season", `The Windigo legend` vs "Wendigo"),
3 stopped at the BRIDGE entity (answered hop 1), 1 `none`.

**Why the bridge fails.** 16w's leaves read both hops of "Ravi Khote has included his music in which 2003 Indian drama?"
(…"Pretty Woman" from "Kal Ho Naa Ho" / Kal Ho Naa Ho … is a 2003 Indian romantic drama) and returned "No relevant
information". The search contract says "otherwise return any relevant information", the oracle has a full context
channel (leaf context turn, internal merge, top-K) — but the corpus had **0 of 1,984** search leaves using it:
narrativeqa's model leaf essentially never answers CONTEXT, niah_novel has nothing but the needle, and a root that
receives only context boxes "answer not found" so such traces are rejected by design. Feasibility on RULER's own pool
(HotpotQA validation, supporting-fact annotations): both supporting sentences share a content keyword with the
question for 90.7% of bridge and 98.8% of comparison questions — one keyword-driven binary pass can surface both hops.

**niah_bridge** (`tasks/niah/generators.py`, oracle = BookQAOracle scripted leaf): two inserted fact sentences in prose
— "The head archivist of Port Evan is Freya Lindqvist." / "Freya Lindqvist was born in Tarsk." — plus distractors
mirroring both hops (same role elsewhere, other roles in the same place, the same relation for other people); the
question asks across the bridge ("Where was the head archivist of Port Evan born?"). Same single binary search; leaves
pass up the fact sentences they hold as relevant context; the ROOT keeps the UNCHANGED search opener (revised
2026-09-24: the root makes no decision from the question — it passes it down verbatim and combines what comes back; a
bridge-specific opener would add a "recognize multi-hop from the wording" decision, and on RULER qa_2 the model writes
the search opener anyway) and the only new text is the chaining final turn ("No single passage answers the question; …
Chaining them: «hop 1» — so X is Freya Lindqvist; «hop 2» — so the answer is Tarsk."). No benchmark data. Verified
100/100 gold at doc 6K/14K (max node 1.7K), niah_novel unchanged 100/100. Example: `trace_snippets/niah_bridge_example.txt`.

**Run 17 draft** (`scripts/run_sft17_warm.sh`, not launched): run-16 recipe + v11 minimal states + v12 vt contrast +
niah_bridge 80 + the six sequential synth tasks 10 -> 20. Dry run: 1,149/1,149 gold, 48,508 datums, 80.5M tokens
(run 16: 71.8M), fold roots 286 (24.9%, was 22.4%), 546 context leaves (was 0; the QA rebalancer counts them as
positive verdicts and duplicates them x2).

## BUG: the model leaf was offered the agent tools — narrativeqa "none" leaves were mostly not judgments (fixed 2026-09-24, cache v13)

`ModelBackend.complete()` (used by BookQAOracle's model-executed leaf and by rl.py's judge) called `sample()` with the
default `tools=True`, so every Haiku leaf call also carried our `read_chunk` / `spawn_subagent` schema with
`tool_choice="auto"`. Replaying 120 narrativeqa training leaves recorded as "No relevant information" WITH the tools:
Haiku CALLED `read_chunk` on **114 (95%)** ("I need to read more of this excerpt…") — no ANSWER/CONTEXT line, so the
parser recorded `none`. Replaying all 369 recorded-`none` leaves WITHOUT tools: ~167 none, ~107 context, ~95 answer.
The leaf prompt was never the problem (it already allows CONTEXT). This is why the corpus had 0 narrativeqa context
leaves, and a large part of why narrativeqa's rejection acceptance was 25%.
Fix: `complete()` -> `sample(..., tools=False)` (eval/backends.py). Scope: SFT narrativeqa/bookqa leaves (every run
that used BOOKQA_LEAF_MODEL) and rl.py's judge; NOT any reported eval number (graders are string-based).

Same regeneration fixes a PLACEHOLDER LEAK: narrativeqa has no bare-question field, so BookQAOracle's q_core fell back to
problem.question — which starts with the harness placeholder line on 30% of questions; 6/30 roots told every child to
answer "[The relevant text is in a separate document …] (Document length: N tokens.) …". sft.py now stores the bare
question as q_core before prepending the placeholder.

v13 narrativeqa (30 traces, 456 leaves): answer 87 -> **135**, context 0 -> **116**, none 369 -> **205**; placeholder in
subtask 6/30 -> **0/30**; acceptance 25% -> **42%**. Run-17 draft dry run: 1,149/1,149 traces, 48,845 datums, 81.0M tokens.

## OOLONG-synth failure audit (16w, all runs: 4K/3K, 8K/5K, 16K/5K, 32K/5K, 32K/8K; 255 rollouts) — 2026-09-24

Method (`scripts/audit_oolong_leaves.py`): regenerate every problem from its seed (documents verified token-identical),
take the TRUE label/user/date of every line, and compare each leaf's boxed tally with the truth for the lines it owns,
in the key space the root chose, under the question's own filters (user subset / month / date range / named labels).
Calibration: the same comparison on PERFECT rollouts also shows large leaf errors — perfect scores often survive leaf
noise because the question (e.g. "is entity more or less common than location?") is insensitive to it.

Score lost on imperfect rollouts (sum of 1 - score = 148 points):
| category | points lost | share |
|---|---|---|
| leaf label errors (leaves misjudge items' labels) | 99.7 | 67% |
| root overflow (state outgrew the budget) | 43.0 | 29% |
| no boxed answer / merge-resolution errors | 4.2 | 3% |

Leaf label accuracy by dataset (clean read: unfiltered 1-D label tallies, 17,765 items):
app_reviews 94.3% · agnews 91.3% · imdb 91.2% · trec_coarse 89.9% · yahoo 83.5% · spam 80.3% · metaphors 65.1% ·
multinli 60.6% · **formality 51.0%** (chance) · **negation 45.7%** (BELOW chance) — overall 77.9%.
negation = judging odd definitional claims true/false ("A thumb is an asset of special worth or utility." -> False;
"Not a single pilot may be someone who is licensed to operate an aircraft in flight." -> False) — lexical world
knowledge + negation; formality is subjective annotator style. These label semantics are what training on OOLONG's
source datasets teaches and what our no-eval-data policy leaves to the base model's judgment.
Overflows (29%): mostly the over-carried joint tallies the v11 minimal-state corpus targets (user A vs B, user-most-with-L,
months L1>L2, first-month); the rest are inherently large states (months-L-single-most, dates-exactly-n).

## sft_general17w (run 17, warm start from 14w) — 2026-09-24 — 4K SCORE-9 0.828; OOLONG @10K 0.619; RULER-13 @10K 0.923 / @40K 0.887

(ckpt `tinker://eb4c3882-959b-5ee6-81a0-35f0715fe8fd:train:0/weights/sft_general17w`; 1,149 traces / 48,849 datums / 81M tokens,
3,054 batches, SFT 15:20→20:40. Script `scripts/run_sft17_warm.sh`: run-16 recipe + v11 minimal states + v12 vt contrast +
niah_bridge 80 + sequential synth x2 + v13 leaf-tools fix. Raw `eval_results/raw/sft_general17w*`.)

**4K scoreboard (16w -> 17w):** SCORE-9 0.810 -> **0.828**; vt 0.88 -> **1.00**, oolong_temporal 0.26 -> 0.35, oolong_user
0.68 -> 0.71, counting 0.51 flat, multiquery 0.95 -> 0.90, cwe 1.00 -> 0.98. (qa_1/qa_2 now on the filtered pool — not
comparable with 16w's 4K numbers.)

**OOLONG on the June chart's problems** (`scripts/eval_oolong_chart.sh`: OOLONG_BASE=2000000, 10/family, 8K budget; standard
limits — no depth cap, MAX_NODES = 4 x nominal tree):
| | 10K | 40K |
|---|---|---|
| 17w | **0.619** (counting 0.46, user 0.78, temporal 0.62) | 0.354 (counting 0.26, user 0.45, temporal 0.35) |
| June OOLONG-only fine-tune (`sft_oolong`) | 0.532 | **0.562** |
| gpt-5.4 single-shot | 0.561 | 0.338 |
17w beats both at 10K. At 40K it falls far below the OOLONG-only fine-tune (same overflow count, 2 each; one 17w tree hit
the 1,020 cap). The gap is mostly oolong_user (0.45 vs 0.82): roots with the right minimal state (per-user tally of one
label; per-label tally over one user) answering wrong, plus one root that never decomposed ("User: N/A") and one
garbled goal. Leaf label accuracy on the same 40K problems where both are auditable: formality 96% (June) vs 75% (17w),
spam 94% vs 82%, negation 87% (June) vs 46% (16w); imdb ~97% both. The OOLONG-trained leaves judge the hard label
semantics far better — the audit's leaf-label ceiling, now measured against a model trained on these label sets.

**RULER-13 @ 8K budget, fresh seeds:** 10K **0.923** (all NIAH 1.00 except multiquery / multivalue 0.90; qa_1 0.80, qa_2 0.40,
vt 1.00); 40K **0.887** (multiquery 0.65, cwe 0.80, fwe 0.93, qa_1 0.80, qa_2 0.40, vt 1.00; max tree 561 < cap).

## Open-ended QA: RULER string match + an LLM equivalence grade — 2026-09-24

The RULER QA question-pool filter (unit-bearing numeric golds) is REVERTED — `tasks/ruler/qa_data.py` serves RULER's full
pool again (SQuAD 5,928, HotpotQA 7,405). Surface-form mismatches are handled at grading time instead: RULER's
string_match_part stays the benchmark-comparable number and an LLM equivalence grade is reported alongside it
(`eval/qa_judge.py`, `scripts/judge_open_qa.py <rollouts.jsonl …>`; judge `anthropic/claude-opus-4-6`, temp 0, binary
"same entity/value/fact as any reference", cache `eval_results/qa_judge_cache.json`). The judge credits number words (`4` /
"four"), aliases (`Caligula` / "Gaius Julius Caesar Augustus Germanicus"; `Mary O'Connell` / "Sister Anthony, S.C."),
implied units (`35` / "35 people") and paraphrases (`Bigg Boss 10` / "the tenth season"); it REJECTS hedges ("KPMG and Ernst &
Young", which string match credits), bridge-stops ("Derek and the Dominos" for "how many") and wrong entities.
Calibration (`scripts/calibrate_qa_judge.py`, real answers from our rollouts, hand-labelled): 20/20.

| rollouts | qa_1 string / judged | qa_2 string / judged |
|---|---|---|
| 14w 4K (RULER-7 run) | 1.00 / 1.00 | 0.40 / 0.80 |
| 16w 4K | 1.00 / 1.00 | 0.40 / 0.80 |
| 16w 8K / 16K / 32K (5K budget) | 0.60 / 1.00 · 0.80 / 1.00 · 0.60 / 1.00 | 0.40 / 0.60 · 0.60 / 0.80 · 0.60 / 0.60 |
| 16w 16K filtered pool, n=10 | 0.70 / 0.90 | 0.30 / 0.60 |
| 17w 4K | 0.80 / 0.80 | 0.80 / 0.80 |
| **17w 10K / 40K (8K budget)** | 0.80 / 1.00 · 0.80 / 0.80 | 0.40 / **1.00** · 0.40 / **1.00** |
17w's qa_2 is judged 5/5 at both lengths (string 2/5): every miss is `4`/"four", `Caligula`, `Bigg Boss 10`; its bridge
chaining now reaches RULER ("Chaining them: «Derek and the Dominos were … formed by Clapton, Whitlock, Radle and
Gordon» — so the answer is 4"). Note: the 17w 10K/40K and 16w filtered-pool runs drew qa questions from the (then)
filtered pool.

## RULER multiquery regression in 17w + fix (cache v14) — 2026-09-24

17w: multiquery 1.00 (16w, all lengths) -> 0.90 @4K / 0.90 @10K / 0.65 @40K. Never a wrong value — asked keys come back
`none`. The owning leaf READ the sentence and wrote "- magic number for soggy-roast (other key, skip)" although soggy-roast
was an asked key. By position in the asked list (all 17w multiquery rollouts): key 1 missed 0/15, key 2 1/15, **key 3 5/15,
key 4 5/15**. No 14w/15w/16w multiquery tree ever wrote an "(other key, skip)" line (RULER's multiquery haystack holds only
the asked keys' needles) — the false skip is new in 17w, whose v11 corpus made skip verdicts far more common.
Cause in the leaf format: niah_multi was the ONLY filtered leaf whose op phrase did not name its filter values ("keeping
each stated magic number for the asked key(s)…") — every other filtered leaf names them ("…of author Ivanov's items only",
"…for `Building` and `Village` only", "…for label=Sports and label=Sci/Tech only"), so this leaf had to recall the list from
its subtask. Fix, no new format: (1) `oracle/prose.py` `_op_phrase` names the kept keys ("keeping each stated magic number
for escarpment-ember, brindle-quarry, marble-tempest, vellum-kiln and cobalt-thicket only (key=value) and skipping every
other key"); hidden mode unchanged. (2) multiquery asks 3-5 keys (3-4 with uuids; was 2-4 / 2-3) so kept keys in positions
3-5 are common. Verified niah_multi 300/300 gold at doc 6K/14K; worst node 2.3K -> 2.74K (5 uuid-ish keys; < 3K).

## STATE BEFORE RUN 18 — 2026-09-24 (handoff summary)

**Best checkpoint: 17w** `tinker://eb4c3882-959b-5ee6-81a0-35f0715fe8fd:train:0/weights/sft_general17w` (warm from 14w;
recipe `scripts/run_sft17_warm.sh`, cache v13). 4K SCORE-9 0.828; OOLONG @10K 0.619 (June OOLONG-only fine-tune 0.532,
gpt-5.4 0.561); OOLONG @40K 0.354 (June fine-tune 0.562, gpt-5.4 0.338); RULER-13 @10K/8K 0.923, @40K/8K 0.887; vt 1.00 at
4K/10K/40K; qa_2 LLM-judged 1.00 at 10K and 40K (string 0.40). Probes (fold / split-decision) NOT yet run on 17w.

**Corpus changes since run 17 launched (all go into run 18):**
1. v14 — niah_multi filtered leaves name the kept keys in the op phrase; multiquery asks 3-5 keys (fixes 17w's multiquery
   false skips of asked keys 3-4: 0.65 @40K). Verified niah_multi 300/300, worst node 2.74K.
2. (Everything else in the run-17 corpus stays: v11 minimal states, v12 vt niah-surface, niah_bridge 80, sequential synth
   x2, v13 leaf-tools fix + bare-question q_core, budget x length jitter.)
Run-18 script: copy `scripts/run_sft17_warm.sh` (cache v14 regenerates niah_multi; everything else hits cache). Starting
checkpoint to decide (14w for the clean one-pass comparison vs 17w for continuity).

**Eval-side changes (no training effect):**
- RULER QA pool: the unit-bearing-numeric filter was added then REVERTED — full pool again. Surface-form mismatches are
  handled by reporting an LLM equivalence grade next to RULER's string match: `eval/qa_judge.py`,
  `scripts/judge_open_qa.py <rollouts.jsonl…>` (Claude Opus 4.6, cached), calibration `scripts/calibrate_qa_judge.py` (20/20).
- `eval/backends.py`: `complete()` never offers tools (v13 bug fix; competitor evals use `sample()` directly — unaffected).
- `scripts/eval_long.sh` (RULER-13 + OOLONG, fresh seeds 3,000,000+) and `scripts/eval_oolong_chart.sh` (OOLONG on the June
  chart's problems, OOLONG_BASE=2000000, 10/family, 8K budget): both no depth cap, non-binding chunk limit, and
  MAX_NODES = 4 x the nominal binary tree (2 x next-pow2(doc/500) - 1) — healthy trees stay <= 2x nominal, every tree past
  ~3.5x failed, so the cap only trims runaways. 4K scoreboard keeps MAX_NODES=150 for comparability.
- OOLONG-real built (eval-only; mapping in task_context), grader validated vs the paper; no fine-tuned checkpoint run on it.
- `scripts/audit_oolong_leaves.py`: regenerates OOLONG problems from seeds and checks every leaf's tally against the truth.

**Open threads:**
- LEAF CLASSIFICATION (next focus): 67% of OOLONG points lost are leaf label errors. Leaf accuracy by dataset (16w):
  app_reviews 94%, agnews 91%, imdb 91%, trec 90%, yahoo 84%, spam 80%, metaphors 65%, multinli 61%, formality 51%, negation
  46% (below chance: odd definitional claims judged true/false). On the same 40K problems the June OOLONG-only fine-tune's
  leaves score formality 96% vs 17w 75%, spam 94% vs 82%, negation 87% — label semantics learned from OOLONG's own label
  sets, which our no-benchmark-data policy excludes. 29% of lost points are root overflows (partly addressed by v11).
- qa_2 multi-hop: solved semantically; string-grader artifacts remain by design (reported with the LLM grade).
- vt fold decision: 17w folds 5/5 at every length; fold probe pending.

## LEAF CLASSIFICATION INVESTIGATION + cache v15 — 2026-09-25

**Finding: 17w's OOLONG leaves label ~74% of items correctly in-tree; the same model labels 87.5% with a plain prompt,
base Qwen 91.3%, the June OOLONG-only fine-tune 93.1% plain / 94.0% in-tree.** Measured item-by-item
(`scripts/leaf_classify_probe.py`: OOLONG items from fresh seeds 7,000,000+, leaf-sized batches of <=500 tokens, temp 0;
in-tree = the leaves' own per-line verdicts in `sft_C17w_{10k,40k}` / `sft_oolong_40k`):

| dataset | base plain | June plain | June in-tree @40K | 17w plain | 17w in-tree @40K |
|---|---|---|---|---|---|
| negation | 86.2 | 91.6 | 88.7 | 73.6 | 57.6 |
| formality | 87.6 | 89.1 | 95.1 | 73.5 | 73.0 |
| multinli | 87.5 | 88.9 | 91.3 | 87.2 | 50.2 |
| metaphors | 97.5 | 96.5 | 98.4 | 93.2 | 64.6 |
| overall | 91.3 | 93.1 | 94.0 | 87.5 | 73.5 |

17w's in-tree errors run one way (false->true 455 vs 16, incorrect->correct 563 vs 44, contradiction->neutral). In-tree
per-item scoring matches verdicts to items by POSITION — shifted leaves (partial first line counted, items dropped) blur it;
rewrite it to match by quoted text before the run-18 comparison. Leaf accuracy is ~equal at 10K (75.8%) and 40K; 10K
scores higher only because fewer items compound fewer errors. Verbatim leaves: `trace_snippets/oolong_17w_leaf_rollouts_40k.txt`.

**Causes found (and refuted):**
1. ROOT wrote lookup contracts. 25/29 17w OOLONG roots @40K named no judgment in the subtask: "(exact count)",
   "(inclusive)", "(no tag means the label is True)", "(no label judgement needed)". Every labeled_records question began
   "Judge each item's label …" and every context said "you must judge each item yourself", so the root's "(judging each
   item's label)" was always copyable; OOLONG never says "judge".
2. Training GOLD often not judgeable from the text. Opus 4.6 (text-only, may answer `unclear`) vs our gold: emotion 59%
   agree (34% disagree outright: "i couldn t feel positive emotions of any sort" = joy), claims 86% (12% need outside
   knowledge: obscure people/places), yelp 91%, dbpedia 99%. Base Qwen scores 91-99% on the Opus-kept rows — the sources
   are easy or noisy, with little hard-but-judgeable material.
3. Leaf headers stated the item count BEFORE listing: 17w @40K 96% headers right but 9% of leaves listed a different number
   than their header (e.g. "Labelling the 7 sentences", 5 listed, the dropped one the only False).
4. 17w leaves mostly write no content before the verdict (`- line at token N: label=entity`, 83% of lines @40K; June quotes
   the sentence 90%) — synth's `field=value` leaf habit on OOLONG's field-record lines.
- REFUTED: quoting metadata as THE cause (17w accuracy by quote kind: content 72.5%, none 73.4%, metadata 77.3%); the
  question naming a label biasing verdicts (75.3% vs 74.1%); a per-item reason before the label (base 91.3 -> 90.1,
  17w 87.5 -> 86.1).

**cache v15 corpus changes (all labeled_records unless noted):**
1. Judgment-wording jitter: question opener (9-way pool, 3 empty, 1 "Judge"), count wording (6-way, incl. "should be
   classified as"), context label sentence (10-way, incl. "The overall sentiment of each review can be classified into
   one of 2 categories"), item noun (items/instances/data points/entries/lines) and "`L` items" / "items with the label
   `L`" / "items labelled `L`". 79% of problems never say "judge"; every root subtask still says "(judging each item's
   label)". author_count's goal now reads "(counting items per author; labels do not matter)" instead of the
   copyable "(no label judgement needed)".
2. ALL oracles: leaf partial/fold headers state no item count ("Judging the items whose line STARTS in a..b …").
3. Label-quality filter: rows kept only where Opus = gold (`scripts/label_judge.py`, `scripts/write_label_keep.py`,
   `~/.cache/infinite-context/labeled/{name}.opus_keep.json`; `LABELED_UNFILTERED=1` bypasses). Kept: dbpedia 3956/4000,
   yelp 3624/3977, claims 4670/5450, emotion 2292/3911. (~$25 of Opus for all seven sources.)
4. New judgment types, sources disjoint from OOLONG's: SNLI (3534 kept, entailment/neutral/contradiction), PAWS (3465,
   paraphrase / not paraphrase), Stanford politeness (Cleanlab copy, 1391 of 2366, polite/neutral/impolite). Pairs share a
   line joined by a per-problem marker (` -> `, ` <--> `, ` | `, ` => `, ` // `) and are quoted IN FULL by the leaf (PAWS
   pairs differ late in the sentence). Source mix: dbpedia 1, emotion 1 (was 2), yelp 2, claims 2, snli 2, paws 1,
   politeness 1.
5. Record layouts: bracket `[S2] [by Kim] text` 40% / pipe `Section: S1 || Posted by: 96016 || Item: …` 30% / key=value
   `section=S1; posted_by=Sato; item=…` 30%; field names jittered; context, questions and the root's contract name the
   layout's fields; the leaf's per-item notation is unchanged.
Verified: 400/400 labeled_records problems solve across layouts and sources, 25,200/25,200 nodes answered, largest node
2,020 tokens, 0 bracket-reference leaks in non-bracket layouts. Snippets: `trace_snippets/labeled_v15_{judgment_wording,
new_sources,record_layouts}.txt`.

**For run 18:** v14 + v15. Measure with the plain probe (does 17w's lost plain-prompt judgment come back toward base
91.3%?) and the in-tree scorer (fixed to match by quote). Warm start (14w vs 17w) still to decide.

## sft_general18w (run 18, warm start from 14w, cache v14+v15) — 2026-09-25 — OOLONG @40K **0.535** (17w 0.354); RULER-13 @10K **0.949** / @40K **0.929**; 4K SCORE-9 0.800

Checkpoint `tinker://85b93811-f926-5129-89e3-5a65c95707f1:train:0/weights/sft_general18w` (also pinned in
`~/.cache/infinite-context/ckpt_sft_general18w.txt`). Recipe `scripts/run_sft18_warm.sh` = run-17 recipe and mix on the
v15 corpus (leaf-classification changes above + v14 multiquery). 1,149 traces / 47,142 agents / 48,794 datums / 81.2M
tokens, 3,050 batches, LR 5e-6, final running NLL ~0.006; trained 01:47-06:05. Post-run chain `scripts/after_run18.sh`
(watcher): gate for the from-base run 18b was SCORE-9 > 0.828 -> NOT passed (0.800), 18b not launched.

**Leaf classification (the target):**

| | 17w | 18w | base Qwen | June OOLONG-only |
|---|---|---|---|---|
| plain-prompt item accuracy (probe, fresh seeds) | 89.9% | **91.6%** | 91.3% | 93.6% |
| in-tree item accuracy @40K (C-chart leaves) | 73.5% | **86.8%** | — | 94.0% |
| leaves listing != owned item count @40K | 338 | **23** | — | — |

In-tree @40K by dataset, 17w -> 18w: multinli 50 -> 88, metaphors 65 -> 92, negation 58 -> 66, formality 73 -> 81,
trec 75 -> 82, yahoo 81 -> 86, spam 84 -> 96, agnews 87 -> 94, app_reviews 89 -> 95, imdb 95 -> 97. Plain negation 74 ->
80 (base 86) — the one judgment still below base. (CORRECTION: 17w plain was 89.9% / formality 94% with the robust parser
that reads `i: label`, `1. label (…)` and label-first forms; the 87.5% / 73.5% reported earlier were parse misses.
In-tree scoring still matches by position — rewrite to match by quote before relying on small differences.)

**4K scoreboard (3K budget, standard seeds; n=5 RULER/OOLONG, n=2 others):** SCORE-9 0.800 (17w 0.828); OVERALL
all tasks 1.00 except: oolong_counting 0.21 (17w 0.51), oolong_user 0.60 (0.71), oolong_temporal 0.53 (0.35), qa_2 0.60
(0.80; LLM-judged 0.80 — `500` for "500-room"), niah_multiquery 0.90 (0.90), vt 0.96 (1.00), qa_1 1.00 (0.80),
niah_multivalue 1.00 (0.90), cwe 1.00 (0.98). The SCORE-9 drop is entirely the 15 OOLONG rollouts (RULER part 0.977 vs
0.980) — not yet inspected.

**OOLONG-synth, June chart problems (OOLONG_BASE 2,000,000, 10 per family = 1 per dataset, 8K budget):**

| | 17w @10K | 18w @10K | 17w @40K | 18w @40K |
|---|---|---|---|---|
| counting | 0.46 | 0.45 | 0.26 | **0.42** |
| user | 0.78 | **0.86** | 0.45 | **0.80** |
| temporal | 0.62 | 0.51 | 0.35 | 0.39 |
| **mean** | **0.619** | 0.606 | 0.354 | **0.535** |

References on these problems: June OOLONG-only fine-tune 0.532 @10K / **0.562** @40K; gpt-5.4 0.561 / 0.338.

**RULER-13 + OOLONG, fresh seeds 3,000,000+ (8K budget; RULER n=5, OOLONG n=20/family):**

| task | 17w @10K | 18w @10K | 17w @40K | 18w @40K |
|---|---|---|---|---|
| niah_single_1/2/3, multikey_1/2/3 | 1.00 | 1.00 | 1.00 | 1.00 |
| niah_multiquery | 0.90 | **1.00** | 0.65 | **1.00** |
| niah_multivalue | 0.90 | **1.00** | 0.95 | **1.00** |
| vt | 1.00 | 1.00 | 1.00 | 1.00 |
| cwe | 1.00 | 1.00 | 0.80 | 0.74 |
| fwe | 1.00 | 0.93 | 0.93 | 0.93 |
| qa_1 (string / LLM-judged) | 0.80 / 1.00 | 0.60 / 0.80 | 0.80 / 0.80 | 0.80 / 1.00 |
| qa_2 (string / LLM-judged) | 0.40 / 1.00 | 0.80 / 0.80 | 0.40 / 1.00 | 0.60 / 0.60 |
| **RULER-13 mean (string)** | 0.923 | **0.949** | 0.887 | **0.929** |
| oolong counting / user / temporal | — | 0.63 / 0.90 / 0.62 | — | 0.57 / 0.75 / 0.50 |

(17w's qa rows drew from the then unit-filtered question pool, 18w's from the restored full pool — the qa rows are not the
same questions. 17w's R runs predate OOLONG in eval_long.) 18w qa judge disagreements: `High risk preparations and other
compounding functions` (string 0, judged 1), `35` for "35 people" (0 -> 1), `KPMG and Ernst & Young` (string 1, judged 0 —
a hedge).

**Reading:** v15 did what it was built for — in-tree leaf accuracy 73.5% -> 86.8%, plain judgment back above base, and
OOLONG @40K 0.354 -> 0.535 (user 0.45 -> 0.80), within 0.03 of the model trained on OOLONG itself. v14 fixed multiquery
(0.65 -> 1.00 @40K). RULER-13 is the best at both lengths. Open: the 4K OOLONG dip (counting 0.51 -> 0.21 on 5 rollouts),
temporal @10K 0.62 -> 0.51, negation still below base, cwe @40K 0.74. Raw: `eval_results/raw/sft_general18w*`,
probe `eval_results/leaf_classify_18w.json`.

### 18w @80K (8K budget) — 2026-09-25 — OOLONG chart **0.561** (June OOLONG-only 0.429, gpt-5.4 0.327); RULER-13 0.885

| | 10K | 40K | 80K |
|---|---|---|---|
| OOLONG chart (June problems) counting / user / temporal | 0.45 / 0.86 / 0.51 | 0.42 / 0.80 / 0.39 | 0.41 / 0.73 / 0.54 |
| **OOLONG chart mean** | 0.606 | 0.535 | **0.561** |
| OOLONG fresh seeds (n=20/family) counting / user / temporal | 0.63 / 0.90 / 0.62 | 0.57 / 0.75 / 0.50 | 0.51 / 0.69 / 0.44 |
| niah (all 9 variants) | 1.00 | 1.00 | 1.00 |
| vt | 1.00 | 1.00 | 0.92 |
| cwe | 1.00 | 0.74 | **0.58** |
| fwe | 0.93 | 0.93 | 1.00 |
| qa_1 / qa_2 (string) | 0.60 / 0.80 | 0.80 / 0.60 | 0.60 / 0.40 (not yet LLM-judged) |
| **RULER-13 mean** | 0.949 | 0.929 | **0.885** |

The June chart's 80K dip (0.562 -> 0.429, temporal tallies overflowing) is gone: 18w holds 0.56 at 80K. Merge capacity at
8K budget: answered merges hold up to **214 entries** (oolong_user `userID:count`; temporal 103; 6 chart / 43 RULER-run
merges at 101-200+); overflows happen when two children bring ~400 entries combined (temporal 20, user 2). fwe's 6
overflowed merges are NOT size (children 54 entries combined) but a repetition loop (`gihmd:1|gihmd:1|…`,
`zrtdgy:3|…` until the budget) — the tree recovered (both rollouts 1.00; fwe's Zipf head survives a lost branch). Same
loop family as cwe's `section (4)` x6. cwe falls 1.00 -> 0.74 -> 0.58: the pruned-tally ceiling (leaves keep <=10 words
seen 2+ times; at 40K+ a common word occurs ~0.4x per 500-token leaf). Simulation on RULER's cwe construction (exact
counts, top-M kept at every node): M=10 0%/0% at 40K/80K, M=50 60%/0%, M=100 80%/90%, M=200 100%/100%.
Raw: `eval_results/raw/sft_general18w_{C18w_80k,R18w_80k_b8k}.*`.

**Deferred format cleanup (2026-09-25):** every scripted leaf writes `Partial = X.` and then `\boxed{X}` — the same X
twice, in every task (`Partial = 0.` / `\boxed{0}`; topk: up to K entries twice). Wasted output tokens everywhere, and
the dominant cost in topk leaves. Not changed now (shared leaf convention, close to the deadline); if changed, change it
for ALL tasks in one pass (base.py leaf turn), never per task.

### 18w open-QA, LLM-judged (Opus 4.6 equivalence judge, `scripts/judge_open_qa.py`) — 2026-09-25

| (8K budget, fresh seeds, n=5) | 10K string / judged | 40K string / judged | 80K string / judged |
|---|---|---|---|
| qa_1 | 0.60 / **0.80** | 0.80 / **1.00** | 0.60 / **0.80** |
| qa_2 | 0.80 / **0.80** | 0.60 / **0.60** | 0.40 / **0.80** |
| 4K scoreboard (3K budget) qa_1 / qa_2 | 1.00 / 1.00 · 0.60 / 0.80 | | |

String-vs-judge disagreements (surface form, not errors): `High risk preparations and other compounding functions`
(gold "…and some other…", judged correct at all lengths), `35` for "35 people", `EY` for "Ernst & Young" (80K), `500` for
"500-room" (4K); `KPMG and Ernst & Young` (40K) is string-correct but judged a hedge. REAL failures @40K/80K (4):
- qa_2 3015004 "Ravi Khote … in which 2003 Indian drama?" fails at BOTH 40K ("No relevant information") and 80K
  ("Baghban (film)" — a 2003 drama "directed by Ravi Chopra", matched on the name): the leaf that read "Ravi "Rags" Khote is a
  playback singer … Some of his songs include … "Kal Ho Naa Ho"" passed up only the "his songs" sentence, losing the
  name. -> v16 niah_bridge pronoun hop-2 units (naming sentence + "They …" quoted together).
- qa_1 3014004 @80K: the leaf whose read held the gold sentence ("Hospital pharmacies usually stock a larger range of
  medications…") returned "No relevant information"; the root answered from related pharmacist context. Leaf recall miss.
- qa_2 3015001 @40K: root hedged between two firms.

## sft_general19w (run 19, warm start from 18w, cache v17) — 2026-09-25 — 4K SCORE-9 **0.864** (best); RULER-13 0.969 / 0.908 / 0.922 @10K/40K/80K; OOLONG chart 0.489 / 0.534 / 0.533

Checkpoint `tinker://9febd22f-7451-540d-ad76-4139c3e759a1:train:0/weights/sft_general19w` (pinned in
`~/.cache/infinite-context/ckpt_sft_general19w.txt`). Recipe `scripts/run_sft19_warm.sh` = run-18 mix, synth_topk 40 -> 120,
cache v17 (v16 topk top-K / one-pass leaf / word ownership / one-line docs / sparse + coded regimes, niah_bridge pronoun
units; v17 results written once, trace gates). 1,229 traces / 52,652 datums / 91.8M tokens, LR 5e-6, final NLL ~0.0017;
trained 14:18-20:37. Watcher `scripts/after_run19.sh` (long evals = OOLONG chart + RULER-13 only; fresh-seed OOLONG cut).

| | 18w | 19w |
|---|---|---|
| 4K SCORE-9 (3K budget) | 0.800 | **0.864** (oolong counting 0.21 -> 0.47, user 0.60 -> 0.80) |
| RULER-13 @10K / 40K / 80K (8K budget) | 0.949 / 0.929 / 0.885 | **0.969** / 0.908 / **0.922** |
| – cwe @40K / 80K | 0.74 / 0.58 | **0.90 / 0.86** (v16 top-K tallies; simulated ceiling ~0.97) |
| – fwe @10K / 40K | 0.93 / 0.93 | **1.00 / 1.00** |
| – qa_1 LLM-judged @10K / 40K / 80K | 0.80 / 1.00 / 0.80 | **1.00 / 1.00 / 1.00** |
| – qa_2 LLM-judged @10K / 40K / 80K | 0.80 / 0.60 / 0.80 | 0.80 / 0.60 / 0.60 |
| OOLONG chart @10K / 40K / 80K | 0.606 / 0.535 / 0.561 | **0.489** / 0.534 / 0.533 |
| – user @10K / 40K / 80K | 0.86 / 0.80 / 0.73 | **0.36** / 0.78 / 0.70 |
| plain leaf probe | 91.6% | 90.7% (negation 80 -> 77) |

**OOLONG user @10K regression (0.86 -> 0.36) is ROOT-CONTRACT breakage, not labels** — 5/10 newly failed:
- imdb "which user is represented the second most often" -> subtask "per-user tally of reviews (judging each review's
  label)" with an INVENTED closed key space "`<user>` is exactly one of `User 1` … `User 5`" -> `User: User 91202`.
- negation "which user is represented most often" -> "per-user tally of the sentences marked as true" (invented label
  filter) -> boxed `true`.
- yahoo -> "`<user>` is the user's `name:` as in the question" (no such field; layout-reference wording leaking) -> `User: none`.
- formality user-subset question -> filter to "user 885999" (not in the question) -> `Label: none`.
- multinli -> wrong user.
@40K/80K user only slips (0.80 -> 0.78, 0.73 -> 0.70); 4K user improved. Hypothesis (unverified): a second pass over the same
v15 labeled_records surfaces (layout field names, author/user wording) drifts the root's user contracts.
**Other @40K dips:** niah_multikey_1 1 -> 0.80 (one `none`); niah_multiquery 1 -> 0.90 (two answers boxed the whole
`key=value|…` state instead of the values).
**Pronoun fix did not take:** "Ravi Khote … 2003 Indian drama" still fails ("(unproven by the document)" @40K,
"Baghban (film)" @80K).

**Temperature study (18w, RULER-13 + fresh-seed OOLONG @40K):** t=0 RULER 0.899 / OOLONG 0.597, 90 overflowed nodes (40
repetition loops), fwe 0.93 -> 0.33; t=0.2 0.929 / 0.606, 54 (10 loops); t=0.4 0.911 / 0.557, 126 (20 loops). OOLONG chart
@40K t=0 0.529 vs 0.535. Keep t=0.2. The `word:1|word:1|…` loop is trained behaviour that greedy decoding exposes.

Raw: `eval_results/raw/sft_general19w*`, `eval_results/qa_judge_19w.txt`, `eval_results/probe_19w.log`.
