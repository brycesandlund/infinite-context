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

## sft_general6 (run 6) — PLANNED

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
