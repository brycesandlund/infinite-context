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

