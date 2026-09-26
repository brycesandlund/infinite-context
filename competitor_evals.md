# Competitor evals

Frontier / base models on the same OOLONG problems as our fine-tunes. Started 2026-09-25 (rerun of the June 2026
comparison in `eval_results/RESULTS.md` §2, §6).

## Protocol

**Single-shot ("full document in context").** The whole document is placed in the prompt; the model answers in one
tool-free call; graded with the same OOLONG grader as our harness runs.
- Code path: `eval/run.py` with `MODE=single` (`eval/agent.py:run_single_shot`). System prompt = the task context +
  "You are given a complete document followed by a question about it. Read the document carefully and answer the
  question. Show your reasoning, then emit your final answer as \boxed{value} and stop."; user message = document +
  question. Output cap 16,384 tokens (`OUT_TOKENS`, counts hidden reasoning).
- Runner: `scripts/competitor_oolong_chart.sh <litellm-model> <tag> <doc lengths...>` (env `TEMP`, default 0;
  `TEMP=none` omits the parameter for models that reject it).
- **Problems: the June "chart" set** — `oolong_spec(task, idx, OOLONG_BASE=2000000)`, 10 per family (counting / user /
  temporal) = one per OOLONG dataset. Same problems our checkpoints are scored on in `eval_oolong_chart.sh`
  (e.g. 18w: `eval_results/raw/sft_general18w_C18w_*`).
- Grading: OOLONG-official (numeric `0.75^|err|`, exact label/user/date membership, word-boundary comparison).
- Raw rollouts: `eval_results/competitor/<tag>_single_chart_<len>.{jsonl,txt,log}`.

**Model versioning.** Aliases can be re-pointed over time, so (a) pin dated snapshots where the provider offers them,
and (b) every rollout records the model the provider actually served: `served_model` on each assistant message and
`served_models` on each rollout row (`eval/backends.py`, `eval/agent.py`, `eval/render.py`, `eval/run.py`; added
2026-09-25). Tinker rollouts leave it empty (the checkpoint path identifies the model).

## Results — OOLONG chart problems, single-shot

OVERALL = mean of the three families (30 problems per length).

| Model | 10K | 20K | 40K | 80K |
|---|---|---|---|---|
| **gpt-5.4** (`gpt-5.4-2026-03-05`, temp 0, 2026-09-25) | **0.620** | **0.556** | **0.399** | **0.476** |
| gpt-5.4, June 2026 (`frontier_ceiling.py`, alias) | 0.561 | 0.583 | 0.338 | 0.327 |
| *ours: 18w, harness, 8K/agent budget* | *0.606* | *—* | *0.535* | *0.561* |

Per family, gpt-5.4 (2026-09-25):

| family | 10K | 20K | 40K | 80K |
|---|---|---|---|---|
| counting | 0.449 | 0.346 | 0.305 | 0.301 |
| user | 0.775 | 0.800 | 0.450 | 0.507 |
| temporal | 0.635 | 0.521 | 0.443 | 0.620 |

All 120 rollouts answered (no truncation at the 16K output cap); every rollout reports `gpt-5.4-2026-03-05`.

### gpt-5.4 WITH reasoning (`reasoning_effort=medium`) — 2026-09-25

**Finding: every gpt-5.4 number above (and June's) is gpt-5.4 WITHOUT reasoning.** In our setup its default reasoning
effort is "none": 0 reasoning tokens on OOLONG prompts, answers in 100-370 output tokens. The OOLONG paper runs GPT-5 at
its default (medium) and RLM uses "medium reasoning", so medium is the literature-matching baseline. OpenAI accepts
ONLY temperature 1 when reasoning is on (temperature 0 → "gpt-5 models don't support temperature=0 ... supported when
reasoning_effort='none'"), so this row is `TEMP=none REASONING_EFFORT=medium OUT_TOKENS=65536`.

| 40K chart | counting | user | temporal | overall |
|---|---|---|---|---|
| **gpt-5.4, medium reasoning** (64K output cap) | 0.237 | 0.875 | 0.689 | **0.600** |
| gpt-5.4, no reasoning (temp 0) | 0.305 | 0.450 | 0.443 | 0.399 |
| *18w, harness, 8K/agent* | *0.42* | *0.80* | *0.39* | *0.535* |

**gpt-5.4 medium reasoning, full chart row** (temperature omitted = 1; all 210 answered, no call hit its cap):

| | 10K | 20K | 40K | 80K | 160K | 320K | 640K |
|---|---|---|---|---|---|---|---|
| overall | **0.708** | **0.662** | **0.600** | **0.500** | **0.556** | **0.479** | **0.419** |
| counting / user / temporal | 0.601 / 0.856 / 0.665 | 0.510 / 0.800 / 0.677 | 0.237 / 0.875 / 0.689 | 0.301 / 0.573 / 0.626 | 0.306 / 0.887 / 0.475 | 0.300 / 0.412 / 0.725 | 0.200 / 0.484 / 0.573 |
| output tokens median / max | 5.0K / 22.3K | 9.8K / 31.5K | 11.8K / 26.1K | 10.3K / 43.1K | 8.8K / 50.0K | 5.1K / 58.9K | 6.3K / 32.2K |
| output cap | 64K | 64K | 64K | 64K | 128K | none | none |

| 80K chart | counting | user | temporal | overall |
|---|---|---|---|---|
| **gpt-5.4, medium reasoning** (64K output cap) | 0.301 | 0.573 | 0.626 | **0.500** |
| gpt-5.4, no reasoning (temp 0) | 0.301 | 0.507 | 0.620 | 0.476 |
| *18w, harness, 8K/agent* | *0.41* | *0.73* | *0.54* | *0.561* |

80K: all 30 answered; output tokens median 10.3K / max 43.1K (4 calls > 32K; none hit 64K).

40K: all 30 answered; reasoning tokens median 11.6K / max 25.7K (output 11.8K / 26.1K) — a 16K cap would have truncated
several. Reasoning lifts user/temporal sharply; exact counts do not improve (single-pass counting of ~800 items stays
off by enough that 0.75^|err| ≈ 0). ~$8 for the length. Raw: `eval_results/competitor/gpt5_4_rmed_out64k_single_chart_40k.*`.
Test on one 10K problem (gold 38): no reasoning 53/53/92 (t0), 58/57/58 (default); medium 49 (14.3K reasoning tokens);
high 43 (24.6K).

### Base Qwen3.6-35B-A3B (no fine-tune), single-shot, 20K chart — 2026-09-25

Tinker, our usual renderer (thinking disabled). The output limit decides this model's score, not the temperature:

| Output limit | temp 0 | temp 0.2 | no answer (t0 / t0.2) | output tokens median / max |
|---|---|---|---|---|
| 16K output cap | 0.370 | 0.529 | 9 / 9 | — (capped) |
| **64K total context** (`SINGLE_CONTEXT=65536`: output = 64K − prompt ≈ 44K) | **0.558** | **0.571** | 2 / 1 | 13.3K / 48.0K (t0) · 11.5K / 47.6K (t0.2) |

Per family, 64K context: t0 counting 0.478 / user 0.675 / temporal 0.523; t0.2 0.448 / 0.675 / 0.589.
- Base Qwen solves these by enumerating items one by one; its median answer is 11-13K tokens, so a 16K cap truncated
  many answers (18 of 60 truncations across both 16K runs; ~2/3 were still making progress, ~1/3 repetition loops such
  as `- Count: 1` x258). With room to finish, temp 0 vs 0.2 differ by 0.013.
- At 20K base Qwen single-shot ≈ gpt-5.4 single-shot (0.56-0.57 vs 0.556).
- **Context ceiling: Tinker serves this model with a 64K window** (the model itself supports 262K; Tinker's tokenizer
  still reports 262,144). Base-Qwen single-shot is therefore limited by Tinker's serving limit, not the model: 40K docs
  leave ~24K for output, 80K docs do not fit. Our harness runs are unaffected (each agent uses <= 8-12K).

**Base Qwen, temp 0, Tinker max context (64K total) — the chart row:**

| | 10K | 20K | 40K | 80K |
|---|---|---|---|---|
| overall | **0.536** | **0.558** | **0.388** | n/a (80K doc > 64K window) |
| counting / user / temporal | 0.507 / 0.642 / 0.460 | 0.478 / 0.675 / 0.523 | 0.359 / 0.475 / 0.330 | — |
| no answer (out of room / no \boxed) | 1 / 1 | 2 / 0 | 5 / 1 | — |
| output tokens median / max | 8.1K / 56.6K | 13.3K / 48.0K | 18.2K / 37.5K | — |

**40K chart, 64K total context:**

| | temp 0 | temp 0.2 |
|---|---|---|
| overall | **0.388** | **0.326** |
| counting / user / temporal | 0.359 / 0.475 / 0.330 | 0.102 / 0.600 / 0.275 |
| no answer: ran out of room / stopped without \boxed{} | 5 / 1 | 10 / 1 |
| output tokens median / max | 18.2K / 37.5K (ceiling) | 21.8K / 37.5K (ceiling) |

- **No consistent temperature effect:** 0.2 leads at 20K (+0.013), 0 leads at 40K (+0.062) — within run-to-run noise
  for 30 problems.
- At 40K the ~24K output room is below base Qwen's median answer length, so it runs out of room often (5 / 10); that,
  not temperature, drives most of the 40K gap (e.g. t0.2 counting 0.102).
- At 40K: base Qwen single-shot 0.33-0.39 ≈ gpt-5.4 single-shot 0.399 < 18w harness 0.535.

## Results — RULER-13, single-shot (same problems as the fine-tune `eval_long.sh` runs)

Runner `scripts/competitor_ruler.sh`: `eval_long.sh`'s exact 16-task order with the three OOLONG tasks at 0 problems
(so each RULER task keeps its seed index), `SEED_OFFSET=3000000`, 5 per task. **Problem identity verified** against
18w's 10K fine-tune rollouts: same 65 (task, seed) pairs; 60/65 byte-identical question+gold; the 5 vt problems differ
only in the "(Document length: N tokens)" line (±1) with the same gold set — vt's generator orders items by
per-process Python hash, so its layout reshuffles between ANY two runs (fine-tune runs included); vt is set-graded.

**Base Qwen3.6-35B-A3B, single-shot, temp 0, Tinker 64K total context:**

| | 10K | 20K | 40K |
|---|---|---|---|
| **RULER-13 mean (string)** | **0.923** | **0.938** | **0.938** |
| RULER-13 mean, qa_2 LLM-judged | 0.938 | 0.954 | 0.954 |
| niah x9, vt, qa_1 | 1.00 each | 1.00 each | 1.00 each |
| cwe | 0.80 | 0.60 | 1.00 |
| fwe | 0.40 | 0.80 | 0.40 |
| qa_2 string / judged | 0.80 / 1.00 | 0.80 / 1.00 | 0.80 / 1.00 |
| no answer (ran out of context) | 4 (cwe 1, fwe 3) | 3 (cwe 2, fwe 1) | 3 (fwe 3) |
| *18w harness (8K/agent, temp 0.2), string* | *0.949* | *—* | *0.929* |

- Base Qwen is perfect on every retrieval / tracking / single-hop task at <=40K when the whole doc fits; its only losses
  are word-frequency tasks (cwe/fwe), almost all from running out of the 64K window while writing out word counts,
  plus the qa_2 `35` vs "35 people" surface form (judged correct).
- 18w @40K per task: cwe 0.74, fwe 0.93, qa_1 0.80, qa_2 0.60 (string), everything else 1.00.
- Raw: `eval_results/competitor/base_qwen_t0_ctx64k_single_ruler_{10,20,40}k.*`.

## Reproducing June's gpt-5.4 numbers — what changed and why the numbers move

- **Same problems.** At 20K and 80K all 30 problems match the June logs exactly (dataset + gold per problem); 10K/40K
  use the same generator. OOLONG grading is unchanged since June (the only `tasks/base.py` changes since added new
  modes, `qa_part` and `oolong_real`).
- **Same model.** The `gpt-5.4` alias resolves to `gpt-5.4-2026-03-05`, the only full gpt-5.4 snapshot (the others are
  mini / nano / pro), released before June — so June used the same weights.
- **Prompt structure differs.** June (`frontier_ceiling.py`, removed in commit 321b3c0): one user message
  `{task_context}\n\n{document}\n\n{question}`, no output cap. Now: task context in a system prompt plus the
  "show your reasoning" line; document + question as the user message; 16K output cap.
- **Temperature 0 really applies** (gpt-5.4 accepts it; LiteLLM's `drop_params` does not strip it) — yet answers still
  vary run to run on these long prompts: two 10K runs today gave 0.591 and 0.620; at 20K, 12 of 30 problems changed
  score vs June in both directions. Exact-count questions swing most (e.g. 80K metaphors: June 17 → now 7, gold 7).
- **June's 80K was deflated by API errors.** The June 80K log contains LiteLLM errors and 2/30 problems with no answer
  (temporal/spam, temporal/multinli), both scored 0; both are answered now (1.000, 0.750) — ≈+0.06 of the 80K gap.
- Net: 10K–40K within ±0.06 of June (noise); 80K +0.15 (errors + count swings). **The June 80K frontier baseline
  (0.327) understated gpt-5.4 by ~0.15.**

## Output limits and temperature (protocol decisions)

- **Output cap.** A cap is fair when it is uniform, fixed in advance, generous enough not to decide answers for a model
  making real progress, and truncations are reported. 16K was NOT generous enough for base Qwen (above); gpt-5.4 never
  approached it. Working choice: 64K (model-total-context limited where the model's window is smaller), truncations
  reported per run.
- **Temperature.** Greedy / temperature 0 is the usual choice for short-answer long-context benchmarks where the model
  allows it; reasoning models that forbid it (Fable 5.1) run at provider defaults, stated per model. Our fine-tune
  evals have run at 0.2 — a same-temperature comparison needs base Qwen and our models at the same value. (OOLONG
  paper's setting not yet checked.)

### Temperature conventions in the literature (checked 2026-09-25)

| Source | Standard / open models | Reasoning / API models |
|---|---|---|
| OOLONG — reference eval (`src/eval/eval_script_batched.py`) + paper §4.2 | — | **Provider defaults**: no temperature / top_p / max_tokens set (a `max_completion_tokens=32768` line is commented out); optional `reasoning_effort`, left at default for all main results |
| RULER — `scripts/config_models.sh` | **Greedy**: `TEMPERATURE="0.0" # greedy`, `TOP_P="1.0"`, `TOP_K="32"` | — |
| HELMET | **Greedy** for all models | — |
| NoLiMa | **Greedy** for instruction-tuned models | defaults for o1 / o3-mini; R1 at temp 0.6, top_p 0.95; reasoning output capped at 1,536 tokens |
| LongBench v2 | temperature 0.1 | — |
| Recursive Language Models (arXiv 2512.24601) | Qwen3-Coder at its model-card recommended settings | GPT-5 "with medium reasoning … and default sampling parameters" |

Convention: **standard models greedy (temp 0); reasoning / API models at provider defaults.** Implications:
- gpt-5.4 (reasoning model) was run at temp 0; to match OOLONG and RLM it should be rerun at provider defaults.
- Fable 5.1 / gpt-6-*: provider defaults (Fable forbids temperature anyway).
- Base Qwen (thinking disabled, instruct): greedy — the temp-0 runs above follow this.
- Our fine-tunes have been evaluated at 0.2 (near-greedy; LongBench v2 uses 0.1). State it, or rerun headline rows
  at 0 to match base Qwen exactly.

Sources: [OOLONG repo](https://github.com/abertsch72/oolong) · [OOLONG paper](https://arxiv.org/abs/2511.02817) ·
[RULER config_models.sh](https://github.com/NVIDIA/RULER/blob/main/scripts/config_models.sh) ·
[HELMET](https://arxiv.org/abs/2410.02694) · [NoLiMa](https://arxiv.org/abs/2502.05167) ·
[LongBench v2](https://arxiv.org/abs/2412.15204) · [Recursive Language Models](https://arxiv.org/abs/2512.24601)

## Candidate models (checked 2026-09-25)

**OpenAI** (API model list + [pricing page](https://developers.openai.com/api/docs/pricing), standard tier, $/1M tokens):

| Model | Released | Input | Output | Context |
|---|---|---|---|---|
| gpt-6-astra (top price tier) | 2026-08-27 | 10.00 | 50.00 | short & long |
| gpt-6-sol | 2026-09-14 | 2.00 | 10.00 | short & long |
| gpt-6-luna | 2026-09-14 | 0.10 | 0.50 | short & long |
| gpt-5.6-sol / terra / luna | 2026-06-23 | 4.00/20.00 · 2.00/12.00 · 0.20/1.20 | | short & long |
| gpt-5.5 (`gpt-5.5-2026-04-23`) | 2026-04-23 | 5.00 | 30.00 | short only (<272K) |
| gpt-5.4 (`gpt-5.4-2026-03-05`) | 2026-03-05 | 2.50 | 15.00 | short only (<272K) |

**Anthropic.** Claude Fable 5.1 (`claude-fable-5-1`): 1M context, 128K max output, $10 / $50. No dated snapshot ID (the
served model is recorded instead). **Rejects `temperature`** (400 "temperature is deprecated for this model", even via
LiteLLM drop_params) — run with `TEMP=none`; thinking is always on, depth via effort (default `high`). May return
`stop_reason: refusal`; for evals we do NOT enable server-side fallbacks (that would mix models) — a refusal is scored
as no answer. Cheaper alternative: Claude Opus 5.5 (`claude-opus-5-5`), $4 / $20.

**Cost estimates** for the chart (30 problems × 10/20/40/80K ≈ 4.5M input tokens; reasoning output assumed 3–8K per
call): $10/$50 models (Fable 5.1, gpt-6-astra) ~$65–110 for all four lengths, ~$50–75 for 40K+80K only; gpt-6-sol
~$15–20 all four; Opus 5.5 ~$25–45 all four.

## Log

- 2026-09-25: gpt-5.4 chart 10/20/40/80K reproduced (above). Fable 5.1 10K started and interrupted before any result
  (cost check); nothing recorded.
- 2026-09-25: base Qwen single-shot 20K at temp 0 / 0.2, with a 16K output cap and with a 64K total context (above).
  `eval/run.py` gained `SINGLE_CONTEXT` (total-context limit for single-shot) and `TEMP=none`.
- 2026-09-25: base Qwen single-shot 40K at temp 0 / 0.2, 64K total context (above).
- 2026-09-25: gpt-5.4 found to run with NO reasoning by default; rerun at reasoning_effort=medium @40K = 0.600,
  @80K = 0.500 (18w 0.561). Plan: drop the output cap (use each model's max) for future runs.
- 2026-09-25: base Qwen t0 / 64K context @10K = 0.536 — base-Qwen chart row complete (80K cannot run on Tinker).
- 2026-09-25: gpt-5.4 medium reasoning @10K = 0.708, @20K = 0.662 — row complete.
- 2026-09-25: gpt-5.4 medium reasoning @160K = 0.556 (128K output cap; max used 50K).
- 2026-09-25: gpt-5.4 medium reasoning @320K, uncapped = 0.479. `OUT_TOKENS=0` = no output cap sent.
- 2026-09-25: base Qwen single-shot RULER-13 @10/20/40K = 0.923 / 0.938 / 0.938 (identical problems to eval_long.sh).
- 2026-09-25: gpt-5.4 medium reasoning @640K, uncapped = 0.419 (rerun at CONCURRENCY=2 after a 429 crash).
  `eval/run.py` `CONCURRENCY` env; `APIBackend` retries 429s with backoff.
  `eval/run.py` gained `REASONING_EFFORT`; API backend output ceiling now follows `OUT_TOKENS`; rollouts record per-call
  `usage` (output + reasoning tokens).

## Next

- Decide the next models (gpt-6-sol all lengths; one $10/$50 model at 40K+80K), then: longer contexts (160K+), base
  Qwen single-shot (262K context), same-harness competitor runs (MODE=decompose).
- Consider 2 samples per problem for frontier models (run-to-run variance at temp 0 is large on exact counts).
