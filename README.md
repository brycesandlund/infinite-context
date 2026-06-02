# infinite-context

Training a small-context language model to solve long-context problems by
**calling itself agentically** — reading a document it can't fit in its window
through a `read_chunk` tool, and delegating sub-ranges to fresh-context copies
of itself via `spawn_subagent`. Reward propagates from the root agent down to
every subagent in the tree, so the whole recursion is trained end-to-end with
GRPO on [Tinker](https://tinker.thinkingmachines.ai/).

The benchmark is [RULER](https://github.com/NVIDIA/RULER) (NVIDIA's long-context
suite), vendored faithfully, with one twist: the model never receives the
document in its prompt — it must reach it through tool calls within a fixed
~10k-token budget.

## The idea

A model with a 10k context cannot read a 15k–∞ token document. Two ways out:
1. **Bigger context** (what frontier models do) — but there's always a document
   bigger than the window.
2. **Decompose**: read a chunk, or spawn a subagent to read a range and report
   back, then aggregate. This is what we train.

Every agent in the tree (root + all subagents) shares:
- the same policy weights,
- the same fixed per-agent context budget (`AGENT_CONTEXT`, default 10k),
- the same two tools (`read_chunk`, `spawn_subagent`),
- one read-only view of the document (token positions are a shared coordinate
  system; each agent has its own budget).

Because every tool result stays in an agent's context, a single agent *cannot*
scan a document larger than its budget — it must delegate. Learning to delegate
well (instead of over-reading into context overflow) is the capability being
trained.

## Project structure

```
harness.py            Shared agent primitives (single source of truth)
train.py              RL training loop (token-level, GRPO, Tinker)
debug.py              Rollout-tree inspection / verbose printing helpers
tasks/                Problem generation (the RULER suite)
  base.py             Problem schema + graders (exact/set/numeric/ruler_*)
  corpus.py           Paul Graham essays + noise-sentence haystacks
  registry.py         Task dispatch + train/eval grader-mode tables
  ruler/              Vendored NVIDIA/RULER generators (Apache-2.0)
    constants.py      RULER templates + the 13-task config (synthetic.yaml)
    generators.py     NIAH / VT / CWE / FWE generators -> Problem
    _common.py        wonderwords vocab, nltk insertion, {context} adapter
eval/                 Multi-backend eval (Qwen / Claude / GPT)
  backends.py         ModelBackend ABC + TinkerBackend + APIBackend (LiteLLM)
  agent.py            Backend-agnostic recursive agent loop (run_agent)
  run.py              Eval entry point: backend x tasks x N -> RULER scores
```

### `harness.py` — the shared seam

Holds the parts that **must be identical** between training and eval so a
Qwen-vs-Claude-vs-GPT comparison isn't confounded by harness differences:
`make_system_prompt`, `read_chunk_impl` (token-slice/decode/cap semantics),
`extract_boxed`, and the canonical tool descriptions / `openai_tool_specs()`.

`train.py` asserts at import time that its cookbook `@tool` docstrings equal the
harness constants, so the two can't silently drift.

### `train.py` — RL training (token-level)

The battle-tested path. Uses the Tinker cookbook's `do_single_rollout` +
`build_agent_tool_env`, which operate at the **token** level because GRPO needs
per-token logprobs (`Transition.ac` = `TokensWithLogprobs`) to build importance-
sampling `Datum`s.

- `ReadChunkTool` / `SubagentTool` — the two cookbook tools. `spawn_subagent`
  recursively rolls out the same policy on a fresh env (depth-capped).
- `LongContextReward` — `format_coef * (format - 1) + grade(...)` on the root's
  `\boxed{}` answer.
- GRPO: `BATCH_SIZE` problems x `GROUP_SIZE` rollouts; advantage =
  reward − per-problem group mean; **every node in a parent's tree inherits the
  parent's advantage** (reward propagation), then `trajectory_to_data` per node.
- `TASK_MIXTURE` — per-step weighted sampling over the 11 trainable RULER tasks.
- Post-training verbose eval block (`EVAL_TASKS`, `EVAL_N_ROLLOUTS`) reporting
  both the strict (training) and RULER-official (eval) graders.

Run: `uv run python train.py` (config is the constant block at the top —
`MODEL_NAME`, `AGENT_CONTEXT`, `MAX_DEPTH`, `TASK_MIXTURE`, checkpoint paths, …).

### `tasks/` — the RULER suite

`make_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem`
where a `Problem` is `(document_tokens, question, gold_answers, task,
task_context, metadata)`. `document_tokens` is the haystack (reached only via
`read_chunk`); `question` is the short user message carrying RULER's instruction.

Generators are vendored from NVIDIA/RULER's `main` branch (the source the
NeMo-Skills "rulerv1-ns" pipeline itself invokes), adapted in exactly two ways:
1. the haystack is served via `read_chunk` instead of inlined in `{context}`;
2. the model answers in `\boxed{}` instead of completing RULER's answer prefix.

`task_context` is empty for RULER tasks — the root agent gets only RULER's
instruction and must infer the needle format and propagate it to subagents,
preserving RULER fidelity.

11 trainable tasks: `niah_single_{1,2,3}`, `niah_multikey_{1,2,3}`,
`niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`. (`qa_1`/`qa_2` are
registered but deferred — they need SQuAD/HotpotQA downloads and are intended as
held-out eval.)

Graders (`tasks/base.py`):
- **training reward** (`grading_mode`): strict `exact` / `set` / `numeric`
  (`0.75**|y-ŷ|` partial credit) — a clean gradient signal.
- **eval scoring** (`eval_grading_mode`): RULER's official `string_match_all` /
  `string_match_part` (lowercased substring) — numbers comparable to NVIDIA's
  leaderboard.

### `eval/` — multi-backend eval (message-level)

A **separate** rollout path from training, so any chat model can play the same
game. Training stays token-level (it needs logprobs); eval is message-level (the
API providers don't expose token-level sampling).

- `ModelBackend`: `count_tokens(messages)` + `async sample(messages, max_tokens)
  -> AssistantTurn`. The message-level seam.
- `TinkerBackend`: drives the Qwen policy through the **same renderer + the exact
  tool specs train.py uses**, so eval rollouts are faithful to training rollouts.
- `APIBackend`: one class over any LiteLLM model (Anthropic, OpenAI, …) by model
  string; counts tokens in each model's own tokenizer.
- `run_agent` (`eval/agent.py`): the single backend-agnostic recursive loop —
  budget enforcement, `read_chunk`, `spawn_subagent` recursion, `\boxed`
  extraction — used by **every** backend, so cross-model comparison is
  equivalent by construction.

Run:
```bash
uv run python -m eval.run                                   # Qwen via Tinker
BACKEND=anthropic/claude-sonnet-4-20250514 uv run python -m eval.run
BACKEND=openai/gpt-5-mini                   uv run python -m eval.run
```
Budget/recursion/data constants are imported from `train.py` so eval and
training can't diverge on them.

## Setup

```bash
uv sync
export TINKER_API_KEY=...        # for training + the Tinker eval backend
export ANTHROPIC_API_KEY=...     # for BACKEND=anthropic/...
export OPENAI_API_KEY=...        # for BACKEND=openai/...
```

Dependencies of note: `tinker` + `tinker-cookbook` (RL + rollouts), `wonderwords`
+ `nltk` (RULER's exact vocab + sentence-boundary insertion), `litellm`
(multi-provider eval).

## Status / findings

- Full RULER suite (11 synthetic tasks) generating and training-ready.
- Multi-backend eval validated: the Tinker backend through `run_agent` reproduces
  the training rollout's behavior; recursion + tool dispatch confirmed.
- Early head-to-head on the recursive harness @ 10k budget (4 tasks, n=3, *not*
  seed-paired): base **Qwen3.5-9B ≈ 0.00**, **Claude Sonnet 4 ≈ 0.42** (RULER
  substring). Claude delegates cleanly but only ~half the time; base Qwen
  delegates chaotically or over-reads into context overflow. The gap — and
  Claude's unreliability — is what training a budget-aware recursive policy aims
  to close.

## Roadmap

- Training run on the RULER mixture (does a trained small model close/beat the
  Claude gap on this harness?).
- OOLONG-synth aggregation tasks (counting / user / temporal) — the
  decomposition thesis's natural home.
- Held-out eval bucket: RULER QA (SQuAD/HotpotQA) + OOLONG-real.
