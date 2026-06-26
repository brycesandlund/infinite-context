# infinite-context

Teaching a **small-context** language model to solve **long-context** problems by
**calling itself recursively**. The model has a fixed ~8K-token budget and never
receives the document in its prompt — it reaches the text only through a
`read_chunk(start, end)` tool, and delegates sub-ranges to fresh-context copies
of itself via `spawn_subagent(subtask)`. A document larger than the budget
*cannot* be answered by one agent; it must be split, delegated, and the partial
results combined — a binary divide-and-conquer over the prompt.

Policy: **Qwen3.6-35B-A3B** (LoRA r32) on [Tinker](https://tinker.thinkingmachines.ai/).
Primary testbed: **OOLONG-synth** aggregation tasks (counting / user / temporal);
**RULER** (NVIDIA's long-context suite) is used for cross-task transfer. Both are
vendored faithfully.

## The idea

A model with an 8K context cannot read a 40K–∞ token document. Two ways out:
1. **Bigger context** (what frontier models do) — but there is always a document
   bigger than the window, and single-pass aggregation over hundreds of in-context
   items degrades even when the text *fits* (see results).
2. **Decompose**: read a small chunk, or spawn a subagent to read a range and
   report back, then aggregate the reports. This is what we train.

Every agent in the tree (root + all subagents) shares the same policy weights, the
same fixed per-agent budget (`AGENT_CONTEXT`, default 8K), the same two tools, and
one read-only view of the document (token positions are a shared coordinate system;
each agent has its own budget). Because every tool result stays in an agent's
context, a single agent cannot scan a document larger than its budget — it must
delegate. Learning to decompose *finely and correctly* (not over-read into context
overflow) is the capability being trained.

### Why some tasks scale and others hit a wall — combine-state complexity

The decisive variable is the **size of the state passed up the tree** (the
"combine"):

- **Bounded / associative combine** → scales gracefully. Counting reduces over a
  fixed ≤6-label count vector (O(1) in document length), so a parent summing its
  children's tallies is always ~30 tokens. Counting holds its score from 10K to 80K.
- **Irreducible O(distinct-keys) combine** → hits a working-memory wall. User and
  temporal carry per-user / per-month×label / per-date tallies whose key count
  *grows with the document*. Up the tree these unions exceed the agent budget, the
  node truncates mid-tally (no `\boxed{}` → overflow), and the overflow triggers a
  re-split/re-spawn cascade that bloats the tree. This is the *only* place our
  method bends, and it's diagnosable, not a method flaw (see results §1/§8).

The fix space for the irreducible case: bigger budget (pushes the wall out),
a non-growing/streaming combine (thresholded/top-k sketches), or a **sequential
left-fold** that threads one running tally instead of unioning pairwise.

## Training pipeline

```
OolongOracle  ──►  sft.py (SFT warm-start)  ──►  rl.py (GRPO RL)
scripted-optimal     the currently EVALUATED        token-level reward
delegation traces    checkpoint (sft_oolong)        propagation, next stage
```

1. **Scripted oracle** (`eval/backends.py :: OolongOracle`, `OracleBackend`): plays
   the canonical binary decomposition through `run_agent`, emitting clean
   "show-your-work" traces — a leaf reads a small range, enumerates each example
   with its classified label, and reports a compact tally; parents sum tallies; the
   root derives the answer. Subagents report *extractable evidence*, never
   gold-derived answers, so the skill is learnable and transfers.
2. **SFT warm-start** (`sft.py`): base Qwen sits in the 0-reward regime on this
   harness (handed the tools untrained, it never even calls `read_chunk`). SFT on
   the oracle traces teaches the *protocol*. The evaluated checkpoint is SFT-only.
3. **GRPO RL** (`rl.py`): token-level recursive-agent RL where **every node in a
   tree inherits the root's advantage** (reward propagation), with an LLM judge
   (`eval/judge.py`) scoring open-ended subagent subtasks. This is the infra for the
   next stage beyond SFT.

## Results (preliminary)

Full numbers, methodology, and raw files: **[`eval_results/RESULTS.md`](eval_results/RESULTS.md)**.
Headlines, all on held-out problems with OOLONG-official grading:

**Graceful degradation vs frontier collapse — the crossover.** OOLONG OVERALL across
document length (our SFT model @8K budget vs gpt-5.4 single-shot, native 1M window,
identical problems):

| OVERALL | 10K | 20K | 40K | 80K |
|---|---|---|---|---|
| **ours** (8K budget, decompose) | 0.532 | 0.514 | **0.562** | **0.429** |
| gpt-5.4 (1M, single-shot) | 0.561 | 0.583 | 0.338 | 0.327 |

Frontier is ahead at 10K–20K, then **collapses at 40K** (single-pass aggregation over
~600+ items fails); our decomposition stays flat and crosses above.

**SFT's entire value is the protocol** (OOLONG counting @10K): same base weights score
**0.000 with the harness** (never reads) → **0.519 after SFT**, which even beats
*gpt-5.4 operating the same harness* (0.376 — it decomposes shallowly, ~3 nodes, vs our
~40). But at 10K **single-shot wins overall** (base Qwen single-shot 0.681) — forced
decomposition is pure added error when the doc fits native context. The harness earns
its keep only past that point.

**The 80K dip is the working-memory wall, confirmed by a budget sweep.** Raising the
budget 8K → 12K at 80K recovers the irreducible-combine families and leaves counting
flat (temporal +0.08, OVERALL 0.429 → 0.473); the worst overflow cascade collapses from
**4293 nodes → 497** once a node can hold its merged tally. The combine is *pushed out,
not solved* — temporal still overflows at the widest tallies.

**RULER cross-task transfer is leaf-op-gated** (zero-shot, SFT was OOLONG-only): transfers
where the leaf-op matches training (`cwe` = counting → 1.0) and fails on untrained
leaf-ops (needle/track/QA). The scaffold is task-general; the leaf-op must be taught.

## Project structure

```
harness.py            Shared agent primitives (single source of truth)
sft.py                SFT warm-start from oracle traces (-> sft_oolong checkpoint)
rl.py                 Recursive-agent RL (token-level, GRPO, Tinker)
metrics.py            Optional W&B logging (no-op unless WANDB=1)
debug.py              Rollout-tree inspection helpers
calibrate_judge.py    One-off: calibrate the LLM judge vs gold on OOLONG counting
frontier_released.py  Frontier single-shot on the OFFICIAL released oolong-synth (paper sanity)

tasks/                Problem generation
  base.py             Problem schema + graders (exact/set/numeric/ruler_*)
  corpus.py           Paul Graham essays + noise-sentence haystacks
  registry.py         Task dispatch + train/eval grader-mode tables
  oolong/             OOLONG-synth (counting / user / temporal aggregation)
    generators.py     Thin wrapper over the vendored OOLONG generation code
    vendored_synth/   Vendored OOLONG task constructors
  ruler/              Vendored NVIDIA/RULER generators (Apache-2.0)
    constants.py      RULER templates + the 13-task config (synthetic.yaml)
    generators.py     NIAH / VT / CWE / FWE -> Problem
    qa_data.py        SQuAD / HotpotQA loaders (qa_1, qa_2)
    _common.py        wonderwords vocab, nltk insertion, {context} adapter

eval/                 Multi-backend eval (one path for Qwen / base / Claude / GPT)
  backends.py         ModelBackend ABC + TinkerBackend + APIBackend + the oracles
  agent.py            Backend-agnostic loops: run_agent (decompose) + run_single_shot
  run.py              Eval entry point: backend × mode × tasks × N -> scores
  judge.py            LLM-as-a-judge (scores open-ended subagent subtasks, for RL)
  render.py           Shared rollout-tree printing (eval + sft + train)

eval_results/         RESULTS.md (findings) + raw/ rollouts + plotting scripts
```

### `harness.py` — the shared seam

Holds the parts that **must be identical** between training, SFT, and eval so a
Qwen-vs-base-vs-GPT comparison isn't confounded: `make_system_prompt`,
`make_single_shot_prompt`, `read_chunk_impl` (token-slice/decode/cap semantics),
`extract_boxed`, and the canonical tool descriptions / `openai_tool_specs()`.
`rl.py` asserts at import that its cookbook `@tool` docstrings equal the harness
constants, so the two can't silently drift.

### `eval/` — one eval path, two modes

A single entry point (`eval/run.py`) drives any backend through the `ModelBackend`
seam, over identical problems and graders, in either of two modes:

- **`MODE=decompose`** (default): the recursive 8K-budget harness — the model must
  `read_chunk` / `spawn_subagent` (`run_agent`).
- **`MODE=single`**: the whole document is placed in context and the model answers in
  one tool-free call (`run_single_shot`) — the raw-ability ceiling (frontier
  single-shot, or an un-finetuned base model). Same problem construction, same grading.

`TinkerBackend` drives the Qwen policy through the **same renderer + exact tool specs
rl.py uses** (eval faithful to training); `APIBackend` covers any LiteLLM model.
Budget/recursion/data constants are imported from `rl.py` so eval and training
can't diverge.

## Setup & running

```bash
uv sync
export TINKER_API_KEY=...        # training + the Tinker eval backend
export OPENAI_API_KEY=...        # BACKEND=gpt-5.4 etc.
export ANTHROPIC_API_KEY=...     # BACKEND=anthropic/...

# Eval the SFT model through the harness on OOLONG @10K
CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) \
  EVAL_TASKS=oolong_counting,oolong_user,oolong_temporal uv run python -m eval.run

# Single-shot ceilings (same problems, no decomposition):
BACKEND=gpt-5.4 MODE=single EVAL_TASKS=oolong_counting uv run python -m eval.run   # frontier
BACKEND=tinker  MODE=single EVAL_TASKS=oolong_counting uv run python -m eval.run   # base Qwen (no CKPT)

# Key env knobs: BACKEND, MODE, CKPT, DOC_SIZE_TOKENS, AGENT_CONTEXT, EVAL_TASKS, N_PER_TASK, TEMP
```

Dependencies of note: `tinker` + `tinker-cookbook` (RL + rollouts), `wonderwords` +
`nltk` (RULER's exact vocab + sentence-boundary insertion), `litellm` (multi-provider
eval), `datasets` (released OOLONG sanity).

## Status / roadmap

- ✅ OOLONG-synth (3 families) + RULER (13 configs) generating; SFT warm-start trained
  and evaluated; multi-backend + single-shot eval unified in `eval/run.py`.
- ✅ Characterized the method: graceful-degradation-vs-collapse crossover (40K), the
  combine-state-complexity wall (80K), SFT-teaches-the-protocol, RULER leaf-op transfer.
- ⏭ **GRPO RL** on top of the SFT warm-start (`rl.py` infra ready).
- ⏭ A **non-growing / sequential-fold combine** to break the irreducible-combine wall
  on user/temporal at scale.
- ⏭ Teach additional leaf-ops (needle/track/QA) to broaden zero-shot RULER transfer.
