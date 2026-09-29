# Trained Agentic Context Management

Code for the paper **Trained Agentic Context Management** (Bryce Sandlund, 2026; arXiv link forthcoming).

Instead of training long context natively or engineering a long-context harness, we fine-tune a model to use
the simplest possible harness — two tools:

- `read_chunk(start, end)`: read tokens `[start, end)` of a document the model never sees in its prompt;
- `spawn_subagent(subtask)`: delegate `subtask` to a fresh-context copy of itself, which returns its `\boxed{}` answer.

Every agent has a small fixed context budget (8K tokens at evaluation), so a long document can only be answered by
decomposing it. We supervise fine-tune **Qwen3.6-35B-A3B** (LoRA rank 32, on [Tinker](https://thinkingmachines.ai/tinker/))
on synthetic demonstrations from scripted oracles that teach three strategies: **tree reduce** for
order-independent questions, **sequential fold** for order-dependent ones, and reading past a range boundary to
finish a record. Nothing about the decomposition is hard-coded in the harness. All code used for the project is contained
in this repo and most artifacts are also present, except large trace files.

## Results

Fine-tuned Qwen3.6-35B-A3B in the harness (8K tokens per agent) vs GPT-5.4 (medium reasoning) and base
Qwen3.6-35B-A3B given the whole document in one call. Training documents are at most 14K tokens.

| OOLONG-synth (30 problems / length) | 10K | 20K | 40K | 80K | 160K | 320K |
|---|---|---|---|---|---|---|
| **Ours** (8K / agent) | 0.606 | 0.540 | 0.535 | 0.561 | 0.464 | 0.470 |
| GPT-5.4 (full document) | 0.718 | 0.666 | 0.600 | 0.539 | 0.556 | 0.479 |
| Qwen3.6-35B-A3B (full document, 64K window) | 0.536 | 0.558 | 0.388 | — | — | — |
| Random baseline (OOLONG §2.3) | 0.226 | 0.225 | 0.184 | 0.181 | 0.191 | 0.161 |

| RULER-13 (65 problems / length) | 10K | 20K | 40K | 80K | 160K | 320K |
|---|---|---|---|---|---|---|
| **Ours** (8K / agent) | 0.949 | 0.949 | 0.929 | 0.885 | 0.868 | 0.858 |
| GPT-5.4 (full document) | 0.954 | 0.938 | 0.954 | 0.954 | 0.929 | 0.954 |
| Qwen3.6-35B-A3B (full document, 64K window) | 0.923 | 0.938 | 0.938 | — | — | — |

Per-family breakdowns, protocols and every run's history are in [`training_runs.md`](training_runs.md) (fine-tunes)
and [`competitor_evals.md`](competitor_evals.md) (GPT-5.4, base Qwen). Raw rollouts are in `eval_results/`.

## Setup

```bash
uv sync
export TINKER_API_KEY=...        # training and the Tinker eval backend
export OPENAI_API_KEY=...        # GPT baselines (LiteLLM)
export ANTHROPIC_API_KEY=...     # Claude baselines, and the model-executed leaf for narrativeqa traces
```

`pyproject.toml` pins `tinker` to a local editable checkout (`[tool.uv.sources]`); remove that entry to install the
published `tinker` package instead.

## Training

Training traces are generated on CPU by scripted oracles (`oracle/`) over procedurally generated tasks (`tasks/`),
then used for supervised fine-tuning with `sft.py` (Tinker). One run script per training run lives in `scripts/`:

```bash
# from base (the first stage of the reported model)
nohup bash scripts/run_sft13_base.sh > /tmp/sft13.log 2>&1 &
# warm-start stages take the previous checkpoint
INIT_CHECKPOINT=tinker://... nohup bash scripts/run_sft18_warm.sh > /tmp/sft18.log 2>&1 &
# dry run: build and print traces without training
INIT_CHECKPOINT=x TRAIN=0 PRINT_TRACES=1 bash scripts/run_sft18_warm.sh
```

The reported model (`sft_general18w`) is run 13 (from base), followed by warm-start runs 14w and 18w. Key knobs in
`sft.py`: `SFT_TASKS`, `N_PER_TASK` / `N_PER_TASK_OVERRIDE`, `BUDGET_MIX` (per-agent budgets and document lengths),
`LR`, `SFT_BATCH_SIZE`, `ROOT_DUP`, `CONTEXT_DROP`.

## Evaluation

One entry point, `eval/run.py`, drives any backend over identical problems and graders, in two modes:
`MODE=decompose` (the harness) and `MODE=single` (the whole document in one tool-free call).

```bash
# fine-tune, harness, RULER-13 + OOLONG-synth at a given document length and per-agent budget
scripts/eval_long.sh <tag> <tinker://checkpoint> <doc_tokens> <budget> [ruler_n] [oolong_n]
# fine-tune, harness, the OOLONG-synth problems plotted in the paper (8K budget)
scripts/eval_oolong_chart.sh <tag> <tinker://checkpoint> <doc_tokens>
# single-shot baselines on the same problems
REASONING_EFFORT=medium TEMP=none OUT_TOKENS=0 scripts/competitor_oolong_chart.sh openai/gpt-5.4-2026-03-05 gpt5_4 10000 20000 40000
REASONING_EFFORT=medium TEMP=none OUT_TOKENS=0 scripts/competitor_ruler.sh openai/gpt-5.4-2026-03-05 gpt5_4 10000 40000
```

Common environment knobs: `BACKEND`, `MODE`, `CKPT`, `DOC_SIZE_TOKENS`, `AGENT_CONTEXT`, `EVAL_TASKS`, `N_PER_TASK`,
`TEMP`, `MAX_NODES`. API runs can be expensive at long lengths; `competitor_evals.md` records the cost of each.

## Repository layout

```
harness.py         system prompt, tool definitions, read_chunk semantics, \boxed{} extraction (shared by train and eval)
sft.py             SFT: builds oracle traces, trains on Tinker
rl.py              recursive-agent RL experiment (GRPO); not used for the reported model
oracle/            scripted oracles that play tree reduce / sequential fold / boundary reads
tasks/             task generators: synthetic records, labelled text, prose haystacks, word lists, RULER, OOLONG
eval/              backends (Tinker, LiteLLM), agent loops (decompose / single), eval runner, rendering
scripts/           run scripts for every training run and eval
paper/             paper source (LaTeX) and literature notes
eval_results/      raw rollouts and analysis
trace_snippets/    example training traces
```

## Third-party code and data

- `tasks/ruler/` vendors task generators from [NVIDIA RULER](https://github.com/NVIDIA/RULER) (Apache-2.0); see
  `tasks/ruler/LICENSE` and `tasks/ruler/NOTICE` for the license and our modifications.
- `tasks/oolong/vendored_synth/` and `tasks/oolong_real/` use code from [OOLONG](https://github.com/abertsch72/oolong)
  (MIT); see the `LICENSE` files in those directories.
- Datasets are downloaded at run time and keep their own licenses (Project Gutenberg texts, Paul Graham essays,
  NarrativeQA, SQuAD, HotpotQA, Yelp polarity, DBpedia-14, dair-ai/emotion, SNLI, PAWS, Stanford politeness, and the
  OOLONG datasets). Eval logs and trace snippets in this repository contain excerpts of these datasets; the MIT
  license below does not apply to that third-party content.

## License

MIT (see [`LICENSE`](LICENSE)), except for third-party code and data as noted above.

## Citation

```bibtex
@misc{sandlund2026trained,
  title  = {Trained Agentic Context Management},
  author = {Sandlund, Bryce},
  year   = {2026},
  note   = {arXiv preprint, forthcoming}
}
```
