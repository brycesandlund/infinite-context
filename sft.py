"""SFT warm-start from scripted-oracle delegation traces.

Base Qwen sits in the 0-reward regime on this harness (it over-reads into
context overflow, or spawns chaotic subagent swarms), so RL-from-scratch has no
reward variance to learn from. This script teaches the *skeleton* of good
delegation first, giving RL a non-zero base to improve on.

Pipeline:
1. Generate gold traces with `OracleBackend` through the exact `run_agent` loop
   (pure CPU — the oracle plays scripted-optimal moves). Each trace is a tree of
   agents: a root that splits the doc into ranges + spawns a subagent per range,
   and subagents that each read once and report their range's findings.
2. Convert every agent's conversation to per-assistant-turn cross-entropy Datums
   (the Qwen3 renderer strips thinking from history, so we build one example per
   assistant turn with TrainOnWhat.LAST_ASSISTANT_MESSAGE rather than training
   multiple assistant messages in one example).
3. SFT: forward_backward(loss_fn="cross_entropy") + optim_step over epochs.
4. Save a checkpoint to warm-start RL from (set train.py's LOAD_CHECKPOINT_PATH
   to it, with RESUME_OPTIMIZER=False).

Run: uv run python sft.py
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import tinker
import torch
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook.supervised import datum_from_model_input_weights

import metrics  # optional W&B logging (no-op unless WANDB=1)
import train  # shared constants + cookbook tool specs
from eval.agent import flatten, run_agent
from eval.backends import make_oracle, neutral_to_cookbook
from eval.run import _rollout_header, _tree_to_text  # shared rollout renderer
from tasks import grade_answer, list_tasks, load_pg_essays_text, make_problem, resolve_eval_grading_mode


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# OOLONG-ONLY warm-start. We isolate the hard aggregation tasks: with RULER out
# of the SFT pool there is no "read-a-wide-6000-chunk" precedent to contaminate
# the policy, so the OOLONG token-width traces teach ONE consistent rule (spawn
# wide ranges; read only narrow <=LEAF_TOKENS leaves). If decomposition transfers
# here, RULER becomes a separate, later concern. Decoupled from train.TASK_MIXTURE
# on purpose (that's the RL mixture).
SFT_TASKS = ["oolong_counting", "oolong_user", "oolong_temporal"]
N_PER_TASK = int(os.environ.get("N_PER_TASK", "30"))  # generous coverage (~3/dataset x 10)
N_PER_TASK_OVERRIDE: dict[str, int] = {}
DATA_SEED = 500_000             # distinct from train/eval seed ranges

EPOCHS = 2                      # NLL saturates by epoch 1 (~0.04) on the scripted
                                # oracle traces; a 3rd epoch just memorizes phrasings
                                # RL would undo. 2 captures the pattern, not the quirks.
SFT_BATCH_SIZE = 16             # datums per optim step
LEARNING_RATE = 1e-5

SAVE_CHECKPOINT_NAME = "sft_oolong"   # OOLONG-only warm-start (distinct from combined)
LAST_SFT_CHECKPOINT_FILE = Path.home() / ".cache" / "infinite-context" / "last_sft_checkpoint.txt"

# Debug toggles (env-overridable):
#   TRAIN=0          -> generate (+optionally print) oracle traces, but SKIP the
#                       Tinker training/save. Pure-CPU dry run to inspect the data.
#   PRINT_TRACES=1   -> print every oracle trace we build SFT datums FROM, using
#                       the eval tree renderer (full text, read_chunk clipped),
#                       and save them to $TRACE_OUT.txt. This is the ground-truth
#                       behaviour we're teaching — check it's actually clean.
TRAIN = os.environ.get("TRAIN", "1") == "1"
PRINT_TRACES = os.environ.get("PRINT_TRACES", "0") == "1"
TRACE_OUT = os.environ.get("TRACE_OUT", "/tmp/sft_traces")

# Shared with training / eval (single source of truth).
MODEL_NAME = train.MODEL_NAME
RENDERER_NAME = train.RENDERER_NAME
LORA_RANK = train.LORA_RANK
AGENT_CONTEXT = train.AGENT_CONTEXT
MAX_CHUNK_TOKENS = train.MAX_CHUNK_TOKENS
DOC_SIZE_TOKENS = train.DOC_SIZE_TOKENS
MAX_DEPTH = train.MAX_DEPTH
MAX_TURNS = train.MAX_TURNS


# ---------------------------------------------------------------------------
# Trace generation (CPU — no sampling client needed; the oracle is scripted)
# ---------------------------------------------------------------------------


async def _gen_traces(corpus_tokens, tokenizer):
    coros, meta = [], []
    for ti, task in enumerate(SFT_TASKS):
        for i in range(N_PER_TASK_OVERRIDE.get(task, N_PER_TASK)):
            seed = DATA_SEED + ti * 1000 + i
            problem = make_problem(task, corpus_tokens, tokenizer, DOC_SIZE_TOKENS, seed)
            oracle = make_oracle(
                problem, tokenizer,
                budget=AGENT_CONTEXT, max_chunk_tokens=MAX_CHUNK_TOKENS,
            )
            coros.append(
                run_agent(
                    oracle,
                    document_tokens=problem.document_tokens,
                    tokenizer=tokenizer,
                    task_context=problem.task_context,
                    question=problem.question,
                    budget=AGENT_CONTEXT,
                    max_chunk_tokens=MAX_CHUNK_TOKENS,
                    max_depth=MAX_DEPTH,
                    max_turns=MAX_TURNS,
                )
            )
            meta.append((task, problem))
    nodes = await asyncio.gather(*coros)
    return list(zip(meta, nodes))


# ---------------------------------------------------------------------------
# Trace -> SFT Datums
# ---------------------------------------------------------------------------


def _node_to_datums(node, renderer, tool_specs) -> list[tinker.Datum]:
    """One cross-entropy Datum per assistant turn in this agent's conversation.

    Per-turn (LAST_ASSISTANT_MESSAGE) rather than ALL_ASSISTANT_MESSAGES because
    the Qwen3 renderer lacks the extension property (it strips thinking from
    history), so each assistant turn must be rendered with its own real prefix.
    """
    cb = neutral_to_cookbook(node.messages, renderer, tool_specs)
    datums: list[tinker.Datum] = []
    for i, m in enumerate(cb):
        if m.get("role") != "assistant":
            continue
        model_input, weights = renderer.build_supervised_example(
            cb[: i + 1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        if float(weights.sum()) == 0.0:  # nothing trainable (shouldn't happen)
            continue
        datums.append(
            datum_from_model_input_weights(
                model_input, weights, max_length=AGENT_CONTEXT, reduction="mean"
            )
        )
    return datums


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    unknown = [t for t in SFT_TASKS if t not in list_tasks()]
    if unknown:
        raise SystemExit(f"Unknown SFT_TASKS: {unknown}. Available: {list_tasks()}")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    tool_specs = [
        train.ReadChunkTool.read_chunk.to_spec(),
        train.SubagentTool.spawn_subagent.to_spec(),
    ]

    print(f"SFT warm-start | tasks={len(SFT_TASKS)} x {N_PER_TASK} | "
          f"doc={DOC_SIZE_TOKENS} budget={AGENT_CONTEXT} epochs={EPOCHS}")
    print("Loading + tokenizing PG-essay corpus...")
    corpus_tokens = tokenizer.encode(load_pg_essays_text(), add_special_tokens=False)

    print("Generating oracle traces (scripted, CPU)...")
    traces = await _gen_traces(corpus_tokens, tokenizer)

    # Optionally dump every oracle trace we train FROM — the ground-truth
    # behaviour being taught. Uses the eval tree renderer (full text, read_chunk
    # clipped) and grades the oracle's answer as a sanity check that it solved.
    if PRINT_TRACES:
        with open(f"{TRACE_OUT}.txt", "w") as tf:
            for (task, problem), node in traces:
                sc = grade_answer(node.answer, problem.gold_answers, resolve_eval_grading_mode(problem))
                ds = problem.metadata.get("dataset")
                qt = problem.metadata.get("task_type")
                hdr = _rollout_header(task, "-", ds, qt, problem.question,
                                      problem.gold_answers, node.answer, node.termination, sc)
                tf.write(hdr)
                tf.write(_tree_to_text(node))
        print(f"Printed {len(traces)} oracle traces -> {TRACE_OUT}.txt")

    # Convert to datums; also sanity-check oracle traces actually solved the task.
    datums: list[tinker.Datum] = []
    n_agents = 0
    for (_task, _problem), node in traces:
        for agent in flatten(node):
            n_agents += 1
            datums.extend(_node_to_datums(agent, renderer, tool_specs))
    print(f"Traces: {len(traces)} | agents (root+subagents): {n_agents} | datums: {len(datums)}")
    if not datums:
        raise SystemExit("No datums produced — check oracle trace generation.")

    if not TRAIN:
        print("TRAIN=0: skipping Tinker training/save (dry run). "
              f"{'Traces at ' + TRACE_OUT + '.txt' if PRINT_TRACES else 'Set PRINT_TRACES=1 to inspect traces.'}")
        return

    # Training client.
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL_NAME, rank=LORA_RANK
    )
    adam_params = tinker.AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95)

    metrics.init(
        project="infinite-context",
        config={
            "phase": "sft", "model": MODEL_NAME, "renderer": RENDERER_NAME,
            "lora_rank": LORA_RANK, "lr": LEARNING_RATE, "epochs": EPOCHS,
            "sft_batch_size": SFT_BATCH_SIZE, "n_datums": len(datums),
            "tasks": SFT_TASKS, "n_per_task": N_PER_TASK,
            "n_per_task_override": N_PER_TASK_OVERRIDE,
        },
    )

    rng = random.Random(0)
    global_batch = 0
    for epoch in range(EPOCHS):
        rng.shuffle(datums)
        n_batches = (len(datums) + SFT_BATCH_SIZE - 1) // SFT_BATCH_SIZE
        epoch_nll = 0.0
        n_logged = 0
        for b in range(n_batches):
            batch = datums[b * SFT_BATCH_SIZE : (b + 1) * SFT_BATCH_SIZE]
            fwd_bwd = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
            optim = await training_client.optim_step_async(adam_params)
            fb_result = await fwd_bwd.result_async()
            await optim.result_async()
            # weighted-mean NLL for logging (best-effort; never block training on it)
            try:
                lp = [o["logprobs"] for o in fb_result.loss_fn_outputs]
                w = [d.loss_fn_inputs["weights"] for d in batch]
                num = sum(float(l.to_torch().dot(wi.to_torch())) for l, wi in zip(lp, w))
                den = sum(float(wi.to_torch().sum()) for wi in w)
                if den:
                    batch_nll = -(num / den)
                    epoch_nll += batch_nll
                    n_logged += 1
                    metrics.log({"sft/batch_nll": batch_nll, "sft/epoch": epoch}, step=global_batch)
            except Exception:
                pass
            global_batch += 1
        nll_str = f"{epoch_nll / n_logged:.4f}" if n_logged else "n/a"
        print(f"Epoch {epoch}: batches {n_batches} | mean batch NLL {nll_str}")
        if n_logged:
            metrics.log({"sft/epoch_nll": epoch_nll / n_logged}, step=global_batch)

        # Checkpoint after EVERY epoch (overwrite). Tinker runs are slow (~15
        # min/epoch) and occasionally drop connection, so a kill/crash should
        # never cost a full redo — last_sft_checkpoint.txt always points at the
        # latest completed epoch's weights, which are usable for RL warm-start.
        print(f"  saving checkpoint '{SAVE_CHECKPOINT_NAME}' (after epoch {epoch})...")
        save_future = await training_client.save_state_async(SAVE_CHECKPOINT_NAME, overwrite=True)
        save_resp = await save_future.result_async()
        LAST_SFT_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_SFT_CHECKPOINT_FILE.write_text(save_resp.path)
        print(f"  saved: {save_resp.path}  (path -> {LAST_SFT_CHECKPOINT_FILE})")
    print("\nTo warm-start RL: set train.py LOAD_CHECKPOINT_PATH to the path above "
          "and RESUME_OPTIMIZER=False.")
    metrics.finish()


if __name__ == "__main__":
    asyncio.run(main())
