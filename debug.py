"""Debug helpers for the recursive-agent RL script.

Kept separate so the main training loop stays uncluttered. Functions here
accept rolled-out `ParentRollout` / `RolloutNode` instances and inspect them —
they never construct policy state themselves.

Tree PRINTING lives in eval/render.py (single source of truth shared with the
eval driver and SFT traces); this module only inspects datum/trajectory shapes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import tinker
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.data_processing import trajectory_to_data

from eval.render import rollout_to_agent_node, tree_to_text

if TYPE_CHECKING:
    from tasks import Problem
    from train import ParentRollout


def _summarize_datum(datum: tinker.Datum) -> dict[str, int]:
    inputs = datum.loss_fn_inputs
    out: dict[str, int] = {"input_length": datum.model_input.length}
    for k in ("target_tokens", "logprobs", "advantages", "mask"):
        if k in inputs:
            shape = inputs[k].shape if hasattr(inputs[k], "shape") else None
            out[k] = shape[-1] if shape else -1
    return out


async def debug_run_rollouts_niah(
    n_rollouts: int,
    sampling_client: tinker.SamplingClient,
    tokenizer,
    renderer: Renderer,
    problems: list["Problem"],
    rollout_fn: Callable[..., Awaitable["ParentRollout"]],
) -> None:
    """Run `n_rollouts` rollouts concurrently. Print every tree (via the shared
    eval/render.py printer) + datum shapes. No training."""
    n = min(n_rollouts, len(problems))
    print("=" * 72)
    print(f"DEBUG: {n} rollouts, concurrent")
    print("=" * 72)
    coros = [
        rollout_fn(problems[i], sampling_client, tokenizer, renderer) for i in range(n)
    ]
    results = await asyncio.gather(*coros)

    fake_adv = 1.0
    n_with_children = 0
    for ri, (problem, result) in enumerate(zip(problems[:n], results)):
        nodes = result.all_nodes()
        max_depth = max(node.depth for node in nodes)
        if max_depth > 0:
            n_with_children += 1
        from train import compute_reward  # post-hoc single-source training reward

        parent_reward = compute_reward(result)
        print()
        print("-" * 72)
        needle_positions = problem.metadata.get("needle_positions", [])
        print(
            f"Rollout {ri}  (task={result.task}, gold={result.gold_answers}, "
            f"needle_pos={needle_positions}/{len(problem.document_tokens)}, "
            f"nodes={len(nodes)}, max_depth={max_depth}, "
            f"parent_reward={parent_reward:.3f})"
        )
        q_short = (
            problem.question if len(problem.question) <= 120
            else problem.question[:117] + "..."
        )
        print(f"Question: {q_short}")
        print("Tree:")
        print(tree_to_text(rollout_to_agent_node(result.root, tokenizer, renderer)))

        total_datums = 0
        for node in nodes:
            if not node.trajectory.transitions:
                print(f"  [depth={node.depth}] empty trajectory — would be skipped")
                continue
            datums = trajectory_to_data(node.trajectory, fake_adv)
            for di, datum in enumerate(datums):
                summary = _summarize_datum(datum)
                print(f"  datum (depth={node.depth}, #{di}): {summary}")
                total_datums += 1
        print(f"Datums for this rollout: {total_datums}")

    print()
    print("=" * 72)
    print(
        f"Summary: {n_with_children}/{n} rollouts spawned at least one subagent. "
        f"Max depth observed across all rollouts = "
        f"{max(max(node.depth for node in r.all_nodes()) for r in results)}"
    )
