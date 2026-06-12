"""Shared rollout-tree renderer — THE one way trees are printed everywhere.

eval/run.py (eval rollouts), sft.py (oracle traces), and train.py (RL rollouts,
both the per-step dump and the post-run eval) all render through these helpers,
so a tree reads identically no matter which pipeline produced it. RL trajectories
are first converted to the same neutral AgentNode shape by
debug.rollout_to_agent_node().
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.agent import AgentNode


def print_tree(node: "AgentNode", indent: int = 0, full: bool = False) -> None:
    """Pretty-print a rollout tree. `full=True` disables ALL truncation (used for
    the OUT.txt dump so saved rollouts are complete); the truncating default keeps
    stdout readable."""
    def clip(s: str, n: int) -> str:
        return s if full or len(s) <= n else s[:n] + " …"

    prefix = "  " * indent
    bar = "=" * max(8, 76 - len(prefix))
    print(f"{prefix}{bar}")
    print(
        f"{prefix}[depth={node.depth}] turns={node.n_turns} "
        f"termination={node.termination} answer={node.answer!r}"
    )
    if node.subtask:
        print(f"{prefix}SUBTASK: {node.subtask}")
    print(f"{prefix}{bar}")
    for m in node.messages:
        role = m["role"]
        raw_content = m.get("content") or ""
        content = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip()
        if role == "assistant" and m.get("tool_calls"):
            calls = "; ".join(f"{tc.name}({tc.arguments})" for tc in m["tool_calls"])
            print(f"{prefix}[assistant] {clip(content, 500)}")
            print(f"{prefix}  -> CALLS: {calls}")
        elif role == "tool":
            # read_chunk returns raw haystack — ALWAYS abbreviate it (even in full
            # mode) so saved rollouts don't bloat with document text. Show the
            # HEAD *and* TAIL of the read so the chunk boundaries are visible (a
            # boundary cutting an example mid-way is a suspected failure mode).
            # Everything else (subagent reports, etc.) follows the full/clip setting.
            if m.get("name") == "read_chunk":
                if len(content) <= 500:
                    body = content
                else:
                    body = f"{content[:240]} …[{len(content)-480} chars clipped]… {content[-240:]}"
            else:
                body = clip(content, 300)
            print(f"{prefix}[tool:{m.get('name')}] {body}")
        else:
            print(f"{prefix}[{role}] {clip(content, 800)}")
    print()
    for c in node.children:
        print_tree(c, indent + 1, full=full)


def node_to_dict(node: "AgentNode") -> dict:
    """JSON-serializable view of a rollout tree (for OUT.jsonl)."""
    return {
        "depth": node.depth, "subtask": node.subtask, "answer": node.answer,
        "termination": node.termination, "n_turns": node.n_turns,
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "tool_calls": [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in (m.get("tool_calls") or [])
                ],
                "name": m.get("name"),
            }
            for m in node.messages
        ],
        "children": [node_to_dict(c) for c in node.children],
    }


def tree_to_text(node: "AgentNode") -> str:
    """Render a tree to a string for OUT.txt — UNtruncated (full=True)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_tree(node, full=True)
    return buf.getvalue()


def rollout_header(task, seed, dataset, qtype, question, gold, answer, term, score) -> str:
    """The ##### banner for one rollout in OUT.txt. The full question is included
    (whitespace-collapsed onto one line, NEVER truncated)."""
    q = " ".join((question or "").split())
    bar = "#" * 90
    return (f"\n{bar}\n# task={task} seed={seed} dataset={dataset} qtype={qtype} "
            f"gold={gold} answer={answer!r} term={term} score={score:.3f}\n"
            f"# Q: {q}\n{bar}\n")
