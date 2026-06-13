"""Shared rollout-tree rendering — THE one way trees are printed everywhere.

eval/run.py (eval rollouts), sft.py (oracle traces), and train.py (RL rollouts,
both the per-step dump and the post-run eval) all render through these helpers,
so a tree reads identically no matter which pipeline produced it. RL trajectories
(token-level) are first converted to the same neutral AgentNode shape by
rollout_to_agent_node() below.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.agent import AgentNode
    from train import RolloutNode


def print_tree(node: "AgentNode", indent: int = 0, full: bool = False) -> None:
    """Pretty-print a rollout tree. `full=True` disables ALL truncation (used for
    the OUT.txt dump so saved rollouts are complete); the truncating default keeps
    stdout readable."""
    def clip(s: str, n: int) -> str:
        return s if full or len(s) <= n else s[:n] + " …"

    prefix = "  " * indent
    bar = "=" * max(8, 76 - len(prefix))
    # RL-credit annotations appear in the header only when populated (the dump),
    # so eval / SFT trees render unchanged.
    credit = ""
    if node.advantage is not None:
        credit += f" adv={node.advantage:+.3f}"
    if node.judge_score is not None:
        credit += f" judge={node.judge_score:.2f}"
    print(f"{prefix}{bar}")
    print(
        f"{prefix}[depth={node.depth}] turns={node.n_turns} "
        f"termination={node.termination} answer={node.answer!r}{credit}"
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


# ---------------------------------------------------------------------------
# RL RolloutNode (token trajectories) -> neutral AgentNode, so RL rollouts
# render through the same printers above as eval + SFT traces.
# ---------------------------------------------------------------------------

_BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.S)
_TOOL_RESP_RE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.S)


def _flatten_ob_tokens(ob) -> list[int]:
    """Pull all token ids out of a tinker.ModelInput, ignoring non-text chunks."""
    out: list[int] = []
    for chunk in ob.chunks:
        toks = getattr(chunk, "tokens", None)
        if toks is not None:
            out.extend(toks)
    return out


def _neutral_tool_calls(cb_tool_calls):
    from eval.backends import ToolCall as NeutralToolCall

    out = []
    for tc in cb_tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {"raw": tc.function.arguments}
        out.append(NeutralToolCall(id=tc.id or "", name=tc.function.name, arguments=args))
    return out


def rollout_to_agent_node(node: "RolloutNode", tokenizer, renderer) -> "AgentNode":
    """Convert an RL rollout tree into the neutral AgentNode shape used by the
    eval driver. Assistant turns are recovered with the renderer's own
    parse_response (exact tool-call parsing); tool results are recovered from the
    token DELTA between consecutive observations (a conversation only grows by
    [prev action][tool results][next generation prompt])."""
    from eval.agent import AgentNode
    from tinker_cookbook.renderers import get_text_content

    traj = node.trajectory
    messages: list[dict] = []

    # System + first user message from the first observation.
    first_ob = _flatten_ob_tokens(traj.transitions[0].ob) if traj.transitions else []
    for role, content in _BLOCK_RE.findall(tokenizer.decode(first_ob)):
        if role in ("system", "user"):
            messages.append({"role": role, "content": content.strip()})

    prev_tokens = list(first_ob)
    for i, tr in enumerate(traj.transitions):
        ac_tokens = list(tr.ac.tokens)
        try:
            parsed, _ = renderer.parse_response(ac_tokens)
        except Exception:
            parsed = {"content": tokenizer.decode(ac_tokens)}
        tool_calls = _neutral_tool_calls(parsed.get("tool_calls"))
        messages.append({
            "role": "assistant",
            "content": get_text_content(parsed) or "",
            "tool_calls": tool_calls,
        })
        prev_tokens += ac_tokens
        if i + 1 < len(traj.transitions):
            next_ob = _flatten_ob_tokens(traj.transitions[i + 1].ob)
            delta = tokenizer.decode(next_ob[len(prev_tokens):])
            # Pair tool responses positionally with this turn's tool calls.
            for j, resp in enumerate(_TOOL_RESP_RE.findall(delta)):
                name = tool_calls[j].name if j < len(tool_calls) else "tool"
                messages.append({"role": "tool", "name": name, "content": resp})
            prev_tokens = list(next_ob)

    tm = getattr(traj, "metrics", None) or {}
    if node.answer is not None:
        termination = "answered"
    elif tm.get("context_overflow") or tm.get("max_tokens_reached"):
        termination = "overflow"
    else:
        termination = "stopped_no_answer"

    return AgentNode(
        depth=node.depth,
        subtask=node.subtask,
        answer=node.answer,
        n_turns=len(traj.transitions),
        termination=termination,
        messages=messages,
        children=[rollout_to_agent_node(c, tokenizer, renderer) for c in node.children],
    )
