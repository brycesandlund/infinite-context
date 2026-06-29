"""Backend-agnostic recursive agent loop (eval).

This is the single control-flow shared across every eval backend (Tinker/Qwen,
Claude, GPT). It mirrors the training rollout's semantics — same system prompt,
same two tools, same per-agent context budget, same recursion and `\\boxed{}`
extraction — but expressed purely in terms of `ModelBackend.count_tokens` and
`ModelBackend.sample`, so it never touches token-level RL machinery.

Document coordinate system: chunks are addressed in the tokenizer used to
generate the problem (Qwen). All backends therefore see identical haystack
*text* via read_chunk; only the per-agent budget is counted in each backend's
own tokenizer (which is the honest meaning of "your context is 10k").
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import harness
from eval.backends import AssistantTurn, ModelBackend, ToolCall


@dataclass
class AgentNode:
    """One agent's conversation + the subagents it spawned."""

    depth: int
    subtask: str                 # parent's spawn request ("" for root)
    answer: str | None           # extracted \boxed{}, or None
    n_turns: int
    termination: str             # answered | stopped_no_answer | overflow | max_turns
    messages: list[dict]
    children: list["AgentNode"] = field(default_factory=list)
    # Optional RL-credit annotations, shown by print_tree when set (None elsewhere,
    # e.g. eval / SFT traces, so the header is unchanged there).
    advantage: float | None = None      # the training advantage this trajectory received
    judge_score: float | None = None    # the judge's score for this node (None for the gold-graded root)


@dataclass
class _RolloutBudget:
    """Shared across every agent in ONE rollout tree: a hard cap on total nodes.
    A runaway policy (e.g. a left-fold chain that never shrinks its range, or an
    overflow-driven re-spawn cascade) would otherwise spin up thousands of agents;
    this kills the tree early. Also a speed governor — bounds the work per rollout.
    asyncio is single-threaded, so the plain counter needs no lock."""

    max_nodes: int | None = None
    count: int = 0

    def exhausted(self) -> bool:
        return self.max_nodes is not None and self.count >= self.max_nodes


def flatten(node: AgentNode) -> list[AgentNode]:
    out = [node]
    for c in node.children:
        out.extend(flatten(c))
    return out


def _tool_msg(tc: ToolCall, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": content}


async def run_agent(
    backend: ModelBackend,
    *,
    document_tokens: list[int],
    tokenizer,
    task_context: str,
    question: str,
    budget: int,
    max_chunk_tokens: int,
    max_depth: int | None,
    max_turns: int | None = None,
    max_nodes: int | None = None,
    depth: int = 0,
    subtask: str = "",
    node_budget: _RolloutBudget | None = None,
) -> AgentNode:
    # One budget object is created at the root (depth 0) and threaded to every
    # descendant, so the cap is on the WHOLE tree, not per-agent.
    if node_budget is None:
        node_budget = _RolloutBudget(max_nodes=max_nodes)
    node_budget.count += 1
    system = harness.make_system_prompt(
        doc_length=len(document_tokens),
        context_budget=budget,
        task_context=task_context,
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    children: list[AgentNode] = []
    answer: str | None = None
    termination = "overflow"   # the context budget is the real terminator
    n_turns = 0

    # No turn cap by default: every turn strictly grows the conversation, so an agent
    # always hits used >= budget eventually. max_turns is an optional safety cap.
    while max_turns is None or n_turns < max_turns:
        used = backend.count_tokens(messages)
        if used >= budget:
            termination = "overflow"
            break
        n_turns += 1
        turn: AssistantTurn = await backend.sample(messages, max_tokens=budget - used)
        messages.append(
            {"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls}
        )

        if not turn.tool_calls:
            answer = harness.extract_boxed(turn.text)
            if answer is not None:
                termination = "answered"
            elif turn.truncated:
                # generation hit the token cap mid-stream (e.g. enumerating too
                # many examples) -> ran out of room == overflow, not a clean stop.
                termination = "overflow"
            else:
                termination = "stopped_no_answer"
            break

        # Dispatch all tool calls in the turn (subagent spawns run concurrently).
        handled = await asyncio.gather(
            *[
                _handle_call(
                    tc,
                    backend=backend,
                    document_tokens=document_tokens,
                    tokenizer=tokenizer,
                    task_context=task_context,
                    budget=budget,
                    max_chunk_tokens=max_chunk_tokens,
                    max_depth=max_depth,
                    max_turns=max_turns,
                    depth=depth,
                    node_budget=node_budget,
                )
                for tc in turn.tool_calls
            ]
        )
        for msg, child in handled:
            messages.append(msg)
            if child is not None:
                children.append(child)
    else:
        termination = "max_turns"   # only reachable when an explicit cap is hit

    return AgentNode(
        depth=depth,
        subtask=subtask,
        answer=answer,
        n_turns=n_turns,
        termination=termination,
        messages=messages,
        children=children,
    )


async def run_single_shot(
    backend: ModelBackend,
    *,
    document_tokens: list[int],
    tokenizer,
    task_context: str,
    question: str,
    max_output_tokens: int,
) -> AgentNode:
    """MODE=single: put the WHOLE document in context and ask for the answer in one
    tool-free call. This is the raw-ability ceiling (frontier single-shot; un-finetuned
    base model) — no decomposition, no budget. Returns the same AgentNode shape as
    run_agent so eval/run.py's scoring + persistence path is identical across modes."""
    system = harness.make_single_shot_prompt(task_context)
    doc_text = tokenizer.decode(document_tokens)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{doc_text}\n\n{question}"},
    ]
    turn = await backend.sample(messages, max_tokens=max_output_tokens, tools=False)
    messages.append({"role": "assistant", "content": turn.text, "tool_calls": []})
    answer = harness.extract_boxed(turn.text)
    if answer is not None:
        termination = "answered"
    elif turn.truncated:
        termination = "overflow"   # ran out of output room (e.g. enumerating too much)
    else:
        termination = "stopped_no_answer"
    return AgentNode(
        depth=0,
        subtask="",
        answer=answer,
        n_turns=1,
        termination=termination,
        messages=messages,
        children=[],
    )


async def _handle_call(
    tc: ToolCall,
    *,
    backend: ModelBackend,
    document_tokens: list[int],
    tokenizer,
    task_context: str,
    budget: int,
    max_chunk_tokens: int,
    max_depth: int | None,
    max_turns: int | None,
    depth: int,
    node_budget: _RolloutBudget,
) -> tuple[dict, AgentNode | None]:
    if tc.name == "read_chunk":
        try:
            start, end = int(tc.arguments["start"]), int(tc.arguments["end"])
        except (KeyError, ValueError, TypeError):
            return _tool_msg(tc, "Error: read_chunk requires integer 'start' and 'end'."), None
        text = harness.read_chunk_impl(document_tokens, tokenizer, start, end, max_chunk_tokens)
        return _tool_msg(tc, text), None

    if tc.name == "spawn_subagent":
        if max_depth is not None and depth >= max_depth:
            return _tool_msg(tc, "Error: max recursion depth reached. Solve directly."), None
        if node_budget.exhausted():
            return _tool_msg(tc, "Error: rollout node budget reached. Solve directly."), None
        child_subtask = str(tc.arguments.get("subtask", "")).strip()
        child = await run_agent(
            backend,
            document_tokens=document_tokens,
            tokenizer=tokenizer,
            task_context=task_context,
            question=child_subtask,
            budget=budget,
            max_chunk_tokens=max_chunk_tokens,
            max_depth=max_depth,
            max_turns=max_turns,
            depth=depth + 1,
            subtask=child_subtask,
            node_budget=node_budget,
        )
        # Tell the parent WHY a subagent failed, so it can react (e.g. shrink the
        # range on overflow, or retry on a non-answer) rather than guessing.
        if child.answer is not None:
            result = child.answer
        elif child.termination == "overflow":
            result = "agent overflowed context"
        else:
            result = "agent did not box an answer"
        return _tool_msg(tc, result), child

    return _tool_msg(tc, f"Error: unknown tool {tc.name!r}."), None
