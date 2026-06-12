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
    max_depth: int,
    max_turns: int,
    depth: int = 0,
    subtask: str = "",
) -> AgentNode:
    system = harness.make_system_prompt(
        doc_length=len(document_tokens),
        context_budget=budget,
        max_chunk_tokens=max_chunk_tokens,
        task_context=task_context,
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    children: list[AgentNode] = []
    answer: str | None = None
    termination = "max_turns"
    n_turns = 0

    for _ in range(max_turns):
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
                )
                for tc in turn.tool_calls
            ]
        )
        for msg, child in handled:
            messages.append(msg)
            if child is not None:
                children.append(child)

    return AgentNode(
        depth=depth,
        subtask=subtask,
        answer=answer,
        n_turns=n_turns,
        termination=termination,
        messages=messages,
        children=children,
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
    max_depth: int,
    max_turns: int,
    depth: int,
) -> tuple[dict, AgentNode | None]:
    if tc.name == "read_chunk":
        try:
            start, end = int(tc.arguments["start"]), int(tc.arguments["end"])
        except (KeyError, ValueError, TypeError):
            return _tool_msg(tc, "Error: read_chunk requires integer 'start' and 'end'."), None
        text = harness.read_chunk_impl(document_tokens, tokenizer, start, end, max_chunk_tokens)
        return _tool_msg(tc, text), None

    if tc.name == "spawn_subagent":
        if depth >= max_depth:
            return _tool_msg(tc, "Error: max recursion depth reached. Solve directly."), None
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
