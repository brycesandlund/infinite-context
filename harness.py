"""Shared agent-harness primitives.

Single source of truth for the parts of the recursive-agent setup that MUST be
identical between the RL training loop (rl.py) and the multi-backend eval
driver (eval/), so a Qwen-vs-Claude-vs-GPT comparison isn't confounded by
harness differences:

- `make_system_prompt`: the system prompt every agent (root + subagents) sees.
- `read_chunk_impl`: the exact token-slice / decode / chunk-cap semantics of the
  read_chunk tool.
- `extract_boxed`: pull the final \\boxed{} answer out of a response.
- Canonical tool descriptions + `openai_tool_specs()`: the tool schema text the
  model is shown. rl.py asserts its cookbook @tool descriptions equal these
  constants at import time (see rl.py), so the two can't drift.

Training keeps its own (token-level, RL) rollout machinery; this module holds
only the backend-agnostic surface.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Canonical tool descriptions (the text the model is shown). rl.py's cookbook
# @tool docstrings must equal these — enforced by an import-time assert there.
# ---------------------------------------------------------------------------

READ_CHUNK_DESCRIPTION = (
    "Read a slice of the document and return the decoded text of tokens "
    "[start, end). The document has a fixed length (stated in the system prompt). "
    "Each call is capped at the chunk limit; for larger ranges, issue multiple "
    "reads or delegate to a subagent."
)
SPAWN_SUBAGENT_DESCRIPTION = (
    "Spawn a fresh-context copy of yourself to solve `subtask`; returns the "
    "child's \\boxed{} answer."
)
READ_CHUNK_START_DESC = "First token position to read (inclusive)."
READ_CHUNK_END_DESC = "Last token position to read (exclusive)."
SPAWN_SUBAGENT_SUBTASK_DESC = "Sub-problem statement (free text) for the child to solve"


def openai_tool_specs() -> list[dict]:
    """Tool schema in OpenAI/LiteLLM function-calling format (also what Anthropic
    consumes via LiteLLM). Built from the canonical constants above."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_chunk",
                "description": READ_CHUNK_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "description": READ_CHUNK_START_DESC},
                        "end": {"type": "integer", "description": READ_CHUNK_END_DESC},
                    },
                    "required": ["start", "end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "spawn_subagent",
                "description": SPAWN_SUBAGENT_DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtask": {"type": "string", "description": SPAWN_SUBAGENT_SUBTASK_DESC},
                    },
                    "required": ["subtask"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# read_chunk semantics (shared by rl.py's cookbook tool and the eval driver)
# ---------------------------------------------------------------------------


def read_chunk_impl(
    document_tokens: list[int],
    tokenizer,
    start: int,
    end: int,
    max_chunk_tokens: int,
) -> str:
    """Return the decoded text of document_tokens[start:end], clamped to bounds
    and capped at max_chunk_tokens. Returns a short error/empty message string
    (not raising) for degenerate ranges, matching the original tool behavior."""
    n = len(document_tokens)
    if start < 0:
        start = 0
    if end > n:
        end = n
    if end <= start:
        return "Empty range."
    if end - start > max_chunk_tokens:
        return (
            f"Range too large ({end - start} tokens > {max_chunk_tokens} cap). "
            f"Issue smaller reads or delegate to a subagent."
        )
    return tokenizer.decode(document_tokens[start:end])


# ---------------------------------------------------------------------------
# System prompt (root + every subagent)
# ---------------------------------------------------------------------------


def make_system_prompt(
    doc_length: int,
    context_budget: int,
    max_chunk_tokens: int,
    task_context: str,
) -> str:
    """Build the system prompt shared by parent and every spawned subagent.

    `task_context` carries per-task instructions (label space, format hint, etc.)
    pinned at the top so a freshly-spawned subagent already knows the task
    without the parent re-explaining it in each subtask string. For RULER tasks
    it is empty (the root receives RULER's instruction in the user message)."""
    task_block = f"{task_context}\n\n" if task_context else ""
    return (
        f"{task_block}"
        f"You are a long-document assistant. The document is {doc_length} tokens "
        f"long; you cannot see it directly. Your own context window is {context_budget} "
        f"tokens — the conversation (system prompt, user message, your responses, "
        f"and all tool results) must fit in this budget or the episode ends.\n\n"
        f"You have two tools:\n"
        f"- `read_chunk(start, end)`: read the document tokens in [start, end). A "
        f"single read returns at most {max_chunk_tokens} tokens — most of your context "
        f"window — so plan accordingly: typically you can do one read of any range and "
        f"then must either answer or delegate further.\n"
        f"- `spawn_subagent(subtask)`: delegate to a fresh-context copy of yourself "
        f"(also {context_budget} tokens) with the same tools and the same task "
        f"instructions above. Pass an explicit subtask string that names a token "
        f"range and the sub-question, e.g. \"Read tokens 5000..7000 and report the "
        f"findings relevant to the task in that range, or 'none'.\" The subagent "
        f"returns its final \\boxed{{}} answer as a string.\n\n"
        f"You may call spawn_subagent multiple times in parallel within a single turn "
        f"to scan disjoint ranges concurrently. When you are confident in the final "
        f"answer, emit it as \\boxed{{value}} and stop."
    )


def make_single_shot_prompt(task_context: str) -> str:
    """System prompt for MODE=single: the whole document is placed directly in the
    user message (no read_chunk, no decomposition), so there is no budget/tool prose
    — only the per-task instructions plus the boxed-answer contract the grader needs.
    This is the frontier-ceiling / raw-base-model protocol: read everything, answer."""
    task_block = f"{task_context}\n\n" if task_context else ""
    return (
        f"{task_block}"
        "You are given a complete document followed by a question about it. Read the "
        "document carefully and answer the question. Show your reasoning, then emit "
        "your final answer as \\boxed{value} and stop."
    )


# ---------------------------------------------------------------------------
# Boxed-answer extraction
# ---------------------------------------------------------------------------


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None
