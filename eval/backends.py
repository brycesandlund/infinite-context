"""Model backends behind a single message-level interface.

`ModelBackend` is the seam that lets one agent loop drive Qwen-via-Tinker,
Claude, or GPT. The interface is message-level (not token-level) because the
API providers don't expose a tokenizer you sample token-IDs from:

- `count_tokens(messages) -> int`   : budget enforcement (each backend counts
  in its own tokenizer — correct, since each model's "10K" is its own).
- `sample(messages, max_tokens) -> AssistantTurn` : one assistant turn, with
  text + any tool calls, in a normalized form.

Message format is a neutral list[dict] (so API-only eval needs no Tinker):
  {"role": "system"|"user"|"assistant"|"tool", "content": str,
   "tool_calls": [ToolCall],            # assistant turns that called tools
   "tool_call_id": str, "name": str}    # tool-result messages

The tool *set* is fixed (read_chunk, spawn_subagent), so each backend injects
its own structured tool schema; the driver only carries the prose system prompt
(harness.make_system_prompt) in messages[0].
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import harness


# ---------------------------------------------------------------------------
# Normalized turn / tool-call types
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None  # provider-native object, for debugging


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ModelBackend(ABC):
    name: str = "backend"

    @abstractmethod
    def count_tokens(self, messages: list[dict]) -> int:
        """Token count of the full conversation (incl. system + tool schema),
        used for the per-agent context budget."""

    @abstractmethod
    async def sample(self, messages: list[dict], max_tokens: int) -> AssistantTurn:
        """Produce one assistant turn given the conversation so far."""


# ---------------------------------------------------------------------------
# Tinker backend (Qwen) — renders identically to training
# ---------------------------------------------------------------------------


class TinkerBackend(ModelBackend):
    """Drives the Tinker-hosted policy through the SAME renderer + tool specs
    that training uses, so eval rollouts are faithful to training rollouts.

    Reuses train.py's actual cookbook tool specs (read_chunk, spawn_subagent)
    so the model sees byte-identical `<tools>` JSON in eval and training.
    """

    def __init__(self, sampling_client, tokenizer, renderer, temperature: float = 1.0):
        self.name = "tinker"
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.temperature = temperature
        # Pull the exact specs training serializes (class-level to_spec()).
        from train import ReadChunkTool, SubagentTool  # tinker-side import

        self._tool_specs = [
            ReadChunkTool.read_chunk.to_spec(),
            SubagentTool.spawn_subagent.to_spec(),
        ]
        self._stop = self.renderer.get_stop_sequences()

    # -- message translation: neutral -> cookbook --------------------------

    def _to_cookbook(self, messages: list[dict]) -> list[dict]:
        from tinker_cookbook.renderers.base import ToolCall as CbToolCall

        system_content = ""
        convo: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                system_content = m["content"]
            elif role == "user":
                convo.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                cb: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    cb["tool_calls"] = [
                        CbToolCall(
                            id=tc.id,
                            function=CbToolCall.FunctionBody(
                                name=tc.name, arguments=json.dumps(tc.arguments)
                            ),
                        )
                        for tc in m["tool_calls"]
                    ]
                convo.append(cb)
            elif role == "tool":
                convo.append(
                    {
                        "role": "tool",
                        "content": m["content"],
                        "tool_call_id": m.get("tool_call_id", ""),
                        "name": m.get("name", ""),
                    }
                )
        prefix = self.renderer.create_conversation_prefix_with_tools(
            tools=self._tool_specs, system_prompt=system_content
        )
        return prefix + convo

    def count_tokens(self, messages: list[dict]) -> int:
        model_input = self.renderer.build_generation_prompt(self._to_cookbook(messages))
        return model_input.length

    async def sample(self, messages: list[dict], max_tokens: int) -> AssistantTurn:
        import tinker

        model_input = self.renderer.build_generation_prompt(self._to_cookbook(messages))
        resp = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.temperature,
                max_tokens=max(1, max_tokens),
                stop=self._stop,
            ),
        )
        parsed, _termination = self.renderer.parse_response(resp.sequences[0].tokens)
        # content may be a str or a multimodal list; get_text_content normalizes to str.
        from tinker_cookbook.renderers import get_text_content

        return AssistantTurn(
            text=get_text_content(parsed) or "",
            tool_calls=_cookbook_tool_calls_to_neutral(parsed.get("tool_calls")),
            raw=parsed,
        )


def _cookbook_tool_calls_to_neutral(cb_calls) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in cb_calls or []:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        out.append(
            ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}", name=tc.function.name, arguments=args)
        )
    return out


# ---------------------------------------------------------------------------
# API backend (Anthropic / OpenAI) via LiteLLM
# ---------------------------------------------------------------------------


class APIBackend(ModelBackend):
    """One backend for any LiteLLM-supported chat model with tool calling.

    model examples: "anthropic/claude-sonnet-4-20250514", "openai/gpt-5-mini".
    Each model counts tokens in its own tokenizer (litellm.token_counter).
    """

    def __init__(self, model: str, temperature: float = 1.0, max_output_cap: int = 8192):
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_output_cap = max_output_cap
        self._tools = harness.openai_tool_specs()

    def _to_openai(self, messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role in ("system", "user"):
                out.append({"role": role, "content": m["content"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in m["tool_calls"]
                    ]
                out.append(msg)
            elif role == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("tool_call_id", ""),
                        "content": m["content"],
                    }
                )
        return out

    def count_tokens(self, messages: list[dict]) -> int:
        import litellm

        try:
            return litellm.token_counter(model=self.model, messages=self._to_openai(messages))
        except Exception:
            # Fallback: rough char/4 estimate if the model isn't in litellm's map.
            return sum(len(m.get("content") or "") for m in messages) // 4

    async def sample(self, messages: list[dict], max_tokens: int) -> AssistantTurn:
        import litellm

        out_cap = max(256, min(max_tokens, self.max_output_cap))
        resp = await litellm.acompletion(
            model=self.model,
            messages=self._to_openai(messages),
            tools=self._tools,
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=out_cap,
        )
        msg = resp.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}", name=tc.function.name, arguments=args)
            )
        return AssistantTurn(text=msg.content or "", tool_calls=tool_calls, raw=msg)
