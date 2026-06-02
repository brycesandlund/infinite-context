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
import re
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
        return neutral_to_cookbook(messages, self.renderer, self._tool_specs)

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


def neutral_to_cookbook(messages: list[dict], renderer, tool_specs: list[dict]) -> list[dict]:
    """Translate neutral messages -> cookbook Message format, with the tool-aware
    system prefix (create_conversation_prefix_with_tools). Shared by TinkerBackend
    (rendering for sampling) and the SFT converter (rendering supervised examples),
    so both produce identical token sequences."""
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
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=tool_specs, system_prompt=system_content
    )
    return prefix + convo


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


# ---------------------------------------------------------------------------
# Oracle backend — scripted-optimal delegation, for SFT warm-start data
# ---------------------------------------------------------------------------


_ORACLE_MARK_RE = re.compile(r"\[ORACLE range=(\d+):(\d+)\]")


class OracleBackend(ModelBackend):
    """Plays the canonical delegation strategy through `run_agent`, producing
    clean gold traces to warm-start SFT.

    Strategy:
    - Root: the document exceeds the budget, so split it into overlapping
      budget-sized chunks and spawn one subagent per chunk. Then aggregate the
      children's findings and box the formatted gold answer.
    - Subagent: read its assigned range once, then box the gold value(s) present
      in that range (or 'none').

    It "knows" the answer by decoding each chunk and substring-matching the gold
    values — overlap between chunks prevents a needle being split across a
    boundary. count_tokens is a (deliberate under-)estimate; by construction the
    oracle never approaches the budget (one read per subagent, small root convo).
    """

    name = "oracle"

    def __init__(
        self,
        document_tokens: list[int],
        gold_answers: list[str],
        tokenizer,
        *,
        budget: int,
        max_chunk_tokens: int,
        chunk_size: int = 6000,
        overlap: int = 1000,
    ):
        self.doc = document_tokens
        self.doc_len = len(document_tokens)
        self.gold = list(gold_answers)
        self.tokenizer = tokenizer
        self.budget = budget
        self.max_chunk_tokens = min(max_chunk_tokens, chunk_size)
        self.chunks = self._compute_chunks(chunk_size, overlap)

    def _compute_chunks(self, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
        step = max(1, chunk_size - overlap)
        starts: list[int] = []
        s = 0
        while True:
            starts.append(s)
            if s + chunk_size >= self.doc_len:
                break
            s += step
        return [(s, min(s + chunk_size, self.doc_len)) for s in starts]

    def _findings(self, start: int, end: int) -> list[str]:
        text = self.tokenizer.decode(self.doc[start:end]).lower()
        return [g for g in self.gold if g.lower() in text]

    def count_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                total += len(self.tokenizer.encode(c, add_special_tokens=False))
        return total + 600  # rough system+tools+header overhead

    async def sample(self, messages: list[dict], max_tokens: int) -> AssistantTurn:
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        has_tool_result = any(m["role"] == "tool" for m in messages)
        m = _ORACLE_MARK_RE.search(user_msg if isinstance(user_msg, str) else "")

        if m is not None:  # subagent
            a, b = int(m.group(1)), int(m.group(2))
            if not has_tool_result:
                return AssistantTurn(
                    text=f"I'll read my assigned range {a}..{b} and look for relevant values.",
                    tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name="read_chunk",
                                         arguments={"start": a, "end": b})],
                )
            present = self._findings(a, b)
            ans = ", ".join(present) if present else "none"
            return AssistantTurn(
                text=f"In my range I found: {ans}.\n\\boxed{{{ans}}}", tool_calls=[]
            )

        # root
        if not has_tool_result:
            calls = []
            for (a, b) in self.chunks:
                sub = (
                    f"[ORACLE range={a}:{b}] Read tokens {a}..{b} of the document and "
                    f"report any values relevant to the question, or 'none' if there are none."
                )
                calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name="spawn_subagent",
                                      arguments={"subtask": sub}))
            return AssistantTurn(
                text=("The document is larger than my context window, so I'll split it into "
                      "ranges and delegate each to a subagent, then combine their findings."),
                tool_calls=calls,
            )

        final = ", ".join(self.gold)
        return AssistantTurn(
            text=f"Combining the findings from all ranges, the answer is:\n\\boxed{{{final}}}",
            tool_calls=[],
        )
