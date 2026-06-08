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
from collections import Counter
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


# Detect "I'm a subagent for range [a,b]" from the natural subtask text the root
# writes ("Read tokens 6000..12000 ..."). No special marker — so the trained
# model reproduces natural subtasks, and the subagent read-and-answer behavior
# keys off text the model actually generates. RULER questions never contain a
# "tokens N..M" span, so this won't misfire on a root.
_RANGE_RE = re.compile(r"tokens (\d+)\.\.(\d+)")


_VT_STMT_RE = re.compile(r"VAR (\w+) = (VAR \w+|\d+)")


class OracleBackend(ModelBackend):
    """Plays the canonical delegation strategy through `run_agent`, producing
    clean gold traces to warm-start SFT.

    Task-family-aware so subagents report *extractable evidence* (what a model
    could actually compute from a chunk), never gold-derived answers — otherwise
    the skill doesn't transfer (a model can't reproduce "I happen to know the
    answer is here"). Root then does the real aggregation/tracing.

    | family | subagent reports (from its chunk)              | root does            |
    |--------|------------------------------------------------|----------------------|
    | niah   | the magic value(s) for the queried key(s)      | collect, dedup       |
    | cwe    | per-chunk word counts (words seen >= 2x)       | sum -> top-10        |
    | fwe    | per-chunk coded-word counts (excluding '...')  | sum -> top-3         |
    | vt     | the raw `VAR x = ...` statements it finds      | trace chain -> names |

    Overlapping chunks prevent a needle/statement being split across a boundary.
    count_tokens is a deliberate under-estimate; by construction the oracle never
    approaches the budget (one read per subagent, small root conversation).
    """

    name = "oracle"

    def __init__(
        self,
        problem,
        tokenizer,
        *,
        budget: int,
        max_chunk_tokens: int,
        chunk_size: int = 6000,
        overlap: int = 1000,
    ):
        self.doc = problem.document_tokens
        self.doc_len = len(self.doc)
        self.gold = list(problem.gold_answers)
        self.task = problem.task
        self.metadata = dict(problem.metadata)
        self.tokenizer = tokenizer
        self.budget = budget
        self.max_chunk_tokens = min(max_chunk_tokens, chunk_size)
        # OOLONG counts examples by which chunk their token-span starts in, so
        # chunks must PARTITION the doc (overlap=0) or examples get double-counted.
        eff_overlap = 0 if self._family() == "oolong" else overlap
        self.chunks = self._compute_chunks(chunk_size, eff_overlap)

    def _family(self) -> str:
        if self.task.startswith("niah"):
            return "niah"
        if self.task.startswith("oolong"):
            return "oolong"
        return self.task

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

    # -- task-aware subtask, per-chunk report, and root aggregation ----------

    def _subtask_for(self, a: int, b: int) -> str:
        fam = self._family()
        head = f"Read tokens {a}..{b} of the document and "
        if fam == "niah":
            keys = self.metadata.get("query", "the queried key")
            return head + (f"report the special magic value(s) for {keys} found in this "
                           f"range, or 'none' if there are none.")
        if self.task == "cwe":
            return head + ("count how many times each word appears in this range; report the "
                           "words that appear more than once as 'word:count', or 'none'.")
        if self.task == "fwe":
            return head + ("count how often each coded word appears in this range (ignore '...'); "
                           "report the most frequent as 'word:count', or 'none'.")
        if self.task == "vt":
            return head + ("report every variable-assignment statement you find verbatim "
                           "(e.g. 'VAR ABCDE = 12345' or 'VAR FGHIJ = VAR ABCDE'), or 'none'.")
        if fam == "oolong":
            labels = ", ".join(self.metadata.get("labels", []))
            return head + (f"classify each example in this range into one of [{labels}] and "
                           f"report the count of each label as 'label:count'.")
        return head + "report any values relevant to the question, or 'none'."

    def _subagent_report(self, a: int, b: int) -> str:
        text = self.tokenizer.decode(self.doc[a:b])
        fam = self._family()
        if fam == "niah":
            present = [g for g in self.gold if g.lower() in text.lower()]
            return ", ".join(present) if present else "none"
        if self.task == "cwe":
            ctr = Counter(re.findall(r"\d+\.\s+(\S+)", text))
            common = [(w, c) for w, c in ctr.most_common() if c >= 2]
            return ", ".join(f"{w}:{c}" for w, c in common) if common else "none"
        if self.task == "fwe":
            ctr = Counter(w for w in text.split() if w != "...")
            top = [(w, c) for w, c in ctr.most_common(8) if c >= 2]
            return ", ".join(f"{w}:{c}" for w, c in top) if top else "none"
        if self.task == "vt":
            stmts = [f"VAR {v} = {rhs}" for v, rhs in _VT_STMT_RE.findall(text)]
            return "; ".join(stmts) if stmts else "none"
        if fam == "oolong":
            # Count true labels of examples whose token-span STARTS in [a, b).
            ctr: Counter = Counter()
            for start, _end, label, _u, _d in self.metadata.get("example_spans", []):
                if a <= start < b:
                    ctr[label] += 1
            return ", ".join(f"{l}:{c}" for l, c in ctr.items()) if ctr else "none"
        present = [g for g in self.gold if g.lower() in text.lower()]
        return ", ".join(present) if present else "none"

    def _aggregate(self, reports: list[str]) -> str:
        fam = self._family()
        usable = [r for r in reports if r and r.strip().lower() != "none"]
        if fam == "niah":
            seen, out = set(), []
            for r in usable:
                for v in (x.strip() for x in r.split(",")):
                    if v and v not in seen:
                        seen.add(v)
                        out.append(v)
            return ", ".join(out)
        if self.task in ("cwe", "fwe"):
            total: Counter = Counter()
            for r in usable:
                for pair in r.split(","):
                    w, sep, c = pair.rpartition(":")
                    if sep:
                        try:
                            total[w.strip()] += int(c.strip())
                        except ValueError:
                            pass
            total.pop("...", None)
            k = 10 if self.task == "cwe" else 3
            return ", ".join(w for w, _ in total.most_common(k))
        if self.task == "vt":
            stmts = [s.strip() for r in usable for s in r.split(";") if s.strip()]
            return ", ".join(self._trace(stmts))
        if fam == "oolong":
            # Sum per-label counts from the subagents' 'label:count' reports.
            total: Counter = Counter()
            for r in usable:
                for pair in r.split(","):
                    lbl, sep, c = pair.rpartition(":")
                    if sep:
                        try:
                            total[lbl.strip()] += int(c.strip())
                        except ValueError:
                            pass
            qtype = self.metadata.get("question_type")
            if qtype == "count":
                return str(total.get(self.metadata.get("target_label", ""), 0))
            if qtype == "common":
                labels = self.metadata.get("labels", [])
                if not labels:
                    return total.most_common(1)[0][0] if total else ""
                return max(labels, key=lambda l: (total.get(l, 0), -labels.index(l)))
            return ", ".join(self.gold)
        return ", ".join(self.gold)

    def _trace(self, stmts: list[str]) -> list[str]:
        """Return the variables transitively assigned the queried value."""
        value = str(self.metadata.get("queried_value", ""))
        assign: dict[str, tuple[str, str]] = {}
        for s in stmts:
            m = _VT_STMT_RE.search(s)
            if not m:
                continue
            var, rhs = m.group(1), m.group(2)
            if rhs.startswith("VAR "):
                assign[var] = ("ref", rhs[4:].strip())
            else:
                assign[var] = ("lit", rhs)
        holds = {v for v, (k, x) in assign.items() if k == "lit" and x == value}
        changed = True
        while changed:
            changed = False
            for v, (k, x) in assign.items():
                if k == "ref" and x in holds and v not in holds:
                    holds.add(v)
                    changed = True
        return sorted(holds)

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
        m = _RANGE_RE.search(user_msg if isinstance(user_msg, str) else "")

        if m is not None:  # subagent: read my range once, then report evidence
            a, b = int(m.group(1)), int(m.group(2))
            if not has_tool_result:
                return AssistantTurn(
                    text=f"I'll read my assigned range {a}..{b} and report what I find.",
                    tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name="read_chunk",
                                         arguments={"start": a, "end": b})],
                )
            report = self._subagent_report(a, b)
            return AssistantTurn(
                text=f"From my range I found: {report}.\n\\boxed{{{report}}}", tool_calls=[]
            )

        # root: split + delegate, then aggregate the children's evidence
        if not has_tool_result:
            calls = [
                ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name="spawn_subagent",
                         arguments={"subtask": self._subtask_for(a, b)})
                for (a, b) in self.chunks
            ]
            return AssistantTurn(
                text=("The document is larger than my context window, so I'll split it into "
                      "ranges, delegate each to a subagent, then combine their findings."),
                tool_calls=calls,
            )

        reports = [m["content"] for m in messages if m["role"] == "tool"]
        final = self._aggregate(reports)
        return AssistantTurn(
            text=f"Combining the findings from all ranges, the answer is:\n\\boxed{{{final}}}",
            tool_calls=[],
        )


# ---------------------------------------------------------------------------
# Tree-reduce counting oracle (prototype) — fine decomposition so no agent
# counts more than ~LEAF_EXAMPLES items, and every parent sums only a handful.
# ---------------------------------------------------------------------------


_ONODE_RE = re.compile(r"\[ONODE (mid|leaf) (\d+):(\d+)\]")


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


class TreeCountOracle(ModelBackend):
    """Demonstrates that aggregation never requires one agent to count 100.

    Three levels (root depth0 -> mids depth1 -> leaves depth2):
    - LEAF reads a small token range (~LEAF_EXAMPLES items) and counts them.
    - MID sums the per-label counts of its few leaves.
    - ROOT sums the few mid subtotals and answers.

    Every agent's arithmetic load stays <= ~max(LEAF_EXAMPLES, MID_LEAVES).
    OOLONG counting family (uses metadata['example_spans']).
    """

    name = "tree_count_oracle"
    LEAF_EXAMPLES = 12
    MID_LEAVES = 5

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens):
        self.doc = problem.document_tokens
        self.doc_len = len(self.doc)
        self.gold = list(problem.gold_answers)
        self.meta = dict(problem.metadata)
        self.spans = self.meta["example_spans"]  # (start,end,label,user,date), in order
        self.tok = tokenizer
        self.budget = budget

    def count_tokens(self, messages):
        total = sum(
            len(self.tok.encode(m["content"], add_special_tokens=False))
            for m in messages if isinstance(m.get("content"), str)
        )
        return total + 600

    # -- range subdivision (by example boundaries) --------------------------

    def _leaf_ranges(self, a, b):
        idxs = [i for i, s in enumerate(self.spans) if a <= s[0] < b]
        out = []
        for k in range(0, len(idxs), self.LEAF_EXAMPLES):
            grp = idxs[k:k + self.LEAF_EXAMPLES]
            out.append((self.spans[grp[0]][0], self.spans[grp[-1]][1]))
        return out

    def _mid_ranges(self):
        leaves = self._leaf_ranges(0, self.doc_len)
        out = []
        for k in range(0, len(leaves), self.MID_LEAVES):
            grp = leaves[k:k + self.MID_LEAVES]
            out.append((grp[0][0], grp[-1][1]))
        return out

    def _leaf_count(self, a, b):
        ctr = Counter()
        for s in self.spans:
            if a <= s[0] < b:
                ctr[s[2]] += 1
        return ctr

    @staticmethod
    def _fmt(ctr):
        return ", ".join(f"{l}:{c}" for l, c in ctr.items()) if ctr else "none"

    @staticmethod
    def _sum_reports(reports):
        total = Counter()
        for r in reports:
            if not r or r.strip().lower() == "none":
                continue
            for pair in r.split(","):
                lbl, sep, c = pair.rpartition(":")
                if sep:
                    try:
                        total[lbl.strip()] += int(c.strip())
                    except ValueError:
                        pass
        return total

    async def sample(self, messages, max_tokens):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        has_tool = any(m["role"] == "tool" for m in messages)
        m = _ONODE_RE.search(user if isinstance(user, str) else "")

        if m and m.group(1) == "leaf":
            a, b = int(m.group(2)), int(m.group(3))
            if not has_tool:
                return AssistantTurn(
                    text=f"Reading my small range {a}..{b} to count its few items.",
                    tool_calls=[ToolCall(_new_id(), "read_chunk", {"start": a, "end": b})],
                )
            ctr = self._leaf_count(a, b)
            return AssistantTurn(
                text=f"This range has {sum(ctr.values())} items: {self._fmt(ctr)}.\n\\boxed{{{self._fmt(ctr)}}}",
                tool_calls=[],
            )

        if m and m.group(1) == "mid":
            a, b = int(m.group(2)), int(m.group(3))
            if not has_tool:
                calls = [
                    ToolCall(_new_id(), "spawn_subagent", {"subtask":
                        f"[ONODE leaf {la}:{lb}] Read tokens {la}..{lb}, classify each "
                        f"example, and report the per-label counts as 'label:count'."})
                    for (la, lb) in self._leaf_ranges(a, b)
                ]
                return AssistantTurn(text="Subdividing my section into small leaf ranges.", tool_calls=calls)
            total = self._sum_reports([mm["content"] for mm in messages if mm["role"] == "tool"])
            return AssistantTurn(
                text=f"Summing my leaves' counts -> {self._fmt(total)}.\n\\boxed{{{self._fmt(total)}}}",
                tool_calls=[],
            )

        # ROOT
        if not has_tool:
            calls = [
                ToolCall(_new_id(), "spawn_subagent", {"subtask":
                    f"[ONODE mid {a}:{b}] Subdivide tokens {a}..{b} into small ranges, "
                    f"count each, and report the per-label totals as 'label:count'."})
                for (a, b) in self._mid_ranges()
            ]
            return AssistantTurn(
                text="The document is large; I'll split it into sections, each of which "
                     "subdivides further so every batch counted is small.",
                tool_calls=calls,
            )
        total = self._sum_reports([mm["content"] for mm in messages if mm["role"] == "tool"])
        ans = self._answer_from_total(user if isinstance(user, str) else "", total)
        return AssistantTurn(
            text=f"Combining section subtotals: {self._fmt(total)}.\n\\boxed{{{ans}}}",
            tool_calls=[],
        )

    def _answer_from_total(self, q, total):
        """Derive the answer to ANY OOLONG counting question from the tree-reduced
        per-label totals — so the boxed answer verifies the recursive arithmetic,
        not a gold leak. Mirrors CountingTasks' four question forms exactly."""
        if not total:
            return ", ".join(self.gold)
        # absolute count: "how many ... classified as label 'X'?"
        m = re.search(r"classified as label '([^']+)'", q)
        if m:
            return str(total.get(m.group(1), 0))
        # A-vs-B comparison: "is label 'A' more common, less common, or the same
        # frequency as label 'B'?" -> phrase matching CountingTasks' answer.
        m = re.search(r"is label '([^']+)' more common, less common, or the same "
                      r"frequency as label '([^']+)'", q)
        if m:
            a, b = total.get(m.group(1), 0), total.get(m.group(2), 0)
            return ("more common than" if a > b
                    else "less common than" if a < b else "same frequency as")
        # most / least common label across the whole document.
        if "is the most common" in q:
            return max(total, key=total.get)
        if "is the least common" in q:
            return min(total, key=total.get)
        return ", ".join(self.gold)


def make_oracle(problem, tokenizer, *, budget, max_chunk_tokens):
    """Pick the scripted oracle that best decomposes this task.

    `oolong_counting` uses the tree-reduce oracle (leaves count <= ~12 items,
    parents sum a handful) so aggregation never requires one agent to tally the
    whole document — the fine-decomposition pattern we want SFT to teach. Every
    other task uses the standard split-and-delegate OracleBackend.
    """
    if problem.task == "oolong_counting":
        return TreeCountOracle(
            problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens
        )
    return OracleBackend(
        problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens
    )
