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
import os
import re
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

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
    truncated: bool = False  # generation hit the token cap (no clean stop) == overflow


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
    async def sample(
        self, messages: list[dict], max_tokens: int, tools: bool = True
    ) -> AssistantTurn:
        """Produce one assistant turn given the conversation so far. With
        `tools=False` the read_chunk/spawn_subagent schema is NOT offered — the
        single-shot full-document mode (MODE=single) needs the model to just answer
        out of context, not to reach for tools that aren't part of that protocol."""

    async def complete(self, messages: list[dict], max_tokens: int = 512) -> str:
        """Plain text completion — used by the LLM judge (eval/judge.py). Shared by
        every backend: run sample() and return its text. The judge prompt offers
        nothing to act on, so sample()'s tools stay unused (tool_choice is 'auto')."""
        turn = await self.sample(messages, max_tokens)
        return turn.text


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

    async def sample(
        self, messages: list[dict], max_tokens: int, tools: bool = True
    ) -> AssistantTurn:
        import tinker

        specs = self._tool_specs if tools else []
        cookbook = neutral_to_cookbook(messages, self.renderer, specs)
        model_input = self.renderer.build_generation_prompt(cookbook)
        resp = await self.sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=self.temperature,
                max_tokens=max(1, max_tokens),
                stop=self._stop,
            ),
        )
        parsed, termination = self.renderer.parse_response(resp.sequences[0].tokens)
        # content may be a str or a multimodal list; get_text_content normalizes to str.
        from tinker_cookbook.renderers import get_text_content

        # No clean stop sequence => the generation was cut off at the token cap
        # (ran out of room). Treated as overflow by the agent loop.
        truncated = not getattr(termination, "is_clean", True)
        return AssistantTurn(
            text=get_text_content(parsed) or "",
            tool_calls=_cookbook_tool_calls_to_neutral(parsed.get("tool_calls")),
            raw=parsed,
            truncated=truncated,
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

    def __init__(self, model: str, temperature: float = 1.0, max_output_cap: int = 16384):
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_output_cap = max_output_cap
        self._tools = harness.openai_tool_specs()
        # Let reasoning models (gpt-5.x, used as policy OR judge backend) silently
        # drop/translate unsupported args (e.g. a fixed temperature, max_tokens ->
        # max_completion_tokens) instead of erroring.
        import litellm

        litellm.drop_params = True

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

    async def sample(
        self, messages: list[dict], max_tokens: int, tools: bool = True
    ) -> AssistantTurn:
        import litellm

        out_cap = max(256, min(max_tokens, self.max_output_cap))
        kwargs = dict(
            model=self.model,
            messages=self._to_openai(messages),
            temperature=self.temperature,
            max_tokens=out_cap,
        )
        if tools:
            kwargs["tools"] = self._tools
            kwargs["tool_choice"] = "auto"
        resp = await litellm.acompletion(**kwargs)
        msg = resp.choices[0].message
        # finish_reason == "length" => generation hit the cap (overflow).
        truncated = getattr(resp.choices[0], "finish_reason", None) == "length"
        tool_calls: list[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(
                ToolCall(id=tc.id or f"call_{uuid.uuid4().hex[:8]}", name=tc.function.name, arguments=args)
            )
        return AssistantTurn(text=msg.content or "", tool_calls=tool_calls, raw=msg, truncated=truncated)


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

    async def sample(
        self, messages: list[dict], max_tokens: int, tools: bool = True
    ) -> AssistantTurn:
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
# Unified OOLONG tree-reduce oracle — one "show-your-work" leaf serves all three
# families (counting / user / temporal). Fine decomposition so no agent ever
# classifies more than ~LEAF_EXAMPLES items, and every parent sums a handful.
#
# Leaf:  reads a small range, ENUMERATES each example (User / Date verbatim +
#        classified Label — the show-your-work the model must learn), then
#        reports a compact family-keyed tally.
# Mid:   subdivides its range into leaves and sums their tallies.
# Root:  splits the whole doc into mids, sums, and derives the answer.
#
# Routing carries NO hidden marker: a node decides leaf-vs-split purely from how
# many example spans fall in the token range named in its (descriptive) subtask.
# The root is the only agent whose user message has no "tokens A..B" range.
# ---------------------------------------------------------------------------


_RANGE_RE = re.compile(r"tokens (\d+)\.\.(\d+)")
_PAIR_SEP = " || "


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


class OolongOracle(ModelBackend):
    """One scripted oracle for all three OOLONG families (counting / user /
    temporal). The leaf shows its work — it enumerates each example, reading the
    User and Date verbatim from the line and committing a classified Label — then
    reports a compact family-keyed tally. Parents only ever SUM a handful of
    tallies, so no single agent classifies more than ~LEAF_EXAMPLES items.

    Family key (what a leaf tallies by):
      counting -> label            user -> 'user|label'       temporal -> 'date|label'

    Root answer derivation: counting + user are derived from the tree-reduced
    tallies (so the boxed answer verifies the recursive arithmetic). Temporal's
    per-date arithmetic (most-frequent-date, before/after, per-month) is the
    stage-2 follow-up; for now it falls back to gold — still a CORRECT trace,
    since the oracle's leaf counts come from ground-truth labels, so the gold IS
    the tree-reduced distribution. The unified show-your-work leaf + structure
    (the classification-bottleneck fix) is identical across all three families.
    """

    name = "oolong_oracle"
    # BINARY recursion: one uniform rule at every node — if range > LEAF_TOKENS, split
    # at the midpoint and spawn 2; else it's a leaf. Depth = ceil(log2(doc/LEAF)), which
    # generalizes to any doc length with the SAME policy (the split point (a+b)//2 is
    # the only thing a node needs to compute). LEAF_TOKENS is small so a leaf counts few
    # examples — the regime where per-leaf counting is reliable (accuracy degrades with
    # chunk size). A leaf reads [a,b] then [b, b+LEAF_OVERLAP]: the seam between the two
    # reads is exactly b, and the second read completes the final example (which may run
    # past b). Counting is by where each example's line STARTS, so each is counted once.
    LEAF_TOKENS = int(os.environ.get("LEAF_TOKENS", "500"))
    LEAF_OVERLAP = int(os.environ.get("LEAF_OVERLAP", "200"))

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens):
        self.doc = problem.document_tokens
        self.doc_len = len(self.doc)
        self.gold = list(problem.gold_answers)
        self.meta = dict(problem.metadata)
        self.spans = self.meta["example_spans"]  # (start,end,label,user,date), in order
        self.family = self.meta.get("family", "counting")
        self.tok = tokenizer
        self.budget = budget
        # QUESTION-ADAPTIVE: only count the labels the question actually asks about
        # (e.g. an A-vs-B comparison needs just A and B, not all 6). targets=None
        # means "all labels" (most/least-common, date questions). This teaches the
        # transferable instinct "extract what's asked" AND keeps the leaf output
        # tiny (the 6-label enumeration was what overflowed).
        self.targets = self._parse_targets(problem.question)
        # User family is ALSO adaptive on the user axis: an "A or B" question needs
        # just those two users, a user-subset question just that subset — not all
        # ~40 users. None = every user (most/2nd-most-often need the full ranking).
        self.user_targets = (
            self._parse_user_targets(problem.question) if self.family == "user" else None
        )
        # Temporal is heterogeneous: each subtype needs a different decomposition
        # (date-window subset -> leaf-filtered label count; month-aggregation ->
        # per-month label tally; date-histogram -> per-date count). _setup_temporal
        # parses the subtype so _key / _relevant / _derive specialize accordingly.
        self.tmode = None
        if self.family == "temporal":
            self._setup_temporal(problem.question)
        # User questions split into label-AGNOSTIC frequency ("which user is
        # represented most/second-most often" -> just count users) and label-based
        # ("...with the label X", subset counting). The frequency mode needs no
        # Instance classification at all, so the leaf counts by user only.
        self.umode = None
        self.uaxis = None
        if self.family == "user":
            self.umode = ("user_freq" if "which user is represented" in problem.question
                          else "user_label")
            # The ANSWER axis decides the (1-D) key: "which user ..." -> tally by user
            # (the label, if any, is a constant filter, not a key dimension); otherwise
            # it's a label question scoped to a user subset -> tally by label (subset
            # filtered at the leaf). Never key by userID|label — one dim is always
            # redundant.
            self.uaxis = "user" if "which user" in problem.question else "label"

    # months <-> number, and the formatted_date parser ("Aug 18, 2024")
    _MONTHS = {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1)}

    @staticmethod
    def _pdate(s: str):
        return datetime.strptime(s.strip(), "%b %d, %Y").date()

    @staticmethod
    def _pcut(s: str):
        """Parse the before/after cutoff, which renders as a dateobj ('2023-04-27',
        possibly with a trailing time component)."""
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date()

    def _setup_temporal(self, q: str):
        """Classify the temporal subtype and pin its parameters. Sets self.tmode and
        (per mode) self.t_win / self.t_cmp / self.t_top / self.t_n, and OVERRIDES
        self.targets to the labels that subtype actually needs."""
        self.t_win = self.t_cmp = self.t_top = self.t_n = None
        # date-range window subset ("... occur between Apr 14, 2023 and Dec 27, 2023, inclusive ...")
        m = re.search(r"occur between (\w+ \d+, \d+) and (\w+ \d+, \d+), inclusive", q)
        if m:
            self.tmode = "window"
            self.t_win = ("range", self._pdate(m.group(1)), self._pdate(m.group(2)))
            return
        # month window subset ("... occur in December of any year ...")
        m = re.search(r"occur in (\w+) of any year", q)
        if m:
            self.tmode = "window"
            self.t_win = ("month", self._MONTHS[m.group(1)])
            return
        # "For how many months does the label 'X' occur more frequently than the label 'Y'?"
        m = re.search(r"how many months does the label '([^']+)' occur more frequently "
                      r"than the label '([^']+)'", q)
        if m:
            self.tmode = "month_more"
            self.t_cmp = (m.group(1), m.group(2))
            self.targets = frozenset(self.t_cmp)
            return
        # "For how many months is the label 'X' the single most frequently occuring label?"
        m = re.search(r"how many months is the label '([^']+)' the single most", q)
        if m:
            self.tmode = "month_top"
            self.t_top = m.group(1)
            self.targets = None  # need every label per month to find the argmax
            return
        # "In which month did the label 'X first occur more often than the label 'Y'?" (sic: open quote)
        m = re.search(r"In which month did the label '(.+?) first occur more often "
                      r"than the label '([^']+)'", q)
        if m:
            self.tmode = "month_first"
            self.t_cmp = (m.group(1).strip(), m.group(2))
            self.targets = frozenset(self.t_cmp)
            return
        # "was label 'X' more common, less common, or the same frequency before T, ... after T"
        m = re.search(r"was label '([^']+)' more common, less common, or the same "
                      r"frequency before (.+?), as compared to after", q)
        if m:
            self.tmode = "before_after"
            self.t_label = m.group(1)
            self.t_cut = self._pcut(m.group(2))
            self.targets = None  # need every label per side for the fraction denominator
            return
        # "how many dates are represented exactly N times"
        m = re.search(r"how many dates are represented exactly (\d+) times", q)
        if m:
            self.tmode = "date_ntimes"
            self.t_n = int(m.group(1))
            self.targets = None  # label-agnostic: we tally date frequencies
            return
        # "which date is represented (the second) most often" (rare; gold format ambiguous)
        if re.search(r"which date is represented .*most often", q):
            self.tmode = "date_2nd" if "second" in q else "date_most"
            self.targets = None

    @staticmethod
    def _parse_user_targets(q: str):
        # "...: User A or User B?" -> just those two users
        m = re.search(r"User (\S+?) or User (\S+?)[?.]", q)
        if m:
            return frozenset({m.group(1), m.group(2)})
        # "...user IDs [a, b, ...]" / "user IDs N" subset -> those users
        m = re.search(r"user(?:s with)? IDs \[?([\d,\s]+?)\]?\s*[.?]", q)
        if m:
            ids = frozenset(re.findall(r"\d+", m.group(1)))
            if ids:
                return ids
        return None  # most/second-most-often -> need the full per-user ranking

    @staticmethod
    def _parse_targets(q: str):
        # A-vs-B comparison -> the two compared labels
        m = re.search(r"is label '([^']+)' more common, less common, or the same "
                      r"frequency as label '([^']+)'", q)
        if m:
            return frozenset({m.group(1), m.group(2)})
        # absolute count -> the single label
        m = re.search(r"classified as label '([^']+)'", q)
        if m:
            return frozenset({m.group(1)})
        # temporal before/after -> the single label
        m = re.search(r"was label '([^']+)' more common, less common", q)
        if m:
            return frozenset({m.group(1)})
        # user "(most|more) instances with the label X" (unquoted, maybe multi-word)
        m = re.search(r"instances with the label (.+?)\s*[?:]", q)
        if m:
            return frozenset({m.group(1).strip()})
        return None  # most/least-common, date questions -> need every label

    def count_tokens(self, messages):
        total = sum(
            len(self.tok.encode(m["content"], add_special_tokens=False))
            for m in messages if isinstance(m.get("content"), str)
        )
        return total + 600

    # -- family key -------------------------------------------------------------

    def _entity_only(self):
        """Subtypes that count BY a single entity (date or user), ignoring the
        Instance label entirely. Returns (field_name, span_index) or None.
        Shared by temporal date-frequency and user-frequency — both just tally how
        many examples fall under each distinct date / user."""
        if self.family == "temporal" and self.tmode in ("date_ntimes", "date_most", "date_2nd"):
            return ("Date", 4)
        if self.family == "user" and self.umode == "user_freq":
            return ("User", 3)
        return None

    def _key(self, span):
        label, user, date = span[2], span[3], span[4]
        ent = self._entity_only()
        if ent:
            return span[ent[1]]                              # count by date / user only
        if self.family == "user":
            return user if self.uaxis == "user" else label  # key by the answer axis
        if self.family == "temporal":
            if self.tmode in ("month_more", "month_top", "month_first"):
                d = self._pdate(date)
                return f"{d.year:04d}-{d.month:02d}|{label}"  # per-month label tally
            if self.tmode == "before_after":
                side = "before" if self._pdate(date) < self.t_cut else "after"
                return f"{side}|{label}"                      # per-side label tally
            return label                                     # window: leaf already date-filtered
        return label

    def _in_window(self, span):
        """temporal 'window' subtype: is this example's Date inside the asked window?
        (True for every other family/subtype — no date filter applies.)"""
        if self.family != "temporal" or self.tmode != "window":
            return True
        d = self._pdate(span[4])
        if self.t_win[0] == "month":
            return d.month == self.t_win[1]
        return self.t_win[1] <= d <= self.t_win[2]

    def _relevant(self, span):
        """Does this example matter for the question? Filters on the label axis
        (targets), the user axis (user_targets), and — for temporal window
        questions — the date axis. None on an axis means "all" on that axis."""
        if self.targets is not None and span[2] not in self.targets:
            return False
        if self.user_targets is not None and str(span[3]) not in self.user_targets:
            return False
        if not self._in_window(span):
            return False
        # before/after excludes examples falling EXACTLY on the cutoff date (vendored
        # walks the boundary index past all ties before splitting before/after).
        if self.tmode == "before_after" and self._pdate(span[4]) == self.t_cut:
            return False
        return True

    def _counts(self, spans):
        ctr = Counter()
        for s in spans:
            if self._relevant(s):
                ctr[self._key(s)] += 1
        return ctr

    # -- range subdivision (binary) ---------------------------------------------

    def _spans_in(self, a, b):
        """Examples whose line STARTS in [a,b). An example straddling a started
        earlier (owned by the left sibling) so it's excluded; one straddling b
        started here so it's included (the leaf's overlap read completes its text)."""
        return [s for s in self.spans if a <= s[0] < b]

    # -- tally formatting / summing --------------------------------------------

    @staticmethod
    def _fmt(ctr):
        return _PAIR_SEP.join(f"{k}:{c}" for k, c in ctr.items()) if ctr else "none"

    @staticmethod
    def _sum_reports(reports):
        total = Counter()
        for r in reports:
            if not r or r.strip().lower() == "none":
                continue
            for pair in r.split(_PAIR_SEP):
                key, sep, c = pair.rpartition(":")
                if sep:
                    try:
                        total[key.strip()] += int(c.strip())
                    except ValueError:
                        pass
        return total

    # -- descriptive subtask (the child only sees THIS, not the root question) --

    def _key_fmt(self):
        ent = self._entity_only()
        if ent:
            return "date:count" if ent[0] == "Date" else "userID:count"
        if self.family == "user":
            return "userID:count" if self.uaxis == "user" else "label:count"
        if self.family == "temporal":
            return {
                "window": "label:count",
                "month_more": "YYYY-MM|label:count",
                "month_top": "YYYY-MM|label:count",
                "month_first": "YYYY-MM|label:count",
                "before_after": "side|label:count",
            }.get(self.tmode, "label:count")
        return "label:count"

    def _task_core(self):
        """The (scope, countwhat, trailer, read_dims) clauses describing the leaf op,
        the same whether the recipient reads directly or fans out. Only asks to read
        the line fields the subtype needs; the trailer carries family-specific extra
        scope (user subset / temporal date filter / per-month grouping)."""
        # entity-frequency subtypes don't classify the Instance at all — just read
        # the one entity (Date or User) and tally how many examples fall under each
        ent = self._entity_only()
        if ent:
            f = ent[0]
            return (
                f"note its {f} (ignore the Instance text — only the {f.lower()} matters here)",
                f"tally how many examples fall under each distinct {f}",
                "", f"read its {f} and ",
            )
        if self.targets is None:
            scope = "classify its Instance into one of the labels described in the task"
            countwhat = "tally every label"
        else:
            tnames = ", ".join(f"'{t}'" for t in sorted(self.targets))
            scope = f"decide whether its Instance should be classified as one of {tnames}"
            countwhat = f"count ONLY {tnames} (ignore the other labels)"
        trailer = ""
        if self.user_targets is not None:
            unames = ", ".join(sorted(self.user_targets))
            trailer = f" Only consider examples from users {unames}; skip all other users."
        elif self.family == "temporal" and self.tmode == "window":
            if self.t_win[0] == "month":
                mname = [k for k, v in self._MONTHS.items() if v == self.t_win[1]][0]
                trailer = (f" Only count examples whose Date falls in {mname} (of any year); "
                           f"skip all examples from other months.")
            else:
                d1 = self.t_win[1].strftime("%b %d, %Y"); d2 = self.t_win[2].strftime("%b %d, %Y")
                trailer = (f" Only count examples whose Date is between {d1} and {d2} "
                           f"inclusive; skip all examples outside that range.")
        elif self.family == "temporal" and self.tmode in ("month_more", "month_top", "month_first"):
            trailer = (" Group your tally by calendar month: key each count as "
                       "'YYYY-MM|label' (e.g. '2024-03|positive').")
        elif self.family == "temporal" and self.tmode == "before_after":
            cut = self.t_cut.strftime("%b %d, %Y")
            trailer = (f" Split by whether the Date is before or after {cut}: key each count as "
                       f"'before|label' or 'after|label' (skip examples dated exactly {cut}).")
        read_dims = {"user": "read its User and ", "temporal": "read its Date and "}.get(
            self.family, ""
        )
        return scope, countwhat, trailer, read_dims

    def _subtask(self, a, b):
        """ONE uniform recursive instruction for any node over tokens [a,b). The first
        'tokens a..b' is the node's own range (what the oracle routes on). The split-or-
        read decision is stated explicitly so the rule is self-similar at every depth."""
        scope, countwhat, userscope, read_dims = self._task_core()
        m = (a + b) // 2
        ob = min(b + self.LEAF_OVERLAP, self.doc_len)
        return (
            f"Count the requested labels in tokens {a}..{b}, reporting '{self._key_fmt()}' "
            f"pairs joined by ' || '.\n"
            f"- If {b}-{a} is larger than {self.LEAF_TOKENS}: split at the midpoint — spawn one "
            f"subagent for {a}..{m} and one for {m}..{b}, then SUM their tallies.\n"
            f"- Otherwise read tokens {a}..{b}; its last line may be a cut-off example, so also "
            f"read tokens {b}..{ob} to finish it — and if that example's line STILL runs past "
            f"{ob} (long entries can), keep reading until its line ends. Then for every example "
            f"whose line STARTS within {a}..{b} (skip a partial first line that began before {a}; "
            f"the final example, completed by the extra read, still counts): "
            f"{read_dims}{scope}; {countwhat}.{userscope}"
        )

    def _snippet(self, span):
        raw = self.tok.decode(self.doc[span[0]:span[1]])
        inst = raw.split("Instance:", 1)[-1].strip()
        return inst[:30].replace("\n", " ")

    def _ex_line(self, s):
        """One enumerated example, with only the dimensions the family needs —
        counting omits User/Date (irrelevant), keeping the line (and budget) small."""
        ent = self._entity_only()
        if ent:                                              # entity-only: no label/snippet
            return f'- User {s[3]}' if ent[0] == "User" else f'- {s[4]}'
        if self.family == "user":
            return f'- User {s[3]}: "{self._snippet(s)}…" -> {s[2]}'
        if self.family == "temporal":
            if self.tmode == "window" and not self._in_window(s):
                return f'- {s[4]}: "{self._snippet(s)}…" -> {s[2]} (outside window — skip)'
            return f'- {s[4]}: "{self._snippet(s)}…" -> {s[2]}'
        return f'- "{self._snippet(s)}…" -> {s[2]}'

    # -- the policy -------------------------------------------------------------

    async def sample(self, messages, max_tokens, tools: bool = True):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        user = user if isinstance(user, str) else ""
        has_tool = any(m["role"] == "tool" for m in messages)
        rng = _RANGE_RE.search(user)  # first 'tokens a..b' = this node's own range

        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if (b - a) <= self.LEAF_TOKENS:
                return self._leaf(a, b, messages)
            return self._internal(a, b, has_tool, messages)
        return self._root(user, has_tool, messages)  # root: original question

    def _leaf(self, a, b, messages):
        """Read [a,b] + [b,b+overlap]; if the final example's line is STILL cut off,
        read the NEXT +overlap block and repeat — each as a separate turn, deciding to
        continue only from what we've actually read (no foreknowledge of where the line
        ends). Once the line is complete, EXHAUSTIVELY enumerate every example whose
        line starts in [a,b) and count."""
        ov = self.LEAF_OVERLAP
        n_read = sum(1 for m in messages if m["role"] == "tool")
        spans_in = self._spans_in(a, b)
        last_end = min(max((s[1] for s in spans_in), default=b), self.doc_len)

        if n_read == 0:                       # turn 1: the range + the first finish-read block
            ob = min(b + ov, self.doc_len)
            return AssistantTurn(
                text=f"Range {a}..{b} is small enough to count directly. Reading it, then the "
                     f"next {ob - b} tokens to finish the final example (its line may run past {b}).",
                tool_calls=[
                    ToolCall(_new_id(), "read_chunk", {"start": a, "end": b}),
                    ToolCall(_new_id(), "read_chunk", {"start": b, "end": ob}),
                ],
            )
        # How far past b we've read so far: the first read is [a,b]; every read after it
        # is a contiguous +overlap block, so coverage extends to b + (n_read-1)*overlap.
        covered = min(b + (n_read - 1) * ov, self.doc_len)
        if covered < last_end:                # we SAW the final line is still open -> read on
            nxt = min(covered + ov, self.doc_len)
            return AssistantTurn(
                text=f"The final example's line is still cut off at {covered} (no end-of-line "
                     f"yet), so I read the next {nxt - covered} tokens ({covered}..{nxt}).",
                tool_calls=[ToolCall(_new_id(), "read_chunk", {"start": covered, "end": nxt})],
            )
        # the final line is now complete -> enumerate + count
        report = self._fmt(self._counts(spans_in))
        # EXHAUSTIVE: list EVERY owned example with its label (accountability per example
        # — "list only matches" silently undercounts), then count the requested labels.
        lines = [self._ex_line(s) for s in spans_in]
        listing = "\n".join(lines) if lines else "  (no complete example starts in this range)"
        ent = self._entity_only()
        verb = (f"Recording the {ent[0]} of every example" if ent
                else "Classifying every example")
        body = (f"{verb} whose line starts in {a}..{b} ({len(spans_in)} of "
                f"them; skipping any partial first line that began earlier):\n{listing}")
        return AssistantTurn(
            text=f"{body}\n\nTally for this range: {report}\n\\boxed{{{report}}}",
            tool_calls=[],
        )

    def _internal(self, a, b, has_tool, messages):
        """Binary: split at the midpoint, spawn 2, sum."""
        if not has_tool:
            m = (a + b) // 2
            calls = [
                ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(a, m)}),
                ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(m, b)}),
            ]
            return AssistantTurn(
                text=f"Range {a}..{b} is wider than {self.LEAF_TOKENS} tokens; splitting at "
                     f"midpoint {m} and delegating {a}..{m} and {m}..{b}.",
                tool_calls=calls,
            )
        total = self._sum_reports([m["content"] for m in messages if m["role"] == "tool"])
        return AssistantTurn(
            text=f"Summing my two subagents' tallies -> {self._fmt(total)}.\n\\boxed{{{self._fmt(total)}}}",
            tool_calls=[],
        )

    def _root(self, question, has_tool, messages):
        if not has_tool:
            m = self.doc_len // 2
            calls = [
                ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(0, m)}),
                ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(m, self.doc_len)}),
            ]
            return AssistantTurn(
                text=f"The document is {self.doc_len} tokens — too large to read. Splitting at "
                     f"midpoint {m} and delegating the two halves, then combining.",
                tool_calls=calls,
            )
        total = self._sum_reports([m["content"] for m in messages if m["role"] == "tool"])
        ans, work = self._derive(question, total)
        if ans is None:
            # Gold fallback: emit ONE acceptable answer (grader accepts any gold by
            # membership) — never join multiple.
            ans, work = (self.gold[0] if self.gold else ""), ""
        # `work` shows the step FROM the combined tally TO the answer (per-month
        # breakdown, fraction compare, argmax) so the root never just "conjures" a
        # number — its reasoning is auditable and learnable.
        body = f"Combining the two halves: {self._fmt(total)}."
        if work:
            body += "\n" + work
        return AssistantTurn(text=f"{body}\n\\boxed{{{ans}}}", tool_calls=[])

    # -- answer derivation from the tree-reduced tally --------------------------

    def _derive(self, q, total):
        """Derive (answer, work) from the tally; work is the shown reasoning between the
        tally and the boxed answer. (None, "") -> fall back to gold."""
        if not total:
            return None, ""
        if self.family == "counting":
            return self._derive_counting(q, total)
        if self.family == "user":
            return self._derive_user(q, total)
        if self.family == "temporal":
            return self._derive_temporal(q, total)
        return None, ""

    def _derive_temporal(self, q, total):
        """Genuinely derive (answer, work) from the tree-reduced tally (no gold leak).
        The tally's key shape matches the subtype (set in _key)."""
        if self.tmode == "window":
            # total is a plain label:count over the in-window examples -> reuse counting
            return self._derive_counting(q, total)
        if self.tmode == "before_after":
            # total keyed 'side|label'; vendored compares FRACTIONS (count / side total)
            bef, aft = Counter(), Counter()
            for k, c in total.items():
                side, _, lbl = k.partition("|")
                (bef if side == "before" else aft)[lbl] += c
            tb, ta = sum(bef.values()), sum(aft.values())
            nb, na = bef.get(self.t_label, 0), aft.get(self.t_label, 0)
            fb = nb / tb if tb else 0.0
            fa = na / ta if ta else 0.0
            ans = "more common" if fb > fa else "less common" if fb < fa else "the same frequency"
            cut = self.t_cut.strftime("%b %d, %Y")
            work = (f"Frequency of '{self.t_label}' before {cut}: {nb}/{tb}={fb:.3f}; "
                    f"after: {na}/{ta}={fa:.3f} -> {ans}.")
            return ans, work
        if self.tmode == "date_ntimes":
            # total is date:count -> the literal count of distinct dates occurring
            # exactly N times (upstream's [:50] cap was dropped in the generator as a
            # benchmark bug; the gold is now the interpretable literal answer).
            n = sum(1 for c in total.values() if c == self.t_n)
            return str(n), f"Distinct dates occurring exactly {self.t_n} time(s): {n}."
        if self.tmode in ("date_most", "date_2nd"):
            return None, ""  # rare; gold-format ambiguous -> fall back to gold for the pick
        # month-aggregation: total is keyed 'YYYY-MM|label'. Rebuild per-month counts.
        per = defaultdict(Counter)
        for k, c in total.items():
            mo, _, lbl = k.partition("|")
            per[mo][lbl] += c
        if self.tmode == "month_more":
            x, y = self.t_cmp
            wins = [mo for mo in sorted(per) if per[mo].get(x, 0) > per[mo].get(y, 0)]
            lines = [f"  {mo}: {x}={per[mo].get(x,0)}, {y}={per[mo].get(y,0)}"
                     + ("  <- " + x if per[mo].get(x, 0) > per[mo].get(y, 0) else "")
                     for mo in sorted(per)]
            work = (f"Per month, {x} vs {y}:\n" + "\n".join(lines)
                    + f"\nMonths where {x} > {y}: {len(wins)}.")
            return str(len(wins)), work
        if self.tmode == "month_top":
            x = self.t_top
            lines, n = [], 0
            for mo in sorted(per):
                ctr = per[mo]; comp = ctr.get(x, 0)
                top = comp > 0 and all(c < comp for l, c in ctr.items() if l != x)
                if top:
                    n += 1
                tally = ", ".join(f"{l}={c}" for l, c in ctr.most_common())
                lines.append(f"  {mo}: {tally}" + ("  <- " + x + " sole top" if top else ""))
            work = (f"Per month, is {x} the sole most-common label?\n" + "\n".join(lines)
                    + f"\nMonths where {x} is the sole top: {n}.")
            return str(n), work
        if self.tmode == "month_first":
            x, y = self.t_cmp
            for mo in sorted(per):  # chronological (YYYY-MM sorts correctly)
                if per[mo][x] > per[mo][y]:
                    yr, mn = mo.split("-")
                    mname = [k for k, v in self._MONTHS.items() if v == int(mn)][0]
                    work = f"Earliest month (chronologically) with {x} > {y}: {mo} -> {mname} {yr}."
                    return f"{mname} {yr}", work
            return None, ""
        return None, ""

    @staticmethod
    def _derive_counting(q, label_counts):
        m = re.search(r"classified as label '([^']+)'", q)
        if m:                                    # the answer IS the tally entry -> no extra work
            return str(label_counts.get(m.group(1), 0)), ""
        m = re.search(r"is label '([^']+)' more common, less common, or the same "
                      r"frequency as label '([^']+)'", q)
        if m:
            la, lb = m.group(1), m.group(2)
            a, b = label_counts.get(la, 0), label_counts.get(lb, 0)
            ans = "more common than" if a > b else "less common than" if a < b else "same frequency as"
            return ans, f"{la}={a} vs {lb}={b} -> {la} is {ans} {lb}."
        if "is the most common" in q:
            mx = max(label_counts.values())
            tied = [l for l, c in label_counts.items() if c == mx]
            w = max(label_counts, key=label_counts.get)
            work = (f"Most common (tie at {mx}): {', '.join(tied)} — answering {w}."
                    if len(tied) > 1 else f"Most common label: {w} ({mx}).")
            return w, work
        if "is the least common" in q:
            mn = min(label_counts.values())
            tied = [l for l, c in label_counts.items() if c == mn]
            w = min(label_counts, key=label_counts.get)
            work = (f"Least common (tie at {mn}): {', '.join(tied)} — answering {w}."
                    if len(tied) > 1 else f"Least common label: {w} ({mn}).")
            return w, work
        return None, ""

    def _derive_user(self, q, counts):
        """User-family (answer, work) from a 1-D tally (keyed by the answer axis; any
        user subset / label filter was already applied at the leaf via _relevant).

        uaxis='label': a label question scoped to a user subset -> counts is a plain
        label tally; most-/least-common / count / A-vs-B all reduce to counting.
        uaxis='user': "which user ..." -> counts is keyed by userID."""
        if self.uaxis == "label":
            return self._derive_counting(q, counts)
        if not counts:
            return None, ""
        if "second most" in q:
            ranked = Counter(counts).most_common()
            if len(ranked) < 2:
                return None, ""
            top = ", ".join(f"{u}({c})" for u, c in ranked[:3])
            return ranked[1][0], f"Users by count: {top}… -> second most = {ranked[1][0]}."
        # "...: User A or User B?" -> whichever of the two has more (of label X)
        m = re.search(r": User (\S+) or User (\S+)\?", q)
        if m:
            a, b = m.group(1), m.group(2)
            ca, cb = counts.get(a, 0), counts.get(b, 0)
            w = a if ca >= cb else b
            return w, f"User {a}={ca} vs User {b}={cb} -> {w}."
        # represented most often / most instances with label X -> the top user
        w = max(counts, key=counts.get)
        return w, f"Top user: {w} ({counts[w]})."


def make_oracle(problem, tokenizer, *, budget, max_chunk_tokens):
    """Pick the scripted oracle that best decomposes this task.

    All three OOLONG families (counting / user / temporal) use the unified
    OolongOracle: a show-your-work leaf (classify <= ~12 enumerated examples)
    plus tree-reduce summing, so no agent ever classifies the whole document.
    Every other task uses the standard split-and-delegate OracleBackend.
    """
    if problem.task in ("oolong_counting", "oolong_user", "oolong_temporal"):
        return OolongOracle(
            problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens
        )
    return OracleBackend(
        problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens
    )
