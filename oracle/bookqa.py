"""BookQAOracle — real-question QA over an anonymized real novel (tasks/bookqa).

Binary scan, driven ONLY by the question (the subtask never names the question's entities —
that would hand the model the retrieval target). The decomposition SKELETON is scripted
(split / route / combine / the iterative read), but the LEAF JUDGMENT — "does the prose I
just read answer the question?" — is not mechanically computable, so it is delegated to a
real model through the `ModelBackend` ABC (`leaf_model`). Each leaf reads its range and the
leaf model returns one of:
  ('answer', answer, evidence_sentence) | ('context', None, [snippets]) | ('none', None, None)
The model's answer bubbles up the tree (combine prefers an answering child), and the root
boxes THE MODEL'S answer — never the gold. Faithfulness is then enforced OUTSIDE the oracle:
the data pipeline grades the boxed answer against gold and KEEPS the trace only if it matches
(rejection sampling). This filters both wrong model reads and noisy gold automatically.

`leaf_model=None` falls back to a scripted span-based leaf (the old substring behavior) so
eval — which only needs the decomposition shape, not a faithful answer — runs without a model.

A node result is serialized through \\boxed{} so a parent can keep combining.
"""

from __future__ import annotations

import re

from oracle.base import ScaffoldOracle, ToolCall, _new_id, AssistantTurn, _RANGE_RE


class BookQAOracle(ScaffoldOracle):
    name = "bookqa_oracle"
    _SEP = " ⟐ "
    _ANS = "ANSWER"   # sentinel: a serialized result whose first field is this carries the answer

    _LEAF_SYS = (
        "You are reading ONE excerpt from a longer document, to help answer a question. "
        "Use ONLY this excerpt — no outside knowledge, no guessing from names alone.\n"
        "If the excerpt DIRECTLY answers the question, reply exactly:\n"
        "ANSWER: <the answer, as short as possible>\n"
        "EVIDENCE: <the sentence or few sentences from the excerpt that establish it, quoted "
        "verbatim>\n"
        "If the excerpt is related but does NOT actually answer the question, reply:\n"
        "CONTEXT: <one short relevant quote>\n"
        "If the excerpt is irrelevant, reply exactly:\nNONE"
    )

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None, leaf_model=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy="binary")
        self.question = problem.question
        self.answer = self.meta["answer"]      # gold — used ONLY by the external rejection gate
        self.k = self.meta.get("k", 12)
        self.leaf_model = leaf_model           # ModelBackend; None => scripted fallback leaf
        # entities live in meta but are used ONLY to build spans at generation time; the
        # oracle never puts them in a prompt.

    # -- async dispatch (the leaf may await a model call) -----------------------

    async def sample(self, messages, max_tokens, tools: bool = True):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        user = user if isinstance(user, str) else ""
        rng = _RANGE_RE.search(user)
        if rng is None:
            return await self._root(messages)
        a, b = int(rng.group(1)), int(rng.group(2))
        return await self._node(a, b, messages)

    def _subtask(self, a: int, b: int) -> str:
        L = self.LEAF_TOKENS
        return (
            f'Find the answer to the question "{self.question}" within the document range in '
            f"tokens {a}..{b}. Recursively split the range at its midpoint, delegating each "
            f"half to a subagent, until the range is less than {L} tokens. When the range is "
            f"less than {L} tokens, read it directly. If this range contains the answer, return "
            f"it with the sentence that states it; otherwise return any relevant information you "
            f'find, or "No relevant information in this range." if there is none.'
        )

    # -- result (answer / context / none) <-> string through \boxed{} -----------

    def _ser(self, result) -> str:
        kind, ans, payload = result
        if kind == "answer":
            return f"{self._ANS}{self._SEP}{ans}{self._SEP}{payload}"
        if kind == "context" and payload:
            return self._SEP.join(payload[: self.k])
        return "none"

    def _parse(self, box):
        parts = (box or "").strip().split(self._SEP)
        if len(parts) >= 2 and parts[0].strip() == self._ANS:
            return ("answer", parts[1].strip(), self._SEP.join(parts[2:]).strip())
        snips = [p.strip() for p in parts if p.strip() and p.strip().lower() != "none"]
        return ("context", None, snips) if snips else ("none", None, None)

    def _merge(self, returns):
        """Combine children: the first answering child wins; else merge their relevant context
        (bounded top-K); else nothing. A wrong pick just fails the external gold gate."""
        answer = None
        ctx = []
        for c in returns:
            kind, ans, payload = self._parse(c)
            if kind == "answer" and answer is None:
                answer = (ans, payload)
            elif kind == "context":
                ctx += payload
        if answer is not None:
            return ("answer", answer[0], answer[1])
        if ctx:
            return ("context", None, ctx[: self.k])
        return ("none", None, None)

    # -- leaf: model-executed (faithful) or scripted span fallback --------------

    async def _leaf(self, a, b, messages):
        if self.leaf_model is not None:
            return await self._leaf_model_call(messages)
        return self._leaf_scripted(a, b)

    async def _leaf_model_call(self, messages):
        # The prose the leaf actually read is in its read_chunk tool results — feed exactly
        # that to the leaf model (no peeking at tokens it didn't read).
        chunk = "".join(
            m["content"] for m in messages
            if m.get("role") == "tool" and m.get("name") == "read_chunk"
        ).strip()
        prompt = [
            {"role": "system", "content": self._LEAF_SYS},
            {"role": "user", "content": f"Question: {self.question}\n\nExcerpt:\n{chunk}"},
        ]
        text = await self.leaf_model.complete(prompt, max_tokens=400)
        return self._parse_model_leaf(text)

    def _parse_model_leaf(self, text):
        t = (text or "").strip()
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", t, re.I)
        if m:
            ans = m.group(1).strip().strip('"').rstrip(".").strip()
            # EVIDENCE may span several sentences — keep the whole block, newlines collapsed.
            ev = re.search(r"EVIDENCE:\s*(.+)", t, re.I | re.S)
            evid = (re.sub(r"\s+", " ", ev.group(1)).strip()[:400] if ev else "")
            if ans and ans.upper() != "NONE":
                return ("answer", ans, evid or ans)
        m = re.search(r"CONTEXT:\s*(.+)", t, re.I | re.S)
        if m:
            snip = m.group(1).strip().split("\n")[0].strip()[:200]
            if snip:
                return ("context", None, [snip])
        return ("none", None, None)

    def _leaf_scripted(self, a, b):
        """Fallback when no leaf_model: the old span-based substring leaf. NOT faithful for
        free-form QA (only used by eval, which needs the shape, not a true answer)."""
        recs = self._recs_in(a, b)
        ans = [s for s in recs if s[5]]
        if ans:
            best = max(ans, key=lambda s: (s[3], -s[2]))
            return ("answer", self.answer, best[4])
        if recs:
            ctx = [s[4] for s in sorted(recs, key=lambda s: (-s[3], s[2]))[: self.k]]
            return ("context", None, ctx)
        return ("none", None, None)

    def _show(self, snips) -> str:
        return "\n".join(f"  • {s}" for s in snips) or "  (none)"

    # -- root: search both halves, then read the answer off the surfaced sentence

    async def _root(self, messages):
        nr, ns = self._reads_spawns(messages)
        if self.doc_len > self.LEAF_TOKENS:
            if ns == 0:
                m = self.doc_len // 2
                return AssistantTurn(
                    text=(f"This document is {self.doc_len} tokens — too long to read in one "
                          f"context. I'll split it in half recursively, search each half for the "
                          f"answer to the question (or any relevant context), and combine; once a "
                          f"half reports the answer, I read it off from the sentence that states it."),
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(0, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(m, self.doc_len)}),
                    ],
                )
            return self._finalize_root(self._merge(self._spawn_returns(messages)))
        rt = self._read_phase(0, self.doc_len, messages)
        if rt is not None:
            return rt
        return self._finalize_root(await self._leaf(0, self.doc_len, messages))

    def _finalize_root(self, result):
        kind, ans, payload = result
        if kind == "answer":
            return AssistantTurn(
                text=(f"A subagent located the answer; the sentence that states it:\n"
                      f"  «{payload}»\nSo the answer is:\n\\boxed{{{ans}}}"),
                tool_calls=[],
            )
        # No leaf answered. Box a non-answer so the trace FAILS the gold gate and is discarded
        # (never conjure the gold here — that would launder an unfaithful trace through).
        shown = self._show(payload) if kind == "context" else "  (none)"
        return AssistantTurn(
            text=(f"No subagent isolated the answer; best available context:\n{shown}\n"
                  f"\\boxed{{answer not found in document}}"),
            tool_calls=[],
        )

    # -- internal/leaf node: search range, box the serialized result ------------

    async def _node(self, a, b, messages):
        nr, ns = self._reads_spawns(messages)
        if (b - a) > self.LEAF_TOKENS:
            if ns == 0:
                m = (a + b) // 2
                return AssistantTurn(
                    text=f"Range {a}..{b} > {self.LEAF_TOKENS}; splitting at midpoint {m} to "
                         f"search each half for the answer.",
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(a, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._subtask(m, b)}),
                    ],
                )
            return self._combine_turn(self._merge(self._spawn_returns(messages)))
        rt = self._read_phase(a, b, messages)
        if rt is not None:
            return rt
        return self._leaf_turn(a, b, await self._leaf(a, b, messages))

    def _combine_turn(self, result):
        kind, ans, payload = result
        if kind == "answer":
            return AssistantTurn(
                text=(f"A subagent found the answer ({ans}); the sentence «{payload}» states it. "
                      f"Passing it up.\n\\boxed{{{self._ser(result)}}}"),
                tool_calls=[],
            )
        if kind == "context":
            return AssistantTurn(
                text=(f"No subagent found the answer; passing up the {len(payload)} most "
                      f"relevant sentence(s):\n{self._show(payload)}\n\\boxed{{{self._ser(result)}}}"),
                tool_calls=[],
            )
        return AssistantTurn(text="Neither half had relevant information.\n\\boxed{none}", tool_calls=[])

    def _leaf_turn(self, a, b, result):
        kind, ans, payload = result
        if kind == "answer":
            return AssistantTurn(
                text=(f"Read tokens {a}..{b}. This passage answers the question — «{payload}» — "
                      f"so the answer is {ans}.\n\\boxed{{{self._ser(result)}}}"),
                tool_calls=[],
            )
        if kind == "context":
            return AssistantTurn(
                text=(f"Read tokens {a}..{b}. No sentence here answers the question; relevant "
                      f"context:\n{self._show(payload)}\n\\boxed{{{self._ser(result)}}}"),
                tool_calls=[],
            )
        return AssistantTurn(
            text=f"Read tokens {a}..{b}. No relevant information in this range.\n\\boxed{{none}}",
            tool_calls=[],
        )
