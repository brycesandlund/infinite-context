"""BookQAOracle — real-question QA over an anonymized real novel (tasks/bookqa).

Binary scan; the leaf retrieves sentences mentioning the question's entities; the combine
keeps the top-K most-relevant evidence sentences (BOUNDED -> no overflow); the root reads
the collected evidence and emits the gold answer (an entity sentence contains it, enforced
at generation). The combine (collect-top-K, not the scalar reduce) and the root (answer,
not finalize-a-state) differ from the default scaffold, so this overrides `_binary_node` /
`_root`; everything else (recursion, the iterative read, subtask plumbing) is inherited.
"""

from __future__ import annotations

from oracle.base import ScaffoldOracle, ToolCall, _new_id, AssistantTurn


class BookQAOracle(ScaffoldOracle):
    name = "bookqa_oracle"
    _SEP = " ⟐ "

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy="binary")
        self.question = problem.question
        self.entities = self.meta["entities"]
        self.answer = self.meta["answer"]
        self.k = self.meta.get("k", 12)

    def _op_phrase(self):
        ents = ", ".join(self.entities) if self.entities else "the question's subject"
        return f"report sentences relevant to the question (mentioning {ents})"

    def _binary_subtask(self, a, b):
        m = (a + b) // 2
        return (
            f"Strategy: BINARY scan for QA evidence. Find the sentences in tokens {a}..{b} "
            f"relevant to the question: \"{self.question}\" — i.e. {self._op_phrase()}.\n"
            f"- If {b}-{a} > {self.LEAF_TOKENS}: split at the midpoint — spawn one subagent "
            f"for tokens {a}..{m} and one for tokens {m}..{b}, then MERGE their reported "
            f"evidence and keep the most relevant sentences.\n"
            f"- Otherwise read the range and report the relevant sentences (or 'none')."
        )

    # -- evidence (rel, snippet) serialization through \boxed{} ------------------

    def _ser(self, evid):
        evid = sorted(evid, key=lambda e: -e[0])[: self.k]
        return self._SEP.join(f"{r}¦{s}" for r, s in evid) if evid else "none"

    def _parse(self, boxed):
        out = []
        for part in (boxed or "").split(self._SEP):
            r, sep, s = part.partition("¦")
            if sep:
                try:
                    out.append((int(r.strip()), s.strip()))
                except ValueError:
                    pass
        return out

    def _leaf_evidence(self, a, b):
        return [(s[3], s[4]) for s in self._recs_in(a, b)]

    def _merge(self, returns):
        evid = []
        for c in returns:
            evid += self._parse(c)
        return sorted(evid, key=lambda e: -e[0])[: self.k]

    def _show(self, evid):
        return "\n".join(f"  [rel {r}] {s}" for r, s in evid) or "  (no relevant evidence found)"

    # -- root: collect evidence, then ANSWER (not box a state) ------------------

    def _root(self, messages):
        nr, ns = self._reads_spawns(messages)
        if self.doc_len > self.LEAF_TOKENS:
            if ns == 0:
                m = self.doc_len // 2
                return AssistantTurn(
                    text=f"The document is {self.doc_len} tokens. Scanning both halves for "
                         f"sentences relevant to the question, then answering from the evidence.",
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(0, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(m, self.doc_len)}),
                    ],
                )
            return self._answer(self._merge(self._spawn_returns(messages)))
        rt = self._read_phase(0, self.doc_len, messages)
        if rt is not None:
            return rt
        return self._answer(self._leaf_evidence(0, self.doc_len))

    def _answer(self, evid):
        evid = sorted(evid, key=lambda e: -e[0])[: self.k]
        return AssistantTurn(
            text=f"Collected evidence (top {len(evid)}):\n{self._show(evid)}\n\n"
                 f"From this evidence, the answer is:\n\\boxed{{{self.answer}}}",
            tool_calls=[],
        )

    # -- internal/leaf: collect + box serialized top-K evidence -----------------

    def _binary_node(self, a, b, messages, is_root=False):
        nr, ns = self._reads_spawns(messages)
        if (b - a) > self.LEAF_TOKENS:
            if ns == 0:
                m = (a + b) // 2
                return AssistantTurn(
                    text=f"Range {a}..{b} > {self.LEAF_TOKENS}; splitting at midpoint {m} and "
                         f"merging the relevant evidence from each half.",
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(a, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(m, b)}),
                    ],
                )
            merged = self._merge(self._spawn_returns(messages))
            return AssistantTurn(
                text=f"Merging my children's evidence, keeping the top {len(merged)}:\n"
                     f"{self._show(merged)}\n\\boxed{{{self._ser(merged)}}}",
                tool_calls=[],
            )
        rt = self._read_phase(a, b, messages)
        if rt is not None:
            return rt
        evid = self._leaf_evidence(a, b)
        return AssistantTurn(
            text=f"Scanning tokens {a}..{b} for question-relevant sentences "
                 f"({self._op_phrase()}); found {len(evid)}:\n{self._show(evid)}\n"
                 f"\\boxed{{{self._ser(evid)}}}",
            tool_calls=[],
        )
