"""Scripted oracle for the abstract synthetic decomposition tasks (tasks/synth).

ONE oracle that can render EITHER decomposition strategy, chosen at construction
(the per-task TRAINING knob). The strategy is dictated by the root and propagated
verbatim into every subtask, so the trained model learns to (a) pick a strategy and
(b) keep using it down the tree.

  strategy="binary"     balanced split: a node splits [a,b] at the midpoint, spawns
                        TWO children, and combines their two results (sum / max). For
                        the bounded-associative tasks (sum/count/max). Depth ~log2.

  strategy="left_fold"  right-leaning chain: a node reads the first leaf-sized slice
                        of its range, folds it into the running accumulator threaded
                        in from its parent, then spawns ONE child for the rest with
                        the updated accumulator; the last slice returns the final
                        value, which bubbles back up. For the stateful task. Depth ~#chunks.

The leaf-op (parse a field) is trivial, so the trace teaches the SCAFFOLD, not the
leaf-op. A leaf enumerates the records it owns + their contribution before boxing.
"""

from __future__ import annotations

import os
import re
import uuid

import harness
from eval.backends import AssistantTurn, ModelBackend, ToolCall

_RANGE_RE = re.compile(r"tokens (\d+)\.\.(\d+)")
_ACC_RE = re.compile(r"accumulator so far = (-?\d+|none)")


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


class SynthOracle(ModelBackend):
    name = "synth_oracle"
    LEAF_TOKENS = int(os.environ.get("LEAF_TOKENS", "500"))
    LEAF_OVERLAP = int(os.environ.get("LEAF_OVERLAP", "120"))

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        self.doc_len = len(problem.document_tokens)
        self.tok = tokenizer
        self.task = problem.task
        self.meta = dict(problem.metadata)
        self.spans = self.meta["record_spans"]   # (start, end, idx, amt, flag, grp)
        self.strategy = strategy or self.meta.get("strategy_default", "binary")
        self.budget = budget
        if self.strategy == "binary" and self.task == "synth_runreset":
            # runreset's combine is non-associative; a binary tree would need a
            # richer monoid. Not rendered yet — the all-binary experiment adds it.
            raise ValueError("synth_runreset has no binary oracle yet; use left_fold")

    # -- per-task leaf-op / combine / fold --------------------------------------

    def _recs_in(self, a: int, b: int) -> list[tuple]:
        """Records whose line STARTS in [a, b), in document order."""
        return [s for s in self.spans if a <= s[0] < b]

    def _identity(self):
        return None if self.task == "synth_max" else 0

    def _leaf_value(self, recs: list[tuple]):
        if self.task == "synth_sum":
            return sum(s[3] for s in recs)
        if self.task == "synth_count":
            return sum(1 for s in recs if s[4] == "Y")
        if self.task == "synth_max":
            return max((s[3] for s in recs), default=None)
        raise ValueError(self.task)

    def _combine(self, vals: list):
        vals = [v for v in vals if v is not None]
        if self.task == "synth_max":
            return max(vals) if vals else None
        return sum(vals)

    def _fold(self, recs: list[tuple], acc):
        for s in recs:
            amt, flag, grp = s[3], s[4], s[5]
            if self.task == "synth_runreset":
                acc = 0 if grp == "RST" else acc + amt
            elif self.task == "synth_sum":
                acc += amt
            elif self.task == "synth_count":
                acc += 1 if flag == "Y" else 0
            elif self.task == "synth_max":
                acc = amt if acc is None else max(acc, amt)
        return acc

    def _contrib(self, s: tuple) -> str:
        idx, amt, flag, grp = s[2], s[3], s[4], s[5]
        if self.task == "synth_count":
            return f"- [{idx:04d}] flag={flag}" + ("  (+1)" if flag == "Y" else "")
        if self.task == "synth_runreset":
            return f"- [{idx:04d}] grp={grp} amt={amt:+d}" + ("  -> RESET to 0" if grp == "RST" else "")
        return f"- [{idx:04d}] amt={amt:+d}"

    def _op_phrase(self) -> str:
        return {
            "synth_sum": "add up the 'amt' fields",
            "synth_count": "count the records with flag=Y",
            "synth_max": "take the maximum 'amt'",
            "synth_runreset": "fold 'amt' left-to-right, resetting to 0 on grp=RST",
        }[self.task]

    def _combine_phrase(self) -> str:
        return "take the max of" if self.task == "synth_max" else "sum"

    # -- subtask construction (carries strategy + range [+ accumulator]) --------

    def _binary_subtask(self, a: int, b: int) -> str:
        m = (a + b) // 2
        return (
            f"Strategy: BINARY split. Compute the partial result for tokens {a}..{b} "
            f"(leaf-op: {self._op_phrase()}).\n"
            f"- If {b}-{a} > {self.LEAF_TOKENS}: split at the midpoint — spawn one "
            f"subagent for tokens {a}..{m} and one for tokens {m}..{b}, then "
            f"{self._combine_phrase()} their two results.\n"
            f"- Otherwise read the range and compute the partial directly."
        )

    def _fold_subtask(self, a: int, b: int, acc) -> str:
        accs = "none" if acc is None else str(acc)
        return (
            f"Strategy: LEFT-FOLD. accumulator so far = {accs}. Process tokens "
            f"{a}..{b} left-to-right (leaf-op: {self._op_phrase()}).\n"
            f"- Read the first ~{self.LEAF_TOKENS} tokens, update the accumulator over "
            f"those records, then spawn ONE subagent for the rest with the updated "
            f"accumulator.\n"
            f"- If the whole range is ≤ {self.LEAF_TOKENS} tokens, process it and return "
            f"the final accumulator."
        )

    def count_tokens(self, messages):
        total = sum(
            len(self.tok.encode(m["content"], add_special_tokens=False))
            for m in messages if isinstance(m.get("content"), str)
        )
        return total + 400   # deliberate under-estimate; oracle never nears budget

    # -- the policy -------------------------------------------------------------

    async def sample(self, messages, max_tokens, tools: bool = True):
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        user = user if isinstance(user, str) else ""
        n_tool = sum(1 for m in messages if m["role"] == "tool")
        rng = _RANGE_RE.search(user)

        if rng is None:
            return self._root(n_tool, messages)
        a, b = int(rng.group(1)), int(rng.group(2))
        if self.strategy == "left_fold":
            return self._fold_node(a, b, self._parse_acc(user), n_tool, messages)
        return self._binary_node(a, b, n_tool, messages)

    def _parse_acc(self, user: str):
        m = _ACC_RE.search(user)
        if not m:
            return self._identity()
        return None if m.group(1) == "none" else int(m.group(1))

    # -- root -------------------------------------------------------------------

    def _root(self, n_tool, messages):
        if n_tool == 0:
            if self.strategy == "left_fold":
                sub = self._fold_subtask(0, self.doc_len, self._identity())
                txt = (f"The document is {self.doc_len} tokens. I'll LEFT-FOLD a running "
                       f"accumulator from the start, threading it through the chain.")
            else:
                sub = self._binary_subtask(0, self.doc_len)
                txt = (f"The document is {self.doc_len} tokens — too large to read. I'll split "
                       f"with a BINARY tree and {self._combine_phrase()} the parts.")
            return AssistantTurn(text=txt, tool_calls=[ToolCall(_new_id(), "spawn_subagent", {"subtask": sub})])
        val = self._child_int(messages)
        return AssistantTurn(text=f"The decomposition returned {val}.\n\\boxed{{{val}}}", tool_calls=[])

    # -- binary node ------------------------------------------------------------

    def _binary_node(self, a, b, n_tool, messages):
        if (b - a) > self.LEAF_TOKENS:
            if n_tool == 0:
                m = (a + b) // 2
                return AssistantTurn(
                    text=f"Range {a}..{b} > {self.LEAF_TOKENS}; splitting at midpoint {m}.",
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(a, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(m, b)}),
                    ],
                )
            vals = [self._extract_int(x["content"]) for x in messages if x["role"] == "tool"]
            res = self._combine(vals)
            return AssistantTurn(
                text=f"{self._combine_phrase().capitalize()} of children {vals} = {res}.\n\\boxed{{{res}}}",
                tool_calls=[],
            )
        # leaf: read the range, then compute the partial
        if n_tool == 0:
            return self._read(a, b)
        recs = self._recs_in(a, b)
        lines = "\n".join(self._contrib(s) for s in recs) or "  (no record starts here)"
        val = self._leaf_value(recs)
        return AssistantTurn(
            text=f"Computing the partial over {len(recs)} records ({self._op_phrase()}):\n"
                 f"{lines}\nPartial = {val}.\n\\boxed{{{val}}}",
            tool_calls=[],
        )

    # -- left-fold node: read slice (n_tool 0->2) -> fold+spawn (2) -> bubble (3) --

    def _fold_node(self, a, b, acc, n_tool, messages):
        cut = min(a + self.LEAF_TOKENS, b)
        if n_tool == 0:
            return self._read(a, cut)
        recs = self._recs_in(a, cut)
        acc_out = self._fold(recs, acc)
        lines = "\n".join(self._contrib(s) for s in recs) or "  (no record starts here)"
        body = f"Folding {len(recs)} records into accumulator {acc}:\n{lines}\n→ accumulator = {acc_out}"
        if cut >= b:                       # final slice — return the accumulator
            return AssistantTurn(text=f"{body} (final).\n\\boxed{{{acc_out}}}", tool_calls=[])
        if n_tool == 2:                    # just read my slice — delegate the rest
            sub = self._fold_subtask(cut, b, acc_out)
            return AssistantTurn(
                text=f"{body}. Delegating the rest {cut}..{b} with accumulator {acc_out}.",
                tool_calls=[ToolCall(_new_id(), "spawn_subagent", {"subtask": sub})],
            )
        final = self._child_int(messages)  # n_tool == 3: child returned the final
        return AssistantTurn(text=f"The chain returned {final}.\n\\boxed{{{final}}}", tool_calls=[])

    # -- shared read (range + small overlap to finish the last line) ------------

    def _read(self, a, b):
        ob = min(b + self.LEAF_OVERLAP, self.doc_len)
        return AssistantTurn(
            text=f"Range {a}..{b} fits; reading it (plus overlap to finish the last line).",
            tool_calls=[
                ToolCall(_new_id(), "read_chunk", {"start": a, "end": b}),
                ToolCall(_new_id(), "read_chunk", {"start": b, "end": ob}),
            ],
        )

    def _child_int(self, messages):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        return self._extract_int(tool_msgs[-1]["content"]) if tool_msgs else 0

    @staticmethod
    def _extract_int(text: str):
        boxed = harness.extract_boxed(text or "")
        src = boxed if boxed is not None else (text or "")
        m = re.search(r"-?\d+", src)
        return int(m.group()) if m else 0
