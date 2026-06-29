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
from collections import Counter

import harness
from eval.backends import AssistantTurn, ModelBackend, ToolCall

_RANGE_RE = re.compile(r"tokens (\d+)\.\.(\d+)")
# accumulator can now be a scalar OR a serialized state (counter/set/dict), so it runs
# to end-of-line; the left-fold subtask puts it on its own "accumulator so far = ..." line.
_ACC_RE = re.compile(r"accumulator so far = (.+)")
_GROUPS_DEFAULT = "K1"


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


class SynthOracle(ModelBackend):
    name = "synth_oracle"
    LEAF_TOKENS = int(os.environ.get("LEAF_TOKENS", "500"))
    # One finish-read block. The leaf reads [a,b]+[b,b+overlap] and KEEPS extending by
    # another block while the last owned record still runs past what it has read — so it
    # generalizes to any record length / document type (an OOLONG example can be ~450
    # tokens). 200 keeps most records to one extra read; long ones trigger the extension.
    LEAF_OVERLAP = int(os.environ.get("LEAF_OVERLAP", "200"))

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        self.doc_len = len(problem.document_tokens)
        self.tok = tokenizer
        self.task = problem.task
        self.meta = dict(problem.metadata)
        self.spans = self.meta["record_spans"]   # (start, end, idx, amt, flag, grp)
        self.strategy = strategy or self.meta.get("strategy_default", "binary")
        self.budget = budget
        self.query_var = self.meta.get("query_var")   # synth_varchain: which var to report
        if self.strategy == "binary" and self.task in ("synth_runreset", "synth_varchain"):
            # non-associative combines; a binary tree would need a richer monoid. Not
            # rendered yet — these are left-fold tasks.
            raise ValueError(f"{self.task} has no binary oracle; use left_fold")

    # -- per-task leaf-op / combine / fold --------------------------------------

    def _recs_in(self, a: int, b: int) -> list[tuple]:
        """Records whose line STARTS in [a, b), in document order."""
        return [s for s in self.spans if a <= s[0] < b]

    # State kind per task — how the partial/accumulator is carried through \boxed{} and
    # subtasks: int (scalar reduce/fold), counter (per-key tally -> argmax), set (distinct
    # -> count), dict (variable bindings). The leaf makes a state, combine/fold merges it,
    # and only the ROOT finalizes it to the boxed answer.
    _KIND = {
        "synth_sum": "int", "synth_count": "int", "synth_max": "int", "synth_min": "int",
        "synth_sumwhere": "int", "synth_runreset": "int",
        "synth_mode": "counter", "synth_distinct": "set", "synth_varchain": "dict",
    }

    def _kind(self):
        return self._KIND.get(self.task, "int")

    def _identity(self):
        if self.task in ("synth_max", "synth_min"):
            return None
        return {"int": 0, "counter": Counter(), "set": set(), "dict": {}}[self._kind()]

    def _leaf_value(self, recs):
        t = self.task
        if t == "synth_sum":      return sum(s[3] for s in recs)
        if t == "synth_count":    return sum(1 for s in recs if s[4] == "Y")
        if t == "synth_sumwhere": return sum(s[3] for s in recs if s[4] == "Y")
        if t == "synth_max":      return max((s[3] for s in recs), default=None)
        if t == "synth_min":      return min((s[3] for s in recs), default=None)
        if t == "synth_mode":     return Counter(s[5] for s in recs if s[5] != "RST")
        if t == "synth_distinct": return {s[5] for s in recs if s[5] != "RST"}
        raise ValueError(t)

    def _combine(self, states):
        t = self.task
        if t == "synth_max":
            v = [s for s in states if s is not None]; return max(v) if v else None
        if t == "synth_min":
            v = [s for s in states if s is not None]; return min(v) if v else None
        if t == "synth_mode":
            tot = Counter()
            for s in states: tot += s
            return tot
        if t == "synth_distinct":
            out = set()
            for s in states: out |= s
            return out
        return sum(s for s in states if s is not None)   # sum / count / sumwhere

    def _fold(self, recs, acc):
        t = self.task
        for s in recs:
            if t == "synth_varchain":
                _, _, _, var, rhs, is_ref = s
                acc = dict(acc); acc[var] = acc.get(rhs, 0) if is_ref else int(rhs)
                continue
            amt, flag, grp = s[3], s[4], s[5]
            if t == "synth_runreset": acc = 0 if grp == "RST" else acc + amt
            elif t == "synth_sum":    acc += amt
            elif t == "synth_count":  acc += 1 if flag == "Y" else 0
        return acc

    def _contrib(self, s) -> str:
        t = self.task
        if t == "synth_varchain":
            _, _, idx, var, rhs, is_ref = s
            return f"- [{idx:04d}] set {var} = {rhs}" + ("  (copy)" if is_ref else "")
        idx, amt, flag, grp = s[2], s[3], s[4], s[5]
        if t == "synth_count":
            return f"- [{idx:04d}] flag={flag}" + ("  (+1)" if flag == "Y" else "")
        if t == "synth_sumwhere":
            return f"- [{idx:04d}] flag={flag} amt={amt:+d}" + ("  (+)" if flag == "Y" else "  (skip)")
        if t == "synth_runreset":
            return f"- [{idx:04d}] grp={grp} amt={amt:+d}" + ("  -> RESET to 0" if grp == "RST" else "")
        if t in ("synth_mode", "synth_distinct"):
            return f"- [{idx:04d}] grp={grp}" + ("  (ignore RST)" if grp == "RST" else "")
        return f"- [{idx:04d}] amt={amt:+d}"

    def _op_phrase(self) -> str:
        return {
            "synth_sum": "add up the 'amt' fields",
            "synth_count": "count the records with flag=Y",
            "synth_max": "take the maximum 'amt'",
            "synth_min": "take the minimum 'amt'",
            "synth_sumwhere": "sum the 'amt' of records with flag=Y",
            "synth_mode": "tally each grp value (ignoring RST)",
            "synth_distinct": "collect the distinct grp values (ignoring RST)",
            "synth_runreset": "fold 'amt' left-to-right, resetting to 0 on grp=RST",
            "synth_varchain": "apply each assignment to the running variable bindings",
        }[self.task]

    def _combine_phrase(self) -> str:
        return {
            "synth_max": "take the max of", "synth_min": "take the min of",
            "synth_mode": "merge the tallies of", "synth_distinct": "union the distinct-sets of",
        }.get(self.task, "sum")

    def _empty_phrase(self) -> str:
        return "  (no record starts here)"

    # -- state <-> string (carried through \boxed{} and the fold subtask) -------

    def _ser_state(self, state) -> str:
        if state is None:
            return "none"
        k = self._kind()
        if k == "int":     return str(state)
        if k == "counter": return "|".join(f"{g}:{c}" for g, c in sorted(state.items())) or "none"
        if k == "set":     return "|".join(sorted(state)) or "none"
        return "|".join(f"{v}={x}" for v, x in sorted(state.items())) or "none"   # dict

    def _parse_state(self, s):
        k = self._kind(); s = (s or "").strip()
        if k == "int":
            if self.task in ("synth_max", "synth_min") and "none" in s.lower():
                return None
            m = re.search(r"-?\d+", s); return int(m.group()) if m else self._identity()
        if s.lower() == "none":
            return self._identity()
        if k == "counter":
            c = Counter()
            for p in s.split("|"):
                g, sep, n = p.partition(":")
                if sep:
                    try: c[g.strip()] += int(re.search(r"-?\d+", n).group())
                    except (ValueError, AttributeError): pass
            return c
        if k == "set":
            return {p.strip() for p in s.split("|") if p.strip()}
        d = {}
        for p in s.split("|"):
            v, sep, x = p.partition("=")
            if sep:
                try: d[v.strip()] = int(re.search(r"-?\d+", x).group())
                except (ValueError, AttributeError): pass
        return d

    def _finalize(self, state) -> str:
        t = self.task
        if t == "synth_mode":
            if not state: return _GROUPS_DEFAULT
            mx = max(state.values()); return min(g for g, n in state.items() if n == mx)
        if t == "synth_distinct":
            return str(len(state))
        if t == "synth_varchain":
            return str(state.get(self.query_var, 0))
        return "0" if state is None else str(state)

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
        return (
            f"Strategy: LEFT-FOLD. Process tokens {a}..{b} left-to-right "
            f"(leaf-op: {self._op_phrase()}).\n"
            f"accumulator so far = {self._ser_state(acc)}\n"
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
        rng = _RANGE_RE.search(user)
        if rng is None:
            return self._root(messages)
        a, b = int(rng.group(1)), int(rng.group(2))
        if self.strategy == "left_fold":
            return self._fold_node(a, b, self._parse_acc(user), messages)
        return self._binary_node(a, b, messages)

    def _parse_acc(self, user: str):
        m = _ACC_RE.search(user)
        return self._parse_state(m.group(1).strip()) if m else self._identity()

    @staticmethod
    def _reads_spawns(messages):
        """Count read_chunk vs spawn_subagent tool results separately — a fold node has
        BOTH (its reads, then its one child spawn), so a single tool-count can't tell the
        read phase from the post-read phase."""
        nr = ns = 0
        for m in messages:
            if m.get("role") != "tool":
                continue
            if m.get("name") == "read_chunk":
                nr += 1
            elif m.get("name") == "spawn_subagent":
                ns += 1
        return nr, ns

    def _spawn_returns(self, messages):
        return [m["content"] for m in messages if m.get("role") == "tool" and m.get("name") == "spawn_subagent"]

    # -- root -------------------------------------------------------------------

    def _root(self, messages):
        # The root IS the top node — it demonstrates the actual operation (binary split of
        # the whole doc, or the first left-fold step), not a wrapper that hands the whole
        # range to one child. It alone FINALIZES the combined state into the boxed answer.
        if self.strategy == "left_fold":
            return self._fold_node(0, self.doc_len, self._identity(), messages, is_root=True)
        return self._binary_node(0, self.doc_len, messages, is_root=True)

    def _box_text(self, state, is_root):
        """The boxed suffix. Internal nodes box the serialized STATE (so the parent can
        keep combining); the root SHOWS its finalize step (argmax / count / var lookup)
        then boxes the finalized answer — so the root never silently conjures it."""
        if is_root:
            return f"{self._finalize_note(state)}\n\\boxed{{{self._finalize(state)}}}"
        return f"\n\\boxed{{{self._ser_state(state)}}}"

    def _finalize_note(self, state) -> str:
        t = self.task
        if t == "synth_mode":
            if not state:
                return ""
            mx = max(state.values())
            return f"\nMost frequent grp: {min(g for g, n in state.items() if n == mx)} ({mx})."
        if t == "synth_distinct":
            return f"\nDistinct grp values: {len(state)}."
        if t == "synth_varchain":
            return f"\nFinal value of {self.query_var}: {state.get(self.query_var, 0)}."
        return ""   # numeric reduce/fold: the boxed answer IS the state

    # -- grounded iterative boundary read (generalizes to any record length) ----

    def _read_phase(self, a, end, messages):
        """Return the next read turn while the last owned record isn't fully covered yet,
        else None (reading done). Observe-then-extend, deciding to continue only from what
        has actually been read: read [a,end] + [end,end+ov], then keep reading +ov blocks
        while the last record whose line starts in [a,end) still runs past what we've read."""
        ov = self.LEAF_OVERLAP
        n_reads, _ = self._reads_spawns(messages)
        recs = self._recs_in(a, end)
        last_end = min(max((s[1] for s in recs), default=end), self.doc_len)
        if n_reads == 0:
            ob = min(end + ov, self.doc_len)
            calls = [ToolCall(_new_id(), "read_chunk", {"start": a, "end": end})]
            if ob > end:
                calls.append(ToolCall(_new_id(), "read_chunk", {"start": end, "end": ob}))
                txt = (f"Range {a}..{end} fits; reading it, then the next {ob - end} tokens to "
                       f"finish the last record (its line may run past {end}).")
            else:
                txt = f"Range {a}..{end} fits and reaches the document end; reading it."
            return AssistantTurn(text=txt, tool_calls=calls)
        # first read is [a,end]; every read after it is a contiguous +ov block
        covered = min(end + (n_reads - 1) * ov, self.doc_len)
        if covered < last_end:
            nxt = min(covered + ov, self.doc_len)
            return AssistantTurn(
                text=f"The last record's line is still cut off at {covered} (no end-of-line yet), "
                     f"so I read the next {nxt - covered} tokens ({covered}..{nxt}) to finish it.",
                tool_calls=[ToolCall(_new_id(), "read_chunk", {"start": covered, "end": nxt})],
            )
        return None

    # -- binary node ------------------------------------------------------------

    def _binary_node(self, a, b, messages, is_root=False):
        nr, ns = self._reads_spawns(messages)
        if (b - a) > self.LEAF_TOKENS:
            if ns == 0:
                m = (a + b) // 2
                return AssistantTurn(
                    text=f"Range {a}..{b} > {self.LEAF_TOKENS}; splitting at midpoint {m}.",
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(a, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(m, b)}),
                    ],
                )
            states = [self._parse_state(c) for c in self._spawn_returns(messages)]
            res = self._combine(states)
            shown = ", ".join(self._ser_state(s) for s in states)
            return AssistantTurn(
                text=f"{self._combine_phrase().capitalize()} my children [{shown}] -> "
                     f"{self._ser_state(res)}.{self._box_text(res, is_root)}",
                tool_calls=[],
            )
        # leaf: read (iteratively, until the last owned record is covered), then compute
        read_turn = self._read_phase(a, b, messages)
        if read_turn is not None:
            return read_turn
        recs = self._recs_in(a, b)
        lines = "\n".join(self._contrib(s) for s in recs) or self._empty_phrase()
        state = self._leaf_value(recs)
        return AssistantTurn(
            text=f"{self._partial_header(a, b, len(recs))}:\n{lines}\n"
                 f"Partial = {self._ser_state(state)}.{self._box_text(state, is_root)}",
            tool_calls=[],
        )

    def _partial_header(self, a, b, n) -> str:
        return (f"Computing the partial over the {n} records whose line STARTS in {a}..{b} "
                f"({self._op_phrase()}; the trailing reads only finish the last line, and any "
                f"record starting at/after {b} belongs to the next range)")

    # -- left-fold node: read slice (iterative) -> fold + spawn rest -> bubble ---

    def _fold_node(self, a, b, acc, messages, is_root=False):
        cut = min(a + self.LEAF_TOKENS, b)
        read_turn = self._read_phase(a, cut, messages)
        if read_turn is not None:
            return read_turn
        recs = self._recs_in(a, cut)
        acc_out = self._fold(recs, acc)
        lines = "\n".join(self._contrib(s) for s in recs) or self._empty_phrase()
        body = (f"Folding the {len(recs)} records whose line STARTS in {a}..{cut} into "
                f"accumulator {self._ser_state(acc)} (in order):\n{lines}\n"
                f"→ accumulator = {self._ser_state(acc_out)}")
        if cut >= b:                            # final slice — return the accumulator/answer
            return AssistantTurn(text=f"{body} (final).{self._box_text(acc_out, is_root)}", tool_calls=[])
        _, ns = self._reads_spawns(messages)
        if ns == 0:                             # reads done, not yet delegated — delegate the rest
            sub = self._fold_subtask(cut, b, acc_out)
            return AssistantTurn(
                text=f"{body}. Delegating the rest {cut}..{b} with accumulator {self._ser_state(acc_out)}.",
                tool_calls=[ToolCall(_new_id(), "spawn_subagent", {"subtask": sub})],
            )
        # child returned the chain's final accumulator (serialized); finalize iff root
        final_state = self._parse_state(self._spawn_returns(messages)[-1])
        return AssistantTurn(
            text=f"The chain returned {self._ser_state(final_state)}.{self._box_text(final_state, is_root)}",
            tool_calls=[],
        )


class RealDocOracle(SynthOracle):
    """realdoc_count over real novel prose. Each 'record' is one occurrence of the
    queried word (contributing +1); the binary tree-reduce sums per-chunk counts — the
    same bounded-associative combine as synth_count, but the leaf-op is 'find the word
    in real prose'. Reuses SynthOracle's scaffold + iterative boundary read wholesale."""

    name = "realdoc_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy="binary")
        self.entity = self.meta["entity"]

    def _leaf_value(self, recs):
        return len(recs)                       # every occurrence counts +1

    def _op_phrase(self):
        return f"count the occurrences of the word '{self.entity}'"

    def _contrib(self, s):
        idx, snip = s[2], s[3]
        return f"- occurrence {idx}: …{snip}…"

    def _partial_header(self, a, b, n) -> str:
        return (f"Counting the {n} occurrences of '{self.entity}' that START in tokens {a}..{b} "
                f"(the trailing reads only finish an occurrence straddling {b}; an occurrence "
                f"starting at/after {b} belongs to the next range)")

    def _empty_phrase(self) -> str:
        return f"  (no occurrences of '{self.entity}' start here)"


class BookQAOracle(SynthOracle):
    """Real-question QA over a real (anonymized) novel. Binary scan; the leaf retrieves
    sentences mentioning the question's entities; the combine keeps the top-K most-relevant
    evidence sentences (BOUNDED -> no overflow); the root reads the collected evidence and
    emits the gold answer (an entity sentence contains it). Reuses SynthOracle's scaffold +
    iterative read; only the combine (collect-top-K) and the root (answer) differ."""

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

    # -- root: collect evidence, then ANSWER (not box evidence) -----------------

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

    def _binary_node(self, a, b, messages):
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
