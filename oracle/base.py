"""ScaffoldOracle — the shared recursive-decomposition scaffold for scripted oracles.

A scaffold oracle plays the canonical decomposition through `run_agent`, producing clean
SFT traces. This base owns everything task-agnostic:

- the recursion harness (root -> binary split / left-fold chain -> leaf), routed purely
  off the "tokens a..b" range in each subtask (no hidden markers);
- the grounded iterative boundary read (read the range, then keep extending while the
  last owned record's line still runs past what's been read);
- carrying a typed STATE through `\\boxed{}` and the fold subtask — int / Counter / set /
  dict — serialized by `_kind()`, finalized to the answer only at the root.

Subclasses provide the task-specific HOOKS: `_leaf_value`, `_combine`, `_fold_lines`,
`_contrib`, `_op_phrase`, `_combine_phrase`, `_finalize`, `_finalize_note`, `_kind`.
SynthOracle / RealDocOracle / BookQAOracle are peers built on this — none is privileged.
"""

from __future__ import annotations

import os
import re
import uuid
from collections import Counter

from eval.backends import AssistantTurn, ModelBackend, ToolCall

_RANGE_RE = re.compile(r"tokens (\d+)\.\.(\d+)")
# The accumulator can be a scalar OR a serialized state (counter/set/dict), so it runs to
# end-of-line; the left-fold subtask puts it on its own "accumulator so far = ..." line.
_ACC_RE = re.compile(r"accumulator so far = (.+)")


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


class ScaffoldOracle(ModelBackend):
    name = "scaffold_oracle"
    LEAF_TOKENS = int(os.environ.get("LEAF_TOKENS", "500"))
    # Left-fold uses a SMALLER leaf: a fold node lists every record in its slice inline (the
    # running accumulator), so a 500-token slice of a dense task like varchain (5 vars, verbose
    # "copied X" annotations) can push the real per-agent context past a tight budget. 400
    # keeps the heaviest fold node comfortably under 3000 incl. the tool-schema prefix.
    FOLD_LEAF_TOKENS = int(os.environ.get("FOLD_LEAF_TOKENS", "400"))
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
        self.spans = self.meta["record_spans"]   # (tok_start, tok_end, idx, *fields)
        self.strategy = strategy or self.meta.get("strategy_default", "binary")
        self.budget = budget

    def _recs_in(self, a: int, b: int) -> list[tuple]:
        """Records whose line STARTS in [a, b), in document order."""
        return [s for s in self.spans if a <= s[0] < b]

    # -- task-specific HOOKS (subclass overrides) -------------------------------

    def _kind(self) -> str:
        """State kind: 'int' | 'counter' | 'set' | 'dict'."""
        return "int"

    def _identity(self):
        return {"int": 0, "counter": Counter(), "set": set(), "dict": {}}[self._kind()]

    def _acc_step(self, acc, s):
        """Accumulate ONE record into `acc`, returning (new_acc, display_line) where the
        line shows the record AND the running state after it. This is THE per-record op —
        the leaf's aggregation and the left-fold step are the same accumulation, so the
        model never has to compute a total over many records in one shot."""
        raise NotImplementedError

    def _accumulate(self, recs, acc):
        """Left-accumulate `recs` into `acc` -> (final_acc, lines). Shared by the binary
        leaf (acc = identity, result = the partial) and the left-fold node (acc = incoming)."""
        lines = []
        for s in recs:
            acc, line = self._acc_step(acc, s)
            lines.append(line)
        return acc, lines

    def _combine(self, states):
        return sum(s for s in states if s is not None)   # default: scalar sum

    def _finalize(self, state) -> str:
        return "0" if state is None else str(state)

    def _finalize_note(self, state) -> str:
        return ""

    def _op_phrase(self) -> str:
        return "process the records"

    def _unit(self) -> str:
        return "records"        # what the spans represent (records / occurrences / sentences)

    def _goal_phrase(self) -> str:
        return "the result"     # noun phrase: what a node computes over its range

    def _verb(self) -> str:
        """Verb applied to the goal in the subtask/preamble. Reductions "compute"; retrieval
        tasks override with "find"/"collect" — run 4 showed a retrieval question phrased as
        "compute the hidden number" gets routed into the numeric-sum template (leaves ADDED the
        magic numbers), so the verb has to separate the two families."""
        return "compute"

    def _sequential_reason(self) -> str:
        """Why this task is order-dependent (fold tasks only) — one clause, read in the preamble
        after 'This task is order-dependent:'."""
        return "each record's effect depends on the state left by the records before it"

    def _state_format(self) -> str:
        """The exact shape a node must box its partial in. Stated by the ROOT in every binary
        subtask, so all siblings serialize the same way and the parent's merge is mechanical
        (run 5: OOLONG temporal children invented three different month/label key formats and the
        merge overflowed). Default by state kind; tasks with structured keys override."""
        k = self._kind()
        if k == "int":     return "the partial as a single integer"
        if k == "counter": return f"the partial tally as `<key>:<count>` entries joined by `|`{self._key_space()}"
        if k == "set":     return "the collected values joined by `|`"
        return f"the partial as `<key>=<value>` entries joined by `|`{self._key_space()}"

    def _empty_partial(self) -> str:
        """Appended to the contract: what a range with no owned items boxes. Audit 2026-09-20: sibling
        leaves boxed `negative:3|positive:6` and `none` under the same contract, which never defined the
        empty case. Scalar states box their identity (0), so only collection states need the clause."""
        return "; `none` if no " + self._unit_singular() + " starts in the range" if self._kind() != "int" else ""

    def _unit_singular(self) -> str:
        u = self._unit()
        if u.endswith("ies"): return u[:-3] + "y"
        if u.endswith("s"): return u[:-1]
        return u

    def _n_units(self, n: int) -> str:
        # No count in leaf headers: the header is written BEFORE the per-item lines, so a stated number was a silent
        # count the listing then had to honour (17w @40K: 9% of OOLONG leaves listed a different number of items than
        # their header, dropping items). The count now emerges from the item-by-item listing. `n` kept for callers.
        return self._unit()

    def _key_space(self) -> str:
        """Appended to the format contract: the allowed KEYS. Closed label sets are enumerated
        ("where `<key>` is exactly one of K1, K2, K3, K4"); open sets state what a key is ("each
        `<key>` is the word exactly as written in the text"). Run 6 (OOLONG negation): the root
        fixed the shape but not the keys, and sibling leaves invented `none`, `unrelated`,
        `not relevant` — the merge summed incompatible vocabularies. Default: no clause."""
        return ""

    def _shape_reason(self) -> str:
        """One sentence the ROOT says about WHY the state (binary partial / fold accumulator) has its shape — which field(s) the question
        groups by ("The question compares labels within each month and the items carry dates, so the
        state is a per-(month, label) tally"). Same device as the fold/binary reason: the run-7/8w
        roots still chose a 1-D per-label tally for per-month questions (OOLONG temporal/yahoo) because
        shape selection was an imitated template with no stated inference behind it. Default: none."""
        return ""

    def _combine_phrase(self) -> str:
        return "combine"

    def _empty_phrase(self) -> str:
        return "  (nothing starts here)"

    def _partial_header(self, a, b, n) -> str:
        return (f"Computing the partial over the {self._n_units(n)} whose line STARTS in {a}..{b} "
                f"({self._op_phrase()}; the trailing reads only finish the last line, and any "
                f"record starting at/after {b} belongs to the next range)")

    def _fold_header(self, n, a, cut) -> str:
        return f"Folding the {self._n_units(n)} whose line STARTS in {a}..{cut} into the accumulator (in order)"

    # Boundary-read wording. Records are one line by default; a task whose record is a
    # multi-line block (longrec) overrides these so the read narration matches its layout.
    def _finish_phrase(self, end) -> str:
        return f"to finish the last record (its line may run past {end})"

    def _cutoff_phrase(self, covered) -> str:
        return f"The last record's line is still cut off at {covered} (no end-of-line yet)"

    # -- typed state <-> string (carried through \boxed{} and the fold subtask) --

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
            m = re.search(r"-?\d+", s)
            return int(m.group()) if m else self._identity()   # "none" -> identity (None for max/min)
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

    # -- subtask construction (carries strategy + range [+ accumulator]) --------

    def _binary_subtask(self, a: int, b: int) -> str:
        # Self-similar directive (no conditional): the same instruction works at every
        # node — split-or-leaf is decided from the range, and the model learns the one
        # handoff shape. The midpoint isn't stated; the parent's two spawn calls show it.
        L, goal, unit, v = self.LEAF_TOKENS, self._goal_phrase(), self._unit(), self._verb()
        return (
            f"Over the {unit} STARTING in tokens {a}..{b}, {v} {goal}. Recursively split "
            f"the range at its midpoint, delegating each half to a subagent, until the range "
            f"is less than {L} tokens. When the range is less than {L} tokens, read it and "
            f"{v} {goal} over the {unit} STARTING in the range. Put the result in \\boxed{{}}: "
            f"{self._state_format()}{self._empty_partial()}."
        )

    def _fold_subtask(self, a: int, b: int, acc) -> str:
        L, unit = self.FOLD_LEAF_TOKENS, self._unit()
        return (
            f"Continue the running accumulator by {self._op_phrase()}, "
            f"over the {unit} STARTING in tokens {a}..{b}.\n"
            f"accumulator so far = {self._ser_state(acc)}\n"
            f"Read the first {L} tokens and update the accumulator over the {unit} STARTING "
            f"in tokens {a}..{b}, then delegate the rest of the range to one subagent with the "
            f"updated accumulator. When the range is less than {L} tokens, process it directly "
            f"and return the final accumulator."
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

    # -- root: the top node, demonstrates the operation + finalizes -------------

    def _root(self, messages):
        # The root IS the top node — it demonstrates the actual operation (binary split of
        # the whole doc, or the first left-fold step), not a wrapper that hands the whole
        # range to one child. It alone FINALIZES the combined state into the boxed answer.
        nr, ns = self._reads_spawns(messages)
        if self.strategy == "left_fold":
            turn = self._fold_node(0, self.doc_len, self._identity(), messages, is_root=True)
        else:
            turn = self._binary_node(0, self.doc_len, messages, is_root=True)
        # On its FIRST turn the root states the plan: it alone has to read the question, turn
        # it into the per-node subtask, and finalize the combined result — so it reasons about
        # the whole strategy once, before acting. Internal nodes stay terse (they just execute).
        if nr == 0 and ns == 0:
            turn.text = f"{self._strategy_preamble()}\n{turn.text}"
        return turn

    def _strategy_preamble(self) -> str:
        n = self.doc_len
        # The strategy is a property of the task, and the root SAYS WHY it picks one: sequential
        # (order-dependent) tasks fold, order-independent tasks split. Stating the reason is what
        # turns the correlation into a rule the model can apply to an unseen question.
        #
        # ORDER OF COMMITMENT (runs 8w/9w/10w). An autoregressive root commits in token order, so
        # sentence two IS the fold-vs-binary decision. 8w: sentence two was task-specific in both
        # strategies (order relation fused with the split/fold plan and the GOAL noun) and there was
        # no separate state sentence — vt 0.88. 9w put a state sentence BEFORE the plan with a per-task
        # unit noun; the root filled both from the question's surface on RULER vt (filter state) —
        # 0.16. 10w made sentence two a fixed generic sentence shared by 71% of roots; it became the
        # zero-effort continuation and 4/5 vt roots went binary — 0.36. So: sentence two names the
        # task's order relation AND the goal (no generic slot, no deferral), and the shape reasoning
        # follows as justification of the goal already named — it explains the commitment, it cannot
        # redirect it.
        # Run 11: PURE 8w preamble — no state-reasoning sentence at all. The shape reasoning (9w/10w)
        # never showed a measurable gain (temporal 0.25 → 0.37 → 0.21 over three runs of five seeds) and
        # cost fold on vt whenever it preceded the commitment. Keep the root simple; the state is named
        # in the goal phrase. `_shape_reason` hooks stay in the oracles for a later experiment.
        shape = ""
        if self.strategy == "left_fold":
            return (
                f"This document is {n} tokens — too long to read in one context. This task is "
                f"order-dependent: {self._sequential_reason()} — so I process the document from the "
                f"start with a running accumulator: read the first slice and update the accumulator by "
                f"{self._op_phrase()}, then hand the rest of the document plus the accumulator to a "
                f"subagent to continue the same way.{shape} Once the final slice is folded in, the "
                f"accumulator holds the whole-document result and I turn it into the final answer."
            )
        return (
            f"This document is {n} tokens — too long to read in one context. The result over a range "
            f"does not depend on the order of the {self._unit()}, so I split the range in half "
            f"recursively, having a subagent {self._verb()} {self._goal_phrase()} over each half and "
            f"combining the two partials.{shape} Once I have it for the whole document, I turn the "
            f"combined result into the final answer."
        )

    def _result_text(self, state, is_root):
        """A leaf's / merge's result, written ONCE (v17). Non-root: the state goes straight into \\boxed{} — it used to
        be restated first ("-> X." then "\\boxed{X}"), the same X twice; 41 of 49 of 18w's OOLONG merge overflows @40K/80K
        ran out while writing that second copy. The root keeps "X." + its finalize note, then boxes the ANSWER."""
        if is_root:
            return f"{self._ser_state(state)}.{self._box_text(state, is_root)}"
        return f"\\boxed{{{self._ser_state(state)}}}"

    def _box_text(self, state, is_root):
        """The boxed suffix. Internal nodes box the serialized STATE (so the parent can keep
        combining); the root SHOWS its finalize step then boxes the answer — so the root
        never silently conjures it."""
        if is_root:
            return f"{self._finalize_note(state)}\n\\boxed{{{self._finalize(state)}}}"
        return f"\n\\boxed{{{self._ser_state(state)}}}"

    # -- grounded iterative boundary read (generalizes to any record length) ----

    def _read_phase(self, a, end, messages, node_end=None):
        """Return the next read turn while the last owned record isn't fully covered yet,
        else None (reading done). Observe-then-extend, deciding to continue only from what
        has actually been read. `node_end` (the full node range end) is set by the left-fold
        node: when end < node_end this read is only the FIRST slice of a larger range that
        will be delegated onward — so it must NOT be narrated as a self-contained leaf."""
        ov = self.LEAF_OVERLAP
        n_reads, _ = self._reads_spawns(messages)
        recs = self._recs_in(a, end)
        last_end = min(max((s[1] for s in recs), default=end), self.doc_len)
        if n_reads == 0:
            ob = min(end + ov, self.doc_len)
            calls = [ToolCall(_new_id(), "read_chunk", {"start": a, "end": end})]
            fold_slice = node_end is not None and end < node_end
            if fold_slice:
                lead = f"Reading the first {end - a} tokens ({a}..{end}) of this range to fold"
            elif ob > end:
                lead = f"Range {a}..{end} fits; reading it"
            else:
                lead = f"Range {a}..{end} fits and reaches the document end; reading it"
            if ob > end:
                calls.append(ToolCall(_new_id(), "read_chunk", {"start": end, "end": ob}))
                txt = f"{lead}, then the next {ob - end} tokens {self._finish_phrase(end)}."
            else:
                txt = f"{lead}."
            return AssistantTurn(text=txt, tool_calls=calls)
        # first read is [a,end]; every read after it is a contiguous +ov block
        covered = min(end + (n_reads - 1) * ov, self.doc_len)
        if covered < last_end:
            nxt = min(covered + ov, self.doc_len)
            return AssistantTurn(
                text=f"{self._cutoff_phrase(covered)}, "
                     f"so I read the next {nxt - covered} tokens ({covered}..{nxt}) to finish it.",
                tool_calls=[ToolCall(_new_id(), "read_chunk", {"start": covered, "end": nxt})],
            )
        return None

    # -- binary node: split -> combine, or read -> leaf -------------------------

    def _binary_node(self, a, b, messages, is_root=False):
        nr, ns = self._reads_spawns(messages)
        # Split while the range is NOT less than LEAF_TOKENS — the same rule the subtask states
        # ("until the range is less than 500 tokens"). The old `>` disagreed with the text at exactly
        # 500, which is what a 4000-token eval doc halves to (4000→2000→1000→500).
        if (b - a) >= self.LEAF_TOKENS:
            if ns == 0:
                m = (a + b) // 2
                return AssistantTurn(
                    text=f"Range {a}..{b} is not below {self.LEAF_TOKENS}; splitting at midpoint {m}.",
                    tool_calls=[
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(a, m)}),
                        ToolCall(_new_id(), "spawn_subagent", {"subtask": self._binary_subtask(m, b)}),
                    ],
                )
            states = [self._parse_state(c) for c in self._spawn_returns(messages)]
            res = self._combine(states)
            # Scalar/set states are restated (cheap; shows the arithmetic). Counter/dict tallies are
            # NOT re-listed — they sit in the two tool results just above, and re-printing both plus
            # the merge tripled the state cost at every internal node (labeled_records date-mode 2-D
            # combines measured 3.6k real tokens; OOLONG's run-5 merge overflow was the same shape).
            if self._kind() in ("counter", "dict"):
                text = (f"{self._combine_phrase().capitalize()} my two children's tallies (above) -> "
                        f"{self._result_text(res, is_root)}")
            else:
                shown = ", ".join(self._ser_state(s) for s in states)
                text = (f"{self._combine_phrase().capitalize()} my children [{shown}] -> "
                        f"{self._result_text(res, is_root)}")
            return AssistantTurn(text=text, tool_calls=[])
        # leaf: read (iteratively, until the last owned record is covered), then accumulate
        # the records ONE AT A TIME, showing the running partial after each (never a total
        # computed in one shot).
        read_turn = self._read_phase(a, b, messages)
        if read_turn is not None:
            return read_turn
        recs = self._recs_in(a, b)
        state, acc_lines = self._accumulate(recs, self._identity())
        lines = "\n".join(acc_lines) or self._empty_phrase()
        return AssistantTurn(
            text=f"{self._partial_header(a, b, len(recs))}:\n{lines}\n"
                 f"Partial = {self._result_text(state, is_root)}",
            tool_calls=[],
        )

    # -- left-fold node: read slice (iterative) -> fold + spawn rest -> bubble ---

    def _fold_node(self, a, b, acc, messages, is_root=False):
        cut = min(a + self.FOLD_LEAF_TOKENS, b)
        read_turn = self._read_phase(a, cut, messages, node_end=b)
        if read_turn is not None:
            return read_turn
        recs = self._recs_in(a, cut)
        acc_out, fold_lines = self._accumulate(recs, acc)
        lines = "\n".join(fold_lines) or self._empty_phrase()
        # The incoming accumulator is already in this agent's user prompt and the outgoing one is
        # stated once below (and again in the spawn arg) — don't reprint it in the header: with
        # large dict states (vt_novel: ~20 five-letter var names ≈ 4 tokens each) each extra
        # reprint costs ~250 tokens and pushed fold nodes over the 3000 budget.
        if recs:
            body = (f"{self._fold_header(len(recs), a, cut)}:\n{lines}\n"
                    f"→ accumulator = {self._ser_state(acc_out)}")
        else:
            # An EMPTY slice is a normal fold step, not a stopping condition: say so explicitly
            # and keep the accumulator. Run 6 (RULER vt on a noise haystack, where most 400-token
            # slices hold no VAR line): one root re-read the next slice itself instead of
            # delegating; one child returned `none` after an empty slice instead of continuing.
            body = (f"No {self._unit()} start in {a}..{cut} — the accumulator is unchanged.\n"
                    f"→ accumulator = {self._ser_state(acc_out)}")
        if cut >= b:                            # final slice — return the accumulator/answer
            return AssistantTurn(text=f"{body} (final).{self._box_text(acc_out, is_root)}", tool_calls=[])
        _, ns = self._reads_spawns(messages)
        if ns == 0:                             # reads done, not yet delegated — delegate the rest
            sub = self._fold_subtask(cut, b, acc_out)
            return AssistantTurn(
                text=f"{body}. Delegating the rest {cut}..{b} with the "
                     f"{'unchanged' if not recs else 'updated'} accumulator.",
                tool_calls=[ToolCall(_new_id(), "spawn_subagent", {"subtask": sub})],
            )
        # child returned the chain's final accumulator (serialized); finalize iff root
        final_state = self._parse_state(self._spawn_returns(messages)[-1])
        return AssistantTurn(
            # Don't restate the returned state in prose — it's in the tool result just above and
            # (for internal nodes) in the box below; the duplicate cost ~1x the accumulator.
            text=f"The chain returned its final accumulator.{self._box_text(final_state, is_root)}",
            tool_calls=[],
        )
