"""SynthOracle — the abstract synthetic decomposition tasks (tasks/synth).

Nine operations across both strategies and four combine-state kinds, all expressed as
hooks on ScaffoldOracle:
  bounded-associative (binary):  sum / count / max / min / sumwhere (int reduce),
                                 mode (Counter -> argmax), distinct (set -> count)
  stateful-sequential (left-fold): runreset (running total), varchain (variable bindings)
The strategy is the per-task TRAINING knob (mixed default, or forced all-binary/all-fold).
"""

from __future__ import annotations

from collections import Counter

from oracle.base import ScaffoldOracle

_GROUPS_DEFAULT = "K1"


class SynthOracle(ScaffoldOracle):
    name = "synth_oracle"

    # State kind per task: int (scalar reduce/fold), counter (per-key tally -> argmax),
    # set (distinct -> count), dict (variable bindings).
    _KIND = {
        "synth_sum": "int", "synth_count": "int", "synth_max": "int", "synth_min": "int",
        "synth_sumwhere": "int", "synth_runreset": "int",
        "synth_mode": "counter", "synth_distinct": "set", "synth_varchain": "dict",
    }

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.query_var = self.meta.get("query_var")   # synth_varchain: which var to report
        if self.strategy == "binary" and self.task in ("synth_runreset", "synth_varchain"):
            # non-associative combines; a binary tree would need a richer monoid. Not
            # rendered yet — these are left-fold tasks.
            raise ValueError(f"{self.task} has no binary oracle; use left_fold")

    def _kind(self):
        return self._KIND.get(self.task, "int")

    def _identity(self):
        if self.task in ("synth_max", "synth_min"):
            return None
        return super()._identity()

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

    def _fold_lines(self, recs, acc):
        """Step-by-step left fold: (final_acc, display_lines) where each line shows a record
        AND the running state after it, so the fold is AUDITABLE. For varchain a `= VAR`
        copy shows the RESOLVED value (the source's current binding)."""
        lines = []
        for s in recs:
            if self.task == "synth_varchain":
                _, _, idx, var, rhs, is_ref = s
                if is_ref:
                    val = acc.get(rhs, 0)
                    acc = {**acc, var: val}
                    lines.append(f"- [{idx:04d}] set {var} = {rhs}  → {var}={val} (copied {rhs}'s current value)")
                else:
                    acc = {**acc, var: int(rhs)}
                    lines.append(f"- [{idx:04d}] set {var} = {rhs}  → {var}={rhs}")
            else:  # synth_runreset
                idx, amt, grp = s[2], s[3], s[5]
                if grp == "RST":
                    acc = 0
                    lines.append(f"- [{idx:04d}] grp=RST  → RESET, total=0")
                else:
                    acc += amt
                    lines.append(f"- [{idx:04d}] amt={amt:+d}  → total={acc}")
        return acc, lines

    def _contrib(self, s) -> str:
        t = self.task
        idx, amt, flag, grp = s[2], s[3], s[4], s[5]
        if t == "synth_count":
            return f"- [{idx:04d}] flag={flag}" + ("  (+1)" if flag == "Y" else "")
        if t == "synth_sumwhere":
            return f"- [{idx:04d}] flag={flag} amt={amt:+d}" + ("  (+)" if flag == "Y" else "  (skip)")
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
        # Must read naturally in BOTH "then {phrase} their two results" (subtask) and
        # "{Phrase} my children [...]" (combine display).
        return {
            "synth_max": "take the max of", "synth_min": "take the min of",
            "synth_mode": "merge", "synth_distinct": "union",
        }.get(self.task, "sum")

    def _empty_phrase(self) -> str:
        return "  (no record starts here)"

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
