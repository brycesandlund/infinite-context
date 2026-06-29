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

    def _acc_step(self, acc, s):
        """One record -> (new_acc, line showing the running state). The binary leaf calls
        this from identity (so its partial is built one record at a time, never summed in
        one shot); the left-fold calls it threading the incoming accumulator."""
        t = self.task
        if t == "synth_varchain":
            _, _, idx, var, rhs, is_ref = s
            if is_ref:
                val = acc.get(rhs, 0)
                return {**acc, var: val}, f"- [{idx:04d}] set {var} = {rhs}  → {var}={val} (copied {rhs}'s current value)"
            return {**acc, var: int(rhs)}, f"- [{idx:04d}] set {var} = {rhs}  → {var}={rhs}"
        idx, amt, flag, grp = s[2], s[3], s[4], s[5]
        if t == "synth_sum":
            acc += amt
            return acc, f"- [{idx:04d}] amt={amt:+d}  → sum={acc}"
        if t == "synth_count":
            if flag == "Y": acc += 1
            return acc, f"- [{idx:04d}] flag={flag}  → count={acc}"
        if t == "synth_sumwhere":
            mark = "(skip)"
            if flag == "Y": acc += amt; mark = "(add)"
            return acc, f"- [{idx:04d}] flag={flag} amt={amt:+d} {mark}  → sum={acc}"
        if t == "synth_max":
            acc = amt if acc is None else max(acc, amt)
            return acc, f"- [{idx:04d}] amt={amt:+d}  → max={acc}"
        if t == "synth_min":
            acc = amt if acc is None else min(acc, amt)
            return acc, f"- [{idx:04d}] amt={amt:+d}  → min={acc}"
        if t == "synth_mode":
            if grp != "RST": acc = acc + Counter([grp])
            return acc, f"- [{idx:04d}] grp={grp}" + ("  (ignore RST)" if grp == "RST" else "") + f"  → {self._ser_state(acc)}"
        if t == "synth_distinct":
            if grp != "RST": acc = acc | {grp}
            return acc, f"- [{idx:04d}] grp={grp}" + ("  (ignore RST)" if grp == "RST" else "") + f"  → {self._ser_state(acc)}"
        if t == "synth_runreset":
            if grp == "RST": return 0, f"- [{idx:04d}] grp=RST  → RESET, total=0"
            acc += amt
            return acc, f"- [{idx:04d}] amt={amt:+d}  → total={acc}"
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

    def _op_phrase(self) -> str:
        # Imperative description: the binary entries narrate the leaf header; the two
        # left-fold entries (runreset/varchain) are the gerund fold-step in the subtask.
        return {
            "synth_sum": "add up the 'amt' fields",
            "synth_count": "count the records with flag=Y",
            "synth_max": "take the maximum 'amt'",
            "synth_min": "take the minimum 'amt'",
            "synth_sumwhere": "sum the 'amt' of records with flag=Y",
            "synth_mode": "tally each grp value (ignoring RST)",
            "synth_distinct": "collect the distinct grp values (ignoring RST)",
            "synth_runreset": "adding each 'amt' to the running total, resetting the total to 0 at each grp=RST",
            "synth_varchain": "applying each assignment in order (a `= VAR` copies that variable's current value)",
        }[self.task]

    def _goal_phrase(self) -> str:
        # Noun goal a node computes over its range (binary tasks only; the fold tasks use
        # the running accumulator instead). Reads as "... compute {goal} over the records ...".
        return {
            "synth_sum": "the SUM of 'amt'",
            "synth_count": "how many have flag=Y",
            "synth_max": "the MAXIMUM 'amt'",
            "synth_min": "the MINIMUM 'amt'",
            "synth_sumwhere": "the SUM of 'amt' (flag=Y records only)",
            "synth_mode": "the per-grp tally (how many have each grp value, ignoring RST)",
            "synth_distinct": "the set of distinct grp values (ignoring RST)",
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
