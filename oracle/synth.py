"""SynthOracle — the abstract synthetic decomposition tasks (tasks/synth).

Seventeen operations across both strategies and four combine-state kinds, all expressed as
hooks on ScaffoldOracle:
  bounded-associative (binary):  sum / count / max / min / sumwhere (int reduce),
                                 count2 / count_cmp / count_range / maxwhere (parameterized
                                 predicate reduce), mode (Counter -> argmax),
                                 distinct (set -> count), sumby / diff (per-key dict -> argmax/subtract)
  stateful-sequential (left-fold): runreset (running total), varchain (variable bindings)
  month-record families:         filter_argmax (filter on flag/mon -> per-grp Counter -> argmax/argmin),
                                 2d (joint mon/grp dict tally -> reduce along one axis: count months
                                 where grp G wins / beats G2, argmax grp within a month, argmax month
                                 for a grp). The question variant is drawn per problem (metadata qtype).
The combining/temporal tasks (count2, diff, sumby, maxwhere, count_cmp, count_range) fuse
two record labels or a numeric range into one predicate. The strategy is the per-task
TRAINING knob (mixed default, or forced all-binary/all-fold).
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
        "synth_count2": "int", "synth_maxwhere": "int", "synth_count_cmp": "int", "synth_count_range": "int",
        "synth_mode": "counter", "synth_distinct": "set",
        "synth_sumby": "dict", "synth_diff": "dict", "synth_varchain": "dict",
        "synth_filter_argmax": "counter", "synth_2d": "dict",
    }
    _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    _GROUPS = ["K1", "K2", "K3", "K4"]

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.query_var = self.meta.get("query_var")   # synth_varchain: which var to report
        # parameterized predicates (count2 / maxwhere / count_cmp / count_range)
        self.qflag, self.qgrp = self.meta.get("qflag"), self.meta.get("qgrp")
        self.op, self.thresh = self.meta.get("op"), self.meta.get("thresh")
        self.lo, self.hi = self.meta.get("lo"), self.meta.get("hi")
        # month-record variant families (filter_argmax / 2d)
        self.qtype = self.meta.get("qtype")
        self.ffield, self.fval, self.agg = self.meta.get("ffield"), self.meta.get("fval"), self.meta.get("agg")
        self.qgrp2, self.qmon = self.meta.get("qgrp2"), self.meta.get("qmon")
        self.months = list(self.meta.get("months", []))
        if self.strategy == "binary" and self.task in ("synth_runreset", "synth_varchain"):
            # non-associative combines; a binary tree would need a richer monoid. Not
            # rendered yet — these are left-fold tasks.
            raise ValueError(f"{self.task} has no binary oracle; use left_fold")

    def _kind(self):
        return self._KIND.get(self.task, "int")

    def _identity(self):
        if self.task in ("synth_max", "synth_min", "synth_maxwhere"):
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
        if t == "synth_sumby":
            if grp != "RST": acc = {**acc, grp: acc.get(grp, 0) + amt}
            return acc, f"- [{idx:04d}] grp={grp} amt={amt:+d}" + ("  (ignore RST)" if grp == "RST" else "") + f"  → {self._ser_state(acc)}"
        if t == "synth_diff":
            acc = {**acc, flag: acc.get(flag, 0) + amt}
            return acc, f"- [{idx:04d}] flag={flag} amt={amt:+d}  → {self._ser_state(acc)}"
        if t == "synth_count2":
            hit = (flag == self.qflag and grp == self.qgrp)
            if hit: acc += 1
            return acc, f"- [{idx:04d}] flag={flag} grp={grp}  ({'match' if hit else 'no'})  → count={acc}"
        if t == "synth_maxwhere":
            if flag == self.qflag:
                acc = amt if acc is None else max(acc, amt)
                return acc, f"- [{idx:04d}] flag={flag} amt={amt:+d}  → max={self._ser_state(acc)}"
            return acc, f"- [{idx:04d}] flag={flag} amt={amt:+d}  (skip)  → max={self._ser_state(acc)}"
        if t == "synth_count_cmp":
            hit = (amt > self.thresh) if self.op == ">" else (amt < self.thresh)
            if hit: acc += 1
            return acc, f"- [{idx:04d}] amt={amt:+d}  ({amt} {self.op} {self.thresh}? {'yes' if hit else 'no'})  → count={acc}"
        if t == "synth_count_range":
            hit = self.lo <= amt <= self.hi
            if hit: acc += 1
            return acc, f"- [{idx:04d}] amt={amt:+d}  (in [{self.lo},{self.hi}]? {'yes' if hit else 'no'})  → count={acc}"
        mon = s[6]
        if t == "synth_filter_argmax":
            fv = flag if self.ffield == "flag" else mon
            if fv == self.fval:
                acc = acc + Counter([grp])
                return acc, f"- [{idx:04d}] {self.ffield}={fv} grp={grp}  (match)  → {self._ser_state(acc)}"
            return acc, f"- [{idx:04d}] {self.ffield}={fv} grp={grp}  (skip)  → {self._ser_state(acc)}"
        if t == "synth_2d":
            # delta only: the joint tally can hold up to 4 months x 4 grps = 16 keys, and
            # reprinting it per record would blow the fold budget (see fold-state notes).
            key = f"{mon}/{grp}"
            acc = {**acc, key: acc.get(key, 0) + 1}
            return acc, f"- [{idx:04d}] mon={mon} grp={grp}  → {key}={acc[key]}"
        raise ValueError(t)

    def _combine(self, states):
        t = self.task
        if t in ("synth_max", "synth_maxwhere"):
            v = [s for s in states if s is not None]; return max(v) if v else None
        if t == "synth_min":
            v = [s for s in states if s is not None]; return min(v) if v else None
        if t in ("synth_mode", "synth_filter_argmax"):
            tot = Counter()
            for s in states: tot += s
            return tot
        if t == "synth_distinct":
            out = set()
            for s in states: out |= s
            return out
        if t in ("synth_sumby", "synth_diff", "synth_2d"):
            out = {}
            for s in states:
                for k, v in s.items(): out[k] = out.get(k, 0) + v
            return out
        # sum / count / sumwhere / count2 / count_cmp / count_range
        return sum(s for s in states if s is not None)

    def _op_phrase(self) -> str:
        # GERUND fold-step description — reads correctly both in the fold subtask ("continue
        # the accumulator by {op}") and the binary leaf header ("({op}; the trailing reads…)").
        # Every bounded task can now render as left_fold (SYNTH_STRATEGY=both), so all entries
        # are gerunds, not just runreset/varchain.
        return {
            "synth_sum": "adding up the 'amt' fields",
            "synth_count": "counting the records with flag=Y",
            "synth_max": "taking the maximum 'amt'",
            "synth_min": "taking the minimum 'amt'",
            "synth_sumwhere": "summing the 'amt' of records with flag=Y",
            "synth_mode": "tallying each grp value (ignoring RST)",
            "synth_distinct": "collecting the distinct grp values (ignoring RST)",
            "synth_sumby": "adding each 'amt' to its grp's running total (ignoring RST)",
            "synth_diff": "adding each 'amt' to its flag's (Y/N) running total",
            "synth_count2": f"counting the records with flag={self.qflag} and grp={self.qgrp}",
            "synth_maxwhere": f"taking the maximum 'amt' among flag={self.qflag} records",
            "synth_count_cmp": f"counting the records with amt {self.op} {self.thresh}",
            "synth_count_range": f"counting the records with amt in [{self.lo}, {self.hi}]",
            "synth_runreset": "adding each 'amt' to the running total, resetting the total to 0 at each grp=RST",
            "synth_varchain": "applying each assignment in order (a `= VAR` copies that variable's current value)",
            "synth_filter_argmax": f"tallying the grp of each record with {self.ffield}={self.fval} (skipping the rest)",
            "synth_2d": "tallying each (mon, grp) pair, i.e. how many records have each month AND grp",
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
            "synth_sumby": "the per-grp total of 'amt' (a running sum of 'amt' for each grp, ignoring RST)",
            "synth_diff": "the per-flag totals of 'amt' (a running sum for flag=Y and for flag=N)",
            "synth_count2": f"how many have flag={self.qflag} and grp={self.qgrp}",
            "synth_maxwhere": f"the MAXIMUM 'amt' among flag={self.qflag} records",
            "synth_count_cmp": f"how many have amt {self.op} {self.thresh}",
            "synth_count_range": f"how many have amt between {self.lo} and {self.hi} (inclusive)",
            "synth_filter_argmax": f"the per-grp tally over ONLY the records with {self.ffield}={self.fval} "
                                   f"(how many of those have each grp value)",
            "synth_2d": "the per-(mon, grp) tally (how many records have each month AND grp combination)",
        }[self.task]

    def _combine_phrase(self) -> str:
        # Must read naturally in BOTH "then {phrase} their two results" (subtask) and
        # "{Phrase} my children [...]" (combine display).
        return {
            "synth_max": "take the max of", "synth_min": "take the min of",
            "synth_maxwhere": "take the max of",
            "synth_mode": "merge", "synth_distinct": "union",
            "synth_sumby": "merge (add per-grp)", "synth_diff": "merge (add per-flag)",
            "synth_filter_argmax": "merge", "synth_2d": "merge (add per mon/grp key)",
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
        if t == "synth_sumby":
            if not state: return _GROUPS_DEFAULT
            mx = max(state.values()); return min(g for g, v in state.items() if v == mx)
        if t == "synth_diff":
            return str(state.get("Y", 0) - state.get("N", 0))
        if t == "synth_varchain":
            return str(state.get(self.query_var, 0))
        if t in ("synth_filter_argmax", "synth_2d"):
            return self._month_resolve(state)[0]
        return "0" if state is None else str(state)

    # -- month-record families: reduce the tally to the answer, showing the work ---------

    @staticmethod
    def _argbest(counts: dict, best, order):
        target = best(counts.values())
        return min((k for k, n in counts.items() if n == target), key=order.index)

    def _by_month(self, state) -> dict:
        """{month: {grp: n}} from the 'mon/grp' composite-key tally, in calendar order."""
        out = {m: {} for m in self.months}
        for k, n in (state or {}).items():
            mon, _, grp = k.partition("/")
            out.setdefault(mon, {})[grp] = n
        return {m: out[m] for m in sorted(out, key=self._MONTHS.index)}

    @staticmethod
    def _fmt_tally(d: dict) -> str:
        return ", ".join(f"{g}={n}" for g, n in sorted(d.items())) or "none"

    def _month_resolve(self, state):
        """-> (answer, note) for the root."""
        if self.task == "synth_filter_argmax":
            if not state:
                return self._GROUPS[0], f"\nNo records with {self.ffield}={self.fval}; defaulting to {self._GROUPS[0]}."
            ans = self._argbest(state, max if self.agg == "most" else min, self._GROUPS)
            return ans, (f"\nTally over {self.ffield}={self.fval} records: {self._fmt_tally(state)}; "
                         f"{self.agg} common is {ans} ({state[ans]}).")
        bm = self._by_month(state)
        if self.qtype == "months_argmax":
            g, hits, rows = self.qgrp, [], []
            for m, d in bm.items():
                other = max((n for k, n in d.items() if k != g), default=0)
                ok = d.get(g, 0) > other
                if ok: hits.append(m)
                rows.append(f"  {m}: {self._fmt_tally(d)} → {g}={d.get(g, 0)} vs best other {other}: {'yes' if ok else 'no'}")
            return str(len(hits)), ("\nPer month, is " + g + " strictly the most common?\n" + "\n".join(rows)
                                    + f"\nMonths where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        if self.qtype == "months_cmp":
            g1, g2, hits, rows = self.qgrp, self.qgrp2, [], []
            for m, d in bm.items():
                a, b = d.get(g1, 0), d.get(g2, 0)
                if a > b: hits.append(m)
                rows.append(f"  {m}: {g1}={a} vs {g2}={b}: {'yes' if a > b else 'no'}")
            return str(len(hits)), (f"\nPer month, {g1} > {g2}?\n" + "\n".join(rows)
                                    + f"\nMonths where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        if self.qtype == "grp_in_month":
            d = bm.get(self.qmon, {})
            if not d:
                return self._GROUPS[0], f"\nNo records with mon={self.qmon}; defaulting to {self._GROUPS[0]}."
            ans = self._argbest(d, max, self._GROUPS)
            return ans, f"\nTally for mon={self.qmon}: {self._fmt_tally(d)}; most common is {ans} ({d[ans]})."
        # month_for_grp
        g = self.qgrp
        per = {m: d.get(g, 0) for m, d in bm.items()}
        if not per:
            return self._MONTHS[0], f"\nNo records seen; defaulting to {self._MONTHS[0]}."
        ans = self._argbest(per, max, self._MONTHS)
        return ans, (f"\n{g} count per month: " + ", ".join(f"{m}={n}" for m, n in per.items())
                     + f"; the most is {ans} ({per[ans]}).")

    def _finalize_note(self, state) -> str:
        t = self.task
        if t in ("synth_filter_argmax", "synth_2d"):
            return self._month_resolve(state)[1]
        if t == "synth_mode":
            if not state:
                return ""
            mx = max(state.values())
            return f"\nMost frequent grp: {min(g for g, n in state.items() if n == mx)} ({mx})."
        if t == "synth_distinct":
            return f"\nDistinct grp values: {len(state)}."
        if t == "synth_sumby":
            if not state: return ""
            mx = max(state.values()); w = min(g for g, v in state.items() if v == mx)
            return f"\nPer-grp totals: {self._ser_state(state)}; largest total is {w} ({mx})."
        if t == "synth_diff":
            y, n = state.get("Y", 0), state.get("N", 0)
            return f"\nflag=Y total = {y}, flag=N total = {n}; difference = {y} - {n} = {y - n}."
        if t == "synth_varchain":
            return f"\nFinal value of {self.query_var}: {state.get(self.query_var, 0)}."
        return ""   # numeric reduce/fold: the boxed answer IS the state
