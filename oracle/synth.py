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
        # sequential ops with a small structured state (all values ints; absent = not yet)
        "synth_peak": "dict",          # {total, peak}
        "synth_streak": "dict",        # {cur, best}
        "synth_adjacent": "dict",      # {prev, count}   (prev absent before the first record)
        "synth_first_exceed": "dict",  # {total, first}  (first absent until reached)
    }
    _SEQUENTIAL = {"synth_runreset", "synth_varchain", "synth_peak", "synth_streak", "synth_adjacent", "synth_first_exceed"}
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
        # 2-D scheme (field names + ordered vocabularies); defaults keep old metadata working
        self.f_out = self.meta.get("f_out", "mon"); self.f_in = self.meta.get("f_in", "grp")
        self.out_vals = list(self.meta.get("out_vals", self.months or self._MONTHS))
        self.in_vals = list(self.meta.get("in_vals", self._GROUPS))
        if self.strategy == "binary" and self.task in self._SEQUENTIAL:
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
        if t == "synth_peak":
            tot = acc.get("total", 0) + amt; peak = max(acc.get("peak", 0), tot)
            mark = "  (new peak)" if peak > acc.get("peak", 0) else ""
            return {"total": tot, "peak": peak}, f"- [{idx:04d}] amt={amt:+d}  → total={tot}, peak={peak}{mark}"
        if t == "synth_streak":
            cur = acc.get("cur", 0) + 1 if flag == "Y" else 0
            best = max(acc.get("best", 0), cur)
            return {"cur": cur, "best": best}, f"- [{idx:04d}] flag={flag}  → current run={cur}, best={best}"
        if t == "synth_adjacent":
            prev = acc.get("prev"); n = acc.get("count", 0)
            if prev is None:
                return {"prev": amt, "count": n}, f"- [{idx:04d}] amt={amt:+d}  (no predecessor)  → count={n}"
            hit = amt > prev
            if hit: n += 1
            return {"prev": amt, "count": n}, f"- [{idx:04d}] amt={amt:+d}  ({amt} > {prev}? {'yes' if hit else 'no'})  → count={n}"
        if t == "synth_first_exceed":
            tot = acc.get("total", 0) + amt
            first = acc.get("first")          # None/absent means "not yet"; serialized as first=-1
            if first is not None and first >= 0:
                return {"total": tot, "first": first}, f"- [{idx:04d}] amt={amt:+d}  → total={tot} (already exceeded at index {first})"
            if tot > self.thresh:
                return {"total": tot, "first": idx}, f"- [{idx:04d}] amt={amt:+d}  → total={tot} > {self.thresh}: FIRST exceed at index {idx}"
            return {"total": tot, "first": -1}, f"- [{idx:04d}] amt={amt:+d}  → total={tot} (not > {self.thresh}; first=-1 means not yet)"
        mon = s[6]   # outer value; `grp` (s[5]) is the inner value
        if t == "synth_filter_argmax":
            fv = flag if self.ffield == "flag" else mon
            if fv == self.fval:
                acc = acc + Counter([grp])
                return acc, f"- [{idx:04d}] {self.ffield}={fv} {self.f_in}={grp}  (match)  → {self._ser_state(acc)}"
            return acc, f"- [{idx:04d}] {self.ffield}={fv} {self.f_in}={grp}  (skip)  → {self._ser_state(acc)}"
        if t == "synth_2d":
            # MINIMAL STATE (audit 2026-09-24): only months_argmax needs the full joint tally ("is g strictly the most
            # common WITHIN each month" needs every inner value). The other qtypes keep only what they name — the full
            # joint tally taught 16w's OOLONG roots to build per-(month, label) over EVERY label and overflow at 16K.
            fo, fi = self.f_out, self.f_in
            if self.qtype == "months_cmp":
                if grp not in (self.qgrp, self.qgrp2):
                    return acc, f"- [{idx:04d}] {fo}={mon} {fi}={grp}  (skip)"
                key = f"{mon}/{grp}"
                acc = {**acc, key: acc.get(key, 0) + 1}
                return acc, f"- [{idx:04d}] {fo}={mon} {fi}={grp}  (match)  → {key}={acc[key]}"
            if self.qtype == "month_for_grp":
                if grp != self.qgrp:
                    return acc, f"- [{idx:04d}] {fo}={mon} {fi}={grp}  (skip)"
                acc = {**acc, mon: acc.get(mon, 0) + 1}
                return acc, f"- [{idx:04d}] {fo}={mon} {fi}={grp}  (match)  → {mon}={acc[mon]}"
            if self.qtype == "grp_in_month":
                if mon != self.qmon:
                    return acc, f"- [{idx:04d}] {fo}={mon} {fi}={grp}  (skip)"
                acc = {**acc, grp: acc.get(grp, 0) + 1}
                return acc, f"- [{idx:04d}] {fo}={mon} {fi}={grp}  (match)  → {grp}={acc[grp]}"
            # months_argmax: delta only (the joint tally can reach 8 x 5 = 40 keys; never reprint it per record)
            key = f"{mon}/{grp}"
            acc = {**acc, key: acc.get(key, 0) + 1}
            return acc, f"- [{idx:04d}] {self.f_out}={mon} {self.f_in}={grp}  → {key}={acc[key]}"
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
            "synth_peak": "adding each 'amt' to the running total and tracking the highest total reached",
            "synth_streak": "extending the current run of consecutive flag=Y records (reset to 0 on flag=N) and tracking the longest run",
            "synth_adjacent": "comparing each 'amt' with the previous record's 'amt' and counting the strict increases",
            "synth_first_exceed": f"adding each 'amt' to the running total and noting the first index at which it exceeds {self.thresh}",
            "synth_filter_argmax": f"tallying the {self.f_in} of each record with {self.ffield}={self.fval} (skipping the rest)",
            "synth_2d": self._op_2d(),
        }[self.task]

    def _op_2d(self) -> str:
        fo, fi = self.f_out, self.f_in
        if self.qtype == "months_cmp":
            return f"tallying each ({fo}, {fi}) pair for {fi}={self.qgrp} and {fi}={self.qgrp2} only (skipping the rest)"
        if self.qtype == "month_for_grp":
            return f"counting the {fi}={self.qgrp} records per {fo} (skipping the rest)"
        if self.qtype == "grp_in_month":
            return f"tallying the {fi} of each record with {fo}={self.qmon} (skipping the rest)"
        return f"tallying each ({fo}, {fi}) pair, i.e. how many records have each {fo} AND {fi}"

    def _sequential_reason(self) -> str:
        return {
            "synth_runreset": "a grp=RST record resets the total, and what a record contributes depends on the resets before it",
            "synth_varchain": "a `set B = A` copies A's value AT THAT POINT, and assignments must be applied in order",
            "synth_peak": "the peak is a property of the running total's path, which depends on every record before it",
            "synth_streak": "a run of consecutive flag=Y records is defined by the records immediately before each one",
            "synth_adjacent": "each record is compared with the record immediately before it",
            "synth_first_exceed": "the answer is the FIRST index where the running total crosses the threshold, which depends on all earlier records",
        }.get(self.task, super()._sequential_reason())

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
            "synth_filter_argmax": f"the per-{self.f_in} tally over ONLY the records with {self.ffield}={self.fval} "
                                   f"(how many of those have each {self.f_in} value)",
            "synth_2d": self._goal_2d(),
        }[self.task]

    def _goal_2d(self) -> str:
        fo, fi = self.f_out, self.f_in
        if self.qtype == "months_cmp":
            return f"the per-({fo}, {fi}) tally over ONLY the {fi}={self.qgrp} and {fi}={self.qgrp2} records"
        if self.qtype == "month_for_grp":
            return f"the per-{fo} count of {fi}={self.qgrp} records"
        if self.qtype == "grp_in_month":
            return (f"the per-{fi} tally over ONLY the records with {fo}={self.qmon} "
                    f"(how many of those have each {fi} value)")
        return f"the per-({fo}, {fi}) tally (how many records have each {fo} AND {fi} combination)"

    def _shape_reason(self) -> str:
        t = self.task
        if t == "synth_2d":
            fo, fi = self.f_out, self.f_in
            if self.qtype == "month_for_grp":
                return (f"The question asks which {fo} has the most {fi}={self.qgrp} records, so the state is a per-{fo} "
                        f"count of {fi}={self.qgrp} records alone — not a per-({fo}, {fi}) tally.")
            if self.qtype == "months_cmp":
                return (f"The question compares {fi}={self.qgrp} with {fi}={self.qgrp2} WITHIN each {fo}, so the state is a "
                        f"per-({fo}, {fi}) tally of just those two {fi} values — not every {fi}.")
            if self.qtype == "grp_in_month":
                return (f"The question is restricted to records with {fo}={self.qmon}, so the state is a per-{fi} tally "
                        f"over those records only.")
            return (f"The question asks whether {fi}={self.qgrp} beats EVERY other {fi} within each {fo}, so the state is a "
                    f"per-({fo}, {fi}) tally over all {fi} values, not a per-{fi} one.")
        if t == "synth_filter_argmax":
            return f"The question is restricted to records with {self.ffield}={self.fval}, so the state is a per-{self.f_in} tally over those records only."
        if t == "synth_mode":
            return "The question asks which grp value is most common over the whole document, so a per-grp tally is the right state."
        if t == "synth_distinct":
            return "The question asks how many DISTINCT grp values appear, so the state is the SET of grp values seen (no counts)."
        if t == "synth_sumby":
            return "The question asks which grp has the largest total, so the state is a per-grp running total."
        if t == "synth_diff":
            return "The question subtracts one flag's total from the other's, so the state keeps a running total per flag."
        if t in ("synth_count", "synth_count2", "synth_count_cmp", "synth_count_range", "synth_sumwhere"):
            return "The question asks for one number over the whole document, so the state is a single running count/sum."
        if t == "synth_sum":
            return "The question asks for one total over the whole document, so the state is a single running sum — no per-grp breakdown is needed."
        if t in ("synth_max", "synth_min"):
            w = "largest" if t == "synth_max" else "smallest"
            return f"The question asks for the single {w} 'amt', so the state is just the best value seen so far (`none` until one is seen) — not a running total."
        if t == "synth_maxwhere":
            return (f"The question asks for the largest 'amt' among flag={self.qflag} records only, so the state is the best value "
                    f"seen among those records (`none` until one is seen) — other records are skipped, not counted.")
        # fold tasks: why the ACCUMULATOR carries what it carries
        if t == "synth_runreset":
            return ("The answer is the running total after the last reset, so the accumulator is just the current total — "
                    "not a per-grp tally and not a count of resets.")
        if t == "synth_varchain":
            return ("The question asks one variable's final value, but a `set B = A` needs A's value at that moment, so the "
                    "accumulator carries the current value of EVERY variable — not just the one asked about.")
        if t == "synth_peak":
            return ("The peak is the highest value the running total ever reaches, so the accumulator carries the running total "
                    "and the highest total so far — the final total alone would lose the peak.")
        if t == "synth_streak":
            return ("The longest run can end anywhere, so the accumulator carries the current run length and the longest run so "
                    "far — a plain count of flag=Y records would not do.")
        if t == "synth_adjacent":
            return ("Each comparison needs the previous record's amt, so the accumulator carries the count of increases and the "
                    "previous amt — the count alone cannot judge the next record.")
        if t == "synth_first_exceed":
            return (f"The answer is the first index at which the running total exceeds {self.thresh}, so the accumulator "
                    f"carries the running total and that first index (-1 until it happens) — the total alone forgets whether and "
                    f"where the threshold was crossed.")
        return ""

    def _state_format_base(self) -> str:
        if self.task == "synth_2d":
            fo, fi, ov, iv = self.f_out, self.f_in, ", ".join(self.out_vals), ", ".join(self.in_vals)
            if self.qtype == "months_cmp":
                return (f"the partial tally as `<{fo}>/<{fi}>=<count>` entries joined by `|`, where `<{fo}>` is one of "
                        f"{ov} and `<{fi}>` is {self.qgrp} or {self.qgrp2} only")
            if self.qtype == "month_for_grp":
                return (f"the partial tally as `<{fo}>=<count>` entries joined by `|`, counting only {fi}={self.qgrp} "
                        f"records, where `<{fo}>` is one of {ov}")
            if self.qtype == "grp_in_month":
                return (f"the partial tally as `<{fi}>=<count>` entries joined by `|`, over only {fo}={self.qmon} "
                        f"records, where `<{fi}>` is one of {iv}")
            return (f"the partial tally as `<{fo}>/<{fi}>=<count>` entries joined by `|`, "
                    f"where `<{fo}>` is one of {ov} and `<{fi}>` is one of {iv}")
        if self.task == "synth_sumby":
            return "the partial as `<grp>=<total>` entries joined by `|`, where `<grp>` is one of K1, K2, K3, K4"
        if self.task == "synth_diff":
            return "the partial as `<flag>=<total>` entries joined by `|`, where `<flag>` is Y or N"
        return super()._state_format()

    def _key_space(self) -> str:
        if self.task == "synth_filter_argmax":
            return f", where `<key>` is exactly one of {', '.join(self.in_vals)}"
        if self.task == "synth_mode":
            return ", where `<key>` is exactly one of K1, K2, K3, K4 (never RST)"
        return ""

    def _state_format(self) -> str:
        if self.task == "synth_distinct":
            return "the collected grp values joined by `|`, each exactly one of K1, K2, K3, K4 (never RST)"
        if self.task in ("synth_max", "synth_min", "synth_maxwhere"):
            return "the partial as a single integer, or `none` if no qualifying record starts in the range"
        return self._state_format_base()

    def _combine_phrase(self) -> str:
        # Must read naturally in BOTH "then {phrase} their two results" (subtask) and
        # "{Phrase} my children [...]" (combine display).
        return {
            "synth_max": "take the max of", "synth_min": "take the min of",
            "synth_maxwhere": "take the max of",
            "synth_mode": "merge", "synth_distinct": "union",
            "synth_sumby": "merge (add per-grp)", "synth_diff": "merge (add per-flag)",
            "synth_filter_argmax": "merge",
            "synth_2d": {"month_for_grp": f"merge (add per {self.f_out})", "grp_in_month": f"merge (add per {self.f_in})"}.get(
                self.qtype, f"merge (add per {self.f_out}/{self.f_in} key)"),
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
        if t == "synth_peak":     return str(state.get("peak", 0))
        if t == "synth_streak":   return str(state.get("best", 0))
        if t == "synth_adjacent": return str(state.get("count", 0))
        if t == "synth_first_exceed":
            return str(state["first"]) if state.get("first", -1) >= 0 else "none"
        if t in ("synth_filter_argmax", "synth_2d"):
            return self._month_resolve(state)[0]
        return "0" if state is None else str(state)

    # -- month-record families: reduce the tally to the answer, showing the work ---------

    @staticmethod
    def _argbest(counts: dict, best, order):
        target = best(counts.values())
        return min((k for k, n in counts.items() if n == target), key=order.index)

    def _by_month(self, state) -> dict:
        """{outer: {inner: n}} from the 'outer/inner' composite-key tally, in the outer's natural
        order. Keys split at the FIRST '/', so inner values may themselves contain '/'."""
        out = {m: {} for m in self.out_vals}
        for k, n in (state or {}).items():
            o, _, i = k.partition("/")
            out.setdefault(o, {})[i] = n
        order = lambda m: self.out_vals.index(m) if m in self.out_vals else len(self.out_vals)
        return {m: out[m] for m in sorted(out, key=order)}

    @staticmethod
    def _fmt_tally(d: dict) -> str:
        return ", ".join(f"{g}={n}" for g, n in sorted(d.items())) or "none"

    def _month_resolve(self, state):
        """-> (answer, note) for the root."""
        fo, fi = self.f_out, self.f_in
        if self.task == "synth_filter_argmax":
            if not state:
                return self.in_vals[0], f"\nNo records with {self.ffield}={self.fval}; defaulting to {self.in_vals[0]}."
            ans = self._argbest(state, max if self.agg == "most" else min, self.in_vals)
            return ans, (f"\nTally over {self.ffield}={self.fval} records: {self._fmt_tally(state)}; "
                         f"{self.agg} common is {ans} ({state[ans]}).")
        if self.qtype == "grp_in_month":
            d = state or {}
            if not d:
                return self.in_vals[0], f"\nNo records with {fo}={self.qmon}; defaulting to {self.in_vals[0]}."
            ans = self._argbest(d, max, self.in_vals)
            return ans, f"\nTally for {fo}={self.qmon}: {self._fmt_tally(d)}; most common is {ans} ({d[ans]})."
        if self.qtype == "month_for_grp":
            g = self.qgrp
            per = {m: (state or {}).get(m, 0) for m in self.out_vals}
            ans = self._argbest(per, max, self.out_vals)
            return ans, (f"\n{g} count per {fo}: " + ", ".join(f"{m}={n}" for m, n in per.items())
                         + f"; the most is {ans} ({per[ans]}).")
        bm = self._by_month(state)
        if self.qtype == "months_argmax":
            g, hits, rows = self.qgrp, [], []
            for m, d in bm.items():
                other = max((n for k, n in d.items() if k != g), default=0)
                ok = d.get(g, 0) > other
                if ok: hits.append(m)
                rows.append(f"  {fo}={m}: {self._fmt_tally(d)} → {g}={d.get(g, 0)} vs best other {other}: {'yes' if ok else 'no'}")
            return str(len(hits)), (f"\nPer {fo}, is {g} strictly the most common {fi}?\n" + "\n".join(rows)
                                    + f"\n{fo} values where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        if self.qtype == "months_cmp":
            g1, g2, hits, rows = self.qgrp, self.qgrp2, [], []
            for m, d in bm.items():
                a_, b_ = d.get(g1, 0), d.get(g2, 0)
                if a_ > b_: hits.append(m)
                rows.append(f"  {fo}={m}: {g1}={a_} vs {g2}={b_}: {'yes' if a_ > b_ else 'no'}")
            return str(len(hits)), (f"\nPer {fo}, {g1} > {g2}?\n" + "\n".join(rows)
                                    + f"\n{fo} values where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        raise ValueError(f"unknown synth_2d qtype {self.qtype!r}")

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
            fmt = lambda v: f"({v})" if v < 0 else str(v)
            return f"\nflag=Y total = {y}, flag=N total = {n}; difference = {fmt(y)} - {fmt(n)} = {y - n}."
        if t == "synth_varchain":
            return f"\nFinal value of {self.query_var}: {state.get(self.query_var, 0)}."
        if t == "synth_peak":     return f"\nHighest running total reached: {state.get('peak', 0)} (final total {state.get('total', 0)})."
        if t == "synth_streak":   return f"\nLongest run of consecutive flag=Y: {state.get('best', 0)}."
        if t == "synth_adjacent": return f"\nRecords with amt above their predecessor: {state.get('count', 0)}."
        if t == "synth_first_exceed":
            return (f"\nThe running total first exceeds {self.thresh} at index {state['first']}." if state.get("first", -1) >= 0
                    else f"\nThe running total never exceeds {self.thresh} (first=-1 throughout; final total {state.get('total', 0)}).")
        if t == "synth_runreset":
            return f"\nFinal running total after the last record (and any resets): {state}."
        return ""   # numeric reduce/fold: the boxed answer IS the state
