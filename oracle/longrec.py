"""LongRecOracle — aggregation over multi-line prose entries (tasks/longrec).

Same ScaffoldOracle machinery as SynthOracle; what differs is the RECORD: a 60-250-token block
whose ownership is decided by where its HEADER starts, and whose body routinely runs past a
leaf/fold boundary (so the iterative boundary read extends across it). The per-entry op is
mechanical: header fields, or whole-word occurrence counts in the body. State kind by qtype:
  count_tag / mention_count -> int          src_most / tag_in_src -> Counter
  src_tag_2d -> dict of src/tag keys        mention_most -> dict {best: entry_no, n: occurrences}
"""

from __future__ import annotations

from collections import Counter

from oracle.base import ScaffoldOracle

_SRCS = ["A", "B", "C"]
_TAGS = ["K1", "K2", "K3", "K4"]


def _argbest(counts: dict, best, order):
    target = best(counts.values())
    return min((k for k, v in counts.items() if v == target), key=order.index)


def _fmt(d: dict, order=None) -> str:
    """Render a tally in the document's natural key order (the tie-break order the question states)."""
    keys = sorted(d, key=(lambda k: (order.index(k) if k in order else len(order), k)) if order else None)
    return ", ".join(f"{k}={d[k]}" for k in keys) or "none"


class LongRecOracle(ScaffoldOracle):
    name = "longrec_oracle"

    _KIND = {"count_tag": "int", "mention_count": "int", "src_most": "counter",
             "tag_in_src": "counter", "src_tag_2d": "dict", "mention_most": "dict",
             "first_reach": "dict"}   # first_reach: {"count": n, "first": entry_no|absent} — sequential

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.qtype = self.meta["qtype"]
        self.qtag, self.qsrc, self.word = self.meta.get("qtag"), self.meta.get("qsrc"), self.meta.get("word")
        self.qn = self.meta.get("qn")
        # header field scheme (names + ordered vocabularies); defaults keep old metadata working
        self.fa, self.fb = self.meta.get("fa", "src"), self.meta.get("fb", "tag")
        self.a_vals = list(self.meta.get("a_vals", _SRCS)); self.b_vals = list(self.meta.get("b_vals", _TAGS))
        if self.qtype == "first_reach" and self.strategy == "binary":
            raise ValueError("first_reach is sequential; use left_fold")

    def _kind(self):
        return self._KIND[self.qtype]

    # -- per-entry accumulation ---------------------------------------------------------

    def _acc_step(self, acc, s):
        _, _, n, src, tag, occ, snips = s
        q = self.qtype
        # Show the matches counted (≤4 snippets) so the count is grounded in the text read.
        ev = ""
        if occ:
            shown = "; ".join("\u201c" + str(x).replace('"', "'") + "\u201d" for x in snips)
            more = f"; +{occ - len(snips)} more" if occ > len(snips) else ""
            ev = f" [{shown}{more}]"
        if q == "count_tag":
            hit = tag == self.qtag
            if hit: acc += 1
            return acc, f"- entry {n} ({self.fb}={tag}) ({'match' if hit else 'no'}) → count={acc}"
        if q == "src_most":
            acc = acc + Counter([src])
            return acc, f"- entry {n} ({self.fa}={src}) → {self._ser_state(acc)}"
        if q == "tag_in_src":
            if src == self.qsrc:
                acc = acc + Counter([tag])
                return acc, f"- entry {n} ({self.fa}={src}, {self.fb}={tag}) (match) → {self._ser_state(acc)}"
            return acc, f"- entry {n} ({self.fa}={src}, {self.fb}={tag}) (skip) → {self._ser_state(acc)}"
        if q == "src_tag_2d":
            key = f"{src}/{tag}"
            acc = {**acc, key: acc.get(key, 0) + 1}
            return acc, f"- entry {n} ({self.fa}={src}, {self.fb}={tag}) → {key}={acc[key]}"
        if q == "mention_count":
            if occ > 0: acc += 1
            return acc, (f"- entry {n}: '{self.word}' appears {occ}x in the body{ev} "
                         f"({'counts' if occ > 0 else 'no'}) → count={acc}")
        if q == "first_reach":
            if "first" in acc:
                return acc, f"- entry {n} ({self.fb}={tag}) (done) → already reached at entry {acc['first']}; unchanged"
            hit = tag == self.qtag
            cnt = acc.get("count", 0) + (1 if hit else 0)
            acc = {**acc, "count": cnt}
            if hit and cnt >= self.qn:
                acc = {**acc, "first": n}
                return acc, f"- entry {n} ({self.fb}={tag}) (match) → count={cnt} — reaches {self.qn} here: first={n}"
            return acc, f"- entry {n} ({self.fb}={tag}) ({'match' if hit else 'no'}) → count={cnt}"
        # mention_most: keep the best (most occurrences; ties -> lowest entry number)
        cur_n, cur_best = acc.get("n", -1), acc.get("best")
        if occ > cur_n or (occ == cur_n and (cur_best is None or n < cur_best)):
            acc = {"best": n, "n": occ}
            return acc, f"- entry {n}: '{self.word}' appears {occ}x in the body{ev} (new best) → {self._ser_state(acc)}"
        return acc, f"- entry {n}: '{self.word}' appears {occ}x in the body{ev} (not better) → {self._ser_state(acc)}"

    def _combine(self, states):
        q = self.qtype
        if q in ("count_tag", "mention_count"):
            return sum(s for s in states if s is not None)
        if q in ("src_most", "tag_in_src"):
            tot = Counter()
            for s in states: tot += s
            return tot
        if q == "src_tag_2d":
            out = {}
            for s in states:
                for k, v in (s or {}).items(): out[k] = out.get(k, 0) + v
            return out
        best = {}
        for s in states:
            if not s: continue
            if (not best or s["n"] > best["n"] or (s["n"] == best["n"] and s["best"] < best["best"])):
                best = dict(s)
        return best

    # -- root finalize -------------------------------------------------------------------

    def _resolve(self, state):
        q, state = self.qtype, state or {}
        if q == "count_tag":
            return str(state or 0), f"\nTotal entries with {self.fb}={self.qtag} over the whole document: {state or 0}."
        if q == "mention_count":
            return str(state or 0), f"\nTotal entries whose body contains '{self.word}': {state or 0}."
        if q == "src_most":
            if not state: return self.a_vals[0], f"\nNo entries; defaulting to {self.a_vals[0]}."
            a = _argbest(state, max, self.a_vals)
            return a, f"\nEntries per {self.fa} (in natural order): {_fmt(state, self.a_vals)}; most is {a} ({state[a]})."
        if q == "tag_in_src":
            if not state: return self.b_vals[0], f"\nNo entries with {self.fa}={self.qsrc}; defaulting to {self.b_vals[0]}."
            a = _argbest(state, max, self.b_vals)
            return a, f"\n{self.fb} tally over {self.fa}={self.qsrc} entries (in natural order): {_fmt(state, self.b_vals)}; most common is {a} ({state[a]})."
        if q == "src_tag_2d":
            per = {s: {} for s in self.a_vals}
            for k, v in state.items():
                s, _, t = k.partition("/")
                per.setdefault(s, {})[t] = v
            hits, rows = [], []
            for s in self.a_vals:
                d = per[s]
                other = max((v for t, v in d.items() if t != self.qtag), default=0)
                ok = d.get(self.qtag, 0) > other
                if ok: hits.append(s)
                rows.append(f"  {self.fa}={s}: {_fmt(d, self.b_vals)} → {self.qtag}={d.get(self.qtag, 0)} vs best other {other}: {'yes' if ok else 'no'}")
            return str(len(hits)), (f"\nPer {self.fa}, is {self.qtag} strictly the most common {self.fb}?\n" + "\n".join(rows)
                                    + f"\n{self.fa} values where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        if q == "first_reach":
            if "first" not in state:
                return "none", f"\nOnly {state.get('count', 0)} entries with {self.fb}={self.qtag} in total — {self.qn} is never reached."
            return str(state["first"]), f"\nThe cumulative count of {self.fb}={self.qtag} entries first reaches {self.qn} at entry {state['first']}."
        if not state:
            return "0", "\nNo entries seen."
        return str(state["best"]), (f"\nThe entry with the most occurrences of '{self.word}' is entry "
                                    f"{state['best']} ({state['n']}x).")

    def _finalize(self, state) -> str:
        return self._resolve(state)[0]

    def _finalize_note(self, state) -> str:
        return self._resolve(state)[1]

    # -- phrasing: the record is a multi-line ENTRY owned by where its header starts --------

    def _unit(self) -> str:
        return "entries"

    def _op_phrase(self) -> str:
        return {
            "count_tag": f"counting the entries with {self.fb}={self.qtag}",
            "src_most": f"tallying each entry's {self.fa}",
            "tag_in_src": f"tallying the {self.fb} of each entry with {self.fa}={self.qsrc} (skipping the rest)",
            "src_tag_2d": f"tallying each ({self.fa}, {self.fb}) pair, i.e. how many entries have each {self.fa} AND {self.fb}",
            "mention_count": f"counting the entries whose body contains the word '{self.word}'",
            "mention_most": f"tracking the entry with the most whole-word occurrences of '{self.word}' in its body",
            "first_reach": f"counting {self.fb}={self.qtag} entries in order and noting the first entry at which the count reaches {self.qn}",
        }[self.qtype]

    def _goal_phrase(self) -> str:
        return {
            "count_tag": f"how many entries have {self.fb}={self.qtag}",
            "src_most": f"the per-{self.fa} tally (how many entries have each {self.fa})",
            "tag_in_src": f"the per-{self.fb} tally over ONLY the entries with {self.fa}={self.qsrc}",
            "src_tag_2d": f"the per-({self.fa}, {self.fb}) tally (how many entries have each {self.fa} AND {self.fb} combination)",
            "mention_count": f"how many entries contain the word '{self.word}' in their body",
            "mention_most": f"the entry with the most occurrences of '{self.word}' in its body (and that count)",
            "first_reach": "(sequential — not used)",
        }[self.qtype]

    def _combine_phrase(self) -> str:
        return {"src_most": "merge", "tag_in_src": "merge", "src_tag_2d": f"merge (add per {self.fa}/{self.fb} key)",
                "mention_most": "take the best of"}.get(self.qtype, "sum")

    def _sequential_reason(self) -> str:
        return (f"the answer is the FIRST entry at which a running count reaches {self.qn}, which "
                f"depends on every entry before it")

    def _shape_reason(self) -> str:
        q = self.qtype
        if q == "src_tag_2d":
            return f"The question compares {self.fb} values WITHIN each {self.fa}, so the state is a per-({self.fa}, {self.fb}) tally."
        if q == "tag_in_src":
            return f"The question is restricted to {self.fa}={self.qsrc} entries, so the state is a per-{self.fb} tally over those only."
        if q == "src_most":
            return f"The question asks which {self.fa} has the most entries, so the state is a per-{self.fa} count."
        if q in ("count_tag", "mention_count"):
            return "The question asks for one total over the whole document, so the state is a single count."
        if q == "mention_most":
            return "The question asks which single entry has the most occurrences, so the state is just the best (entry, count) seen so far."
        if q == "first_reach":
            return (f"The answer is the entry at which the running count of {self.fb}={self.qtag} entries first reaches {self.qn}, "
                    f"so the accumulator carries the running count and that entry once it is reached.")
        return ""

    def _state_format(self) -> str:
        if self.qtype == "src_tag_2d":
            return (f"the partial tally as `<{self.fa}>/<{self.fb}>=<count>` entries joined by `|`, where "
                    f"`<{self.fa}>` is one of {', '.join(self.a_vals)} and `<{self.fb}>` is one of {', '.join(self.b_vals)}")
        if self.qtype == "mention_most":
            return "the best entry so far as `best=<entry number>|n=<occurrences>`"
        return super()._state_format()

    def _key_space(self) -> str:
        if self.qtype == "src_most":
            return f", where `<key>` is exactly one of {', '.join(self.a_vals)}"
        if self.qtype == "tag_in_src":
            return f", where `<key>` is exactly one of {', '.join(self.b_vals)}"
        return ""

    def _empty_phrase(self) -> str:
        return "  (no entry header starts here — this range is inside an entry owned by the range before it)"

    def _partial_header(self, a, b, n) -> str:
        return (f"Computing the partial over the {self._n_units(n)} whose HEADER STARTS in {a}..{b} "
                f"({self._op_phrase()}; the trailing reads only finish the last entry's body, and "
                f"any entry whose header starts at/after {b} belongs to the next range)")

    def _fold_header(self, n, a, cut) -> str:
        return f"Folding the {self._n_units(n)} whose HEADER STARTS in {a}..{cut} into the accumulator (in order)"

    def _finish_phrase(self, end) -> str:
        return f"to finish the last entry's body (it may run well past {end})"

    def _cutoff_phrase(self, covered) -> str:
        return (f"The last entry's body is still cut off at {covered} (no blank line / next header "
                f"yet)")
