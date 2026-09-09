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


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"


class LongRecOracle(ScaffoldOracle):
    name = "longrec_oracle"

    _KIND = {"count_tag": "int", "mention_count": "int", "src_most": "counter",
             "tag_in_src": "counter", "src_tag_2d": "dict", "mention_most": "dict"}

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.qtype = self.meta["qtype"]
        self.qtag, self.qsrc, self.word = self.meta.get("qtag"), self.meta.get("qsrc"), self.meta.get("word")

    def _kind(self):
        return self._KIND[self.qtype]

    # -- per-entry accumulation ---------------------------------------------------------

    def _acc_step(self, acc, s):
        _, _, n, src, tag, occ, snips = s
        q = self.qtype
        # Show the matches counted (≤4 snippets) so the count is grounded in the text read.
        ev = (" [" + "; ".join(f'"{x}"' for x in snips) + ("; …" if occ > len(snips) else "") + "]") if occ else ""
        if q == "count_tag":
            hit = tag == self.qtag
            if hit: acc += 1
            return acc, f"- entry {n} (tag={tag})  ({'match' if hit else 'no'})  → count={acc}"
        if q == "src_most":
            acc = acc + Counter([src])
            return acc, f"- entry {n} (src={src})  → {self._ser_state(acc)}"
        if q == "tag_in_src":
            if src == self.qsrc:
                acc = acc + Counter([tag])
                return acc, f"- entry {n} (src={src}, tag={tag})  (match)  → {self._ser_state(acc)}"
            return acc, f"- entry {n} (src={src}, tag={tag})  (skip)  → {self._ser_state(acc)}"
        if q == "src_tag_2d":
            key = f"{src}/{tag}"
            acc = {**acc, key: acc.get(key, 0) + 1}
            return acc, f"- entry {n} (src={src}, tag={tag})  → {key}={acc[key]}"
        if q == "mention_count":
            if occ > 0: acc += 1
            return acc, (f"- entry {n}: '{self.word}' appears {occ}x in the body{ev}  "
                         f"({'counts' if occ > 0 else 'no'})  → count={acc}")
        # mention_most: keep the best (most occurrences; ties -> lowest entry number)
        cur_n, cur_best = acc.get("n", -1), acc.get("best")
        if occ > cur_n or (occ == cur_n and (cur_best is None or n < cur_best)):
            acc = {"best": n, "n": occ}
            return acc, f"- entry {n}: '{self.word}' appears {occ}x{ev}  (new best)  → {self._ser_state(acc)}"
        return acc, f"- entry {n}: '{self.word}' appears {occ}x{ev}  → {self._ser_state(acc)}"

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
            return str(state or 0), ""
        if q == "mention_count":
            return str(state or 0), ""
        if q == "src_most":
            if not state: return _SRCS[0], f"\nNo entries; defaulting to {_SRCS[0]}."
            a = _argbest(state, max, _SRCS)
            return a, f"\nEntries per src: {_fmt(state)}; most is {a} ({state[a]})."
        if q == "tag_in_src":
            if not state: return _TAGS[0], f"\nNo entries with src={self.qsrc}; defaulting to {_TAGS[0]}."
            a = _argbest(state, max, _TAGS)
            return a, f"\nTag tally over src={self.qsrc} entries: {_fmt(state)}; most common is {a} ({state[a]})."
        if q == "src_tag_2d":
            per = {s: {} for s in _SRCS}
            for k, v in state.items():
                s, _, t = k.partition("/")
                per.setdefault(s, {})[t] = v
            hits, rows = [], []
            for s in _SRCS:
                d = per[s]
                other = max((v for t, v in d.items() if t != self.qtag), default=0)
                ok = d.get(self.qtag, 0) > other
                if ok: hits.append(s)
                rows.append(f"  src={s}: {_fmt(d)} → {self.qtag}={d.get(self.qtag, 0)} vs best other {other}: {'yes' if ok else 'no'}")
            return str(len(hits)), (f"\nPer src, is {self.qtag} strictly the most common tag?\n" + "\n".join(rows)
                                    + f"\nSrcs where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
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
            "count_tag": f"counting the entries with tag={self.qtag}",
            "src_most": "tallying each entry's src",
            "tag_in_src": f"tallying the tag of each entry with src={self.qsrc} (skipping the rest)",
            "src_tag_2d": "tallying each (src, tag) pair, i.e. how many entries have each src AND tag",
            "mention_count": f"counting the entries whose body contains the word '{self.word}'",
            "mention_most": f"tracking the entry with the most whole-word occurrences of '{self.word}' in its body",
        }[self.qtype]

    def _goal_phrase(self) -> str:
        return {
            "count_tag": f"how many entries have tag={self.qtag}",
            "src_most": "the per-src tally (how many entries have each src)",
            "tag_in_src": f"the per-tag tally over ONLY the entries with src={self.qsrc}",
            "src_tag_2d": "the per-(src, tag) tally (how many entries have each src AND tag combination)",
            "mention_count": f"how many entries contain the word '{self.word}' in their body",
            "mention_most": f"the entry with the most occurrences of '{self.word}' in its body (and that count)",
        }[self.qtype]

    def _combine_phrase(self) -> str:
        return {"src_most": "merge", "tag_in_src": "merge", "src_tag_2d": "merge (add per src/tag key)",
                "mention_most": "take the best of"}.get(self.qtype, "sum")

    def _empty_phrase(self) -> str:
        return "  (no entry header starts here — this range is inside an entry owned by the range before it)"

    def _partial_header(self, a, b, n) -> str:
        return (f"Computing the partial over the {n} entries whose HEADER line STARTS in {a}..{b} "
                f"({self._op_phrase()}; the trailing reads only finish the last entry's body, and "
                f"any entry whose header starts at/after {b} belongs to the next range)")

    def _fold_header(self, n, a, cut) -> str:
        return f"Folding the {n} entries whose HEADER line STARTS in {a}..{cut} into the accumulator (in order)"

    def _finish_phrase(self, end) -> str:
        return f"to finish the last entry's body (it may run well past {end})"

    def _cutoff_phrase(self, covered) -> str:
        return (f"The last entry's body is still cut off at {covered} (no blank line / next header "
                f"yet)")
