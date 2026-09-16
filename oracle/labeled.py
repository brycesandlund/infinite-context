"""LabeledOracle — gold-label classification + aggregation over real labeled text (tasks/labeled).

Structurally RuleLabelOracle with the label read from the DATA (the dataset's gold) instead of a
rule, plus an author axis for the filter question. The leaf line shows the judgment per item:
  - S2 [by Kim] "Absolutely loved the brunch here, the st…" → label: positive → positive:3
State kinds: count -> int; most_common / relative / author_most -> Counter; sections_cmp /
section_most -> dict of `S<n>/<label>` keys. Binary (order-independent).
"""

from __future__ import annotations

from collections import Counter

from oracle.base import ScaffoldOracle


class LabeledOracle(ScaffoldOracle):
    name = "labeled_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        m = self.meta
        self.qtype, self.labels, self.sections = m["qtype"], list(m["labels"]), list(m["sections"])
        self.qlabel, self.qa, self.qb, self.qauthor = m.get("qlabel"), m.get("qa"), m.get("qb"), m.get("qauthor")

    def _kind(self):
        return {"count": "int", "most_common": "counter", "relative": "counter", "author_most": "counter",
                "sections_cmp": "dict", "section_most": "dict"}[self.qtype]

    def _unit(self):
        return "items"

    def _acc_step(self, acc, s):
        _, _, idx, label, sec, au, snip = s
        q = self.qtype
        head = f"- {sec} [by {au}] \"{snip}\" → label: {label}"
        if q == "count":
            hit = label == self.qlabel
            if hit: acc += 1
            return acc, f"{head} {'(counts)' if hit else '(no)'} → count={acc}"
        if q == "author_most":
            if au == self.qauthor:
                acc = acc + Counter([label])
                return acc, f"{head} (author matches) → {label}: {acc[label]}"
            return acc, f"{head} (other author, skip)"
        if q in ("most_common", "relative"):
            acc = acc + Counter([label])
            return acc, f"{head} → {label}: {acc[label]}"
        key = f"{sec}/{label}"
        acc = {**acc, key: acc.get(key, 0) + 1}
        return acc, f"{head} → {key}={acc[key]}"

    def _combine(self, states):
        if self.qtype == "count":
            return sum(s for s in states if s is not None)
        if self.qtype in ("most_common", "relative", "author_most"):
            tot = Counter()
            for s in states: tot += s
            return tot
        out = {}
        for s in states:
            for k, v in (s or {}).items(): out[k] = out.get(k, 0) + v
        return out

    def _per_section(self, state):
        per = {s: {} for s in self.sections}
        for k, n in (state or {}).items():
            sec, _, lab = k.partition("/")
            per.setdefault(sec, {})[lab] = n
        return per

    def _argmax_label(self, c):
        best = max((c.get(l, 0) for l in self.labels), default=0)
        return next(l for l in self.labels if c.get(l, 0) == best)     # labels sorted -> alphabetical tie

    def _fmt(self, c):
        return ", ".join(f"{l}={c.get(l, 0)}" for l in self.labels if c.get(l, 0))

    def _resolve(self, state):
        q = self.qtype
        if q == "count":
            return str(state or 0), ""
        if q in ("most_common", "author_most"):
            c = state or Counter()
            if not c:
                return self.labels[0], f"\nNo matching items; defaulting to {self.labels[0]}."
            a = self._argmax_label(c)
            who = f" over author {self.qauthor}'s items" if q == "author_most" else ""
            return a, f"\nLabel tally{who}: {self._fmt(c)}; most common is {a} ({c[a]})."
        if q == "relative":
            c = state or Counter(); a, b = c.get(self.qa, 0), c.get(self.qb, 0)
            ans = "more common than" if a > b else "less common than" if a < b else "equally common as"
            return ans, f"\n{self.qa}={a} vs {self.qb}={b} → {self.qa} is {ans} {self.qb}."
        per = self._per_section(state)
        if q == "sections_cmp":
            hits, rows = [], []
            for s in self.sections:
                a, b = per[s].get(self.qa, 0), per[s].get(self.qb, 0)
                if a > b: hits.append(s)
                rows.append(f"  {s}: {self.qa}={a} vs {self.qb}={b}: {'yes' if a > b else 'no'}")
            return str(len(hits)), (f"\nPer section, {self.qa} > {self.qb}?\n" + "\n".join(rows)
                                    + f"\nSections where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        counts = {s: per[s].get(self.qlabel, 0) for s in self.sections}
        best = max(counts.values()) if counts else 0
        ans = next((s for s in self.sections if counts[s] == best), self.sections[0])
        return ans, (f"\n{self.qlabel} per section: " + ", ".join(f"{s}={n}" for s, n in counts.items())
                     + f"; the most is {ans} ({best}).")

    def _finalize(self, state): return self._resolve(state)[0]
    def _finalize_note(self, state): return self._resolve(state)[1]

    # -- phrasing ---------------------------------------------------------------------------

    def _labelling(self):
        return f"judging each item's label (one of {', '.join(self.labels)})"

    def _op_phrase(self):
        return {
            "count": f"{self._labelling()} and counting the `{self.qlabel}` ones",
            "most_common": f"{self._labelling()} and tallying the labels",
            "relative": f"{self._labelling()} and tallying the labels",
            "author_most": f"{self._labelling()} and tallying the labels of author {self.qauthor}'s items only",
            "sections_cmp": f"{self._labelling()} and tallying each (section, label) pair",
            "section_most": f"{self._labelling()} and tallying each (section, label) pair",
        }[self.qtype]

    def _goal_phrase(self):
        return {
            "count": f"how many items are `{self.qlabel}` ({self._labelling()})",
            "most_common": f"the per-label tally ({self._labelling()})",
            "relative": f"the per-label tally ({self._labelling()})",
            "author_most": f"the per-label tally over ONLY author {self.qauthor}'s items ({self._labelling()})",
            "sections_cmp": f"the per-(section, label) tally ({self._labelling()})",
            "section_most": f"the per-(section, label) tally ({self._labelling()})",
        }[self.qtype]

    def _state_format(self):
        labs = ", ".join(f"`{l}`" for l in self.labels)
        if self.qtype == "count":
            return "the partial as a single integer"
        if self.qtype in ("most_common", "relative", "author_most"):
            return f"the partial tally as `<label>:<count>` entries joined by `|`, where `<label>` is exactly one of {labs}"
        return (f"the partial tally as `<section>/<label>=<count>` entries joined by `|`, where `<section>` is the "
                f"line's `S<n>` tag and `<label>` is exactly one of {labs}")

    def _combine_phrase(self):
        return {"count": "sum", "most_common": "merge", "relative": "merge", "author_most": "merge"}.get(
            self.qtype, "merge (add per section/label key)")

    def _empty_phrase(self):
        return "  (no item starts here)"

    def _partial_header(self, a, b, n):
        return (f"Judging the {n} items whose line STARTS in {a}..{b} one at a time and "
                f"{'counting' if self.qtype == 'count' else 'tallying'} them ({self._op_phrase()}; the trailing "
                f"reads only finish the last line, and any item starting at/after {b} belongs to the next range)")

    def _finish_phrase(self, end):
        return f"to finish the last item (its line may run past {end})"
