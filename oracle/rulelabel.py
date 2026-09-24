"""RuleLabelOracle — per-sentence rule classification + aggregation (tasks/rulelabel).

The leaf line SHOWS the check it made on each sentence before tallying it, e.g.
  - [S2] "Come in," she said.  → contains a quotation mark? yes → dialogue   (dialogue:3|narration:4)
so the trained habit is: judge each item, write the judgment with its evidence, then tally — never a
one-line `label=x count=10`. State kinds: count -> int; most_common/relative -> Counter over the two
labels; sections_cmp/section_most -> dict of `S<n>/<label>` keys (2-D). Binary (order-independent).
"""

from __future__ import annotations

from collections import Counter

from oracle.base import ScaffoldOracle


class RuleLabelOracle(ScaffoldOracle):
    name = "rulelabel_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        m = self.meta
        self.qtype, self.lt, self.lf = m["qtype"], m["label_true"], m["label_false"]
        self.rule_text, self.sections = m["rule_text"], list(m["sections"])
        self.qlabel, self.qa, self.qb = m.get("qlabel"), m.get("qa"), m.get("qb")
        # the yes/no question the leaf asks of each sentence
        self.check_q = {
            "dialogue": "contains a quotation mark?", "question": "ends with a question mark?",
            "numeric": "contains a digit?", "named": "capitalized word after the first?",
        }[m["rule"]]

    def _kind(self):
        return {"count": "int", "most_common": "counter", "relative": "counter",
                "sections_cmp": "dict", "section_most": "dict"}[self.qtype]

    def _unit(self):
        return "sentences"

    # -- per-sentence: show the check, then the label, then the running state ----------------

    def _acc_step(self, acc, s):
        _, _, idx, label, sec, snip = s
        yes = label == self.lt
        check = f"{self.check_q} {'yes' if yes else 'no'} → {label}"   # the per-item judgment, shown
        q = self.qtype
        if q == "count":
            hit = label == self.qlabel
            if hit: acc += 1
            return acc, f"- {sec} \"{snip}\" → {check} {'(counts)' if hit else '(no)'} → count={acc}"
        if q in ("most_common", "relative"):
            acc = acc + Counter([label])
            return acc, f"- {sec} \"{snip}\" → {check} → {label}: {acc[label]}"
        if q == "section_most":   # MINIMAL STATE (audit 2026-09-24): one label's count per section, not both labels
            if label != self.qlabel:
                return acc, f"- {sec} \"{snip}\" → {check} (not `{self.qlabel}`, skip)"
            acc = {**acc, sec: acc.get(sec, 0) + 1}
            return acc, f"- {sec} \"{snip}\" → {check} (counts for {sec}) → {sec}={acc[sec]}"
        key = f"{sec}/{label}"
        acc = {**acc, key: acc.get(key, 0) + 1}
        return acc, f"- {sec} \"{snip}\" → {check} → {key}={acc[key]}"

    def _combine(self, states):
        if self.qtype == "count":
            return sum(s for s in states if s is not None)
        if self.qtype in ("most_common", "relative"):
            tot = Counter()
            for s in states: tot += s
            return tot
        out = {}
        for s in states:
            for k, v in (s or {}).items(): out[k] = out.get(k, 0) + v
        return out

    # -- root ------------------------------------------------------------------------------

    def _per_section(self, state):
        per = {s: {} for s in self.sections}
        for k, v in (state or {}).items():
            sec, _, lab = k.partition("/")
            per.setdefault(sec, {})[lab] = v
        return per

    def _resolve(self, state):
        q, lt, lf = self.qtype, self.lt, self.lf
        if q == "count":
            return str(state or 0), ""
        if q == "most_common":
            a, b = state.get(lt, 0), state.get(lf, 0)
            ans = lt if a >= b else lf
            return ans, f"\n{lt}={a}, {lf}={b} → more common: {ans}" + (" (tie → the rule's positive label)" if a == b else "") + "."
        if q == "relative":
            a, b = state.get(self.qa, 0), state.get(self.qb, 0)
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
        counts = {s: (state or {}).get(s, 0) for s in self.sections}   # section_most: {section: count of qlabel}
        best = max(counts.values()) if counts else 0
        ans = next((s for s in self.sections if counts[s] == best), self.sections[0])
        return ans, (f"\n{self.qlabel} per section: " + ", ".join(f"{s}={n}" for s, n in counts.items())
                     + f"; the most is {ans} ({best}).")

    def _finalize(self, state): return self._resolve(state)[0]
    def _finalize_note(self, state): return self._resolve(state)[1]

    # -- phrasing --------------------------------------------------------------------------

    def _rule(self):
        return f"labelling each sentence `{self.lt}` if it {self.rule_text}, else `{self.lf}`"

    def _op_phrase(self):
        return {
            "count": f"{self._rule()} and counting the `{self.qlabel}` ones",
            "most_common": f"{self._rule()} and tallying the labels",
            "relative": f"{self._rule()} and tallying the labels",
            "sections_cmp": f"{self._rule()} and tallying each (section, label) pair",
            "section_most": f"{self._rule()} and counting the `{self.qlabel}` sentences per section",
        }[self.qtype]

    def _goal_phrase(self):
        return {
            "count": f"how many sentences are `{self.qlabel}` ({self._rule()})",
            "most_common": f"the per-label tally ({self._rule()})",
            "relative": f"the per-label tally ({self._rule()})",
            "sections_cmp": f"the per-(section, label) tally ({self._rule()})",
            "section_most": f"the per-section count of `{self.qlabel}` sentences ({self._rule()})",
        }[self.qtype]

    def _shape_reason(self):
        if self.qtype == "sections_cmp":
            return "The question compares labels WITHIN each section, so the state is a per-(section, label) tally, not a per-label one."
        if self.qtype == "section_most":
            return (f"The question compares one label's count ACROSS sections, so the state is a per-section count of "
                    f"`{self.qlabel}` alone — not a per-(section, label) tally.")
        if self.qtype == "count":
            return f"The question asks for one label's total, so the state is a single count of `{self.qlabel}` sentences."
        return "The question asks about labels over the whole document, so a per-label tally is the right state."

    def _state_format(self):
        labs = f"`{self.lt}` or `{self.lf}`"
        if self.qtype == "count":
            return "the partial as a single integer"
        if self.qtype in ("most_common", "relative"):
            return f"the partial tally as `<label>:<count>` entries joined by `|`, where `<label>` is exactly {labs}"
        if self.qtype == "section_most":
            return (f"the partial tally as `<section>=<count>` entries joined by `|`, counting only `{self.qlabel}` "
                    f"sentences, where `<section>` is the line's `S<n>` tag")
        return (f"the partial tally as `<section>/<label>=<count>` entries joined by `|`, where `<section>` is "
                f"the line's `S<n>` tag and `<label>` is exactly {labs}")

    def _combine_phrase(self):
        return {"count": "sum", "most_common": "merge", "relative": "merge",
                "section_most": "merge (add per section)"}.get(self.qtype, "merge (add per section/label key)")

    def _empty_phrase(self):
        return "  (no sentence starts here)"

    def _partial_header(self, a, b, n):
        return (f"Labelling the {n} sentences whose line STARTS in {a}..{b} one at a time and "
                f"{'counting' if self.qtype == 'count' else 'tallying'} them ({self._op_phrase()}; the "
                f"trailing reads only finish the last line, and any sentence starting at/after {b} belongs "
                f"to the next range)")

    def _finish_phrase(self, end):
        return f"to finish the last sentence (its line may run past {end})"
