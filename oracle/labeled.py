"""LabeledOracle — gold-label classification + aggregation over real labeled text (tasks/labeled).

Structurally RuleLabelOracle with the label read from the DATA (the dataset's gold) instead of a
rule, plus an author axis and two KEY MODES for the outer axis:
  section — the line's `[S3]` tag is the key (copied);
  date    — the line carries `[Jul 28, 2022]` and the key is DERIVED: the month `Jul 2022`. The leaf
            line shows the derivation, since bucketing a value into a key is the step OOLONG temporal
            needs and no other task demonstrates.
Leaf line:  - [Jul 28, 2022 → Jul 2022] [by Kim] "Absolutely loved the brunch…" → label: positive → Jul 2022/positive=3
State kinds: count -> int; most_common / relative / author_most / author_top -> Counter;
sections_cmp / section_most -> dict `<outer>/<label>`; dates_rep_k -> dict `<date>` -> count.
"""

from __future__ import annotations

from collections import Counter

from oracle.base import ScaffoldOracle

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _dkey(date):
    mon, day, year = date.replace(",", "").split()
    return (int(year), _MONTHS.index(mon), int(day))


class LabeledOracle(ScaffoldOracle):
    name = "labeled_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        m = self.meta
        self.qtype, self.labels, self.sections = m["qtype"], list(m["labels"]), list(m["sections"])
        self.authors = list(m.get("authors", []))
        self.key_mode = m.get("key_mode", "section")
        self.unit_word = "section" if self.key_mode == "section" else "month"
        self.qlabel, self.qa, self.qb = m.get("qlabel"), m.get("qa"), m.get("qb")
        self.qauthor, self.qk, self.qmon, self.qdate = m.get("qauthor"), m.get("qk"), m.get("qmon"), m.get("qdate")
        self.qa1, self.qa2 = m.get("qa1"), m.get("qa2")

    def _kind(self):
        return {"count": "int", "most_common": "counter", "relative": "counter", "author_most": "counter",
                "author_top": "counter", "author_count": "counter", "author_least": "counter", "author_cmp": "counter",
                "sections_cmp": "dict", "section_most": "dict", "section_mode_count": "dict",
                "dates_rep_k": "dict", "first_month_cmp": "dict", "before_after": "dict"}[self.qtype]

    def _unit(self):
        return "items"

    # -- one line per item: tag (with derivation in date mode), author, snippet, judged label ---

    def _tag(self, outer, date):
        if self.key_mode == "date":
            return f"[{date} → {outer}]"
        return f"[{outer}]"

    def _acc_step(self, acc, s):
        _, _, idx, label, outer, au, snip, date = s
        q = self.qtype
        by = f" [by {au}]" if q in ("author_most", "author_top", "author_count", "author_least", "author_cmp") else ""   # author only when it matters
        head = f"- {self._tag(outer, date)}{by} \"{snip}\" → label: {label}"
        if q == "count":
            hit = label == self.qlabel
            if hit: acc += 1
            return acc, f"{head} {'(counts)' if hit else '(no)'} → count={acc}"
        if q == "author_count":
            acc = acc + Counter([au])
            return acc, f"{head} → {au}: {acc[au]}"
        if q in ("author_most", "author_least"):
            if au == self.qauthor:
                acc = acc + Counter([label])
                return acc, f"{head} (author matches) → {label}: {acc[label]}"
            return acc, f"{head} (other author, skip)"
        if q in ("author_top", "author_cmp"):
            if label == self.qlabel:
                acc = acc + Counter([au])
                return acc, f"{head} (counts for {au}) → {au}: {acc[au]}"
            return acc, f"{head} (not `{self.qlabel}`, skip)"
        if q in ("most_common", "relative"):
            acc = acc + Counter([label])
            return acc, f"{head} → {label}: {acc[label]}"
        if q == "before_after":
            side = "before" if _dkey(date) < _dkey(self.qdate) else "on_or_after"
            key = f"{side}/{label if label == self.qlabel else 'other'}"
            acc = {**acc, key: acc.get(key, 0) + 1}
            return acc, f"- [{date} → {side} {self.qdate}] \"{snip}\" → label: {label} → {key}={acc[key]}"
        if q == "dates_rep_k":
            if outer != self.qmon:
                return acc, f"- [{date} → {outer}] \"{snip}\" (not {self.qmon}, skip)"
            acc = {**acc, date: acc.get(date, 0) + 1}
            return acc, f"- [{date} → {outer}] \"{snip}\" → date {date} seen {acc[date]}x"
        key = f"{outer}/{label}"
        acc = {**acc, key: acc.get(key, 0) + 1}
        return acc, f"{head} → {key}={acc[key]}"

    def _combine(self, states):
        q = self.qtype
        if q == "count":
            return sum(s for s in states if s is not None)
        if q in ("most_common", "relative", "author_most", "author_top", "author_count", "author_least", "author_cmp"):
            tot = Counter()
            for s in states: tot += s
            return tot
        out = {}
        for s in states:
            for k, v in (s or {}).items(): out[k] = out.get(k, 0) + v
        return out

    # dict keys in date mode contain spaces and commas ("Jul 28, 2022") — the base serializer
    # handles them (split on '|' and '='), but sort by outer natural order where we can.
    def _per_section(self, state):
        per = {s: {} for s in self.sections}
        for k, n in (state or {}).items():
            outer, _, lab = k.partition("/")
            per.setdefault(outer, {})[lab] = n
        return per

    def _argmax(self, c, order):
        best = max((c.get(x, 0) for x in order), default=0)
        return next(x for x in order if c.get(x, 0) == best)

    def _fmt(self, c, order):
        return ", ".join(f"{x}={c.get(x, 0)}" for x in order if c.get(x, 0))

    def _resolve(self, state):
        q, uw = self.qtype, self.unit_word
        if q == "count":
            return str(state or 0), ""
        if q in ("most_common", "author_most"):
            c = state or Counter()
            if not c:
                return self.labels[0], f"\nNo matching items; defaulting to {self.labels[0]}."
            a = self._argmax(c, self.labels)
            who = f" over author {self.qauthor}'s items" if q == "author_most" else ""
            return a, f"\nLabel tally{who}: {self._fmt(c, self.labels)}; most common is {a} ({c[a]})."
        if q == "author_cmp":
            c = state or Counter()
            n1, n2 = c.get(self.qa1, 0), c.get(self.qa2, 0)
            ans = self.qa1 if n1 > n2 else self.qa2 if n2 > n1 else min(self.qa1, self.qa2)
            return ans, (f"\n`{self.qlabel}` items: author {self.qa1} has {n1}, author {self.qa2} has {n2} → "
                         f"{'tie, alphabetical first' if n1 == n2 else 'more for'} {ans}. The answer is the tag content "
                         f"itself, {ans} — not the word 'author'.")
        if q == "author_count":
            c = state or Counter()
            a = self._argmax(c, self.authors)
            return a, f"\nItems per author: {self._fmt(c, self.authors) or 'none'}; the most is {a} ({c.get(a, 0)})."
        if q == "author_least":
            c = state or Counter()
            present = [l for l in self.labels if c.get(l, 0) > 0]
            if not present:
                return self.labels[0], f"\nNo items by author {self.qauthor}; defaulting to {self.labels[0]}."
            a = min(present, key=lambda l: (c[l], l))
            return a, (f"\nLabel tally over author {self.qauthor}'s items: {self._fmt(c, self.labels)}; among the labels "
                       f"that appear, the least common is {a} ({c[a]}).")
        if q == "section_mode_count":
            per = self._per_section(state)
            hits, rows = [], []
            for s in self.sections:
                mine = per[s].get(self.qlabel, 0)
                other = max((v for l, v in per[s].items() if l != self.qlabel), default=0)
                ok = mine > other
                if ok: hits.append(s)
                rows.append(f"  {s}: {self.qlabel}={mine} vs best other {other}: {'yes' if ok else 'no'}")
            return str(len(hits)), (f"\nPer {uw}, is `{self.qlabel}` strictly the most common label?\n" + "\n".join(rows)
                                    + f"\n{uw.capitalize()}s where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        if q == "author_top":
            c = state or Counter()
            a = self._argmax(c, self.authors)
            return a, f"\n`{self.qlabel}` items per author: {self._fmt(c, self.authors) or 'none'}; the most is {a} ({c.get(a, 0)})."
        if q == "relative":
            c = state or Counter(); a, b = c.get(self.qa, 0), c.get(self.qb, 0)
            ans = "more common than" if a > b else "less common than" if a < b else "equally common as"
            return ans, f"\n{self.qa}={a} vs {self.qb}={b} → {self.qa} is {ans} {self.qb}."
        if q == "dates_rep_k":
            c = state or {}
            hits = sorted(d for d, n in c.items() if n == self.qk)
            shown = ", ".join(hits[:12]) + (f", … (+{len(hits) - 12} more)" if len(hits) > 12 else "")
            return str(len(hits)), (f"\nDistinct {self.qmon} dates: {len(c)}; represented exactly {self.qk}x: "
                                    f"{shown or 'none'} → {len(hits)}.")
        if q == "before_after":
            c = state or {}
            bL, bO = c.get(f"before/{self.qlabel}", 0), c.get("before/other", 0)
            aL, aO = c.get(f"on_or_after/{self.qlabel}", 0), c.get("on_or_after/other", 0)
            fb = bL / max(1, bL + bO); fa = aL / max(1, aL + aO)
            ans = "more common" if fb > fa + 1e-9 else "less common" if fb < fa - 1e-9 else "the same frequency"
            return ans, (f"\nBefore {self.qdate}: {bL} of {bL + bO} items are `{self.qlabel}` ({fb:.0%}); on/after: {aL} of "
                         f"{aL + aO} ({fa:.0%}). Comparing shares, `{self.qlabel}` was {ans} before.")
        per = self._per_section(state)
        if q == "first_month_cmp":
            rows = []
            hit = None
            for m in self.sections:
                a, b = per[m].get(self.qa, 0), per[m].get(self.qb, 0)
                rows.append(f"  {m}: {self.qa}={a} vs {self.qb}={b}: {'yes' if a > b else 'no'}")
                if hit is None and a > b: hit = m
            return (hit or "none"), (f"\nIn chronological order, {self.qa} > {self.qb}?\n" + "\n".join(rows)
                                     + f"\nFirst month where yes: {hit or 'none'}.")
        if q == "sections_cmp":
            hits, rows = [], []
            for s in self.sections:
                a, b = per[s].get(self.qa, 0), per[s].get(self.qb, 0)
                if a > b: hits.append(s)
                rows.append(f"  {s}: {self.qa}={a} vs {self.qb}={b}: {'yes' if a > b else 'no'}")
            return str(len(hits)), (f"\nPer {uw}, {self.qa} > {self.qb}?\n" + "\n".join(rows)
                                    + f"\n{uw.capitalize()}s where yes: {', '.join(hits) or 'none'} → {len(hits)}.")
        counts = {s: per[s].get(self.qlabel, 0) for s in self.sections}
        best = max(counts.values()) if counts else 0
        ans = next((s for s in self.sections if counts[s] == best), self.sections[0])
        return ans, (f"\n`{self.qlabel}` per {uw}: " + ", ".join(f"{s}={n}" for s, n in counts.items())
                     + f"; the most is {ans} ({best}).")

    def _finalize(self, state):
        ans = self._resolve(state)[0]
        form = self.meta.get("answer_form")
        return f"{form}: {ans}" if form else ans      # fill the requested template exactly once
    def _finalize_note(self, state):
        note = self._resolve(state)[1]
        form = self.meta.get("answer_form")
        return note + (f"\nThe question asks for the form '{form}: [X]', so I box it that way." if form else "")

    # -- phrasing ---------------------------------------------------------------------------

    def _labelling(self):
        # the label set is stated in the task context and enumerated in the format contract; not repeated
        # here (the goal phrase appears twice per subtask and once in the preamble)
        return "judging each item's label"

    def _outer_note(self):
        return ("; an item's month is the month and year of its date" if self.key_mode == "date" else "")

    def _op_phrase(self):
        return {
            "count": f"{self._labelling()} and counting the `{self.qlabel}` ones",
            "most_common": f"{self._labelling()} and tallying the labels",
            "relative": f"{self._labelling()} and tallying the labels",
            "author_most": f"{self._labelling()} and tallying the labels of author {self.qauthor}'s items only",
            "author_top": f"{self._labelling()} and tallying, per author, the `{self.qlabel}` items",
            "author_count": "tallying how many items each author has (the label is irrelevant)",
            "author_cmp": f"{self._labelling()} and tallying, per author, the `{self.qlabel}` items",
            "author_least": f"{self._labelling()} and tallying the labels of author {self.qauthor}'s items only",
            "section_mode_count": f"{self._labelling()} and tallying each ({self.unit_word}, label) pair{(' (' + self._outer_note()[2:] + ')') if self._outer_note() else ''}",
            "sections_cmp": f"{self._labelling()} and tallying each ({self.unit_word}, label) pair{(' (' + self._outer_note()[2:] + ')') if self._outer_note() else ''}",
            "section_most": f"{self._labelling()} and tallying each ({self.unit_word}, label) pair{(' (' + self._outer_note()[2:] + ')') if self._outer_note() else ''}",
            "dates_rep_k": f"tallying how many items carry each exact date, for items dated in {self.qmon} only",
            "first_month_cmp": f"{self._labelling()} and tallying each (month, label) pair (an item's month is the month and year of its date)",
            "before_after": f"{self._labelling()} and tallying, separately for items dated before {self.qdate} and on/after it, how many are `{self.qlabel}` and how many are any other label",
        }[self.qtype]

    def _goal_phrase(self):
        return {
            "count": f"how many items are `{self.qlabel}` ({self._labelling()})",
            "most_common": f"the per-label tally ({self._labelling()})",
            "relative": f"the per-label tally ({self._labelling()})",
            "author_most": f"the per-label tally over ONLY author {self.qauthor}'s items ({self._labelling()})",
            "author_top": f"the per-author tally of `{self.qlabel}` items ({self._labelling()})",
            "author_count": "the per-author tally of items (no label judgement needed)",
            "author_cmp": f"the per-author tally of `{self.qlabel}` items ({self._labelling()})",
            "author_least": f"the per-label tally over ONLY author {self.qauthor}'s items ({self._labelling()})",
            "section_mode_count": f"the per-({self.unit_word}, label) tally ({self._labelling()}{self._outer_note()})",
            "sections_cmp": f"the per-({self.unit_word}, label) tally ({self._labelling()}{self._outer_note()})",
            "section_most": f"the per-({self.unit_word}, label) tally ({self._labelling()}{self._outer_note()})",
            "dates_rep_k": f"the per-date tally over ONLY items dated in {self.qmon} (how many items carry each exact date)",
            "first_month_cmp": f"the per-(month, label) tally ({self._labelling()}; an item's month is the month and year of its date)",
            "before_after": f"the before/on-or-after {self.qdate} tally of `{self.qlabel}` vs other labels ({self._labelling()})",
        }[self.qtype]

    def _shape_reason(self):
        q, uw = self.qtype, self.unit_word
        derive = " (the items carry dates; an item's month is the month and year of its date)" if self.key_mode == "date" else ""
        if q in ("sections_cmp", "first_month_cmp"):
            return f"The question compares labels WITHIN each {uw}{derive}, so the state is a per-({uw}, label) tally, not a per-label one."
        if q == "section_most":
            return f"The question compares one label's count ACROSS {uw}s{derive}, so the state is a per-({uw}, label) tally, not a per-label one."
        if q == "section_mode_count":
            return (f"The question asks in how many {uw}s `{self.qlabel}` beats EVERY other label{derive}, so the state is a "
                    f"per-({uw}, label) tally over all labels — not a per-{uw} count of `{self.qlabel}` alone.")
        if q == "author_cmp":
            return (f"The question compares two named authors on `{self.qlabel}`, so the state is a per-author count of "
                    f"`{self.qlabel}` items — the same tally as for any author question, read off for {self.qa1} and {self.qa2} at the end.")
        if q == "author_count":
            return ("The question asks which author has the most items, so the state is a per-author count — the labels "
                    "play no part, and the authors are whatever names or ids the `[by …]` tags carry.")
        if q == "author_least":
            return f"The question is restricted to author {self.qauthor}'s items, so the state is a per-label tally over those items only."
        if q == "before_after":
            return (f"The question compares the label's share before vs on/after {self.qdate}, so the state keeps two periods "
                    f"x (`{self.qlabel}` vs other) — four counts — not one overall tally.")
        if q == "author_top":
            return f"The question asks WHICH AUTHOR has the most `{self.qlabel}` items, so the state is a per-author count of `{self.qlabel}` items."
        if q == "author_most":
            return f"The question is restricted to author {self.qauthor}'s items, so the state is a per-label tally over those items only."
        if q == "dates_rep_k":
            return f"The question asks how many dates in {self.qmon} appear exactly {self.qk} time{'s' if self.qk != 1 else ''}, so the state is a per-DATE count, reduced at the root."
        if q == "count":
            return f"The question asks for one label's total, so the state is a single count of `{self.qlabel}` items."
        return "The question asks about labels over the whole document, so a per-label tally is the right state."

    def _state_format(self):
        labs = ", ".join(f"`{l}`" for l in self.labels)
        q = self.qtype
        if q == "count":
            return "the partial as a single integer"
        if q in ("most_common", "relative", "author_most", "author_least"):
            return f"the partial tally as `<label>:<count>` entries joined by `|`, where `<label>` is exactly one of {labs}"
        if q in ("author_top", "author_count", "author_cmp"):
            # OPEN key space: authors are read off the text, never enumerated by the root (run 9w on OOLONG-user:
            # the root invented `User A`..`User E` and every leaf returned `none:38`)
            return (f"the partial tally as `<author>:<count>` entries joined by `|`, where `<author>` is the author "
                    f"exactly as written in the item's `[by …]` tag")
        if q == "dates_rep_k":
            return f"the partial tally as `<date>=<count>` entries joined by `|`, where `<date>` is a date in {self.qmon} written exactly as in the item's tag"
        if q == "before_after":
            return (f"the partial as `<period>/<which>=<count>` entries joined by `|`, where `<period>` is `before` or "
                    f"`on_or_after` (relative to {self.qdate}) and `<which>` is `{self.qlabel}` or `other`")
        outer = (f"the line's `S<n>` tag" if self.key_mode == "section"
                 else "the month and year of the item's date, one of " + ", ".join(f"`{m}`" for m in self.sections))
        return (f"the partial tally as `<{self.unit_word}>/<label>=<count>` entries joined by `|`, where "
                f"`<{self.unit_word}>` is {outer}; and `<label>` is exactly one of {labs}")

    def _combine_phrase(self):
        return {"count": "sum", "most_common": "merge", "relative": "merge", "author_most": "merge",
                "author_top": "merge", "author_count": "merge", "author_least": "merge", "author_cmp": "merge", "dates_rep_k": "merge (add per date)", "before_after": "merge (add per period/which key)"}.get(
            self.qtype, f"merge (add per {self.unit_word}/label key)")

    def _empty_phrase(self):
        return "  (no item starts here)"

    def _partial_header(self, a, b, n):
        if self.qtype == "dates_rep_k":
            return (f"Tallying the {self._n_units(n)} whose line STARTS in {a}..{b} by exact date ({self._op_phrase()}; the "
                    f"trailing reads only finish the last line, and any item starting at/after {b} belongs to the next range)")
        verb = "counting" if self.qtype == "count" else "tallying"
        return (f"Judging the {self._n_units(n)} whose line STARTS in {a}..{b} one at a time and {verb} them "
                f"({self._op_phrase()}; the trailing reads only finish the last line, and any item starting "
                f"at/after {b} belongs to the next range)")

    def _finish_phrase(self, end):
        return f"to finish the last item (its line may run past {end})"
