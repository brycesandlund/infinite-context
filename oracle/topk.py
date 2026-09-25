"""TopKOracle — k most frequent words over an open vocabulary (tasks/topk).

The contract this task exists to teach: a range returns a PRUNED `word:count` tally (its most
frequent words, at least the top m), never a bare word list; the merge ADDS counts; only the root
ranks. Leaves narrate per LINE (a 250-token leaf holds ~80 words) and then state the pruning.
"""

from __future__ import annotations

from collections import Counter

from oracle.base import ScaffoldOracle


class TopKOracle(ScaffoldOracle):
    name = "topk_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.k, self.layout = self.meta["k"], self.meta["layout"]
        # K = budget / 50 (v16): every partial keeps its top-K words WITH one-offs, exact counts, ties in reading
        # order. The old contract (words seen 2+ times, at most 10-16) was exact only when hot words are dense
        # per leaf; RULER cwe at 40K+ has a common word ~0.4x per 500-token leaf, so leaves dropped it as a
        # one-off (18w cwe 1.00 / 0.74 / 0.58 at 10K / 40K / 80K). Simulated on RULER's construction, B/50 recovers
        # 0.97-0.99 at 10-80K (8K budget, K=160). The arithmetic is stated in the contract so it carries to any budget.
        self.keep_m = max(2 * self.k, self.budget // 50)
        if self.strategy != "binary":
            raise ValueError("synth_topk prunes its partials; a fold would drop counts — binary only")

    def _kind(self):
        return "counter"

    def _unit(self):
        return "words"

    # -- leaf: tally per line, then prune -------------------------------------------------------

    def _acc_step(self, acc, s):
        raise NotImplementedError("synth_topk narrates per block of lines; see _accumulate")

    def _accumulate(self, recs, acc):
        acc = Counter(acc) if acc else Counter()
        lines, by_line = [], {}
        for s in recs:
            by_line.setdefault(s[0], []).append(s[3])
        for words in by_line.values():
            acc.update(words)
        # Per block of lines, name the words that RECUR in this range (with their block counts) and count the rest; the
        # words seen once are not re-listed here — they appear once, in the partial (v16 keeps them there, in reading
        # order). Listing every word per line AND in a "range tally" AND in the partial put coded-word leaves at 3.6K
        # tokens on a 3K budget.
        starts = list(by_line)
        for i in range(0, len(starts), 3):                  # narrate in blocks of 3 lines
            blk = starts[i:i + 3]
            c = Counter(w for st in blk for w in by_line[st])
            keep = [(w, n) for w, n in c.items() if acc[w] >= 2]
            others = sum(n for w, n in c.items() if acc[w] < 2)
            shown = ", ".join(f"{w}×{n}" if n > 1 else w for w, n in keep)
            span = f"line at token {blk[0]}" if len(blk) == 1 else f"lines at tokens {blk[0]}–{blk[-1]}"
            lines.append(f"- {span}: {shown or '(no recurring words)'}"
                         + (f"; +{others} word{'s' if others != 1 else ''} seen once" if others else ""))
        if acc:
            lines.append(f"{len(acc)} distinct words here; " + (
                f"keeping the top {self.keep_m} by count, ties in reading order (words seen once included)."
                if len(acc) > self.keep_m else f"all of them fit in {self.keep_m}, so all are kept (words seen once included)."))
        return acc, lines

    def _top(self, c):
        # v16: one-offs kept; stable sort -> ties stay in reading order (Counter keeps first-seen order, and the
        # merge adds the left child before the right)
        return sorted(c.items(), key=lambda kv: -kv[1])[:self.keep_m]

    def _ser_counter(self, items):
        return "|".join(f"{w}:{n}" for w, n in items) or "none"

    # serialize PRUNED (parents only ever see the top m); parse is the base counter parser
    def _ser_state(self, state):
        if not state:
            return "none"
        return self._ser_counter(self._top(Counter(state)))

    def _combine(self, states):
        tot = Counter()
        for s in states: tot += Counter(s or {})
        return tot

    def _finalize(self, state):
        ranked = sorted(Counter(state or {}).items(), key=lambda kv: (-kv[1], kv[0]))[:self.k]
        return ", ".join(w for w, _ in ranked)

    def _finalize_note(self, state):
        ranked = sorted(Counter(state or {}).items(), key=lambda kv: (-kv[1], kv[0]))
        top = ", ".join(f"{w} ({n})" for w, n in ranked[:self.k])
        nxt = f"; next: {ranked[self.k][0]} ({ranked[self.k][1]})" if len(ranked) > self.k else ""
        return f"\nRanking the merged counts: top {self.k} = {top}{nxt}."

    # -- phrasing ---------------------------------------------------------------------------

    def _op_phrase(self):
        return ("tallying how many times each word appears" +
                (" (ignoring the list numbers)" if self.layout == "numbered" else " (ignoring the dots)" if self.layout == "dotted" else ""))

    def _goal_phrase(self):
        return "the per-word tally of the most frequent words"

    def _combine_phrase(self):
        return "merge (add per word)"

    def _shape_reason(self):
        return ("The question asks for the most frequent words over an open vocabulary, so each partial is a pruned "
                "`word:count` tally (never a bare word list) and only the root ranks.")

    def _state_format(self):
        return (f"the partial as `<word>:<count>` entries joined by `|` for the {self.budget}/50 = {self.keep_m} most "
                f"frequent words in the range (exact counts; words seen once included; ties kept in reading order) "
                f"— a tally with counts, never a bare word list"
                + (", and never list numbers as words" if self.layout == "numbered" else ""))

    def _empty_phrase(self):
        return "  (no line starts here)"

    def _partial_header(self, a, b, n):
        return (f"Tallying the words on the lines that START in {a}..{b} ({self._op_phrase()}; the trailing "
                f"reads only finish the last line, and any line starting at/after {b} belongs to the next range)")

    def _finish_phrase(self, end):
        return f"to finish the last line (it may run past {end})"
