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
        # ONE PASS, FIRST SIGHT (v16). Old leaves wrote per-"line" blocks with range-wide counts, "+N one-off words" and
        # "M words appear once" — totals stated before anything was written; on 18w's RULER cwe/fwe leaves those were 0%
        # exact, per-block counts 20%, the partial 35%, and the "lines" were invented (RULER docs are one line). Now the
        # leaf walks the words in reading order and writes each word ONCE, where it first appears, marked with how many
        # times it appears in the range (`×n` when more than once); a word already written is not repeated. The partial
        # is then the top K of that list by count, ties in the list's order.
        acc = Counter(acc) if acc else Counter()
        c = Counter(s[3] for s in recs)                     # insertion order = first-seen order
        acc.update(c)
        lines = []
        if c:
            lines.append("In reading order, each word once where it first appears (×n = appears n times here): "
                         + " ".join(f"{w}×{n}" if n > 1 else w for w, n in c.items()))
        if len(acc) > self.keep_m:
            lines.append(f"More than {self.keep_m} different words, so the partial keeps the {self.keep_m} most frequent "
                         f"(ties in reading order).")
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
        return "  (no word starts here)"

    def _partial_header(self, a, b, n):
        return (f"Going through the words that START in {a}..{b} in reading order ({self._op_phrase()}; "
                f"the trailing reads only finish the last word, and any word starting at/after {b} belongs to the next range)")

    def _finish_phrase(self, end):
        return f"to finish the last word (it may run past {end})"
