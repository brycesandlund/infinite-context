"""RealDocOracle — realdoc_count over real novel prose (tasks/realdoc).

Each 'record' is one occurrence of the queried word (contributing +1); the binary
tree-reduce sums per-chunk counts (a bounded-associative int combine). Only the leaf-op
differs from the synthetic count — here it's 'find the word in real prose' — so this is
just a thin set of hooks on ScaffoldOracle (no fold; int state; default sum combine)."""

from __future__ import annotations

from oracle.base import ScaffoldOracle


class RealDocOracle(ScaffoldOracle):
    name = "realdoc_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy="binary")
        self.entity = self.meta["entity"]

    def _leaf_value(self, recs):
        return len(recs)                       # every occurrence counts +1

    def _unit(self):
        return f"occurrences of the word '{self.entity}'"

    def _goal_phrase(self):
        return "the total count"

    def _contrib(self, s):
        idx, snip = s[2], s[3]
        return f"- occurrence {idx}: …{snip}…"

    def _partial_header(self, a, b, n) -> str:
        return (f"Counting the {n} occurrences of '{self.entity}' that START in tokens {a}..{b} "
                f"(the trailing reads only finish an occurrence straddling {b}; an occurrence "
                f"starting at/after {b} belongs to the next range)")

    def _empty_phrase(self) -> str:
        return f"  (no occurrences of '{self.entity}' start here)"
