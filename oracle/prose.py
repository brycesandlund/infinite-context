"""Oracles for the hidden-sentences-in-novel tasks (tasks/niah): niah_multi and vt_novel.

Both are thin hook-sets on ScaffoldOracle's `dict` state — the same machinery as synth_sumby /
synth_varchain — with mechanical (scripted, faithful) leaves:

- NiahMultiOracle: binary collect-then-resolve. Each leaf folds the hidden facts starting in its
  range into {key: value, QUERY: key_j}; combine is dict-merge; the ROOT resolves the multi-hop
  lookup state[state["QUERY"]]. The RULER multikey/multiquery analog.
- VtNovelOracle: left-fold variable tracking (non-associative, like synth_varchain): thread the
  bindings dict through the chain; root reports every variable whose final value is the target
  (set-graded). The RULER vt analog.
"""

from __future__ import annotations

from oracle.base import ScaffoldOracle

_NOT_FOUND = "answer not found in document"


class NiahMultiOracle(ScaffoldOracle):
    name = "niah_multi_oracle"

    def _kind(self):
        return "dict"

    # Values are strings (a 7-digit number or a key name), so override the base dict
    # serializer, which parses values as ints and would DROP the QUERY entry.
    def _ser_state(self, state) -> str:
        if not state:
            return "none"
        return "|".join(f"{k}={v}" for k, v in sorted(state.items()))

    def _parse_state(self, s):
        s = (s or "").strip()
        if not s or s.lower() == "none":
            return {}
        d = {}
        for part in s.split("|"):
            k, sep, v = part.partition("=")
            if sep:
                d[k.strip()] = v.strip()
        return d

    def _acc_step(self, acc, s):
        _, _, idx, key, value, is_query = s
        if is_query:
            acc = {**acc, "QUERY": key}
            return acc, f"- lookup instruction: the key to look up is {key}  → {self._ser_state(acc)}"
        acc = {**acc, key: value}
        return acc, f"- magic number for {key} = {value}  → {self._ser_state(acc)}"

    def _combine(self, states):
        out = {}
        for st in states:
            out.update(st or {})
        return out

    def _finalize(self, state) -> str:
        q = (state or {}).get("QUERY")
        return state.get(q, _NOT_FOUND) if q else _NOT_FOUND

    def _finalize_note(self, state) -> str:
        q = (state or {}).get("QUERY")
        if not q or q not in state:
            return "\nThe collected facts do not pin down the lookup."
        return f"\nThe lookup instruction names {q}; the magic number for {q} is {state[q]}."

    # Phrases must read naturally inside the templated subtask: "Over the {unit} STARTING in
    # tokens a..b, compute {goal}." — keep them noun-phrases with no dangling clauses.
    def _op_phrase(self) -> str:
        return "collecting each stated magic number (key=number) and the one lookup instruction"

    def _unit(self) -> str:
        return "hidden fact sentences"

    def _goal_phrase(self) -> str:
        return "the hidden facts (each key's magic number, plus which key to look up)"

    def _combine_phrase(self) -> str:
        return "merge"

    def _empty_phrase(self) -> str:
        return "  (no hidden fact sentence starts here)"

    def _partial_header(self, a, b, n) -> str:
        return (f"Collecting the {n} hidden fact sentence(s) whose line STARTS in {a}..{b} "
                f"({self._op_phrase()}; the trailing reads only finish a sentence straddling {b}; "
                f"one starting at/after {b} belongs to the next range)")


class VtNovelOracle(ScaffoldOracle):
    name = "vt_novel_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.target = int(self.meta["target_value"])
        if self.strategy == "binary":
            # 'VAR B = VAR A' needs A's CURRENT value: sequential state, no associative combine.
            raise ValueError("vt_novel has no binary oracle; use left_fold")

    def _kind(self):
        return "dict"

    def _acc_step(self, acc, s):
        _, _, idx, name, rhs, is_ref = s
        if is_ref:
            val = acc.get(rhs, 0)
            return {**acc, name: val}, f"- VAR {name} = VAR {rhs}  → {name}={val} (copied {rhs}'s current value)"
        return {**acc, name: int(rhs)}, f"- VAR {name} = {rhs}  → {name}={rhs}"

    def _finalize(self, state) -> str:
        names = sorted(n for n, v in (state or {}).items() if v == self.target)
        return ", ".join(names) if names else "none"

    def _finalize_note(self, state) -> str:
        names = sorted(n for n, v in (state or {}).items() if v == self.target)
        return f"\nVariables whose final value is {self.target}: {', '.join(names) or 'none'}."

    def _op_phrase(self) -> str:
        return "applying each VAR assignment in order (VAR A = VAR B copies B's current value)"

    def _unit(self) -> str:
        return "VAR assignment lines"

    def _goal_phrase(self) -> str:
        return "the variable bindings"

    def _empty_phrase(self) -> str:
        return "  (no VAR assignment starts here)"
