"""Oracles for the hidden-sentences-in-a-haystack tasks (tasks/niah): niah_multi and vt_novel.

Both are thin hook-sets on ScaffoldOracle's `dict` state — the same machinery as synth_sumby /
synth_varchain — with mechanical (scripted, faithful) leaves. Each task is a VARIANT FAMILY (the
variant is drawn per problem and carried in metadata), so the oracle reads the mode from `meta`:

- NiahMultiOracle (binary collect-then-resolve). Leaves fold the hidden facts starting in their
  range into {key: value[,value…], QUERY: key_j}; combine is a per-key union merge (multivalue
  keys hold several comma-joined values); the ROOT resolves by mode:
    hidden     -> state[state["QUERY"]]           (a real multi-hop lookup)
    explicit   -> state[target_key]
    multiquery -> state[k] for each asked key      (set answer)
    multivalue -> all values under the one key     (set answer)
- VtNovelOracle (left-fold variable tracking; non-associative like synth_varchain). Thread the
  bindings dict through the chains; the root answers by qtype:
    which_vars  -> every variable whose final value is the target (set answer)
    final_value -> the final value of the queried variable (exact)
"""

from __future__ import annotations

from oracle.base import ScaffoldOracle

_NOT_FOUND = "answer not found in document"


def _union(a: str, b: str) -> str:
    """Comma-joined value sets (values never contain commas)."""
    return ",".join(sorted(set(a.split(",")) | set(b.split(","))))


class NiahMultiOracle(ScaffoldOracle):
    name = "niah_multi_oracle"

    def __init__(self, problem, tokenizer, *, budget, max_chunk_tokens, strategy=None):
        super().__init__(problem, tokenizer, budget=budget,
                         max_chunk_tokens=max_chunk_tokens, strategy=strategy)
        self.mode = self.meta.get("mode", "hidden")
        self.target_keys = list(self.meta.get("target_keys", []))
        self.vword = "number" if self.meta.get("value_type", "numbers") == "numbers" else "uuid"

    def _kind(self):
        return "dict"

    # Values are strings (a number, a uuid, or a key name), so override the base dict
    # serializer, which parses values as ints and would DROP non-numeric entries.
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

    # Per-record lines show only the DELTA (the binding just added), not the whole accumulator:
    # with uuid keys/values and up to ~6 needles the full dict is ~150 tokens, and reprinting it
    # per line pushed fold nodes past the 3000 budget. The fold node prints the full accumulator
    # once at the end ("→ accumulator = …"), which is where the model needs it.
    def _acc_step(self, acc, s):
        _, _, idx, key, value, is_query = s
        if is_query:
            acc = {**acc, "QUERY": key}
            return acc, f"- lookup instruction: the key to look up is {key}  → QUERY={key}"
        if key in acc:   # multivalue: a second value for a key we've already seen
            acc = {**acc, key: _union(acc[key], value)}
            return acc, f"- another magic {self.vword} for {key}: {value}  → {key}={acc[key]}"
        acc = {**acc, key: value}
        return acc, f"- magic {self.vword} for {key} = {value}  → {key}={value}"

    def _combine(self, states):
        out: dict = {}
        for st in states:
            for k, v in (st or {}).items():
                out[k] = _union(out[k], v) if k in out and k != "QUERY" else v
        return out

    def _resolve(self, state):
        """-> (answer_str, note) for the root, by mode. Set answers are ', '-joined."""
        state = state or {}
        if self.mode == "hidden":
            q = state.get("QUERY")
            if not q or q not in state:
                return _NOT_FOUND, "The collected facts do not pin down the lookup."
            return state[q].replace(",", ", "), f"The lookup instruction names {q}; the magic {self.vword} for {q} is {state[q]}."
        if self.mode == "explicit":
            k = self.target_keys[0]
            return (state.get(k, _NOT_FOUND).replace(",", ", "),
                    f"The magic {self.vword} for {k} is {state.get(k, _NOT_FOUND)}.")
        if self.mode == "multiquery":
            vals = [state.get(k, _NOT_FOUND) for k in self.target_keys]
            pairs = "; ".join(f"{k} = {v}" for k, v in zip(self.target_keys, vals))
            return ", ".join(vals), f"Reading off each asked key: {pairs}."
        k = self.target_keys[0]   # multivalue
        vals = state.get(k, _NOT_FOUND).split(",")
        return ", ".join(vals), f"All magic {self.vword}s stated for {k}: {', '.join(vals)}."

    def _finalize(self, state) -> str:
        return self._resolve(state)[0]

    def _finalize_note(self, state) -> str:
        return "\n" + self._resolve(state)[1]

    # Phrases must read naturally inside the templated subtask: "Over the {unit} STARTING in
    # tokens a..b, compute {goal}." — noun-phrases, no dangling clauses.
    def _op_phrase(self) -> str:
        return f"collecting each stated magic {self.vword} (key=value) and any lookup instruction"

    def _unit(self) -> str:
        return "hidden fact sentences"

    def _verb(self) -> str:
        return "collect"

    def _shape_reason(self) -> str:
        if self.mode == "hidden":
            return (f"The question asks for the magic {self.vword} of a key that only a hidden lookup instruction names, so "
                    f"the state is the set of collected key=value facts plus that instruction, resolved only at the root.")
        if self.mode == "multivalue":
            return (f"The question asks for EVERY magic {self.vword} stated for one key, so the state collects all values "
                    f"per key (several values under one key are comma-joined), merged at the root.")
        return (f"The question asks for specific keys' magic {self.vword}s, so the state is the set of collected "
                f"key=value facts, resolved only at the root.")

    def _state_format(self) -> str:
        return (f"the collected facts as `<key>=<magic {self.vword}>` entries joined by `|` (several values for one "
                f"key comma-joined: `<key>=<v1>,<v2>`), where each `<key>` is the key name exactly as written in the "
                f"text, and a lookup instruction as `QUERY=<key>`")

    def _goal_phrase(self) -> str:
        # Name what the question asks for, so every subtask carries the KEY(s) down the tree
        # (run 4: the model dropped the key — "compute the hidden number" — and leaves returned
        # whichever magic number they saw). Other keys are still collected: the leaf can't know
        # in advance which sentence matters for a hidden-query lookup.
        if self.mode == "explicit":
            return (f"the magic {self.vword} stated for {self.target_keys[0]} (collecting every "
                    f"stated key=magic {self.vword} fact in the range)")
        if self.mode == "multiquery":
            return (f"the magic {self.vword}s stated for {', '.join(self.target_keys)} (collecting "
                    f"every stated key=magic {self.vword} fact in the range)")
        if self.mode == "multivalue":
            return (f"every magic {self.vword} stated for {self.target_keys[0]} (collecting every "
                    f"stated key=magic {self.vword} fact in the range)")
        return f"the hidden facts (each key's magic {self.vword}, plus any lookup instruction)"

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
        self.qtype = self.meta.get("qtype", "which_vars")
        self.query_var = self.meta.get("query_var")
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

    def _holders(self, state):
        return sorted(n for n, v in (state or {}).items() if v == self.target)

    def _finalize(self, state) -> str:
        if self.qtype == "final_value":
            return str((state or {}).get(self.query_var, 0))
        names = self._holders(state)
        return ", ".join(names) if names else "none"

    def _finalize_note(self, state) -> str:
        if self.qtype == "final_value":
            return f"\nFinal value of VAR {self.query_var}: {(state or {}).get(self.query_var, 0)}."
        return f"\nVariables whose final value is {self.target}: {', '.join(self._holders(state)) or 'none'}."

    def _verb(self) -> str:
        return "collect"

    def _sequential_reason(self) -> str:
        return ("a `VAR A = VAR B` line copies B's value AT THAT POINT, and a variable can be "
                "reassigned later, which makes the order of the bindings matter")

    def _op_phrase(self) -> str:
        return "applying each VAR assignment in order (VAR A = VAR B copies B's current value)"

    def _unit(self) -> str:
        return "VAR assignment lines"

    def _goal_phrase(self) -> str:
        return "the variable bindings"

    def _empty_phrase(self) -> str:
        return "  (no VAR assignment starts here)"
