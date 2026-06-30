"""Scripted oracles — they play the canonical decomposition through `run_agent` to
produce clean SFT warm-start traces.

- ScaffoldOracle (base.py): the shared recursive-decomposition scaffold.
- SynthOracle / RealDocOracle / BookQAOracle: peer subclasses for the new corpus
  (abstract synthetic ops / real-novel counting / real-novel QA retrieval).
- OracleBackend / OolongOracle (classic.py): the original RULER + OOLONG oracles.

`make_oracle(problem, ...)` dispatches to the right one for a problem's task.
"""

from oracle.base import ScaffoldOracle
from oracle.bookqa import BookQAOracle
from oracle.realdoc import RealDocOracle
from oracle.synth import SynthOracle


def make_oracle(problem, tokenizer, *, budget, max_chunk_tokens, strategy=None, leaf_model=None):
    """Pick the scripted oracle that best decomposes this task.

    - bookqa / realdoc_* / synth_*: the new ScaffoldOracle-based corpus. `strategy`
      (the per-task training knob) only applies to synth_*; None = the task default.
    - oolong_* : the unified OOLONG show-your-work oracle.
    - everything else (RULER niah/vt/cwe/fwe): the split-and-delegate OracleBackend.

    `leaf_model` (a ModelBackend) only applies to bookqa: it makes the leaf JUDGMENT a real
    model call (faithful free-form QA). None keeps bookqa's scripted span fallback.
    """
    task = problem.task
    if task == "bookqa":
        return BookQAOracle(problem, tokenizer, budget=budget,
                            max_chunk_tokens=max_chunk_tokens, leaf_model=leaf_model)
    if task.startswith("realdoc_"):
        return RealDocOracle(problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens)
    if task.startswith("synth_"):
        return SynthOracle(problem, tokenizer, budget=budget,
                           max_chunk_tokens=max_chunk_tokens, strategy=strategy)

    from oracle.classic import OolongOracle, OracleBackend  # heavier (datetime/regex); lazy

    if task in ("oolong_counting", "oolong_user", "oolong_temporal"):
        return OolongOracle(problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens)
    return OracleBackend(problem, tokenizer, budget=budget, max_chunk_tokens=max_chunk_tokens)


__all__ = ["make_oracle", "ScaffoldOracle", "SynthOracle", "RealDocOracle", "BookQAOracle"]
