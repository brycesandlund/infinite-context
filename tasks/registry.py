"""Task dispatch.

`make_problem(task, ...)` → Problem for any registered task.
`grading_mode(task)` → the GradingMode used for the *training* reward signal
                       (strict — exact / set / numeric). Cleaner gradient.
`eval_grading_mode(task)` → the GradingMode used to report *eval* numbers
                            (RULER's official substring matchers). Comparable
                            to NVIDIA's published leaderboard.

Adding a new task family: implement a generator with signature
`(corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem`, register it
in `_GENERATORS`, and add its training+eval grader modes below.
"""

from __future__ import annotations

from tasks.base import GradingMode, Problem
from tasks.oolong import OOLONG_TASKS, make_oolong_problem
from tasks.ruler import RULER_TASKS, make_ruler_problem
from tasks.synth import SYNTH_TASKS, make_synth_problem
from tasks.realdoc import REALDOC_TASKS, make_realdoc_problem
from tasks.bookqa import BOOKQA_TASKS, make_bookqa_problem


# All 13 RULER tasks resolve through the same vendored builder; per-task
# config lives in tasks/ruler/constants.py (RULER_TASKS). Each entry binds
# the task name as the first arg so the registry's invocation signature stays
# uniform: (corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem.
def _bind(make_fn, name: str):
    def gen(corpus_tokens, tokenizer, doc_size_tokens, seed):
        return make_fn(name, corpus_tokens, tokenizer, doc_size_tokens, seed)
    return gen


_GENERATORS = {name: _bind(make_ruler_problem, name) for name in RULER_TASKS}
_GENERATORS.update({name: _bind(make_oolong_problem, name) for name in OOLONG_TASKS})
_GENERATORS.update({name: _bind(make_synth_problem, name) for name in SYNTH_TASKS})
_GENERATORS.update({name: _bind(make_realdoc_problem, name) for name in REALDOC_TASKS})
_GENERATORS.update({name: _bind(make_bookqa_problem, name) for name in BOOKQA_TASKS})


# Training graders: strict equality / set / numeric. Clean reward signal —
# no spurious credit for the chain-of-thought happening to mention the gold
# answer in passing. (NIAH single-answer = exact; multi-answer = set.)
_TRAIN_GRADING_MODES: dict[str, GradingMode] = {
    "niah_single_1": "exact", "niah_single_2": "exact", "niah_single_3": "exact",
    "niah_multikey_1": "exact", "niah_multikey_2": "exact", "niah_multikey_3": "exact",
    "niah_multivalue": "set", "niah_multiquery": "set",
    "vt": "set", "cwe": "set", "fwe": "set",
    # QA (qa_1, qa_2) deferred to held-out eval; substring match.
    "qa_1": "ruler_part", "qa_2": "ruler_part",
    # OOLONG-synth families: per-PROBLEM grader (Problem.grading_mode) since
    # answer types are mixed within a family; these are fallbacks only.
    "oolong_counting": "oolong_exact", "oolong_user": "oolong_exact", "oolong_temporal": "oolong_exact",
    # Synthetic decomposition tasks: integer answers -> numeric partial credit.
    "synth_sum": "numeric", "synth_count": "numeric", "synth_max": "numeric", "synth_runreset": "numeric",
    "realdoc_count": "numeric",
    "bookqa": "ruler_part",
}


# Eval graders: RULER's official string_match. Directly comparable to NVIDIA's
# published scores. Used at post-training eval time, NOT for the reward signal.
_EVAL_GRADING_MODES: dict[str, GradingMode] = {
    "niah_single_1": "ruler_all", "niah_single_2": "ruler_all", "niah_single_3": "ruler_all",
    "niah_multikey_1": "ruler_all", "niah_multikey_2": "ruler_all", "niah_multikey_3": "ruler_all",
    "niah_multivalue": "ruler_all", "niah_multiquery": "ruler_all",
    "vt": "ruler_all", "cwe": "ruler_all", "fwe": "ruler_all",
    "qa_1": "ruler_part", "qa_2": "ruler_part",
    # OOLONG uses its own metric (exact categorical / numeric partial credit),
    # decided per-problem via Problem.grading_mode; these are fallbacks only.
    "oolong_counting": "oolong_exact", "oolong_user": "oolong_exact", "oolong_temporal": "oolong_exact",
    "synth_sum": "numeric", "synth_count": "numeric", "synth_max": "numeric", "synth_runreset": "numeric",
    "realdoc_count": "numeric",
    "bookqa": "ruler_part",
}


def list_tasks() -> list[str]:
    return list(_GENERATORS.keys())


def make_problem(
    task: str,
    corpus_tokens: list[int],
    tokenizer,
    doc_size_tokens: int,
    seed: int,
) -> Problem:
    if task not in _GENERATORS:
        raise ValueError(f"Unknown task: {task!r}. Known: {list_tasks()}")
    return _GENERATORS[task](corpus_tokens, tokenizer, doc_size_tokens, seed)


def grading_mode(task: str) -> GradingMode:
    """Training-reward grader (strict). Use for the RL signal."""
    if task not in _TRAIN_GRADING_MODES:
        raise ValueError(f"Unknown task: {task!r}. Known: {list_tasks()}")
    return _TRAIN_GRADING_MODES[task]


def resolve_grading_mode(problem) -> GradingMode:
    """Training-reward grader for a specific problem: its softer reward override
    (Problem.reward_mode) if set, else its eval override (Problem.grading_mode),
    else the registry default for its task."""
    return problem.reward_mode or problem.grading_mode or grading_mode(problem.task)


def resolve_eval_grading_mode(problem) -> GradingMode:
    """Eval grader for a specific problem (per-problem override, else registry)."""
    return problem.grading_mode or eval_grading_mode(problem.task)


def eval_grading_mode(task: str) -> GradingMode:
    """RULER-official grader (substring). Use for eval numbers comparable to
    NVIDIA's leaderboard."""
    if task not in _EVAL_GRADING_MODES:
        raise ValueError(f"Unknown task: {task!r}. Known: {list_tasks()}")
    return _EVAL_GRADING_MODES[task]
