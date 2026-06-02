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
from tasks.ruler import RULER_TASKS, make_ruler_problem


# All 13 RULER tasks resolve through the same vendored builder; per-task
# config lives in tasks/ruler/constants.py (RULER_TASKS). Each entry binds
# the task name as the first arg so the registry's invocation signature stays
# uniform: (corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem.
def _bind_ruler(name: str):
    def gen(corpus_tokens, tokenizer, doc_size_tokens, seed):
        return make_ruler_problem(name, corpus_tokens, tokenizer, doc_size_tokens, seed)
    return gen


_GENERATORS = {name: _bind_ruler(name) for name in RULER_TASKS}


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
}


# Eval graders: RULER's official string_match. Directly comparable to NVIDIA's
# published scores. Used at post-training eval time, NOT for the reward signal.
_EVAL_GRADING_MODES: dict[str, GradingMode] = {
    "niah_single_1": "ruler_all", "niah_single_2": "ruler_all", "niah_single_3": "ruler_all",
    "niah_multikey_1": "ruler_all", "niah_multikey_2": "ruler_all", "niah_multikey_3": "ruler_all",
    "niah_multivalue": "ruler_all", "niah_multiquery": "ruler_all",
    "vt": "ruler_all", "cwe": "ruler_all", "fwe": "ruler_all",
    "qa_1": "ruler_part", "qa_2": "ruler_part",
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


def eval_grading_mode(task: str) -> GradingMode:
    """RULER-official grader (substring). Use for eval numbers comparable to
    NVIDIA's leaderboard."""
    if task not in _EVAL_GRADING_MODES:
        raise ValueError(f"Unknown task: {task!r}. Known: {list_tasks()}")
    return _EVAL_GRADING_MODES[task]
