"""Long-context task suite.

A `Problem` is what gets shipped to a rollout: a long-document `document_tokens`
haystack that the agent must access via `read_chunk`, a short `question` the
agent receives as a user message, accepted `gold_answers`, a task identifier
for grader dispatch, and `task_context` — a per-task string pinned to every
agent's system prompt (so a freshly-spawned subagent already knows the task's
label space / format without the parent re-explaining it).

Today: RULER's NIAH-family ×8, VT, CWE, FWE. Adding a task family is a
generator + registry entry; the agent loop, tools, and training code don't
move.
"""

from tasks.base import (
    GradingMode,
    Problem,
    grade_answer,
    is_answer_correct,
    normalize_answer,
    split_list,
)
from tasks.corpus import load_pg_essays_text, NOISE_SENTENCE
from tasks.registry import eval_grading_mode, grading_mode, list_tasks, make_problem

__all__ = [
    "GradingMode",
    "NOISE_SENTENCE",
    "Problem",
    "eval_grading_mode",
    "grade_answer",
    "grading_mode",
    "is_answer_correct",
    "list_tasks",
    "load_pg_essays_text",
    "make_problem",
    "normalize_answer",
    "split_list",
]
