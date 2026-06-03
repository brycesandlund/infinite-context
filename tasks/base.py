"""Shared task primitives: Problem schema, graders, helpers.

Grader contract:
- `grade_answer(extracted, gold, mode) -> float` returns a score in [0, 1].
- For binary modes ("exact", "set"), the score is 0.0 or 1.0.
- For "numeric", the score is OOLONG's 0.75**|y-y_hat| partial credit, so
  "close" numeric answers get gradient signal even when not exact.

`is_answer_correct` is a thin bool wrapper used by logging for binary tasks.
For numeric-graded tasks, log the mean score instead of a correct/N ratio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


GradingMode = Literal["exact", "set", "numeric", "ruler_all", "ruler_part"]


@dataclass
class Problem:
    """One task instance the agent must solve.

    `document_tokens` is the haystack — the agent never sees this directly;
    it accesses ranges via the `read_chunk` tool. `question` is the user
    message (small; carries any answer-format hint). `task_context` is pinned
    to every agent's system prompt (parent + every spawned subagent) so child
    agents inherit task-level instructions (e.g. label space, format) without
    the parent burning budget restating them in subtask strings.
    """

    document_tokens: list[int]
    question: str
    gold_answers: list[str]
    task: str
    task_context: str = ""
    # Optional per-problem grader override. Most tasks have a fixed grader per
    # task name (registry tables), but OOLONG questions mix answer types
    # (numeric counts vs exact labels/users/dates) within a family, so the
    # grader is decided per problem. None -> fall back to the registry by task.
    grading_mode: "GradingMode | None" = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Strip commas, dollar signs, whitespace, surrounding quotes; lowercase."""
    s = s.replace(",", "").replace("$", "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s.lower()


_LIST_SPLIT_RE = re.compile(r"[,;\n]")


def split_list(text: str) -> list[str]:
    """Split a model-emitted answer on common list separators."""
    return [p.strip() for p in _LIST_SPLIT_RE.split(text) if p.strip()]


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_number(s: str) -> float | None:
    """Pull the first signed decimal number out of `s`, or None."""
    s = s.replace(",", "")
    m = _NUMERIC_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


def grade_answer(
    extracted: str | None, gold_answers: list[str], mode: GradingMode
) -> float:
    """Score an extracted answer against the gold(s). Returns a [0, 1] float."""
    if extracted is None or not gold_answers:
        return 0.0

    if mode == "exact":
        return 1.0 if normalize_answer(extracted) == normalize_answer(gold_answers[0]) else 0.0

    if mode == "set":
        # Partial-credit set F1 (precision * recall), NOT binary set-equality.
        # Binary all-or-nothing gives RL no within-group variance on multi-answer
        # tasks (3/4 correct scores the same 0 as 0/4) — so the gradient vanishes
        # exactly where it's needed (vt, cwe, fwe, multivalue, multiquery). F1
        # rewards partial progress AND penalizes over-listing (precision), so it
        # can't be gamed by dumping every candidate. Exact set match -> 1.0.
        model_items = {normalize_answer(s) for s in split_list(extracted)}
        gold_items = {normalize_answer(g) for g in gold_answers}
        if not gold_items:
            return 0.0
        inter = len(model_items & gold_items)
        if inter == 0:
            return 0.0
        precision = inter / len(model_items)  # model_items nonempty since inter>0
        recall = inter / len(gold_items)
        return 2 * precision * recall / (precision + recall)

    if mode == "numeric":
        y_hat = _extract_number(extracted)
        y = _extract_number(gold_answers[0])
        if y_hat is None or y is None:
            return 0.0
        # OOLONG's partial-credit: 0.75**|y - y_hat|. Exact = 1.0, off-by-one = 0.75.
        return float(0.75 ** abs(y - y_hat))

    if mode == "ruler_all":
        # RULER's `string_match_all`: average over golds of "is gold (lowercased)
        # a substring of the prediction (lowercased)?". Returns [0, 1].
        # Vendored from RULER/scripts/eval/synthetic/constants.py.
        pred_lc = extracted.lower()
        hits = sum(1.0 for g in gold_answers if g.lower() in pred_lc)
        return hits / len(gold_answers)

    if mode == "ruler_part":
        # RULER's `string_match_part`: max over golds. Used for QA where any
        # single matching gold counts. Returns {0.0, 1.0} per example.
        pred_lc = extracted.lower()
        return 1.0 if any(g.lower() in pred_lc for g in gold_answers) else 0.0

    raise ValueError(f"Unknown grading mode: {mode!r}")


def is_answer_correct(
    extracted: str | None, gold_answers: list[str], mode: GradingMode
) -> bool:
    """Thin bool wrapper for logging. For numeric tasks this requires an exact
    hit; report mean score separately if you care about partial credit."""
    return grade_answer(extracted, gold_answers, mode) >= 1.0
