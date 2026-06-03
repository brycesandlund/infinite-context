"""OOLONG-synth generators — thin wrapper over the VENDORED OOLONG code.

We call NVIDIA-of-OOLONG's actual generation modules (tasks/oolong/vendored_synth/,
verbatim from abertsch72/oolong) so problems are faithful: their 10 mislabel-
filtered datasets, their skewed user/date synthesis, and their exact question
templates across the Counting / User / Timeline families. Two adaptations match
the rest of our suite:
1. the packed examples become `document_tokens` (read via read_chunk), while the
   dataset description goes into `task_context` and the question into `question`;
2. the model answers in `\\boxed{}` (we override OOLONG's "Give your final answer
   in the form ..." with "answer inside \\boxed{}") — content/gold/scoring stay
   faithful (exact for label/user/date/comparison, numeric 0.75^|err| for counts).

OOLONG questions mix answer types within a family, so each Problem carries its
own `grading_mode` (numeric vs exact).
"""

from __future__ import annotations

import os
import random
import sys

from tasks.base import Problem

# Make the vendored modules importable (they cross-import by bare module name).
_VENDORED = os.path.join(os.path.dirname(__file__), "vendored_synth")
if _VENDORED not in sys.path:
    sys.path.insert(0, _VENDORED)

from constants import ANSWER_TYPE  # noqa: E402  (vendored)
from datasets_loader import SUPPORTED_DATASETS  # noqa: E402
from example_constructor import construct_context  # noqa: E402
from task_constructors import CountingTasks, TemporalTasks, UserTasks  # noqa: E402


# Family task names -> which vendored Task constructor produces the questions.
OOLONG_TASKS = {
    "oolong_counting": "counting",   # most/least common, A-vs-B, absolute counts
    "oolong_user": "user",           # which user (subset / per-label) most often
    "oolong_temporal": "temporal",   # dates, before/after, month & range subsets
}

_DATASETS = list(SUPPORTED_DATASETS.keys())


def _build_tasks(family, true_counts, final_data):
    if family == "counting":
        return CountingTasks(true_counts).tasks
    if family == "user":
        return UserTasks(true_counts, in_context=final_data).tasks
    if family == "temporal":
        return TemporalTasks(true_counts, in_context=final_data).tasks
    raise ValueError(f"Unknown OOLONG family: {family!r}")


def make_oolong_problem(
    task_name: str,
    corpus_tokens: list[int],  # unused — OOLONG builds from its own datasets
    tokenizer,
    doc_size_tokens: int,
    seed: int,
) -> Problem:
    if task_name not in OOLONG_TASKS:
        raise ValueError(f"Unknown OOLONG task: {task_name!r}")
    family = OOLONG_TASKS[task_name]
    rng = random.Random(seed)

    dataset = rng.choice(_DATASETS)
    data_obj = SUPPORTED_DATASETS[dataset]()
    num_labels = len(data_obj.label_list)
    temperature = rng.randint(2, 6)               # OOLONG's label-skew temperature
    num_in_context = max(8, data_obj.get_examples_in_context(doc_size_tokens))

    final_data, true_counts, desc_fn = construct_context(
        data_obj, num_labels, num_in_context, temperature, seed
    )

    tasks = _build_tasks(family, true_counts, final_data)
    if not tasks:  # degenerate context for this family — fall back to counting
        tasks = CountingTasks(true_counts).tasks
    task = rng.choice(tasks)

    # Build the document (unlabeled instance lines), recording per-example spans
    # (start_tok, end_tok, label, user, date) for the classify-and-count oracle.
    ordered = final_data.sort("x").shuffle(seed=42)
    doc_tokens: list[int] = []
    spans: list[tuple] = []
    for ex in ordered:
        line = f"Date: {ex['formatted_date']} || User: {ex['user_id']} || Instance: {ex['x'].replace(chr(10), ' ')}\n"
        toks = tokenizer.encode(line, add_special_tokens=False)
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        spans.append((start, len(doc_tokens), ex["y"], ex["user_id"], ex["formatted_date"]))

    task_context = desc_fn(
        intro=True,
        num_examples=len(ordered),
        num_labels=num_labels,
        labels_chosen=list(true_counts.keys()),
    ).strip()

    question = task.question.strip() + (
        "\n\nReply with ONLY the final answer inside \\boxed{} — just the value "
        "itself (a single label, number, user ID, or the exact comparison phrase). "
        "Do NOT add counts, prefixes like 'Label:'/'User:', or any extra words."
    )
    gold = [str(a).strip() for a in task.answer]
    # numeric counts -> partial-credit numeric grader; categorical answers
    # (label/user/date/comparison) -> substring match (ruler_part), which is
    # robust to the model wrapping the answer (e.g. boxing 'spam: 12' for gold
    # 'spam'). Slightly more lenient than OOLONG's official exact-match, to fit
    # our \boxed harness; swap to "exact" for strict OOLONG comparability.
    grading_mode = "numeric" if task.answer_type == ANSWER_TYPE.NUMERIC else "ruler_part"

    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=gold,
        task=task_name,
        task_context=task_context,
        grading_mode=grading_mode,
        metadata={
            "dataset": dataset,
            "family": family,
            "answer_type": task.answer_type.value,
            "task_type": task.task_type.value,
            "n_examples": len(ordered),
            "true_counts": dict(true_counts),
            "example_spans": spans,
        },
    )
