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
import warnings

import datasets as _hf_datasets

from tasks.base import Problem

# The vendored synth pipeline runs many datasets .filter()/.sort()/.shuffle()
# calls per problem — each spews a tqdm progress bar ("Filter: ...%", "Flattening
# the indices: ...") into training/eval logs. Silence them globally; we generate
# hundreds of problems per run.
_hf_datasets.disable_progress_bars()
# Likewise the vendored example_constructor's bare softmax() triggers a per-problem
# torch deprecation warning; not ours to fix (vendored verbatim), so filter it.
warnings.filterwarnings(
    "ignore", message="Implicit dimension choice for softmax", category=UserWarning
)

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

# Deterministic problem indexing shared by SFT and eval, so a given (base, task,
# idx) is the IDENTICAL problem in both — letting us train and probe on the same
# questions. The dataset round-robins for even coverage; the per-task offset keeps
# tasks from colliding; `base` separates train vs held-out draws.
_OOLONG_TASK_ORD = {"oolong_counting": 0, "oolong_user": 1, "oolong_temporal": 2}


def oolong_spec(task: str, idx: int, base: int) -> tuple[int, str]:
    """(seed, dataset) for OOLONG problem `idx` of `task` under seed `base`.
    Same (base, task, idx) -> same problem everywhere. Use a different `base`
    to draw a disjoint set (e.g. held-out eval vs SFT train)."""
    if task not in _OOLONG_TASK_ORD:
        raise ValueError(f"Not an OOLONG task: {task!r}")
    seed = base + _OOLONG_TASK_ORD[task] * 100_000 + idx
    return seed, _DATASETS[idx % len(_DATASETS)]


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
    dataset: str | None = None,  # pin the source dataset (else sampled from seed)
) -> Problem:
    if task_name not in OOLONG_TASKS:
        raise ValueError(f"Unknown OOLONG task: {task_name!r}")
    family = OOLONG_TASKS[task_name]
    rng = random.Random(seed)

    # Dataset is normally sampled per-seed; eval can pin it for even coverage
    # across the 10 datasets (so per-dataset stats aren't starved). Consume the
    # same rng draw either way so the rest of the synthesis is seed-stable.
    sampled = rng.choice(_DATASETS)
    if dataset is not None and dataset not in _DATASETS:
        raise ValueError(f"Unknown OOLONG dataset {dataset!r}. Known: {_DATASETS}")
    dataset = dataset or sampled
    data_obj = SUPPORTED_DATASETS[dataset]()
    num_labels = len(data_obj.label_list)
    temperature = rng.randint(2, 6)               # OOLONG's label-skew temperature
    num_in_context = max(8, data_obj.get_examples_in_context(doc_size_tokens))

    final_data, true_counts, desc_fn = construct_context(
        data_obj, num_labels, num_in_context, temperature, seed
    )

    # Pin task selection: the vendored *Tasks classes draw the cutoff date, month,
    # date range, label-pair order, user subset, etc. from the GLOBAL random module,
    # whose state after construct_context (HF dataset ops in between) is process-
    # dependent. Reseed so a given (seed, dataset) always yields the SAME question —
    # essential for reproducible SFT traces and train/eval problem alignment.
    random.seed(seed)
    tasks = _build_tasks(family, true_counts, final_data)
    if not tasks:  # degenerate context for this family — fall back to counting
        tasks = CountingTasks(true_counts).tasks
    # Normalize task ORDER before choosing: the vendored builders iterate over
    # `set(labels)`, whose order is randomized per process (PYTHONHASHSEED), so the
    # list order — and thus rng.choice — would otherwise vary run-to-run. Sorting by
    # (question, answer) makes the pick reproducible for a given seed.
    tasks = sorted(tasks, key=lambda t: (t.question, str(t.answer)))
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

    # Official OOLONG prompt verbatim (it already specifies the answer format,
    # e.g. "Give your final answer in the form 'Label: answer'"); we only add the
    # minimal \boxed{} mechanism our harness extracts from. OOLONG's own grader
    # parse (split(":")[-1]) handles the "Label: X" form, so this stays faithful.
    question = task.question.strip() + "\n\nPut your final answer inside \\boxed{}."
    gold = [str(a).strip() for a in task.answer]
    is_numeric = task.answer_type == ANSWER_TYPE.NUMERIC
    is_comparison = task.answer_type == ANSWER_TYPE.COMPARISON
    # EVAL grader = OOLONG-official, by answer type: 0.75^|err| numeric for counts;
    # word-boundary containment of the gold phrase for COMPARISON ("A is [X] B",
    # gold is only [X] — exact can't match); exact membership (split(":")[-1]) for
    # categorical / user / date. Leaderboard-faithful per the question format.
    grading_mode = (
        "numeric" if is_numeric else "oolong_compare" if is_comparison else "oolong_exact"
    )
    # TRAINING REWARD = softer/denser: numeric stays partial-credit; categorical
    # uses whole-word membership (oolong_soft) so RL gets gradient toward the
    # answer even when the format is off, WITHOUT substring false-positives
    # (e.g. 'correct' must not match 'incorrect'). Eval stays exact (above).
    reward_mode = "numeric" if is_numeric else "oolong_soft"

    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=gold,
        task=task_name,
        task_context=task_context,
        grading_mode=grading_mode,
        reward_mode=reward_mode,
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
