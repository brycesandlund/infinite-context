"""RULER task templates and per-task configuration.

`TASK_TEMPLATES` is RULER's `scripts/data/synthetic/constants.py` verbatim
(template + answer_prefix per task family).

`RULER_TASKS` is RULER's `scripts/synthetic.yaml` ported to a Python dict —
the 13-task spec NeMo-Skills uses. The keys are the canonical task names that
appear in NVIDIA's published RULER scores (niah_single_1, niah_single_2, ...).

Vendored from NVIDIA/RULER (Apache-2.0); see tasks/ruler/__init__.py.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Per-task-family templates. {context} is replaced with the haystack in
# RULER's reference; we replace it with a "[document accessible via read_chunk]"
# placeholder when rendering the question, and put the haystack tokens into
# Problem.document_tokens. The answer_prefix is the RULER-mandated prompt
# completion target; we drop it for the agent setup and inject a "Reply with
# the value inside \\boxed{}." instruction instead.
# ---------------------------------------------------------------------------


TASK_TEMPLATES = {
    "niah": {
        "tokens_to_generate": 128,
        "template": (
            "Some special magic {type_needle_v} are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
            "{context}\n"
            "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
        ),
        "answer_prefix": (
            " The special magic {type_needle_v} for {query} mentioned in the provided text are"
        ),
    },
    "variable_tracking": {
        "tokens_to_generate": 30,
        "template": (
            "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
            "{context}\n"
            "Question: Find all variables that are assigned the value {query} in the text above."
        ),
        "answer_prefix": (
            " Answer: According to the chain(s) of variable assignment in the text above, "
            "{num_v} variables are assigned the value {query}, they are: "
        ),
    },
    "common_words_extraction": {
        "tokens_to_generate": 120,
        "template": (
            "Below is a numbered list of words. In these words, some appear more often than others. "
            "Memorize the ones that appear most often.\n"
            "{context}\n"
            "Question: What are the 10 most common words in the above list?"
        ),
        "answer_prefix": " Answer: The top 10 words that appear most often in the list are:",
    },
    "freq_words_extraction": {
        "tokens_to_generate": 50,
        "template": (
            "Read the following coded text and track the frequency of each coded word. Find the three "
            "most frequently appeared coded words. {context}\n"
            "Question: Do not provide any explanation. Please ignore the dots '....'. "
            "What are the three most frequently appeared words in the above coded text?"
        ),
        "answer_prefix": (
            " Answer: According to the coded text above, the three most frequently appeared words are:"
        ),
    },
    "qa": {
        "tokens_to_generate": 32,
        "template": (
            "Answer the question based on the given documents. Only give me the answer and do not "
            "output any other words.\n\n"
            "The following are given documents.\n\n"
            "{context}\n\n"
            "Answer the question based on the given documents. Only give me the answer and do not "
            "output any other words.\n\n"
            "Question: {query}"
        ),
        "answer_prefix": " Answer:",
    },
}


# ---------------------------------------------------------------------------
# The 13 canonical RULER task configurations (RULER/scripts/synthetic.yaml).
# Each entry: {"task": <template family>, "args": <kwargs forwarded to generator>}.
# Task names match NVIDIA's published RULER leaderboard so our eval numbers
# are directly comparable.
# ---------------------------------------------------------------------------


RULER_TASKS: dict[str, dict] = {
    "niah_single_1": {
        "task": "niah",
        "args": {
            "type_haystack": "noise", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1,
        },
    },
    "niah_single_2": {
        "task": "niah",
        "args": {
            "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1,
        },
    },
    "niah_single_3": {
        "task": "niah",
        "args": {
            "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "uuids",
            "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1,
        },
    },
    "niah_multikey_1": {
        "task": "niah",
        "args": {
            "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1,
        },
    },
    "niah_multikey_2": {
        "task": "niah",
        "args": {
            "type_haystack": "needle", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1,
        },
    },
    "niah_multikey_3": {
        "task": "niah",
        "args": {
            "type_haystack": "needle", "type_needle_k": "uuids", "type_needle_v": "uuids",
            "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1,
        },
    },
    "niah_multivalue": {
        "task": "niah",
        "args": {
            "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1,
        },
    },
    "niah_multiquery": {
        "task": "niah",
        "args": {
            "type_haystack": "essay", "type_needle_k": "words", "type_needle_v": "numbers",
            "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 4,
        },
    },
    "vt": {
        "task": "variable_tracking",
        "args": {"type_haystack": "noise", "num_chains": 1, "num_hops": 4},
    },
    "cwe": {
        "task": "common_words_extraction",
        "args": {"freq_cw": 30, "freq_ucw": 3, "num_cw": 10},
    },
    "fwe": {
        "task": "freq_words_extraction",
        "args": {"alpha": 2.0},
    },
    "qa_1": {"task": "qa", "args": {"dataset": "squad"}},
    "qa_2": {"task": "qa", "args": {"dataset": "hotpotqa"}},
}


# ---------------------------------------------------------------------------
# RULER's substring-match accuracy thresholds, per RULER's eval/synthetic/constants.py.
# `string_match_all` averages per-gold substring hits; `string_match_part` takes
# the max over golds. Used at eval time so our reported numbers line up with
# NVIDIA's leaderboard scores.
# ---------------------------------------------------------------------------


METRIC_TYPE = {
    "niah": "string_match_all",
    "variable_tracking": "string_match_all",
    "common_words_extraction": "string_match_all",
    "freq_words_extraction": "string_match_all",
    "qa": "string_match_part",
}
