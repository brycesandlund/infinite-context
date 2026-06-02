"""Shared primitives for vendored RULER generators.

- `_words()` mirrors RULER's NIAH adjective×noun key vocabulary.
- `_essay_words()` lazily loads the PG-essay corpus as a word list (RULER
  works in word-space, splits on whitespace, then sentence-tokenizes after
  joining).
- `_ensure_punkt()` lazily downloads nltk's punkt models on first use.
- `_render_question()` is the one adaptation point: takes RULER's task
  template + the rendered query, strips `{context}`, and produces our
  agent-facing question string.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache

from tasks.corpus import NOISE_SENTENCE, load_pg_essays_text


# ---------------------------------------------------------------------------
# nltk punkt download (one-shot, thread-safe)
# ---------------------------------------------------------------------------

_punkt_lock = threading.Lock()
_punkt_loaded = False


def ensure_punkt() -> None:
    """Idempotent: download punkt + punkt_tab if not already on disk.
    RULER's prepare.py does the equivalent at import time."""
    global _punkt_loaded
    if _punkt_loaded:
        return
    with _punkt_lock:
        if _punkt_loaded:
            return
        import nltk
        for resource, lookup in [("punkt", "tokenizers/punkt"),
                                  ("punkt_tab", "tokenizers/punkt_tab")]:
            try:
                nltk.data.find(lookup)
            except LookupError:
                nltk.download(resource, quiet=True)
        _punkt_loaded = True


# ---------------------------------------------------------------------------
# RULER's NIAH key vocabulary: adj×noun, mirrors niah.py:
#   nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
#   adjs  = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
#   words = sorted(set(f"{adj}-{noun}" for adj in adjs for noun in nouns))
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def niah_key_words() -> list[str]:
    from wonderwords import random_word
    nouns = random_word._get_words_from_text_file("nounlist.txt")
    adjs = random_word._get_words_from_text_file("adjectivelist.txt")
    return sorted({f"{a}-{n}" for a in adjs for n in nouns})


@lru_cache(maxsize=1)
def cwe_word_pool() -> list[str]:
    """RULER's CWE uses nouns + adjs + verbs unioned."""
    from wonderwords import random_word
    nouns = random_word._get_words_from_text_file("nounlist.txt")
    adjs = random_word._get_words_from_text_file("adjectivelist.txt")
    verbs = random_word._get_words_from_text_file("verblist.txt")
    return sorted(set(nouns + adjs + verbs))


# ---------------------------------------------------------------------------
# Essay corpus as a flat word list (RULER's niah.py does this exactly):
#   essay = re.sub(r"\s+", " ", essay_text).split(" ")
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def essay_words() -> list[str]:
    return re.sub(r"\s+", " ", load_pg_essays_text()).split(" ")


# ---------------------------------------------------------------------------
# The "noise" haystack RULER uses for niah_single_1 and vt:
#   "The grass is green. The sky is blue. The sun is yellow. Here we go.
#    There and back again."
# repeated N times.
# ---------------------------------------------------------------------------


def noise_sentence() -> str:
    return NOISE_SENTENCE


# ---------------------------------------------------------------------------
# Question rendering: RULER bakes {context} into the template; we extract the
# context out, hand the agent a "[document via read_chunk]" placeholder in
# the question, and put the actual context text into Problem.document_tokens.
# ---------------------------------------------------------------------------


CONTEXT_PLACEHOLDER = "[The relevant text is in a separate document accessible via the read_chunk tool — see the system prompt for usage.]"


def render_question(template: str, *, context: str, doc_token_count: int, **fields) -> tuple[str, str]:
    """Split RULER's `template` into (context_text, question_text).

    `context_text` is the haystack-with-needles (will be tokenized and stored
    in Problem.document_tokens). `question_text` is the template prose with
    `{context}` replaced by a placeholder telling the agent the document is
    accessible via `read_chunk`, plus an instruction to put the final answer
    in \\boxed{}.
    """
    # Substitute everything except {context}, then replace {context} with the
    # placeholder. Done in two steps so the placeholder can't accidentally
    # consume curly braces in user-supplied fields.
    body = template.replace("{context}", "__RULER_CONTEXT_SENTINEL__")
    body = body.format(context="__RULER_CONTEXT_SENTINEL__", **fields)
    placeholder = (
        f"{CONTEXT_PLACEHOLDER} (Document length: {doc_token_count} tokens.)"
    )
    body = body.replace("__RULER_CONTEXT_SENTINEL__", placeholder)
    body += "\n\nReply with the answer inside \\boxed{}."
    return context, body
