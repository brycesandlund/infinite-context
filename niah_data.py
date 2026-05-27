"""Synthesized Needle-in-a-Haystack problems from Paul Graham essays.

The haystack corpus is the Paul Graham essays (concatenated, ~250k tokens).
A problem is a random `doc_size_tokens`-long slice with a single key-value
needle ("The magic number is X.") inserted at a random position in the
middle 80%. The question asks for that value.
"""

from __future__ import annotations

import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


_CACHE_DIR = Path.home() / ".cache" / "infinite-context"
_PG_ESSAYS_CACHE = _CACHE_DIR / "pg_essays.txt"

_PG_BASE_URL = (
    "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/"
    "main/needlehaystack/PaulGrahamEssays/"
)

# Subset of Paul Graham essays in the NIAH github repo. Enough for ~200-300k
# tokens after tokenization.
_PG_ESSAY_FILES = [
    "addiction.txt", "aord.txt", "apple.txt", "avg.txt", "before.txt",
    "bias.txt", "boss.txt", "copy.txt", "corpdev.txt", "desres.txt",
    "diff.txt", "ecw.txt", "founders.txt", "foundervisa.txt", "gap.txt",
    "gba.txt", "gh.txt", "goodtaste.txt", "hubs.txt", "iflisp.txt",
    "island.txt", "know.txt", "langdes.txt", "laundry.txt", "love.txt",
    "mod.txt", "newideas.txt", "nft.txt", "philosophy.txt", "popular.txt",
    "pow.txt", "rootsoflisp.txt", "rss.txt", "siliconvalley.txt",
    "startuplessons.txt", "submarine.txt", "sun.txt", "superangels.txt",
    "todo.txt", "unions.txt", "useful.txt", "vb.txt", "vcsqueeze.txt",
    "vw.txt", "want.txt", "web20.txt", "weird.txt", "wisdom.txt", "worked.txt",
]


def _fetch_one(filename: str) -> str | None:
    url = _PG_BASE_URL + filename
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError):
        return None


def _download_pg_essays() -> str:
    chunks: list[str] = []
    n_failed = 0
    for fname in _PG_ESSAY_FILES:
        text = _fetch_one(fname)
        if text is None:
            n_failed += 1
            continue
        chunks.append(text)
    if not chunks:
        raise RuntimeError("Failed to download any Paul Graham essays")
    if n_failed:
        print(f"NIAH corpus: fetched {len(chunks)}/{len(_PG_ESSAY_FILES)} essays "
              f"({n_failed} failed)")
    return "\n\n".join(chunks)


def load_pg_essays_text(force_refresh: bool = False) -> str:
    """Load Paul Graham essays as one concatenated string. Cached on disk under
    ~/.cache/infinite-context/pg_essays.txt."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _PG_ESSAYS_CACHE.exists() and not force_refresh:
        return _PG_ESSAYS_CACHE.read_text()
    text = _download_pg_essays()
    _PG_ESSAYS_CACHE.write_text(text)
    return text


@dataclass
class NIAHProblem:
    document_tokens: list[int]
    question: str
    gold_answer: str
    needle_position: int  # token index where the needle was inserted


def make_niah_problem(
    corpus_tokens: list[int],
    tokenizer,
    doc_size_tokens: int,
    seed: int,
    needle_template: str = " The magic number is {value}. ",
    question_template: str = "What is the magic number?",
) -> NIAHProblem:
    """Build one NIAH problem from a random slice of the corpus + a single needle."""
    rng = random.Random(seed)
    if len(corpus_tokens) < doc_size_tokens:
        raise ValueError(
            f"Corpus has only {len(corpus_tokens)} tokens, need at least {doc_size_tokens}."
        )
    span_start = rng.randint(0, len(corpus_tokens) - doc_size_tokens)
    haystack = list(corpus_tokens[span_start : span_start + doc_size_tokens])

    needle_value = str(rng.randint(1000, 9999))
    needle_text = needle_template.format(value=needle_value)
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)

    insert_pos = rng.randint(doc_size_tokens // 10, 9 * doc_size_tokens // 10)
    doc_tokens = haystack[:insert_pos] + needle_tokens + haystack[insert_pos:]

    return NIAHProblem(
        document_tokens=doc_tokens,
        question=question_template,
        gold_answer=needle_value,
        needle_position=insert_pos,
    )
