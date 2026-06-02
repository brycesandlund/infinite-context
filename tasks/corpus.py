"""Haystack corpora.

- Paul Graham essays: real-text background for the agent's haystack. Cached on
  disk after the first download.
- Noise sentence: the Mohtashami & Jaggi "The grass is green..." filler, used
  as a low-entropy haystack for one of RULER's S-NIAH subtasks (and as the
  default haystack for VT).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path


# ---------------------------------------------------------------------------
# Paul Graham essays
# ---------------------------------------------------------------------------


_CACHE_DIR = Path.home() / ".cache" / "infinite-context"
_PG_ESSAYS_CACHE = _CACHE_DIR / "pg_essays.txt"

_PG_BASE_URL = (
    "https://raw.githubusercontent.com/gkamradt/LLMTest_NeedleInAHaystack/"
    "main/needlehaystack/PaulGrahamEssays/"
)

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
        print(
            f"PG essays corpus: fetched {len(chunks)}/{len(_PG_ESSAY_FILES)} "
            f"({n_failed} failed)"
        )
    return "\n\n".join(chunks)


def load_pg_essays_text(force_refresh: bool = False) -> str:
    """Load Paul Graham essays as one concatenated string. Cached on disk."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _PG_ESSAYS_CACHE.exists() and not force_refresh:
        return _PG_ESSAYS_CACHE.read_text()
    text = _download_pg_essays()
    _PG_ESSAYS_CACHE.write_text(text)
    return text


# ---------------------------------------------------------------------------
# Noise sentence (Mohtashami & Jaggi 2023, also used by RULER)
# ---------------------------------------------------------------------------


NOISE_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again."
)
