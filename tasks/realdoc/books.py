"""Public-domain novel corpus (Project Gutenberg) for the realdoc tasks.

Real long prose to ask REAL aggregation/retrieval questions over — distinct from
every eval set (RULER synthetic, OOLONG classification, our PG-*essay* filler,
SQuAD/HotpotQA). Books are cached locally on first use; Gutenberg texts are public
domain in the US."""

from __future__ import annotations

import os
import re
import urllib.request

_CACHE = os.path.expanduser("~/.cache/infinite-context/gutenberg")

# name -> Gutenberg ebook id. Varied authors/eras so the prose isn't monocultural.
BOOKS: dict[str, int] = {
    "pride_and_prejudice": 1342,
    "sherlock_holmes": 1661,
    "moby_dick": 2701,
    "tale_of_two_cities": 98,
    "frankenstein": 84,
    "great_expectations": 1400,
    "dracula": 345,
    "alice_in_wonderland": 11,
}


def _url(gid: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"


def _strip_gutenberg(text: str) -> str:
    """Drop the Gutenberg license header/footer, keeping just the work itself."""
    start = re.search(r"\*\*\*\s*START OF.*?\*\*\*", text, re.S)
    end = re.search(r"\*\*\*\s*END OF.*?\*\*\*", text, re.S)
    s = start.end() if start else 0
    e = end.start() if end else len(text)
    return text[s:e].strip()


def load_book_text(name: str) -> str:
    """Return the cleaned text of `name`, downloading + caching it on first use."""
    if name not in BOOKS:
        raise ValueError(f"Unknown book {name!r}. Known: {sorted(BOOKS)}")
    os.makedirs(_CACHE, exist_ok=True)
    path = os.path.join(_CACHE, f"{name}.txt")
    if not os.path.exists(path):
        raw = urllib.request.urlopen(_url(BOOKS[name]), timeout=60).read().decode("utf-8", "ignore")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_strip_gutenberg(raw))
    return open(path, encoding="utf-8").read()
