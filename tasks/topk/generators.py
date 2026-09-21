"""synth_topk — the k most frequent words over an OPEN vocabulary, with pruned `word:count` partials.

Why: the model's root learned to dictate a format contract, and on an open key space (RULER cwe/fwe)
it dictated a lossy one — "the 10 most common words as a comma-separated list", no counts — so the
merge became a set union (cwe 0.96 → 0.52 in run 7). This task teaches the contract that works:
every range returns a PRUNED TALLY (`word:count` for its most frequent words, at least the top m),
the merge adds counts, and only the root ranks. Also teaches that list numerals are not words.

Document: a word list in one of two layouts —
    numbered   "1. lantern 2. gravel 3. lantern 4. ..."      (several per line)
    stream     "lantern gravel lantern ... "                  (plain, several per line)
Vocabulary: PG-essay corpus words (not RULER's noun lists). A few `hot` words repeat many times; the
rest appear once or a handful of times. Question: "Which k words appear most often? (k of 3/5/8)".
Gold = the k hot words (set grading); hot counts are separated from the background by construction.
`record_spans` = (tok_start, tok_end, idx, word) per word occurrence (line-based ownership).
"""

from __future__ import annotations

import random
import re
from collections import Counter

from tasks.base import Problem

TOPK_TASKS: dict[str, dict] = {
    "synth_topk": {"family": "bounded", "strategy": "binary"},
}

_WORD = re.compile(r"\b[a-z]{4,9}\b")
_VOCAB: list[str] = []


def _vocab(corpus_tokens, tokenizer) -> list[str]:
    global _VOCAB
    if not _VOCAB:
        text = tokenizer.decode(corpus_tokens[:400_000]) if corpus_tokens else ""
        c = Counter(_WORD.findall(text.lower()))
        _VOCAB = sorted(w for w, n in c.items() if 2 <= n <= 200)   # plain, mid-frequency words
        if len(_VOCAB) < 500:
            _VOCAB = sorted(set(_WORD.findall(text.lower()))) or ["alpha", "bravo", "charlie", "delta"]
    return _VOCAB


def _context(layout: str) -> str:
    if layout == "numbered":
        return ("The document is a numbered list of words: each entry is `<n>. <word>` and several entries "
                "appear on each line. The numbers are positions, not words.")
    return "The document is a plain sequence of words separated by spaces, several per line."


def make_topk_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in TOPK_TASKS:
        raise ValueError(f"Unknown topk task: {task!r}")
    rng = random.Random(seed)
    vocab = _vocab(corpus_tokens, tokenizer)
    layout = rng.choice(["numbered", "stream"])
    k = rng.choice([3, 5, 5, 8])
    hot = rng.sample(vocab, k)
    # Each LINE carries each hot word with p=0.75 (so a ~250-token leaf of 4-6 lines holds every hot
    # word 3-5 times) plus a few background words (mostly one-off; ~20% drawn from a small pool of
    # words that recur 2-4 times in total). Hot words therefore dominate every range, and a leaf's
    # pruned tally (words seen 2+ times) is exact for the top-k by construction — RULER cwe's regime.
    bg_words = [w for w in vocab if w not in hot]
    rng.shuffle(bg_words)
    repeaters = bg_words[:12]
    fresh = iter(bg_words[12:])
    per_line_bg = rng.randint(9, 15)      # ~15-23 words per line -> ~12-15 lines per 250-token leaf
    doc_tokens, spans, idx, entry_no = [], [], 0, 0
    while True:
        words = [w for w in hot if rng.random() < 0.75]
        for _ in range(per_line_bg):
            # when the fresh supply runs out (long docs), fall back to a uniform draw over ALL background
            # words — never the small repeater pool, which would otherwise accumulate hot-like counts
            words.append(rng.choice(repeaters) if rng.random() < 0.2 else next(fresh, rng.choice(bg_words[12:] or bg_words)))
        rng.shuffle(words)
        if layout == "numbered":
            line = " ".join(f"{entry_no + j + 1}. {w}" for j, w in enumerate(words)) + "\n"
        else:
            line = " ".join(words) + "\n"
        toks = tokenizer.encode(line, add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        for w in words:
            spans.append((start, len(doc_tokens), idx, w)); idx += 1
        entry_no += len(words)
    counts = Counter(w for *_, w in spans)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    gold = [w for w, _ in ranked[:k]]
    q = (f"Which {k} words appear MOST often in the document? {'Ignore the list numbers. ' if layout == 'numbered' else ''}"
         f"List the {k} words, comma-separated, in \\boxed{{}}.")
    return Problem(
        document_tokens=doc_tokens, question=q, gold_answers=gold, task=task,
        task_context=_context(layout), grading_mode="set",
        metadata={"family": "bounded", "strategy_default": "binary", "task": task, "layout": layout,
                  "k": k, "keep_m": max(10, 2 * k), "hot": hot, "n_records": len(spans),
                  "record_spans": spans, "gold_int": None},
    )
