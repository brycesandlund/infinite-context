"""synth_topk — the k most frequent words over an OPEN vocabulary, with pruned `word:count` partials.

Why: the model's root learned to dictate a format contract, and on an open key space (RULER cwe/fwe)
it dictated a lossy one — "the 10 most common words as a comma-separated list", no counts — so the
merge became a set union (cwe 0.96 → 0.52 in run 7). This task teaches the contract that works:
every range returns a PRUNED TALLY (`word:count` for its most frequent words, at least the top m),
the merge adds counts, and only the root ranks. Also teaches that list numerals are not words.

Document: a word list in one of two layouts —
    numbered   "1. lantern 2. gravel 3. lantern 4. ..."      (several per line)
    dotted     "lantern .... gravel .... lantern .... ..."    (RULER fwe's separator)
Either layout may be one unbroken line (40%, as RULER cwe/fwe are); words may be English or coded strings.
Vocabulary: PG-essay corpus words (not RULER's noun lists). A few `hot` words repeat many times; the
rest appear once or a handful of times. Question: "Which k words appear most often? (k of 3/5/8)".
Gold = the k hot words (set grading); hot counts are separated from the background by construction.
`record_spans` = (tok_start, tok_end, idx, word) — ONE RECORD PER WORD, spanning that word's own tokens (ownership =
the word STARTS in the range); 40% of documents are a single line (RULER cwe/fwe).
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


# CODED words (v16): 35% of problems use random 6-letter strings instead of English words. RULER fwe (18w) failed by
# COPY errors on coded words — `hlosvz` tallied beside `lhosvz`, near-duplicate keys (`xyxzroc`/`vxzroc`) splitting
# counts; training only ever tallied real words. Each hot word also gets two one-swap NEAR-MISS decoys in the
# background, so an exact copy is what separates them. Half of coded problems separate words with " .... ".
_CODED_FRAC = 0.35
# SPARSE regime (v16): half the problems put the hot words at ~12-30 occurrences over the WHOLE document against a
# background whose words recur 1-3 times (RULER cwe's regime: 30 vs 3). In a 14K doc a hot word then appears <1x per
# 500-token leaf, so one-offs must be kept (K = budget/50) — the dense regime (hot word on 75% of lines) never showed
# why. k includes 10 here (cwe asks for 10).
_SPARSE_FRAC = 0.5
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _code(rng) -> str:
    return "".join(rng.choice(_LETTERS) for _ in range(6))


def _near_miss(rng, w: str) -> str:
    i = rng.randrange(len(w) - 1)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:] if w[i] != w[i + 1] else w[:i] + rng.choice(_LETTERS) + w[i + 1:]


# LAYOUTS (v16): numbered (`12. word`, RULER cwe) and dotted (`word .... word`, RULER fwe), for English and coded words
# alike. The plain space-separated layouts were dropped: at ~1.1 tokens/word a 500-token leaf owns ~450 words, and the
# enumerate-first leaf then overflows a 3K budget (3.1-3.5K).
def _context(layout: str, coded: bool = False, oneline: bool = False) -> str:
    kind = "coded words (random letter strings)" if coded else "words"
    lines = "all on one line" if oneline else "several per line"
    if layout == "dotted":
        return f"The document is a sequence of {kind} separated by ` .... `, {lines}. The dots are separators, not words."
    return (f"The document is a numbered list of {kind}: each entry is `<n>. <word>`, {lines}. The numbers are "
            f"positions, not words.")


def make_topk_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in TOPK_TASKS:
        raise ValueError(f"Unknown topk task: {task!r}")
    rng = random.Random(seed)
    vocab = _vocab(corpus_tokens, tokenizer)
    coded = random.Random(f"coded-{seed}").random() < _CODED_FRAC
    layout = rng.choice(["numbered", "dotted"])
    k = rng.choice([3, 5, 5, 8])
    hot = rng.sample(vocab, k)
    if coded:
        crng = random.Random(f"codedvocab-{seed}")
        layout = crng.choice(["numbered", "dotted"])
        vocab = sorted({_code(crng) for _ in range(6000)})
        hot = crng.sample(vocab, k)
        decoys = {_near_miss(crng, w) for w in hot for _ in range(2)} - set(hot)
    # Each LINE carries each hot word with p=0.75 (so a ~250-token leaf of 4-6 lines holds every hot
    # word 3-5 times) plus a few background words (mostly one-off; ~20% drawn from a small pool of
    # words that recur 2-4 times in total). Hot words therefore dominate every range, and a leaf's
    # pruned tally (words seen 2+ times) is exact for the top-k by construction — RULER cwe's regime.
    bg_words = [w for w in vocab if w not in hot]
    rng.shuffle(bg_words)
    if coded:                                   # decoys recur like repeaters: 2-4 times in total, never hot-like
        bg_words = sorted(decoys) + [w for w in bg_words if w not in decoys]
    repeaters = bg_words[:12]
    fresh = iter(bg_words[12:])
    per_line_bg = rng.randint(9, 15)      # ~15-23 words per line -> ~12-15 lines per 250-token leaf
    srng = random.Random(f"sparse-{seed}")
    if srng.random() < _SPARSE_FRAC:
        return _sparse_problem(task, srng, tokenizer, doc_size_tokens, hot, bg_words, layout, coded, seed)
    def dense_lines():
        while True:
            words = [w for w in hot if rng.random() < 0.75]
            for _ in range(per_line_bg):
                # when the fresh supply runs out (long docs), fall back to a uniform draw over ALL background
                # words — never the small repeater pool, which would otherwise accumulate hot-like counts
                words.append(rng.choice(repeaters) if rng.random() < 0.2 else next(fresh, rng.choice(bg_words[12:] or bg_words)))
            rng.shuffle(words)
            yield words
    doc_tokens, spans = _emit(tokenizer, dense_lines(), layout, doc_size_tokens, _oneline(seed))
    return _finish(task, tokenizer, doc_tokens, spans, k, layout, coded, hot, "dense")


# ONE-LINE documents (v16): RULER cwe/fwe are a single line (" ".join of the entries — no newlines at 40K). 18w's leaves
# narrated invented "lines at tokens 312-467" (synth_topk docs always had lines) and ownership-by-line would give a
# one-line doc wholly to the first leaf. Records are now WORDS (span = the word's own tokens, ownership = the word
# STARTS in the range) and 40% of documents have no line breaks at all.
_ONELINE_FRAC = 0.4


def _oneline(seed) -> bool:
    return random.Random(f"oneline-{seed}").random() < _ONELINE_FRAC


def _emit(tokenizer, lines, layout, doc_size_tokens, oneline):
    """Lay the words out one PIECE per word (separator + entry), tokenized piece by piece so every word has its own
    token span; line breaks between lines unless `oneline`. Stops before exceeding doc_size_tokens."""
    sep = {"dotted": " .... "}.get(layout, " ")
    doc_tokens, spans, n, first = [], [], 0, True
    for words in lines:
        for j, w in enumerate(words):
            lead = "" if first else ("\n" if (j == 0 and not oneline) else sep)
            entry = f"{n + 1}. {w}" if layout == "numbered" else w
            toks = tokenizer.encode(lead + entry, add_special_tokens=False)
            if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
                return doc_tokens, spans
            spans.append((len(doc_tokens), len(doc_tokens) + len(toks), n, w))
            doc_tokens.extend(toks); n += 1; first = False
    return doc_tokens, spans


def _line(words, layout, entry_no):
    if layout == "numbered":
        return " ".join(f"{entry_no + j + 1}. {w}" for j, w in enumerate(words)) + "\n"
    if layout == "dotted":
        return " .... ".join(words) + "\n"
    return " ".join(words) + "\n"


def _sparse_problem(task, rng, tokenizer, doc_size_tokens, hot, bg_words, layout, coded, seed):
    k = rng.choice([3, 5, 8, 10])
    # extra hot words (k > the dense draw) come from past the decoy/repeater head of bg_words — a coded problem's
    # near-miss decoys must stay background, never become hot themselves
    if k > len(hot):
        sig = {"".join(sorted(h)) for h in hot}          # a decoy is a letter swap: same sorted letters as its hot word
        extra = [w for w in bg_words[12:] if w not in hot and "".join(sorted(w)) not in sig]
        hot = hot + extra[:k - len(hot)]
    else:
        hot = hot[:k]
    bg = [w for w in bg_words if w not in hot]
    per_word = len(tokenizer.encode(_line(bg[:200], layout, 1000), add_special_tokens=False)) / 200
    n_words = int(doc_size_tokens / per_word * 1.05)
    # Background words recur at most `max_bg` times each (RULER: 3). The vocabulary is finite (~3.4K English words), so
    # a long doc needs more repeats; max_bg grows with the words needed per background word, and hot
    # words stay >= 3x max_bg so the top-k is always strictly separated.
    need = n_words / max(1, len(bg))
    max_bg = max(3, int(need * 1.6) + 1)
    c_hot = rng.randint(3 * max_bg, max(3 * max_bg, 30))
    seq = [w for w in hot for _ in range(c_hot)]
    bgc = Counter()
    for w in bg:
        if len(seq) >= n_words: break
        r = min(max_bg, max(1, round(rng.gauss(need, 0.8)))) if need > 1.5 else rng.choice([1, 1, 2, 3])
        seq += [w] * r; bgc[w] = r
    for w in bg:                                     # still short (rare): top words up to max_bg, never beyond
        if len(seq) >= n_words: break
        extra = max_bg - bgc[w]
        seq += [w] * extra; bgc[w] += extra
    rng.shuffle(seq)
    per_line = rng.randint(12, 20)
    doc_tokens, spans = _emit(tokenizer, (seq[i:i + per_line] for i in range(0, len(seq), per_line)), layout,
                              doc_size_tokens, _oneline(seed))
    return _finish(task, tokenizer, doc_tokens, spans, k, layout, coded, hot, "sparse")


def _finish(task, tokenizer, doc_tokens, spans, k, layout, coded, hot, regime):
    counts = Counter(w for *_, w in spans)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    gold = [w for w, _ in ranked[:k]]
    q = (f"Which {k} {'coded ' if coded else ''}words appear MOST often in the document? "
         f"{'Ignore the list numbers. ' if layout == 'numbered' else 'Ignore the dots. ' if layout == 'dotted' else ''}"
         f"List the {k} words, comma-separated, in \\boxed{{}}.")
    return Problem(
        document_tokens=doc_tokens, question=q, gold_answers=gold, task=task,
        task_context=_context(layout, coded, "\n" not in tokenizer.decode(doc_tokens)), grading_mode="set",
        metadata={"family": "bounded", "strategy_default": "binary", "task": task, "layout": layout,
                  "k": k, "regime": regime, "hot": hot, "oneline": "\n" not in tokenizer.decode(doc_tokens), "n_records": len(spans),
                  "record_spans": spans, "gold_int": None},
    )
