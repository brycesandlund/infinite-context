"""Long-record aggregation: the document is a sequence of MULTI-LINE entries (a short header
plus 60-250 tokens of real prose), separated by blank lines, and the question aggregates
over the entries.

Every other training record is one short line, so "a record is a long block, a leaf owns
only a few of them, and one may straddle my range" is never demonstrated — that is the
layout regime of OOLONG-style data (long items, aggregation over many of them). The per-entry
op stays MECHANICAL (header fields, or whole-word counts in the body) so the oracle is exact;
what the model learns is record-boundary discipline on long items + aggregation on top.

Variant family (drawn per problem):
  layout ∈ {bar: `=== entry 17 | src=B | tag=K3 ===`, kv: `Entry: 17 / Source: B / Tag: K3` lines}
  prose  ∈ {novel, essay}
  qtype  ∈ {count_tag, src_most, tag_in_src (filter->argmax), src_tag_2d (2-D reduce),
            mention_count (entries whose body contains a word), mention_most (entry with the
            most occurrences of a word)}
`metadata["record_spans"]` = (tok_start, tok_end, entry_no, src, tag, word_occurrences, snippets).
"""

from __future__ import annotations

import random
import re
from collections import Counter

from tasks.base import Problem
from tasks.phrasing import filtered_question
from tasks.niah.generators import _filler

LONGREC_TASKS: dict[str, dict] = {
    "long_records": {"family": "bounded", "strategy": "binary"},
}

_SRCS = ["A", "B", "C"]
_TAGS = ["K1", "K2", "K3", "K4"]
_QTYPES = ["count_tag", "src_most", "tag_in_src", "src_tag_2d", "mention_count", "mention_most"]

_CONTEXT = {
    "bar": (
        "The document is a sequence of multi-line entries separated by blank lines. Each entry "
        "starts with a header line formatted as:\n"
        "  === entry <n> | src=<A|B|C> | tag=<K1|K2|K3|K4> ===\n"
        "followed by one or more paragraphs of body text (real prose). Entries are numbered in "
        "order starting at 1."
    ),
    "kv": (
        "The document is a sequence of multi-line entries separated by blank lines. Each entry "
        "starts with three header lines:\n"
        "  Entry: <n>\n  Source: <A|B|C>\n  Tag: <K1|K2|K3|K4>\n"
        "followed by one or more paragraphs of body text (real prose). Entries are numbered in "
        "order starting at 1."
    ),
}

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\b[a-z]{4,}\b")


def _render(layout: str, n: int, src: str, tag: str, body: str) -> str:
    if layout == "bar":
        return f"=== entry {n} | src={src} | tag={tag} ===\n{body}\n\n"
    return f"Entry: {n}\nSource: {src}\nTag: {tag}\n{body}\n\n"


def _bodies(rng: random.Random, sentences: list[str], tokenizer):
    """Yield prose bodies of ~60-250 tokens (1-2 paragraphs) from a sentence stream."""
    i = 0
    while i < len(sentences):
        target = rng.randint(60, 250)
        picked, ntok = [], 0
        while i < len(sentences) and ntok < target:
            s = sentences[i].strip()
            i += 1
            if not s:
                continue
            picked.append(s)
            ntok += len(tokenizer.encode(s, add_special_tokens=False)) + 1
        if not picked:
            return
        if len(picked) >= 4 and rng.random() < 0.4:      # two paragraphs
            k = rng.randint(2, len(picked) - 2)
            yield " ".join(picked[:k]) + "\n" + " ".join(picked[k:])
        else:
            yield " ".join(picked)


def _occ(word: str, body: str) -> tuple[int, list[str]]:
    """(occurrence count, up to 4 short '…a b WORD c d…' snippets) — the snippets let the
    oracle's leaf line SHOW the matches it counted rather than assert a number."""
    # whole word: not glued to letters, digits, or a hyphen ('earth-boxes' is not 'earth')
    ms = list(re.finditer(rf"(?<![\w-]){re.escape(word)}(?![\w-])", body))
    snips = []
    for m in ms[:4]:
        pre = body[max(0, m.start() - 25):m.start()].split(" ")[-3:]
        post = body[m.end():m.end() + 25].split(" ")[:3]
        snips.append(("…" + " ".join(pre) + word + " ".join(post) + "…").replace("\n", " "))
    return len(ms), snips


def _pick_word(rng: random.Random, bodies: list[str]) -> str:
    """A lowercase word (4+ letters) present in 15-70% of the entries — enough hits to make
    the aggregation non-trivial, few enough that most leaves see both hits and misses."""
    presence = Counter()
    for b in bodies:
        presence.update(set(_WORD.findall(b)))
    n = len(bodies)
    ok = [w for w, c in presence.items() if 0.15 * n <= c <= 0.70 * n]
    if not ok:
        ok = [w for w, _ in presence.most_common(20)] or ["the"]
    return rng.choice(sorted(ok))


def _argbest(counts: dict, best, order):
    target = best(counts.values())
    return min((k for k, v in counts.items() if v == target), key=order.index)


def _question_gold(qtype: str, recs: list[dict], word: str, rng: random.Random):
    """-> (question, gold, grading_mode, params)."""
    if qtype == "count_tag":
        t = rng.choice(_TAGS)
        gold = sum(1 for r in recs if r["tag"] == t)
        return (f"How many entries have tag={t}? Give the single integer in \\boxed{{}}.",
                gold, "numeric", {"qtag": t})
    if qtype == "src_most":
        c = Counter(r["src"] for r in recs)
        gold = _argbest(c, max, _SRCS) if c else _SRCS[0]
        return ("Which src has the MOST entries? Break ties by the alphabetically first src. "
                "Give the src letter (e.g. B) in \\boxed{}.", gold, "exact", {})
    if qtype == "tag_in_src":
        s = rng.choice(_SRCS)
        c = Counter(r["tag"] for r in recs if r["src"] == s)
        gold = _argbest(c, max, _TAGS) if c else _TAGS[0]
        q = filtered_question(
            rng, "entries", f"src={s}", "which tag is the MOST common",
            "Consider only tags that appear at least once among those entries; break ties by the "
            "alphabetically first tag. Give the tag (e.g. K2) in \\boxed{}.")
        return q, gold, "exact", {"qsrc": s}
    if qtype == "src_tag_2d":
        t = rng.choice(_TAGS)
        per = {s: Counter(r["tag"] for r in recs if r["src"] == s) for s in _SRCS}
        gold = sum(1 for s in _SRCS
                   if per[s][t] > max((n for k, n in per[s].items() if k != t), default=0))
        return (f"For how many of the three srcs (A, B, C) is tag={t} the single most common tag "
                f"among that src's entries — i.e. STRICTLY more tag={t} entries than entries of "
                f"any other one tag? Give the single integer in \\boxed{{}}.",
                gold, "numeric", {"qtag": t})
    if qtype == "mention_count":
        gold = sum(1 for r in recs if r["occ"] > 0)
        return (f"How many entries contain the word '{word}' anywhere in their body text "
                f"(whole-word, case-sensitive; a hyphenated compound does not count)? Give the single integer in \\boxed{{}}.",
                gold, "numeric", {"word": word})
    # mention_most
    best = max(recs, key=lambda r: (r["occ"], -r["n"]))
    return (f"Which entry contains the word '{word}' the MOST times (whole-word, case-sensitive, not part of a hyphenated compound; "
            f"counted within its body text)? Break ties by the lowest entry number. Give the "
            f"entry number in \\boxed{{}}.", best["n"], "numeric", {"word": word})


def make_longrec_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in LONGREC_TASKS:
        raise ValueError(f"Unknown longrec task: {task!r}")
    rng = random.Random(seed)
    layout = rng.choice(["bar", "kv"])
    prose_kind = "essay" if corpus_tokens and rng.random() < 0.5 else "novel"
    # Headers cost ~15 tokens per entry; pull a little extra prose so we never run short.
    prose = _filler(rng, tokenizer, int(doc_size_tokens * 1.15) + 500, prose_kind, corpus_tokens)
    prose = re.sub(r"\s*\n\s*", " ", prose).strip()
    sentences = _SENT.split(prose)

    doc_tokens, recs, spans = [], [], []
    for body in _bodies(rng, sentences, tokenizer):
        n = len(recs) + 1
        src, tag = rng.choice(_SRCS), rng.choice(_TAGS)
        toks = tokenizer.encode(_render(layout, n, src, tag, body), add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        recs.append({"n": n, "src": src, "tag": tag, "body": body, "start": start, "end": len(doc_tokens)})
    word = _pick_word(rng, [r["body"] for r in recs])
    for r in recs:
        r["occ"], r["snips"] = _occ(word, r["body"])
    qtype = rng.choice(_QTYPES)
    question, gold, grading, params = _question_gold(qtype, recs, word, rng)
    spans = [(r["start"], r["end"], r["n"], r["src"], r["tag"], r["occ"], r["snips"]) for r in recs]
    return Problem(
        document_tokens=doc_tokens, question=question, gold_answers=[str(gold)], task=task,
        task_context=_CONTEXT[layout], grading_mode=grading,
        metadata={"family": "bounded", "strategy_default": "binary", "task": task,
                  "qtype": qtype, "layout": layout, "prose": prose_kind,
                  "n_records": len(recs), "record_spans": spans, "gold_int": gold, **params},
    )
