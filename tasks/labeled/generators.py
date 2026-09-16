"""labeled_records — classification + aggregation over REAL labeled text, gold labels from the
dataset (no model calls), the recipe that reached >60% on OOLONG counting in the OOLONG-only SFT.

The document is one item per line, each tagged with a section and an author:
    [S2] [by Kim] Absolutely loved the brunch here, the staff were …
The task context names the LABEL SET (the dataset's taxonomy) and one line per label; the leaf
must judge each item's label — there is no rule to check, the label is a judgment — and the trace
shows the gold label per item before tallying:  - S2 "Absolutely loved the brunch…" → label: positive → positive:3
Datasets (cached by scripts/cache_labeled.py): dbpedia (14 ontology classes), emotion (6), yelp (2).
Questions: count / most_common / relative / sections_cmp (2-D) / section_most (2-D) / author_most
(filter by author -> most common label).  `record_spans` = (ts, te, idx, label, section, author, snippet).
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter

from tasks.base import Problem
from tasks.phrasing import filtered_question

LABELED_TASKS: dict[str, dict] = {
    "labeled_records": {"family": "bounded", "strategy": "binary"},
}

_CACHE = os.path.expanduser("~/.cache/infinite-context/labeled")
_DESC = {
    "dbpedia": "Each item is a short encyclopedia-style description of a thing; its label is the KIND of thing described.",
    "emotion": "Each item is a short personal message; its label is the EMOTION the writer expresses.",
    "yelp": "Each item is a customer review of a business; its label is the review's overall SENTIMENT.",
}
_AUTHORS = ["Cho", "Diaz", "Han", "Ivanov", "Kim", "Lee", "Okafor", "Park", "Rossi", "Sato"]
_QTYPES = ["count", "count", "most_common", "relative", "sections_cmp", "section_most", "author_most"]
_ROWS: dict[str, list[dict]] = {}


def _rows(name: str) -> list[dict]:
    if name not in _ROWS:
        with open(f"{_CACHE}/{name}.jsonl") as f:
            _ROWS[name] = [json.loads(l) for l in f]
    return _ROWS[name]


def _context(name: str, labels: list[str]) -> str:
    return (
        "The document is a list of text items, ONE per line. Each line starts with a section tag "
        "`[S<n>]` (sections are numbered from 1 and appear in order) and an author tag `[by <name>]`, "
        f"followed by the item's text. {_DESC[name]} The label is NOT written in the text — you must "
        f"judge each item yourself. The label set is exactly: {', '.join(labels)}."
    )


def make_labeled_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in LABELED_TASKS:
        raise ValueError(f"Unknown labeled task: {task!r}")
    rng = random.Random(seed)
    name = rng.choice(["dbpedia", "emotion", "emotion", "yelp", "yelp"])
    rows = _rows(name)
    labels = sorted({r["label"] for r in rows})
    # Restrict dbpedia to a random subset of 4-6 of its 14 classes per problem (a 14-way tally is
    # too wide for the budget and rarely what a question is about); items are drawn from those.
    if name == "dbpedia":
        labels = sorted(rng.sample(labels, rng.randint(4, 6)))
        rows = [r for r in rows if r["label"] in labels]
    n_sections = rng.randint(3, 6)
    authors = sorted(rng.sample(_AUTHORS, rng.randint(3, 5)))
    per_sec = max(1, (doc_size_tokens // 45) // n_sections)     # ~45 tok/item

    order = rng.sample(range(len(rows)), min(len(rows), 600))
    doc_tokens, recs = [], []
    for i, ri in enumerate(order):
        r = rows[ri]
        sec = min(n_sections, i // per_sec + 1)
        au = rng.choice(authors)
        line = f"[S{sec}] [by {au}] {r['text']}\n"
        toks = tokenizer.encode(line, add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        t = r["text"]
        recs.append({"idx": i, "sec": f"S{sec}", "au": au, "label": r["label"],
                     "snip": (t[:42] + "…") if len(t) > 45 else t, "start": start, "end": len(doc_tokens)})
    sections = sorted({r["sec"] for r in recs}, key=lambda x: int(x[1:]))
    spans = [(r["start"], r["end"], r["idx"], r["label"], r["sec"], r["au"], r["snip"]) for r in recs]

    tally = Counter(r["label"] for r in recs)
    per = {s: Counter(r["label"] for r in recs if r["sec"] == s) for s in sections}
    qtype = rng.choice(_QTYPES)
    head = f"Judge each item's label (one of: {', '.join(labels)}). "
    ex = labels[0]
    if qtype == "count":
        L = rng.choice(labels)
        gold, grading, params = tally[L], "numeric", {"qlabel": L}
        q = head + f"How many items are labelled `{L}`? Give the single integer in \\boxed{{}}."
    elif qtype == "most_common":
        gold = min((l for l in labels if tally[l] == max(tally[l2] for l2 in labels)))
        grading, params = "exact", {}
        q = head + f"Which label is the most common overall? Break ties by the alphabetically first label. Give the label in \\boxed{{}}."
    elif qtype == "relative":
        a, b = rng.sample(labels, 2)
        gold = ("more common than" if tally[a] > tally[b] else "less common than" if tally[a] < tally[b] else "equally common as")
        grading, params = "exact", {"qa": a, "qb": b}
        q = head + (f"Is `{a}` more common than, less common than, or equally common as `{b}`? Answer with exactly "
                    f"one of: more common than / less common than / equally common as, in \\boxed{{}}.")
    elif qtype == "sections_cmp":
        a, b = rng.sample(labels, 2)
        gold = sum(1 for s in sections if per[s][a] > per[s][b])
        grading, params = "numeric", {"qa": a, "qb": b}
        q = head + f"In how many sections are there STRICTLY more `{a}` items than `{b}` items? Give the single integer in \\boxed{{}}."
    elif qtype == "section_most":
        L = rng.choice(labels)
        best = max(per[s][L] for s in sections)
        gold = next(s for s in sections if per[s][L] == best)
        grading, params = "exact", {"qlabel": L}
        q = head + f"Which section has the MOST `{L}` items? Break ties by the earlier section. Give the section tag (e.g. {sections[0]}) in \\boxed{{}}."
    else:  # author_most
        au = rng.choice(authors)
        c = Counter(r["label"] for r in recs if r["au"] == au)
        gold = min((l for l in labels if c[l] == max(c[l2] for l2 in labels))) if c else labels[0]
        grading, params = "exact", {"qauthor": au}
        q = head + filtered_question(rng, "items", f"author {au} (the `[by {au}]` tag)", "which label is the MOST common",
                                     f"Break ties by the alphabetically first label. Give the label (e.g. {ex}) in \\boxed{{}}.")
    return Problem(
        document_tokens=doc_tokens, question=q, gold_answers=[str(gold)], task=task,
        task_context=_context(name, labels), grading_mode=grading,
        metadata={"family": "bounded", "strategy_default": "binary", "task": task, "qtype": qtype,
                  "dataset": name, "labels": labels, "sections": sections, "authors": authors,
                  "n_records": len(recs), "record_spans": spans, "gold_int": gold, **params},
    )
