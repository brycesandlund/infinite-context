"""labeled_records — classification + aggregation over REAL labeled text, gold labels from the
dataset (no model calls), the recipe that reached >60% on OOLONG counting in the OOLONG-only SFT.

The document is one item per line, each tagged with a section and an author:
    [S2] [by Kim] Absolutely loved the brunch here, the staff were …
The task context names the LABEL SET (the dataset's taxonomy) and one line per label; the leaf
must judge each item's label — there is no rule to check, the label is a judgment — and the trace
shows the gold label per item before tallying:  - S2 "Absolutely loved the brunch…" → label: positive → positive:3
Datasets (cached by scripts/cache_labeled.py): dbpedia (14 ontology classes), emotion (6), yelp (2).
Questions: count / most_common / relative / sections_cmp (2-D) / section_most (2-D) / author_most
(filter by author -> most common label) / author_top (which author has the most items of label L —
argmax over the OUTER key) / dates_rep_k (how many distinct dates appear exactly k times — a
histogram-of-counts reduction).
KEY MODES: half the problems tag items with a section `[S3]`; the other half with a full DATE
`[Jul 28, 2022]`, and the 2-D questions are per MONTH — the outer key must be DERIVED (Jul 28, 2022 →
Jul 2022), which no other training task requires and OOLONG temporal does.
`record_spans` = (ts, te, idx, label, outer_key, author, snippet, date_or_None).
"""

from __future__ import annotations

import json
import os
import random
import re
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
    "claims": "Each item is a one-sentence factual claim about a named thing; its label is whether the claim is True or False.",
}
# the labelled property, for the "The {prop} can be classified …" context variants
_PROP = {"dbpedia": "kind of thing each item describes", "emotion": "emotion each message expresses",
         "yelp": "overall sentiment of each review", "claims": "truth of each claim"}
_AUTHORS = ["Cho", "Diaz", "Han", "Ivanov", "Kim", "Lee", "Okafor", "Park", "Rossi", "Sato"]
_QTYPES = ["count", "count", "most_common", "relative", "sections_cmp", "section_most", "author_most", "author_top",
           # run 10: OOLONG-user / temporal shapes we lacked — "which author has the most items" (open key space of
           # names/ids, no label), "least common label among one author's items", "in how many months is L the
           # single most common label" (strict mode per outer key)
           "author_count", "author_least", "section_mode_count",
           # two named authors compared on one label; the question names them WITH the descriptor word
           # ("author 30140 or author 92806") and the form wants only the id — the OOLONG-user yahoo seed lost
           # the same way in 3 runs (`User: User 30140`, `user 30140`) with the tally itself correct
           "author_cmp",
           # run 15: filtered-subset questions were 2 of 12 qtypes and run 14w's OOLONG-user leaves dropped the user
           # filter (tallied every line). Author-filtered share raised to ~30%: a filtered COUNT ("how many of user
           # X's items are L" — the agnews seed shape we never taught) and extra author_most/least draws.
           "author_label_count", "author_label_count", "author_most", "author_least"]
# author_count / author_top ask about a SUBSET of authors this often (OOLONG-user: "only consider the subset of users
# with IDs …; which user is represented most often / has the most instances with the label L"). The state then holds
# only the listed authors. Audit 2026-09-24: no training question had this shape.
_AUTHOR_SUBSET_FRAC = 0.5
# Author-filtered questions name 2 authors this often (OOLONG: "associated with user IDs …" may list several).
_MULTI_AUTHOR_FRAC = 0.3
# 40% of documents tag authors with numeric ids instead of names, so the open-set contract ("the author exactly
# as written in the tag") and the `Author: [X]` form both see id-like keys (run 7/8w/9w: `User: User 30140`).
_ID_AUTHOR_FRAC = 0.4
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_ROWS: dict[str, list[dict]] = {}


def _pick_authors(rng, authors) -> list[str]:
    """1 author, or 2 (sorted) with prob _MULTI_AUTHOR_FRAC."""
    return sorted(rng.sample(authors, 2)) if rng.random() < _MULTI_AUTHOR_FRAC else [rng.choice(authors)]


def _pick_subset(rng, authors) -> list[str]:
    """2-3 of the document's 3-5 authors (always a proper subset), sorted."""
    return sorted(rng.sample(authors, rng.randint(2, min(3, len(authors) - 1))))


def _author_cond(aus) -> str:
    if len(aus) == 1:
        return f"author = {aus[0]} (the `[by {aus[0]}]` tag)"
    return f"author {' or '.join(aus)} (the `[by …]` tag)"


def _rows(name: str) -> list[dict]:
    if name not in _ROWS:
        with open(f"{_CACHE}/{name}.jsonl") as f:
            _ROWS[name] = [json.loads(l) for l in f]
    return _ROWS[name]


# How much of the document's SCHEMA the system prompt gives away. OOLONG — the eval this task is the
# proxy for — describes only the item type and the label set ("86 general-knowledge questions ... one of
# 6 categories: ..."); it never describes its `Date: … || User: … || Instance: …` columns, which the
# model must discover by reading. We always described our `[S<n>]` / `[by <author>]` / `[Mon DD, YYYY]`
# tags AND the month-derivation rule, so the root learned to be TOLD the key space — and at eval it
# invented one (`U0`..`U4`, "a number 1..12", "the month name as written"). `schema_lite` matches
# OOLONG's information content: item type + label set only.
_SCHEMA_LITE_FRAC = 0.4
# JUDGMENT-WORDING JITTER (v15). Every question used to open with "Judge each item's label (one of: …)." and every
# task context said "you must judge each item yourself", so the root's "(judging each item's label)" subtask phrase was
# always copyable from its input. OOLONG never says "judge" ("how many data points should be classified as label 'X'?",
# "…can be classified as one of…"); 17w's roots then wrote "(exact count)", "(inclusive)", "(no tag means the label is
# True)" — lookup contracts — and leaves looked labels up instead of judging (25/29 OOLONG roots @40K had no judgment
# instruction). Now the question opener, the count wording and the context's label sentence are each drawn from a pool
# (most variants never say "judge"; the direction — the label comes from the item's text — stays clear). The oracle's
# subtask is unchanged, so the root learns to STATE the judgment whatever its input says.
_Q_HEADS = [
    "",
    "",
    "",
    "Judge each item's label (one of: {L}). ",
    "Classify every item as one of: {L}. ",
    "Each item belongs to one of these labels: {L}. ",
    "Items can be classified as {L}. ",
    "The possible labels are {L}. ",
    "Consider each item's label ({L}). ",
]
_COUNT_ASKS = [
    "How many items are labelled `{q}`?",
    "How many items should be classified as label `{q}`?",
    "How many items have the label `{q}`?",
    "How many of the items are `{q}`?",
    "How many data points should be classified as `{q}`?",
    "Count the items whose label is `{q}`.",
]
_LABEL_SENTENCES = [
    "The label is NOT written in the text — you must judge each item yourself. The label set is exactly: {L}.",
    "The label is not written anywhere; it follows from the item's text. The label set is exactly: {L}.",
    "Each item can be classified as one of: {L}.",
    "The possible labels are: {L}.",
    "Items fall into exactly one of these labels: {L}.",
    "Every item has one label (not shown in the document), one of: {L}.",
    "The items can be classified into {n} labels: {L}.",
    "The {prop} can be classified into one of {n} categories: {L}.",
    "Each item falls into one of {n} categories: {L}.",
    "The {prop} can be classified as one of: {L}.",
]


# The question's item noun and "`L` items" construction (OOLONG: "instances with the label X", "data points … label
# 'X'"). Applied to the finished question, so every qtype gets it.
_ITEM_NOUNS = [("item", "items")] * 3 + [("instance", "instances"), ("data point", "data points"), ("entry", "entries"),
                                         ("line", "lines")]
_LABELLED_FORMS = ["`{L}` {n}", "{n} with the label `{L}`", "{n} labelled `{L}`"]


def _jitter_item_words(q: str, seed) -> str:
    sg, pl = random.Random(f"noun-{seed}").choice(_ITEM_NOUNS)
    form = _jitter(seed, "lform", _LABELLED_FORMS)
    q = re.sub(r"`([^`]+)` items\b", lambda m: form.format(L=m.group(1), n="items"), q)
    q = re.sub(r"\bitems\b", pl, q)
    q = re.sub(r"\bItems\b", pl[0].upper() + pl[1:], q)
    return re.sub(r"\bitem\b", sg, q)


def _jitter(seed, what: str, pool: list[str]) -> str:
    """Seeded pick from a wording pool, on its own rng so it never shifts any other draw of an existing seed."""
    return random.Random(f"{what}-{seed}").choice(pool)


def _context(name: str, labels: list[str], key_mode: str, schema_lite: bool = False, seed=None) -> str:
    prop = _PROP[name]
    lab = _jitter(seed, "labsent", _LABEL_SENTENCES).format(L=", ".join(labels), n=len(labels), prop=prop)
    if schema_lite:
        return f"The document is a list of text items, ONE per line. {_DESC[name]} {lab}"
    if key_mode == "section":
        tag, note = "a section tag `[S<n>]` (sections are numbered from 1 and appear in order)", ""
    else:
        tag = "the item's date `[Mon DD, YYYY]`"
        note = " Questions about MONTHS refer to the month and year of that date (e.g. `Jul 28, 2022` is in month `Jul 2022`)."
    return (
        f"The document is a list of text items, ONE per line. Each line starts with {tag} and an author tag "
        f"`[by <author>]`, followed by the item's text.{note} {_DESC[name]} {lab}"
    )


def _date_key(date: str):
    mon, day, year = date.replace(",", "").split()
    return (int(year), _MONTHS.index(mon), int(day))


def _month_key(date: str) -> str:
    mon, _, year = date.replace(",", "").split()
    return f"{mon} {year}"


def make_labeled_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in LABELED_TASKS:
        raise ValueError(f"Unknown labeled task: {task!r}")
    rng = random.Random(seed)
    name = rng.choice(["dbpedia", "emotion", "emotion", "yelp", "yelp", "claims", "claims"])
    # Long-item regime: 20% of problems use full-length yelp reviews (up to ~700 tokens), so a leaf
    # owns at most one item, must extend its read 2-3 times to finish it, and neighbouring leaves see
    # only a fragment they do not own — the OOLONG-imdb shape that broke ownership in run 8w.
    long_items = rng.random() < 0.2
    if long_items:
        name = "yelp"
    rows = _rows(name)
    text_key = "text"
    if long_items and rows and "text_long" in rows[0]:
        text_key = "text_long"
        rows = [r for r in rows if len(r["text_long"].split()) >= 220]   # genuinely long reviews (~300-700 tokens)
    labels = sorted({r["label"] for r in rows})
    # Restrict dbpedia to a random subset of 4-6 of its 14 classes per problem (a 14-way tally is
    # too wide for the budget and rarely what a question is about); items are drawn from those.
    if name == "dbpedia":
        labels = sorted(rng.sample(labels, rng.randint(4, 6)))
        rows = [r for r in rows if r["label"] in labels]
    key_mode = rng.choice(["section", "date"])
    # Joint outer x label key space capped at ~18 (the binary combine holds two children's tallies plus
    # the merge; 6 months x 6 emotions = 36 multi-token keys measured 3.6k real tokens at internal nodes).
    n_sections = rng.randint(3, min(6, max(3, 18 // len(labels))))
    if rng.random() < _ID_AUTHOR_FRAC:
        authors = sorted(str(x) for x in rng.sample(range(10000, 99999), rng.randint(3, 5)))
    else:
        authors = sorted(rng.sample(_AUTHORS, rng.randint(3, 5)))
    per_sec = max(1, (doc_size_tokens // 45) // n_sections)     # ~45 tok/item
    if key_mode == "date":
        # 3-6 months (calendar order, may span years); each month gets a pool of ~5-9 specific dates so
        # dates REPEAT (for dates_rep_k) and every month has several items.
        ym = sorted(rng.sample([(y, m) for y in (2022, 2023, 2024, 2025) for m in range(12)], n_sections))
        months = [f"{_MONTHS[m]} {y}" for y, m in ym]
        date_pool = {mk: [f"{mk.split()[0]} {d:02d}, {mk.split()[1]}" for d in sorted(rng.sample(range(1, 29), rng.randint(8, 14)))]  # many dates per month, each carried by ~1-4 items (for dates_rep_k)
                     for mk in months}

    order = rng.sample(range(len(rows)), min(len(rows), 600))
    doc_tokens, recs = [], []
    for i, ri in enumerate(order):
        r = rows[ri]
        au = rng.choice(authors)
        if key_mode == "section":
            # section = position in the document, so all n_sections are populated whatever the item length
            outer, date = f"S{min(n_sections, len(doc_tokens) * n_sections // doc_size_tokens + 1)}", None
            tag = f"[{outer}]"
        else:
            outer = rng.choice(months)
            date = rng.choice(date_pool[outer])
            tag = f"[{date}]"
        line = f"{tag} [by {au}] {r[text_key]}\n"
        toks = tokenizer.encode(line, add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        t = r[text_key]
        recs.append({"idx": i, "sec": outer, "au": au, "label": r["label"], "date": date,
                     # claims: the judged content is the PREDICATE, so quote most of the sentence
                     "snip": t if name == "claims" else ((t[:32] + "…") if len(t) > 35 else t),
                     "start": start, "end": len(doc_tokens)})
    if key_mode == "section":
        sections = sorted({r["sec"] for r in recs}, key=lambda x: int(x[1:]))
    else:
        sections = [m for m in months if any(r["sec"] == m for r in recs)]
    spans = [(r["start"], r["end"], r["idx"], r["label"], r["sec"], r["au"], r["snip"], r["date"]) for r in recs]
    unit_word = "section" if key_mode == "section" else "month"

    tally = Counter(r["label"] for r in recs)
    per = {s_: Counter(r["label"] for r in recs if r["sec"] == s_) for s_ in sections}
    qtypes = _QTYPES + (["dates_rep_k", "dates_rep_k", "first_month_cmp", "first_month_cmp", "before_after", "before_after"]
                        if key_mode == "date" else [])
    qtype = rng.choice(qtypes)
    head = _jitter(seed, "qhead", _Q_HEADS).format(L=", ".join(labels))
    ex = labels[0]
    if qtype == "count":
        L = rng.choice(labels)
        gold, grading, params = tally[L], "numeric", {"qlabel": L}
        ask = _jitter(seed, "countask", _COUNT_ASKS).format(q=L)
        q = head + f"{ask} Give the single integer in \\boxed{{}}."
    elif qtype == "most_common":
        gold = min((l for l in labels if tally[l] == max(tally[l2] for l2 in labels)))
        grading, params = "exact", {}
        q = head + f"Which label is the most common overall? Break ties by the alphabetically first label. Give the label in \\boxed{{}}."
    elif qtype == "relative":
        a, b = rng.sample(labels, 2)
        gold = ("more common than" if tally[a] > tally[b] else "less common than" if tally[a] < tally[b] else "equally common as")
        grading, params = "exact", {"qa": a, "qb": b}
        q = head + (f"Is `{a}` more common than, less common than, or equally common as `{b}`? Answer with exactly "
                    f"one of: more common than / less common than / equally common as. Put it in \\boxed{{}}.")
    elif qtype == "sections_cmp":
        a, b = rng.sample(labels, 2)
        gold = sum(1 for s_ in sections if per[s_][a] > per[s_][b])
        grading, params = "numeric", {"qa": a, "qb": b}
        q = head + f"In how many {unit_word}s are there STRICTLY more `{a}` items than `{b}` items? Give the single integer in \\boxed{{}}."
    elif qtype == "section_most":
        L = rng.choice(labels)
        best = max(per[s_][L] for s_ in sections)
        gold = next(s_ for s_ in sections if per[s_][L] == best)
        grading, params = "exact", {"qlabel": L}
        q = head + (f"Which {unit_word} has the MOST `{L}` items? Break ties by the earlier {unit_word}. Give the "
                    f"{unit_word} (e.g. {sections[0]}) in \\boxed{{}}.")
    elif qtype == "author_count":
        sub = _pick_subset(rng, authors) if rng.random() < _AUTHOR_SUBSET_FRAC else None
        pool = sub or authors
        c = Counter(r["au"] for r in recs)
        best = max(c[a] for a in pool)
        gold = next(a for a in pool if c[a] == best)              # pool sorted -> first in sort order on ties
        grading, params = "exact", ({"qauthor": sub[0], "qauthors": sub} if sub else {})
        tail = (f"Break ties by the author that comes first in alphabetical order. Give the author exactly as written in "
                f"the `[by …]` tag (e.g. {pool[0]}) in \\boxed{{}}.")
        q = (filtered_question(rng, "items", _author_cond(sub), "which author has the MOST items (regardless of label)", tail)
             if sub else f"Which author has the MOST items overall (regardless of label)? {tail}")
    elif qtype == "section_mode_count":
        L = rng.choice(labels)
        gold = sum(1 for s_ in sections if per[s_][L] > max((per[s_][l] for l in labels if l != L), default=0))
        grading, params = "numeric", {"qlabel": L}
        q = head + (f"For how many {unit_word}s is `{L}` the single most common label — i.e. that {unit_word} has "
                    f"STRICTLY more `{L}` items than items of any other one label? Give the single integer in \\boxed{{}}.")
    elif qtype == "author_label_count":
        aus = _pick_authors(rng, authors)
        L = rng.choice(labels)
        gold = sum(1 for r in recs if r["au"] in aus and r["label"] == L)
        grading, params = "numeric", {"qauthor": aus[0], "qauthors": aus, "qlabel": L}
        q = head + filtered_question(rng, "items", _author_cond(aus), f"how many are labelled `{L}`",
                                     "Give the single integer in \\boxed{}.")
    elif qtype == "author_least":
        aus = _pick_authors(rng, authors)
        c = Counter(r["label"] for r in recs if r["au"] in aus)
        present = [l for l in labels if c[l] > 0]
        gold = min(present, key=lambda l: (c[l], l)) if present else labels[0]
        grading, params = "exact", {"qauthor": aus[0], "qauthors": aus}
        q = head + filtered_question(rng, "items", _author_cond(aus),
                                     "which label is the LEAST common among the labels that appear at least once",
                                     f"Break ties by the alphabetically first label. Give the label (e.g. {ex}) in \\boxed{{}}.")
    elif qtype == "author_cmp":
        L = rng.choice(labels)
        a1, a2 = rng.sample(authors, 2)
        c = Counter(r["au"] for r in recs if r["label"] == L)
        gold = a1 if c[a1] > c[a2] else a2 if c[a2] > c[a1] else min(a1, a2)   # tie -> alphabetical first (stated)
        grading, params = "exact", {"qlabel": L, "qa1": a1, "qa2": a2}
        q = head + (f"Which author has more `{L}` items: author {a1} or author {a2}? If they are tied, answer the one "
                    f"that comes first in alphabetical order. Give the author exactly as written in the `[by …]` tag "
                    f"(e.g. {a1}) in \\boxed{{}}.")
    elif qtype == "author_top":
        L = rng.choice(labels)
        sub = _pick_subset(rng, authors) if rng.random() < _AUTHOR_SUBSET_FRAC else None
        pool = sub or authors
        c = Counter(r["au"] for r in recs if r["label"] == L)
        best = max((c[a] for a in pool), default=0)
        gold = next(a for a in pool if c[a] == best)              # pool sorted -> alphabetical tie
        grading, params = "exact", ({"qlabel": L, "qauthor": sub[0], "qauthors": sub} if sub else {"qlabel": L})
        tail = (f"Break ties by the author that comes first in alphabetical order. Give the author exactly as written in "
                f"the `[by …]` tag (e.g. {pool[0]}) in \\boxed{{}}.")
        q = head + (filtered_question(rng, "items", _author_cond(sub), f"which author has the MOST `{L}` items", tail)
                    if sub else f"Which author has the MOST `{L}` items? {tail}")
    elif qtype == "dates_rep_k":
        # Scoped to ONE month so the per-date tally stays <= ~14 keys (a whole-document per-date dict
        # overflowed the root); the leaf must derive each item's month to decide whether it counts.
        # Whole-document variant (run 12): OOLONG asks "how many dates are represented exactly k times" over the
        # WHOLE document, and the 11w root had to improvise the contract (one leaf then returned a scalar
        # `dates=10`). Taught when the document's distinct dates fit the budget (<= 40); otherwise month-scoped.
        all_dates = Counter(r["date"] for r in recs)
        whole = len(all_dates) <= 40 and rng.random() < 0.5
        mon = None if whole else rng.choice(sections)
        dc = all_dates if whole else Counter(r["date"] for r in recs if r["sec"] == mon)
        k = rng.choice([1, 1, 2, 3])
        gold, grading, params = sum(1 for v in dc.values() if v == k), "numeric", {"qk": k, "qmon": mon}
        scope = "In the whole document" if whole else f"Considering only items dated in {mon}"
        q = (f"{scope}: how many distinct dates are represented exactly {k} "
             f"time{'s' if k != 1 else ''} (i.e. exactly {k} item{'s carry' if k != 1 else ' carries'} that exact date)? "
             f"Give the single integer in \\boxed{{}}.")
    elif qtype == "first_month_cmp":
        a, b = rng.sample(labels, 2)
        hit = next((m for m in sections if per[m][a] > per[m][b]), None)
        gold, grading, params = (hit or "none"), "exact", {"qa": a, "qb": b}
        q = head + (f"In which month did `{a}` FIRST occur more often than `{b}`? Months are in chronological order; "
                    f"answer with the month and year exactly as written in the items' dates (e.g. {sections[0]}), or "
                    f"none if it never happens. Give it in \\boxed{{}}.")
    elif qtype == "before_after":
        # OOLONG-temporal semantics: compare the SHARE of items with label L before date D vs on/after D.
        L = rng.choice(labels)
        dates_sorted = sorted({r["date"] for r in recs}, key=_date_key)
        D = dates_sorted[rng.randint(len(dates_sorted) // 4, 3 * len(dates_sorted) // 4)]
        before = [r for r in recs if _date_key(r["date"]) < _date_key(D)]
        after = [r for r in recs if _date_key(r["date"]) >= _date_key(D)]
        fb = sum(r["label"] == L for r in before) / max(1, len(before))
        fa = sum(r["label"] == L for r in after) / max(1, len(after))
        gold = "more common" if fb > fa + 1e-9 else "less common" if fb < fa - 1e-9 else "the same frequency"
        grading, params = "exact", {"qlabel": L, "qdate": D}
        q = head + (f"Was `{L}` more common, less common, or the same frequency among items dated before {D} as "
                    f"compared to items dated on or after {D}? 'Common' means the share of that period's items "
                    f"with the label. Answer with exactly one of: more common / less common / the same frequency. Put it in \\boxed{{}}.")
    else:  # author_most
        aus = _pick_authors(rng, authors)
        c = Counter(r["label"] for r in recs if r["au"] in aus)
        gold = min((l for l in labels if c[l] == max(c[l2] for l2 in labels))) if c else labels[0]
        grading, params = "exact", {"qauthor": aus[0], "qauthors": aus}
        q = head + filtered_question(rng, "items", _author_cond(aus), "which label is the MOST common",
                                     f"Break ties by the alphabetically first label. Give the label (e.g. {ex}) in \\boxed{{}}.")
    # Answer-form following (run 7: the model boxed `User: User 30140` for "in the form 'User: [X]'").
    # A third of exact-answer questions state a template; the gold is the template filled ONCE.
    answer_form = None
    if grading == "exact" and rng.random() < 0.35:
        word = {"most_common": "Label", "relative": "Answer", "section_most": unit_word.capitalize(),
                "author_most": "Label", "author_top": "Author", "author_count": "Author",
                "author_least": "Label", "author_cmp": "Author"}.get(qtype, "Answer")
        answer_form = word
        # drop the question's own "Give … in \boxed{}." sentence and replace it with the form instruction
        # the question's closing "Give … in \boxed{}." sentence may contain dots ("e.g. S1", "e.g. 30140"), so
        # match up to the boxed instruction at the END rather than stopping at the first dot
        q = re.sub(r"\s*Give [^\n]*? in \\boxed\{\}\.\s*$", "", q)        # "Give the author … (e.g. Kim) in \boxed{}."
        q = re.sub(r"\s*Put it in \\boxed\{\}\.\s*$", "", q)              # "…equally common as. Put it in \boxed{}."
        what = {"Label": "the label", "Author": "the author exactly as written in the `[by …]` tag (just the name or id)"}.get(word, "the answer")
        q += f" Give your final answer in the form '{word}: [X]', where [X] is {what}; put it inside \\boxed{{}}."
        gold = f"{word}: {gold}"
    q = _jitter_item_words(q, seed)
    return Problem(
        document_tokens=doc_tokens, question=q, gold_answers=[str(gold)], task=task,
        task_context=_context(name, labels, key_mode, schema_lite=rng.random() < _SCHEMA_LITE_FRAC, seed=seed),
        grading_mode=grading,
        metadata={"family": "bounded", "strategy_default": "binary", "task": task, "qtype": qtype, "q_head": head.strip(),
                  "dataset": name, "labels": labels, "sections": sections, "authors": authors,
                  "key_mode": key_mode, "answer_form": answer_form, "long_items": long_items,
                  "n_records": len(recs), "record_spans": spans, "gold_int": gold, **params},
    )
