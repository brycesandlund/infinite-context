"""rule_label — classification-SHAPED aggregation over real prose with an exactly checkable label.

The document is real prose (novel or essay), one sentence per line, each line tagged with its
section: `[S3] "Come in," she said.` The question defines a LABEL RULE the reader must apply to
every sentence — a mechanical property, so gold is exact — and asks an aggregate over the labels:

  rules (one per problem; the rule is stated in the question AND the task context):
    dialogue/narration   sentence contains a quotation mark (" “ ”)
    question/statement   sentence ends with ?
    numeric/plain        sentence contains a digit
    named/unnamed        sentence contains a capitalized word that is not its first word
  questions (qtype, drawn per problem):
    count        how many sentences get label L                        (int)
    most_common  which label is more common                             (exact)
    relative     is L1 more common / less common / equally common vs L2 (exact)
    sections_cmp in how many sections are there strictly more L1 than L2 (int; 2-D)
    section_most which section has the most L sentences                (exact; 2-D)

Why: run 6's OOLONG misses were leaves that JUDGED ten items and wrote one line, and leaves that
invented label names. Every other training leaf reads its label off the record; here the leaf must
decide a label per sentence, write the check it made, and tally — the per-item procedure of
classification — while the decision stays verifiable. Not OOLONG's data, labels, or layout.
`metadata["record_spans"]` = (tok_start, tok_end, idx, label, section, snippet).
"""

from __future__ import annotations

import random
import re
from collections import Counter

from tasks.base import Problem
from tasks.niah.generators import _filler
from tasks.phrasing import filtered_question

RULELABEL_TASKS: dict[str, dict] = {
    "rule_label": {"family": "bounded", "strategy": "binary"},
}

_QUOTES = '"“”'
_RULES = {
    # name: (label_if_true, label_if_false, rule text, checker)
    "dialogue": ("dialogue", "narration",
                 'contains a quotation mark (any of " “ ”)',
                 lambda s: any(q in s for q in _QUOTES)),
    "question": ("question", "statement",
                 "ends with a question mark",
                 lambda s: s.rstrip().rstrip('"”’\'').endswith("?")),
    "numeric": ("numeric", "plain",
                "contains a digit (0-9)",
                lambda s: any(ch.isdigit() for ch in s)),
    "named": ("named", "unnamed",
              "contains a capitalized word that is NOT its first word",
              lambda s: any(w[0].isupper() for w in re.findall(r"[A-Za-z][A-Za-z'’-]*", s)[1:])),
}
_QTYPES = ["count", "count", "most_common", "relative", "sections_cmp", "section_most"]

# Two fixed-width look-behinds (Python re disallows variable-width): sentence ends with .!? or
# with .!? followed by one closing quote, then whitespace, then an opening quote / capital / digit.
_SENT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’']))\s+(?=[\"“‘'(A-Z0-9])")


def _snip(s: str, n: int = 60) -> str:
    """Short evidence quote: keep both the head and the TAIL so the end-of-sentence mark is visible."""
    if len(s) <= n + 3:
        return s
    return s[:n - 18].rstrip() + " … " + s[-18:].lstrip()


def _sentences(prose: str) -> list[str]:
    prose = re.sub(r"\s*\n\s*", " ", prose).strip()
    out = []
    for s in _SENT.split(prose):
        s = s.strip()
        if 6 <= len(s.split()) <= 60 and not s.isupper() and any(c.islower() for c in s):
            out.append(s)
    return out


def _context(rule_text, lt, lf) -> str:
    return (
        "The document is real prose, ONE sentence per line. Each line starts with its section tag "
        "`[S<n>]` (sections are numbered from 1 and appear in order). Label every sentence by the "
        f"rule: it is `{lt}` if it {rule_text}, otherwise `{lf}`. The label is not written in the "
        "text — you must apply the rule to each sentence yourself."
    )


def make_rulelabel_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in RULELABEL_TASKS:
        raise ValueError(f"Unknown rulelabel task: {task!r}")
    rng = random.Random(seed)
    rule = rng.choice(list(_RULES))
    lt, lf, rule_text, check = _RULES[rule]
    prose_kind = "essay" if corpus_tokens and rng.random() < 0.4 else "novel"
    sents = _sentences(_filler(rng, tokenizer, int(doc_size_tokens * 1.3) + 800, prose_kind, corpus_tokens))
    n_sections = rng.randint(3, 6)

    # Pack sentences (one per line) until the doc budget; section tags cut the stream into
    # n_sections contiguous blocks of roughly equal size.
    doc_tokens, recs = [], []
    per_sec = max(1, (doc_size_tokens // 24) // n_sections)     # ~24 tok/sentence
    for i, s in enumerate(sents):
        sec = min(n_sections, i // per_sec + 1)
        line = f"[S{sec}] {s}\n"
        toks = tokenizer.encode(line, add_special_tokens=False)
        if doc_tokens and len(doc_tokens) + len(toks) > doc_size_tokens:
            break
        start = len(doc_tokens)
        doc_tokens.extend(toks)
        recs.append({"idx": i, "sec": f"S{sec}", "label": lt if check(s) else lf,
                     "snip": _snip(s), "start": start, "end": len(doc_tokens)})
    sections = sorted({r["sec"] for r in recs}, key=lambda x: int(x[1:]))
    spans = [(r["start"], r["end"], r["idx"], r["label"], r["sec"], r["snip"]) for r in recs]

    tally = Counter(r["label"] for r in recs)
    per = {s: Counter(r["label"] for r in recs if r["sec"] == s) for s in sections}
    qtype = rng.choice(_QTYPES)
    head = f"Label each sentence by the rule (`{lt}` if it {rule_text}, otherwise `{lf}`). "
    if qtype == "count":
        L = rng.choice([lt, lf])
        gold, grading = tally[L], "numeric"
        q = head + f"How many sentences are `{L}`? Give the single integer in \\boxed{{}}."
        params = {"qlabel": L}
    elif qtype == "most_common":
        gold = lt if tally[lt] >= tally[lf] else lf      # tie -> the rule's positive label
        grading = "exact"
        q = head + (f"Which label is more common overall? If tied, answer `{lt}`. Give the label in "
                    f"\\boxed{{}}.")
        params = {}
    elif qtype == "relative":
        a, b = (lt, lf) if rng.random() < 0.5 else (lf, lt)
        gold = ("more common than" if tally[a] > tally[b] else
                "less common than" if tally[a] < tally[b] else "equally common as")
        grading = "exact"
        q = head + (f"Is `{a}` more common than, less common than, or equally common as `{b}`? "
                    f"Answer in \\boxed{{}} with exactly one of: more common than / less common than / equally common as.")
        params = {"qa": a, "qb": b}
    elif qtype == "sections_cmp":
        a, b = (lt, lf) if rng.random() < 0.5 else (lf, lt)
        gold = sum(1 for s in sections if per[s][a] > per[s][b])
        grading = "numeric"
        q = head + (f"In how many sections are there STRICTLY more `{a}` sentences than `{b}` "
                    f"sentences? Give the single integer in \\boxed{{}}.")
        params = {"qa": a, "qb": b}
    else:  # section_most
        L = rng.choice([lt, lf])
        best = max(per[s][L] for s in sections)
        gold = next(s for s in sections if per[s][L] == best)      # tie -> earliest section
        grading = "exact"
        q = head + (f"Which section has the MOST `{L}` sentences? Break ties by the earlier section. "
                    f"Give the section tag (e.g. {sections[0]}) in \\boxed{{}}.")
        params = {"qlabel": L}
    return Problem(
        document_tokens=doc_tokens, question=q, gold_answers=[str(gold)], task=task,
        task_context=_context(rule_text, lt, lf), grading_mode=grading,
        metadata={"family": "bounded", "strategy_default": "binary", "task": task, "qtype": qtype,
                  "rule": rule, "label_true": lt, "label_false": lf, "rule_text": rule_text,
                  "sections": sections, "prose": prose_kind, "n_records": len(recs),
                  "record_spans": spans, "gold_int": gold, **params},
    )
