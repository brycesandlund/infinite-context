"""Synthetic claim verification from DBpedia-14 descriptions (exact gold, no OOLONG data).

Each description parses as "<Subject> is/was <predicate>". A TRUE claim is the sentence itself; a FALSE
claim swaps in the predicate of a description from a DIFFERENT class ("Carmeuse is a village in …").
Predicates are clipped to ~25 words. Output: ~/.cache/infinite-context/labeled/claims.jsonl rows
{"text", "label" in {True, False}}, balanced. This is the negation-style leaf judgment ("X may refer
to Y", true/false) on non-OOLONG data.
"""
import json, os, random, re

CACHE = os.path.expanduser("~/.cache/infinite-context/labeled")
rows = [json.loads(l) for l in open(f"{CACHE}/dbpedia.jsonl")]
pat = re.compile(r"^(?P<subj>[^.]{2,60}?)\s+(?P<verb>is|was|are|were)\s+(?P<pred>.+)$")
parsed = []
for r in rows:
    first = re.split(r"(?<=[a-z\)])\.\s", r["text"], maxsplit=1)[0]
    m = pat.match(first)
    if not m:
        continue
    subj, verb, pred = m.group("subj").strip(), m.group("verb"), m.group("pred").strip().rstrip(".")
    pw = pred.split()
    if len(pw) < 4:
        continue
    pred = " ".join(pw[:25]) + ("…" if len(pw) > 25 else "")
    parsed.append({"subj": subj, "verb": verb, "pred": pred, "cls": r["label"]})

rng = random.Random(0)
by_cls = {}
for p in parsed:
    by_cls.setdefault(p["cls"], []).append(p)
out = []
for p in parsed:
    out.append({"text": f"{p['subj']} {p['verb']} {p['pred']}.", "label": "True"})
    other_cls = rng.choice([c for c in by_cls if c != p["cls"]])
    q = rng.choice(by_cls[other_cls])
    out.append({"text": f"{p['subj']} {q['verb']} {q['pred']}.", "label": "False"})
rng.shuffle(out)
with open(f"{CACHE}/claims.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
from collections import Counter
print("claims", len(out), dict(Counter(r["label"] for r in out)))
for r in out[:4]:
    print("  ", r["label"], "|", r["text"][:120])
