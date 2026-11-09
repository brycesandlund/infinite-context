"""Cache small labeled text-classification samples for the `labeled_records` task.

Datasets with taxonomies DISJOINT from OOLONG's (TREC / IMDB / AG News / negation / Yahoo):
  dbpedia  — DBpedia-14 ontology classes (Company, Artist, Village, ...)
  emotion  — dair-ai/emotion (sadness, joy, love, anger, fear, surprise)
  yelp     — Yelp polarity (positive / negative)  [reviews, distinct from IMDB movie reviews]
Writes ~/.cache/infinite-context/labeled/<name>.jsonl rows {"text", "label"} with human-readable
labels; texts are clipped to ~60 words so an item is one line of 20-90 tokens.
"""
import json, os, random, re
import datasets as hf
hf.disable_progress_bars()

OUT = os.path.expanduser("~/.cache/infinite-context/labeled")
os.makedirs(OUT, exist_ok=True)
N = 4000

SPECS = {
    "dbpedia": ("fancyzhx/dbpedia_14", "train", "content", "label"),
    "emotion": ("dair-ai/emotion", "train", "text", "label"),
    "yelp": ("fancyzhx/yelp_polarity", "train", "text", "label"),
}

def clip(t: str, max_words=60) -> str:
    t = t.replace("\\n", " ").replace('\\"', '"').replace("\\'", "'")   # CSV-style escapes in yelp
    t = re.sub(r"\s+", " ", t).strip()
    w = t.split()
    return " ".join(w[:max_words]) + ("…" if len(w) > max_words else "")

for name, (path, split, tcol, lcol) in SPECS.items():
    ds = hf.load_dataset(path, split=split)
    names = ds.features[lcol].names
    rng = random.Random(0)
    idx = rng.sample(range(len(ds)), min(N, len(ds)))
    rows = []
    for i in idx:
        r = ds[i]
        text = clip(r[tcol])
        if len(text.split()) < 5:
            continue
        lab = names[r[lcol]]
        if name == "yelp":                       # HF feature names are "1"/"2"
            lab = {"1": "negative", "2": "positive"}[lab]
        row = {"text": text, "label": lab}
        if name == "yelp":                       # long form for the long-item regime (~400-700 tokens)
            row["text_long"] = clip(r[tcol], max_words=480)
        rows.append(row)
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(name, len(rows), "rows | labels:", dict(Counter(r["label"] for r in rows)))


# --- v15: harder judgment types (sentence-pair relations, style), from sources DISJOINT from OOLONG's (it uses
# multi_nli, bigbench metaphors, pavlick formality). Pair rows keep both halves (`a`, `b`); the generator joins them
# with a per-problem marker. `text` (canonical "a -> b") is only the row's identity. Each is Opus-filtered after
# caching (scripts/label_judge.py -> scripts/write_label_keep.py).
import csv, io, urllib.request

def _write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows), "rows | labels:", dict(Counter(r["label"] for r in rows)))

from collections import Counter

snli = hf.load_dataset("stanfordnlp/snli", split="train")
names = snli.features["label"].names
rng = random.Random(0)
rows = []
for i in rng.sample(range(len(snli)), 12000):
    r = snli[i]
    if r["label"] < 0: continue
    a, b = clip(r["premise"], 40), clip(r["hypothesis"], 40)
    rows.append({"a": a, "b": b, "text": f"{a} -> {b}", "label": names[r["label"]]})
    if len(rows) >= N: break
_write("snli", rows)

paws = hf.load_dataset("google-research-datasets/paws", "labeled_final", split="train")
rng = random.Random(0)
rows = []
for i in rng.sample(range(len(paws)), N):
    r = paws[i]
    a, b = clip(r["sentence1"], 40), clip(r["sentence2"], 40)
    rows.append({"a": a, "b": b, "text": f"{a} -> {b}", "label": "paraphrase" if r["label"] == 1 else "not paraphrase"})
_write("paws", rows)

rows = []
for part in ("fine-tuning/train_full.csv", "fine-tuning/test.csv"):
    url = f"https://huggingface.co/datasets/Cleanlab/stanford-politeness/resolve/main/{part}"
    for r in csv.DictReader(io.StringIO(urllib.request.urlopen(url).read().decode())):
        t, lab = clip(r.get("text") or r.get("prompt") or ""), (r.get("label_cat") or r.get("completion") or "").strip()
        if len(t.split()) >= 5 and lab in ("polite", "neutral", "impolite"):
            rows.append({"text": t, "label": lab})
rows = list({r["text"]: r for r in rows}.values())
_write("politeness", rows)
