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
    t = re.sub(r"\s+", " ", t.replace("\\n", " ")).strip()
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
