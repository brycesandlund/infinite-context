"""SQuAD / HotpotQA loaders for the RULER QA tasks (qa_1, qa_2).

These are held-out *eval* tasks: real-text document QA where the golden
paragraph(s) are buried among distractor paragraphs from the same dataset.
Datasets are downloaded + cached on first use (RULER's download_qa_dataset.sh
URLs). Parsing mirrors RULER's read_squad / read_hotpotqa exactly.

Returned per dataset: (qas, docs) where
  docs : list[str]  — the full paragraph pool (golden + distractors)
  qas  : list[dict] — {query, outputs, context (golden doc idxs),
                       more_context (same-article distractor idxs, squad only)}
"""

from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "infinite-context"
# SQuAD is served from rajpurkar's stable URL. HotpotQA's original CMU host
# (curtis.ml.cmu.edu) is dead, so we load it from HuggingFace instead.
_SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


def _ensure_squad() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / "squad.json"
    if not path.exists():
        print(f"Downloading squad from {_SQUAD_URL} (cached to {path}) ...")
        tmp = path.with_suffix(".json.tmp")
        urllib.request.urlretrieve(_SQUAD_URL, tmp)
        tmp.rename(path)
        print(f"  done: {path} ({path.stat().st_size // (1024 * 1024)} MB)")
    return path


def _read_squad(path: Path):
    with open(path) as f:
        data = json.load(f)
    total_docs = sorted({p["context"] for d in data["data"] for p in d["paragraphs"]})
    idx = {c: i for i, c in enumerate(total_docs)}
    qas = []
    for d in data["data"]:
        more_docs = [idx[p["context"]] for p in d["paragraphs"]]
        for p in d["paragraphs"]:
            ctx_idx = idx[p["context"]]
            for q in p["qas"]:
                if not q.get("is_impossible") and q.get("answers"):
                    qas.append({
                        "query": q["question"],
                        "outputs": [a["text"] for a in q["answers"]],
                        "context": [ctx_idx],
                        "more_context": [i for i in more_docs if i != ctx_idx],
                    })
    return qas, total_docs


def _read_hotpotqa_hf():
    """Load HotpotQA distractor (validation) from HuggingFace and build the same
    (qas, docs) structure RULER's read_hotpotqa produced. Each example's context
    is its 10 paragraphs (2 golden supporting + 8 distractors); a doc is
    'title\\n<joined sentences>'."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    def ex_docs(ex) -> list[str]:
        return [
            f"{t}\n{''.join(s)}"
            for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])
        ]

    total_docs = sorted({d for ex in ds for d in ex_docs(ex)})
    idx = {c: i for i, c in enumerate(total_docs)}
    qas = [
        {
            "query": ex["question"],
            "outputs": [ex["answer"]],
            "context": [idx[d] for d in ex_docs(ex)],
        }
        for ex in ds
    ]
    return qas, total_docs


@lru_cache(maxsize=2)
def load_qa(dataset: str):
    """Return (qas, docs) for 'squad' or 'hotpotqa'. Cached in-process + on disk."""
    if dataset == "squad":
        return _read_squad(_ensure_squad())
    if dataset == "hotpotqa":
        return _read_hotpotqa_hf()
    raise ValueError(f"Unknown QA dataset {dataset!r}. Known: ['squad', 'hotpotqa']")
