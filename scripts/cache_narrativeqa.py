"""Cache the NarrativeQA short-extractive subset for the `narrativeqa` bookqa-style task.

NarrativeQA answers are ABSTRACTIVE (human paraphrases) and its documents are whole books /
scripts (~200k tokens), so it does NOT drop into our window-around-the-answer + substring
pipeline as-is. We keep the tractable subset: questions with a SHORT reference answer (<=4
words) that appears VERBATIM in the full text, and store a ~60k-char window centered on that
occurrence (not the whole book — keeps the cache small and lets the generator re-window to
DOC_SIZE). Schema matches enqa.jsonl (id, question, answer, context) so the same
BookQAOracle + qa_part grading + model-executed leaf + rejection sampling apply unchanged.

Run:  N=2000 MAX_SEEN=8000 uv run python scripts/cache_narrativeqa.py
"""

import json
import os

from datasets import load_dataset

N = int(os.environ.get("N", "2000"))          # target kept candidates
MAX_SEEN = int(os.environ.get("MAX_SEEN", "9000"))  # bound streamed rows (bandwidth/time)
HALF_WIN = int(os.environ.get("HALF_WIN", "30000"))  # chars each side of the answer occurrence
MAX_WORDS = int(os.environ.get("MAX_WORDS", "4"))    # "short" answer cutoff

cache = os.path.expanduser("~/.cache/infinite-context/bookqa")
os.makedirs(cache, exist_ok=True)
out = os.path.join(cache, "narrativeqa.jsonl")


def _pick_short_verbatim(answers, text_lc):
    """Shortest reference answer (>=3 chars, <=MAX_WORDS words) that occurs verbatim in the
    full text (case-insensitive), or None."""
    cands = []
    for a in answers:
        a = a.strip().strip('"').strip()
        if len(a) >= 3 and len(a.split()) <= MAX_WORDS and a.lower() in text_lc:
            cands.append(a)
    return min(cands, key=len) if cands else None


ds = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
kept = seen = 0
with open(out, "w", encoding="utf-8") as f:
    for ex in ds:
        seen += 1
        q = ex["question"]["text"].strip()
        text = ex["document"]["text"]
        text_lc = text.lower()
        ans = _pick_short_verbatim([a["text"] for a in ex["answers"]], text_lc)
        if ans:
            pos = text_lc.find(ans.lower())
            lo = max(0, pos - HALF_WIN)
            window = text[lo: pos + HALF_WIN]
            f.write(json.dumps({
                "id": f"nqa_{kept}", "question": q, "answer": ans, "context": window,
            }) + "\n")
            kept += 1
        if kept >= N or seen >= MAX_SEEN:
            break
        if seen % 500 == 0:
            print(f"  seen={seen} kept={kept}", flush=True)
print(f"seen={seen} kept={kept} -> {out} ({os.path.getsize(out) // 1024 // 1024}MB)")
