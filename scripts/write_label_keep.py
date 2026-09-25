"""Write the labeled_records keep-lists from a label_judge.py run: rows whose Opus verdict equals gold.

    PYTHONPATH=. uv run python scripts/write_label_keep.py eval_results/label_judge_full.json
"""
import collections, json, os, sys

from tasks.labeled.generators import _CACHE, _row_id

recs = json.load(open(sys.argv[1]))
keep, n = collections.defaultdict(list), collections.Counter()
for r in recs:
    n[r["src"]] += 1
    if r["opus_cat"] == "agree":
        keep[r["src"]].append(_row_id(r))
for src, ids in keep.items():
    json.dump(sorted(set(ids)), open(os.path.join(_CACHE, f"{src}.opus_keep.json"), "w"))
    print(f"{src:8s} kept {len(ids):5d} / {n[src]:5d} ({len(ids) / n[src]:.0%})")
