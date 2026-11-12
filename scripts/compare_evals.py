"""Paired comparison of two eval runs on the SAME problems (e.g. a checkpoint on Tinker vs the same weights merged and
served by our vLLM stack).

    PYTHONPATH=. uv run python scripts/compare_evals.py <label>=<ref.jsonl>,<new.jsonl> [<label>=...] [--tasks=a,b,c]

Rollouts are matched on (task, seed) and kept only if the gold set is the same in both files (generators change; a
mismatch is reported, not scored). Per task: n, mean ref, mean new. Per file pair: the mean of the per-task means (how
SCORE / RULER-13 / the OOLONG chart are reported), the rollout-level paired difference with a sign-flip permutation
p-value and a bootstrap 95% interval, and the win/loss/tie counts. Both runs sample at TEMP > 0, so single rollouts
differ even between two runs of the same server; only the aggregate difference is a test.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict


def load(path: str) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["task"], r["seed"])] = r
    return out


def compare(label: str, ref_path: str, new_path: str, tasks: set[str] | None) -> None:
    ref, new = load(ref_path), load(new_path)
    keys = [k for k in ref if k in new and (tasks is None or k[0] in tasks)]
    # Same problem = same gold set. (RULER vt orders its variables by set iteration, so the same seed gives the same
    # variables and value in a different layout from one process to the next: counted as matched, reported as reordered.)
    same = [k for k in keys if sorted(map(str, ref[k]["gold"])) == sorted(map(str, new[k]["gold"]))]
    changed, reordered = defaultdict(int), defaultdict(int)
    for k in keys:
        if k not in same:
            changed[k[0]] += 1
        elif ref[k]["question"] != new[k]["question"] or ref[k]["gold"] != new[k]["gold"]:
            reordered[k[0]] += 1
    print(f"\n=== {label}: {len(same)} matched rollouts (ref {len(ref)}, new {len(new)})")
    if changed:
        print(f"    different problems for the same seed (excluded): {dict(changed)}")
    if reordered:
        print(f"    same gold set, different document layout (included): {dict(reordered)}")
    by = defaultdict(list)
    for k in same:
        by[k[0]].append((ref[k]["score"], new[k]["score"], ref[k].get("n_agents", 0), new[k].get("n_agents", 0)))
    print(f"    {'task':<18}{'n':>4}{'ref':>8}{'new':>8}{'diff':>8}   agents ref/new")
    tm_ref, tm_new = [], []
    for t in sorted(by):
        v = by[t]
        a, b = sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)
        tm_ref.append(a)
        tm_new.append(b)
        print(f"    {t:<18}{len(v):>4}{a:>8.3f}{b:>8.3f}{b - a:>+8.3f}   "
              f"{sum(x[2] for x in v) / len(v):.0f} / {sum(x[3] for x in v) / len(v):.0f}")
    if not tm_ref:
        return
    print(f"    mean of task means: ref {sum(tm_ref) / len(tm_ref):.3f}   new {sum(tm_new) / len(tm_new):.3f}   "
          f"diff {sum(tm_new) / len(tm_new) - sum(tm_ref) / len(tm_ref):+.3f}")
    d = [new[k]["score"] - ref[k]["score"] for k in same]
    n = len(d)
    mean = sum(d) / n
    rng = random.Random(0)
    reps = 20000
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d) / n) >= abs(mean) - 1e-12 for _ in range(reps))
    boots = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(4000))
    wins, losses = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    print(f"    rollout-level paired diff (new - ref): {mean:+.3f}   95% bootstrap [{boots[100]:+.3f}, {boots[3899]:+.3f}]   "
          f"sign-flip p = {hits / reps:.3f}")
    print(f"    new better on {wins}, worse on {losses}, equal on {n - wins - losses}")


def main() -> None:
    tasks = None
    for a in sys.argv[1:]:
        if a.startswith("--tasks="):
            tasks = set(a.split("=", 1)[1].split(","))
    for a in sys.argv[1:]:
        if a.startswith("--"):
            continue
        label, paths = a.split("=", 1)
        ref_path, new_path = paths.split(",")
        compare(label, ref_path, new_path, tasks)


if __name__ == "__main__":
    main()
