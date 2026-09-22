# Held-out scoreboard across runs

Computed from `eval_results/raw/*.jsonl` on 2026-09-22 (regenerate with the snippet at the bottom).
Doc 4,000 tokens; agent budget 3,000; temp 0.2; MAX_NODES 150; identical seeds across runs.
Runs 3-5 used 3 seeds/task and lack some tasks, so SCORE is only comparable from run 6 on (5 seeds/task).
SCORE = unweighted mean over the 9 held-out tasks; diagnostics = in-distribution tasks, excluded from SCORE.

| task | run3 | run4 | run5 | run6 | run7 | 7w | 7w2 | 8w | 9w | 10w | 11w | 12w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| oolong_counting | 0.34* | 0.64* | 0.34* | 0.09 | 0.60 | 0.32 | 0.71 | 0.36 | 0.63 | 0.50 | 0.57 | 0.42 |
| oolong_user | 0.00* | 0.33* | 0.00* | 0.60 | 0.40 | 0.75 | 0.40 | 0.60 | 0.40 | 0.55 | 0.64 | 0.75 |
| oolong_temporal | 0.16* | 0.48* | 0.39* | 0.46 | 0.27 | 0.17 | 0.15 | 0.25 | 0.37 | 0.21 | 0.20 | 0.21 |
| niah_single_1 | 0.67* | 0.33* | 1.00* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| niah_multikey_1 | 0.33* | 0.00* | 1.00* | 0.80 | 1.00 | 1.00 | 1.00 | 0.80 | 0.60 | 0.80 | 0.40 | 0.80 |
| niah_multiquery | — | 0.67* | 0.83* | 0.80 | 0.95 | 0.90 | 0.80 | 1.00 | 0.85 | 1.00 | 0.70 | 0.70 |
| vt | 0.00* | 1.00* | 0.20* | 0.44 | 0.48 | 0.36 | 0.48 | 0.88 | 0.16 | 0.36 | 0.88 | 0.96 |
| cwe | 0.97* | 0.00* | 0.93* | 0.96 | 0.52 | 0.68 | 0.92 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 |
| fwe | 0.89* | 0.00* | 0.78* | 0.93 | 0.73 | 0.73 | 0.53 | 0.93 | 0.80 | 1.00 | 1.00 | 1.00 |
| **SCORE** | — | **0.384** | **0.609** | **0.675** | **0.662** | **0.657** | **0.666** | **0.758** | **0.646** | **0.711** | **0.709** | **0.760** |
| diagnostics | 0.900 | 0.954 | 0.886 | 0.946 | 0.911 | 0.953 | 0.989 | 0.958 | 1.000 | 1.000 | 0.958 | 1.000 |

`*` = 3 seeds rather than 5.

**Best-of-per-task oracle over the 5-seed runs: 0.876** — oolong_counting 0.71 (7w2), oolong_user 0.75 (7w), oolong_temporal 0.46 (run6), niah_single_1 1.00 (run7), niah_multikey_1 1.00 (run7), niah_multiquery 1.00 (8w), vt 0.96 (12w), cwe 1.00 (9w), fwe 1.00 (12w)

No single checkpoint reaches this; it is the ceiling of the recipe as of run 12.

## Checkpoints

| run | checkpoint | lineage |
|---|---|---|
| run 7 | `tinker://8ec58e87-7310-55dd-a31d-9f9caf3c0382:train:0/weights/sft_general7` | from base |
| 8w | `tinker://75b29e89-caf6-5e79-8415-5f4e64af39a3:train:0/weights/sft_general8w` | warm from run 7 |
| 9w / 10w / 11w / 12w | `…:train:0/weights/sft_general{9w,10w,11w,12w}` | each warm from 8w |
| 12w | `tinker://a8b5c74c-a05e-52cc-9cbc-797739ee23c5:train:0/weights/sft_general12w` | warm from 8w — best SCORE |
| run 13 | (training 2026-09-22) | FROM BASE on run-12 text at run-7 scale |

## Regenerate

```bash
python3 - <<'PY'
import json, collections
S=['oolong_counting','oolong_user','oolong_temporal','niah_single_1','niah_multikey_1','niah_multiquery','vt','cwe','fwe']
rows=[json.loads(l) for l in open('eval_results/raw/sft_general12w.jsonl')]
d=collections.defaultdict(list)
for r in rows: d[r['task']].append(float(r.get('score') or 0))
print(sum(sum(d[t])/len(d[t]) for t in S)/9)
PY
```
