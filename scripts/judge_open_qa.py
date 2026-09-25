"""Report RULER-string AND LLM-judged accuracy for the open-ended QA tasks in saved eval rollouts.

    PYTHONPATH=. uv run python scripts/judge_open_qa.py eval_results/raw/sft_general17w_R17w_10k_b8k.jsonl [...]

For every rollout of qa_1 / qa_2 / narrativeqa: the string score already stored (RULER string_match_part / qa_part) and
an equivalence verdict from eval/qa_judge.py. Judgments are cached in eval_results/qa_judge_cache.json, so re-running
over the same rollouts costs nothing. Also lists every DISAGREEMENT (string != judge) for inspection.
"""
import asyncio
import collections
import json
import sys

from eval.qa_judge import QAJudge, bare_question

TASKS = {"qa_1", "qa_2", "narrativeqa"}
CACHE = "eval_results/qa_judge_cache.json"


async def main(paths):
    judge = QAJudge(CACHE)
    for path in paths:
        rows = [r for r in map(json.loads, open(path)) if r["task"] in TASKS]
        if not rows:
            continue
        verdicts = await asyncio.gather(*[judge.grade(bare_question(r["question"]), r["gold"], r["answer"]) for r in rows])
        judge.save()
        by = collections.defaultdict(lambda: [0.0, 0, 0])
        for r, v in zip(rows, verdicts):
            b = by[r["task"]]; b[0] += r["score"]; b[1] += v["correct"]; b[2] += 1
        print(f"== {path}")
        for t, (s, j, n) in sorted(by.items()):
            print(f"   {t:12s} string {s / n:.2f} | LLM-judged {j / n:.2f}  (n={n})")
        for r, v in zip(rows, verdicts):
            if round(r["score"]) != v["correct"]:
                print(f"     [{r['task']} {r['seed']}] string={r['score']:.0f} judge={v['correct']} gold={r['gold'][:2]} "
                      f"answer={str(r['answer'])[:70]!r}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
