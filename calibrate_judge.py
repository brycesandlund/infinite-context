"""One-off: calibrate the LLM judge against gold on OOLONG counting subtasks.

For real OOLONG chunks we know the exact label tally (example_spans), so we build
answers whose correctness we KNOW and measure whether the judge's score tracks it
— especially the false-positive rate (a WRONG answer scored high), the failure
mode that corrupts RL reward.

Two arms isolate where the judge struggles:
  raw  : source = chunk text only (no labels) -> judge must classify AND count
  gold : source = chunk with each example's gold label appended -> judge only counts

Dumps the FULL prompt + FULL judge response (incl. hidden reasoning if litellm
exposes it) for every trial to OUT, and prints aggregate metrics to stdout.

Run:  OPENAI_API_KEY=... uv run python calibrate_judge.py
"""

from __future__ import annotations

import asyncio
import os

from tinker_cookbook import tokenizer_utils

import rl
from eval.judge import JUDGE_SYSTEM, _SCORE_RE, Judge, make_judge
from tasks.oolong import OOLONG_DATASETS, make_oolong_problem

N_DATASETS = 5                # keep the full-response dump readable
RANGE_EXAMPLES = 18           # ~one leaf's worth per range
JUDGE_MAX_TOKENS = 4000       # give the reasoning model room (512 truncated to empty)
OUT = "/tmp/judge_calib.txt"


def _build_trials(tokenizer):
    trials = []
    for ds in OOLONG_DATASETS[:N_DATASETS]:
        p = make_oolong_problem("oolong_counting", [], tokenizer, 10000, seed=7, dataset=ds)
        spans = p.metadata["example_spans"]
        labels = sorted({s[2] for s in spans})
        grp = spans[:RANGE_EXAMPLES]
        if len(grp) < 4 or not labels:
            continue
        a, b = grp[0][0], grp[-1][1]
        label = labels[0]
        gold = sum(1 for s in grp if s[2] == label)
        chunk = tokenizer.decode(p.document_tokens[a:b])
        annotated = "\n".join(
            f"{tokenizer.decode(p.document_tokens[s[0]:s[1]]).split('Instance:',1)[-1].strip()[:80]} -> {s[2]}"
            for s in grp
        )
        task = (f"Read these examples. Count how many should be classified as label "
                f"'{label}'. Return only the integer count.")
        for given in sorted({gold, gold + 1, gold + 3, 0}):
            for arm, source in (("raw", chunk), ("gold", annotated)):
                trials.append(dict(
                    dataset=ds, label=label, gold=gold, given=given, err=abs(given - gold),
                    arm=arm, task=task, answer=str(given), source=source,
                ))
    return trials


def _build_robustness_trials(tokenizer):
    """REASONABLE (plausible count from a real chunk) vs BROKEN (the real failure
    modes: stalled, refusal, hedge, ungrounded). Tests whether the judge is a
    useful 'did this rollout behave sanely' signal, independent of exact counting."""
    trials = []
    for ds in OOLONG_DATASETS[:N_DATASETS]:
        p = make_oolong_problem("oolong_counting", [], tokenizer, 10000, seed=7, dataset=ds)
        spans = p.metadata["example_spans"]
        labels = sorted({s[2] for s in spans})
        grp = spans[:RANGE_EXAMPLES]
        if len(grp) < 4 or not labels:
            continue
        a, b = grp[0][0], grp[-1][1]
        label = labels[0]
        gold = sum(1 for s in grp if s[2] == label)
        chunk = tokenizer.decode(p.document_tokens[a:b])
        task = (f"Read these examples. Count how many should be classified as label "
                f"'{label}'. Return only the integer count.")
        def add(cat, answer, source=chunk):
            trials.append(dict(dataset=ds, label=label, gold=gold, given=answer, err=0,
                               cat=cat, arm=cat, task=task, answer=answer, source=source))
        # reasonable: a real agent that did the work and gave a plausible count
        add("reasonable", str(gold))
        add("reasonable", str(gold + 1))
        # broken: the failure modes we actually see in rollouts
        add("broken", "(no answer given)")
        add("broken", "I could not determine the count from the text.")
        add("broken", "approximately 5-8, but the line boundaries are unclear")
        add("broken", str(gold), source=None)   # ungrounded: plausible number, never read
    return trials


async def _grade_raw(model, system, user, max_tokens):
    """Return (visible_content, hidden_reasoning, finish_reason)."""
    import litellm

    litellm.drop_params = True
    resp = await litellm.acompletion(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    ch = resp.choices[0]
    msg = ch.message
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    return (msg.content or ""), reasoning, getattr(ch, "finish_reason", None)


# Sweep one or more grader models (comma-separated), so we can see where
# discrimination kicks in: GRADERS=openai/gpt-5.4-nano,openai/gpt-5.4-mini,openai/gpt-5.4
GRADERS = os.environ.get("GRADERS", rl.JUDGE_MODEL).split(",")


async def _grade_all(model, trials, helper):
    async def run_one(t):
        msgs = helper._prompt(t["task"], t["answer"], t["source"], None)
        content, reasoning, finish = await _grade_raw(
            model, msgs[0]["content"], msgs[1]["content"], JUDGE_MAX_TOKENS
        )
        m = list(_SCORE_RE.finditer(content))
        return {**t, "content": content, "reasoning": reasoning, "finish": finish,
                "score": (float(m[-1].group(1)) if m else None)}
    return await asyncio.gather(*(run_one(t) for t in trials))


def _report(model, graded, helper):
    slug = model.replace("/", "_")
    out = f"/tmp/judge_calib_{slug}.txt"
    with open(out, "w") as f:
        for i, t in enumerate(graded):
            msgs = helper._prompt(t["task"], t["answer"], t["source"], None)
            f.write(f"\n{'#'*90}\n# trial {i} | dataset={t['dataset']} arm={t['arm']} "
                    f"label={t['label']!r} gold={t['gold']} given={t['given']} err={t['err']} "
                    f"| parsed_score={t['score']} finish={t['finish']}\n{'#'*90}\n")
            f.write(f"[system]\n{msgs[0]['content']}\n\n[user]\n{msgs[1]['content']}\n\n")
            f.write(f"[judge.reasoning_content]\n{t['reasoning'] or '(none exposed)'}\n\n")
            f.write(f"[judge.content]\n{t['content'] or '(EMPTY)'}\n")

    parsed = sum(1 for t in graded if t["score"] is not None)
    print(f"\n################  {model}  ################")
    print(f"parse rate: {parsed}/{len(graded)}  | full dump -> {out}")
    for arm in ("raw", "gold"):
        rows = [t for t in graded if t["arm"] == arm and t["score"] is not None]
        if not rows:
            print(f"  arm={arm}: no parsed rows"); continue
        correct = [t for t in rows if t["err"] == 0]
        wrong = [t for t in rows if t["err"] > 0]
        mean = lambda xs: (sum(t["score"] for t in xs) / len(xs)) if xs else float("nan")
        fp = sum(1 for t in wrong if t["score"] > 0.5) / len(wrong) if wrong else 0.0
        fn = sum(1 for t in correct if t["score"] <= 0.5) / len(correct) if correct else 0.0
        gap = mean(correct) - mean(wrong)
        print(f"  arm={arm:4s} | correct {mean(correct):.2f}  wrong {mean(wrong):.2f}  "
              f"GAP {gap:+.2f} | FP {fp:.0%} (n={len(wrong)})  FN {fn:.0%} (n={len(correct)})")


async def main():
    tokenizer = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
    trials = _build_trials(tokenizer)
    helper = Judge(backend=make_judge(rl.JUDGE_MODEL).backend)  # reuse _prompt()
    print(f"{len(trials)} trials | max_tokens={JUDGE_MAX_TOKENS} | graders: {GRADERS}\n"
          "(want: GAP large, FP low, FN low)")
    for model in GRADERS:
        try:
            graded = await _grade_all(model.strip(), trials, helper)
            _report(model.strip(), graded, helper)
        except Exception as e:
            print(f"\n################  {model}  ################\n  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
