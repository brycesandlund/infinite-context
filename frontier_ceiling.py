"""Single-shot frontier-model ceiling on OOLONG-synth: give the model the WHOLE
document + question in one call (no harness, no decomposition, no budget limit),
grade with the same grader, on the SAME problems as eval/run.py. Tells us whether
the counting gap is the task (classification+counting) or our decomposition."""
import asyncio, os, re
import litellm
from tinker_cookbook import tokenizer_utils
import harness, tasks
from tasks.oolong import make_oolong_problem, oolong_spec

litellm.drop_params = True
MODEL = os.environ.get("FRONTIER", "gpt-5.4")
BASE = 2_000_000
N = int(os.environ.get("N", "10"))
TASKS = os.environ.get("TASKS", "oolong_counting,oolong_user,oolong_temporal").split(",")
SEM = asyncio.Semaphore(int(os.environ.get("CONC", "6")))
tok = tokenizer_utils.get_tokenizer("Qwen/Qwen3.6-35B-A3B")

async def one(task, idx):
    seed, ds = oolong_spec(task, idx, BASE)
    p = make_oolong_problem(task, [], tok, 10000, seed, dataset=ds)
    doc = tok.decode(p.document_tokens)
    prompt = f"{p.task_context}\n\n{doc}\n\n{p.question}"
    async with SEM:
        try:
            r = await litellm.acompletion(model=MODEL,
                messages=[{"role": "user", "content": prompt}], temperature=0)
            txt = r.choices[0].message.content or ""
        except Exception as e:
            txt = f"(error: {e})"
    ans = harness.extract_boxed(txt)
    sc = tasks.grade_answer(ans, p.gold_answers, tasks.resolve_eval_grading_mode(p))
    return task, ds, p.metadata.get("task_type"), p.gold_answers, ans, sc

async def main():
    print(f"FRONTIER single-shot ceiling | model={MODEL} | {N}/task | tasks={TASKS}")
    for task in TASKS:
        res = await asyncio.gather(*[one(task, i) for i in range(N)])
        for t, ds, qt, gold, ans, sc in res:
            print(f"  {ds:11s} {qt:18s} gold={str(gold):14s} ans={str(ans):14s} score={sc:.3f}")
        m = sum(x[5] for x in res) / len(res)
        print(f"== {task}: {m:.3f} ==")

asyncio.run(main())
