"""Held-out OOLONG eval with PER-DATASET + per-question-type breakdown.

Runs the recursive-agent harness (eval/agent.py) against a checkpoint over the
oolong_counting family, with EVEN coverage of all 10 source datasets (pinned, so
no dataset is starved), and scores with the OFFICIAL OOLONG grader. Separates
"can it aggregate" (the recursive count) from "can it classify" (per dataset),
which a single blended number hides.

    CKPT=$(cat ~/.cache/infinite-context/last_checkpoint.txt) uv run python -u eval_oolong.py
"""
import asyncio
import os
from collections import defaultdict

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

import train
from eval.agent import run_agent
from eval.backends import TinkerBackend
from tasks import grade_answer, resolve_eval_grading_mode
from tasks.oolong.generators import _DATASETS, make_oolong_problem

CKPT = os.environ.get("CKPT") or None
FAMILY = os.environ.get("FAMILY", "oolong_counting")
K_PER_DATASET = int(os.environ.get("K", "3"))   # problems per dataset
SEED_BASE = 4_000_000
CONCURRENCY = 4


async def main():
    tok = tokenizer_utils.get_tokenizer(train.MODEL_NAME)

    # Build work: K problems per dataset, dataset pinned for even coverage.
    work = []
    for di, ds in enumerate(_DATASETS):
        for k in range(K_PER_DATASET):
            seed = SEED_BASE + di * 1000 + k
            p = make_oolong_problem(FAMILY, [], tok, train.DOC_SIZE_TOKENS, seed, dataset=ds)
            work.append((ds, p))
    print(f"{FAMILY} | {len(_DATASETS)} datasets x {K_PER_DATASET} = {len(work)} problems "
          f"| ckpt={CKPT or 'BASE'}")

    import tinker

    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(base_model=train.MODEL_NAME, rank=train.LORA_RANK)
    if CKPT:
        fut = await tc.load_state_async(CKPT)
        await fut.result_async()
    sampling = await tc.save_weights_and_get_sampling_client_async()
    renderer = get_renderer(train.RENDERER_NAME, tok)
    backend = TinkerBackend(sampling, tok, renderer, temperature=1.0)

    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(ds, p):
        async with sem:
            root = await run_agent(
                backend, document_tokens=p.document_tokens, tokenizer=tok,
                task_context=p.task_context, question=p.question,
                budget=train.AGENT_CONTEXT, max_chunk_tokens=train.MAX_CHUNK_TOKENS,
                max_depth=train.MAX_DEPTH, max_turns=train.MAX_TURNS,
            )
        score = grade_answer(root.answer, p.gold_answers, resolve_eval_grading_mode(p))
        return ds, p, root, score

    results = await asyncio.gather(*[run_one(ds, p) for ds, p in work])

    by_dataset = defaultdict(list)
    by_qtype = defaultdict(list)
    for ds, p, root, score in results:
        by_dataset[ds].append(score)
        by_qtype[p.metadata["task_type"]].append(score)

    def line(name, scores):
        return f"  {name:22s} {sum(scores)/len(scores):.3f}  (n={len(scores)})"

    print("\n" + "=" * 60 + f"\nOOLONG {FAMILY} — official grader — ckpt={CKPT or 'BASE'}\n" + "=" * 60)
    print("\nBY DATASET (classification difficulty lens):")
    for ds in sorted(by_dataset, key=lambda d: -sum(by_dataset[d]) / len(by_dataset[d])):
        print(line(ds, by_dataset[ds]))
    print("\nBY QUESTION TYPE (aggregation lens):")
    for qt in sorted(by_qtype):
        print(line(qt, by_qtype[qt]))
    alls = [s for _, _, _, s in results]
    print(f"\nOVERALL: {sum(alls)/len(alls):.3f}  (n={len(alls)})")

    # A few concrete examples per outcome bucket for sanity.
    print("\nSAMPLES (dataset | gold -> answer | score):")
    for ds, p, root, score in sorted(results, key=lambda r: -r[3])[:4] + sorted(results, key=lambda r: r[3])[:4]:
        print(f"  {ds:14s} {str(p.gold_answers):>14s} -> {str(root.answer)[:24]:24s} {score:.2f}  "
              f"[{p.metadata['task_type']}]")


if __name__ == "__main__":
    asyncio.run(main())
