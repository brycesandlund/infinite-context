"""Held-out OOLONG eval with PER-DATASET breakdown + trace dumps.

Runs the recursive-agent harness against a checkpoint over ALL THREE OOLONG
families, with EVEN coverage of the 10 source datasets (pinned), scores with the
OFFICIAL OOLONG grader, and dumps a few full rollout trees per family (a pass +
a fail) so we can see how the policy actually decomposes/classifies.

    CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) uv run python -u eval_oolong.py
Env: K (per dataset, default 2), FAMILIES (csv, default all 3), TRACES (per family, default 2).
"""
import asyncio
import os
from collections import defaultdict

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

import train
from eval.agent import flatten, run_agent
from eval.backends import TinkerBackend
from eval.run import _print_tree
from tasks import grade_answer, resolve_eval_grading_mode
from tasks.oolong.generators import _DATASETS, make_oolong_problem

CKPT = os.environ.get("CKPT") or None
FAMILIES = os.environ.get("FAMILIES", "oolong_counting,oolong_user,oolong_temporal").split(",")
K = int(os.environ.get("K", "2"))            # problems per dataset per family
TRACES = int(os.environ.get("TRACES", "2"))  # full trees to dump per family
SEED_BASE = 4_000_000
CONCURRENCY = 4


async def main():
    tok = tokenizer_utils.get_tokenizer(train.MODEL_NAME)
    import tinker

    sc = tinker.ServiceClient()
    tc = await sc.create_lora_training_client_async(base_model=train.MODEL_NAME, rank=train.LORA_RANK)
    if CKPT:
        print(f"Loading checkpoint: {CKPT}")
        fut = await tc.load_state_async(CKPT)
        await fut.result_async()
    sampling = await tc.save_weights_and_get_sampling_client_async()
    renderer = get_renderer(train.RENDERER_NAME, tok)
    backend = TinkerBackend(sampling, tok, renderer, temperature=1.0)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_one(p):
        async with sem:
            root = await run_agent(
                backend, document_tokens=p.document_tokens, tokenizer=tok,
                task_context=p.task_context, question=p.question,
                budget=train.AGENT_CONTEXT, max_chunk_tokens=train.MAX_CHUNK_TOKENS,
                max_depth=train.MAX_DEPTH, max_turns=train.MAX_TURNS,
            )
        return root, grade_answer(root.answer, p.gold_answers, resolve_eval_grading_mode(p))

    overall = {}
    for fam in FAMILIES:
        work = []
        for di, ds in enumerate(_DATASETS):
            for k in range(K):
                p = make_oolong_problem(fam, [], tok, train.DOC_SIZE_TOKENS,
                                        SEED_BASE + di * 1000 + k, dataset=ds)
                work.append((ds, p))
        print(f"\n### {fam}: {len(_DATASETS)} datasets x {K} = {len(work)} problems")
        res = await asyncio.gather(*[run_one(p) for _, p in work])

        by_ds = defaultdict(list)
        rows = []
        for (ds, p), (root, score) in zip(work, res):
            by_ds[ds].append(score)
            rows.append((ds, p, root, score))
        overall[fam] = rows

        print(f"--- {fam} BY DATASET (official grader) ---")
        for ds in sorted(by_ds, key=lambda d: -sum(by_ds[d]) / len(by_ds[d])):
            s = by_ds[ds]
            print(f"  {ds:14s} {sum(s)/len(s):.3f}  (n={len(s)})")
        alls = [s for _, _, _, s in rows]
        print(f"  {'OVERALL':14s} {sum(alls)/len(alls):.3f}  (n={len(alls)})")
        # overflow / no-answer health
        no_ans = sum(1 for _, _, r, _ in rows if r.answer is None)
        sizes = [len(flatten(r)) for _, _, r, _ in rows]
        print(f"  health: no_answer={no_ans}/{len(rows)}  mean_tree={sum(sizes)/len(sizes):.1f}")

    # Trace dumps: a fail + a pass per family.
    for fam in FAMILIES:
        rows = overall[fam]
        fails = [r for r in rows if r[3] < 1.0]
        passes = [r for r in rows if r[3] >= 1.0]
        picks = (fails[:TRACES] + passes[:1]) if fails else passes[:TRACES]
        for ds, p, root, score in picks:
            print("\n" + "#" * 80)
            print(f"# TRACE {fam} | dataset={ds} | qtype={p.metadata['task_type']} | "
                  f"gold={p.gold_answers} | answer={root.answer!r} | score={score:.2f}")
            print(f"# Q: {p.question.splitlines()[0][:150]}")
            print("#" * 80)
            _print_tree(root)


if __name__ == "__main__":
    asyncio.run(main())
