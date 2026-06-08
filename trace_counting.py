"""One verbose oolong_counting rollout against a checkpoint (default: sft_warmstart).

Shows how the current policy attempts the tree-reduce decomposition + final count.
    CKPT=$(cat ~/.cache/infinite-context/last_sft_checkpoint.txt) uv run python trace_counting.py
"""
import asyncio
import os

from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

import train
from eval.agent import flatten, run_agent
from eval.backends import TinkerBackend
from eval.run import _print_tree
from tasks import grade_answer, load_pg_essays_text, make_problem, resolve_eval_grading_mode

CKPT = os.environ.get("CKPT") or None


async def main():
    tok = tokenizer_utils.get_tokenizer(train.MODEL_NAME)
    corpus = tok.encode(load_pg_essays_text(), add_special_tokens=False)

    # Find a numeric count question (the clearest tree-reduce target).
    problem = None
    for s in range(3_000_000, 3_000_060):
        cand = make_problem("oolong_counting", corpus, tok, train.DOC_SIZE_TOKENS, s)
        if cand.metadata["answer_type"] == "numeric" and "classified as label" in cand.question:
            problem = cand
            break
    print(f"dataset={problem.metadata['dataset']} n_examples={problem.metadata['n_examples']} "
          f"true_counts={problem.metadata['true_counts']}")
    print(f"QUESTION: {problem.question}")
    print(f"GOLD: {problem.gold_answers}\n")

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

    root = await run_agent(
        backend,
        document_tokens=problem.document_tokens,
        tokenizer=tok,
        task_context=problem.task_context,
        question=problem.question,
        budget=train.AGENT_CONTEXT,
        max_chunk_tokens=train.MAX_CHUNK_TOKENS,
        max_depth=train.MAX_DEPTH,
        max_turns=train.MAX_TURNS,
    )
    score = grade_answer(root.answer, problem.gold_answers, resolve_eval_grading_mode(problem))
    nodes = flatten(root)
    depth_counts = {}
    for n in nodes:
        depth_counts[n.depth] = depth_counts.get(n.depth, 0) + 1
    print(f"\n{'='*72}\nRESULT: answer={root.answer!r} gold={problem.gold_answers} score={score:.3f}")
    print(f"tree: {len(nodes)} agents  depths={depth_counts}\n{'='*72}\n")
    _print_tree(root)


if __name__ == "__main__":
    asyncio.run(main())
