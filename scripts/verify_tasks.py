"""Sweep tasks with their OWN strategy (strategy is a task property): every oracle rollout must
grade 1.0 against gold, and the REAL per-agent prompt (renderer + tool schema) must stay under
the SFT budget. Dumps DUMP_PER traces per (task, variant) for reading.

  TASKS=synth_2d,long_records N=20 DOCS=6000,14000 DUMP_PER=1 OUT=/tmp/verify.txt \
      uv run python scripts/verify_tasks.py
"""
import asyncio, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rl, sft
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from eval.agent import flatten
from eval.backends import neutral_to_cookbook
from eval.run import _rollout_header, _tree_to_text
from oracle import make_oracle
from tasks import load_pg_essays_text, make_problem, grade_answer, resolve_eval_grading_mode

N = int(os.environ.get("N", "20"))
DOCS = [int(x) for x in os.environ.get("DOCS", "6000,14000").split(",")]
OUT = os.environ.get("OUT", "/tmp/verify_traces.txt")
DUMP_PER = int(os.environ.get("DUMP_PER", "1"))
SEED0 = int(os.environ.get("SEED0", "900000"))
BUDGET = sft.AGENT_CONTEXT


async def main():
    tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
    renderer = get_renderer(rl.RENDERER_NAME, tok)
    specs = [rl.ReadChunkTool.read_chunk.to_spec(), rl.SubagentTool.spawn_subagent.to_spec()]
    corpus = tok.encode(load_pg_essays_text(), add_special_tokens=False)
    dumped = defaultdict(int)
    with open(OUT, "w") as tf:
        for task in os.environ["TASKS"].split(","):
            for doc in DOCS:
                scores, maxlen, variants, strats, worst = [], 0, Counter(), Counter(), None
                for i in range(N):
                    seed = SEED0 + i
                    prob = make_problem(task, corpus, tok, doc, seed)
                    orc = make_oracle(prob, tok, budget=BUDGET, max_chunk_tokens=sft.MAX_CHUNK_TOKENS)
                    node = await sft._one_trace(orc, prob, tok)
                    sc = grade_answer(node.answer, prob.gold_answers, resolve_eval_grading_mode(prob))
                    scores.append(sc)
                    v = prob.metadata.get("qtype") or prob.metadata.get("mode") or "-"
                    if task == "rule_label":
                        v += "/" + prob.metadata["rule"]
                    variants[v] += 1
                    strats[getattr(orc, "strategy", "?")] += 1
                    for ag in flatten(node):
                        cb = neutral_to_cookbook(ag.messages, renderer, specs)
                        L = renderer.build_generation_prompt(cb).length
                        if L > maxlen:
                            maxlen, worst = L, (v, seed)
                    if sc < 1.0:
                        print(f"  FAIL {task} doc={doc} seed={seed} {v} gold={prob.gold_answers} ans={node.answer!r}", flush=True)
                    key = (task, v)
                    if doc == DOCS[0] and dumped[key] < DUMP_PER:
                        dumped[key] += 1
                        tf.write(_rollout_header(f"{task}[{getattr(orc, 'strategy', '?')}]", seed, None, v, prob.question,
                                                 prob.gold_answers, node.answer, node.termination, sc))
                        tf.write(_tree_to_text(node))
                print(f"{task:20s} doc={doc:6d} pass={sum(s == 1.0 for s in scores)}/{N} max_ctx={maxlen} "
                      f"(worst {worst}) strat={dict(strats)} variants={dict(variants)}", flush=True)
    print("traces ->", OUT)


asyncio.run(main())
