"""Sample ONLY the root's first turn (preamble + spawned subtask) on many OOLONG-user questions and count broken
contracts — cheap (one sample per question, no tree). Compares checkpoints on the failure mode seen in 19w @10K.

    CKPTS="18w=tinker://...,19w=tinker://..." PYTHONPATH=. uv run python scripts/root_contract_probe.py [n] [doc_tokens]
Defects: closed user list (`User 1`…), label mention / "judging" on a LABEL-FREE question, a backticked field that is
not in the data (only `User:`/`Date:`/`Instance:` exist), a 5-6 digit user ID in the subtask that the question lacks,
or no spawn at all.
"""
import asyncio, collections, json, os, re, sys

import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

import rl
import harness
from eval.backends import TinkerBackend
from tasks.oolong.generators import make_oolong_problem, oolong_spec

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DOC = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
BUDGET = 8000
tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
LABELS = r"true|false|positive|negative|formal|informal|spam|ham|correct|incorrect|entailment|neutral|contradiction"


def defects(q, sub, labelfree):
    head = re.sub(r"tokens \d+\.\.\d+", "", sub.split("Recursively")[0])
    out = []
    if re.search(r"one of `?User \d", sub): out.append("closed user list")
    if labelfree and (re.search(r"judging|labell?ing|marked as", head, re.I) or re.search(rf"\b({LABELS})\b", head, re.I)):
        out.append("label/judging on label-free question")
    for f in re.findall(r"`([A-Za-z_ ]+)[:=]`", sub):
        if f.strip().lower() not in ("user", "date", "instance"): out.append(f"non-existent field `{f}:`")
    qids = set(re.findall(r"\b\d{5,6}\b", q)); sids = set(re.findall(r"\b\d{5,6}\b", head))
    if sids - qids: out.append("user ID not in question")
    return out


async def main():
    ckpts = dict(kv.split("=", 1) for kv in os.environ["CKPTS"].split(","))
    probs = []
    for i in range(N):
        seed, ds = oolong_spec("oolong_user", i, 5_000_000)
        p = make_oolong_problem("oolong_user", None, tok, DOC, seed, dataset=ds)
        q = p.question.split("\n")[0]
        labelfree = not re.search(r"label", q, re.I)
        sys_ = harness.make_system_prompt(doc_length=len(p.document_tokens), context_budget=BUDGET, task_context=p.task_context)
        probs.append((ds, q, labelfree, [{"role": "system", "content": sys_}, {"role": "user", "content": p.question}]))
    sc = tinker.ServiceClient(); renderer = get_renderer(rl.RENDERER_NAME, tok)
    res = {}
    for name, ck in ckpts.items():
        tc = await sc.create_lora_training_client_async(base_model=rl.MODEL_NAME, rank=rl.LORA_RANK)
        await (await tc.load_state_async(ck)).result_async()
        be = TinkerBackend(await tc.save_weights_and_get_sampling_client_async(), tok, renderer, temperature=0.2)
        sem = asyncio.Semaphore(32)
        async def one(m):
            async with sem:
                return await be.sample(m, max_tokens=1500)
        turns = await asyncio.gather(*[one(m) for *_, m in probs])
        rows = []
        for (ds, q, lf, _), t in zip(probs, turns):
            sub = next((c.arguments.get("subtask", "") for c in t.tool_calls if c.name == "spawn_subagent"), None)
            d = ["no spawn"] if sub is None else defects(q, sub, lf)
            rows.append({"ds": ds, "q": q, "labelfree": lf, "subtask": sub, "defects": d})
        res[name] = rows
        C = collections.Counter(); byk = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            k = "label-free" if r["labelfree"] else "label-conditioned"; byk[k][1] += 1; byk[k][0] += bool(r["defects"])
            for x in r["defects"]: C[re.sub(r"`.*`", "`…`", x)] += 1
        tot = sum(bool(r["defects"]) for r in rows)
        print(f"{name}: defective root contracts {tot}/{len(rows)} ({tot / len(rows):.1%}) | "
              + " | ".join(f"{k} {v[0]}/{v[1]}" for k, v in byk.items()) + f" | {dict(C)}", flush=True)
    json.dump(res, open(f"eval_results/root_contract_probe_{DOC // 1000}k.json", "w"), indent=1)


asyncio.run(main())
