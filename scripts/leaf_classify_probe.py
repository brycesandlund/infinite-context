"""Per-item classification accuracy on OOLONG-synth items, in LEAF-SIZED batches (<=500 tokens of lines), for the BASE
model (or a checkpoint via CKPT=tinker://...). Isolates label judgment from decomposition/tallying.

    PYTHONPATH=. uv run python scripts/leaf_classify_probe.py [n_problems_per_dataset]

Items come from fresh-seed OOLONG problems (base 7_000_000, never used for train or eval). The prompt is the problem's own
task_context (OOLONG's dataset description) + the label set; the model labels every numbered line as `i: label`.
Thinking disabled (same renderer as our agents). Temperature 0.
"""
import asyncio, collections, json, os, re, sys

import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer, get_text_content

import rl
from tasks.oolong.generators import make_oolong_problem, _DATASETS
from datasets_loader import SUPPORTED_DATASETS  # vendored (path set by tasks.oolong.generators)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CKPT = os.environ.get("CKPT")
TAG = os.environ.get("TAG", "base" if not CKPT else "ckpt")
LEAF_TOKENS = 500
BASE = 7_000_000

tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
renderer = get_renderer(rl.RENDERER_NAME, tok)


def batches(p):
    """Contiguous runs of items whose lines fit in LEAF_TOKENS, like a leaf's owned range."""
    out, cur, start = [], [], None
    for s in p.metadata["example_spans"]:
        if cur and s[1] - start > LEAF_TOKENS:
            out.append(cur); cur = []
        if not cur: start = s[0]
        cur.append(s)
    if cur: out.append(cur)
    return out


def prompt(p, labels, items):
    lines = [re.sub(r"\s+", " ", tok.decode(p.document_tokens[a:b])).strip() for a, b, *_ in items]
    lines = [l.split("Instance:", 1)[-1].strip() for l in lines]
    sysm = (p.task_context.split("\n\nYou will be asked")[0].strip() +
            f"\n\nThe possible labels are: {', '.join(labels)}.")
    user = ("Classify every line below. Reply with one line per item, exactly `i: label`, using only the listed labels, "
            "nothing else.\n\n" + "\n".join(f"{i + 1}. {l}" for i, l in enumerate(lines)))
    return [{"role": "system", "content": sysm}, {"role": "user", "content": user}]


async def main():
    sc = tinker.ServiceClient()
    if CKPT:
        tc = await sc.create_lora_training_client_async(base_model=rl.MODEL_NAME, rank=rl.LORA_RANK)
        await (await tc.load_state_async(CKPT)).result_async()
        client = await tc.save_weights_and_get_sampling_client_async()
    else:
        client = sc.create_sampling_client(base_model=rl.MODEL_NAME)
    sem = asyncio.Semaphore(32)

    async def run(msgs):
        async with sem:
            mi = renderer.build_generation_prompt(msgs)
            r = await client.sample_async(mi, num_samples=1, sampling_params=tinker.SamplingParams(
                temperature=0.0, max_tokens=600, stop=renderer.get_stop_sequences()))
            return get_text_content(renderer.parse_response(r.sequences[0].tokens)[0]) or ""

    jobs = []
    for di, ds in enumerate(_DATASETS):
        labels = [str(x) for x in SUPPORTED_DATASETS[ds]().label_list]
        for k in range(N):
            p = make_oolong_problem("oolong_counting", None, tok, 4000, BASE + 1000 * di + k, dataset=ds)
            for items in batches(p):
                jobs.append((ds, labels, items, prompt(p, labels, items)))
    print(f"{len(jobs)} leaf batches, {sum(len(j[2]) for j in jobs)} items", flush=True)
    outs = await asyncio.gather(*[run(j[3]) for j in jobs])

    acc = collections.defaultdict(lambda: [0, 0, 0]); conf = collections.defaultdict(collections.Counter); dump = []
    for (ds, labels, items, msgs), out in zip(jobs, outs):
        got = {}
        for m in re.finditer(r"^\s*(\d+)\s*[:.)]\s*(.+?)\s*$", out, re.M):
            got[int(m.group(1))] = m.group(2).strip().strip("`*'\"").lower()
        lab = {l.lower(): l for l in labels}
        for i, s in enumerate(items, 1):
            g = str(s[2]).lower(); pr = got.get(i)
            pr = pr if pr in lab else ("<unparsed>" if pr is None else "<other>")
            a = acc[ds]; a[0] += pr == g; a[1] += 1; a[2] += pr.startswith("<")
            conf[ds][(g, pr)] += 1
        dump.append({"ds": ds, "prompt": msgs, "out": out, "gold": [s[2] for s in items]})
    print(f"\n== {TAG}: per-item accuracy in leaf-sized batches (N={N} problems/dataset @4K)")
    tot = [0, 0]
    for ds in sorted(acc, key=lambda d: acc[d][0] / acc[d][1]):
        c, n, bad = acc[ds]; tot[0] += c; tot[1] += n
        maj = max(collections.Counter(g for (g, _), v in conf[ds].items() for _ in range(v)).values()) / n
        print(f"  {ds:12s} {c / n:6.1%}  (n={n}, majority-class {maj:.0%}, unparsed/other {bad})")
        for (g, pr), v in sorted(conf[ds].items(), key=lambda kv: -kv[1])[:6]:
            if g != pr: print(f"        gold {g!r:24s} -> {pr!r}: {v}")
    print(f"  {'OVERALL':12s} {tot[0] / tot[1]:6.1%}")
    json.dump(dump, open(f"eval_results/leaf_classify_{TAG}.json", "w"), indent=1)


asyncio.run(main())
