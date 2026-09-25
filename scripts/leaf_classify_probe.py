"""Per-item classification accuracy in LEAF-SIZED batches, isolated from decomposition/tallying.

    PYTHONPATH=. uv run python scripts/leaf_classify_probe.py [n]

Env:
  CKPT=tinker://...   sample a checkpoint instead of the base model
  TAG=name            output tag (eval_results/leaf_classify_{TAG}.json)
  SOURCE=oolong       OOLONG-synth items from fresh-seed problems (base 7_000_000, never train/eval); n = problems/dataset
  SOURCE=labeled      OUR labeled_records training rows (dbpedia/emotion/yelp/claims) with their gold labels; n = items/source
                      -> does the gold we train on follow from the text? (agreement of an untrained judge with it)
  MODE=plain          reply `i: label`
  MODE=reason         reply `i: <a few words on what decides it> -> label`  (does a per-item reason help judgment?)

Prompt = the dataset description + the label set; the model labels every numbered line. Thinking disabled (the
same renderer our agents use). Temperature 0.
"""
import asyncio, collections, json, os, random, re, sys

import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer, get_text_content

import rl

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CKPT = os.environ.get("CKPT")
SOURCE = os.environ.get("SOURCE", "oolong")
MODE = os.environ.get("MODE", "plain")
TAG = os.environ.get("TAG", ("base" if not CKPT else "ckpt") + ("" if SOURCE == "oolong" else f"_{SOURCE}")
                     + ("" if MODE == "plain" else f"_{MODE}"))
LEAF_TOKENS = 500

tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
renderer = get_renderer(rl.RENDERER_NAME, tok)

INSTR = {
    "plain": "Classify every line below. Reply with one line per item, exactly `i: label` (i = the item's number), "
             "using only the listed labels, nothing else.",
    "reason": "Classify every line below. Reply with one line per item, exactly `i: <a few words on what in the item "
              "decides its label> -> label` (i = the item's number), using only the listed labels for the label, "
              "nothing else.",
}[MODE]


def _msgs(desc, labels, texts):
    return [{"role": "system", "content": f"{desc}\n\nThe possible labels are: {', '.join(labels)}."},
            {"role": "user", "content": INSTR + "\n\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))}]


def _pack(items, cost):
    """Contiguous batches whose summed token cost stays under LEAF_TOKENS (a leaf's owned range)."""
    out, cur, used = [], [], 0
    for it in items:
        c = cost(it)
        if cur and used + c > LEAF_TOKENS:
            out.append(cur); cur, used = [], 0
        cur.append(it); used += c
    if cur: out.append(cur)
    return out


def oolong_jobs():
    from tasks.oolong.generators import make_oolong_problem, _DATASETS
    from datasets_loader import SUPPORTED_DATASETS  # vendored (path set by tasks.oolong.generators)
    jobs = []
    for di, ds in enumerate(_DATASETS):
        labels = [str(x) for x in SUPPORTED_DATASETS[ds]().label_list]
        for k in range(N):
            p = make_oolong_problem("oolong_counting", None, tok, 4000, 7_000_000 + 1000 * di + k, dataset=ds)
            desc = p.task_context.split("\n\nYou will be asked")[0].strip()
            items = [(re.sub(r"\s+", " ", tok.decode(p.document_tokens[a:b])).split("Instance:", 1)[-1].strip(),
                      str(y), b - a) for a, b, y, *_ in p.metadata["example_spans"]]
            for batch in _pack(items, lambda it: it[2]):
                jobs.append((ds, labels, [t for t, _, _ in batch], [y for _, y, _ in batch], desc))
    return jobs


def labeled_jobs():
    from tasks.labeled.generators import _rows, _DESC
    jobs, rng = [], random.Random(0)
    for name in _DESC:
        rows = _rows(name)
        all_labels = sorted({r["label"] for r in rows})
        picked = rng.sample(rows, min(N, len(rows)))
        if name == "dbpedia":          # training shows 4-6 of the 14 classes per problem; group items by a class subset
            groups = collections.defaultdict(list)
            subsets = [sorted(rng.sample(all_labels, 5)) for _ in range(max(1, N // 40))]
            for r in picked:
                s = next((s for s in subsets if r["label"] in s), None)
                if s is None:
                    s = subsets[rng.randrange(len(subsets))]; s = sorted(set(s[:4]) | {r["label"]})
                    subsets.append(s)
                groups[tuple(s)].append(r)
            parts = list(groups.items())
        else:
            parts = [(tuple(all_labels), picked)]
        desc = f"The document is a list of text items, ONE per line. {_DESC[name]}"
        for labels, rs in parts:
            items = [(r["text"].replace("\n", " "), str(r["label"]), len(tok.encode(r["text"])) + 12) for r in rs]
            for batch in _pack(items, lambda it: it[2]):
                jobs.append((name, list(labels), [t for t, _, _ in batch], [y for _, y, _ in batch], desc))
    return jobs


def parse(out, n):
    """-> list of n predicted labels (lowercased) or None. Numbered lines by number; otherwise by position."""
    rows = [l for l in out.splitlines() if l.strip()]
    got = {}
    for pos, l in enumerate(rows):
        m = re.match(r"^\s*(\d+|i)\s*[:.)]\s*(.+?)\s*$", l)
        if not m: continue
        idx = int(m.group(1)) if m.group(1).isdigit() else pos + 1
        body = m.group(2)
        if MODE == "reason":
            parts = re.split(r"\s*(?:->|→)\s*", body)
            body = parts[-1]
        got[idx] = body.strip().strip("`*'\".").lower()
    return [got.get(i + 1) for i in range(n)]


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
                temperature=0.0, max_tokens=900 if MODE == "reason" else 600, stop=renderer.get_stop_sequences()))
            return get_text_content(renderer.parse_response(r.sequences[0].tokens)[0]) or ""

    jobs = oolong_jobs() if SOURCE == "oolong" else labeled_jobs()
    print(f"{len(jobs)} leaf batches, {sum(len(j[2]) for j in jobs)} items ({SOURCE}, {MODE}, {TAG})", flush=True)
    outs = await asyncio.gather(*[run(_msgs(desc, labels, texts)) for _, labels, texts, _, desc in jobs])

    acc = collections.defaultdict(lambda: [0, 0, 0]); conf = collections.defaultdict(collections.Counter); dump = []
    for (ds, labels, texts, gold, desc), out in zip(jobs, outs):
        preds = parse(out, len(gold))
        lab = {l.lower() for l in labels}
        for g, pr in zip(gold, preds):
            g = g.lower(); pr = pr if pr in lab else ("<unparsed>" if pr is None else "<other>")
            a = acc[ds]; a[0] += pr == g; a[1] += 1; a[2] += pr.startswith("<")
            conf[ds][(g, pr)] += 1
        dump.append({"ds": ds, "labels": labels, "texts": texts, "gold": gold, "out": out})
    print(f"\n== {TAG}: per-item accuracy in leaf-sized batches")
    tot = [0, 0]
    for ds in sorted(acc, key=lambda d: acc[d][0] / acc[d][1]):
        c, n, bad = acc[ds]; tot[0] += c; tot[1] += n
        print(f"  {ds:12s} {c / n:6.1%}  (n={n}, unparsed/other {bad})")
        for (g, pr), v in sorted(conf[ds].items(), key=lambda kv: -kv[1])[:4]:
            if g != pr: print(f"        gold {g!r:24s} -> {pr!r}: {v}")
    print(f"  {'OVERALL':12s} {tot[0] / tot[1]:6.1%}")
    json.dump(dump, open(f"eval_results/leaf_classify_{TAG}.json", "w"), indent=1)


asyncio.run(main())
