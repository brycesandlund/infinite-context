"""Do our labeled_records training labels follow from the item text? A strong judge (Opus, may answer `unclear`,
text-only) and base Qwen (forced choice) label the SAME rows in the SAME leaf-sized batches; cross-tabulate vs gold.

    PYTHONPATH=. uv run python scripts/label_judge.py [rows_per_source]

Opus verdicts: = gold -> text supports the label (keep); `unclear` -> ambiguous / needs outside knowledge (drop);
other label -> likely mislabeled (drop). Base Qwen on the KEPT rows tells whether the filter keeps hard-but-
distinguishable items (Qwen < 100%) or only easy ones. Opus calls cached in eval_results/label_judge_cache.json.
"""
import asyncio, collections, hashlib, json, os, random, re, sys

import litellm
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer, get_text_content

import rl
from tasks.labeled.generators import _rows, _DESC

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
JUDGE = os.environ.get("LABEL_JUDGE_MODEL", "anthropic/claude-opus-4-6")
CACHE = "eval_results/label_judge_cache.json"
OUT = os.environ.get("OUT", "eval_results/label_judge_sample.json")
BATCH_TOKENS = 500

tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
renderer = get_renderer(rl.RENDERER_NAME, tok)

OPUS_SYS = ("You label items from a dataset. Use ONLY the item's own text: do not use outside knowledge of the world "
            "(e.g. whether a named person, film or place really has the stated property), and do not guess. If the "
            "text alone clearly supports one label, give it; if it is ambiguous between labels, or deciding would "
            "need outside knowledge, answer `unclear`.")


def batches(name, rows):
    labels_all = sorted({r["label"] for r in _rows(name)})
    rng = random.Random(f"batches-{name}")
    groups = collections.defaultdict(list)
    for r in rows:
        if name == "dbpedia":      # training shows 4-6 of the 14 classes per problem
            s = sorted(set(rng.sample([l for l in labels_all if l != r["label"]], 4)) | {r["label"]})
            groups[tuple(s)].append(r)
        else:
            groups[tuple(labels_all)].append(r)
    out = []
    for labels, rs in groups.items():
        cur, used = [], 0
        for r in rs:
            c = len(tok.encode(r["text"])) + 12
            if cur and used + c > BATCH_TOKENS:
                out.append((list(labels), cur)); cur, used = [], 0
            cur.append(r); used += c
        if cur: out.append((list(labels), cur))
    return out


def user_msg(labels, rs, allow_unclear):
    opt = " or `unclear`" if allow_unclear else ""
    return (f"The possible labels are: {', '.join(labels)}.\n\nLabel every line below. Reply with one line per item, "
            f"exactly `i: label` (i = the item's number) using only the listed labels{opt}, nothing else.\n\n"
            + "\n".join(f"{i + 1}. {r['text'].replace(chr(10), ' ')}" for i, r in enumerate(rs)))


def parse(out, n, labels, allow_unclear):
    ok = {l.lower() for l in labels} | ({"unclear"} if allow_unclear else set())
    got = {}
    for pos, l in enumerate(x for x in out.splitlines() if x.strip()):
        m = re.match(r"^\s*(\d+|i)\s*[:.)]\s*(.+?)\s*$", l)
        if m:
            v = m.group(2).strip().strip("`*'\".").lower()
            got[int(m.group(1)) if m.group(1).isdigit() else pos + 1] = v if v in ok else "<other>"
    return [got.get(i + 1, "<missing>") for i in range(n)]


async def main():
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    sem_o, sem_q = asyncio.Semaphore(int(os.environ.get("OPUS_CONC", "32"))), asyncio.Semaphore(32)
    done = [0]
    qwen = tinker.ServiceClient().create_sampling_client(base_model=rl.MODEL_NAME)

    async def opus(desc, labels, rs):
        u = user_msg(labels, rs, True)
        k = hashlib.sha1(json.dumps([JUDGE, OPUS_SYS, desc, u]).encode()).hexdigest()
        if k not in cache:
            async with sem_o:
                for attempt in range(5):
                    try:
                        r = await litellm.acompletion(model=JUDGE, temperature=0, max_tokens=800, messages=[
                            {"role": "system", "content": f"{OPUS_SYS}\n\n{desc}"}, {"role": "user", "content": u}])
                        cache[k] = r.choices[0].message.content or ""
                        done[0] += 1
                        if done[0] % 100 == 0:                 # save progress (a killed run resumes from cache)
                            json.dump(cache, open(CACHE, "w")); print(f"opus {done[0]} batches", flush=True)
                        break
                    except Exception as e:
                        print("opus retry", e); await asyncio.sleep(5 * (attempt + 1))
        return cache.get(k, "")

    async def base(desc, labels, rs):
        async with sem_q:
            mi = renderer.build_generation_prompt([{"role": "system", "content": desc},
                                                   {"role": "user", "content": user_msg(labels, rs, False)}])
            r = await qwen.sample_async(mi, num_samples=1, sampling_params=tinker.SamplingParams(
                temperature=0.0, max_tokens=600, stop=renderer.get_stop_sequences()))
            return get_text_content(renderer.parse_response(r.sequences[0].tokens)[0]) or ""

    jobs = []
    for name in [x for x in os.environ.get("SOURCES", ",".join(_DESC)).split(",") if x]:
        rows = random.Random(f"sample-{name}").sample(_rows(name), min(N, len(_rows(name))))
        desc = f"The document is a list of text items, ONE per line. {_DESC[name].replace('{m}', '->')}"
        jobs += [(name, desc, labels, rs) for labels, rs in batches(name, rows)]
    print(f"{len(jobs)} batches, {sum(len(j[3]) for j in jobs)} rows", flush=True)
    o_out = await asyncio.gather(*[opus(d, l, rs) for _, d, l, rs in jobs])
    json.dump(cache, open(CACHE, "w"))
    q_out = await asyncio.gather(*[base(d, l, rs) for _, d, l, rs in jobs])

    recs = []
    for (name, desc, labels, rs), oo, qo in zip(jobs, o_out, q_out):
        for r, ov, qv in zip(rs, parse(oo, len(rs), labels, True), parse(qo, len(rs), labels, False)):
            g = str(r["label"]).lower()
            recs.append({"src": name, "text": r["text"], "gold": r["label"], "labels": labels, "opus": ov, "qwen": qv,
                         "opus_cat": "agree" if ov == g else "unclear" if ov == "unclear" else "disagree",
                         "qwen_ok": qv == g})
    json.dump(recs, open(OUT, "w"), indent=1)

    print(f"\nOpus verdict vs gold  |  base-Qwen accuracy within each Opus category")
    for src in sorted({r["src"] for r in recs}):
        rr = [r for r in recs if r["src"] == src]
        cats = collections.Counter(r["opus_cat"] for r in rr)
        qa = {c: sum(r["qwen_ok"] for r in rr if r["opus_cat"] == c) for c in cats}
        line = "  ".join(f"{c} {cats[c]:3d} ({cats[c] / len(rr):4.0%}, qwen {qa[c] / cats[c]:4.0%})"
                         for c in ("agree", "unclear", "disagree") if cats[c])
        print(f"  {src:8s} n={len(rr):3d}  qwen overall {sum(r['qwen_ok'] for r in rr) / len(rr):4.0%}  |  {line}")
    rnd = random.Random(1)
    for src in sorted({r["src"] for r in recs}):
        for cat in ("unclear", "disagree"):
            ex = [r for r in recs if r["src"] == src and r["opus_cat"] == cat]
            for r in rnd.sample(ex, min(3, len(ex))):
                print(f"   [{src} {cat}] gold={r['gold']} opus={r['opus']} qwen={r['qwen']} | {r['text'][:140]}")


asyncio.run(main())
