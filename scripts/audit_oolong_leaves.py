"""Leaf-level audit of 16w's OOLONG-synth rollouts: regenerate each problem, compute the TRUE tally for every leaf's
owned range in the key space the root chose, and categorize every imperfect rollout."""
import json, re, collections, sys
from datetime import datetime
import rl
from tinker_cookbook import tokenizer_utils
from tasks.oolong import make_oolong_problem

tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
RUNS = [("4K/3K", "eval_results/raw/sft_general16w.jsonl", 4000), ("8K/5K", "eval_results/raw/sft_general16w_long_8k.jsonl", 8000),
        ("16K/5K", "eval_results/raw/sft_general16w_long_16k.jsonl", 16000), ("32K/5K", "eval_results/raw/sft_general16w_long_32k.jsonl", 32000),
        ("32K/8K", "eval_results/raw/sft_general16w_long_32k_b8k.jsonl", 32000)]
MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FULLMON = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def pdate(s):
    for f in ("%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%B %d, %Y"):
        try: return datetime.strptime(s.strip(), f)
        except Exception: pass
    return None


def walk(n):
    yield n
    for c in n.get("children", []): yield from walk(c)


def boxed(t):
    m = re.findall(r"\\boxed\{(.*)\}", t or "", re.S)
    return m[-1].strip() if m else None


def parse_state(s):
    if s is None: return None
    s = s.strip()
    if re.fullmatch(r"-?\d+", s): return int(s)
    if s.lower() in ("none", ""): return {}
    d = {}
    for part in s.split("|"):
        m = re.match(r"\s*(.+?)\s*[:=]\s*(-?\d+)\s*$", part)
        if not m: return None
        d[m.group(1).strip().strip("'\"`")] = d.get(m.group(1).strip().strip("'\"`"), 0) + int(m.group(2))
    return d


def part_kind(p, labels):
    pl = p.lower().strip("'\" ")
    if pl in labels: return "label"
    if re.fullmatch(r"(user\s*)?\d{3,}", pl): return "user"
    if re.fullmatch(r"[a-z]{3,9}\.? \d{1,2},? \d{4}", pl) or re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", pl) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", pl): return "date"
    if re.fullmatch(r"[a-z]{3,9}\.? \d{4}", pl) or re.fullmatch(r"\d{4}-\d{2}", pl): return "month"
    return None


def norm(kind, v, labels):
    v = v.lower().strip("'\" ")
    if kind == "user": return re.sub(r"\D", "", v)
    if kind == "month":
        m = re.match(r"([a-z]+)\.? (\d{4})", v)
        if m: return f"{m.group(1)[:3]} {m.group(2)}"
        m = re.match(r"(\d{4})-(\d{2})", v); return f"{MON[int(m.group(2)) - 1]} {m.group(1)}" if m else v
    if kind == "date":
        d = pdate(v.title()) or pdate(v)
        return d.strftime("%Y-%m-%d") if d else v
    return v


def gold_key(kind, rec):
    _, _, y, u, dt = rec
    d = pdate(dt)
    if kind == "label": return str(y).lower()
    if kind == "user": return str(u)
    if kind == "month": return f"{d.strftime('%b').lower()} {d.year}" if d else None
    if kind == "date": return d.strftime("%Y-%m-%d") if d else None


def qfilter(q):
    """Return a predicate over records from the question's subset prefix (user IDs / month / date range)."""
    preds = []
    m = re.search(r"user IDs ([\d,\s]+(?:and\s+\d+)?)", q)
    if m:
        ids = set(re.findall(r"\d+", m.group(1))); preds.append(lambda r, ids=ids: str(r[3]) in ids)
    m = re.search(r"occur in (\w+) of any year", q)
    if m and m.group(1).lower() in FULLMON:
        mi = FULLMON.index(m.group(1).lower()) + 1; preds.append(lambda r, mi=mi: (pdate(r[4]) or datetime(1, 1, 1)).month == mi)
    m = re.search(r"between (.+?) and (.+?), inclusive", q)
    if m:
        a, b = pdate(m.group(1)), pdate(m.group(2))
        if a and b: preds.append(lambda r, a=a, b=b: a <= (pdate(r[4]) or datetime(1, 1, 1)) <= b)
    return lambda r: all(p(r) for p in preds)


def leaf_range(node):
    m = re.search(r"tokens (\d+)\.\.(\d+)", node["messages"][1]["content"] if len(node["messages"]) > 1 else "")
    return (int(m.group(1)), int(m.group(2))) if m else None


cats = collections.Counter(); per_run = collections.defaultdict(collections.Counter); examples = collections.defaultdict(list)
leaf_err = collections.Counter(); per_ds = collections.defaultdict(collections.Counter); lost = collections.Counter()
for tag, path, doc in RUNS:
    for r in map(json.loads, open(path)):
        if not r["task"].startswith("oolong"): continue
        if r["score"] >= 0.999: per_run[tag]["perfect"] += 1; continue
        p = make_oolong_problem(r["task"], None, tok, doc, r["seed"], dataset=r["dataset"])
        spans = p.metadata["example_spans"]; labels = {str(x).lower() for x in p.metadata["true_counts"]}
        q = r["question"]; keep = qfilter(q)
        ml = re.search(r"label '([^']+)'", q); qlabel = ml.group(1).lower() if ml else None
        named = {x.lower() for x in re.findall(r"'([^']+)'", q)} & labels     # labels the question names
        root = r["tree"]
        cat = None
        if r["root_termination"] == "overflow": cat = "overflow"
        elif r["answer"] is None: cat = "no boxed answer"
        # leaf audit
        leaves = [n for n in walk(root) if not n.get("children") and any(m.get("name") == "read_chunk" for m in n["messages"] if m["role"] == "tool")]
        tot_abs = tot_gold = n_ok = n_bad = 0; kinds_seen = set(); overflow_leaves = 0
        for lf in leaves:
            if lf["termination"] == "overflow": overflow_leaves += 1; continue
            rg = leaf_range(lf); st = parse_state(boxed(lf["messages"][-1]["content"]))
            if rg is None or st is None: continue
            recs = [s for s in spans if rg[0] <= s[0] < rg[1] and keep(s)]
            if isinstance(st, int):
                if qlabel is None: continue
                g = sum(1 for s in recs if str(s[2]).lower() == qlabel); kinds_seen.add("int")
                tot_abs += abs(st - g); tot_gold += g; n_ok += st == g; n_bad += st != g; continue
            keys = list(st)
            if not keys:
                # an empty partial is right iff no relevant item is owned (relevant = in the question's filter, and of a
                # named label when the question names labels)
                g = sum(1 for s in recs if not named or str(s[2]).lower() in named)
                tot_gold += g; tot_abs += g; n_ok += g == 0; n_bad += g != 0; continue
            parts0 = keys[0].split("/")
            ks = tuple(part_kind(x, labels) for x in parts0)
            if None in ks: kinds_seen.add("unrecognized:" + keys[0][:25]); continue
            kinds_seen.add("/".join(ks))
            if any(len(key.split("/")) != len(ks) for key in st): kinds_seen.add("mixed key shapes"); continue
            model = collections.Counter({tuple(norm(k, x, labels) for k, x in zip(ks, key.split("/"))): v for key, v in st.items()})
            gold = collections.Counter(tuple(gold_key(k, s) for k in ks) for s in recs)
            if "label" in ks and named:
                li = ks.index("label")
                if all(k[li] in named for k in model):     # the leaf tallied only the named labels -> so does gold
                    gold = collections.Counter({k: v for k, v in gold.items() if k[li] in named})
            err = sum(abs(model[k] - gold[k]) for k in set(model) | set(gold))
            tot_abs += err; tot_gold += sum(gold.values()); n_ok += err == 0; n_bad += err != 0
        leaf_err["records"] += tot_gold; leaf_err["abs_err"] += tot_abs; leaf_err["leaves_ok"] += n_ok; leaf_err["leaves_bad"] += n_bad
        if cat is None:
            if overflow_leaves: cat = "subagent overflow (partial lost)"
            elif n_ok + n_bad == 0: cat = "unauditable state: " + ",".join(sorted(kinds_seen))[:40]
            elif n_bad == 0: cat = "leaves all correct -> merge/resolution/format error"
            else:
                rate = tot_abs / max(1, tot_gold)
                cat = f"leaf tally errors ({'<10%' if rate < .1 else '10-30%' if rate < .3 else '>30%'} of owned items)"
        cats[cat] += 1; per_run[tag][cat] += 1; per_ds[r["dataset"]][cat.split(" (")[0]] += 1; lost[cat.split(" (")[0]] += 1 - r["score"]
        if len(examples[cat]) < 3: examples[cat].append((tag, r["task"], r["dataset"], r["seed"], round(r["score"], 2), q[-160:].replace("\n", " ")))
print("IMPERFECT OOLONG ROLLOUTS BY CATEGORY (all 16w runs)")
for k, v in cats.most_common(): print(f"  {v:3d}  {k}")
print("\nscore lost per category (sum of 1-score):", {k: round(v, 1) for k, v in lost.most_common()})
print("\nper dataset:")
for d, c in sorted(per_ds.items()): print(f"  {d:12s} {dict(c)}")
print("\nper run:")
for tag, c in per_run.items(): print(f"  {tag:7s} {dict(c)}")
print(f"\nleaf audit over all imperfect rollouts: {leaf_err['leaves_ok']} leaves exact, {leaf_err['leaves_bad']} with errors; "
      f"sum |model-gold| = {leaf_err['abs_err']} over {leaf_err['records']} owned relevant items")
json.dump({k: v for k, v in examples.items()}, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/oolong_audit_examples.json", "w"), indent=1)
