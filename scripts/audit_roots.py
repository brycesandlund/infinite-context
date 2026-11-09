r"""Curate a question + ROOT-response audit sample from a trace dump: for every task and every distinct
question SHAPE (question text with numbers/names/quoted values normalized), print the document
description (from the system prompt), the question, and EVERY root turn with its tool calls (tool results
summarised). Subtrees are omitted — this is for checking the FORMAT the model must follow.

  uv run python scripts/audit_roots.py DUMP OUT [per_shape=1]
"""
import re, sys
from collections import OrderedDict

dump, out = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 1
text = open(dump).read()
traces = re.split(r"\n(?=#{20,}\n# task=)", text)

def shape(q):
    q = re.sub(r"`[^`]*`", "`_`", q)
    q = re.sub(r"'[^']*'", "'_'", q)
    q = re.sub(r"\b\d[\d,.]*\b", "N", q)
    q = re.sub(r"\b[A-Z][A-Z]{3,}\b", "VAR", q)                 # VAR names / K1..K4 stay distinct enough
    q = re.sub(r"\b(than|as|of|for|by|with|author =|label|value|section|month|dated) ([A-Z][a-z]+)\b", r"\1 X", q)
    return q

groups = OrderedDict()
for tr in traces:
    m = re.search(r"^# task=(\S+?)(?:\[\w+\])?\s", tr, re.M)
    if not m: continue
    task = m.group(1)
    header, _, rest = tr.partition("\n[depth=0]")
    root = rest.split("\n  ====", 1)[0]
    qm = re.search(r"^# Q: (.*)$", header, re.M)
    q = qm.group(1) if qm else "?"
    sysm = re.search(r"\[system\] (.*?)\n\[user\]", root, re.S)
    sysd = sysm.group(1) if sysm else ""
    # document description = everything between the first paragraph and "You have two tools"
    parts = sysd.split("\n\n")
    desc = "\n\n".join(p for p in parts[1:] if not p.startswith("You have two tools") and not p.startswith("When you are confident")).strip()
    # every root turn, each with its tool calls; tool RESULTS (read_chunk text, child returns) are
    # summarised to one line so the file stays about the root's own text
    body = root.split("\n[user]", 1)[1] if "\n[user]" in root else root
    turns = re.findall(r"^(\[assistant\] .*?)(?=^\[assistant\]|\Z)", body, re.S | re.M)
    def clean(t):
        out = []
        for ln in t.rstrip().split("\n"):
            if ln.startswith("[tool:read_chunk]"): out.append("[tool:read_chunk] …")
            elif ln.startswith("[tool:spawn_subagent]"): out.append(ln[:300])
            elif out and out[-1] == "[tool:read_chunk] …" and not ln.startswith("[") and not ln.startswith("  ->"): continue
            else: out.append(ln)
        return "\n".join(out)
    turns = [clean(t) for t in turns]
    key = (task, shape(q))
    groups.setdefault(key, []).append((q, desc, turns))

lines = [f"# Question + root-response audit — {sum(len(v) for v in groups.values())} traces, "
         f"{len(groups)} (task, question-shape) groups; {per} per group. Source: {dump}\n"]
for (task, sh), items in groups.items():
    for q, desc, turns in items[:per]:
        lines.append("=" * 110)
        lines.append(f"TASK {task}   ({len(items)} traces with this question shape)")
        lines.append("-" * 110)
        lines.append(f"[document description]\n{desc}\n")
        lines.append(f"[question]\n{q}\n")
        for i, t in enumerate(turns, 1):
            lines.append(f"[root turn {i}]\n{t}\n")
open(out, "w").write("\n".join(lines) + "\n")
print(f"{len(groups)} groups -> {out}")
for (task, sh), items in groups.items():
    print(f"  {task:20s} {len(items):3d}  {sh[:90]}")
