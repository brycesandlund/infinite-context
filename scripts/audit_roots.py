r"""Curate a question + ROOT-response audit sample from a trace dump: for every task and every distinct
question SHAPE (question text with numbers/names/quoted values normalized), print the document
description (from the system prompt), the question, the root's first turn (preamble + first action) and
the root's final turn. Subtrees are omitted — this is for checking the FORMAT the model must follow.

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
    turns = re.findall(r"^\[assistant\] (.*?)(?=^\[assistant\]|^\[tool:|^\[user\]|\Z)", root, re.S | re.M)
    first = turns[0].strip() if turns else "?"
    last = turns[-1].strip() if len(turns) > 1 else ""
    key = (task, shape(q))
    groups.setdefault(key, []).append((q, desc, first, last))

lines = [f"# Question + root-response audit — {sum(len(v) for v in groups.values())} traces, "
         f"{len(groups)} (task, question-shape) groups; {per} per group. Source: {dump}\n"]
for (task, sh), items in groups.items():
    for q, desc, first, last in items[:per]:
        lines.append("=" * 110)
        lines.append(f"TASK {task}   ({len(items)} traces with this question shape)")
        lines.append("-" * 110)
        lines.append(f"[document description]\n{desc}\n")
        lines.append(f"[question]\n{q}\n")
        lines.append(f"[root turn 1]\n{first}\n")
        if last:
            lines.append(f"[root final turn]\n{last}\n")
open(out, "w").write("\n".join(lines) + "\n")
print(f"{len(groups)} groups -> {out}")
for (task, sh), items in groups.items():
    print(f"  {task:20s} {len(items):3d}  {sh[:90]}")
