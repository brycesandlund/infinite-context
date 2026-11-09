r"""Pull COMPLETE depth-0 (root) episodes out of a trace dump — system prompt, question, the root's
turns, tool calls and tool results, final answer — one per requested (task[, qtype]) so the root
behaviour can be read in context without the subtree.

  uv run python scripts/extract_roots.py DUMP OUT task[:regex][:n] ...
`regex` is searched over the trace's header + root text (question, preamble); e.g.
  ... sft9_dry2_traces.txt trace_snippets/root_turns_v9.txt synth_peak "vt_novel:document is 1[45]\d\d\d tokens" "labeled_records:Which section has the MOST:2"
"""
import re, sys

dump, out, *wants = sys.argv[1:]
text = open(dump).read()
# a trace = header block + tree; the header line starts with "# task="
traces = re.split(r"\n(?=#{20,}\n# task=)", text)
picked = []
for w in wants:
    parts = w.split(":"); task = parts[0]
    rx = re.compile(parts[1]) if len(parts) > 1 and parts[1] else None
    n = int(parts[2]) if len(parts) > 2 else 1
    got = 0
    for tr in traces:
        m = re.search(r"^# task=(\S+?)(?:\[\w+\])? .*?qtype=(\S+)", tr, re.M)
        if not m or m.group(1) != task: continue
        header = tr.split("\n[depth=0]")[0]
        root = tr.split("\n[depth=0]", 1)[1]
        root = root.split("\n  ====", 1)[0]       # subtree (indented, depth>=1) starts here
        if rx and not rx.search(header + root): continue
        picked.append(header.rstrip() + "\n[depth=0]" + root.rstrip() + "\n")
        got += 1
        if got >= n: break
    if got < n: print(f"WARNING: only {got}/{n} for {w}", file=sys.stderr)
open(out, "w").write(("\n\n" + "=" * 100 + "\n\n").join(picked) + "\n")
print(f"{len(picked)} roots -> {out}")
