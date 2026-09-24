"""One-time preparation of OOLONG-real (oolongbench/oolong-real, config `dnd`) for our harness.

Input:  ~/.cache/infinite-context/oolong_real/dnd/{test,validation}.jsonl  (HF snapshot, ~9.3 GB each; every
        question row repeats its full context window text)
Output: ~/.cache/infinite-context/oolong_real/prepared/<split>/
          windows/<context_window_id>.npy   document tokens (Qwen tokenizer, int32) — stored ONCE per window
          windows/<context_window_id>.instr.txt   instruction paragraph + player->character mapping (-> task_context)
          index.jsonl   one row per question: id, context_window_id, question, answer, question_type,
                        episodes, campaign, doc_tokens

Split of each context_window_text (verbatim, nothing rewritten): everything before the first "[START OF EPISODE]"
— the instruction paragraph AND the player->character mapping block ("…Use this mapping when answering the questions
below.") — becomes the task_context, i.e. the system prompt every agent sees. The official harness sends the whole
window as the system message; the mapping is task-level context (who plays whom), not data to aggregate, so like our
OOLONG-synth adapter's description it goes where every subagent can use it. The transcripts ([START OF EPISODE]…
[END OF EPISODE] blocks) are the document. A window without the markers aborts the run (never silently mis-split).

    PYTHONPATH=. uv run python scripts/prepare_oolong_real.py [split ...]
"""
import json
import os
import sys

import numpy as np
from tinker_cookbook import tokenizer_utils

import rl

ROOT = os.path.expanduser("~/.cache/infinite-context/oolong_real")
MARKER = "The following lines contain the mapping between player names and character names."
EPISODE = "[START OF EPISODE]"


def main(splits):
    tok = tokenizer_utils.get_tokenizer(rl.MODEL_NAME)
    for split in splits:
        out = os.path.join(ROOT, "prepared", split)
        os.makedirs(os.path.join(out, "windows"), exist_ok=True)
        seen: dict[str, int] = {}
        n = 0
        with open(os.path.join(ROOT, "dnd", f"{split}.jsonl")) as f, \
                open(os.path.join(out, "index.jsonl"), "w") as idx:
            for line in f:
                r = json.loads(line)
                cw = r["context_window_id"]
                if cw not in seen:
                    text = r["context_window_text"]
                    m = text.find(MARKER)
                    i = text.find(EPISODE, m) if m >= 0 else -1
                    if m < 0 or i < 0:
                        raise SystemExit(f"{split} window {cw}: mapping/episode marker not found — refusing to guess a split")
                    instr, doc = text[:i].strip(), text[i:]
                    ids = tok.encode(doc, add_special_tokens=False)
                    np.save(os.path.join(out, "windows", f"{cw}.npy"), np.asarray(ids, dtype=np.int32))
                    with open(os.path.join(out, "windows", f"{cw}.instr.txt"), "w") as g:
                        g.write(instr)
                    seen[cw] = len(ids)
                    print(f"  {split} window {len(seen):3d}: {len(r['episodes']):2d} episode(s), "
                          f"{len(text):>8d} chars -> {len(ids):>8d} doc tokens", flush=True)
                idx.write(json.dumps({
                    "id": r["id"], "context_window_id": cw, "question": r["question"], "answer": r["answer"],
                    "question_type": r["question_type"], "episodes": r["episodes"], "campaign": r["campaign"],
                    "doc_tokens": seen[cw],
                }) + "\n")
                n += 1
        L = sorted(seen.values())
        print(f"{split}: {n} questions, {len(seen)} windows, doc tokens min/median/max {L[0]}/{L[len(L) // 2]}/{L[-1]}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["test", "validation"])
