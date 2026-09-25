"""LLM equivalence grader for OPEN-ENDED QA (RULER qa_1 / qa_2, narrativeqa) — reported ALONGSIDE the string graders.

Why: RULER's string_match_part needs the whole gold string inside the prediction, so correct answers in another surface
form score 0: `4` vs "four", `Caligula` vs "Gaius Julius Caesar Augustus Germanicus", `Bigg Boss 10` vs "the tenth
season", `35` vs "35 people". It also credits hedges ("KPMG and Ernst & Young" for "Ernst & Young"). The string score
stays the benchmark-comparable number; this grade is the semantic one, with its calibration reported.

Verdict is binary: the candidate gives the same entity / value / fact as ANY reference answer.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re

JUDGE_MODEL = os.environ.get("QA_JUDGE_MODEL", "anthropic/claude-opus-4-6")

SYSTEM = (
    "You grade short answers to reading-comprehension questions. You get the QUESTION, one or more REFERENCE answers "
    "(matching any one of them is enough), and a CANDIDATE answer.\n"
    "The candidate is CORRECT if it gives the same entity, value, or fact as a reference, even in a different surface "
    "form: an alias or the full vs. short name of the same person or thing; digits vs. number words or ordinals "
    "(4 = four, 10th = tenth); a unit left out when the question already implies it; minor spelling variants; or extra "
    "detail that is consistent with the reference.\n"
    "The candidate is INCORRECT if it names a different entity or value, hedges between several possible answers, "
    "gives only something related (e.g. the intermediate entity of a multi-step question instead of the final answer), "
    "is 'none' / no answer, or contradicts the reference.\n"
    "Use only the information given here; do not rely on your own knowledge of the answer except to recognize that two "
    "names refer to the same thing.\n"
    "Reply with one short sentence of reasoning, then a final line exactly: VERDICT: CORRECT or VERDICT: INCORRECT"
)
_VERDICT = re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.I)


def _key(question, golds, answer, model):
    return hashlib.sha1(json.dumps([question, sorted(golds), answer, model]).encode()).hexdigest()


class QAJudge:
    def __init__(self, cache_path: str, model: str = JUDGE_MODEL, concurrency: int = 8):
        self.model, self.cache_path = model, cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.sem = asyncio.Semaphore(concurrency)

    def save(self):
        json.dump(self.cache, open(self.cache_path, "w"), indent=0)

    async def grade(self, question: str, golds: list[str], answer: str | None) -> dict:
        """-> {"correct": 0|1, "reason": str}. An empty / None answer is INCORRECT without a call."""
        if not answer or not str(answer).strip():
            return {"correct": 0, "reason": "no answer"}
        k = _key(question, golds, str(answer), self.model)
        if k in self.cache:
            return self.cache[k]
        import litellm
        user = (f"QUESTION: {question}\nREFERENCE answers: {json.dumps(golds, ensure_ascii=False)}\n"
                f"CANDIDATE: {answer}")
        async with self.sem:
            for attempt in range(5):
                try:
                    r = await litellm.acompletion(model=self.model, temperature=0, max_tokens=300,
                                                  messages=[{"role": "system", "content": SYSTEM},
                                                            {"role": "user", "content": user}])
                    text = r.choices[0].message.content or ""
                    break
                except Exception as e:  # rate limits / transient errors
                    text = f"ERROR {e}"
                    await asyncio.sleep(5 * (attempt + 1))
        m = _VERDICT.findall(text)
        out = {"correct": int(bool(m) and m[-1].upper() == "CORRECT"), "reason": text.strip()[:400], "parsed": bool(m)}
        self.cache[k] = out
        return out


def bare_question(q: str) -> str:
    """RULER qa prompts wrap the question in instructions + our placeholder; keep the 'Question: …' line if present."""
    m = re.search(r"Question:\s*(.+)", q)
    return (m.group(1) if m else q).strip().split("\n")[0]
