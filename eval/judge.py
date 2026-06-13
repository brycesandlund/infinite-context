"""LLM-as-a-judge — a backend-agnostic grader built on the ModelBackend interface.

Why this exists: the policy's SUBAGENT subtasks are open-ended (the parent invents
them at rollout time), so there is no precomputed gold to grade a subagent's answer
against — and on real / RULER tasks we have no per-example labels at all. A judge
reads (task, answer[, source]) and returns a scalar in [0, 1], filling the gap that
the synthetic-only `example_spans` oracle can't.

Backend-agnostic on purpose: any ModelBackend can judge (Tinker policy, Claude, GPT),
so the judge is swappable and independent of the policy being trained. make_judge()
defaults to GPT-5.4 nano via APIBackend — cheap, and decoupled from the policy so it
can't trivially be reward-hacked by the policy judging itself.

This module is just the grader plumbing; wiring it into the RL reward is a separate
step.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

from eval.backends import APIBackend, ModelBackend

# Match the final "SCORE: 0.7" line. Accept 0..1 (and a bare ".7"); take the LAST
# occurrence so any score mentioned mid-reasoning doesn't win over the verdict.
_SCORE_RE = re.compile(r"SCORE\s*[:=]\s*(1(?:\.0+)?|0?\.\d+|0)", re.IGNORECASE)

JUDGE_SYSTEM = (
    "You are a strict grader. You receive a TASK that was assigned to an agent, the "
    "agent's ANSWER, and optionally the SOURCE text the agent worked from.\n"
    "Judge how correctly and completely the ANSWER satisfies the TASK:\n"
    "- If SOURCE is given, verify the answer against it; do NOT credit claims that the "
    "SOURCE does not support.\n"
    "- Reward a correct, on-format answer; penalize wrong, irrelevant, fabricated, or "
    "incomplete answers.\n"
    "Think briefly, then end with exactly one final line:\n"
    "SCORE: <number from 0 to 1>\n"
    "where 1.0 = fully correct, 0.0 = wrong/irrelevant/unsupported, partial in between."
)


@dataclass
class JudgeVerdict:
    score: float          # clamped to [0, 1]
    reasoning: str        # the judge's text (for inspection/logging)
    parsed: bool          # whether a SCORE line was found (False => `default` used)


@dataclass
class Judge:
    """Grades (task, answer[, source]) -> JudgeVerdict via any ModelBackend."""

    backend: ModelBackend
    max_tokens: int = 512
    default: float = 0.0   # score used when no SCORE line parses (treat as failure)

    def _prompt(self, task: str, answer: str, source: str | None, rubric: str | None) -> list[dict]:
        parts = [f"TASK:\n{task.strip()}", f"ANSWER:\n{(answer or '').strip() or '(no answer given)'}"]
        if source:
            parts.append(f"SOURCE:\n{source.strip()}")
        if rubric:
            parts.append(f"RUBRIC:\n{rubric.strip()}")
        return [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    async def score(
        self,
        *,
        task: str,
        answer: str,
        source: str | None = None,
        rubric: str | None = None,
    ) -> JudgeVerdict:
        text = await self.backend.complete(
            self._prompt(task, answer, source, rubric), max_tokens=self.max_tokens
        )
        matches = list(_SCORE_RE.finditer(text))
        if matches:
            score = float(matches[-1].group(1))
            return JudgeVerdict(score=max(0.0, min(1.0, score)), reasoning=text.strip(), parsed=True)
        return JudgeVerdict(score=self.default, reasoning=text.strip(), parsed=False)

    async def score_batch(self, items: list[dict]) -> list[JudgeVerdict]:
        """Concurrently score many {task, answer, source?, rubric?} dicts."""
        return await asyncio.gather(*(self.score(**it) for it in items))


def make_judge(model: str | None = None, temperature: float = 0.0, max_tokens: int = 512) -> Judge:
    """Build a judge. Defaults to GPT-5.4 nano (override via JUDGE_MODEL env or arg).
    temperature=0 for grading stability (dropped automatically for models that fix it).
    max_tokens also raises the backend's output ceiling — reasoning models (gpt-5.x)
    spend hidden reasoning tokens against this budget, and too small a cap truncates
    the response to empty (no SCORE line)."""
    model = model or os.environ.get("JUDGE_MODEL", "openai/gpt-5.4-nano")
    backend = APIBackend(model, temperature=temperature, max_output_cap=max_tokens)
    return Judge(backend=backend, max_tokens=max_tokens)
