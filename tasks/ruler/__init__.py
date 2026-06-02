"""Vendored RULER task generators (NVIDIA/RULER, Apache-2.0).

This package contains task generators adapted from
https://github.com/NVIDIA/RULER/tree/main/scripts/data/synthetic
(commit ab17b78, the source-of-truth pointed to by NeMo-Skills'
`nemo_skills/dataset/ruler/prepare.py`).

Two adaptations for the recursive-agent setting:
1. The haystack (background + inserted needles) is returned in
   `Problem.document_tokens` and accessed by the agent via the `read_chunk`
   tool, rather than inlined into the prompt's `{context}` slot.
2. The model emits its answer in `\\boxed{...}` rather than completing RULER's
   "Task Answer Prefix", because the agent's final turn contains thinking +
   tool calls, not a single prefix-completion. RULER's official string_match
   grader works fine on whatever's inside the box.

Everything else — task templates, needle phrasing, value generation (7-digit
numbers, uuid4 keys/values, wonderwords adj/noun/verb keys), sentence-boundary
insertion via nltk.sent_tokenize, Zeta(α) word sampling for FWE, etc. —
matches RULER's reference implementation byte-for-byte.
"""

from tasks.ruler.constants import RULER_TASKS, TASK_TEMPLATES
from tasks.ruler.generators import make_ruler_problem

__all__ = ["RULER_TASKS", "TASK_TEMPLATES", "make_ruler_problem"]
