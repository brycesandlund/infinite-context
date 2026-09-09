"""Question-phrasing diversity for FILTERED aggregation questions.

The root's first turn turns the question into the op phrase every subtask carries; children see
only that subtask, so a condition the root fails to restate is simply gone from the tree. In run 4
(OOLONG user, seed 2100000) the root applied the user filter in its own leaf but wrote "tallying
each label" for the rest of the chain — the condition was stated as a separate LEADING sentence,
a shape our filtered training questions (always "Among ONLY the records with X, …") never used.
So the model had learned one surface position for the condition, not "restate whatever condition
the question imposes". These variants put the same condition in several positions and wordings;
gold and oracle op phrase are unchanged (the oracle reads metadata, never the question text).
"""

from __future__ import annotations

import random


def filtered_question(rng: random.Random, unit: str, cond: str, body: str, tail: str) -> str:
    """`unit` plural noun ("records" / "entries"); `cond` a field condition ("flag=Y",
    "src=B"); `body` the question clause, lowercase, no trailing "?" ("which grp value is the
    MOST common"); `tail` the tie/format sentence(s) ending with the \\boxed{} instruction."""
    B = body[0].upper() + body[1:]
    forms = [
        f"Among ONLY the {unit} with {cond}, {body}? {tail}",
        f"For this question, consider only the {unit} with {cond}. Among those {unit}, {body}? {tail}",
        f"Restrict attention to the subset of {unit} where {cond}. {B}? {tail}",
        f"Ignore the {unit} that do not have {cond}. Among the remaining {unit}, {body}? {tail}",
        f"{B}, considering only the {unit} with {cond}? {tail}",
        f"Looking only at {unit} that have {cond}: {body}? {tail}",
        f"Filter the {unit} to those with {cond}, then answer: {body}? {tail}",
    ]
    return rng.choice(forms)
