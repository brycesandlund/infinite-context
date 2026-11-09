"""Realdoc tasks — real Gutenberg-novel prose with exact-computable-gold questions.
See generators.py. v1: realdoc_count (count a word's occurrences; sum combine)."""

from tasks.realdoc.generators import REALDOC_TASKS, make_realdoc_problem

__all__ = ["REALDOC_TASKS", "make_realdoc_problem"]
