"""Abstract synthetic decomposition tasks — train the scaffold (split + combine +
strategy) on trivial-leaf-op data, then transfer it zero-shot to RULER / OOLONG.

See generators.py. Two families: bounded-associative (binary tree-reduce) and
stateful-sequential (left-fold)."""

from tasks.synth.generators import SYNTH_TASKS, gold_for, make_synth_problem

__all__ = ["SYNTH_TASKS", "make_synth_problem", "gold_for"]
