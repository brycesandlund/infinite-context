"""OOLONG-synth: information-aggregation over in-context classification.

Unlike RULER (find/count literal strings), OOLONG requires the model to
*classify* each example (sentiment, topic, ...) and then aggregate over the
labels — counting, user-conditioned, or temporal questions. This is the home of
the decomposition thesis: label every chunk atomically, then aggregate.

Built from source text-classification datasets (we own the per-example labels,
which the classify-and-count oracle needs). v1 = the Counting question family.
"""

from tasks.oolong.generators import OOLONG_TASKS, make_oolong_problem

__all__ = ["OOLONG_TASKS", "make_oolong_problem"]
