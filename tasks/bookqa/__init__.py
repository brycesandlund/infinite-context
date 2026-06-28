"""Book-QA tasks — real human questions over anonymized real novels (InfiniteBench
En.QA). See generators.py. Leaf retrieves question-entity sentences; bounded top-K
evidence combine; root answers from the collected evidence."""

from tasks.bookqa.generators import BOOKQA_TASKS, make_bookqa_problem

__all__ = ["BOOKQA_TASKS", "make_bookqa_problem"]
