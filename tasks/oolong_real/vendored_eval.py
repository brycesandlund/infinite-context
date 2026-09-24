"""VENDORED OOLONG-real scoring — verbatim from abertsch72/oolong, src/eval/eval_helpers.py
(commit 0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c, MIT license: LICENSE_oolong). Only the DnD-split parse
functions are copied unchanged; `dnd_score` is `dnd_process_response`'s scoring block with the bookkeeping
dict stripped (same branches, same order, same arithmetic).
"""
import re

def dnd_parse_answer(answer) -> int | str | list[str]:
    """Parse the answer into int, str, or list of str."""
    # Try to convert to int first
    try:
        return int(answer)
    except ValueError:
        pass

    # Check if it contains commas (list case)
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]

    # Otherwise return as string
    return answer


def dnd_parse_response(answer) -> tuple[str, str]:
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer) or re.search(
        r"\\boxed[\{]+([^}]*)[\}]+", answer
    )
    if match:
        answer = match.group(1)
    else:
        return answer, "low"
    return dnd_parse_answer(answer), "high"



def dnd_score(gold_raw: str, output: str) -> float:
    """== dnd_process_response(datapoint, output, model)["score"] with datapoint["answer"] = gold_raw."""
    gold = dnd_parse_answer(gold_raw)
    trimmed_output, parse_confidence = dnd_parse_response(output)
    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        score = 0.75 ** abs(gold - trimmed_output)
    elif isinstance(gold, str) and isinstance(trimmed_output, str):
        score = float(gold.strip().lower() == trimmed_output.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        score = len(overlap) / len(gold) if gold else 0.0
    return score
