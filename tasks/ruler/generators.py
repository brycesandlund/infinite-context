"""Vendored RULER task generators.

The `_<family>_make_context` functions are direct ports of `generate_input_output`
from RULER's reference scripts:
- _niah_make_context     ← scripts/data/synthetic/niah.py
- _vt_make_context       ← scripts/data/synthetic/variable_tracking.py
- _cwe_make_context      ← scripts/data/synthetic/common_words_extraction.py
- _fwe_make_context      ← scripts/data/synthetic/freq_words_extraction.py

Each was refactored to take its inputs as function args (instead of argparse
globals) and to be reseedable per-call via an `rng` argument. The actual
needle-construction / chain-construction / sampling logic is unchanged from
the reference implementation.

`make_ruler_problem` is the orchestrator: looks up the task config from
RULER_TASKS, drives a binary search over the haystack size to land near the
caller's `doc_size_tokens` target, renders RULER's template with the context
replaced by our `read_chunk` placeholder, and packages the result as a
`tasks.base.Problem`.

Vendored from NVIDIA/RULER (Apache-2.0); see tasks/ruler/__init__.py.
"""

from __future__ import annotations

import heapq
import math
import random
import re
import string
import uuid

from tasks.base import Problem
from tasks.ruler._common import (
    cwe_word_pool,
    ensure_punkt,
    essay_words,
    niah_key_words,
    noise_sentence,
    render_question,
)
from tasks.ruler.constants import RULER_TASKS, TASK_TEMPLATES


# ---------------------------------------------------------------------------
# NIAH — vendored from RULER/scripts/data/synthetic/niah.py
# ---------------------------------------------------------------------------


_NIAH_NEEDLE_FMT = "One of the special magic {type_needle_v} for {key} is: {value}."

# RULER's depth grid: 40 evenly-spaced positions between 0% and 100% of the
# document. Insertion picks one depth per needle without replacement.
_DEPTHS = [round(i * 100 / 39) for i in range(40)]


def _gen_random_number(rng: random.Random, num_digits: int = 7) -> str:
    lo, hi = 10 ** (num_digits - 1), 10 ** num_digits - 1
    return str(rng.randint(lo, hi))


def _gen_random_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _gen_random_word(rng: random.Random) -> str:
    return rng.choice(niah_key_words())


def _gen_random(rng: random.Random, type_needle: str) -> str:
    if type_needle == "numbers":
        return _gen_random_number(rng)
    if type_needle == "words":
        return _gen_random_word(rng)
    if type_needle == "uuids":
        return _gen_random_uuid(rng)
    raise NotImplementedError(f"Needle type {type_needle!r} not implemented.")


def _niah_make_context(
    rng: random.Random,
    *,
    num_haystack: int,
    type_haystack: str,
    type_needle_k: str,
    type_needle_v: str,
    num_needle_k: int,
    num_needle_v: int,
    num_needle_q: int,
) -> tuple[str, list[str], str, str]:
    # RULER's niah.py applies this exact adjustment at startup: when there are
    # more queries than keys (as in niah_multiquery: k=1, q=4), bump k to match.
    num_needle_k = max(num_needle_k, num_needle_q)
    """Returns (context_text, answers, query_string, effective_type_needle_v).

    Effective_type_needle_v is the singularized noun ("number" vs "numbers") that
    must be reused when rendering the template — RULER's niah.py special-cases
    this when num_q*num_v==1.
    """
    keys: list[str] = []
    values: list[list[str]] = []
    needles: list[str] = []
    for _ in range(num_needle_k):
        keys.append(_gen_random(rng, type_needle_k))
        v_list = []
        for _ in range(num_needle_v):
            v_list.append(_gen_random(rng, type_needle_v))
            needles.append(
                _NIAH_NEEDLE_FMT.format(
                    type_needle_v=type_needle_v, key=keys[-1], value=v_list[-1],
                )
            )
        values.append(v_list)
    rng.shuffle(needles)

    if type_haystack == "essay":
        from nltk.tokenize import sent_tokenize
        ensure_punkt()
        pool = essay_words()
        if num_haystack <= len(pool):
            text = " ".join(pool[:num_haystack])
        else:
            repeats = (num_haystack + len(pool) - 1) // len(pool)
            text = " ".join((pool * repeats)[:num_haystack])
        sents = sent_tokenize(text.strip())
        # Pick distinct depths and convert each to a sentence-index split point.
        depth_sample = rng.sample(_DEPTHS, len(needles))
        insertions = sorted(int(len(sents) * (d / 100)) for d in depth_sample)
        insertions = [0, *insertions, len(sents)]
        parts: list[str] = []
        for i in range(1, len(insertions)):
            parts.append(" ".join(sents[insertions[i - 1] : insertions[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)
    elif type_haystack == "noise":
        sents = [noise_sentence()] * num_haystack
        positions = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
        for pos, needle_str in zip(positions, needles):
            sents.insert(pos, needle_str)
        context = "\n".join(sents)
    elif type_haystack == "needle":
        sents = [
            _NIAH_NEEDLE_FMT.format(
                type_needle_v=type_needle_v,
                key=_gen_random(rng, type_needle_k),
                value=_gen_random(rng, type_needle_v),
            )
            for _ in range(num_haystack)
        ]
        positions = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
        for pos, needle_str in zip(positions, needles):
            sents.insert(pos, needle_str)
        context = "\n".join(sents)
    else:
        raise NotImplementedError(f"Haystack {type_haystack!r} not implemented.")

    # Pick queried keys (RULER samples num_needle_q distinct keys); answers are
    # all values associated with those keys.
    qi = rng.sample(range(num_needle_k), num_needle_q)
    queried = [keys[i] for i in qi]
    answers = [v for i in qi for v in values[i]]
    query = (
        ", ".join(queried[:-1]) + ", and " + queried[-1] if len(queried) > 1 else queried[0]
    )

    # Singular/plural fix: RULER's niah.py special-cases num_q*num_v==1 by
    # stripping the trailing 's' from type_needle_v (and other template edits;
    # we handle those via the template substitution itself by adjusting the
    # template before format).
    effective = type_needle_v
    if num_needle_q * num_needle_v == 1:
        effective = type_needle_v[:-1]  # "numbers" → "number", "uuids" → "uuid"

    return context, answers, query, effective


def _niah_render_template(
    template: str, *, num_q_v: int
) -> str:
    """Apply RULER's plural→singular template edits when only one value is queried."""
    if num_q_v == 1:
        template = template.replace("Some", "A")
        template = template.replace("are all", "is")
        template = template.replace("are", "is")
        template = template.replace("answers", "answer")
    return template


# ---------------------------------------------------------------------------
# Variable tracking — vendored from RULER/scripts/data/synthetic/variable_tracking.py
# ---------------------------------------------------------------------------


def _vt_generate_chains(rng: random.Random, num_chains: int, num_hops: int) -> tuple[list[list[str]], list[list[str]]]:
    """Generate `num_chains` chains of `num_hops + 1` variable names + the
    assignment statements that bind them. Variables are 5-letter uppercase
    strings (RULER's choice). The first chain gets the queried value; other
    chains get distinct distractor values.
    """
    k = 5
    needed = num_chains * (num_hops + 1)
    seen: set[str] = set()
    while len(seen) < needed:
        seen.add("".join(rng.choices(string.ascii_uppercase, k=k)))
    all_vars = list(seen)
    chains: list[list[str]] = []
    statements: list[list[str]] = []
    used_values: set[str] = set()
    for ci in range(num_chains):
        chain_vars = all_vars[ci * (num_hops + 1) : (ci + 1) * (num_hops + 1)]
        chains.append(chain_vars)
        while True:
            val = str(rng.randint(10000, 99999))
            if val not in used_values:
                used_values.add(val)
                break
        stmts = [f"VAR {chain_vars[0]} = {val}"]
        for h in range(num_hops):
            stmts.append(f"VAR {chain_vars[h + 1]} = VAR {chain_vars[h]} ")
        statements.append(stmts)
    return chains, statements


def _vt_shuffle_sublists(rng: random.Random, lst: list[list[str]]) -> list[str]:
    """RULER's heap-based interleave: shuffles across sublists but preserves
    within-sublist order (so a chain's statements always appear in correct
    order — X1=N, X2=X1, X3=X2 — just possibly interleaved with other chains).
    """
    heap: list[tuple[float, int, int]] = []
    for i in range(len(lst)):
        heapq.heappush(heap, (rng.random(), i, 0))
    out: list[str] = []
    while heap:
        _, li, ei = heapq.heappop(heap)
        out.append(lst[li][ei])
        if ei + 1 < len(lst[li]):
            heapq.heappush(heap, (rng.random(), li, ei + 1))
    return out


def _vt_make_context(
    rng: random.Random,
    *,
    num_noises: int,
    num_chains: int,
    num_hops: int,
    type_haystack: str,
) -> tuple[str, list[str], str, int]:
    chains, statements = _vt_generate_chains(rng, num_chains, num_hops)
    # Query value comes from the first chain's literal-assignment statement.
    queried_value = statements[0][0].split("=")[-1].strip()
    flat = _vt_shuffle_sublists(rng, statements)

    if type_haystack == "essay":
        from nltk.tokenize import sent_tokenize
        ensure_punkt()
        text = " ".join(essay_words()[:num_noises])
        sents = sent_tokenize(text.strip())
        depth_sample = rng.sample(_DEPTHS, len(flat))
        insertions = sorted(int(len(sents) * (d / 100)) for d in depth_sample)
        insertions = [0, *insertions, len(sents)]
        parts: list[str] = []
        for i in range(1, len(insertions)):
            parts.append(" ".join(sents[insertions[i - 1] : insertions[i]]))
            if i - 1 < len(flat):
                parts.append(flat[i - 1].strip() + ".")
        context = " ".join(parts)
    elif type_haystack == "noise":
        sents = [noise_sentence()] * num_noises
        for chain in statements:
            positions = sorted(rng.sample(range(len(sents)), len(chain)))
            # Insert in order so the heap interleave is honored.
            for offset, (pos, stmt) in enumerate(zip(positions, chain)):
                sents.insert(pos + offset, stmt)
        context = "\n".join(sents)
    else:
        raise NotImplementedError(f"VT haystack {type_haystack!r} not implemented.")

    context = context.replace(". \n", ".\n")
    return context, chains[0], queried_value, num_hops + 1


# ---------------------------------------------------------------------------
# CWE — vendored from RULER/scripts/data/synthetic/common_words_extraction.py
# ---------------------------------------------------------------------------


def _cwe_make_context(
    rng: random.Random,
    *,
    num_words: int,
    freq_cw: int,
    freq_ucw: int,
    num_cw: int,
) -> tuple[str, list[str]]:
    pool = cwe_word_pool()
    if num_words <= len(pool):
        word_list_full = rng.sample(pool, num_words)
    else:
        # RULER falls back to a 466K-word "randle" wordlist; we don't ship that
        # but we can sample with replacement to fill the requested length.
        word_list_full = [rng.choice(pool) for _ in range(num_words)]
    common, uncommon = word_list_full[:num_cw], word_list_full[num_cw:]
    word_list = common * freq_cw + uncommon * freq_ucw
    rng.shuffle(word_list)
    context = " ".join(f"{i + 1}. {w}" for i, w in enumerate(word_list))
    return context, common


# ---------------------------------------------------------------------------
# FWE — vendored from RULER/scripts/data/synthetic/freq_words_extraction.py
# ---------------------------------------------------------------------------


def _zeta_truncated(alpha: float, n_terms: int) -> float:
    """Truncated Riemann zeta — replaces scipy.special.zeta. Identical to
    RULER's denominator for finite vocab_size, which is what RULER uses."""
    return sum(1.0 / (k ** alpha) for k in range(1, n_terms + 1))


def _fwe_make_context(
    rng: random.Random,
    *,
    num_words: int,
    alpha: float,
    coded_wordlen: int,
    vocab_size: int,
) -> tuple[str, list[str]]:
    vocab_set: set[str] = set()
    while len(vocab_set) < vocab_size:
        vocab_set.add("".join(rng.choices(string.ascii_lowercase, k=coded_wordlen)))
    vocab = sorted(vocab_set)
    rng.shuffle(vocab)
    vocab[0] = "..."  # RULER: top-ranked is noise

    zeta = _zeta_truncated(alpha, len(vocab))
    counts = [int(num_words * (k ** -alpha) / zeta) for k in range(1, len(vocab) + 1)]
    stream: list[str] = []
    for w, c in zip(vocab, counts):
        stream.extend([w] * c)
    rng.shuffle(stream)
    context = " ".join(stream)
    return context, list(vocab[1:4])


# ---------------------------------------------------------------------------
# Orchestrator: binary-search the haystack size so context ≈ doc_size_tokens,
# then render the template and build a Problem.
# ---------------------------------------------------------------------------


def _binary_search_haystack(
    *,
    sizer,                # callable: (num_haystack) -> int token count
    target_tokens: int,
    incremental: int,
    slack: float = 0.95,
    max_iters: int = 32,
) -> int:
    """Find the largest haystack size whose context fits under target_tokens.
    Mirrors RULER's binary-search-with-3x-upper-bound heuristic."""
    sample_tokens = sizer(incremental)
    tokens_per_unit = max(sample_tokens / incremental, 1e-6)
    lower = incremental
    upper = max(int((target_tokens / tokens_per_unit) * 3), incremental * 2)
    best = incremental
    for _ in range(max_iters):
        if lower > upper:
            break
        mid = (lower + upper) // 2
        size = sizer(mid)
        if size <= target_tokens:
            best = mid
            lower = mid + 1
        else:
            upper = mid - 1
    # Some safety: if even `incremental` overflows, give back the smallest viable.
    if best == incremental and sizer(incremental) > target_tokens:
        return max(1, incremental // 2)
    return best


def _build_niah_problem(
    task_name: str,
    args: dict,
    tokenizer,
    doc_size_tokens: int,
    seed: int,
) -> Problem:
    incremental = 500 if args["type_haystack"] == "essay" else 25

    def sizer(n: int) -> int:
        ctx, _, _, _ = _niah_make_context(
            random.Random(seed), num_haystack=n, **args,
        )
        return len(tokenizer.encode(ctx, add_special_tokens=False))

    num_haystack = _binary_search_haystack(
        sizer=sizer, target_tokens=doc_size_tokens, incremental=incremental,
    )

    rng = random.Random(seed)
    context, answers, query, effective_v = _niah_make_context(
        rng, num_haystack=num_haystack, **args,
    )
    template = TASK_TEMPLATES["niah"]["template"]
    template = _niah_render_template(
        template, num_q_v=args["num_needle_q"] * args["num_needle_v"],
    )
    doc_tokens = tokenizer.encode(context, add_special_tokens=False)
    _, question = render_question(
        template,
        context=context,
        doc_token_count=len(doc_tokens),
        type_needle_v=effective_v,
        query=query,
    )
    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=answers,
        task=task_name,
        # RULER fidelity: the root agent must receive only RULER's task template
        # (carried in `question`). No needle-format hint here — the model infers
        # it from the instruction, exactly as in RULER. Subagents inherit task
        # knowledge via the parent's `subtask` string. (To re-enable per-subagent
        # hints during training, populate task_context for depth>0 only.)
        task_context="",
        metadata={"num_haystack": num_haystack, "query": query, **args},
    )


def _build_vt_problem(
    task_name: str, args: dict, tokenizer, doc_size_tokens: int, seed: int,
) -> Problem:
    incremental = 50 if args["type_haystack"] == "essay" else 5

    def sizer(n: int) -> int:
        ctx, _, _, _ = _vt_make_context(
            random.Random(seed), num_noises=n, **args,
        )
        return len(tokenizer.encode(ctx, add_special_tokens=False))

    num_noises = _binary_search_haystack(
        sizer=sizer, target_tokens=doc_size_tokens, incremental=incremental,
    )

    rng = random.Random(seed)
    context, gold_vars, queried_value, num_v = _vt_make_context(
        rng, num_noises=num_noises, **args,
    )
    template = TASK_TEMPLATES["variable_tracking"]["template"]
    doc_tokens = tokenizer.encode(context, add_special_tokens=False)
    _, question = render_question(
        template, context=context, doc_token_count=len(doc_tokens), query=queried_value,
    )
    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=list(gold_vars),
        task=task_name,
        task_context="",  # RULER fidelity — see _build_niah_problem note.
        metadata={"num_noises": num_noises, "queried_value": queried_value, **args},
    )


def _build_cwe_problem(
    task_name: str, args: dict, tokenizer, doc_size_tokens: int, seed: int,
) -> Problem:
    incremental = 50

    def sizer(n: int) -> int:
        ctx, _ = _cwe_make_context(random.Random(seed), num_words=n, **args)
        return len(tokenizer.encode(ctx, add_special_tokens=False))

    # CWE's context grows ~linearly in num_words.
    num_words = _binary_search_haystack(
        sizer=sizer, target_tokens=doc_size_tokens, incremental=incremental,
    )

    rng = random.Random(seed)
    context, gold_common = _cwe_make_context(rng, num_words=num_words, **args)
    template = TASK_TEMPLATES["common_words_extraction"]["template"]
    doc_tokens = tokenizer.encode(context, add_special_tokens=False)
    _, question = render_question(
        template, context=context, doc_token_count=len(doc_tokens),
    )
    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=list(gold_common),
        task=task_name,
        task_context="",  # RULER fidelity — see _build_niah_problem note.
        metadata={"num_words": num_words, **args},
    )


def _build_fwe_problem(
    task_name: str, args: dict, tokenizer, doc_size_tokens: int, seed: int,
) -> Problem:
    # RULER scales vocab roughly with input length.
    vocab_size = max(doc_size_tokens // 50, 200)
    coded_wordlen = 6
    incremental = 200

    def sizer(n: int) -> int:
        ctx, _ = _fwe_make_context(
            random.Random(seed),
            num_words=n,
            alpha=args["alpha"],
            coded_wordlen=coded_wordlen,
            vocab_size=vocab_size,
        )
        return len(tokenizer.encode(ctx, add_special_tokens=False))

    num_words = _binary_search_haystack(
        sizer=sizer, target_tokens=doc_size_tokens, incremental=incremental,
    )
    rng = random.Random(seed)
    context, gold_top3 = _fwe_make_context(
        rng,
        num_words=num_words,
        alpha=args["alpha"],
        coded_wordlen=coded_wordlen,
        vocab_size=vocab_size,
    )
    template = TASK_TEMPLATES["freq_words_extraction"]["template"]
    doc_tokens = tokenizer.encode(context, add_special_tokens=False)
    _, question = render_question(
        template, context=context, doc_token_count=len(doc_tokens),
    )
    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=list(gold_top3),
        task=task_name,
        task_context="",  # RULER fidelity — see _build_niah_problem note.
        metadata={"num_words": num_words, "vocab_size": vocab_size, **args},
    )


# ---------------------------------------------------------------------------
# QA (SQuAD / HotpotQA) — vendored from RULER/scripts/data/synthetic/qa.py
# Held-out eval: golden paragraph(s) buried among distractor paragraphs.
# ---------------------------------------------------------------------------


_QA_DOCUMENT_PROMPT = "Document {i}:\n{document}"


def _qa_make_context(rng, qas, docs, index, num_docs):
    """Build the document block for QA question `index` with `num_docs` total
    documents (golden always included), shuffled. Returns (context, query, answers)."""
    curr_q = qas[index]["query"]
    curr_a = qas[index]["outputs"]
    curr_docs = qas[index]["context"]
    curr_more = qas[index].get("more_context", [])
    if num_docs < len(docs):
        if (num_docs - len(curr_docs)) > len(curr_more):
            taken = set(curr_docs) | set(curr_more)
            addition = [i for i in range(len(docs)) if i not in taken]
            n_extra = max(0, num_docs - len(curr_docs) - len(curr_more))
            all_idx = curr_docs + curr_more + rng.sample(addition, min(n_extra, len(addition)))
        else:
            all_idx = curr_docs + rng.sample(curr_more, num_docs - len(curr_docs))
        all_docs = [docs[i] for i in all_idx]
    else:
        repeats = (num_docs + len(docs) - 1) // len(docs)
        all_docs = (docs * repeats)[:num_docs]
    rng.shuffle(all_docs)
    context = "\n\n".join(
        _QA_DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs)
    )
    return context, curr_q, curr_a


def _build_qa_problem(
    task_name: str, args: dict, tokenizer, doc_size_tokens: int, seed: int,
) -> Problem:
    from tasks.ruler.qa_data import load_qa

    qas, docs = load_qa(args["dataset"])
    index = seed % len(qas)
    template = TASK_TEMPLATES["qa"]["template"]

    def sizer(n: int) -> int:
        ctx, _, _ = _qa_make_context(random.Random(seed), qas, docs, index, n)
        return len(tokenizer.encode(ctx, add_special_tokens=False))

    num_docs = _binary_search_haystack(
        sizer=sizer, target_tokens=doc_size_tokens, incremental=10,
    )
    rng = random.Random(seed)
    context, query, answers = _qa_make_context(rng, qas, docs, index, num_docs)
    doc_tokens = tokenizer.encode(context, add_special_tokens=False)
    _, question = render_question(
        template, context=context, doc_token_count=len(doc_tokens), query=query,
    )
    return Problem(
        document_tokens=doc_tokens,
        question=question,
        gold_answers=list(answers),
        task=task_name,
        task_context="",  # RULER fidelity — instruction lives in the question.
        metadata={"num_docs": num_docs, "qa_index": index, "dataset": args["dataset"]},
    )


_BUILDERS = {
    "niah": _build_niah_problem,
    "variable_tracking": _build_vt_problem,
    "common_words_extraction": _build_cwe_problem,
    "freq_words_extraction": _build_fwe_problem,
    "qa": _build_qa_problem,
}


def make_ruler_problem(
    task_name: str,
    corpus_tokens: list[int],  # unused — kept for registry signature compat
    tokenizer,
    doc_size_tokens: int,
    seed: int,
) -> Problem:
    if task_name not in RULER_TASKS:
        raise ValueError(f"Unknown RULER task: {task_name!r}")
    cfg = RULER_TASKS[task_name]
    family = cfg["task"]
    if family not in _BUILDERS:
        raise NotImplementedError(
            f"RULER task family {family!r} not implemented yet (task={task_name})."
        )
    return _BUILDERS[family](task_name, dict(cfg["args"]), tokenizer, doc_size_tokens, seed)
