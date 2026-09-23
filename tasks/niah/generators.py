"""Synthetic "hidden sentences in a REAL haystack" tasks — teach decomposition of prose haystacks
with MECHANICAL, faithful leaf-ops, across a BROAD variant family (the goal is general capability,
not fitting one eval config).

Why: the model only decomposes documents that look like its training data; on unfamiliar
haystacks it single-shots / over-reads its whole range and overflows. Hidden mechanical facts in
real filler + the scaffold transferred to RULER (niah_novel took cwe 0.00 -> 0.97). All leaves are
scripted matches — no model, no rejection sampling.

Tasks (each draws its VARIANT per seed, so one task name covers a family):
- `niah_novel` : ONE needle, explicit key in the question. BookQAOracle scripted fallback.
- `niah_multi` : the RULER-niah family generalized. Per seed: query MODE in {hidden (a "the key you
                 must look up is K" sentence is itself hidden in the doc), explicit (key in the
                 question; 1-6 needles = multikey), multiquery (2-4 keys asked -> list all values),
                 multivalue (one key, 2-4 values -> list all)} x key type {words, uuids} x value type
                 {numbers, uuids} x FILLER {novel, essay, noise}. Collect-then-resolve on dict state
                 (leaf folds facts, combine merges, root resolves). Oracle: NiahMultiOracle.
- `vt_novel`   : RULER-vt variable chains generalized. Per seed: 2-4 chains x 3-7 hops, question in
                 {which_vars ("which variables hold value V?", set), final_value ("final value of
                 VAR X?", exact)} x the same three fillers. Left-fold. Oracle: VtNovelOracle.

Span layouts (`metadata["record_spans"]`, one per inserted sentence, doc order):
  niah_novel : (tok_start, tok_end, idx, relevance=1, snippet, has_answer=True)
  niah_multi : (tok_start, tok_end, idx, key, value_or_None, is_query)
  vt_novel   : (tok_start, tok_end, idx, name, rhs, is_ref)
"""

from __future__ import annotations

import random
import re
import string
import uuid

from tasks.base import Problem
from tasks.realdoc.books import BOOKS
from tasks.realdoc.generators import _book_tokens

NIAH_TASKS: dict[str, dict] = {
    "niah_novel": {"family": "niah", "strategy": "binary"},
    "niah_multi": {"family": "niah", "strategy": "binary"},
    "vt_novel": {"family": "niah", "strategy": "left_fold"},
}

# Key pool — plain nouns, à la RULER's magic-number keys.
_KEYS = [
    "harbor", "lantern", "compass", "meadow", "cascade", "ember", "thicket", "quarry",
    "beacon", "trellis", "cavern", "orchard", "granite", "willow", "marble", "cinder",
    "vellum", "saffron", "obsidian", "juniper", "cobalt", "bramble", "hollow", "pewter",
    "mariner", "citadel", "gallows", "tempest", "furrow", "lattice", "verdigris", "gable",
    "sextant", "brindle", "cistern", "palisade", "reliquary", "escarpment", "byway", "kiln",
]

# RULER's needle template, verbatim: type word is "numbers" or "uuids".
_NEEDLE = "One of the special magic {vtype} for {key} is: {value}."
_QUERY_NEEDLE = "For this task, the key you must look up is {key}."
_NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "

_QUESTION = (
    "What is the special magic number for {key}? A sentence somewhere in the document states "
    "it. Give the number in \\boxed{{}}."
)

# Question-phrasing diversity for the retrieval tasks. The ROOT has to map an arbitrary
# question onto the per-node subtask; with one template per task it learns the template, not
# the mapping (run 4: RULER's "memorize it, I will quiz you… mentioned in the provided text?"
# phrasing produced a subtask with literal START..END placeholders). Each problem draws an
# optional PREFACE (instructional framing, not part of the question) + one QUESTION form +
# one ANSWER-FORMAT tail. None of these copies RULER's wording.
_PREFACES = [
    "", "", "",
    "Some facts are buried in the text below. ",
    "Read carefully — a detail in this passage matters later. ",
    "You will be asked about a detail from this text. ",
    "A note about this passage: it contains a few planted facts. ",
    "Keep track of any special values you come across. ",
]
_ANSWER_TAILS = [
    "Give it in \\boxed{}.",
    "Put the answer in \\boxed{}.",
    "Reply with just the value inside \\boxed{}.",
    "Answer with the value, inside \\boxed{}.",
]
_Q_SINGLE = [
    "What is the special magic {vword} for {key}? A sentence somewhere in the document states it.",
    "Which special magic {vword} is given for {key}?",
    "Find the special magic {vword} associated with {key}.",
    "According to the text, what is the special magic {vword} for {key}?",
    "One sentence names the special magic {vword} for {key} — what is it?",
    "What special magic {vword} does the passage assign to {key}?",
]
_Q_MULTIQUERY = [
    "What are the special magic {vword}s for {ks}? One sentence per key states it. List all of them, comma-separated.",
    "Report the special magic {vword} for each of {ks}, comma-separated.",
    "For each of {ks}, what is its special magic {vword}? List them all, comma-separated.",
    "Which special magic {vword}s are given for {ks}? Give every one, comma-separated.",
]
_Q_MULTIVALUE = [
    "What are all the special magic {vword}s for {key}? Several sentences each state one. List all of them, comma-separated.",
    "List every special magic {vword} the text gives for {key}, comma-separated.",
    "{key} is assigned more than one special magic {vword}. What are all of them? Comma-separated.",
    "Collect all the special magic {vword}s stated for {key}, comma-separated.",
]
_Q_HIDDEN = [
    "This document hides several sentences, each stating the special magic {vword} for some key, and ONE sentence saying which key you must look up. Find that instruction, then report the magic {vword} for that key.",
    "Somewhere in the text, one sentence tells you which key to look up; other sentences give each key's special magic {vword}. What is the magic {vword} for the key you are told to look up?",
    "The passage names a key to look up and, elsewhere, the special magic {vword} of several keys. Report the magic {vword} of the named key.",
    "Find the lookup instruction hidden in the text, then answer it: what is the special magic {vword} for the key it names?",
]
_Q_VT_WHICH = [
    "Which variables end up assigned the value {t_val}? List every such variable name, comma-separated.",
    "After all assignments, which variables hold the value {t_val}? List them all, comma-separated.",
    "Name every variable whose final value is {t_val}, comma-separated.",
    "Which VARs finish with the value {t_val}? Give all of them, comma-separated.",
    # FILTER-phrased forms (no "final"/"end up" cue): the question names the variables holding one value,
    # but the accumulator must still carry every variable — this is the phrasing family whose root the
    # run-9w model answered with a set filter on RULER vt. The contrastive state sentence needs
    # questions that look like this to bite on.
    "Which variables are assigned the value {t_val}, directly or through a chain of copies? List them all, comma-separated.",
    "Find every variable that ends up with the value {t_val} (a copy passes the value along). List them, comma-separated.",
]
_Q_VT_FINAL = [
    "What is the final value of VAR {qvar}? Give the integer.",
    "After all the assignments are applied, what value does VAR {qvar} hold?",
    "What does VAR {qvar} equal at the end of the text?",
    "Trace the assignments: what is VAR {qvar}'s final value?",
]


_LIST_TAILS = [   # for questions whose answer is a LIST of names — "the value" would be the wrong noun
    "Give them in \\boxed{}.",
    "Put the list in \\boxed{}.",
    "Reply with just the answers, comma-separated, inside \\boxed{}.",
    "Answer with the list inside \\boxed{}.",
]


_VT_PREFACES = [   # vt's own prefaces: about assignments, not "planted facts"/"special values" (needle vocabulary)
    "", "", "",
    "The assignments in this passage form chains; a later line can overwrite an earlier one. ",
    "Some lines in this passage assign variables; the rest is ordinary prose. ",
    "Assignments are scattered through the text and must be applied in the order they appear. ",
    "Keep a running picture of every variable's current value as you go. ",
]


def _phrase(rng, forms, tails=None, prefaces=None, **kw) -> str:
    return rng.choice(prefaces or _PREFACES) + rng.choice(forms).format(**kw) + " " + rng.choice(tails or _ANSWER_TAILS)
_CONTEXT = (
    "The document is a long passage of text with a single factual sentence hidden inside it. "
    "Answer the question using ONLY this passage."
)
_MULTI_CONTEXT = (
    "The document is a long passage of text with a few short factual sentences hidden inside "
    "it. Answer the question using ONLY this passage."
)
_NEEDLE_CONTEXT = (   # needle haystack: the "hidden inside a passage" descriptions above would be false
    "The document is a long run of short factual sentences, each stating a special magic value for a "
    "different key. Answer the question using ONLY this passage."
)
_VT_LIT = "VAR {name} = {value}."
_VT_REF = "VAR {name} = VAR {rhs}."
_VT_CONTEXT = (
    "The document is a long passage of text with short 'VAR ... = ...' assignment lines hidden "
    "inside it, in order. Answer the question using ONLY this passage."
)
_VT_PREAMBLE = (
    "Memorize and track the chains of variable assignment hidden in the text (a line "
    "'VAR A = VAR B' copies B's current value into A). "
)
# BARE framing: same task, no copy-semantics gloss. The gloss ("copies B's CURRENT value") states
# the order-dependence fact outright, and it used to be on 100% of vt_novel questions while RULER's
# vt question states nothing about the format at all — so a root trained only on the glossed form
# learns to be TOLD the task is sequential instead of deriving it from the VAR lines it reads.
# Run 13 (from base) folded 2/2 in-distribution and went binary 5/5 on RULER vt.
_VT_PREAMBLE_BARE = "Memorize and track the chains of variable assignment hidden in the text. "
_VT_GLOSS_FRAC = 0.6      # 60% keep the gloss, 40% must infer order-dependence from the document
# Order-dependence is signalled to the root by THREE independent things: the system description
# ("...assignment lines hidden inside it, in order"), the question's copy-semantics gloss, and the
# optional order preface. RULER's vt has none of the three. Left independent, the all-bare case is
# only 0.5 x 0.4 x 0.43 ~= 9% of traces — too thin for the condition that actually matters — so a
# fixed share is drawn BARE ON ALL THREE AXES at once, and the rest keep the independent mix.
_VT_BARE_FRAC = 0.35


# ---------------------------------------------------------------------------
# shared: fillers + ordered multi-sentence insertion + token spans
# ---------------------------------------------------------------------------


def _filler(rng, tokenizer, doc_size_tokens, kind="novel", corpus_tokens=None) -> str:
    """Haystack text of ~doc_size_tokens. novel = Gutenberg prose; essay = the PG-essay corpus
    (RULER's 'essay' haystack; falls back to novel if no corpus given); noise = RULER's repeated
    filler sentences."""
    if kind == "noise":
        reps = doc_size_tokens * 4 // len(_NOISE) + 1
        return (_NOISE * reps)[: doc_size_tokens * 4]
    if kind == "essay" and corpus_tokens:
        start = rng.randint(0, max(0, len(corpus_tokens) - doc_size_tokens))
        return tokenizer.decode(corpus_tokens[start:start + doc_size_tokens])
    name = rng.choice(sorted(BOOKS))
    bt = _book_tokens(name, tokenizer)
    start = rng.randint(0, max(0, len(bt) - doc_size_tokens))
    return tokenizer.decode(bt[start:start + doc_size_tokens])


def _boundaries(filler: str, k: int, rng) -> list[int]:
    """k distinct sentence-boundary char positions well inside the filler, ascending."""
    bounds = [m.end() for m in re.finditer(r"[.!?]\s", filler)]
    bounds = [p for p in bounds if 0.05 * len(filler) < p < 0.95 * len(filler)]
    if len(bounds) < k:   # degenerate filler — spread evenly instead
        bounds = [int(len(filler) * (i + 1) / (k + 1)) for i in range(k)]
    return sorted(rng.sample(bounds, k))


def _insert(filler: str, positions: list[int], sentences: list[str]):
    """Insert sentences[i] at ascending positions[i] (DOCUMENT ORDER == list order).
    Returns (doc_text, [(char_start, char_end)]) per inserted sentence."""
    out, last, spans, cur = [], 0, [], 0
    for p, s in zip(positions, sentences):
        out.append(filler[last:p]); cur += p - last
        spans.append((cur, cur + len(s)))
        out.append(s + " "); cur += len(s) + 1
        last = p
    out.append(filler[last:])
    return "".join(out), spans


def _tok_spans(offs, char_spans):
    """Map char spans -> (tok_start, tok_end) over an offsets mapping (one monotone pass)."""
    res, i = [], 0
    for cs, ce in char_spans:
        while i < len(offs) and offs[i][1] <= cs:
            i += 1
        j = i
        while j < len(offs) and offs[j][0] < ce:
            j += 1
        res.append((i, j))
    return res


def _pick_filler_kind(rng) -> str:
    return rng.choice(["novel", "novel", "essay", "noise"])


# RULER's `type_haystack: needle` (niah_multikey_2/3): the haystack is NOTHING but distractor needles
# of the same shape as the real one, so every leaf range is dense with wrong key=value lines and the
# leaf must filter by key. Run 14w: multikey_2 0.80, multikey_3 0.00 (4/5 overflow) — no training
# problem had ever put more than a handful of needles in a range.
_NEEDLE_HAYSTACK_FRAC = 0.25


def _compound_key(rng) -> str:
    """Hyphenated two-word key (RULER's needle haystack uses adjective-noun keys; 40 plain nouns
    can't key ~200 distinct distractors)."""
    a, b = rng.sample(_KEYS, 2)
    return f"{a}-{b}"


def _mk_key(rng, ktype: str, compound: bool) -> str:
    if ktype == "uuids":
        return _uuid(rng)
    return _compound_key(rng) if compound else rng.choice(_KEYS)


def _needle_filler(rng, tokenizer, doc_size_tokens, ktype, vtype, exclude) -> str:
    """Distractor needles only (never a key in `exclude`), ~doc_size_tokens long. The distractors are
    NOT records: they carry no answer for the asked key, so a filtering leaf reports nothing for them."""
    mk_val = (lambda: _uuid(rng)) if vtype == "uuids" else (lambda: str(rng.randint(1_000_000, 9_999_999)))
    probe = _NEEDLE.format(vtype=vtype, key=_mk_key(rng, ktype, True), value=mk_val())
    per = max(1, len(tokenizer(probe, add_special_tokens=False)["input_ids"]))
    out, seen = [], set(exclude)
    while len(out) < doc_size_tokens // per + 1:
        k = _mk_key(rng, ktype, True)
        if k in seen:
            continue
        seen.add(k)
        out.append(_NEEDLE.format(vtype=vtype, key=k, value=mk_val()))
    return " ".join(out) + " "


def _uuid(rng) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _interleave(rng, chains: list[list]) -> list:
    """Merge several ordered lists into one, randomly, preserving each list's internal order."""
    idx = [0] * len(chains)
    out = []
    while any(i < len(c) for i, c in zip(idx, chains)):
        live = [ci for ci, c in enumerate(chains) if idx[ci] < len(c)]
        weights = [len(chains[ci]) - idx[ci] for ci in live]
        ci = rng.choices(live, weights=weights)[0]
        out.append(chains[ci][idx[ci]]); idx[ci] += 1
    return out


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def make_niah_problem(task, corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    if task not in NIAH_TASKS:
        raise ValueError(f"Unknown niah task: {task!r}")
    if task == "niah_multi":
        return _make_niah_multi(corpus_tokens, tokenizer, doc_size_tokens, seed)
    if task == "vt_novel":
        return _make_vt_novel(corpus_tokens, tokenizer, doc_size_tokens, seed)

    rng = random.Random(seed)
    # Key/value types vary (was: word key -> number, always). RULER's uuid-keyed niah_multikey_3 was the
    # only uuid key the root had ever seen outside niah_multi's collect template, so it collected.
    ktype = rng.choice(["words", "words", "uuids"])
    vtype = rng.choice(["numbers", "numbers", "uuids"])
    vword = "number" if vtype == "numbers" else "uuid"
    needle_hay = rng.random() < _NEEDLE_HAYSTACK_FRAC
    key = _mk_key(rng, ktype, compound=needle_hay)
    value = _uuid(rng) if vtype == "uuids" else str(rng.randint(1_000_000, 9_999_999))
    filler = (_needle_filler(rng, tokenizer, doc_size_tokens, ktype, vtype, exclude={key}) if needle_hay
              else _filler(rng, tokenizer, doc_size_tokens, "novel"))
    needle = _NEEDLE.format(vtype=vtype, key=key, value=value)
    doc_text, cspans = _insert(filler, _boundaries(filler, 1, rng), [needle])
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    (ts, te), = _tok_spans(enc["offset_mapping"], cspans)
    spans = [(ts, te, 0, 1, needle.strip(), True)]
    return Problem(
        document_tokens=enc["input_ids"], question=_phrase(rng, _Q_SINGLE, vword=vword, key=key),
        gold_answers=[value], task=task, task_context=_NEEDLE_CONTEXT if needle_hay else _CONTEXT,
        grading_mode="qa_part",
        metadata={"family": "niah", "strategy_default": "binary", "task": task,
                  "answer": value, "key": key, "record_spans": spans, "k": 12,
                  "key_type": ktype, "value_type": vtype, "filler": "needle" if needle_hay else "novel",
                  # bare question, embedded verbatim in every subtask by BookQAOracle (no preface /
                  # answer-format tail — those conflict with the leaf protocol)
                  "q_core": f"What is the special magic {vword} for {key}?"},
    )


def _make_niah_multi(corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    rng = random.Random(seed)
    mode = rng.choice(["hidden", "hidden", "explicit", "multiquery", "multivalue"])
    ktype = rng.choice(["words", "words", "uuids"])
    vtype = rng.choice(["numbers", "numbers", "uuids"])
    fkind = _pick_filler_kind(rng)
    # Needle haystack only where the question names the key(s): hidden mode must collect every fact,
    # which over ~200 needles is not a 3K-budget problem (and RULER has no such variant).
    needle_hay = mode != "hidden" and rng.random() < _NEEDLE_HAYSTACK_FRAC
    if needle_hay:
        fkind = "needle"

    def mk_val():
        return _uuid(rng) if vtype == "uuids" else str(rng.randint(1_000_000, 9_999_999))

    # A uuid is ~20 tokens and the fold protocol restates the accumulator at every hop, so cap
    # the fact count whenever uuids are involved (measured: 5 uuid=uuid facts -> 3.5k-token node).
    heavy = ktype == "uuids" or vtype == "uuids"
    if mode == "multivalue":
        keys = [_mk_key(rng, ktype, needle_hay)]
        n_vals = rng.randint(2, 3) if heavy else rng.randint(2, 4)
        facts = [(keys[0], mk_val(), False) for _ in range(n_vals)]
        target_keys = keys
    else:
        if mode == "multiquery":
            n_keys = rng.randint(2, 3) if heavy else rng.randint(2, 4)
        else:
            n_keys = rng.randint(1, 3) if heavy else rng.randint(1, 6)
        keys = []
        while len(keys) < n_keys:
            k = _mk_key(rng, ktype, needle_hay)
            if k not in keys:
                keys.append(k)
        facts = [(k, mk_val(), False) for k in keys]
        target_keys = keys if mode == "multiquery" else [rng.choice(keys)]
        if mode == "hidden":
            facts.append((target_keys[0], None, True))
    rng.shuffle(facts)
    vword = "number" if vtype == "numbers" else "uuid"
    filler = (_needle_filler(rng, tokenizer, doc_size_tokens, ktype, vtype, exclude=set(keys)) if needle_hay
              else _filler(rng, tokenizer, doc_size_tokens, fkind, corpus_tokens))
    sentences = [
        (_QUERY_NEEDLE.format(key=k) if is_q else _NEEDLE.format(vtype=vtype, key=k, value=v))
        for k, v, is_q in facts
    ]
    doc_text, cspans = _insert(filler, _boundaries(filler, len(sentences), rng), sentences)
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [(ts, te, i, k, v, is_q)
             for i, ((ts, te), (k, v, is_q)) in enumerate(zip(_tok_spans(enc["offset_mapping"], cspans), facts))]

    # Question + gold + grading per mode.
    if mode == "hidden":
        question = _phrase(rng, _Q_HIDDEN, vword=vword)
        gold = [v for k, v, q in facts if not q and k == target_keys[0]]
        grading = "qa_part"
    elif mode == "explicit":
        question = _phrase(rng, _Q_SINGLE, vword=vword, key=target_keys[0])
        gold = [v for k, v, q in facts if k == target_keys[0]]
        grading = "qa_part"
    elif mode == "multiquery":
        ks = ", ".join(target_keys[:-1]) + f" and {target_keys[-1]}"
        question = _phrase(rng, _Q_MULTIQUERY, tails=_LIST_TAILS, vword=vword, ks=ks)
        gold = [v for k, v, q in facts]
        grading = "set"
    else:  # multivalue
        question = _phrase(rng, _Q_MULTIVALUE, tails=_LIST_TAILS, vword=vword, key=target_keys[0])
        gold = [v for k, v, q in facts]
        grading = "set"
    return Problem(
        document_tokens=enc["input_ids"], question=question, gold_answers=gold, task="niah_multi",
        task_context=_NEEDLE_CONTEXT if needle_hay else _MULTI_CONTEXT, grading_mode=grading,
        metadata={"family": "niah", "strategy_default": "binary", "task": "niah_multi",
                  "mode": mode, "key_type": ktype, "value_type": vtype, "filler": fkind,
                  "target_keys": target_keys, "answer": gold[0], "n_needles": len(facts),
                  "record_spans": spans},
    )


def _var_names(rng, n) -> list[str]:
    names: set[str] = set()
    while len(names) < n:
        names.add("".join(rng.choice(string.ascii_uppercase) for _ in range(5)))
    out = sorted(names)
    rng.shuffle(out)
    return out


def _make_vt_novel(corpus_tokens, tokenizer, doc_size_tokens, seed) -> Problem:
    """RULER-vt shape generalized: n chains (root literal + hops that copy the previous var),
    interleaved but each in order; the question targets one chain."""
    rng = random.Random(seed)
    qtype = rng.choice(["which_vars", "which_vars", "final_value"])
    # vt haystacks lean NOISE (RULER vt is noise-filled) and chains are often SPARSE (one chain of
    # 4-7 lines in 6k tokens), so most 400-token fold slices are empty — the regime where run 6's
    # fold chains broke (self-loop / early `none`). Still 2-3 chains half the time.
    fkind = rng.choice(["noise", "noise", "novel", "essay"])
    filler = _filler(rng, tokenizer, doc_size_tokens, fkind, corpus_tokens)
    # Keep the total bindings dict <= ~17 vars: 5-letter random names tokenize to ~4 tokens each,
    # and the fold accumulator is carried in the subtask + stated per node, so this bounds the
    # per-agent context under the 3000 budget (measured max ~2.5k).
    n_chains = rng.choice([1, 1, 2, 3])
    lens = [rng.randint(3, 6) + 1] + [rng.randint(2, 4) + 1 for _ in range(n_chains - 1)]  # vars per chain
    names = _var_names(rng, sum(lens))
    vals = rng.sample(range(10_000, 99_999), n_chains)
    chains, pos = [], 0
    for L, v in zip(lens, vals):
        ns = names[pos:pos + L]; pos += L
        chains.append([(ns[0], str(v), False)] + [(ns[i], ns[i - 1], True) for i in range(1, L)])
    recs = _interleave(rng, chains)
    # REASSIGNMENTS: with each var assigned once, collect-then-resolve would also work (RULER vt
    # is like that), and in run 5 the root did exactly that (binary "collect every assignment")
    # and resolved wrong. Re-binding a var later makes order genuinely matter: a copy made BEFORE
    # the re-binding keeps the old value. Gold is computed by threading the final record order.
    if rng.random() < 0.6:
        for _ in range(rng.randint(1, 2)):
            j = rng.randint(len(recs) // 2, len(recs) - 1)          # somewhere in the second half
            earlier = [n for n, _, _ in recs[:j]]
            name = rng.choice(earlier)
            if rng.random() < 0.5:
                recs.insert(j, (name, str(rng.randint(10_000, 99_999)), False))
            else:
                src = rng.choice([n for n in earlier if n != name] or earlier)
                recs.insert(j, (name, src, True))
    binding: dict[str, int] = {}
    for n, r, ref in recs:
        binding[n] = binding.get(r, 0) if ref else int(r)
    sentences = [(_VT_REF.format(name=n, rhs=r) if ref else _VT_LIT.format(name=n, value=r)) for n, r, ref in recs]
    doc_text, cspans = _insert(filler, _boundaries(filler, len(sentences), rng), sentences)
    enc = tokenizer(doc_text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [(ts, te, i, n, r, ref)
             for i, ((ts, te), (n, r, ref)) in enumerate(zip(_tok_spans(enc["offset_mapping"], cspans), recs))]

    t_val = vals[0]
    holders = sorted(n for n, v in binding.items() if v == t_val)
    bare = rng.random() < _VT_BARE_FRAC          # RULER-matched: no description, no gloss, no preface
    lead = _VT_PREAMBLE_BARE if bare else (_VT_PREAMBLE if rng.random() < _VT_GLOSS_FRAC else _VT_PREAMBLE_BARE)
    prefaces = [""] if bare else _VT_PREFACES
    if qtype == "which_vars":
        question = lead + _phrase(rng, _Q_VT_WHICH, tails=_LIST_TAILS, prefaces=prefaces, t_val=t_val)
        gold, grading, qvar = (holders or ["none"]), "set", None
    else:
        qvar = rng.choice([n for n, _, _ in chains[0]])
        question = lead + _phrase(rng, _Q_VT_FINAL, prefaces=prefaces, qvar=qvar)
        gold, grading = [str(binding[qvar])], "exact"
    return Problem(
        document_tokens=enc["input_ids"], question=question, gold_answers=gold, task="vt_novel",
        task_context=("" if bare else _VT_CONTEXT), grading_mode=grading,
        metadata={"family": "niah", "strategy_default": "left_fold", "task": "vt_novel",
                  "qtype": qtype, "filler": fkind, "n_chains": n_chains, "target_value": t_val,
                  "query_var": qvar, "record_spans": spans},
    )
