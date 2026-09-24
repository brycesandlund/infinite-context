# Design: `session_events` — a training analogue for OOLONG-real (draft for review, 2026-09-24)

Goal: nothing in OOLONG-real is completely out-of-distribution, WITHOUT training on the eval. OOLONG-real stays
eval-only (Critical Role transcripts, CritRoleStats labels — sft.py refuses `oolong_real`).

## 1. What OOLONG-real actually asks (official `test` split, 6,072 questions)

| family | share | computational shape |
|---|---|---|
| count with a filter (character / player / roll type / natural value / crit = value in {1,20} / named spell) | 29.8% | filtered count |
| **cumulative total by the end of episode N** (optionally by character) | **19.1%** | prefix count up to a SEGMENT ORDINAL |
| **n-th spell / first k / last k spells (in episode N or this episode)** | **15.3%** | positional within a segment |
| **first / last spell (by character X) in EACH episode** — list | **9.9%** | positional, one answer per segment, in order |
| most / least common roll type / natural value / spell — **all ties as a list**; least-common with a threshold ("more than one roll") | 9.3% | full per-key tally, tie-list contract |
| total count | 3.6% | count |
| percentage of rolls with value N (rounded) | 3.6% | two counts |
| how many characters cast spell S | 3.6% | filtered DISTINCT (set of casters) |
| upcast: count, or unique list, of spells cast above their base level | 3.5% | filter needing a base-level fact; count / set |
| cantrips count | 1.8% | filter needing a base-level fact |
| first / last spell by EACH character, order-preserving list | 0.3% | positional per key |
| spells cast by more than one character | 0.1% | set-valued map (spell -> casters) |

**44.5% is order-dependent**, and the dependence is structural: episodes are delimited by UNNUMBERED markers
(`[START OF EPISODE]` … `[END OF EPISODE]`), windows hold non-consecutive episodes (e.g. [2, 16, 64]), and "episode N"
means the N-th episode IN THE WINDOW (verified: 1,844 questions use an ordinal ≤ #episodes, never the real number). A
leaf in the middle of the transcript cannot know which episode it is in — the segment ordinal is running state.

Leaf reality (read from the transcripts): events are spoken, often across two lines — GM "Make a persuasion roll." /
player "14." — with natural vs total values ("A 10. Plus my constitution… thirteen"); spells are named in speech ("I'm
gonna cast Protection from Poison on Keyleth"); many vocabulary hits are NOT events ("Roll good", "I can cast spells",
"What'd you roll?"). Lines are labelled by PLAYER; questions name CHARACTERS; the player->character mapping is a header
block before the transcripts (the official harness sends it, with everything else, as the system message; our adapter
puts it in task_context). Gold is fan-annotated (noisy) — our analogue is exact by construction.

## 2. Coverage against the current corpus

| OOLONG-real need | closest current analogue | status |
|---|---|---|
| filtered counts over records | labeled `count`, `author_label_count`, synth `count2` | covered — NEW surface only (events in dialogue) |
| per-key tally -> argmax/argmin | labeled `most_common` / `author_least` | partial — NEW: **tie-list answer**, **min-count threshold** |
| share of two counts | labeled `before_after` | partial — NEW: percentage of total |
| distinct count | synth `distinct` (fixed 4-key vocab) | partial — NEW: **filtered distinct over an open vocabulary** |
| set-valued map | — | NEW (0.1% of eval; low priority) |
| alias resolution (character -> player via the mapping in the system prompt) | — | **NEW** |
| multi-line records inside free dialogue + non-event distractors | niah (single-line facts), long_records (header-owned blocks) | **NEW** |
| unnumbered segments; "the N-th segment" | — | **NEW (drives 44.5% of the eval)** |
| positional: first / n-th / last k / first-last per segment | synth `first_exceed`, long `first_reach` (first index only) | **NEW** |
| prefix count to a segment boundary | — | **NEW** |

## 3. Principles

1. **Analogue, not copy.** Our own speakers, characters and filler; no Critical Role text, names or stats. Question
   wording in paraphrase families, not the eval's templates verbatim.
2. **Mechanical oracle.** Events are inserted by the generator, so gold is exact and leaves are scripted (the faithful
   regime); the leaf line shows the utterance(s) and the judgment.
3. **Minimal sufficient state** (audit 2026-09-24) and the existing filtered-leaf conventions.
4. **Strategy follows the question**: order-independent -> binary (existing root); segment-ordinal / positional ->
   left fold (existing root template, task-specific sentence two: "…the session a line belongs to is only known by
   counting the session markers before it…"). No third root commitment (ordered-binary merge) — sentence two is fragile.
5. **Length stays within the training ceiling** (≤ 14K target, budget x length jitter as in run 16): 1-4 short sessions
   per document. The eval's 33K-1.3M is length generalization, not trained.

## 4. Documents

- **Header:** player -> character mapping — in `task_context` most of the time (as our OOLONG-real adapter now does,
  matching the official system-message protocol), in the document for a minority so header-reading stays trained —
  in varied layouts ("Priya plays the character Vessa."
  / "Players: Priya (Vessa), Tomas (Brakka)…" / a small table), plus a GM/narrator line. Our own name pools.
- **Instruction paragraph -> task_context** (as the OOLONG adapters do), with prompt-diversity variants: full / lite /
  none.
- **Sessions:** 1-4 per document, delimited by markers drawn from a set — `[START OF SESSION]`…`[END OF SESSION]`,
  `=== session begins ===`…, `[BEGIN TRANSCRIPT]`…, and (decision 1) the eval's `[START OF EPISODE]`…. Mostly
  UNNUMBERED (the eval's regime); a minority numbered so "session 3" is also seen as a label.
- **Filler dialogue:** `Name: utterance` lines built from quoted speech + narration sentences in our Gutenberg novels
  (conversational register, license-clean), GM lines from narration. Per-line lengths matched to real transcripts.
- **Events** (the records), rendered in varied phrasings, some split across two lines:
  - rolls: GM prompt naming the type ("Give me a Perception check", "Roll initiative", "Dexterity save") + player reply
    with the value; natural vs total sometimes both ("17 — that's a natural 14 plus 3"); crits ("Nat 20!", "natural 1").
  - spells/abilities: "I cast Mage Hand", "I'm upcasting Cure Wounds at 3rd level", "Hex on the guard".
- **Distractors (non-events), ~as frequent as events:** questions ("What'd you roll?"), encouragement ("Roll well!"),
  hypotheticals ("if I cast Fireball we all die"), recaps ("last session you cast Shield"), rules talk ("can you
  cast?"). The leaf says why each is skipped.
- **Skins (decision 3):** tabletop RPG (closest; SRD-style spell/skill vocabulary) + 1-2 other event transcripts on the
  same abstract schema (actor, type, value, item, session) — e.g. a board-game night (dice sums, cards played) or a
  match commentary booth (shots by player, type, outcome) — so the model learns event aggregation over speaker
  transcripts, not D&D trivia.

## 5. Question families, strategies and minimal states

Binary (order-independent):

| family | minimal state | leaf line (sketch) |
|---|---|---|
| count, filtered by actor (player OR character -> resolved via the mapping in the system prompt; from the document header in the minority variant) / type / value / value-set / item | int | `- GM: "Give me a Perception check" / Tomas: "Nat 20!" → roll by Tomas, Perception, natural 20 (counts) → count=4` |
| percentage with value v | {v: n, total: n} | `→ roll, natural 13 (matches) → match=2, total=9` |
| most / least common (tie LIST; optional threshold "at least 2") | full per-key tally (needed) | `→ roll type Perception → Perception: 3` |
| distinct actors who used item S | set of actors, only for S | `→ cast by Vessa (Mage Hand) → casters={Vessa, Brakka}` |
| upcast / category (decision 2) | int, or set of items | `→ Cure Wounds at 3rd level, base 1 → upcast (counts)` |
| items used by >1 actor | item -> set of actors (needed) | — (0.1% of eval; include lightly) |

Fold (order-dependent; accumulator always carries the current SESSION ORDINAL):

| family | minimal accumulator |
|---|---|
| cumulative count to the end of session N (optionally filtered) | session=k, count=c (frozen once session N ends) |
| n-th / first k event in session N | session=k, list of ≤ k items (only while in session N) |
| last k events in session N / in the window | session=k, sliding window of k items |
| first / last event (by actor X) in EACH session — list | session=k, one item per session so far |
| first / last event by EACH actor, order-preserving | session=k, actor -> (item, position) |

Fold lines follow the existing fold conventions (delta-only lines, "→ accumulator = …" once per slice); a session
marker is itself a line: `- [START OF SESSION] → session=2 (count so far frozen at 14 for session 1)`.

## 6. Size, mix, budget

- One task `session_events`, ~150-200 traces: ~55% binary families (weighted toward filtered counts and tie-list
  argmax), ~45% fold families (mirroring the eval's 44.5%). That adds ~70-90 fold roots on a NEW surface (fold roots
  226 -> ~300, 22% -> ~28%) — the fold-prior increase, without more VAR chains.
- Budget x length jitter as run 16 (≤ 14K). State bounds: ≤ 4 sessions, ≤ 6 actors, ≤ ~25 event keys.
- Verification as usual: gold-grade + worst-node budget per family at doc 6K/14K; rendered roots/leaves for review.

## 7. Eval plan (unchanged policy: OOLONG-real is never trained on)

OOLONG-real `test`, stratified by question family and bucketed by #episodes (1 / 2 / 3 ≈ the paper's 55K / 118K /
175K); per-family scores; reference points: paper Table 4 and our GPT-5-mini reproduction (51.22 @ 1 episode). Budget
5K or 8K; MAX_NODES raised (175K: ~1,000-agent binary trees, ~440-hop folds). Plus the fold probe on RULER vt and on
session_events positional questions before any full eval.

## 8. Decisions needed

1. **Eval's exact markers** (`[START OF EPISODE]`) as one marker variant among several? (A document-format
   convention, like the harness placeholder we already train on — or too close to the eval?)
2. **Knowledge-dependent filters** (cantrips, "above its base level", 5.3% of the eval) need spell base levels the
   transcript never states. (a) real SRD spells, leaf states the base level from knowledge; (b) state levels in the text
   (mechanical, but eval-mismatched); (c) skip.
3. **Skins:** tabletop-only, or tabletop + 1-2 other transcript genres on the same schema (my recommendation).
4. **Positional questions:** fold (proposed; reuses the trained root) vs teaching an ordered-binary merge.
5. **Filler source:** Gutenberg dialogue + narration (proposed) vs another corpus.
