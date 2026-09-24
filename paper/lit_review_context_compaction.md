# Context compaction: a literature review

*Prepared 2026-09-23 for the ICLR 2027 submission. Companion to `lit_review_long_context_harnesses.md` (what enters the window) and `lit_review_native_long_context.md` (how big the window is). This review covers what **stays** in the window once it fills. Every arXiv identifier was verified against the arXiv API on the date above (title, first author, month; venue from the arXiv comment field where present). Harness behavior was read from source code at the commits listed in the source index, or from vendor documentation where the harness is closed source; constants in harness code change often, so every number here is a snapshot.*

## 0. Scope and taxonomy

**Compaction** is any operation applied *during* a task that shrinks what the model conditions on, typically because the accumulated context approaches a hard limit or a cost threshold. It differs from the two neighboring literatures: native long-context work makes the window larger, and harness design decides what enters the window in the first place. Compaction decides what survives once the window is full.

Three axes organize the field.

1. **Representation of what survives.**
   - *text*: the original tokens, pruned or masked, or a natural-language summary written by a model;
   - *soft tokens*: learned continuous vectors standing in for text (gist tokens, memory slots, beacons);
   - *KV-cache entries*: the model's own keys and values, evicted, merged, or quantized;
   - *weights*: the context distilled into parameters (LoRA adapters, trained KV "cartridges", test-time memories);
   - *external store with a pointer*: the content moves to a file or memory tool, and only a reference stays in context.
2. **Who decides what survives.** A fixed rule (keep the last *n* observations), a separate summarizer call with a hand-written prompt, a learned scorer (attention-based eviction), or the policy itself, trained to compact.
3. **Reversibility.** Most compaction is lossy and irreversible. A minority is *restorable*: the dropped content can be re-fetched because a pointer to it was kept.

| family | representation | decider | reversible | where it runs | production use |
|---|---|---|---|---|---|
| masking / dropping (§2) | text | rule | sometimes (pointer) | harness | universal, usually as the first tier |
| LLM summarization (§3) | text | prompted model | no | harness or provider API | the production default |
| trained compaction (§4) | text, or opaque | the policy | no | model | GPT-5.1-Codex-Max; research |
| soft compression (§5) | soft tokens | trained encoder | no | model internals | essentially none in agents |
| parametric (§6) | weights / trained KV | offline training | no | model internals | research |
| KV-cache compaction (§7) | KV entries | attention statistics | no | serving stack | serving systems; not exposed to API users |
| externalization (§8) | external store | model or harness | yes | harness | memory tools, `CLAUDE.md`, file systems |
| delegation (§9) | text returned by a child | the caller | n/a (never compacts) | harness | sub-agents; our work |

## 1. Why compaction exists

Three pressures, only one of which is the hard limit.

- **The limit.** An agent's trajectory grows with every tool call. Coding and research agents routinely produce millions of tokens per task, far beyond any window.
- **Degradation before the limit.** Accuracy falls as context grows well before the window is full [chroma2025contextrot, hsieh2024ruler, liu2024lostmiddle]; the native-context review (§9) collects the evidence. Anthropic's compaction documentation gives this as a reason to compact at all: "response quality degrades as a conversation grows." Compaction is therefore also a quality intervention, not only a capacity one.
- **Cost, via caching.** Agent contexts are append-heavy. Manus reports an average input-to-output ratio of about 100:1 and a 10× price difference between cached and uncached input tokens, and names the KV-cache hit rate "the single most important metric for a production-stage AI agent" [ji2025manus]. This creates a tension that runs through the whole field: *any edit to the middle of the context invalidates the cached prefix after it.* Anthropic's tool-result clearing exposes a `clear_at_least` parameter so that clearing only happens when it frees enough tokens to be worth the cache break; Codex trims "from the beginning to preserve cache (prefix-based)" during compaction. Compaction has to pay for the cache it destroys.

## 2. Masking and dropping: no model call

The cheapest compaction removes content by rule.

**Observation masking** replaces old tool outputs with a placeholder while keeping the actions that produced them. Anthropic calls tool-result clearing "one of the safest lightest touch forms of compaction" [anthropic2025contextengineering]. SWE-agent's `LastNObservations` history processor keeps the last *n* observations and replaces each earlier one with "Old environment output: (*k* lines omitted)" [yang2024sweagent]. The Anthropic API ships the same idea server-side as `clear_tool_uses_20250919`: once input exceeds a trigger (default 100,000 tokens) it replaces the oldest tool results with placeholders, keeping the 3 most recent tool-use pairs by default, with options to exempt named tools and to clear tool *inputs* too; a sibling strategy, `clear_thinking_20251015`, clears old reasoning blocks. Claude Code's documented behavior is to clear older tool outputs *first* and to summarize only if that is not enough. OpenCode prunes old tool outputs when at least 20,000 tokens are reclaimable, protecting the most recent 40,000 tokens of tool output, and caps each serialized tool output at 2,000 characters. Gemini CLI truncates function responses outside the verbatim tail to a 50,000-token budget before it summarizes.

**Masking is competitive with summarization.** Lindenbauer et al. compared the two directly in SWE-agent on SWE-bench Verified across five model configurations: observation masking *halves cost relative to the unmanaged agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization*; a hybrid of the two cut cost a further 7% and 11% relative to masking alone and summarization alone, with initial evidence that the result carries over to OpenHands [lindenbauer2025complexitytrap].

**Discarding can beat summarizing outright.** DeepSeek-V3.2's search agent triggers context management when usage exceeds 80% of the window and compares three strategies on BrowseComp [deepseekai2025deepseekv32]. Starting from 51.4 without management, *Summary* (summarize the overflowed trajectory and restart) reaches 60.2 while stretching trajectories to an average of 364 steps; *Discard-all* (drop all prior tool-call history) reaches **67.6**, comparable to parallel sampling at far fewer steps. A third strategy, *Discard-75%*, drops the oldest three quarters of tool history. On this task the lossiest strategy won.

**Restorable compression** is the principled middle ground: drop the content but keep a pointer that makes it recoverable. Manus drops web-page bodies but keeps the URL, drops file contents but keeps the path, and treats the file system as "the ultimate context: unlimited in size, persistent by nature, and directly operable by the agent itself" [ji2025manus]. Self-GC formalizes this: context objects (user turns, tool spans, skill state) are indexed, a side-channel planner proposes fold, mask, and prune actions, and the harness writes recoverable sidecars before committing at cache-safe boundaries; it prunes 43.95% of prefix tokens while leaving 84.85% of future continuations unaffected, against 54.55–69.70% for heuristic baselines [hao2026selfgc].

**The hidden cost of dropping.** Liu measures what task-completion metrics miss: in a controlled tool-using environment, compression leaves completion statistically unchanged while forcing the agent to *reacquire* dropped state. For GPT-5.5, completion moved from 80% to 85% (p = 1.0) while retrieval calls rose from 21.0 to 63.9 (p = .002); retrieval calls rose in all six model-regime comparisons [liu2026compressioncost]. Compaction that "works" by the headline metric can triple interaction cost.

## 3. LLM summarization: the production default

When masking is not enough, every major harness falls back to having a model summarize the history. The implementations differ in their details but have converged on a common design.

### 3.1 What each harness does

| harness | trigger | kept verbatim | summary form | notable |
|---|---|---|---|---|
| **Claude Code** (closed source; docs + observation) | automatic as the window fills; `/compact [focus]` manually | "your requests and key code snippets"; `CLAUDE.md` and auto memory are reloaded, not summarized | not documented; see below | clears old tool outputs before summarizing; a "Compact Instructions" section in `CLAUDE.md` steers what is kept; stops after repeated compactions that immediately refill ("thrashing") |
| **Codex CLI** | **90%** of the context window (configurable, clamped to 90%) | the user's messages, newest first, up to **20,000** tokens | free-form "handoff summary for another LLM" | three modes: local summary, *remote* server-side compaction returning an opaque item, and a token-budget reset with no summary at all; warns users that "long threads and multiple compactions can cause the model to be less accurate" |
| **Gemini CLI** | **50%** of the model's token limit | newest **30%** of history | XML `<state_snapshot>` (`overall_goal`, `active_constraints`, `key_knowledge`, `task_state`, …) written after a private `<scratchpad>` | *anchored*: a new snapshot must integrate the previous one; a second **"probe" pass** asks the model to check its own snapshot for omitted file paths, tool results, or constraints and regenerate; explicit prompt-injection guard |
| **OpenHands** | **240** events | the first **2** events plus a recent tail | sectioned: `USER_CONTEXT`, `TASK_TRACKING`, `COMPLETED`, `PENDING`, `CURRENT_STATE`, `CODE_STATE`, `TESTS`, `CHANGES`, `DEPS`, `VERSION_CONTROL_STATUS` | condensers are pluggable and composable into pipelines |
| **Aider** | history above a token budget (1,024 in the summarizer's constructor; set per model in practice) | a tail of about half the budget | prose, written in the user's voice ("I asked you…"), with "less detail about older parts" | *recursive*: summarizes the head, and recurses up to depth 3 if the result is still too large |
| **OpenCode** | overflow of the usable window | recent 25% of the usable window (2,000–15,000 tokens) | structured sections, "preserve exact file paths and identifiers" | two tiers: prune tool outputs first, then summarize |
| **Anthropic API** | a token threshold (`compact_20260112`) or on demand (beta `compact-2026-09-04`) | optionally recent turns, word for word | model-written summary; a custom prompt can replace the default | can run in the background while work continues |
| **OpenAI API** | `context_management` with `compact_threshold`, or `POST /v1/responses/compact` | — | an **encrypted compaction item**, opaque and "not intended to be human-interpretable" | the client passes the compacted window back instead of the transcript |

A source-code study of eleven coding agents reaches the same picture from the other direction: threshold-triggered compaction is used by seven of the eleven; Mini-SWE-Agent keeps an unbounded linear history with no compaction at all; Aider summarizes recursively; OpenHands' condensers are pluggable; and the others add variations such as compaction over a branching session tree (Pi), "lineage" compaction (Hermes), and two-tier reactive compaction (Mistral Vibe) [barbaste2026harnessengineering].

**Claude Code, observed.** Claude Code's summarization prompt is not published. One data point we can report first-hand: the session in which this review was prepared was compacted by Claude Code with `/compact` and a free-text focus instruction, and the resulting summary had nine numbered sections: *Primary Request and Intent; Key Technical Concepts; Files and Code Sections; Errors and fixes; Problem Solving; All user messages; Pending Tasks; Current Work; Optional Next Step.* This is one session's output, not documentation, but it shows the same design as the open-source harnesses: a fixed section checklist, the user's messages preserved nearly verbatim, and an explicit resumption point.

### 3.2 The convergent design

Harnesses built independently arrived at the same pattern:

1. **Mask first, summarize second.** Claude Code, OpenCode, and Gemini CLI drop or truncate tool output before calling the summarizer. This is exactly the hybrid that Lindenbauer et al. found cheapest [lindenbauer2025complexitytrap].
2. **Keep a verbatim tail**, and keep the user's own words. Codex keeps 20,000 tokens of user messages, Gemini the newest 30% of history, OpenCode 25% of the usable window.
3. **Force structure.** Sectioned summaries act as checklists: a section must be filled or explicitly left empty, so a file path cannot be silently dropped. Factory's "anchored iterative summarization" makes this its core argument [factory2025compression], and OpenHands' and Gemini CLI's schemas embody it.
4. **Update incrementally, anchored on the previous summary**, rather than re-summarizing a summary. Gemini CLI instructs the model to integrate the prior `<state_snapshot>`; Factory summarizes only the newly truncated span and merges it in.
5. **Frame the summary as a handoff to another model.** Codex prefixes the summary with "Another language model started to solve this problem and produced a summary of its thinking process."

### 3.3 Known failure modes

- **Detail erodes under repeated rewriting.** Zhang et al. name two failure modes of iterative context rewriting: *brevity bias*, which drops domain insight in favor of concise summaries, and **context collapse**, "where iterative rewriting erodes details over time"; their fix is localized delta updates instead of monolithic rewrites [zhang2026ace]. Codex's own user-facing warning about multiple compactions is the practitioner's version of the same observation.
- **Delegated edits corrupt content.** Across 52 professional domains, frontier models corrupt about 25% of a document by the end of a long editing workflow [laban2026delegate52]. Summarization is a repeated rewrite of the agent's memory, so this failure mode is to be expected.
- **Fold versus tree.** For book-length summarization, *incrementally updating* a running summary yields more detail but lower coherence than *hierarchically merging* chunk summaries [chang2024booookscore]. This is the fold-versus-tree trade-off appearing inside summarization itself. Aider's recursive summarizer and every threshold harness above are folds.
- **Evaluation is thin.** Most harness compaction is judged by end-task success, which Liu shows can hide a tripling of interaction cost [liu2026compressioncost]. Factory's probe-based evaluation, which asks recall, artifact, continuation, and decision questions about the compacted session, is a better design, but it is a vendor evaluating its own method: over 36,611 production messages its structured summarizer scored 3.70 overall, against 3.44 for Anthropic's compaction and 3.35 for OpenAI's [factory2025compression]. The most informative number is the one all three share: **artifact trail**, whether the summary still knows which files were touched, is the weakest dimension for every method (2.45, 2.33, 2.19 out of 5).

## 4. Trained compaction: the summarizer moves into the policy

The frontier is to stop hand-prompting a summarizer and train the policy to compact.

**In production.** OpenAI describes GPT-5.1-Codex-Max (November 2025) as its "first model natively trained to operate across multiple context windows through a process called compaction," working coherently over millions of tokens and on tasks running more than 24 hours [openai2025codexmax]. The implied contrast is with harness-side summaries that a model was never trained to consume. Cursor reports the first numbers for this approach [cursor2026selfsummarization]. Composer is trained to *self-summarize* when it runs out of context during RL, so one rollout becomes several generations chained by summaries, and "the final reward [is used] for all tokens produced by the model in the chain," which rewards summaries that kept what mattered with no separate summary-quality signal. Against a hand-written prompt of "nearly a dozen carefully worded sections" whose summaries averaged over 5,000 tokens, self-summary "consistently reduces the error from compaction by 50% … while using one-fifth of the tokens"; learned summaries average about 1,000 tokens. The baseline it beats is the sectioned-checklist design of §3.2, so this is also evidence that training can overtake the best hand-designed summary format. On the Anthropic side, recent Claude models have **context awareness**: the API injects the remaining token budget into the conversation, and the model is "natively trained" to use it, persisting to the end of a task rather than guessing how much room is left [anthropic2026contextwindows].

**In research**, trained compaction has split by what the model learns to write:

- **A running memory, overwritten each step.** MemAgent (RL, overwrite strategy) [yu2025memagent], MEM1 (a constant-size internal state learned by RL) [zhou2025mem1], InfoMem (an information-gain reward on the memory itself) [han2026infomem].
- **Periodic summaries in a ReAct loop.** ReSum invokes a summary tool to condense history into a compact reasoning state and trains with ReSum-GRPO, reaching 33.3% on BrowseComp-zh and 18.3% on BrowseComp-en from 1K training samples [wu2025resum]. SUPO derives a policy gradient that optimizes tool use and the summaries jointly, end to end [lu2025supo].
- **Folding at multiple granularities.** AgentFold learns, by SFT alone, to fold its trajectory either finely (condensing a single step) or deeply (abstracting a whole sub-task), reaching 36.2% on BrowseComp with a 30B-A3B model (ICLR 2026) [ye2026agentfold]. Context-Folding branches into a sub-trajectory and folds it back into a summary, trained with FoldGRPO [sun2025contextfolding]; FoldAct targets the stability of this training [shao2025foldact].
- **Learning the compaction rule rather than the weights.** ACON optimizes natural-language compression guidelines from failure analysis [kang2025acon]; AdaCoM trains an external manager that edits a frozen agent's context [yi2026adacom]; CompactionRL trains models to compact their own history mid-rollout [li2026compactionrl].
- **Choosing among strategies.** AgentSwing expands several differently managed branches at each trigger point and routes to the most promising, matching static strategies with up to 3× fewer turns [feng2026agentswing].

The representation is splitting along a visible line. Anthropic's compaction and almost all research methods produce **readable text**. OpenAI's API returns an **encrypted item**: the client cannot inspect what was kept, and its internal form is not disclosed.

## 5. Soft compression: learned tokens in place of text

A research line replaces the text with a small number of learned continuous vectors that the model conditions on.

- **Memory tokens across segments.** The Recurrent Memory Transformer passes memory tokens between segments of a long input [bulatov2022rmt].
- **Prompt compression into tokens.** Gisting trains a model to compress a prompt into a few "gist" tokens via a modified attention mask [mu2023gist]. AutoCompressors recursively compress segments into summary vectors that accumulate across a document [chevalier2023autocompressors]. The In-context Autoencoder learns to encode a context into memory slots that a frozen decoder can use [ge2024icae]. 500xCompressor pushes ratios much further [li2024compressor500x], and xRAG compresses each retrieved document to a single token [cheng2024xrag].
- **Compression inside the forward pass.** Compressed Context Memory compresses online interaction into a growing compressed state [kim2024ccm]; Activation Beacon condenses activations of each chunk into "beacon" tokens [zhang2024activationbeacon].

None of these appears in a production agent harness. They require a modified or co-trained model, the compressed state is unreadable, quality falls at high ratios, and the representations do not survive a change of model, which harnesses that switch models mid-session (Codex's compaction handles model changes explicitly) cannot tolerate.

## 6. Parametric compaction: context into weights

A further step moves the context into parameters. Temp-LoRA trains a temporary adapter on the text generated so far during long generation [wang2024templora]. Generative Adapter maps a context to adapter weights in a single forward pass [chen2024generativeadapter], and Text-to-LoRA generates adapters from a task description [charakorn2025texttolora]. LLoCO compresses a corpus offline and adds a LoRA to read it [tan2024lloco].

**Cartridges** is the strongest result in this family. It trains a small KV cache offline for a given corpus using *self-study*, a context-distillation objective on synthetic conversations about the corpus. Naive next-token training on the corpus is not competitive, but self-study matches in-context learning while using **38.6× less memory** and giving **26.4× higher throughput**, and extends effective context on MTOB from 128K to 484K tokens [eyuboglu2025cartridges]. The training cost is amortized across all queries about the same corpus, which suits a fixed codebase and does not suit an agent whose context changes every step. Titans' test-time memory, covered in the native-context review, is the same idea built into the architecture [behrouz2024titans].

## 7. KV-cache compaction: inference-level

Serving systems compact the KV cache directly. This literature is the largest by volume and the least visible to agent builders, because API users cannot touch the cache.

- **Eviction by attention statistics.** StreamingLLM keeps attention-sink tokens plus a recent window [xiao2023streamingllm]; H2O keeps "heavy hitters" by accumulated attention [zhang2023h2o]; Scissorhands relies on the persistence of token importance [liu2023scissorhands]; TOVA recasts decoder Transformers as multi-state RNNs and evicts the lowest-attention state [oren2024tova]; SnapKV selects positions using the prompt's own attention pattern [li2024snapkv]; FastGen profiles each head and applies a head-specific policy [ge2024fastgen]; PyramidKV and Ada-KV allocate budgets across layers and heads [cai2024pyramidkv, feng2025adakv].
- **Head specialization.** RazorAttention and DuoAttention keep a full cache only for "retrieval heads" and a short window for the rest [tang2024razorattention, xiao2025duoattention].
- **Merging and quantization.** MiniCache merges caches across layers [liu2024minicache]; KIVI quantizes keys and values to 2 bits [liu2024kivi]; KVQuant targets 10M-token inference [hooper2024kvquant].
- **Selection without eviction.** Quest, InfLLM, ShadowKV, and RetrievalAttention keep the full cache (often offloaded) and select what to load per query [tang2024quest, xiao2024infllm, sun2025shadowkv, liu2024retrievalattention]. These are sparse attention rather than compaction, and they are what the evidence below favors.

**The central result: query-conditioned compression breaks when the future query is unknown.** Most eviction methods choose what to keep by looking at the *current* query. SCBench evaluates methods over the full lifecycle of a shared, reused cache, and finds that "sub-O(n) memory methods suffer in multi-turn scenarios, while sparse encoding with O(n) memory … perform[s] robustly" [li2025scbench]. KVzip reports that query-aware eviction methods "suffer from performance degradation even at a 90% cache budget ratio under multi-query scenarios," and fixes this by scoring each KV pair by how well it lets the model reconstruct the original context, a query-agnostic criterion that cuts the cache 3–4× with negligible loss [kim2025kvzip]. Benchmarks of the whole family find that the savings come with task-dependent losses [yuan2024kvcompressionbench], and Bhaskar et al. find that aggregation-style tasks tolerate less compression than retrieval-style ones [bhaskar2025cacheme]. A recent ACL 2026 survey covers the system-level landscape [jiang2026kvsurvey].

This is the same problem harness compaction faces in text: **to compact, you must decide what to keep before you know what will be asked.** KV eviction fails in exactly the multi-turn, changing-question regime that agents live in, which is one reason harnesses compact in text instead.

The architectural versions of KV compaction — MLA's latent KV, DeepSeek's sparse and compressed attention, linear-attention states — are covered in the native-context review.

## 8. Externalization: memory stores

Externalization moves content out of the window into storage the agent can query, keeping it recoverable. MemGPT framed this as virtual-memory paging between the window and external storage [packer2023memgpt]; recursive summarization of dialogue memory is an early LLM-era instance [wang2023recursivesumm]; Mem0 and A-MEM are production-oriented and agentic memory systems [chhikara2025mem0, xu2025amem]. In current products, Anthropic's memory tool lets the agent record what it learns in files and read them back on demand, and is recommended *in combination with* compaction: "compaction keeps the active context small …, and memory preserves the information that must survive summarization." Claude Code reloads `CLAUDE.md` and auto memory on every session rather than trusting them to a summary, and Anthropic's context-engineering guidance calls this *structured note-taking*, citing a Pokémon-playing agent that kept precise tallies across thousands of steps and many context resets [anthropic2025contextengineering]. Manus's file-system-as-context is the same pattern [ji2025manus].

## 9. Delegation as compaction

The last family avoids compaction instead of performing it. A sub-agent explores in its own fresh context, and only a condensed result returns to the caller; the sub-agent's full trajectory is discarded. Anthropic's context-engineering guidance lists sub-agent architectures alongside compaction and note-taking as a technique for long horizons, noting that "each subagent might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work (often 1,000-2,000 tokens)" [anthropic2025contextengineering]. Claude Code's sub-agents work this way ("the subagent's tool calls stay out of your context, and Claude gets back a summary"), as do Anthropic's research system [anthropic2025multiagent], Recursive Language Models' sub-calls [zhang2025rlm], and Chain of Agents' communication units [zhang2024coa]. Cognition argues the other side for coding: parallel sub-agents make conflicting implicit decisions, and a dedicated compression model over a single shared trace is preferable [yan2025dontbuild].

## 10. What the evidence says

1. **Cheap compaction is competitive.** Observation masking matches LLM summarization at half the cost [lindenbauer2025complexitytrap]; discarding all tool history beats summarizing on BrowseComp, 67.6 versus 60.2 [deepseekai2025deepseekv32]. A model-written summary has to earn its cost, and often does not.
2. **Headline metrics hide the losses.** Compression can triple reacquisition work with no change in task completion [liu2026compressioncost]; iterative rewriting collapses detail [zhang2026ace]; long delegated workflows corrupt a quarter of a document [laban2026delegate52]; Codex warns its own users about repeated compaction.
3. **Structure preserves information.** Sectioned, anchored, incrementally updated summaries beat free-form rewrites [factory2025compression, zhang2026ace], and the open-source harnesses converged on this design independently.
4. **Deciding what to keep before the question is known is the core difficulty.** It defeats query-aware KV eviction in multi-turn use [li2025scbench, kim2025kvzip], and it is why every text summarizer is told to keep everything that *might* matter: goals, constraints, file paths, errors.
5. **Training is where the field is heading**, and the output format is diverging: readable text in Anthropic's API and in research, an opaque encrypted item in OpenAI's.

## 11. Where our work sits

Our harness has no compaction step. Every agent works within a fixed 3,000-token budget. The parent writes the child's subtask, *including the form of the state the child must return*, before the child starts. When the child finishes, its entire context is discarded and only its `\boxed{}` answer crosses the boundary. This is delegation (§9) made the only mechanism, and it changes each of the problems above.

- **What to keep is decided in advance, with the question in hand.** Compaction chooses what to keep after the window fills, without knowing what will be asked next, which is point 4's core difficulty. Our root states the contract (a keyed tally, an accumulator, a found-or-not flag) at the moment it splits the task, knowing the question. That is precisely the condition under which query-conditioned compression works.
- **The summary is specified, not improvised.** A child does not decide what to preserve; the contract does. That is what makes exact supervision possible: the scripted oracle knows the correct return value for every child.
- **Failure is visible.** When a contracted state is too large for the budget, the merge overflows and the episode ends (`eval-budget-state-ceiling`). The alternative failures documented in §3.3, silently dropped detail and reacquisition cost, do not show up in completion metrics at all.
- **The state stays readable text**, unlike soft tokens (§5), trained KV (§6), or an encrypted compaction item (§4), so every intermediate result in a trace can be inspected and checked.
- **The fold-versus-tree choice is explicit.** Summarization research found that incremental updating and hierarchical merging trade detail for coherence [chang2024booookscore], and every threshold harness is a fold. Our root chooses between the two per task, from whether the task depends on order.

The honest limitation: compaction serves open-ended interactive sessions in which the future request is genuinely unknown, and our decomposition requires the question up front. The two are complementary rather than competing. A long agent session could compact across turns and delegate within a turn, and our results bear on the second half of that.

## 12. Suggested related-work paragraph

*Context compaction.* Production harnesses shrink a filling context by masking old tool outputs and then summarizing with an LLM [anthropic2025contextengineering, barbaste2026harnessengineering]; masking alone matches summarization at half the cost [lindenbauer2025complexitytrap], and discarding tool history can outperform summarizing [deepseekai2025deepseekv32]. Iterative summarization erodes detail [zhang2026ace] and incurs costs that completion metrics miss [liu2026compressioncost]. Recent work trains the compaction step into the policy [openai2025codexmax, cursor2026selfsummarization, wu2025resum, ye2026agentfold, lu2025supo, sun2025contextfolding], while soft-token [mu2023gist, ge2024icae], parametric [eyuboglu2025cartridges], and KV-cache methods [zhang2023h2o, li2024snapkv] compress below the text level; the last degrade when later queries differ from the current one [li2025scbench, kim2025kvzip]. All of these decide what to keep *after* the context has grown. Our agents never compact: each parent specifies the form of its child's result before the child runs, so the summary is a contract written with the question in hand.

## Source index

**Already in `iclr2027_conference.bib`:** `chroma2025contextrot`, `hsieh2024ruler`, `liu2024lostmiddle`, `deepseekai2025deepseekv32`, `laban2026delegate52`, `yu2025memagent`, `zhou2025mem1`, `han2026infomem`, `sun2025contextfolding`, `kang2025acon`, `yi2026adacom`, `li2026compactionrl`, `packer2023memgpt`, `anthropic2025multiagent`, `zhang2025rlm`, `zhang2024coa`, `yan2025dontbuild`, `behrouz2024titans`.

**Verified arXiv entries** (all checked 2026-09-23; venue from the arXiv comment where given; ✓ = already in the bib under this key):

| key | arXiv | title (first author, date; venue) |
|---|---|---|
| li2023selectivecontext | 2310.06201 | Compressing Context to Enhance Inference Efficiency of LLMs (Li, 2023-10; EMNLP 2023) |
| jiang2023llmlingua | 2310.05736 | LLMLingua (Jiang, 2023-10; EMNLP 2023) |
| jiang2024longllmlingua | 2310.06839 | LongLLMLingua (Jiang, 2023-10; ACL 2024) |
| pan2024llmlingua2 | 2403.12968 | LLMLingua-2 (Pan, 2024-03; Findings of ACL 2024) |
| xu2024recomp | 2310.04408 | RECOMP (Xu, 2023-10) |
| li2025promptcompressionsurvey | 2410.12388 | Prompt Compression for LLMs: A Survey (Li, 2024-10) |
| bulatov2022rmt | 2207.06881 | Recurrent Memory Transformer (Bulatov, 2022-07; NeurIPS 2022) |
| mu2023gist | 2304.08467 | Learning to Compress Prompts with Gist Tokens (Mu, 2023-04; NeurIPS 2023) |
| chevalier2023autocompressors | 2305.14788 | Adapting Language Models to Compress Contexts (Chevalier, 2023-05; EMNLP 2023) |
| ge2024icae | 2307.06945 | In-context Autoencoder (Ge, 2023-07; ICLR 2024) |
| kim2024ccm | 2312.03414 | Compressed Context Memory (Kim, 2023-12; ICLR 2024) |
| zhang2024activationbeacon | 2401.03462 | Long Context Compression with Activation Beacon (Zhang, 2024-01) |
| li2024compressor500x | 2408.03094 | 500xCompressor (Li, 2024-08) |
| cheng2024xrag | 2405.13792 | xRAG (Cheng, 2024-05; NeurIPS 2024) |
| tan2024lloco | 2404.07979 | LLoCO (Tan, 2024-04; EMNLP 2024) |
| wang2024templora | 2401.11504 | Inference-Time Training Helps Long Text Generation (Wang, 2024-01; COLM 2024) |
| chen2024generativeadapter | 2411.05877 | Generative Adapter (Chen, 2024-11) |
| charakorn2025texttolora | 2506.06105 | Text-to-LoRA (Charakorn, 2025-06; ICML 2025) |
| eyuboglu2025cartridges | 2506.06266 | Cartridges (Eyuboglu, 2025-06) |
| liu2023scissorhands | 2305.17118 | Scissorhands (Liu, 2023-05) |
| oren2024tova | 2401.06104 | Transformers are Multi-State RNNs (Oren, 2024-01) |
| ge2024fastgen | 2310.01801 | Model Tells You What to Discard (Ge, 2023-10; ICLR 2024) |
| cai2024pyramidkv | 2406.02069 | PyramidKV (Cai, 2024-06) |
| feng2025adakv | 2407.11550 | Ada-KV (Feng, 2024-07; NeurIPS 2025) |
| tang2024razorattention | 2407.15891 | RazorAttention (Tang, 2024-07) |
| xiao2025duoattention | 2410.10819 | DuoAttention (Xiao, 2024-10) |
| liu2024minicache | 2405.14366 | MiniCache (Liu, 2024-05) |
| liu2024kivi | 2402.02750 | KIVI (Liu, 2024-02; ICML 2024) |
| hooper2024kvquant | 2401.18079 | KVQuant (Hooper, 2024-01; NeurIPS 2024) |
| tang2024quest | 2406.10774 | Quest (Tang, 2024-06; ICML 2024) |
| xiao2024infllm | 2402.04617 | InfLLM (Xiao, 2024-02) |
| sun2025shadowkv | 2410.21465 | ShadowKV (Sun, 2024-10) |
| liu2024retrievalattention | 2409.10516 | RetrievalAttention (Liu, 2024-09) |
| yuan2024kvcompressionbench | 2407.01527 | KV Cache Compression, But What Must We Give in Return? (Yuan, 2024-07) |
| li2025scbench | 2412.10319 | SCBench (Li, 2024-12; ICLR 2025) |
| kim2025kvzip | 2505.23416 | KVzip (Kim, 2025-05; NeurIPS 2025 oral) |
| jiang2026kvsurvey | 2607.08057 | System-Aware KV Cache Optimization survey (Jiang, 2026-07; Findings of ACL 2026) |
| wang2023recursivesumm | 2308.15022 | Recursively Summarizing Enables Long-Term Dialogue Memory (Wang, 2023-08; Neurocomputing) |
| chang2024booookscore | 2310.00785 | BooookScore (Chang, 2023-10; ICLR 2024) |
| yang2024sweagent | 2405.15793 | SWE-agent (Yang, 2024-05) |
| wang2025openhands | 2407.16741 | OpenHands (Wang, 2024-07; ICLR 2025) |
| lindenbauer2025complexitytrap ✓ | 2508.21433 | The Complexity Trap (Lindenbauer, 2025-08; DL4Code workshop, NeurIPS 2025) |
| zhang2026ace | 2510.04618 | Agentic Context Engineering (Zhang, 2025-10; ICLR 2026) |
| wu2025resum | 2509.13313 | ReSum (Wu, 2025-09) |
| ye2026agentfold | 2510.24699 | AgentFold (Ye, 2025-10; ICLR 2026 per ML Anthology) |
| lu2025supo | 2510.06727 | Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management (Lu, 2025-10) |
| shao2025foldact | 2512.22733 | FoldAct (Shao, 2025-12) |
| feng2026agentswing | 2603.27490 | AgentSwing (Feng, 2026-03) |
| hao2026selfgc | 2607.00692 | Self-GC (Hao, 2026-07) |
| liu2026compressioncost | 2608.16370 | What Does Context Compression Cost an Agent? (Liu, 2026-08) |
| barbaste2026harnessengineering ✓ | 2609.00006 | Harness Engineering: A Source-Code Study of Eleven Systems (Barbaste, 2026-07) |
| bui2026terminalagents | 2603.05344 | Building Effective AI Coding Agents for the Terminal (Bui, 2026-03) |
| chhikara2025mem0 | 2504.19413 | Mem0 (Chhikara, 2025-04) |
| xu2025amem | 2502.12110 | A-MEM (Xu, 2025-02; NeurIPS 2025) |

From the native-context review's index, also cited here (✓ where now in the bib): `xiao2023streamingllm` ✓ (2309.17453), `zhang2023h2o` ✓ (2306.14048), `li2024snapkv` ✓ (2404.14469), `bhaskar2025cacheme` (2506.17121).

**Industry and documentation sources** (no arXiv record; ✓ = already in the bib as a `@misc` entry):

| key | source |
|---|---|
| anthropic2025contextengineering ✓ | Anthropic, "Effective context engineering for AI agents," engineering blog, September 29, 2025 — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents |
| anthropic2026compaction | Anthropic, Claude API docs: *Compaction overview* and *Context editing* — https://platform.claude.com/docs/en/build-with-claude/compaction, …/context-editing |
| anthropic2026contextwindows | Anthropic, Claude API docs: *Context windows* (context awareness) — https://platform.claude.com/docs/en/build-with-claude/context-windows |
| anthropic2026claudecode | Anthropic, Claude Code docs: *How Claude Code works* — https://code.claude.com/docs/en/how-claude-code-works |
| openai2025codexmax ✓ | OpenAI, "Building more with GPT-5.1-Codex-Max," November 2025 — https://openai.com/index/gpt-5-1-codex-max/ |
| cursor2026selfsummarization ✓ | F. Cassano and S. Rush, "Training Composer for longer horizons," Cursor blog, March 17, 2026 — https://cursor.com/blog/self-summarization |
| openai2026compaction | OpenAI, API docs: *Compaction* — https://developers.openai.com/api/docs/guides/compaction |
| ji2025manus | Y. Ji, "Context Engineering for AI Agents: Lessons from Building Manus," Manus blog, 2025 — https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus |
| factory2025compression | Factory, "Evaluating Context Compression for AI Agents," December 16, 2025 — https://factory.com/news/evaluating-compression (formerly factory.ai) |

**Harness source code inspected** (default branch, commit, commit date):

| harness | repository | commit | files |
|---|---|---|---|
| Codex CLI | openai/codex | `5f371ba30a` (2026-09-23) | `codex-rs/prompts/templates/compact/{prompt,summary_prefix}.md`, `core/src/compact.rs`, `core/src/compact_remote_v2.rs`, `core/src/compact_token_budget.rs`, `protocol/src/openai_models.rs` |
| Gemini CLI | google-gemini/gemini-cli | `f50ba8608e` (2026-09-23) | `packages/core/src/context/chatCompressionService.ts`, `packages/core/src/prompts/snippets.ts` |
| OpenHands | OpenHands/software-agent-sdk | `5b36cacccc` (2026-09-23) | `openhands/sdk/context/condenser/llm_summarizing_condenser.py`, `…/prompts/summarizing_prompt.j2` |
| SWE-agent | SWE-agent/SWE-agent | `3ea751c087` (2026-07-16) | `sweagent/agent/history_processors.py` |
| Aider | Aider-AI/aider | `5dc9490bb3` (2026-05-22) | `aider/history.py`, `aider/prompts.py` |
| OpenCode | anomalyco/opencode (formerly sst/opencode) | `1d6c3c0e29` (2026-09-23) | `packages/opencode/src/session/compaction.ts`, `…/agent/prompt/compaction.txt` |

Claude Code is closed source; its behavior is taken from Anthropic's documentation plus the single-session observation in §3.1.
