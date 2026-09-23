# Native long-context models: a literature review

*Prepared 2026-09-22 for the ICLR 2027 submission. Every arXiv identifier below was verified against the arXiv API on that date (title, first author, and submission month confirmed); venues are given where a proceedings record exists. This review is the companion to `paper/lit_review_long_context_harnesses.md`: that one covers inference-time procedures around a fixed window, this one covers making the window itself longer.*

## 0. Scope and organization

**Native long-context modeling** means any change to the architecture, the positional scheme, the training recipe, or the serving stack whose purpose is to let a single forward pass attend over more tokens. The defining property, and the one that matters for our paper, is that the input is *resident*: every token is in the window at once, and the model is expected to attend to whatever part of it the task needs.

This literature is driven by three distinct costs, and most confusion about it comes from conflating them.

1. **Compute.** Self-attention is quadratic in sequence length [vaswani2017attention]. At 1M tokens the attention matrix alone is $10^{12}$ entries per head per layer.
2. **Memory.** Autoregressive decoding caches keys and values for every past token. The KV cache grows linearly with context and quickly dominates weights: for a 70B model at 128K tokens it runs to tens of gigabytes, which is why so much 2024–2026 work is about the cache rather than about attention.
3. **Generalization.** A model trained at 4K tokens does not automatically work at 128K. Positional encodings go out of distribution, and the data distribution contains few genuinely long-range dependencies to learn from.

Sections 1–4 cover approaches that attack compute (sparsity, linearity, recurrence); Section 5 covers memory (KV cache); Section 6 covers systems; Section 7 covers positional extension and data recipes, which attack generalization; Section 8 covers what the frontier actually ships; Section 9 covers the evidence that none of it closes the gap, which is where our paper enters.

## 1. Sparse and structured attention

The first response to quadratic cost was to compute only part of the attention matrix. Sparse Transformers factorized attention into strided and local patterns [child2019sparse]. Longformer combined sliding-window attention with a few global tokens, scaling linearly and targeting document-level NLP [beltagy2020longformer]. BigBird added random connections and proved the resulting attention is a universal approximator and Turing complete, which mattered because it showed sparsity need not cost expressiveness in principle [zaheer2020bigbird]. Linformer projected keys and values to a fixed low-rank dimension [wang2020linformer].

These were mostly bolted onto pretrained models or trained at modest scale, and the frontier largely passed them by, for a reason worth stating: a sparsity pattern fixed by the architect is a hardcoded decision about what matters, and the pattern that is right for one task is wrong for another.

The 2025 revival fixed exactly that by making sparsity **learned and hardware-aligned**. DeepSeek's Native Sparse Attention (NSA) runs three parallel branches — coarse compressed tokens, dynamically selected fine-grained blocks, and a local sliding window — combined by a learned gate, and is trainable end to end rather than applied post hoc; it reports up to 9x faster forward and 6x faster backward passes at 64K while matching or exceeding full attention on general, long-context, and reasoning benchmarks [yuan2025nsa] (ACL 2025 best paper). Moonshot's MoBA applies a mixture-of-experts formulation to blocks of context, letting each query attend to a learned subset of blocks, and is designed to switch between sparse and full attention [lu2025moba].

NSA and MoBA are the most direct architectural rebuttal to our thesis, and worth naming as such: they *train* the model to decide what part of a long input to look at. The difference is that their decision is made inside one forward pass over a resident sequence and is bounded by it, whereas a delegation decision is made in token space and composes recursively without bound.

## 2. Linear attention and state-space models

A second line removes the quadratic term entirely by giving the model a **fixed-size state**. Katharopoulos et al. showed that replacing the softmax with a kernel feature map turns a Transformer into an RNN with linear-time inference [katharopoulos2020lineartransformers]; Performers approximated softmax attention with random features and unbiased estimators [choromanski2021performer]. RWKV scaled an RNN formulation to Transformer-era sizes with parallelizable training [peng2023rwkv], RetNet proposed retention with parallel, recurrent, and chunkwise-recurrent forms [sun2023retnet], and Gated Linear Attention made the gated variant hardware-efficient [yang2024gla]. xLSTM revisited LSTMs with exponential gating and matrix memory [beck2024xlstm].

Mamba is the pivot: a selective state-space model whose parameters are input-dependent, giving linear-time sequence modeling with content-based reasoning and 5x higher inference throughput than a comparable Transformer [gu2023mamba]. Mamba-2 then established the structured state-space duality showing SSMs and attention are closely related, and made the layer substantially faster [dao2024mamba2].

The trade-off is not subtle and is the reason no frontier lab has shipped a pure SSM: a constant-size state cannot losslessly retain an arbitrarily long history, so recall of specific facts degrades where attention's growing cache does not. This is a real and empirically robust limitation. (A 2026 preprint attempted to formalize it as an "impossibility triangle" of efficiency, compactness, and recall, and classified 52 architectures accordingly; **it was withdrawn in August 2026 for a substantive error in the main theorem's proof and must not be cited.** The intuition remains sound, but no rigorous statement of it should be attributed to that paper.)

## 3. Hybrids, which is what actually shipped

The resolution the field converged on is to interleave a minority of full-attention layers among a majority of linear or recurrent ones, keeping exact recall where it is needed and paying linear cost elsewhere. Jamba pioneered this at scale, mixing Transformer and Mamba layers with MoE and reporting a ratio near 7:1 [lieber2024jamba]; Zamba offered a compact 7B hybrid [glorioso2024zamba]; MiniMax-01 combined lightning attention (a gated linear variant) with periodic full attention and MoE [minimax2025minimax01]; NVIDIA's Nemotron-H replaced roughly 92% of attention layers with Mamba-2 [nvidia2025nemotronh]. Bae et al. give the systematic study — layer-wise versus intra-layer fusion, ratios, placement — and find hybrids achieve lower peak memory, higher throughput, smaller caches, and, notably, *robust long-context retrieval* rather than degraded retrieval [bae2025hybrid].

For our purposes the relevant fact is that hybrids change the cost curve, not the semantics: the input is still resident, and the model is still expected to do the whole task in one pass.

## 4. Recurrence and architectural memory

A related family adds an explicit memory rather than a state. Transformer-XL introduced segment-level recurrence with relative positions, the ancestor of all of this [dai2019transformerxl]. Memorizing Transformers added a non-differentiable kNN lookup over a cache of past key-value pairs [wu2022memorizing]. Infini-attention combined a compressive memory with local attention in a single block, claiming bounded memory over unbounded input [munkhdalai2024infiniattention]. Titans learns to memorize at test time, updating a neural memory module as it reads [behrouz2024titans].

This family is the architectural mirror of the *fold* harnesses in the companion review: a running state updated left to right. The difference is where the state lives — in activations versus in tokens the model wrote — and therefore whether it is inspectable and whether the update rule is learned implicitly or demonstrated.

## 5. The KV cache: the binding constraint in practice

By 2024 the practical wall was memory, not FLOPs, and the largest recent literature is on compressing or evicting the cache. StreamingLLM identified **attention sinks** — models dump attention onto the first few tokens — and showed that keeping those plus a sliding window enables stable streaming over millions of tokens, though without true long-range recall [xiao2023streamingllm]. H2O evicts by accumulated attention score, keeping "heavy hitters" [zhang2023h2o]. SnapKV compresses by observing which positions the prompt's own attention heads already select [li2024snapkv]. The area has since proliferated — selection by retrieval heads, low-rank merging, quantization, hierarchical semantic caches — and now has its own ACL 2026 survey and dedicated benchmarks measuring the quality/throughput trade-off.

Bhaskar et al. give the sharpest result for our purposes: they ask how many KV entries are actually needed for effective long-context behavior and find the answer is task-dependent, with aggregation-style tasks tolerating far less compression than retrieval-style ones [bhaskar2025cacheme]. That is the same task-structure distinction our root makes, arrived at from the opposite direction.

## 6. Systems: exact attention, made to fit

A parallel line keeps attention mathematically exact and attacks the constant factors. FlashAttention made attention IO-aware by tiling to SRAM and never materializing the matrix [dao2022flashattention], with FlashAttention-2 improving work partitioning [dao2023flashattention2]. Ring Attention distributes blockwise attention across devices so context scales with device count, enabling near-infinite context in principle [liu2023ringattention]. PagedAttention (vLLM) applied virtual-memory paging to the KV cache, cutting fragmentation and raising throughput several-fold [kwon2023pagedattention].

This is the honest core of native scaling: no approximation, no lost recall, just engineering. It is also why million-token windows exist at all, and why the harness literature cannot claim native context is a dead end — it demonstrably works, at a cost.

## 7. Positional extension and data recipes

Making a short-context model long is now a standard post-training step. RoPE [su2021roformer] is the near-universal positional scheme, and ALiBi showed linear biases extrapolate beyond training length [press2022alibi]. Position Interpolation rescales RoPE frequencies to fit a longer window and extends LLaMA to 32K within 1,000 fine-tuning steps [chen2023positioninterpolation]. YaRN adds NTK-by-parts interpolation plus attention temperature scaling, reaching state of the art with fine-tuning on ~0.1% of pretraining data [peng2023yarn]. LongRoPE searches non-uniform interpolation progressively to reach 2M tokens from 256K training [ding2024longrope]. Xiong et al. give the frontier-scale recipe — attention base frequency adjustment plus continued pretraining on long data [xiong2023effectivelongcontext].

Data matters as much as positions. Fu et al. show 128K context is largely a *data engineering* problem, achievable with ~500M tokens of continued pretraining if the length distribution and domain mixture are right [fu2024dataengineering]. LongAlign addresses long instruction-following alignment [bai2024longalign]. Gao et al.'s ProLong gives the most careful controlled study of what actually helps [gao2024prolong]. An et al. attack "lost in the middle" directly with information-intensive training so the model uses all positions [an2024film].

## 8. What the frontier ships, and what it discloses

Claimed windows now run to 1M (Gemini 1.5, which reported near-perfect synthetic recall to 10M in ablations [geminiteam2024gemini15]), 10M (Llama 4 Scout [meta2025llama4], SubQ [dangel2026subq], Pokee-Isaac 28B [zhu2026pokeeisaac]) and 100M (Magic LTM-2-mini [magic2024ltm2mini]).

Disclosure is thin and asymmetric, and this is worth a sentence in the paper. Gemini 1.5 is a detailed technical report; Llama 4 and Magic are blog posts; Pokee-Isaac's report calls the model "non-decoder-only" in its abstract and then contains **no architecture section at all** — no parameter breakdown, no attention description, no training details — while reporting 93.3% on RULER at 10M and flat ~335 tok/s decode at that length on a single B200. Flat decode throughput at 10M is the signature of a state that does not grow with sequence length, i.e. some hybrid or recurrent design, but that is inference from a performance profile, not a disclosed fact. Cite these for their *claims*, never for an architecture.

## 9. The gap that will not close

The evidence that claimed context overstates usable context is now overwhelming and comes from many independent directions: position-dependent U-shaped accuracy [liu2024lostmiddle], RULER's finding that models fall well short of their claimed sizes once tasks go beyond single-needle retrieval [hsieh2024ruler], NoLiMa's collapse once literal lexical overlap is removed [modarressi2025nolima], BABILong's reasoning-in-a-haystack [kuratov2024babilong], HELMET's application-centric evaluation [yen2025helmet], LongBench v2's 503 reasoning questions where experts reach 53.7% [bai2024longbenchv2], Chroma's context-rot measurements [chroma2025contextrot], and OOLONG's aggregation tasks where no witness set exists at all [bertsch2025oolong]. Multiple 2026 analyses report effective utilization saturating far below the nominal window and degrading non-uniformly well before the documented limit.

Three summary works are worth having: the comprehensive survey of long-context language modeling [liu2025survey], the "Thus Spake Long-Context LLM" position piece [liu2025thusspake], and Xu et al.'s divide-and-conquer noise decomposition, which is the theoretical statement of why chunking can beat a stronger single-shot model [xu2025divideconquer].

## 10. Where our work sits relative to this literature

Our paper is not a competitor to native scaling and should not be written as one. The relationship to state:

- **Sparse attention (NSA, MoBA) trains a model to choose what to look at within one resident pass.** We train a model to choose what to look at *and* how to combine what it finds, in token space, recursively, with no resident copy of the document. Their decision is bounded by the window; ours composes without bound.
- **Hybrids and SSMs trade recall for a constant state.** We keep exact attention and make the *context* constant instead: 3,000 tokens per agent regardless of document length, with depth absorbing the growth. The constraint we hit is state size at the merge (see `eval-budget-state-ceiling`), which is the same constraint SSMs hit, arrived at explicitly and legibly rather than in activations.
- **Fold-style architectural memory (Infini-attention, Titans) learns an implicit update rule.** Our fold writes its accumulator in natural language, so the update is inspectable and can be supervised exactly by a scripted oracle.
- **KV-cache work finds that aggregation tolerates less compression than retrieval** [bhaskar2025cacheme] — independent confirmation, from the systems side, of the task distinction our root is trained to make.
- **Positional extension and data recipes are orthogonal and complementary.** They make the leaf better. Our harness is bounded by leaf quality, so every advance there raises our ceiling; nothing in our method competes with them.

The one-line framing: native scaling makes the window bigger, and the evidence of Section 9 is that bigger windows are not reliably usable windows. We fix the window at 3,000 tokens and train the model to manage what enters it, which turns a long-context problem into a short-context problem plus a decomposition policy.

## Source index

Verified arXiv identifiers, for bib entries not yet in `iclr2027_conference.bib`:

| key | arXiv | title (first author, date) |
|---|---|---|
| vaswani2017attention | 1706.03762 | Attention Is All You Need (Vaswani, 2017-06) |
| dai2019transformerxl | 1901.02860 | Transformer-XL (Dai, 2019-01) |
| child2019sparse | 1904.10509 | Generating Long Sequences with Sparse Transformers (Child, 2019-04) |
| beltagy2020longformer | 2004.05150 | Longformer (Beltagy, 2020-04) |
| wang2020linformer | 2006.04768 | Linformer (Wang, 2020-06) |
| katharopoulos2020lineartransformers | 2006.16236 | Transformers are RNNs (Katharopoulos, 2020-06) |
| zaheer2020bigbird | 2007.14062 | Big Bird (Zaheer, 2020-07) |
| choromanski2021performer | 2009.14794 | Rethinking Attention with Performers (Choromanski, 2020-09) |
| su2021roformer | 2104.09864 | RoFormer / RoPE (Su, 2021-04) |
| press2022alibi | 2108.12409 | Train Short, Test Long: ALiBi (Press, 2021-08) |
| wu2022memorizing | 2203.08913 | Memorizing Transformers (Wu, 2022-03) |
| dao2022flashattention | 2205.14135 | FlashAttention (Dao, 2022-05) |
| peng2023rwkv | 2305.13048 | RWKV (Peng, 2023-05) |
| zhang2023h2o | 2306.14048 | H2O: Heavy-Hitter Oracle (Zhang, 2023-06) |
| chen2023positioninterpolation | 2306.15595 | Position Interpolation (Chen, 2023-06) |
| sun2023retnet | 2307.08621 | Retentive Network (Sun, 2023-07) |
| dao2023flashattention2 | 2307.08691 | FlashAttention-2 (Dao, 2023-07) |
| peng2023yarn | 2309.00071 | YaRN (Peng, 2023-08) |
| kwon2023pagedattention | 2309.06180 | PagedAttention / vLLM (Kwon, 2023-09) |
| xiong2023effectivelongcontext | 2309.16039 | Effective Long-Context Scaling (Xiong, 2023-09) |
| xiao2023streamingllm | 2309.17453 | StreamingLLM / attention sinks (Xiao, 2023-09) |
| liu2023ringattention | 2310.01889 | Ring Attention (Liu, 2023-10) |
| gu2023mamba | 2312.00752 | Mamba (Gu, 2023-12) |
| yang2024gla | 2312.06635 | Gated Linear Attention (Yang, 2023-12) |
| bai2024longalign | 2401.18058 | LongAlign (Bai, 2024-01) |
| fu2024dataengineering | 2402.10171 | Data Engineering for 128K Context (Fu, 2024-02) |
| ding2024longrope | 2402.13753 | LongRoPE (Ding, 2024-02) |
| lieber2024jamba | 2403.19887 | Jamba (Lieber, 2024-03) |
| munkhdalai2024infiniattention | 2404.07143 | Infini-attention (Munkhdalai, 2024-04) |
| li2024snapkv | 2404.14469 | SnapKV (Li, 2024-04) |
| an2024film | 2404.16811 | Make Your LLM Fully Utilize the Context (An, 2024-04) |
| beck2024xlstm | 2405.04517 | xLSTM (Beck, 2024-05) |
| glorioso2024zamba | 2405.16712 | Zamba (Glorioso, 2024-05) |
| dao2024mamba2 | 2405.21060 | Transformers are SSMs / Mamba-2 (Dao, 2024-05) |
| gao2024prolong | 2410.02660 | How to Train Long-Context LMs Effectively (Gao, 2024-10) |
| behrouz2024titans | 2501.00663 | Titans (Behrouz, 2024-12) |
| minimax2025minimax01 | 2501.08313 | MiniMax-01 (MiniMax, 2025-01) |
| yuan2025nsa | 2502.11089 | Native Sparse Attention (Yuan, 2025-02; ACL 2025 best paper) |
| lu2025moba | 2502.13189 | MoBA (Lu, 2025-02) |
| liu2025thusspake | 2502.17129 | Thus Spake Long-Context LLM (Liu, 2025-02) |
| nvidia2025nemotronh | 2504.03624 | Nemotron-H (NVIDIA, 2025-04) |
| bhaskar2025cacheme | 2506.17121 | Cache Me If You Can (Bhaskar, 2025-06) |
| bae2025hybrid | 2510.04800 | Hybrid Architectures for LMs (Bae, 2025-10) |

Already in the bib: `geminiteam2024gemini15`, `meta2025llama4`, `dangel2026subq`, `zhu2026pokeeisaac`, `magic2024ltm2mini`, `liu2024lostmiddle`, `hsieh2024ruler`, `modarressi2025nolima`, `kuratov2024babilong`, `yen2025helmet`, `bai2024longbenchv2`, `chroma2025contextrot`, `bertsch2025oolong`, `liu2025survey`, `xu2025divideconquer`, `kamradt2023niah`.

**Do not cite:** arXiv 2605.05066 ("The Impossibility Triangle of Long-Context Modeling"), withdrawn August 2026 for an error in the main theorem's proof.
