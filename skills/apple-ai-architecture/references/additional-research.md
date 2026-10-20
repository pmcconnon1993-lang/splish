# Additional research for faster Apple-native LLMs

Reviewed 20 October 2026. This chapter adds thirteen selected studies to the [main synthesis](synthesis.md), [Splash analysis](splash-mechanisms.md), and [model co-design chapter](model-codesign.md). It is a bounded mechanism survey, not an exhaustive literature review. Its subject is architecture and implementation, not any person's hardware. No models were installed, trained or benchmarked. The [source ledger](additional-research-sources.json) records evidence depth and unresolved implementation claims.

The studies strengthen a common conclusion: **reduce the expensive part of the complete inference path, while preserving the required output behavior.** They offer different ways to do that. A drafter reduces sequential target passes; compression reduces traffic; attention kernels improve reuse; serving systems eliminate repeated work. Their benefits do not automatically multiply because one optimization changes the bottleneck faced by the next.

## 1. Predict more useful tokens per expensive target pass

### ReDrafter: make a small predictor exploit target features

Apple's ReDrafter uses an RNN conditioned on target hidden states, trained by distillation, to propose candidate continuations. Its paper combines beam search with dynamic tree attention to avoid evaluating duplicate prefixes. Apple reports an MLX/Metal implementation and up to 2.3× acceleration in its experiments; this is author evidence, not a result reproduced here. Adapting the method requires a compatible trained drafter, target features, vocabulary, verifier and cache updates. Beam width must be retuned for the backend: additional candidates increase acceptance opportunities but also verification work. [Paper](https://arxiv.org/abs/2403.09919v5), [Apple overview](https://machinelearning.apple.com/research/recurrent-drafter).

### EAGLE-3: train for token prediction rather than feature imitation

EAGLE-3 removes an earlier feature-prediction constraint, predicts tokens directly and combines target features from multiple layers. Its training-time test method improves how additional training data benefits the drafter. The transferable lesson is that a draft's learning objective can constrain acceleration even when the target remains unchanged. An Apple implementation needs the trained auxiliary network, access to the specified features, and efficient candidate verification; ordinary model conversion is insufficient. More accurate proposals still lose if collecting features or running the drafter costs more than the saved target passes. [Paper](https://arxiv.org/abs/2503.01840v3), [author implementation](https://github.com/SafeAILab/EAGLE).

### Medusa: share the backbone, then verify multiple heads' proposals

Medusa adds heads that predict future tokens from a shared backbone and verifies candidate trees. Medusa-1 trains heads with the backbone frozen; Medusa-2 jointly adapts the backbone and heads. This offers a smaller auxiliary-weight burden than maintaining an independent full draft model. Apple transfer requires trained heads, tree masks, efficient verification and state commitment. Crucially, Medusa distinguishes rejection sampling from its faster *typical acceptance* option: similar benchmark quality under typical acceptance does not mean the original sampling distribution is preserved. Head training and acceptance policy must therefore be recorded separately. [Paper, especially §2.3.1](https://arxiv.org/html/2401.10774v3), [author implementation](https://github.com/FasterDecoding/Medusa).

### LayerSkip: train early layers to serve as the draft

LayerSkip combines layer dropout and an early-exit loss so a model's earlier layers can make useful predictions. Self-speculation drafts with those layers, then verifies with the remaining layers while reusing activations and KV state. An Apple engine could avoid separate draft weights, but needs an adapted checkpoint and an execution path that preserves this reuse. Merely stopping an ordinary model early is a different, quality-changing operation. The paper's reported inference experiments use greedy decoding; they should not be presented as a reproduced guarantee for every sampling mode or runtime. [Paper](https://arxiv.org/html/2404.16710v4), [author implementation](https://github.com/facebookresearch/LayerSkip).

### Lookahead Decoding: spend spare arithmetic without a trained draft

Lookahead develops Jacobi-style parallel token updates into a pool of candidate n-grams, then verifies promising continuations. Its algorithm is designed to preserve output behavior without training or maintaining an auxiliary model. For Apple, the appeal is a controlled exchange: additional parallel work may reduce serial weight-reading passes. The implementation still needs specialized attention masks, candidate management and verification. It is most interesting when the target has spare compute relative to its traffic cost; on a saturated or heavily batched engine, the extra arithmetic may erase the gain. [Paper](https://arxiv.org/html/2402.02057v1), [author implementation](https://github.com/hao-ai-lab/LookaheadDecoding).

### Mirror: overlap the draft and target critical paths

Mirror proposes starting draft work from early target signals while the target suffix continues, with complementary correction paths and multi-token speculative streaming. Its distinct contribution is scheduling across heterogeneous accelerators rather than only improving draft accuracy. Apple transfer would require suitable trained draft/early-exit components, supported execution on both processors, asynchronous coordination and correct acceptance/state handling. Shared memory alone does not provide this schedule. Treat its reported speedups as unverified here: the paper's hardware description has an unresolved inconsistency recorded in the ledger, so this chapter uses Mirror as an architectural research direction. [Apple overview](https://machinelearning.apple.com/research/mirror), [paper](https://arxiv.org/html/2510.13161).

## 2. Reduce weight and sequence-state traffic

### AWQ: protect sensitive channels without mixed-precision exceptions

AWQ identifies important weight channels from activation statistics and rescales them before low-bit weight quantization. The equivalent rescaling protects sensitive values without retaining scattered weights in higher precision. It is calibration-based, with no gradient training required by the original method. For Apple, the useful pairing is calibrated weights plus efficient packing and quantized matrix kernels; the existing [co-design chapter](model-codesign.md) already records an MLX-LM AWQ route. Calibration quality, export format and task distribution still matter. TinyChat's published speedups are implementation-specific evidence, not a speed promise for an arbitrary MLX conversion. [Paper](https://arxiv.org/abs/2306.00978v6), [author implementation](https://github.com/mit-han-lab/llm-awq).

### QuaRot: change the basis to make low precision easier

QuaRot rotates model representations to spread activation outliers, making weights, activations and KV state easier to quantize. The unquantized transformation preserves the represented computation; subsequent quantization introduces approximation. This expands the design space beyond weight-only compression. Apple transfer requires the compatible transformed model, any runtime rotations and efficient low-bit operators; a file labeled four-bit does not establish those capabilities. Evaluate the cost of rotation, packing and accumulation together. Its quality results on studied models do not imply that every architecture supports the same transformation or that every Apple backend benefits. [Paper](https://arxiv.org/abs/2404.00456v2), [author implementation](https://github.com/spcl/QuaRot).

### KIVI: keys and values need different compression policies

KIVI observes different statistical structure in attention keys and values, using per-channel key quantization and per-token value quantization. Its tuning-free two-bit cache method targets the state that grows with sequence length and concurrency. An Apple port would need cache packing, scale metadata, efficient attention consumption and incremental updates; weight quantization alone does not implement it. Lower cache traffic can help long contexts, while conversion overhead can dominate short ones. The authors' throughput gains partly follow from fitting larger batches, so they must not be read as equal single-request latency improvements. [Paper](https://arxiv.org/abs/2402.02750v2), [author implementation](https://github.com/jy-yuan/KIVI).

### LLM in a flash: make necessary storage reads fewer and larger

Apple's study keeps model weights in flash when DRAM is insufficient, predicts needed neurons, reuses recently active rows and bundles reads into larger contiguous transfers. It turns sparsity into an I/O policy rather than assuming scattered reads are cheap. Transfer requires a suitable sparse or adapted model, trained activation predictors, a storage layout and a managed cache. Predictor errors can change quality. This addresses capacity-limited inference: improvements over naive loading do not establish an advantage over a fully resident model. The enduring lesson is to optimize transferred bytes *and* transfer shape. [Paper](https://arxiv.org/html/2312.11514v3).

## 3. Improve execution and avoid repeated computation

### FlashAttention-2: memory savings still need good work partitioning

FlashAttention-2 builds on exact tiled attention by reducing non-matrix operations, exposing more parallel work and reducing shared-memory communication. It shows why avoiding a large intermediate tensor is only the first step: the remaining kernel can still waste execution resources. Apple transfer requires Metal-specific choices of threadgroups, SIMD work, tile sizes and on-chip storage; CUDA warp arrangements are not portable settings. No checkpoint training is required. Attention semantics are retained, subject to numerical differences, but full attention's underlying arithmetic remains. Published A100 utilization is evidence for that implementation, not an Apple utilization target. [Paper](https://arxiv.org/abs/2307.08691v1), [author implementation](https://github.com/Dao-AILab/flash-attention).

### PagedAttention: control fragmentation and share valid state

PagedAttention maps logical KV sequences onto physical blocks, allowing state to grow without reserving large contiguous regions and supporting sharing across requests. The main opportunity is better use of cache capacity, which can admit more useful concurrent work. An Apple engine needs a block allocator, sequence mappings, ownership rules and attention kernels that consume that layout; no model training is required. This is application-managed KV paging, not operating-system swap. Capacity and throughput benefits may be large even when a single isolated request becomes only slightly faster, or pays additional addressing overhead. [Paper](https://arxiv.org/abs/2309.06180v1), [vLLM implementation](https://github.com/vllm-project/vllm).

### SGLang: expose program structure so the runtime can reuse work

SGLang combines a language-model programming interface with runtime optimizations, including radix-tree KV reuse and faster constrained decoding through compressed state machines. Repeated prefixes and structured generation become optimization opportunities visible above individual kernels. For Apple, the transferable design is a server/program contract that retains reusable state and supplies generation constraints; it does not require training new weights. Cache keys must represent the actual model and input state, and constrained output intentionally changes allowed continuations. Gains depend on prefix repetition and program structure, so a warmed agent workflow and a unique cold prompt need separate evaluation. [Paper](https://arxiv.org/abs/2312.07104v2), [author implementation](https://github.com/sgl-project/sglang).

## What the ReDrafter source adds beyond its paper

The actual [MLX generation entry point](https://github.com/apple-aiml-research/ml-recurrent-drafter/blob/main/recurrent_drafting/mlx/cmd/generate.py) constructs `ReDrafterModel`; its [generation implementation](https://github.com/apple-aiml-research/ml-recurrent-drafter/blob/main/recurrent_drafting/mlx/recurrent_drafting.py) calls the drafter, verifies candidate beams, chooses a valid prefix and updates state. The ledger records inspected blob identities. This establishes a concrete implementation route, not current runtime compatibility.

Two limits materially affect reuse. The repository [README](https://github.com/apple-aiml-research/ml-recurrent-drafter/blob/main/README.md) warns that its non-greedy implementation may not exactly match the target distribution. Also, the inspected MLX `_verify_candidates` function has a TODO to enable tree compression and position IDs. Consequently, the paper's dynamic prefix-deduplication claim should not be attributed without qualification to that MLX function.

Its [MLX experiment notes](https://github.com/apple-aiml-research/ml-recurrent-drafter/blob/main/recurrent_drafting/mlx/experiments/README.md) provide operational lessons: materialize lazy model parameters before timing; avoid unnecessary scalar extraction that forces synchronization; and measure individual stages as well as whole generation. Those older experiments preferred much narrower beams on Apple GPUs than on their CUDA comparisons. The general lesson is to retune branching for the complete backend and workload, rather than copy a successful setting from another accelerator. These notes describe a different experiment from the paper's headline maximum; neither was reproduced for this study.

## Conclusions that change an implementation decision

| Observed limiting cost | Research family to investigate | Necessary evidence before claiming a win |
| --- | --- | --- |
| Many sequential target passes | ReDrafter, EAGLE-3, Medusa, LayerSkip, Lookahead | Whole-cycle time divided by valid emitted tokens, with declared acceptance semantics |
| Draft itself dominates the critical path | Mirror-style overlap | Actual concurrent execution, synchronization/transfer costs and correct state handling |
| Weight traffic | AWQ; QuaRot where operators permit | Matched-quality, matched-format end-to-end comparison |
| Long-context KV traffic or capacity | KIVI and PagedAttention | Total cache footprint, attention time and task quality at fixed concurrency |
| Attention kernels underuse available compute | FlashAttention-2 design principles | Actual model shapes, kernel correctness and complete-model timing |
| Repeated prefixes or structured calls | SGLang design principles | Cold and warm workflow latency with equivalent valid output |
| Weights cannot remain resident | LLM in a flash | Storage traffic, predictor quality and a relevant capacity-constrained baseline |

Three deductions connect these results. First, **changing precision changes the optimal speculation policy**: a faster target can make drafting overhead more prominent, while a different numerical target can alter acceptance. Second, **compression can buy latency indirectly through concurrency**, so throughput, isolated latency and queueing latency must remain distinct. Third, **training is most useful when it exposes a concrete execution opportunity**—better proposals, reliable early exits, low-bit robustness or predictable sparsity—and the runtime actually implements that opportunity. These are engineering deductions from the studies, not additional benchmark findings.

Exact speculative sampling, greedy agreement, approximate acceptance and quantization quality are separate contracts. Exact sampling relative to a quantized or adapted target does not recover the original unmodified model's distribution. Record the chosen target, sampling policy and numerical format before comparing speed. No numerical speedups in this chapter should be combined into a predicted multiplier.
