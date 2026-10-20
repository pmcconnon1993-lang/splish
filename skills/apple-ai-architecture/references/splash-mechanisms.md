# Why Splash works, and what can generalize

Scope: mechanisms, model adaptation and research directions. This chapter excludes any particular user setup. It uses static source from `incoai/splash` commit `be3b083c5f413673d44e44881042e875e674e333`, original research and the authors’ DFlash 2 description. No model, application, training job or benchmark was run. [Evidence manifest](splash-research-sources.json).

Splash combines three kinds of advantage: it makes each pass efficient, gets several useful output tokens from a target pass, and avoids repeating work already represented in valid cached state. These advantages interact. A faster matrix kernel can lose overall if its rounding reduces draft acceptance; a faster decoder barely helps a task dominated by uncached input processing. The reusable objective is **less wall time per correct completed task**, with explicit model and sampling semantics.

## 1. The arithmetic behind prefill and decode

Consider one dense projection, `Y = X Wᵀ`, where `X` has `M` token rows, input width `K`, and output width `N`. It takes approximately `2MNK` floating-point operations. Reading a `b`-bit weight matrix once costs approximately `NKb/8` bytes, before scales, metadata and padding. Ignoring activation traffic, its optimistic arithmetic intensity is therefore `16M/b` FLOPs per weight byte.

For four-bit weights, that is roughly 4 FLOPs/byte at one row, versus 32 at eight rows. The second computation performs more arithmetic but can reuse each loaded weight across more outputs. Actual reuse depends on tiles, caches and scheduling; this is a derivation, not a measured bandwidth or speedup.

**Prefill** knows the input tokens in advance, so it can apply projections to many token rows together. **Autoregressive decode** must select a token before the next unknown token is available, often giving dense projections too little work per weight read. A simplified lower bound is `time ≥ max(FLOPs / effective compute rate, bytes / effective bandwidth)`; launch, synchronization and non-overlapped work add time. Eight GB of traffic at an assumed effective 400 GB/s alone takes 20 ms. These hypothetical values illustrate a bound, not a Splash result.

Attention adds another cost. During decode, full-attention layers read historical K/V; that traffic grows with context length, KV heads, head dimension and storage precision. Prefill also computes interactions among prompt positions. Tiling and online normalization can avoid materializing a full score matrix in external memory without eliminating the attention arithmetic. This is the IO-aware principle established by [FlashAttention](https://arxiv.org/abs/2205.14135). Splash implements its own paged tile/split/reduce path with threadgroup score/probability tiles and row statistics; this observation does not equate its kernels with the original CUDA implementation. [Prefill attention source][attention]

For MoE, do not substitute total parameter count into a dense-model bandwidth estimate. Routed experts, shared experts and the union of experts selected across verification rows determine what work and weight traffic actually occur. Increasing rows may improve reuse or activate more experts. [Qwen hybrid dimensions][hybrid]

## 2. How the engine reduces the cost of a pass

**Allocation and layout.** Weight images are Shared Metal buffers written through their host-visible address. Section views refer into those allocations. There is no inherent need for a second discrete VRAM copy, but CPU/GPU ordering and memory bandwidth still matter. Quantized weights remain compact in storage; the kernel’s tile format determines how efficiently codes and scales become matrix operands. [WeightImages.cpp:19–24][weights]

**Stage close to compute.** Splash’s staged kernel decodes a small weight tile into threadgroup memory, performs `matmul2d`, and loads the next payload/metadata while the current multiply runs. This aims to overlap latency and reuse a tile across token rows. It is not a full-model dequantization into a second large allocation. [gguf_staged_tile.h:56–115][staged]

**Fuse where the producer and consumer agree.** A projection can include its residual or gate epilogue. Compatible gate/up projections share one staged pass. Some normalization/mixer producers emit the next operator’s preferred input representation, avoiding a preparation dispatch and intermediate traffic. Fusion is beneficial only if saved traffic/launches outweigh register pressure, scratch size, occupancy loss and reduced scheduling freedom. The source’s fixed policies select by shape, format and device family rather than universally choosing the largest tile. [LinearGguf.cpp:280–315, 379–417][linear] · [upstream fusion investigation][numerics]

**Submit useful groups of work.** Ordinary decode builds one graph containing the draft, target verification, policy, acceptance and draft-context commit before submission. The CPU does not wait after each small operator. Constraints introduce genuine host dependencies: draft proposals are needed to simulate grammar masks, while target forward work can proceed independently. Shared allocation does not make those dependencies disappear. [Runtime.cpp:2630–2655][decode-graph] · [Runtime.cpp:1935–2031][constraints]

These mechanisms generalize more readily than their constants. The reusable ideas are tiling, operand reuse, fusion, compact storage and fewer synchronization points; tile sizes, reduction order and dispatch plans require evidence for each architecture and workload.

## 3. DFlash, DFlash 2 and the inspected native contract

The [original DFlash paper, v2](https://arxiv.org/html/2602.06036v2), uses a lightweight block-diffusion drafter conditioned on hidden features from selected target layers. Those features are projected into each draft layer’s KV context. Masked positions are predicted together in one pass, and the target verifies them. Training freezes the target, uses randomly selected anchor positions and masked blocks, and constrains attention to prevent information leakage between blocks. Its experiments do not establish the performance of every runtime, target or block length.

The [DFlash 2 article](https://inco.ai/blog/dflash2/) adds two complementary mechanisms: a short dynamic convolution improves local communication inside the block, and a lightweight selector scores transitions among top candidates to choose a more coherent path. The expensive backbone remains parallel; walking a small precomputed candidate graph is cheap compared with another full model pass. The article is an author technical report, distinct from the original peer-reviewed paper.

The inspected Splash source explicitly requires `DFlash2DraftModel`, a block of eight, a 2,048-position draft window, matching capture layers and mask token, supported convolution geometry, selector rank and candidate count. It is not a generic adapter that accepts every DFlash checkpoint. [ModelDescriptor.mm:544–590][draft-config]

The code confirms two convolution stages around both attention and MLP work; it shares the target vocabulary projection. The selector builds predecessor/candidate edge scores, then selects a seven-token path and records its proposal probabilities. [DFlashDraft.cpp:215–295][draft-forward] · [sampling.metal:1215–1367][selector]

## 4. Walk through one anchor plus seven proposals

Assume a prompt has been processed and token `a` has been selected as the next anchor. It is selected output, but its target KV row has not yet been computed. The runtime distinguishes those facts; confusing them produces off-by-one cache and streaming bugs.

1. **Prepare.** Create draft rows `[a, MASK, MASK, MASK, MASK, MASK, MASK, MASK]`, with the correct logical/rotary positions and previously committed draft context. [Runtime.cpp:1545–1574][prepare]
2. **Draft.** Run the small network over all eight rows. The selector produces `[d1,…,d7]`; later candidate selection uses the chosen predecessor’s edge scores. For sampled requests, retain the actual conditional proposal probabilities `q_i`.
3. **Verify.** Run the target causally over `[a,d1,…,d7]`. Its row for `a` supplies the distribution for `d1`; the `d1` row supplies the distribution for `d2`, and so on. The final row can supply a bonus token. All input guesses are known, allowing batched projections without allowing future guesses to influence earlier causal distributions. [Runtime.cpp:2637–2644][decode-graph]
4. **Accept a prefix.** Suppose `d1,d2,d3` pass and `d4` fails. Discard the suffix beginning at `d4`; draw a corrected token `c` from the residual distribution described below. Do not independently keep later lucky guesses, because they were verified under the rejected prefix.
5. **Commit only valid history.** The causal model advances over `[a,d1,d2,d3]`; `c` becomes the next pending anchor. KV slots written for rejected rows remain outside the committed length and are overwritten later. GDN recurrent state uses retained-row metadata/tape, and the draft context commits the matching valid rows. Anchor streaming is accounted separately, so do not count `a` twice when comparing output rates. [paged-KV contract][kv-commit] · [QwenState.cpp:404–410][state-commit] · [Runtime.cpp:1850–1909][finalize]

If all seven proposals pass, the next anchor comes from the eighth verifier distribution. Stop tokens and output budgets can shorten retention. The graph still has the eight-row physical shape in this revision; a shorter remaining output budget does not proportionally reduce its computation. [Runtime.cpp:2602–2608][shape]

## 5. What probabilistic “exactness” means

At a valid prefix, let `p(x)` be the normalized target distribution **after** its requested temperature, filtering, penalties and constraints, and `q(x)` the actual proposal distribution. Sample `x` from `q`, accept with `min(1,p(x)/q(x))`, and on rejection sample from `r(x) ∝ max(p(x)−q(x),0)`. Accepted mass is `min(p,q)`; the rejection correction supplies the remaining positive mass, yielding `p`. This is the speculative-sampling construction in [Leviathan et al., Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).

Splash implements `u*q < p` acceptance and residual sampling over the target vocabulary. `q` is the selector’s conditional candidate distribution, not the raw draft logits before selection. A sparse top-16 proposal distribution can still produce any token in target support through correction; it does not turn the target into a top-16 model. Greedy mode instead retains the longest prefix matching target argmax. [acceptance source][acceptance] · [residual source][residual]

There are important boundaries to that claim:

- It preserves the specified target distribution, not the distribution of an unquantized or differently configured model.
- Distributional equality does not mean identical text for the same seed: speculative execution consumes randomness differently.
- Finite precision, changed reduction order and different verification batching can perturb logits. The mathematical guarantee assumes the verifier represents the same conditional target distributions as the reference.
- Token filters, grammar state, penalties, positions and partial-acceptance state must match the accepted history. A correct probability formula cannot repair an incorrect cache.
- It does not make the target more knowledgeable or prove task quality. It avoids replacing the target’s decisions with unchecked guesses.

Upstream recorded a reduction that passed numerical tests but altered acceptance enough to hurt throughput. Numeric closeness, distribution fidelity and useful speed are related but separate acceptance criteria. [Recorded investigation][numerics]

## 6. Why more compute can finish sooner—and when it loses

For engineering, use the cycle model

`latency per useful token ≈ (draft + verification + policy/commit + non-overlapped host time) / mean useful tokens per cycle`.

This extends DFlash’s draft-plus-verification model to include serving overhead. An illustrative target costs 18 ms per serial token; an eight-row verification costs 27 ms, drafting 5 ms and other cycle work 2 ms. If the average yield is five tokens, the result is `34/5 = 6.8 ms/token`, about 2.65× faster. At a yield of 1.5, it is `22.7 ms/token`, slower. The break-even yield is `34/18 ≈ 1.89`. These are invented arithmetic examples, not benchmarks.

A large bandwidth-limited target can amortize its expensive weight reads across several verified rows, while a relatively small draft adds little time. That can give it a greater **relative** speedup than a small target. It does not imply larger models are intrinsically faster, or that a parameter-count ranking predicts the outcome. A faster target kernel can even make fixed draft overhead a larger fraction of the cycle.

Speculation can lose when target compute is already saturated by concurrent requests; the draft’s memory footprint evicts valuable state; long-context verification repeatedly reads too much KV; acceptance drops on difficult or unfamiliar distributions; an MoE block activates many additional experts; output is too short to amortize setup; or constrained-policy/host work dominates. Longer blocks trade more potential yield against more verification, scratch and rejected work. The inspected source’s fixed eight-row contract means adaptive block size is an implementation/training project, not a free setting.

## 7. What “hyper-train another model” becomes technically

There are two separate projects: **adapt the runtime to execute a target efficiently**, and **train or obtain a compatible auxiliary drafter**. Kernel optimization usually changes execution code and layouts. Draft training learns to predict the target’s continuation using its hidden features. Neither is automatically training a better general-purpose target.

A proposed, testable draft-development recipe is:

1. Freeze an exact target revision and its tokenizer/embedding/head semantics. Define which hidden layers the drafter reads and how they are projected.
2. Choose a block/window architecture the intended runtime can execute. Match the target’s positions, vocabulary and special tokens; include DFlash 2 convolution/selector modules if that is the execution contract.
3. Build a representative training set of prompts and target continuations, including required domains, long context and structured outputs. Use the paper’s masked-anchor training concept with correct causal visibility. This is a proposal; this investigation did not validate a complete reproducible training package.
4. Evaluate the running target/draft pair, not only training loss. Measure acceptance by position, candidate recall, selector accuracy, draft milliseconds, verification milliseconds and memory. High candidate recall with poor selection suggests selector work; poor candidate recall suggests representation/training work.
5. Test held-out task correctness and distributional behavior, then compare total latency under fixed request policies. Reevaluate after target fine-tuning, tokenizer changes or numeric-format changes.

The expensive target weights can remain frozen while the smaller auxiliary model is trained. The evidence does not justify promising that every family admits an equally good draft or that the same trained draft transfers between related checkpoints.

## 8. Extending Splash to a new family

The present `TargetLayout` is a variant of two concrete Qwen layouts. The model interface offers lifecycle, state snapshot/restore, batching and telemetry boundaries, but dimensions alone do not describe an arbitrary architecture. [ModelDescriptor.hpp:18][target-layout] · [Model.hpp:573–653][model-interface]

| Adaptation | Required invariant |
|---|---|
| Target graph and loader | Correct layer order, normalization, activation, position encoding, attention mask and tensor interpretation; support operators before tuning them |
| Recurrent/hybrid state | Capture every state component needed to resume; partial acceptance must leave exactly the state of the accepted prefix |
| Draft and feature capture | Exact selected layers, hidden widths, context projection, vocabulary/head compatibility and trained block contract |
| Tokenizer and template | Identical text→token IDs, special tokens and stop semantics; same vocabulary size alone is insufficient |
| Quantized layout | Decode codes/scales/zeros correctly; validate packed layout and arithmetic against independent references |
| Vision or other modality | Correct encoder/projector, preprocessing, positions, token insertion and content identity; text-only success proves none of these |
| Cache identity | Include model/format/state layout and request-dependent inputs; equivalent text is insufficient if images, templates or semantics differ |
| Scheduler and memory | Budget actual weights, draft, state and scratch; preserve ownership across async work, cancellation and eviction |
| Kernel tuning | Valid masks/padding/barriers, supported shapes and numerics; end-to-end benefit after all overhead |

For a transformer with only ordinary attention, recurrent GDN state may disappear; for a different state-space or hybrid model it must be replaced, not ignored. A different vision tower may need a new operator graph. A new family might initially reuse generic reference operations and specialize its dominant shapes later. Those are design options, not supported features of the inspected revision.

An extension is credible only after checking layer/logit references, greedy traces, sampled distribution fixtures, rejection at every proposal position, repeated cached continuation, batch-width transitions, cancellation/restart and modality/tool cases. Then profile normal fused execution and compare latency, memory and task results. A microkernel win alone does not establish a family-level improvement.

## Research hypotheses worth isolating

- **Better conditional draft selection:** improve yield more than the selector adds cycle time; isolate candidate recall from path selection.
- **Family-specific verification kernels:** reuse weights across the actual verification width; measure occupancy and extra active-expert traffic.
- **State-aware replay placement:** retain checkpoints that avoid the most recomputation per byte, with correct hybrid restoration.
- **Adaptive speculation:** select a supported block length or bypass drafting using measured marginal cost and acceptance. This requires new contracts and validation in Splash.
- **Producer/consumer fusion:** remove materialized intermediates only where representation, lifetime and rounding invariants hold.

These mechanisms are grounded directions for experiments. No new family, trained draft, generalized speedup or implementation correctness is claimed by this chapter.

[attention]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/metal/kernels/prefill/paged_attention.metal#L3-L92
[hybrid]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/QwenHybridLayout.hpp#L15-L81
[weights]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/WeightImages.cpp#L19-L24
[staged]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/metal/kernels/common/gguf_staged_tile.h#L56-L115
[linear]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/ops/LinearGguf.cpp#L280-L417
[numerics]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/dev/benchmarks/remaining-decode-optimizations.md#L7-L22
[decode-graph]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Runtime.cpp#L2630-L2655
[constraints]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Runtime.cpp#L1935-L2031
[draft-config]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/ModelDescriptor.mm#L544-L590
[draft-forward]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/DFlashDraft.cpp#L215-L295
[selector]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/metal/kernels/decode/sampling.metal#L1215-L1367
[prepare]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Runtime.cpp#L1545-L1574
[kv-commit]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/metal/abi/PagedAttention.h#L31-L42
[state-commit]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/QwenState.cpp#L404-L410
[finalize]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Runtime.cpp#L1850-L1909
[shape]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Runtime.cpp#L2602-L2608
[acceptance]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/metal/kernels/decode/sampling.metal#L1403-L1430
[residual]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/metal/kernels/decode/sampling.metal#L682-L825
[target-layout]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/ModelDescriptor.hpp#L18
[model-interface]: https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Model.hpp#L573-L653
