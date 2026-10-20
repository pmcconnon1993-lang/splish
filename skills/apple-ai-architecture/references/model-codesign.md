# Designing and adapting models for Apple execution

Research reviewed 19 October 2026, America/Chicago. Scope: general architecture, training, compression and execution design. This report makes no claim of an executed training run or measured Apple speedup. Primary-source identities and evidence limits are recorded in [training-sources.json](training-sources.json).

## What “hyper-training a model for Apple” can mean

The useful interpretation is **hardware-aware model adaptation and co-design**: train or adapt a model so that its mathematical structure, numerical format and generation behavior work well with the target runtime and processors. “Hyper-training” is not a single established technique. More training alone does not automatically improve hardware efficiency.

There are three connected choices:

- **Model:** how much work and state an answer requires—layers, widths, attention, expert routing, image tokens and output tokens.
- **Training:** what quality survives after compression, structural changes or cheaper prediction methods.
- **Execution:** whether kernels, memory layouts, batching and verification turn that smaller workload into lower latency and energy.

An existing checkpoint can often be adapted without starting over. Quantization, distillation, trained pruning, draft-model training and selected attention conversions form a spectrum. At the other end, changing the tokenizer, attention/state architecture and layer layout together approaches training a new model. Neither path requires “manually unifying” RAM and GPU memory. MLX already exposes shared arrays to supported CPU/GPU devices; the remaining challenge is efficient computation and synchronization. [MLX design](https://github.com/ml-explore/mlx/blob/f41bec734ac9fc96723ab7d4e64e385796f15561/README.md).

Apple's own research demonstrates that this is a real design discipline. Its 2025 model report describes 2-bit quantization-aware training for the on-device model, cross-layer KV sharing, and a server architecture with parallel transformer tracks and MoE. It also describes GPU texture-decompression hardware used for compressed server weights. These are concrete architecture/runtime choices, not proof that every technique is available as a public MLX conversion command. [Apple's model-design overview](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates).

## The software layers have different jobs

**MLX** supplies arrays, automatic differentiation, neural-network/optimizer components, lazy graph evaluation, and CPU/GPU execution. It can implement and train new architectures. **MLX-LM** supplies language-model implementations, checkpoint conversion/quantization, generation, and fine-tuning built on MLX. **oMLX** is a serving application: API translation, model lifecycle, scheduling/batching and reusable prompt state. A serving improvement can reduce repeated work without changing weights. A trained model can become cheaper without changing the server API. Splash is an alternative inference engine whose supported formats and model structures must match the candidate; it is not a general-purpose trainer. [MLX-LM](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/README.md), [oMLX routing](https://github.com/jundot/omlx/blob/c4605a2dcdf03bf8cb58594d34fb2b0089e60304/omlx/engine_pool.py#L2084).

This separation matters when choosing an experiment. A numerical representation can be trainable in MLX but unsupported by an optimized inference kernel. Conversely, an engine may execute an existing checkpoint faster without any training. The deployable artifact is the model **plus** its tokenizer/template, format, supported operators and execution path. [MLX quantized matrix multiplication contract](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html).

## Adaptation options, from smaller changes to structural changes

| Method | What it changes | Existing checkpoint? | Principal cost or limitation |
|---|---|---|---|
| Post-training quantization (PTQ) | Numerical weight representation; sometimes calibration scales/clipping | Usually | Accuracy and kernel-format sensitivity |
| QAT / quantization-aware distillation | Learns under low-precision behavior or from a teacher | Yes, often as a later training phase | Training compute, activation/optimizer memory, export compatibility |
| LoRA / task fine-tuning | Behavior through learned low-rank updates | Yes | Usually retains the base model's size and core work |
| Smaller student distillation | Transfers behavior into a cheaper model | Yes; student can be independently initialized or pretrained | Teacher computation/data, capability loss, distribution shift |
| Structured pruning plus recovery | Removes layers, heads, channels or experts | Yes | Recovery training and valid kernel-friendly dimensions |
| GQA or hybrid conversion | Changes attention/state structure | Selected architectures | Substantial adaptation; cannot just reinterpret tensors |
| Draft or MTP training | Improves proposals or multiple-token prediction | Often | Runtime verification/rollback, acceptance rate, extra state |
| New model co-design | Joint architecture, tokenizer, precision and objectives | Can reuse components; often needs large-scale training | Largest research/data/compute burden |

The table is a decision framework, not a claim that a single tool implements every row.

### PTQ, QAT and learned quantization

PTQ starts with trained weights and reduces their representation. Calibration methods can account for observed activations or layer sensitivity. QAT puts quantization effects inside training, typically using approximate gradients for the discrete rounding operation, so weights can adapt. It can begin from a pretrained checkpoint; it does not necessarily require pretraining from zero. It is also not the same as doing all training arithmetic and optimizer storage at the final inference bit width.

Apple's compute-allocation study explores models from 86 million to 2.2 billion parameters and finds that the useful fraction of QAT depends on the training budget, bit width and model scale. Its findings argue against a universal rule such as “always finish with 10% QAT.” They do not establish a universally optimal schedule for much larger models, every dataset or every hardware format. [Compute-Optimal Quantization-Aware Training](https://arxiv.org/html/2509.22935v2).

MLX-LM's inspected learned-quantization guide includes AWQ, GPTQ, dynamic precision allocation and Distilled Weight Quantization (DWQ). DWQ is particularly relevant to adapting existing models: the source unfreezes scales/biases for eligible affine-quantized modules below eight bits and minimizes a teacher/student KL-divergence loss, along with other trainable parameters. This is a specific supported method, not evidence of a universal full-weight QAT trainer or arbitrary architecture distillation. Calibration data must resemble the tasks and sequence lengths that matter. [Learned quantization](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/LEARNED_QUANTS.md), [DWQ implementation](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/quant/dwq.py#L68).

A smaller bit count is not sufficient evidence of faster execution. Group metadata, packing/unpacking, supported matrix shapes, accumulation precision and available kernels all matter. A format that forces dequantization into a large temporary matrix can lose its expected benefit. Weight quantization, activation quantization and KV-state quantization are distinct experiments and need separate quality checks.

### Distillation and LoRA do different things

Distillation teaches a student to reproduce useful teacher behavior, using generated examples, output distributions or intermediate representations. A smaller student offers an architectural route to lower work per token. It can also learn a particular tool protocol or task more reliably, reducing retries. Teacher mistakes and narrow training data can transfer too; a good specialist is not automatically an equivalent general model.

LoRA freezes the base model and learns low-rank changes. MLX-LM documents LoRA, DoRA and full fine-tuning, including quantized-base LoRA. Its adapter implementation adds low-rank matrix products; fusion merges the update into the base weight shape. Thus LoRA can improve task success or answer brevity, but by itself does not remove the base layers, narrow their matrices or shrink their KV cache. Unfused adapters can add execution work. QLoRA reduces the training burden of adapting a quantized base; it is not synonymous with QAT. [MLX-LM fine-tuning](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/LORA.md), [adapter operations](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/tuner/lora.py#L11), [original LoRA paper](https://arxiv.org/abs/2106.09685).

### Pruning must remove executed work

Unstructured pruning sets selected individual weights to zero. A dense matrix kernel may still load and multiply those entries. Speed requires a sparse representation and kernel that benefits from the particular pattern and density. Structured pruning removes whole channels, heads, layers or experts, creating smaller operations that ordinary kernels can often execute. Even then, awkward dimensions or lost batching efficiency can defeat the theoretical savings.

NVIDIA's Minitron work demonstrates pruning width/depth and using distillation to recover capability. This establishes a model-adaptation method, not an Apple runtime implementation or transferable speed number. Apple's compression evaluation also shows why perplexity alone is insufficient: knowledge-intensive performance can degrade differently from retrieval or summarization. Use task-level tests, not just a sparsity percentage and a reassuring language-model loss. [Minitron research](https://arxiv.org/abs/2408.11796), [Apple compression evaluation](https://machinelearning.apple.com/research/compressing-llms).

## Architecture determines memory traffic and persistent state

For a dense model, most layer parameters participate for each token. In an MoE model, a router selects a subset of experts per token. Active parameters help estimate arithmetic, while **total** parameters largely determine weight storage when all experts remain resident. Routing, gathering, expert reuse, batch shape and memory traffic also affect time. Mixtral's published example distinguishes approximately 47B total from 13B active parameters; it is not a 13B-sized checkpoint. Streaming experts changes the memory/speed tradeoff and must account for storage latency. [Mixtral paper](https://arxiv.org/abs/2401.04088).

A first-order inference budget is:

`resident weights + quantization metadata + persistent sequence state + temporary tensors + runtime overhead`.

Training adds gradients, optimizer state and saved/recomputed activations; a model that fits for generation may not fit for full fine-tuning. Parameter-efficient training and gradient checkpointing address different parts of that extra memory. Unified memory does not remove these terms. The important comparison is the full working set for the intended workload, not checkpoint download size.

### GQA/MQA: shrink the attention state intentionally

For ordinary full-attention layers with equal key/value head dimensions, an uncompressed KV cache is approximately:

`KV bytes = 2 × batch × sequence_length × attention_layers × KV_heads × head_dimension × bytes_per_value`.

The factor two is keys plus values. With batch 1, 32 layers, 32,768 tokens, head dimension 128 and two-byte values, 32 KV heads require 16 GiB; eight KV heads require 4 GiB; one requires 0.5 GiB. These are calculated examples, excluding padding, metadata and all other memory. Sliding windows, cross-layer sharing, cache compression and hybrid layers change the formula.

Multi-query attention (MQA) shares one KV head; grouped-query attention (GQA) uses several. The GQA paper demonstrates converting multi-head checkpoints and continuing training with a fraction of original pretraining compute. Its reported recipe used 5%; that is evidence of feasibility, not a universal cheap conversion cost. Simply discarding or averaging heads without adaptation can harm quality. [GQA paper](https://arxiv.org/abs/2305.13245).

### Recurrent, sliding-window and hybrid models

Full attention keeps token-indexed history and grows state with context. Recurrent/state-space and linear-attention alternatives compress history into structured state; for a fixed architecture, recurrent state need not grow with every preceding token. Their information-access behavior differs from full attention, so long-context retrieval, copying, multi-hop reasoning and position extrapolation need direct evaluation. Hybrid models retain some full-attention layers alongside cheaper layers; those remaining attention layers still accumulate KV. Mamba and Mamba-2 show that architecture and a suitable parallel algorithm must be designed together. [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-2](https://arxiv.org/abs/2405.21060).

Conversion is possible in selected cases. *The Mamba in the Llama* reuses transformer projections and distills a hybrid model after replacing attention layers. This gives a research route between ordinary fine-tuning and training from scratch. Its reported models and accelerator implementation do not establish ready-made MLX/Splash support. Recurrent-state checkpointing, speculative rollback and prefix-cache restoration are also more involved than slicing attention KV. [Conversion research](https://arxiv.org/abs/2408.15237), [oMLX hybrid cache handling](https://github.com/jundot/omlx/blob/c4605a2dcdf03bf8cb58594d34fb2b0089e60304/omlx/cache/prefix_cache.py#L66).

## Train to reduce sequential generation work

Autoregressive decoding creates a serial dependency: each accepted next token contributes to later input. A draft model can propose several tokens, letting the target verify them together. A properly implemented exact speculative-sampling algorithm preserves the target distribution; draft quality affects acceptance and speed rather than replacing the target's judgment. This guarantee belongs to the specific sampling algorithm and its assumptions, not to every system described as speculative. [Exact speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html).

Multi-token prediction (MTP) adds training objectives or prediction heads for future tokens, often sharing the main trunk. It can improve useful proposals and make self-speculation practical. It still needs an inference path that exploits those predictions, handles rejected tokens and maintains valid state. Jointly trained heads, separately trained drafts and approximate standalone multi-token models are different designs. Compare accepted tokens per target pass against extra draft/verification work and memory; more proposed tokens can be slower when rejection rises. [Multi-token training research](https://arxiv.org/abs/2404.19737).

Tokenizer and output length are another part of co-design. Fewer tokens for the same useful text can reduce sequential steps, while a larger vocabulary enlarges embedding/output tables. Tokenization also changes learned structure and multilingual/code behavior; compression alone does not establish quality. Replacing a pretrained tokenizer requires adapting token embeddings/output mappings and usually substantial continued training. Often the less invasive experiment is to teach accurate concise answers or a compact, valid tool protocol. Measure completed-task time and correctness, not just tokens/second: a fast model that emits twice the unnecessary text may finish later. [Tokenizer research](https://arxiv.org/abs/2402.18376).

## Make the graph and representations match the kernels

Hardware awareness can start before changing the learning objective. Choose candidate matrix widths, attention head dimensions, quantization groups and expert batch shapes that the intended Metal kernels handle efficiently. Favor operations that can fuse without materializing large intermediates. Preserve numerical checks against a reference implementation: an algebraically equivalent fusion should not be justified by a broad quality benchmark if a direct operator comparison can expose a bug. A trained low-precision replacement, by contrast, intentionally changes numerical behavior and needs both operator validation and downstream quality evaluation.

Vision introduces another architectural budget. A vision encoder converts image patches into features, and a projector or resampler turns those into language-model inputs. Training a smaller encoder, reducing image-token count, or distilling a resampler can reduce work for every subsequent attention layer. But fewer visual tokens may lose tiny text, spatial relationships or fine detail. Evaluate document OCR separately from broad scene recognition. Apple's model overview describes different vision backbones and an alignment adapter for on-device and server use; the reusable lesson is to design the visual representation for its task and cost, rather than merely move unchanged image processing to another processor. [Apple vision architecture](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates).

Training does not have to occur on the same hardware used for deployment. The important requirement is that the exported model implements the trained numerical/structural assumptions and has an efficient target execution path. A model trained elsewhere can be designed for Apple execution; a model trained with MLX can still have inefficient deployment shapes. Include export/reload equivalence in the experiment before investing in a long run.

## A hardware-aware training objective

The goal is a Pareto frontier: candidates where better quality cannot be obtained without paying more latency, memory or energy. One proposed search objective is:

`J = task_loss + λT × latency/Tref + λE × energy/Eref`, subject to `peak_memory ≤ budget` and minimum task-quality constraints.

This is a suggested experimental formulation, not a formula implemented by MLX-LM. Specify whether latency means first token, per-token decode or complete task; use representative short/long prompts, image inputs and batch sizes. Normalize units before choosing weights. Treat memory as a hard feasibility bound where appropriate. Measure energy over an entire completed workload, with a stated idle baseline and thermal conditions.

Latency is not generally differentiable through the model. Practical methods alternate weight training with discrete architecture/precision choices, use a measured operator-cost table or surrogate to prune the search, then remeasure complete candidates. Surrogates can misrank models because fusion, shape, cache locality and synchronization interact. Hardware-aware search research demonstrates this general approach, but reported results on another accelerator are hypotheses for Apple execution. [Hardware-conditioned architecture search](https://arxiv.org/abs/2605.17653).

Optimization should count the adaptation cost too. If a candidate saves `Δcost` per inference but costs `Ctrain` to create and validate, an elementary break-even estimate is `Ctrain / Δcost` inferences, assuming positive savings and comparable quality. For changing workloads or rarely used specialists, a simpler conversion may be preferable to a lengthy training project. This is accounting, not a prediction of training duration.

## Suggested experiments and evidence ladder

Begin with a small supported model and a fixed evaluation set. Establish a high-precision or otherwise trusted reference, then vary one factor. Keep training/calibration data separate from evaluation, and include factual knowledge, long-context retrieval, code/tool validity and multimodal tasks when claimed. Record exact model/code revisions, tokens trained, optimizer, precision format, template and inference configuration. No experiment below has been run for this report.

| Experiment | Smallest informative design | Cost/risk and stop condition |
|---|---|---|
| PTQ versus learned quantization | Same checkpoint: ordinary supported quantization, calibrated method, then DWQ at matched storage budget | Teacher/calibration cost; stop if held-out quality does not recover or kernels erase latency benefit |
| Behavior versus structure | Same model with/without task LoRA; compare quality, retries and output length, not just decode speed | Dataset/overfitting risk; do not claim smaller core computation |
| Compact student | Distill one bounded task into an already-supported smaller architecture; compare with that student's original checkpoint | Teacher generation/training cost; reject knowledge or robustness regressions outside accepted scope |
| Structured pruning | Remove one modest layer/channel group and recover with fixed data; retain kernel-friendly dimensions | Architecture/export engineering; compare against a native small model before scaling |
| GQA or hybrid conversion | Reproduce an existing small research recipe before adapting another family | Significant retraining and new operators/state handling; stop at unsupported correctness boundaries |
| Draft/MTP adaptation | Fixed target, train one compatible small draft/head and compare accepted tokens per verification | Additional runtime engineering and memory; require correctness plus net full-turn improvement |
| New architecture/QAT schedule | Small matched-token pilots over width, KV heads and a few supported precisions | Highest research cost; require a stable quality/cost advantage before increasing scale |

Use four evidence levels:

1. **Published mechanism:** a paper explains why it might help and reports its own conditions.
2. **Implementation compatibility:** the exact graph/format exports and runs through the intended operators.
3. **Controlled result:** matched evaluation demonstrates quality, latency, memory and energy on the target workload.
4. **Deployment result:** repeated, diverse tasks confirm reliability, cache behavior, concurrency and recovery.

This report establishes level 1 and selected source evidence toward level 2. It does not establish levels 3 or 4. A strong first research direction is therefore a supported compact student or learned quantization of an existing model, followed by a carefully measured structural experiment. A radical architecture is justified when simpler adaptations leave a clearly measured bottleneck that the new design addresses.
