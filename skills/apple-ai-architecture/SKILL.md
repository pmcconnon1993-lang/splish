---
name: apple-ai-architecture
description: "Explain and improve Apple Silicon AI inference by connecting model architecture, training, MLX, oMLX, Splash/Splish, Metal and exo. Use to assess how to accelerate or port a model, design compatible drafts or quantization, compare execution strategies, or study hardware-aware AI. Routine installation and live app troubleshooting use the relevant runtime skill."
---

# Apple AI architecture

Translate an efficiency question into a causal explanation or a testable design spanning model, training, inference engine and hardware. Treat “aerodynamic AI” as reducing memory traffic, wasted computation, sequential dependencies and serving overhead while preserving useful task quality. Do not substitute the user's detected installation for the architectural target they requested.

## Select the useful depth

- Start with [the synthesis and design program](references/synthesis.md) for the joint argument and future experiments.
- For M5 Max/Ultra, memory hierarchy, Metal execution, tensor hardware, synchronization and scaling, use [hardware and Metal](references/hardware-metal.md).
- For what MLX is and how oMLX turns model execution into a service, use [the runtime stack](references/runtime-stack.md).
- For Splash's kernels, state, draft/verify/accept cycle and extending it to another model, use [Splash mechanisms](references/splash-mechanisms.md).
- For what retraining can change, quantization, distillation, pruning, attention/MoE/hybrid design and training costs, use [model co-design](references/model-codesign.md).
- For “Splash this model,” compatibility triage, porting and measured acceleration, use [the optimization playbook](references/optimization-playbook.md).
- For multi-machine execution and communication costs, use [exo and distributed inference](references/exo-distribution.md). Distributed memory is not one physically unified memory pool.
- For alternative methods and the research behind candidate experiments, use [additional research](references/additional-research.md). The [source index](references/source-index.md) supports targeted further reading.
- For artifact compatibility, specialized model paths, EXL3 versus Apple formats and model-table versus prefix-state offload, use [the vMLX/DeepSeek case study](references/format-serving-case-study.md).
- For numerical estimates, use [cost arithmetic](references/cost-arithmetic.md) and the standard-library helper. All calculations are conditional illustrations, never benchmarks.

Load only the references needed to answer the current question. Short conceptual explanations need no setup inventory, profiling session or research ritual. Source pins and dated hardware specifications are reference evidence; verify changing specifications and APIs when a recommendation or implementation depends on them.

## Build the explanation from the computation

Trace one representative request: token/image preparation → model state → prefill → decode or speculative cycle → sampling → usable response. Identify the state and mathematical dependency at each boundary. For a proposal, state which of these it changes:

1. Required model work or sequential depth.
2. Bytes moved and useful reuse per memory read.
3. Operator shape, precision and mapping to hardware.
4. Runtime scheduling, synchronization or reusable state.
5. Total generated work, tool retries and task completion.

Connect the mechanism to the expected benefit and to the condition under which it fails. A list of techniques without those causal links is insufficient. Clearly label source-established behavior, published measurements, analytical estimates and proposed experiments.

## Preserve the distinctions that matter

- RAM stores data; CPU/GPU/ANE execute operations. Unified DRAM is already shared, but caches, layouts, allocations, conversions and producer/consumer dependencies still matter. Private Metal resources are not a second physical VRAM bank on Apple GPUs.
- Metal is the resource/command API; Metal Shading Language describes GPU work; metal-cpp is host bindings. MLX is an array/autodiff/execution framework. MLX-LM/VLM supply model implementations. oMLX is a serving system. Splash is a specialized native engine, not a percentage of MLX “offload.” A checkpoint format does not identify its runtime.
- GPU tensor/neural arithmetic and the separate Apple Neural Engine are distinct. Public Metal support is not equivalent to private ANE runtime paths. A compiled path need not be the executed path.
- Prefill, single-sequence decode, multi-request throughput, multimodal preprocessing and training have different reuse and dependency patterns. Do not rank hardware or engines using one phase alone.
- Parameter count, active parameters, model artifact bytes and full resident runtime memory are different quantities. MoE sparsity does not make all inactive experts disappear from memory. Unstructured zeros do not create speedups without an applicable representation and kernel.
- Autoregressive target verification and speculative proposals are distinct. Distribution-preserving rejection sampling requires the actual proposal probabilities and correct residual sampling. Quantizing/retraining the target changes the distribution being preserved; numerical differences can also affect acceptance. A drafter need not have the target's independent capability to be useful.
- Native CPU/GPU synchronization can enforce correctness. Removing a wait or timing an unevaluated MLX graph is not an established speedup. Fusing more operations can increase register pressure and lose performance.
- Prefix reuse must restore the architecture's complete necessary state, including recurrent and image-dependent state where applicable. Cache identities must account for changed weights/adapters/tokenization/media and other relevant semantics.

## Interpret “hyper-train” constructively

There is no single technique implied by that phrase. Resolve the desired change: better task behavior, fewer bytes, fewer operations, shorter generation, or more successful speculation. Then choose among existing-weight PTQ, recovery/QAT, distillation, structured pruning with recovery, architecture conversion/uptraining, draft/MTP training, or training a new architecture.

Ordinary LoRA/SFT does not automatically reduce forward-pass work. It may reduce end-to-end latency through better task behavior or shorter successful output, but that needs matched quality checks. Distillation can produce a smaller capable student; it does not guarantee identical general intelligence. Training for quantization helps only if the deployed packing, precision, kernels and numerical behavior match the trained assumptions. Published proprietary model techniques are evidence of a mechanism, not an available feature in every public runtime.

Treat model, draft, quantization and engine as a jointly optimized system. Improve quality per unit of completed-task cost, not raw token rate at any cost. Identify what can be changed in an existing checkpoint and what needs architectural implementation, recovery training or substantial new pretraining. Training hardware need not be the deployment hardware.

## Turn a promising idea into a useful experiment

Specify the intended task distribution and minimum quality, then record the chosen hardware target, model/draft identity, formats, context, batching and cache state. Compare a matched baseline. Measure time to first useful output, completion, task success, memory and energy when observable. For speculation also measure total emitted target-valid tokens per cycle and complete cycle time. For model changes include held-out quality, tools/structured output and vision/long-context tests when relevant.

Use analytical bounds to reject implausible claims; use actual traces/benchmarks to select winners. An optimization may move the bottleneck. Hardware core/bandwidth ratios are not universal speedup ratios. Compare M5 Max and M5 Ultra under explicitly matched conditions and public supported specifications; do not invent undisclosed topology or efficiency.

Resolve “Splash a model” into an already-supported execution path, a format conversion, a new model-family implementation, or a draft-training project. These are different amounts of work. Check the exact artifact, operators, packing and state; renaming files is not conversion. Combining engines requires a real state/transport integration, not just an OpenAI-compatible endpoint. Apply optimizations to the engine actually serving the request.

For distributed inference, compare capacity, per-request latency and aggregate throughput separately. Check model-specific sharding, transport/topology and per-node working memory. Recompute headline speedups from matched raw times; fastest-stream rate multiplied by concurrency is not necessarily observed batch throughput. Treat numerical fast-math/synchronization flags as implementation-specific experiments with correctness and stability obligations, not default advice.

For a design request, propose a small set of experiments with mechanism, prerequisites, measured variables, acceptance criterion and disconfirming result. Do not launch training, download weights, change running servers, or replace a managed environment merely because the user requested a study. Actual implementation follows the user's requested scope and permissions.

## Deliver knowledge that survives the example

Give the answer at the user's level, then the causal chain and evidence. Preserve the selected theoretical targets and keep personal installation facts out of an architecture-only study. For a detailed study, include worked arithmetic and a decision table, then unresolved questions with the experiment that would resolve each. Source understanding and a packaged skill do not train model weights or prove a runtime improvement.
