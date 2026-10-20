# From a model to a faster Apple implementation

Use this chapter when the request is “Splash this model” or “make it as fast as possible.” The deliverable is a supported acceleration path and, when implementation is requested, a measured improvement at a stated quality level. “Fastest” always needs a workload: the best short-turn latency, long-context latency, multi-user throughput and memory footprint can belong to different configurations.

This is an engineering workflow derived from the [runtime stack](runtime-stack.md), [Splash source study](splash-mechanisms.md), [training analysis](model-codesign.md), and [additional research](additional-research.md). It is not a universal model converter or a claim that these experiments have already run.

## Resolve the artifact before changing it

Record the exact repository/revision, model architecture, tokenizer and chat template, weight format, precision/grouping, context/position scheme, and any adapters or draft weights. Read the config and conversion code. File extensions identify containers; they do not establish mathematical or kernel compatibility. Two safetensors checkpoints can contain entirely different packed representations. Renaming an EXL3, GPTQ, AWQ or MLX artifact does not change its operators or packing.

If conversion is needed, prefer the appropriate original or high-quality source weights. Dequantizing and requantizing an already lossy representation can compound error; it cannot recover discarded information. An architecture supported by MLX-LM is not automatically supported by Splash. Fine-tunes can preserve an architecture while changing how well an existing drafter predicts the target. Verify both model execution and draft suitability.

For a theoretical question, establish these distinctions without inspecting anyone's machine. For an implementation request, inspect only the requested environment and preserve unrelated managed environments.

## Choose the route that changes the limiting cost

| Evidence about the workload | First useful experiment | What must be checked |
| --- | --- | --- |
| Model is already supported by Splash | Matched ordinary versus speculative generation, then model-shape kernel tuning | Full cycle time, acceptance/yield, quality, context and concurrency |
| Model works in MLX-LM but has no Splash implementation | Establish the MLX reference, profile actual operations, choose native MLX improvement or scoped port | Architecture/state contract and export compatibility |
| New architecture or unsupported packing | Correct reference implementation and a minimal loader/operator path | Layer outputs and state correctness before speed claims |
| Single-sequence generation repeatedly reads large weights | Supported lower precision; useful draft; fused/reused operations | Unpacking cost, target quality, draft overhead, actual bytes moved |
| Long prompt dominates | Efficient attention/projection shapes, chunk policy, valid prefix reuse | Time to first output, context length, warm/cold cache separation |
| Many requests dominate | Continuous batching and scheduling/cache policy | Per-request latency distribution alongside aggregate throughput |
| Long-context state dominates memory/traffic | Supported KV quantization, GQA/state reduction, cache policy | Retrieval/copying quality, actual resident memory and restore overhead |
| Model cannot fit a single target | Supported distributed execution or measured streaming design | Per-node working sets, communication, supported topology and recovery |
| Model spends time failing tools or writing unnecessary output | Task/protocol training or better task allocation | Successful task completion and retries, not only tokens/second |

The table gives hypotheses to test. A “smaller” format with an inefficient decoder can lose. A faster draft with poor acceptance can lose. A larger batch can improve throughput while worsening interactive latency. A cluster can enable a model without speeding up a model that already fits. See [exo distribution](exo-distribution.md).

## What a real Splash port requires

1. **An architectural specification.** Identify every block, tensor shape, normalization, attention/recurrent update, routing rule, positional scheme and vocabulary operation. Map it to an independently working reference.
2. **A representation contract.** Implement weight names/layouts, quantization groups/scales, packing and runtime buffers. Test conversions with known tensors and selected real layer outputs. Unsupported operators require actual code, not a config alias.
3. **Correct ordinary generation.** Implement prefill, incremental decode, sampling, stopping and full model state. Compare selected layers/logits and representative output behavior before adding speculation. Numerical tolerances must be justified by the computation and required quality.
4. **A compatible draft contract.** Decide whether to use an existing supported draft, train a new one, use built-in MTP, or develop a different strategy. Match hidden features, tokenization, positions and the verifier. Do not assume a draft trained on one family or precision is effective for another.
5. **Correct speculative state.** Handle rejection at every proposal position, accepted-prefix commitment, pending correction/bonus tokens, recurrent state, attention caches, cancellation and subsequent requests. Distribution-preserving sampling needs actual proposal probabilities and the proper rejection correction; a greedy match test alone does not establish sampled correctness.
6. **Efficient execution.** Tune the real shapes and complete cycle. The fastest isolated matmul is not automatically the fastest model if its numerical differences reduce acceptance or its buffers increase pressure. Measure command submission, synchronization, memory traffic and the host critical path.

This ordering protects the ability to localize failures. It does not require implementing every conceivable feature before a bounded experiment. A text-only prototype can remain text-only if that scope is explicit. It should not be advertised as a complete multimodal port.

The concrete source map and DFlash2 equations are in [Splash mechanisms](splash-mechanisms.md). For other drafting approaches, read [additional research](additional-research.md). The source contracts must be refreshed when porting against a different release.

## Train only what addresses the measured problem

If the target is already capable, a compatible drafter may be a smaller training project than changing the target. Collect representative target features/outputs using the intended tokenizer, numerical representation and sampling regime. Train the required draft architecture, then validate emitted target-valid tokens per complete cycle on held-out workloads. Draft training loss is an intermediate signal, not the deployment objective.

If weight traffic dominates, investigate a supported quantization method and quality recovery. The deployed packing and kernels are part of that experiment. If the target itself is too expensive, distillation or structural changes can be appropriate, with an explicit capability budget. LoRA alone usually preserves the base forward-pass shapes; better task behavior can reduce completion time, but it is not evidence that the same matrix computation became cheaper.

Architecture changes can include head sharing, reduced layers/width, expert restructuring or recurrence. Each requires an implementation and a training/recovery plan appropriate to the change. Reuse an established method before inventing a broad new architecture. [Model and training co-design](model-codesign.md) distinguishes these routes and their evidence.

## Make the comparison explainable

Keep a compact record with enough information to repeat the result:

```yaml
question: "Does the candidate reduce completion latency at acceptable quality?"
model: {repository: "...", revision: "...", format: "...", quantization: "..."}
draft: {repository: "... or none", revision: "...", format: "..."}
runtime: {repository: "...", revision: "...", build: "...", engine_path: "..."}
target: {chip_variant: "...", os: "...", device_count: 1, transport: "none"}
workload: {dataset_revision: "...", input_lengths: [], output_limit: 0, concurrency: 1}
generation: {template_revision: "...", sampler: "...", seed_policy: "..."}
state: {weights_loaded: true, compile_warm: true, prefix_cache: "cold or warm"}
metrics: {prefill_ms: null, completion_ms: null, decode_tokens_per_second: null,
          emitted_tokens_per_cycle: null, peak_memory_bytes: null, task_success: null}
quality_threshold: "Defined before comparing candidates"
repetitions: "Enough to expose observed variance; record raw results"
result: "measured / inconclusive / not run"
```

The placeholders are an example record, not runnable benchmark settings. Record queueing, compilation, model load and cache restoration separately when they matter. Synchronize/evaluate the work whose time is claimed. Compare the same prompts and quality constraints; sampled output can differ even when a sampling algorithm preserves a distribution, so identical seeds across different engines are not an equality proof.

For speculative changes, report both cycle cost and total valid emitted tokens, including fallback/bonus tokens where relevant. For multi-user work, include p50/p95 latency and aggregate throughput. For training changes, check held-out reasoning, tools/structured output, long context and vision as required by the intended use. A fast failure is not a successful result.

## Turn an improvement into reusable knowledge

Publish the exact changed mechanism, supported model/runtime versions, raw comparison conditions and known limits. Keep failed experiments when they explain a useful boundary: for example, extra fusion losing occupancy or a faster numerical kernel reducing draft acceptance. State whether the result is a source observation, an author measurement, an analytical estimate or a reproduced experiment.

Use a finite experiment budget. Stop a candidate when correctness fails, quality misses the agreed threshold, or measured end-to-end benefit disappears within noise. Retain the baseline and move to a different mechanism when evidence warrants it. “Make it fast” is a request to find and verify useful improvements; it cannot establish a mathematical global optimum across all models and future runtimes.
