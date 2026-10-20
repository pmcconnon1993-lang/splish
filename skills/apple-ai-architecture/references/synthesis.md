# Making AI fit Apple architecture

An architecture study of MLX, oMLX, Splash, Metal, and model/training co-design for M5 Max and M5 Ultra. Researched 19–20 October 2026. The subject is how these systems work and how their ideas can be extended. No personal installation or currently loaded model is part of the analysis.

## The central argument

**A model becomes more efficient on Apple hardware when its mathematical work, numerical representation, execution schedule and trained behavior are designed together.** Unified memory is the foundation, not the optimization itself. It makes some data-sharing arrangements easier. The remaining problem is to spend fewer expensive memory accesses and fewer sequential steps producing a correct answer.

The useful object to optimize is a combination:

`target architecture + learned weights + numerical format + optional drafter + kernels + state representation + serving schedule`.

Changing one part can alter the best choice for the others. This is the study's engineering synthesis, grounded in the component analyses below. It does not claim that one universal combination wins every workload.

In that sense, “aerodynamic” is a useful metaphor: remove resistance along the actual path. In computation, the resistance includes repeated movement of large weight arrays, poorly shaped operations, intermediate tensors, synchronization, repeated input processing and unnecessary output. A design that performs fewer nominal operations can still be slower if those operations are irregular or poorly supported. A design that performs extra speculative arithmetic can finish sooner by reducing expensive sequential target passes.

The detailed chapters establish the mechanisms:

1. [M5 hardware and Metal](hardware-metal.md): bandwidth, compute, memory hierarchy, SIMD/threadgroups, tensor operations and scaling.
2. [The runtime stack](runtime-stack.md): model versus framework versus server; what MLX and oMLX contribute.
3. [Why Splash works](splash-mechanisms.md): specialization, fusion, state and the complete draft/verify/accept cycle.
4. [Model and training co-design](model-codesign.md): adapting existing checkpoints and building different architectures.
5. [exo and distributed inference](exo-distribution.md): capacity, tensor/pipeline parallelism, communication and scaling.
6. [Additional research](additional-research.md): alternative drafting, quantization, state and serving methods.
7. [From a model to a faster implementation](optimization-playbook.md): compatibility triage, porting and meaningful experiments.
8. [Formats and specialized serving paths](format-serving-case-study.md): vMLX, EXL3, Engram and why recognizing an architecture is not full support.

## What makes each system work

**Metal exposes GPU computation and resource control.** Programs allocate buffers, encode work, choose execution geometry and establish dependencies. Metal Shading Language describes GPU operations; metal-cpp supplies C++ host bindings. Neither is a pretrained model. Apple GPU memory hierarchy and tensor facilities reward suitable data reuse and parallel shapes. GPU Neural Accelerators inside shader cores are distinct from the separate Apple Neural Engine. [Apple's M5 ML architecture presentation](https://developer.apple.com/videos/play/tech-talks/111432/).

**MLX supplies a general way to express and execute the mathematics.** Arrays share unified memory, operations form graphs, differentiation supports training, and compilation/kernels turn eligible computations into efficient execution. MLX does not know every possible model automatically: concrete model implementations, formats and operator paths still matter. [MLX design](https://github.com/ml-explore/mlx/blob/f41bec734ac9fc96723ab7d4e64e385796f15561/README.md), [MLX compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html).

**oMLX coordinates useful requests over an engine.** Its contributions include batching, state reuse, cache restoration and lifecycle management. Avoiding a second computation can matter more than accelerating that computation by a modest percentage. It can also select an external Splash path for eligible models, which is a separate execution engine with its own internal schedule and cache. [oMLX scheduler](https://github.com/jundot/omlx/blob/c4605a2dcdf03bf8cb58594d34fb2b0089e60304/omlx/scheduler.py#L3629), [engine routing](https://github.com/jundot/omlx/blob/c4605a2dcdf03bf8cb58594d34fb2b0089e60304/omlx/engine_pool.py#L2084).

**Splash specializes the execution system around supported model families.** It controls weight layout, kernels, state, submission and speculative decoding together. The inspected engine already uses shared allocations. Its ordinary decode graph combines drafting, target verification, acceptance and state commitment rather than asking the host to wait after each small operation. [Shared weights](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/WeightImages.cpp#L19), [combined decode graph](https://github.com/incoai/splash/blob/be3b083c5f413673d44e44881042e875e674e333/runtime/model/Runtime.cpp#L2630).

That specialization offers opportunities and imposes obligations. Extending Splash to another model family requires implementing the actual architecture and state transitions, not merely accepting a differently named checkpoint. Its existing DFlash2 draft contract is concrete. The general strategy can transfer; its weights, tensor layouts and constants cannot be assumed to transfer unchanged.

## The most important causal chain: memory reuse meets prediction

Ordinary single-sequence decoding has a serial dependency: the next input token is unknown until the current decision is made. For a large dense projection, reading the weights for a single row can expose relatively little arithmetic per byte. Processing several known rows together can reuse those weights. Prompt processing naturally provides many known rows; speculative decoding manufactures candidate rows and lets the target check them.

DFlash conditions a small parallel drafter on target hidden features. The inspected Splash DFlash2 path uses an anchor plus seven proposals and runs an eight-row target verification. It accepts a valid prefix, corrects a rejection where needed, and commits only valid state. Correct rejection sampling preserves the selected target distribution under its assumptions; the drafter's role is to make that process economical. It does not need to replace the target's broad intelligence. [DFlash research](https://arxiv.org/html/2602.06036v2), [DFlash 2 author report](https://inco.ai/blog/dflash2/), [exact speculative sampling](https://proceedings.mlr.press/v202/leviathan23a.html).

The engineering quantity is:

`useful generation cost ≈ complete speculative-cycle time / mean valid emitted tokens per cycle`.

A more accurate draft is useful only if its extra acceptance outweighs its additional computation and memory. A cheaper but slightly less accurate draft can win. A faster target can change the optimal draft because fixed draft overhead becomes relatively larger. These are deductions from the cost relationship, not published universal rankings. [Worked trace and break-even examples](splash-mechanisms.md).

This is one route to making other models faster without retraining their expensive target weights: train an auxiliary predictor that matches the target and execute verification efficiently. It remains model-specific work involving hidden features, vocabulary, tokenization, positions, sampling and cache state.

## Can models themselves be trained to fit Apple hardware?

**Yes. Several established approaches can adapt existing models, and more ambitious approaches can change the architecture.** “Hyper-training” does not name one standard operation. The right method depends on which cost must fall.

| Desired change | Appropriate training or adaptation route | What it does not guarantee |
| --- | --- | --- |
| Fewer bytes per weight with useful quality | Calibration/PTQ, learned quantization, QAT or quantization-aware distillation | Efficient execution of an unsupported format |
| A cheaper model with useful task capability | Distill into a smaller supported student | Identical general intelligence or robustness |
| Fewer executed layers/channels/experts | Structured pruning with recovery/distillation | Speed from arbitrary zeros in dense kernels |
| Less state and context traffic | GQA conversion/uptraining, KV sharing, supported hybrid/recurrent design | Equivalent long-context retrieval behavior |
| More output per costly target pass | Compatible draft training or MTP/self-speculation | Benefit without an efficient verifier and correct rollback |
| Fewer retries or unnecessary output tokens | Task fine-tuning, protocol training and concise-output training | Reduced base-model matrix work from LoRA alone |
| Fewer expensive vision tokens | Train/distill the encoder, projector or resampler | Preserved tiny-text and spatial detail without testing |

The literature contains existing-checkpoint adaptation routes, including GQA uptraining and transformer-to-hybrid distillation. Starting from zero is therefore not the only way to change structure. Conversely, ordinary fine-tuning is not a universal conversion tool. A substantial change can require new operators, tensor shapes, data, continued training and export/runtime support. [GQA](https://arxiv.org/abs/2305.13245), [The Mamba in the Llama](https://arxiv.org/abs/2408.15237).

Apple's own model research shows co-design in practice: low-bit quantization-aware training, KV sharing and hardware-aware representations appear in its model-design reports. These demonstrate mechanisms; they do not make every proprietary deployment technique a public MLX feature. Public MLX-LM does provide concrete adaptation tools, including LoRA/DoRA and learned quantization methods whose exact behavior must be distinguished. [Apple foundation-model design](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates), [MLX-LM learned quantization](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/LEARNED_QUANTS.md).

A model does not learn physical cache placement merely by reading more text. To favor hardware efficiency, the training/design process must expose a constraint: low-precision behavior, a smaller architecture, a draft objective, a state budget, a measured latency model or another concrete mechanism. The exported graph and the engine must realize that constraint. Training can occur on other hardware; deployment measurements on the intended Apple target close the loop.

## What an Apple-oriented model design could look like

The following is a **research design**, not an implemented model or a claim that every component should be combined.

Start with a small set of task requirements. Select widths, head dimensions, quantization groups and operator types that have efficient Metal implementations. Use enough KV heads or attention layers to meet the quality requirement while controlling context traffic. Choose dense versus MoE based on the working set and actual routed traffic, not active parameter count alone. Train with the eventual numerical representation in mind. If generation is a major cost, train compatible draft/MTP machinery and include its state and verification path in the design. For multimodal work, learn a useful visual representation whose token budget matches the intended detail.

The unit of evaluation should be a correct completed task under a stated latency, memory and energy budget. A Pareto comparison is often more useful than one combined score: identify candidates that improve one cost without unacceptably worsening another. A model may be best for short interactive turns yet lose for long-document prefill or many simultaneous requests.

Two implications follow:

1. **A hardware-friendly architecture need not be Apple-only.** Many useful ideas—reduced state, low precision, data reuse and fewer serial steps—generalize. Apple-specific work often lies in the supported shapes, packing, fusion, memory policy and execution schedule.
2. **The best format is part of training design.** The same nominal bit width can have different group overhead, unpacking cost and quality. Training against one approximation and deploying another can lose both accuracy and expected efficiency.

These implications combine the [hardware analysis](hardware-metal.md), [runtime mechanisms](runtime-stack.md) and [training evidence](model-codesign.md). They should guide experiments, not replace them.

## M5 Max and M5 Ultra: same design questions, different resource balance

Apple publicly lists M5 Max variants with 460 or 614 GB/s memory bandwidth and M5 Ultra variants with 1.2 TB/s. The corresponding GPU counts differ too. The detailed hardware chapter records the exact public variants and separates GPU Neural Accelerators from the dedicated Neural Engine. [Apple specifications](https://www.apple.com/mac-studio/specs/).

Those figures suggest greater available throughput, not a universal application multiplier. For a bandwidth-dominated region, the relevant limit is bytes divided by achieved bandwidth. For compute-dense verification or prefill, supported tensor arithmetic, reuse and parallel work become more important. For a small kernel or a serial host stage, additional cores can offer little benefit. A larger GPU may favor different tiles or batch sizes, while the latency objective may still favor small batches.

This study does not infer undocumented M5 Ultra interconnect characteristics from older Ultras or multiply Max specifications into an invented topology. The public facts and the remaining topology uncertainty are explicitly separated in [hardware and Metal](hardware-metal.md). This is a general comparison of architecture targets, not an assessment of anyone's computer.

## A concrete research program

Each project below is independently useful. Begin with the one that tests the proposed mechanism, rather than changing every layer at once.

| Project | Concrete artifact | Decisive experiment | What would disconfirm it |
| --- | --- | --- | --- |
| Supported low-precision target | Same model family exported in a compatible ordinary and learned-quantized form | Matched storage/quality comparisons; prefill, decode and full task measurements | Recovery fails or unpacking/kernel costs erase the benefit |
| Better target/draft pair | Frozen target plus one trained compatible draft | Acceptance by position, cycle components and completed-task latency | Better draft accuracy costs more than the saved target passes |
| Compact specialist student | Smaller supported model trained on a bounded teacher task distribution | Compare original student, distilled student and teacher on held-out tasks | Narrow gains hide unacceptable capability loss or retries |
| Efficient long-context structure | A reproduced GQA/hybrid conversion or trained small variant | State footprint plus long-context retrieval, copying and reasoning tests | Smaller state compromises required information access |
| Kernel/graph specialization | One reference-checked operator or fused sequence | Actual model shapes, then complete inference | Local kernel gains worsen occupancy, acceptance or end-to-end time |
| Better multimodal representation | Smaller/cheaper encoder or visual-token resampler | OCR, spatial detail and scene tasks with stage timings | Apparent speed comes from losing required visual information |

For every project, maintain the baseline, exact model/draft/tokenizer/format identity, held-out evaluation and stated quality threshold. Test the complete engine integration, not just a training loss or an exported file. Layer outputs, state restoration, speculative rejection positions and tool/vision behavior can expose failures that a single benchmark score misses.

The narrowest practical starting point is usually an already-supported model with either a supported learned-quantization experiment or a compatible auxiliary drafter. That is an engineering recommendation based on the integration burden, not proof those routes always beat a compact student. Radical new architectures become more attractive when an identified bottleneck persists after simpler changes and the quality/cost case justifies the training expense.

## What this study establishes

The combined evidence explains why shared memory helps, why it does not eliminate data movement, how Metal maps work onto the GPU, why MLX is useful for both expressing and training models, why oMLX can save work at the serving layer, and why Splash's specialization and speculation can reduce generation latency. It also identifies credible ways to adapt other models rather than merely translating their code into C++.

The central synthesis is the connected design argument: **train and implement the representation, predictor, state and schedule as one system, then judge the cost of correct outcomes.** That conclusion follows from the mechanisms, but specific candidate designs remain hypotheses. No training run or M5 benchmark was performed for this study.

The accompanying skill preserves these distinctions, source references, examples and experiment choices. It provides reusable reasoning, not a claim that an assistant has acquired new model weights or that an untested model is now faster.
