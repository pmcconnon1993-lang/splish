# EXO: what distribution can accelerate on Apple hardware

Studied 20 October 2026. Source inspection only; no dependencies installed, models downloaded, services started, or inference benchmarks run. This is an architecture study for Apple systems generally, including the requested M5 Max/Ultra target class; it does not characterize the user's machines or extrapolate M3 measurements into M5 results.

**EXO is a real MLX-based distributed inference system. Its most credible route to faster single-request decoding is model-specific tensor parallelism over a low-latency JACCL/RDMA mesh. Pipeline partitioning chiefly makes a larger model fit and can overlap prompt chunks, but does not make a single autoregressive token traverse its sequential layers simultaneously.** The fastest design depends on whether the objective is first-token delay, subsequent-token delay, total job completion, or concurrent-user capacity.

## Source identity and evidence boundary

- EXO: [`bc412e2230f395e652c23426a5eb88dfa515769b`](https://github.com/exo-explore/exo/commit/bc412e2230f395e652c23426a5eb88dfa515769b), main observed on 20 October, commit dated 13 October 2026. The inspected source tree had no modifications; the source repository is not bundled with this skill.
- The manifest says package version `0.3.70`; this should not be substituted for the commit or confused with the repository's app release numbering.
- macOS MLX dependency: [`0e3ff3643b1c3719f78814b98e0d222afbad867c`](https://github.com/ml-explore/mlx/commit/0e3ff3643b1c3719f78814b98e0d222afbad867c). The lock resolves MLX-LM to a fork, [`rltakashige/mlx-lm@6a3df6cd6b00a347ee40f12d97a182aaf86ea599`](https://github.com/rltakashige/mlx-lm/commit/6a3df6cd6b00a347ee40f12d97a182aaf86ea599), from `leo/deepseek-v4`. Consequently, compatibility with arbitrary installed/upstream MLX-LM versions is unproven. [Manifest](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/pyproject.toml#L45-L113), [lock](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/uv.lock#L1429-L1558).

The accompanying [source ledger](exo-sources.json) records paths, pinned links, hashes, evidence type and remaining limits. Repository tests are evidence of intended checks, not results obtained in this study.

## From discovery to generated tokens

| Stage | Actual implementation at the pinned commit | Implication |
| --- | --- | --- |
| Discover peers | Rust custom IPv6 UDP multicast, namespace/nonce handshake; discovered peers connect to Zenoh over TCP. Zenoh liveliness and topics carry coordination. | Discovery is separate from the tensor transport. Finding another node does not establish usable model sharding or RDMA. |
| Establish shared state | Each Python `Node` runs routing, election, API and optional worker; the elected master maintains placement/task state. Hardware/network reports form a graph with socket and RDMA edges. | A decentralized discovery layer still has an elected coordinating master. |
| Select nodes | Filter topology cycles by minimum/required nodes, estimated weight capacity, model/backend and sharding constraints. Prefer smallest eligible cycles; select using existing download fraction and available memory, with a leaf preference. | This is a placement heuristic, not evidence of measured optimal end-to-end latency. |
| Assign model | Pipeline assigns contiguous layers approximately in proportion to available RAM. Tensor gives each rank every layer index, then partitions that layer's operations/weights. | The two approaches spend communication and compute differently. |
| Connect runners | Build Ring host files or a JACCL RDMA device matrix and TCP coordinator; call `mx.distributed.init(..., strict=True)`. | Ring and JACCL are explicit separate instance types. |
| Load and execute | Lazy MLX-LM model load → model-specific sharding → evaluation/barrier → batch generator by default, prefix cache and token/tool parsers. | A generic API wrapper is only the outer interface; execution is MLX model code. |

Sources: [discovery](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/rust/networking/src/discovery.rs#L20-L174), [TCP/Zenoh setup](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/rust/networking/src/lib.rs#L20-L100), [node lifecycle](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/main.py#L52-L196), [placement](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/placement.py#L192-L419), [layer assignment](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/placement_utils.py#L47-L125), [MLX initialization/loading](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/utils_mlx.py#L91-L147), [generator selection](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/builder.py#L64-L117).

The README describes topology-aware decisions using latency and bandwidth. In the inspected path, `find_ip_prioritised` ranks interface categories and explicitly leaves actual speed profiling as a TODO. [PR #2297](https://github.com/exo-explore/exo/pull/2297), proposing measured bandwidth/latency placement, was open and unmerged when checked. Do not promote that proposed behavior into current main. [Interface priority code](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/placement_utils.py#L339-L378).

### Which backends and model architectures?

The Apple text path selects `MlxBuilder`; the image path selects `MfluxBuilder`. No Splash, oMLX, Core ML or ANE backend was found in the inspected engine, dependency and documentation scope. The placement backend table allows MLX Metal/CUDA/CPU for Ring and **MLX Metal only for JACCL**. The manifest contains CUDA extras and detection code advertises CUDA, while the README still says Linux CPU-only. This is a source/documentation discrepancy: source scaffolding exists, but this study does not establish a working Linux GPU configuration. A `Vllm` enum/detection entry alone does not make vLLM a selectable distributed runner here. [Runner dispatch](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/runner/bootstrap.py#L89-L118), [placement backend table](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/placement.py#L68-L71).

Tensor sharding has explicit strategies for Llama/Ministral3, DeepSeek V3/V3.2/V4 and Kimi K2.5, MiniMax, GLM variants, Qwen3/Next/3.5/MoE/VL, GPT-OSS, Step3.5, Nemotron-H and Gemma4 classes. This is a class dispatch list, not proof every checkpoint/quantization/context works. Unsupported classes raise an error. Placement additionally checks model-card permission and dimensional divisibility, with architecture-specific exceptions; it explicitly excludes distributed pipeline use for Gemma4 and the named DeepSeek-V3.1-8bit model. [Dispatch](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/auto_parallel.py#L456-L603), [gating](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/placement.py#L233-L278).

## Pipeline, tensor, transport: separate decisions

**Pipeline:** each device keeps a contiguous section of layers. First/last layer wrappers receive/send activations, and the decode path gathers the output. A token still depends on all preceding layers. EXO also implements chunked pipeline prefill with staggered stages; the prefill selector invokes it for sufficiently long prompts. That is useful overlap between prompt chunks, and should not be confused with parallel execution of one token's dependent layers. [Wrappers](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/auto_parallel.py#L130-L209), [pipeline prefill](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/generator/generate.py#L163-L270).

**Tensor:** all ranks participate within each layer. For example, Llama's attention Q/K/V and feed-forward gate/up projections are partitioned; output/down projections combine results. This reduces work/weight traffic per rank but introduces collectives throughout the network. It requires a matching sharding strategy, valid dimensions, balanced devices, and low communication overhead. Some specialized MoE strategies partition experts internally; this does not prove a general cross-runtime expert-routing system. [Llama strategy](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/auto_parallel.py#L628-L649), [DeepSeek V4 strategy](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/auto_parallel.py#L755-L940).

| Transport | Verified source requirements | Performance meaning |
| --- | --- | --- |
| MLX Ring | IP connectivity between neighboring ranks; EXO prioritizes Thunderbolt interfaces, then Ethernet categories, then Wi-Fi. | TCP over Thunderbolt is still TCP. Capacity can scale without low enough latency for efficient fine-grained tensor collectives. |
| MLX JACCL | RDMA enabled on participating nodes, full pairwise RDMA connectivity, RDMA interface matrix, reachable coordinator. | Lower communication latency makes tensor splitting more attractive; remote memory remains remote. |

Official MLX documentation places Thunderbolt RDMA at macOS 26.2+ on Thunderbolt 5 Macs. JACCL requires a full mesh and network configuration; recovery-mode enablement is a prerequisite, not an action performed here. Four machines require six pairwise cables for a single-link mesh. The current upstream library documents multiple cables per pair, but this EXO commit's matrix stores one interface string and selects the first matching connection, so upstream multi-link support must not be assumed exposed here. [MLX distributed documentation](https://ml-explore.github.io/mlx/build/html/usage/distributed.html), [EXO matrix](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/placement_utils.py#L296-L325).

An Ultra's on-package fabric and multi-Mac Thunderbolt are different interconnect levels. Adding Macs does not create one cache-coherent GPU with local-memory access latency. It creates communicating processes with separately resident shards. Neither an M5 label nor a sum of memory bandwidth specifications proves useful scaling.

## A latency model that exposes when distribution wins

These are analytical models, not EXO benchmark results. Let `C_i` be a stage's compute/weight-read time, `A` communicated activation bytes, `alpha` message/synchronization latency, `B_net` measured effective bandwidth, `N` nodes, and `m` independent prompt chunks or microbatches.

```text
Transfer time                 ≈ alpha + A / B_net
Single-stream pipeline token  ≈ sum(C_i) + sum(transfers) + coordination
Balanced pipeline m chunks    ≈ (m + N - 1) × stage_time
Pipeline efficiency upper model ≈ m / (m + N - 1)
Tensor-parallel token         ≈ sum_over_layers(max_rank C_layer,rank
                                                + collective_time_layer)
```

The pipeline approximation assumes balanced stages and a stable schedule. `m=1` exposes the fill/drain problem; a long prefill or many requests can occupy more stages. Tensor parallelism can reduce each `C_layer,rank`, but every layer also pays coordination and the slowest rank's time. For a ring all-reduce of a sufficiently large message, a common bandwidth/latency model is `2(N−1)alpha + 2(N−1)A/(N B_net)`; it is **not** a model of JACCL's implementation or small-message performance.

For single-stream dense decode, a useful first lower-bound estimate is `bytes actually read per token / effective local memory bandwidth`. Tensor partitioning may spread those reads over devices, subject to replicated weights, KV access, compute and collectives. For MoE, total weights determine resident capacity, while selected experts and other operations determine each token's traffic; active parameter count alone cannot predict either one.

Capacity must budget weights **plus** KV/hybrid state, temporary activations, runtime buffers, replicated components and loading peaks on **each** rank. EXO's weight-size placement estimate is not a proof of that full budget. Its Llama strategy materializes each layer before sharding it, illustrating why the loading peak can exceed the steady shard size. A heterogeneous memory-proportional layer split can also be compute-imbalanced.

Amdahl's law remains relevant after kernel acceleration: if local compute becomes faster while communication is unchanged, distribution's relative overhead rises. Improvements to local kernels, batching and cache reuse should therefore be measured before assuming more nodes will improve one user's token cadence.

## Prefill/decode disaggregation exists, with narrower behavior than the blog

Current main contains MLX-to-MLX remote prefill, an instance-link API and a benchmark script. `ENABLE_DISAGGREGATION` defaults to false. The master selects an endpoint among linked prefill instances by in-flight task count and reachable interface; this is not the blog's automatically profiled phase-placement optimizer. [Flag](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/shared/constants.py#L110), [endpoint selection](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/master/main.py#L134-L173), [link API](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/api/main.py#L751-L794).

The inspected generator completes `run_prefill_for_request` and then writes cache state. The client receives the complete message sequence before injecting it into MLX caches. Transport is a TCP socket with MessagePack-framed byte tensors. Accepted wire dtypes are BF16/FP16/FP32; the sender explicitly rejects `QuantizedKVCache`, `CacheList` and `DeepseekV4Cache`. These paths do not demonstrate concurrent layer-compute/KV-transfer overlap, quantized wire transport or Splash cache interchange. [Serve sequence](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/runner/llm_inference/batch_generator.py#L303-L317), [sender](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/disaggregated/adapter.py#L64-L156), [client](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/engines/mlx/disaggregated/client.py#L49-L110).

The decision test is whether `remote prefill + cache serialization/transfer/injection + scheduling` is less than local prefill at the required queueing load. Genuine overlap could replace part of that sum with a maximum, but only after its implementation and timing are established. For ordinary attention, approximate KV bytes are `2 × layers × prompt_tokens × KV_heads × head_dim × bytes_per_value`; hybrid, sliding-window and latent-attention models require their actual cache layout instead.

## Performance claims and their boundaries

The headline claims below come from the [pinned EXO README](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/README.md#L20-L67); primary benchmark evidence is identified separately.

| Claim/source | What the primary evidence supports | What it does not establish |
| --- | --- | --- |
| EXO README: up to 1.8× on two devices / 3.2× on four | An upstream performance claim. The linked Qwen3-235B-A22B chart shows EXO 19.5 → 26.2 → 31.9 tokens/s for one/two/four M3 Ultras: approximately 1.34× / 1.64× versus its EXO single-node result. | Those particular headline maxima are not reproduced by that chart. No M5 general speedup follows. |
| README: 99% lower device latency | A marketing claim; MLX's original JACCL PR supplies a reproducible four-way all-reduce microbenchmark and demonstrates the intended latency advantage. | A communication microbenchmark is not an end-to-end model speedup. The exact 99% claim needs its own baseline/message size/raw results. |
| EXO Spark blog headline: 4× | Its Llama-3.1-8B FP16, 8,192-input/32-output table reports 6.42 s M3 Ultra vs 2.32 s combined, approximately 2.77× overall. | Fourfold total latency improvement, arbitrary models/contexts, or current-main automatic overlap/placement. |

The Qwen chart is preserved in the pinned repository and was visually inspected. Its labels compare **llama.cpp/TCP with EXO/RDMA**, so it changes framework, sharding and transport together; it does not isolate RDMA's contribution. The linked December 2025 report used Apple-loaned M3 Ultra machines (two 512 GB and two 256 GB), and discloses prerelease instability. It is historical primary testing, not a current build certification. [Pinned chart](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/docs/benchmarks/jeffgeerling/mac-studio-cluster-ai-full-1-qwen3-235b.jpeg), [original tester report](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5/), [JACCL PR](https://github.com/ml-explore/mlx/pull/2808), [EXO phase-splitting blog](https://blog.exolabs.net/nvidia-dgx-spark/).

The current benchmark endpoint disables prefix caching by default, forces fixed output length and bypasses normal output parsing. Its generation metric measures inter-token intervals; it is not TTFT or total request latency. The benchmark code also computes aggregate TPS as the **fastest** per-request rate multiplied by concurrency. For this study's future validation, retain that reported figure but independently calculate completed output tokens divided by observed batch wall time, failures, TTFT and per-request inter-token percentiles. [Methodology](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/bench/METHODOLOGY.md), [aggregation implementation](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/bench/exo_bench.py#L827-L855).

EXO's bootstrap automatically enables `MLX_METAL_FAST_SYNCH` for JACCL. Current upstream MLX documentation warns this faster CPU/GPU synchronization mode can deadlock and leaves it off by default. This discrepancy matters to a fastest-*reliable* system: do not treat it as an unconditional tuning recommendation. [EXO choice](https://github.com/exo-explore/exo/blob/bc412e2230f395e652c23426a5eb88dfa515769b/src/exo/worker/runner/bootstrap.py#L48-L85), [MLX warning](https://ml-explore.github.io/mlx/build/html/usage/distributed.html#getting-started-with-jaccl).

## Combining EXO, MLX, oMLX and Splash

| Combination | Evidence status | Architectural interpretation |
| --- | --- | --- |
| EXO + MLX/MLX-LM | Existing, statically traced | EXO controls distributed groups, shards and its generation/cache orchestration on MLX. |
| oMLX + Splash | Existing documented integration at `jundot/omlx@85d299585d95129a525b10ce683937bd88a02f50` | oMLX manages a separate Splash process and proxies its API; the model executes in Splash. oMLX's MLX acceleration switches do not automatically apply. |
| EXO + Splash tensor/pipeline execution | No integration found in the inspected EXO scope | Would require a supported distributed layer/collective contract in Splash or a substantial port. An OpenAI-compatible endpoint cannot expose hidden activations, tensor partitions or KV ownership. |
| EXO + oMLX shared in-process batching/cache | No integration found in the inspected EXO scope | Both control runtime lifecycle and scheduling. Sharing model format or an MLX dependency does not merge those controllers. |
| Router sending independent requests to each engine | Plausible separate engineering option | Could increase aggregate task throughput without synchronizing every layer. It does not reduce one request's decode latency by itself. |
| Splash prefill → EXO decode, or the reverse | Hypothesis only | Needs identical model/token semantics plus compatible cache layouts, dtypes, positions, rotary state, hybrid state and serialization; standard chat APIs provide none of this. |

The oMLX integration has concrete version/model-format restrictions and now passes image input for checkpoints with a vision tower. This says nothing about EXO compatibility. [Pinned oMLX Splash documentation](https://github.com/jundot/omlx/blob/85d299585d95129a525b10ce683937bd88a02f50/docs/Splash.md).

## Independent research that improves the design

These are sources for mechanisms and evaluation questions, not Apple-compatible packages or benchmark predictions:

| Primary work | Useful mechanism | Question for an Apple design |
| --- | --- | --- |
| [Petals: distributed inference over the Internet](https://arxiv.org/abs/2312.08361) | Fault-tolerant block serving and load balancing across uneven, changing devices. | When capacity forces layer splitting, what replication/rerouting cost is required to survive a node departure? Its comparison against offloading is a different baseline from one fast Mac. |
| [Helix](https://arxiv.org/abs/2406.01566) | Joint placement and request scheduling modeled as flow through heterogeneous compute/network resources. | Measure device and link capacities; compare a workload-aware placement policy with EXO's memory/category heuristic. |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | Separate prefill/decode resources and optimize requests satisfying both TTFT and token-latency targets. | Is enough queueing interference removed to pay for KV transfer? Evaluate goodput, not only peak tokens/s. |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Match prompt processing and generation to different hardware; explicitly account for state transfer. | Can an Apple GPU/ANE or a second machine perform one phase efficiently enough to cover conversion and transfer? Their GPU-cluster result does not establish Apple or EXO compatibility. |

## Architecture recommendation and smallest decisive experiment

Start with the strongest compatible single-node engine as the baseline. Compare EXO tensor/JACCL when a supported model is large enough for weight-read savings to exceed collective overhead and the machines form a suitable mesh. Use pipeline splitting when capacity is the requirement, or when measured prompt-chunk/concurrency overlap is valuable. Prefer independent replicas/workers when the objective is many independent tasks and each model already fits; replication spends memory to avoid per-layer network dependencies.

The smallest decisive future experiment is **one fixed checkpoint, quantization and workload**, run on one device and two matched devices with Ring/pipeline, Ring/tensor and JACCL/tensor where supported. Hold prompt/output length, cache state, reasoning mode and concurrent requests constant. Record exact commits, link topology, effective communication latency/bandwidth, per-rank peak memory, TTFT, inter-token p50/p95, end-to-end completion, aggregate wall-time throughput and repeatability. Then extend to long prefill, cache reuse, four devices and representative heterogeneous devices. No such runtime claim has been made by this study.
