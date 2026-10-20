# A concrete compatibility case: vMLX, EXL3 and DeepSeek V4.1

Reviewed 20 October 2026. This chapter incorporates a completed parallel source study, rechecking the core implementation distinctions in its pinned public sources. It retains general architectural knowledge and excludes personal installation recommendations. No inference result or model conversion was produced.

## A serving product is a set of execution paths

The inspected vMLX source is `jjang-ai/vmlx@aa098f6d966426c821358e898e863d025290e538`. Its published `v1.6.80` release resolves to a different revision. vMLX packages MLX-native serving, cache/batching mechanisms, draft and prompt-lookup paths, and JANG mixed-precision handling. These features make it an additional implementation to assess. Its speed page said the matched benchmark table was pending when reviewed; it did not establish a measured win over oMLX or Splash. [Project source](https://github.com/jjang-ai/vmlx/tree/aa098f6d966426c821358e898e863d025290e538), [speed page](https://vmlx.net/#speed).

A feature available in one model path may be disabled or unsupported in another. Generic distributed execution, external drafting, cache quantization and specialized architecture handlers have different contracts. vMLX and oMLX are different products; similar names and APIs do not make their settings interchangeable. Likewise, oMLX's external Splash integration runs the model in Splash, with its own execution and caching. [oMLX Splash integration](https://github.com/jundot/omlx/blob/85d299585d95129a525b10ce683937bd88a02f50/docs/Splash.md).

One concrete distinction is vMLX's Smelt loader. Its inspected code builds a subset of experts and adds a large negative selection bias for unloaded experts. That can reduce resident work, but changes routing. It must be assessed as a quality-changing model/execution variant, not described as transparent fetching of every expert the unchanged router would have selected. [Selection bias](https://github.com/jjang-ai/vmlx/blob/aa098f6d966426c821358e898e863d025290e538/vmlx_engine/utils/smelt_loader.py#L1078-L1084).

## Containers, codecs and architecture support are separate

The examined `dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED` artifact declares a `deepseek_v41` architecture and packed EXL3 trellis weights. Safetensors stores those tensors; it does not supply the decoder or matrix kernels needed to consume them. ExLlamaV3's documented runtime/build requires CUDA, and no EXL3 runtime loader was identified in the reviewed vMLX source. This is a statement about the inspected versions, not impossibility of a future Metal backend. [Pinned artifact](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED/tree/89af9bf193575b7eeeb4f1dd85463ec24ea5bb4c), [ExLlamaV3 requirements](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/README.md#installation).

The reviewed upstream MLX-LM `deepseek_v41` class provides a particularly clear counterexample to filename-based compatibility. Its argument validation rejects nonempty Engram layers because required operations are missing. Its sanitation also removes vision and MTP prefixes. Finding a matching architecture filename therefore does not prove that the complete multimodal/Engram/drafting checkpoint works. [Validation](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/models/deepseek_v41.py#L69-L74), [sanitation](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/models/deepseek_v41.py#L1105-L1115).

A native Apple port of that exact artifact would need the codec and efficient projections/experts, the full target architecture and state, and correct optional vision/MTP behavior. A different Apple-compatible conversion can be a more practical experiment, but it is a different artifact. Dequantizing a lossy checkpoint cannot recover discarded information, and requantizing it can add error. Preserve an intended fine-tune or edit deliberately; do not substitute an unmodified-base conversion and claim identical behavior.

## SSD can hold two very different kinds of state

oMLX's `v0.7.1rc1` source includes a dedicated DeepSeek V4.1 implementation with `DiskEngramEmbedding`. It maps table data, reads required rows/pages and prefetches, rather than requiring the entire table to be resident as ordinary model arrays. That capability was checked in the release-tag source as well as main. The tag is a release candidate, not a claim about what every stable or installed release contains. [Storage implementation](https://github.com/jundot/omlx/blob/v0.7.1rc1/omlx/patches/deepseek_v41/storage.py), [loader](https://github.com/jundot/omlx/blob/v0.7.1rc1/omlx/patches/deepseek_v41/loading.py), [release](https://github.com/jundot/omlx/releases/tag/v0.7.1rc1).

This is model-table access. An SSD prefix cache instead stores previously computed conversation state to avoid repeated prompt processing. Both use storage, but their traffic pattern, correctness contract and latency are different. A sparse row-access design can make a very large table usable without loading it all at once; it does not make SSD bandwidth equal to DRAM bandwidth. Page locality, prefetch accuracy, concurrent requests and filesystem residency still matter.

The `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2e-mtp` card describes a separate Apple-oriented representation with Engram/table handling. Treat it as evidence of an available artifact and its stated requirements. The card's disk-size breakdown is not a measured total working set, and retained MTP/vision tensors do not prove that a chosen server activates those paths. [Conversion card](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2e-mtp).

## The reusable inference

Model acceleration needs a compatibility tuple:

`architecture + exact weights/edit + numerical packing + tokenizer/template + runtime path + cache/state + optional draft/vision support`.

Check each component at a pinned version. Then choose whether the cheapest valid next step is using a supported artifact, converting appropriate source weights, extending a loader/operator, training a compatible draft, or implementing an entire new model family. Those are distinct engineering projects. The [optimization playbook](optimization-playbook.md) turns that distinction into a sequence of testable decisions.
