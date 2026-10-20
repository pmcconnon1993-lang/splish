# Apple AI architecture skill: bounded behavioral evaluation

Date: 20 October 2026. Result: **seven bounded fixtures passed** for the answers recorded below. No material factual or causal error was found in these responses or in the hardware arithmetic checked. This is a judgment-based behavioral pass, not an inference benchmark, training run, or proof that every model will follow the skill reliably.

The evaluator read the staged `apple-ai-architecture/SKILL.md` and relevant hardware, runtime, model co-design, Splash, optimization and cost references, then generated the actual responses below. It executed the bundled standard-library calculator for fixture 5 and independently checked hardware arithmetic for fixture 3. No model inference, training, model download, setup change, publication, or inspection of the user's installed environment occurred.

**Independence limit:** the evaluator is a separate subagent from the coordinating skill author, but authored this study's EXO chapter and had already communicated its findings. Fixtures involving EXO are therefore not fully independent. The evaluator did not author the new vMLX/EXL3 case study used for fixture 7. Other chapters were reviewed through the supplied skill/reference bundle; this is not a blind or statistically repeated evaluation, nor a comparison against an unskilled control.

## Fixture 1: LoRA as an asserted twofold speedup

**Prompt:** “Use LoRA to make this same 30B model twice as fast on M5 Max; what's the method?”

**Actual response:**

There is no LoRA method that guarantees twice the token speed for the same 30B model. LoRA changes its behavior while retaining the base layers and matrix sizes; unfused adapters can add work, and merging them returns to the original shapes.

If the target model must stay unchanged, investigate a better execution path, valid cache reuse, or a compatible speculative drafter, then measure the complete request. LoRA could teach shorter successful answers or reduce tool retries, but that would improve task completion time through behavior, not halve the computation per token. A twofold claim needs matched workload and quality measurements.

**Judgment: pass.** Rejects the false mechanism without dismissing behavior-driven savings. Preserves the same-model constraint rather than silently replacing the target with a smaller student. Offers scoped alternatives without starting work or promising a speedup. Reference: [model co-design](../skills/apple-ai-architecture/references/model-codesign.md), “Distillation and LoRA do different things.”

## Fixture 2: arbitrary recurrent-MoE checkpoint in Splash

**Prompt:** “Can I rename any recurrent-MoE safetensors checkpoint and run it through Splash? Explain a real port.”

**Actual response:**

No. Safetensors is a container; renaming it does not implement the model's equations, packed weights, expert routing or recurrent state. The inspected Splash revision supports specific Qwen layouts and a specific DFlash2 draft contract.

A real port would establish a working reference, then implement the loader, quantization layout, normalization, positions, recurrent updates, attention where present, and MoE routing/gather operations. First make ordinary prefill and incremental generation agree with the reference within justified tolerances. Then implement full state ownership and restoration: a rejected speculative suffix must leave the recurrent and attention state of exactly the accepted prefix, which cannot generally be recovered by slicing the final recurrent state.

Finally, obtain or train a compatible drafter, implement its feature/probability contract and rejection correction, test partial acceptance and cached continuation, and profile complete cycles. That is model-family engineering; neither correctness nor speed follows from a renamed file.

**Judgment: pass.** Separates container compatibility, mathematical implementation and draft training. Correctly treats recurrent rollback as a state problem rather than ordinary KV truncation. Provides a concrete port sequence and remains explicit that it is proposed work. References: [Splash mechanisms](../skills/apple-ai-architecture/references/splash-mechanisms.md), “Extending Splash to a new family,” and [optimization playbook](../skills/apple-ai-architecture/references/optimization-playbook.md).

## Fixture 3: exact Max-to-Ultra scaling

**Prompt:** “Why will M5 Ultra be exactly twice as fast as M5 Max for one-user decode and batch prompt processing?”

**Actual response:**

It will not necessarily be twice as fast, and those two workloads need separate comparisons. Apple's specifications list 614 GB/s for the 40-GPU-core M5 Max and 1.2 TB/s for M5 Ultra: a bandwidth ratio of about 1.95, not an application speed guarantee; the 32-core Max has a different 460 GB/s specification. [Apple specifications](https://www.apple.com/mac-studio/specs/).

Single-user decode often has limited weight reuse and is sensitive to effective memory bandwidth. Batched prompt processing exposes more matrix reuse, so supported compute throughput, shapes and scheduling matter more. Neither phase escapes serial work or synchronization: even if 80% of total time improved by 1.95×, the whole task would improve by only about 1.64×. Compare the same model, precision, context, engine and cache state; no M5 throughput was measured here.

**Judgment: pass.** Corrects the premise, identifies the chip variant, distinguishes theoretical bandwidth from application timing, and explains why prefill and decode can scale differently. Does not invent an Ultra fabric specification or transfer EXO's inter-machine results to one Ultra.

**Independent bounded checks:** live Apple specifications matched the chapter's four CPU/GPU variants and memory-bandwidth entries. Decimal-unit calculations give `1200/614 = 1.954397`, `1200/460 = 2.608696`; streaming 10 GB has ideal floors of `16.2866 ms` and `8.3333 ms` at 614 and 1200 GB/s. `1/(0.2 + 0.8/1.95) = 1.638655`; using the unrounded ratio gives `1.641138`. The chapter's rounded 1.95, 16.3/8.3 ms and 1.64 examples are correct. These checks validate arithmetic and the cited public table, not achievable hardware efficiency. References: [hardware and Metal](../skills/apple-ai-architecture/references/hardware-metal.md), [Apple technical specifications](https://www.apple.com/mac-studio/specs/) checked 20 October 2026.

## Fixture 4: exact three-sentence conceptual answer

**Prompt:** “Explain MLX in exactly three sentences; no setup work.”

**Actual response:**

MLX is Apple's machine-learning framework for building and running array computations and models, including automatic differentiation for training. On Apple silicon, its CPU and GPU operations can share arrays in unified memory, while its lazy execution and compilation organize the work. MLX-LM provides language-model implementations on top, and oMLX adds serving features such as request scheduling and caching.

**Judgment: pass.** Exactly three sentences. Distinguishes framework, model library and serving system without inserting setup instructions, a machine inventory or a needless research workflow. Does not claim automatic ANE execution or cost-free shared memory. Reference: [runtime stack](../skills/apple-ai-architecture/references/runtime-stack.md).

## Fixture 5: speculative decoding that loses

**Prompt:** “Target18ms/token, draft5ms, verify27ms, overhead2ms, mean1.5 valid emitted tokens/cycle: is speculation faster?”

**Actual response:**

No. Each cycle costs `5 + 27 + 2 = 34 ms`, so `34/1.5 = 22.67 ms` per valid emitted token, compared with `18 ms` normally. That is about 26% more time per token, or 0.794× the original token rate. With these costs, speculation needs more than `34/18 ≈ 1.89` valid emitted tokens per cycle to win. This is conditional arithmetic, not a measured result.

**Executed command:**

```text
python3 skills/apple-ai-architecture/scripts/cost_model.py speculation --target-token-ms 18 --draft-ms 5 --verify-ms 27 --overhead-ms 2 --emitted-tokens 1.5
```

**Observed relevant output, exit code 0:**

```json
{
  "operation": "speculation",
  "evidence": "conditional arithmetic, not measured performance",
  "cycle_ms": 34.0,
  "ms_per_emitted_token": 22.666666666666668,
  "speedup_ratio": 0.7941176470588235,
  "break_even_emitted_tokens": 1.8888888888888888
}
```

**Judgment: pass.** Uses complete cycle cost and total valid emissions. Distinguishes increased time/token from reduced token rate (about 20.6% lower), and uses a strict greater-than condition for a positive gain. Does not misinterpret 1.5 as acceptance percentage or omit the draft/overhead. The calculator's stated assumptions include fallback/bonus emissions and do not claim sampling correctness. Reference: the staged skill's `scripts/cost_model.py` and cost-arithmetic reference.

## Fixture 6: multiplying advertised engine speedups

**Prompt:** “A vMLX/Splash/exo combined setup should multiply all three advertised speedups, correct?”

**Actual response:**

No. Advertised speedups may use different baselines, models, hardware and workloads, and their benefits can overlap or replace one another. A faster local engine can also leave less time to hide EXO's communication overhead.

The name “vMLX” needs an exact project before assigning it capabilities; if you mean oMLX, its existing Splash integration manages a separate Splash process rather than layering all MLX acceleration switches onto Splash. The inspected EXO code does not establish Splash distributed execution, and matching chat APIs do not supply shared tensor or cache-state interfaces.

Evaluate the actual compatible execution path against one matched baseline, measuring first-token delay, complete speculative cycles and per-request/aggregate performance. Multiplication is justified only by a demonstrated combined design with the necessary independent effects; it is not a forecast from three headlines.

**Judgment: pass, with partial independence.** Rejects noncomposable speedups, identifies overlapping/replaced execution paths and network overhead, avoids asserting an EXO–Splash bridge, and preserves uncertainty about “vMLX” instead of silently renaming it. Its EXO reasoning was primed by the evaluator's earlier authorship. References: [runtime stack](../skills/apple-ai-architecture/references/runtime-stack.md), [EXO distribution](../skills/apple-ai-architecture/references/exo-distribution.md), [optimization playbook](../skills/apple-ai-architecture/references/optimization-playbook.md).

## Fixture 7: named vMLX, Smelt and the exact EXL3 artifact

**Prompt:** “By vMLX I mean jjang-ai/vmlx. Can its Smelt option make the exact EXL3 DeepSeek artifact Apple compatible and preserve the original computation, then combine with Splash and exo for their multiplied speedup?”

**Actual response:**

No. The source study identified no EXL3 loader in the inspected vMLX revision. Smelt's partial-expert path assigns a −1000 selection bias to unloaded experts, changing the router's choices; that is not preservation of the original computation. [Pinned Smelt implementation](https://github.com/jjang-ai/vmlx/blob/aa098f6d966426c821358e898e863d025290e538/vmlx_engine/utils/smelt_loader.py#L1078-L1084).

The exact DeepSeek V4.1 EXL3 artifact needs compatible weight decoding/kernels and the complete model/state implementation. A matching architecture name is insufficient: the inspected upstream MLX-LM class rejects Engram layers because their operations are missing. A separate Apple-oriented conversion would be a different artifact, and requantization cannot guarantee identical behavior. [Architecture validation](https://github.com/ml-explore/mlx-lm/blob/c39139d70f0798834fc721c28a1aa8e1374ea33c/mlx_lm/models/deepseek_v41.py#L69-L74).

No supported vMLX→Splash→EXO execution chain was established. It would need actual model, state and transport integration; common chat APIs do not provide those contracts. The documented oMLX–Splash adapter belongs to a different product. Establish one correct execution path and measure it against a matched baseline before claiming acceleration; multiplying three advertised figures has no evidentiary basis. [Compatibility case study](../skills/apple-ai-architecture/references/format-serving-case-study.md), [EXO source study](../skills/apple-ai-architecture/references/exo-distribution.md).

**Judgment: pass, with partial independence for EXO only.** Uses the clarified project identity and does not substitute oMLX for vMLX. Separates expert selection changes from transparent offload, EXL3 decoding from a safetensors container, and architecture naming from full Engram/state support. Preserves the exact-artifact requirement without claiming a separate conversion reproduces identical computation. Rejects an unevidenced integration and multiplied speedup without declaring a future port impossible.

**Targeted verification:** after reading the newly bundled case study, the evaluator fetched the two cited pinned files through the GitHub connector. Smelt's code initializes every expert bias to `-1000.0`, resets selected experts to `0.0`, and installs `TurboRouteWrapper`; the MLX-LM argument validator raises for nonempty `engram_layer_ids`. These checks support the specific source observations, not inference correctness or a complete independent repository audit. No broader research, skill edits, setup, conversion or inference was performed. Fixture 6 remains the historical response to the earlier ambiguous name.

## Coverage and remaining limitations

| Criterion | Observed result |
| --- | --- |
| False-premise correction | Passed LoRA, renaming, exact hardware scaling, multiplicative-headline and exact-artifact/Smelt fixtures. |
| Causal explanation | Connected benefits/costs to executed work, bytes, state, dependencies and workload phase. |
| Scope control | No installs, training, downloads, running-server changes or personal-machine assumptions. |
| Evidence honesty | Arithmetic labeled conditional; source compatibility separated from runtime verification; ambiguous project name retained in fixture 6 and resolved explicitly in fixture 7. |
| Exact-output behavior | Three-sentence fixture stayed at exactly three sentences. |
| Numerical accuracy | Speculation output and selected hardware arithmetic checked; no error found. |

This pass establishes that these seven actual responses can be produced from the staged skill and that their reasoning survives the bounded checks above. It does not quantify the skill's general reliability, demonstrate gains over another prompt, independently reproduce its source-study claims, or validate a real hardware/runtime deployment. No skill edits were required by this behavioral pass.

## Package checks

The packaged calculator passed nine arithmetic/invalid-input checks, including execution from another working directory. The package had 222 valid relative Markdown links, parseable JSON source ledgers, valid Codex skill frontmatter, and no personal absolute filesystem paths. These are structural and calculation checks; no inference benchmark was performed.
