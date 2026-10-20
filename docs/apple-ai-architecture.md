# Apple AI architecture and acceleration

A reusable Codex skill and source-backed study for making AI efficient on Apple Silicon: MLX, oMLX, Splash/Splish, Metal, model training and exo distribution.

- **Use the skill:** [apple-ai-architecture](../skills/apple-ai-architecture/SKILL.md).
- **Read the connected argument:** [Making AI fit Apple architecture](../skills/apple-ai-architecture/references/synthesis.md).
- **Start an acceleration project:** [From a model to a faster implementation](../skills/apple-ai-architecture/references/optimization-playbook.md).
- **Find the evidence:** [Source index](../skills/apple-ai-architecture/references/source-index.md).

The central idea is to optimize the model, precision, draft, state and engine together. The skill distinguishes supported execution from a new architecture port, sampled correctness from approximate acceptance, per-request latency from throughput, and distributed capacity from speed. It includes conditional arithmetic for weight traffic, KV memory, speculation break-even and Amdahl's law.

The references cover hardware and Metal execution, MLX/oMLX serving, Splash's DFlash2 cycle, quantization and training, exo transport/sharding, additional primary research, and a model-format case study. M5 Max and M5 Ultra are general architectural targets. Personal installations and private setup records are excluded.

Copy the complete `skills/apple-ai-architecture` folder to a Codex skills directory, preserving its `references`, `scripts` and `agents` subfolders. Avoid overwriting an existing customized copy without reviewing the differences. Invoke it as `$apple-ai-architecture`; its description also supports automatic selection for relevant requests.

This package contains research, source inspection, arithmetic and behavioral validation. It does not itself convert or train model weights, change the Splash engine, or establish a new inference speed record. Published author measurements and proposed experiments are labeled in the references. Consult each upstream project's license before reusing its code or models; the study links to those sources and does not bundle their weights.
