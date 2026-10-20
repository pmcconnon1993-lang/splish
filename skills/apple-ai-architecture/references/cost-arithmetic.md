# Conditional cost arithmetic

Use `python3 scripts/cost_model.py --help` from the skill directory, or invoke its absolute path. The helper uses the Python standard library, performs no inference or network access, and labels its assumptions in JSON. Input units are explicit. These examples are invented illustrations, not M5 or Splash measurements.

## Dense weight sweep

```sh
python3 scripts/cost_model.py weight-sweep --parameters-billion 8 --bits 4 --bandwidth-gbps 400
```

Eight billion 4-bit parameters occupy 4 decimal GB before quantization metadata and other state. One ideal 400 GB/s sweep takes 10 ms. This implies a conditional 100 tokens/s traffic ceiling **only if** the weights are read once from DRAM for each emitted token, with no batching/speculation or useful cross-token weight caching. It excludes KV traffic, compute, temporary values and synchronization. Supply metadata and an explicitly justified efficiency when useful; do not pass a chosen efficiency off as measured.

## Conventional attention KV

```sh
python3 scripts/cost_model.py kv-cache --layers 32 --tokens 8192 --kv-heads 8 --head-dim 128 --bytes-per-element 2
```

`2 × layers × batch × tokens × KV heads × head dimension × bytes` gives 1 GiB here. The two represents K plus V. This formula does not describe recurrent state, latent KV compression, cross-layer sharing or sliding windows. Account separately for metadata, padding and allocation overhead.

## Speculative break-even

```sh
python3 scripts/cost_model.py speculation --target-token-ms 20 --draft-ms 5 --verify-ms 30 --emitted-tokens 4
```

This hypothetical cycle takes 35 ms and emits an average of four valid tokens, or 8.75 ms/token versus a 20 ms baseline: about 2.29×. Break-even requires more than 1.75 emitted tokens per cycle. Emitted tokens include any fallback/bonus output; acceptance rate alone is not this value. Add cycle overhead if it is not already included. Different contexts, quality, cache states or baselines invalidate a comparison.

## Fraction of the task improved

```sh
python3 scripts/cost_model.py amdahl --fraction .2 --stage-speedup 2
```

If a nonoverlapping stage accounts for 20% of original task time and becomes twice as fast, total speedup is `1/(0.8+0.2/2) = 1.111…`. This assumes other work and overhead stay unchanged. Re-evaluate when overlap, bottlenecks, output length or retries change.

These calculations expose assumptions and check units. They cannot prove accuracy, energy use, supported model formats, speculative sampling correctness or realized speed. The study's hardware, runtime, Splash and training references explain when the assumptions apply.
