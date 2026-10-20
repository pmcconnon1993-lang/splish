#!/usr/bin/env python3
"""Conditional arithmetic for AI architecture reasoning; never a benchmark."""
import argparse
import json
import math


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def estimate(args):
    result = {'operation': args.operation, 'evidence': 'conditional arithmetic, not measured performance'}
    if args.operation == 'weight-sweep':
        if args.efficiency > 1:
            raise ValueError('efficiency must be no greater than 1')
        weight_bytes = args.parameters_billion * 1e9 * args.bits / 8 + args.metadata_gb * 1e9
        seconds = weight_bytes / (args.bandwidth_gbps * 1e9 * args.efficiency)
        result.update(weight_bytes=weight_bytes, sweep_ms=seconds * 1000,
                      conditional_tokens_per_second=1 / seconds,
                      assumptions=['Dense-model weights read from DRAM once per output token; no batching, speculation or cross-token weight-cache benefit.',
                                   'Metadata is only what the caller supplies; KV traffic, activations, arithmetic, scheduling and synchronization are excluded.',
                                   'GB and GB/s use decimal units. Effective bandwidth is advertised bandwidth multiplied by the supplied efficiency.'])
    elif args.operation == 'kv-cache':
        size = 2 * args.layers * args.batch * args.tokens * args.kv_heads * args.head_dim * args.bytes_per_element
        result.update(kv_bytes=size, kv_gib=size / 2**30,
                      assumptions=['Conventional full-attention K and V of identical dimensions and precision for every layer.',
                                   'No cross-layer sharing, sliding windows, latent compression, recurrent state, quantization metadata or allocator overhead.'])
    elif args.operation == 'speculation':
        cycle = args.draft_ms + args.verify_ms + args.overhead_ms
        result.update(cycle_ms=cycle, ms_per_emitted_token=cycle / args.emitted_tokens,
                      speedup_ratio=args.target_token_ms * args.emitted_tokens / cycle,
                      break_even_emitted_tokens=cycle / args.target_token_ms,
                      assumptions=['emitted-tokens is mean total valid output per cycle, including any fallback or bonus token; it is not simply the draft acceptance rate.',
                                   'Use consistent complete-cycle timings and a matched autoregressive baseline. Proposed values do not prove target-distribution correctness or quality.'])
    elif args.operation == 'amdahl':
        if args.fraction > 1:
            raise ValueError('fraction must be no greater than 1')
        remaining = 1 - args.fraction + args.fraction / args.stage_speedup
        result.update(total_speedup=1 / remaining, remaining_time_fraction=remaining,
                      assumptions=['Original nonoverlapping execution fraction; all other work and overhead remain unchanged.',
                                   'Does not model changed overlap, bottleneck migration, quality or retries.'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    weights = commands.add_parser('weight-sweep')
    weights.add_argument('--parameters-billion', type=positive, required=True)
    weights.add_argument('--bits', type=positive, required=True)
    weights.add_argument('--metadata-gb', type=nonnegative, default=0)
    weights.add_argument('--bandwidth-gbps', type=positive, required=True)
    weights.add_argument('--efficiency', type=positive, default=1)
    kv = commands.add_parser('kv-cache')
    for name in ('layers', 'tokens', 'kv-heads', 'head-dim'):
        kv.add_argument('--' + name, type=positive_int, required=True)
    kv.add_argument('--batch', type=positive_int, default=1)
    kv.add_argument('--bytes-per-element', type=positive, required=True)
    spec = commands.add_parser('speculation')
    for name in ('target-token-ms', 'verify-ms', 'emitted-tokens'):
        spec.add_argument('--' + name, type=positive, required=True)
    spec.add_argument('--draft-ms', type=nonnegative, required=True)
    spec.add_argument('--overhead-ms', type=nonnegative, default=0)
    amdahl = commands.add_parser('amdahl')
    amdahl.add_argument('--fraction', type=nonnegative, required=True)
    amdahl.add_argument('--stage-speedup', type=positive, required=True)
    args = parser.parse_args()
    try:
        result = estimate(args)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
