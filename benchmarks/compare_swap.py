"""Compare tiled SWAP, the original pair exchange, and native FlagQuantum."""

import argparse
import hashlib
import inspect
import json
import statistics
import sys
from datetime import datetime, timezone
from functools import partial
from itertools import combinations
from pathlib import Path

import torch
import triton
import triton.testing

import flagquantum as fq
from flagquantum.core.runtime_config import RuntimeConfig, runtime_config
from flag_gems_quantum import apply_swap_inplace
from flag_gems_quantum.statevector.apply_permutation_inplace import (
    apply_permutation_inplace_kernel,
)
from compare_native_gates import native_call, revision, route_label, trace_native


def original_swap(state, first, second):
    """Launch the unchanged kernel used by SWAP before tile writeback."""
    size, batch = state.shape[-1], state.shape[0]
    with torch.get_device_module(state.device).device(state.device):
        apply_permutation_inplace_kernel[(triton.cdiv(size // 4, 256), batch)](
            torch.view_as_real(state), SIZE=size, FIRST=first, SECOND=second,
            SWAP=True, CONTROL_MASK=0, BLOCK=256, num_warps=4,
        )
    return state


def bit_pairs(n, all_pairs):
    if all_pairs:
        return list(combinations(range(n), 2))
    # Low bits, tile boundaries, middle/high bits and long-distance pairs.
    candidates = [(0, 1), (0, 2), (1, 2), (2, 3), (0, n - 1), (1, n - 1),
                  (2, n - 1), (3, n - 1), (4, n - 1), (6, n - 1),
                  (7, n - 1), (8, n - 1), (3, 4), (7, 8), (8, 9), (n // 2 - 1, n // 2),
                  (n // 2, n - 1), (n - 2, n - 1), (n - 1, 0)]
    return list(dict.fromkeys(pair for pair in candidates
                             if min(pair) >= 0 and max(pair) < n and pair[0] != pair[1]))


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SWAP tile writeback comparison", "",
        f"Device: {report['device_name']} ({report['device']}), complex64, forward only.",
        f"FlagQuantum: `{report['flagquantum_revision']}`.", "",
        "GPU time from warmed CUDA-graph replay; initialization, JIT and Python are excluded.",
        "Three implementations run in alternating order on the same device. Buffers are reused",
        "without cache flushing. Original and tiled SWAP are in place; native FlagQuantum",
        "keeps its normal output allocation. The original kernel is still used by CSWAP.",
        "Every row is checked bitwise against independent CPU tensor-axis transposition.",
        f"Rounds: {report['rounds']}; graph target: {report['replay_ms']} ms per round.", "",
        "| Qubits | Batch | Bits (LSB) | Original us | Tiled us | Native us | Original / tiled | Native / tiled |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report['results']:
        times = row['timings']
        lines.append(
            f"| {row['qubits']} | {row['batch']} | {tuple(row['bits'])} | "
            f"{times['original']['median_us']:.3f} | {times['tiled']['median_us']:.3f} | "
            f"{times['native']['median_us']:.3f} | {row['original_over_tiled']:.3f}x | "
            f"{row['native_over_tiled']:.3f}x |")
    lines.extend(["", f"Complete: {report['complete']}. Rows: {len(report['results'])}.", ""])
    path.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qubits', type=int, nargs='+', default=[20])
    parser.add_argument('--batch', type=int, nargs='+', default=[1])
    parser.add_argument('--all-pairs', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--replay-ms', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.qubits) < 2 or min(args.batch + [args.rounds, args.replay_ms]) < 1:
        parser.error('qubits must be >= 2; batch, rounds and replay-ms must be positive')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('an available CUDA-interface device is required')
    torch.set_num_threads(1)
    config = RuntimeConfig(backend='pytorch', device=str(device), complex_dtype='complex64')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'benchmarks/compare_native_gates.py',
               *sorted((root / 'src/flag_gems_quantum').rglob('*.py'))]
    report = {
        'complete': False, 'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'single-gate GPU time; CUDA graph; initialization/JIT/Python excluded',
        'cache_flush': False, 'distribution_semantics': 'single_device_fast_path',
        'scalability_claim_allowed': False, 'dtype': 'complex64',
        'device': str(device), 'device_name': torch.cuda.get_device_name(device),
        'torch_version': torch.__version__, 'triton_version': triton.__version__,
        'triton_module': triton.__file__, 'python_executable': sys.executable,
        'flagquantum_revision': revision(Path(fq.__file__).resolve().parents[1]),
        'source_sha256': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        'original_kernel_file': inspect.getfile(apply_permutation_inplace_kernel.fn),
        'rounds': args.rounds, 'replay_ms': args.replay_ms,
        'all_pairs': args.all_pairs, 'results': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(2051)
    print(f"{report['device_name']} | original / tiled / native SWAP", flush=True)
    with torch.no_grad(), torch.cuda.device(device), runtime_config(config):
        for n in args.qubits:
            for batch in args.batch:
                source = torch.randn((batch, 1 << n), dtype=torch.complex64, generator=generator)
                states = {name: source.to(device) for name in ('original', 'tiled', 'native')}
                for bits in bit_pairs(n, args.all_pairs):
                    expected = source.view(torch.int64).reshape((batch,) + (2,) * n).transpose(
                        n - bits[0], n - bits[1]).reshape(source.shape)
                    native, steps = native_call(states['native'], 'swap', bits, {}, config)
                    calls = {'original': partial(original_swap, states['original'], *bits),
                             'tiled': partial(apply_swap_inplace, states['tiled'], *bits),
                             'native': native}
                    aliases = {}
                    for name, call in calls.items():
                        states[name].copy_(source)
                        if name == 'native':
                            actual, trace = trace_native(call)
                        else:
                            actual = call()
                        assert torch.equal(actual.cpu().view(torch.int64), expected), (n, batch, bits, name)
                        aliases[name] = actual.data_ptr() == states[name].data_ptr()
                    timings = {name: {'rounds_us': [], 'samples_us': []} for name in calls}
                    for iteration in range(args.rounds):
                        names = list(calls)
                        if (iteration + len(report['results'])) % 2:
                            names.reverse()
                        for name in names:
                            states[name].copy_(source)
                            samples = [time * 1000 for time in triton.testing.do_bench_cudagraph(
                                calls[name], rep=args.replay_ms, return_mode='all')]
                            timings[name]['rounds_us'].append(samples)
                            timings[name]['samples_us'].extend(samples)
                    for item in timings.values():
                        samples = sorted(item['samples_us'])
                        item.update(median_us=statistics.median(samples),
                                    p10_us=samples[int(.1 * (len(samples) - 1))],
                                    p90_us=samples[int(.9 * (len(samples) - 1))])
                    tiled = timings['tiled']['median_us']
                    row = {'qubits': n, 'batch': batch, 'bits': bits, 'bitwise_correct': True,
                           'output_aliases_input': aliases, 'native_route': route_label(trace),
                           'native_program_steps': steps, 'native_trace': trace,
                           'timings': timings,
                           'original_over_tiled': timings['original']['median_us'] / tiled,
                           'native_over_tiled': timings['native']['median_us'] / tiled}
                    report['results'].append(row)
                    write_report(args.output, report)
                    print(f"q={n} B={batch} bits={bits}: "
                          f"{timings['original']['median_us']:.3f} / {tiled:.3f} / "
                          f"{timings['native']['median_us']:.3f} us", flush=True)
    report['complete'] = True
    write_report(args.output, report)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
