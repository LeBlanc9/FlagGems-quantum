"""Compare dedicated gates to our generic matrix operators on the same buffer.

Warmed CUDA-graph GPU time excludes Python dispatch, parameter/matrix preparation
and compilation. Replays reuse buffers without cache flushing. This is not a
comparison to FlagQuantum's optimized circuit executor; use compare_flagquantum
with --workload mixed for that comparison.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
import triton.testing

from flagquantum.core.ir import Instruction
from flagquantum.simulation.gate_matrix import gate_matrix

import flag_gems_quantum as fgq


GATES = ("h", "x", "swap", "z", "s", "t", "cz", "rx", "ry", "rz", "rzz")
ROTATIONS = {"rx", "ry", "rz", "rzz"}
TWO_QUBIT = {"swap", "cz", "rzz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[16, 20])
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--gates", nargs="+", choices=GATES, default=GATES)
    parser.add_argument("--replay-ms", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.qubits) < 2 or min(args.batch) < 1 or args.replay_ms < 1:
        parser.error("qubits must be >= 2; batch and replay-ms must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("an available CUDA-interface device is required")
    report = {
        "scope": "warmed CUDA-graph GPU time; same buffers, no cache flush",
        "baseline": "our generic apply_matrix_1q/2q, with out=state",
        "matrix_and_parameter_preparation": "excluded from timing",
        "device_name": torch.cuda.get_device_name(device), "device": str(device),
        "torch_version": torch.__version__, "triton_version": triton.__version__,
        "dtype": "complex64", "replay_ms": args.replay_ms, "results": [],
    }
    print(f"{report['device_name']} | complex64 | CUDA-graph GPU time", flush=True)
    print("gate   q   B   bits      generic_us   special_us   generic/special", flush=True)
    generator = torch.Generator().manual_seed(2031)
    with torch.no_grad(), torch.cuda.device(device):
        for n in args.qubits:
            for batch in args.batch:
                source = torch.randn((batch, 1 << n), dtype=torch.complex64, generator=generator)
                state = source.to(device)
                # Angles differ across samples; B=1 also tests the shared tensor path.
                theta_cpu = torch.tensor([0.37 + i * 0.13 for i in range(batch)])
                theta = theta_cpu.to(device)
                for gate in args.gates:
                    arity = 2 if gate in TWO_QUBIT else 1
                    bits_list = ([(0,), (n // 2,), (n - 1,)] if arity == 1 else
                                 [(0, n - 1), (n - 1, 0), (n // 2, n // 2 - 1)])
                    instruction = Instruction(gate, tuple(range(arity)),
                                              {"theta": theta_cpu} if gate in ROTATIONS else {})
                    matrix_cpu = gate_matrix(instruction, bsz=batch, device="cpu",
                                             dtype=torch.complex64)
                    matrix = matrix_cpu.to(device)
                    for bits in dict.fromkeys(bits_list):
                        # Independent CPU tensor-axis reference; no GPU indexing reuse.
                        axes = tuple(n - bit for bit in bits)
                        destinations = tuple(range(n + 1 - arity, n + 1))
                        grouped = source.reshape((batch,) + (2,) * n).movedim(axes, destinations)
                        expected = (grouped.reshape(batch, -1, 1 << arity)
                                    @ matrix_cpu.transpose(-1, -2))
                        expected = expected.reshape(grouped.shape).movedim(destinations, axes).reshape(source.shape)
                        generic = fgq.apply_matrix_1q if arity == 1 else fgq.apply_matrix_2q
                        parameters = {"theta": theta} if gate in ROTATIONS else {}
                        special = partial(getattr(fgq, f"apply_{gate}_inplace"),
                                          state, *bits, **parameters)
                        calls = {"generic": partial(generic, state, matrix, *bits, out=state),
                                 "specialized": special}
                        timings, errors = {}, {}
                        names = list(calls)
                        if len(report["results"]) % 2:
                            names.reverse()
                        for name in names:
                            state.copy_(source)
                            actual = calls[name]()
                            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
                            errors[name] = (actual.cpu() - expected).abs().max().item()
                            state.copy_(source)
                            median, p10, p90 = triton.testing.do_bench_cudagraph(
                                calls[name], rep=args.replay_ms, quantiles=[0.5, 0.1, 0.9])
                            timings[name] = {"median_us": median * 1000,
                                             "p10_us": p10 * 1000, "p90_us": p90 * 1000}
                        ratio = timings["generic"]["median_us"] / timings["specialized"]["median_us"]
                        print(f"{gate:5} {n:2} {batch:3} {str(bits):10} "
                              f"{timings['generic']['median_us']:11.3f} "
                              f"{timings['specialized']['median_us']:12.3f} {ratio:16.3f}x", flush=True)
                        report["results"].append({"gate": gate, "qubits": n, "batch": batch,
                                                   "bits": bits, "timings": timings,
                                                   "max_amplitude_error": errors,
                                                   "generic_over_specialized": ratio})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
