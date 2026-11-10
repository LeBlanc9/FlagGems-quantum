"""Compare CX implementations using warmed CUDA-graph device timing.

This measures repeated execution on the same buffers without cache flushing.
Compilation, tensor preparation, validation, and Python dispatch are outside the
captured work. It is not end-to-end circuit timing. The FlagQuantum Triton entry
is called directly; the other baseline is its unfused Torch permutation path.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import triton
import triton.testing

import flagquantum as fq
from flagquantum.kernels.triton.statevector_gates import apply_complex64_local_cx_inplace
from flagquantum.simulation.statevector.operations import _apply_cx_permutation

import flag_gems_quantum as fgq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[16, 20, 22])
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--bits",
        type=int,
        nargs=2,
        action="append",
        metavar=("CONTROL", "TARGET"),
        help="Bit pair to measure; repeat for multiple pairs (default: three layouts)",
    )
    parser.add_argument("--replay-ms", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.qubits) < 2 or min(args.batch) < 1 or args.replay_ms < 1:
        parser.error("qubits must be >= 2; batch and replay-ms must be positive")
    if args.bits and any(
        c == t or min(c, t) < 0 or max(c, t) >= min(args.qubits)
        for c, t in args.bits
    ):
        parser.error("bits must be distinct and valid for every requested qubit count")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("an available CUDA-interface device is required")

    report = {
        "scope": "warmed CUDA-graph GPU time per CX; same buffers, no cache flush",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "flagquantum_version": fq.__version__,
        "flag_gems_quantum_version": fgq.__version__,
        "dtype": "complex64",
        "cx_access_policy": "control_one_pairs_only",
        "replay_ms": args.replay_ms,
        "implementations": {
            "ours_generic": "apply_matrix_2q with out=state and a prepared CX matrix",
            "ours_cx": "apply_cx_inplace",
            "fq_triton_cx": "direct apply_complex64_local_cx_inplace; no dispatch",
            "fq_torch_cx": "_apply_cx_permutation: Torch flip/stack, out of place",
        },
        "results": [],
    }
    print(f"{report['device_name']} | complex64 | CUDA-graph GPU time per CX", flush=True)
    print("Bits use LSB=0. FlagQuantum's Triton kernel is a direct-call baseline.")
    print(" q   B   c   t   generic_us  special_us  FQ_Triton_us  FQ_Torch_us  generic/special")
    with torch.no_grad(), torch.cuda.device(device):
        matrix = torch.tensor(
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)),
            dtype=torch.complex64,
            device=device,
        )
        generator = torch.Generator().manual_seed(2026)
        for n in args.qubits:
            size = 1 << n
            pairs = args.bits or list(
                dict.fromkeys(((n - 1, 0), (0, n - 1), (n // 2, n // 2 - 1)))
            )
            for batch in args.batch:
                source_cpu = torch.randn(
                    (batch, size), dtype=torch.complex64, generator=generator
                )
                state = source_cpu.to(device)
                indices = torch.arange(size)
                for control, target in pairs:
                    mapped = indices ^ (((indices >> control) & 1) << target)
                    expected = source_cpu[:, mapped]
                    calls = {
                        "ours_generic": partial(
                            fgq.apply_matrix_2q, state, matrix, control, target, out=state
                        ),
                        "ours_cx": partial(fgq.apply_cx_inplace, state, control, target),
                        "fq_triton_cx": partial(
                            apply_complex64_local_cx_inplace,
                            state,
                            control_bit_position=control,
                            target_bit_position=target,
                        ),
                        "fq_torch_cx": partial(
                            _apply_cx_permutation,
                            state,
                            (n - 1 - control, n - 1 - target),
                            n,
                        ),
                    }
                    timings = {}
                    names = list(calls)
                    # Rotate order across cases to reduce systematic order effects.
                    rotation = len(report["results"]) % len(names)
                    for name in names[rotation:] + names[:rotation]:
                        state.copy_(source_cpu)
                        actual = calls[name]()
                        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                        del actual
                        state.copy_(source_cpu)
                        median, p10, p90 = triton.testing.do_bench_cudagraph(
                            calls[name], rep=args.replay_ms, quantiles=[0.5, 0.1, 0.9]
                        )
                        timings[name] = {
                            "median_us": median * 1000,
                            "p10_us": p10 * 1000,
                            "p90_us": p90 * 1000,
                        }
                    values = [timings[name]["median_us"] for name in calls]
                    ratio = values[0] / values[1]
                    print(
                        f"{n:2} {batch:3} {control:3} {target:3} "
                        f"{values[0]:12.3f} {values[1]:11.3f} {values[2]:13.3f} "
                        f"{values[3]:12.3f} {ratio:16.3f}x",
                        flush=True,
                    )
                    report["results"].append(
                        {
                            "qubits": n,
                            "batch": batch,
                            "control_bit": control,
                            "target_bit": target,
                            "correctness": "exact match with independent CPU gather",
                            "timings": timings,
                            "generic_over_special": ratio,
                            "fq_triton_over_special": values[2] / values[1],
                        }
                    )
    print("All outputs matched exactly. These timings exclude host overhead and fusion.")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
