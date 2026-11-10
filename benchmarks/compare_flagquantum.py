"""Compare warmed statevector execution through the same FlagQuantum API.

The baseline is FlagQuantum's existing pytorch backend, including its normal
kernel dispatch and fusion. Plans are built once; every fq.run(plan) recomputes
the state. Synchronized wall time includes initialization, gate preparation,
execution, and result packaging, but excludes planning and backend registration.
The first call can use existing compiler caches; it is not guaranteed cold JIT.
"""

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
import triton

import flagquantum as fq
from flagquantum.benchmarking.simulator_workload_corpus import (
    WORKLOAD_NAMES,
    build_workload,
)

import flag_gems_quantum as fgq
from flag_gems_quantum.integrations import flagquantum as adapter
from flag_gems_quantum.integrations.flagquantum import use_backend


def make_circuit(qubits: int, layers: int, batch: int, workload: str = "hcx") -> fq.Circuit:
    """Build a local circuit or reuse an upstream corpus circuit at complex64."""
    if workload in WORKLOAD_NAMES:
        # Preserve the upstream gates, angles, seed, and fixed depths. Both
        # backends receive the same IR; only precision and batch are adapted.
        source = build_workload(workload, n_qubits=qubits)
        return fq.Circuit.from_ir(source.to_ir(), dtype=torch.complex64, bsz=batch)
    if workload not in ("hcx", "mixed"):
        raise ValueError(f"unknown workload: {workload}")
    circuit = fq.Circuit(qubits, bsz=batch, dtype=torch.complex64)
    for layer in range(layers):
        for wire in range(layer % 2, qubits, 2):
            circuit.h(wire)
        for wire in range(layer % 2, qubits - 1, 2):
            circuit.cx(wire, wire + 1)
        if workload == "mixed":
            for wire in range(qubits):
                gate = ("x", "z", "s", "t", "rx", "ry", "rz")[(wire + layer) % 7]
                if gate in {"rx", "ry", "rz"}:
                    getattr(circuit, gate)(wire, 0.37 + 0.13 * layer)
                else:
                    getattr(circuit, gate)(wire)
            for wire in range(layer % 2, qubits - 1, 2):
                gate = ("cz", "swap", "rzz")[(wire // 2 + layer) % 3]
                if gate == "rzz":
                    circuit.rzz(wire, wire + 1, 0.51)
                else:
                    getattr(circuit, gate)(wire, wire + 1)
    return circuit


def timed_run(
    plan: fq.ExecutionPlan,
    device: torch.device,
    cx_override: Callable[[torch.Tensor, int, int], torch.Tensor] | None = None,
) -> tuple[fq.ExecutionResult, float]:
    context = (
        patch.dict(adapter._GATE_DISPATCH, {"cx": ((), cx_override)})
        if cx_override is not None
        else nullcontext()
    )
    with context:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = fq.run(plan)
        torch.cuda.synchronize(device)
        return result, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument(
        "--layers", type=int, default=2,
        help="Depth for hcx/mixed; corpus depths are fixed upstream",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workload", choices=("hcx", "mixed", *WORKLOAD_NAMES), default="hcx")
    parser.add_argument("--output", type=Path, help="Optional JSON result file")
    parser.add_argument(
        "--compare-generic-cx",
        action="store_true",
        help="Also run our same backend with generic CX and a prebuilt 4x4 matrix",
    )
    args = parser.parse_args()
    if args.qubits < 2 or min(args.layers, args.batch, args.repeats) < 1:
        parser.error("qubits must be >= 2; layers, batch, and repeats must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    corpus_workload = args.workload in WORKLOAD_NAMES
    if corpus_workload and (args.qubits < 4 or args.layers != 2):
        parser.error(
            "corpus workloads require qubits >= 4 and retain their fixed "
            "upstream depths; omit --layers"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this backend comparison requires an available CUDA device")

    program = make_circuit(args.qubits, args.layers, args.batch, args.workload).to_ir()
    print(
        f"{torch.cuda.get_device_name(device)} | qubits={args.qubits} "
        f"workload={args.workload} gates={len(program.instructions)} "
        f"batch={args.batch} complex64",
        flush=True,
    )
    rows = {}
    plans = {}
    overrides = {}
    with torch.no_grad(), torch.cuda.device(device), use_backend(device=str(device)) as ours:
        variants = [
            ("pytorch", replace(ours, backend="pytorch")),
            ("flag_gems_quantum", ours),
        ]
        if args.compare_generic_cx:
            cx_matrix = torch.tensor(
                ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)),
                dtype=torch.complex64,
                device=device,
            )

            def generic_cx(
                state: torch.Tensor, control: int, target: int
            ) -> torch.Tensor:
                return fgq.apply_matrix_2q(state, cx_matrix, control, target, out=state)

            overrides["flag_gems_generic_cx"] = generic_cx
            variants.append(("flag_gems_generic_cx", ours))
        for name, options in variants:
            start = time.perf_counter()
            plans[name] = fq.plan(program, options=options)
            rows[name] = {
                "plan_ms": (time.perf_counter() - start) * 1000,
                "compiled_gate_count": plans[name].analysis.n_instructions,
                "samples_ms": [],
            }

        initial = {}
        for name, plan in plans.items():
            initial[name], rows[name]["first_call_ms"] = timed_run(
                plan, device, overrides.get(name)
            )
            rows[name]["runtime"] = dict(initial[name].runtime)

        reference = initial["pytorch"].state
        max_error = 0.0
        max_relative_l2_error = 0.0
        max_norm_error = 0.0
        reference_norm = torch.linalg.vector_norm(reference).item()
        for name in plans:
            candidate = initial[name].state
            torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)
            max_error = max(max_error, (candidate - reference).abs().max().item())
            relative_l2_error = (
                torch.linalg.vector_norm(candidate - reference).item() / reference_norm
            )
            norm_error = (torch.linalg.vector_norm(candidate, dim=-1) - 1).abs().max().item()
            if relative_l2_error > 2e-5 or norm_error > 2e-5:
                raise AssertionError(
                    f"{name}: relative L2 error={relative_l2_error}, norm error={norm_error}"
                )
            max_relative_l2_error = max(max_relative_l2_error, relative_l2_error)
            max_norm_error = max(max_norm_error, norm_error)
        del initial, reference, candidate
        print(
            f"Correctness passed; max amplitude error={max_error:.3g}, "
            f"relative L2 error={max_relative_l2_error:.3g}", flush=True,
        )

        # Rotate the order to reduce systematic order/thermal bias.
        names = tuple(plans)
        for iteration in range(args.warmup + args.repeats):
            rotation = iteration % len(names)
            for name in names[rotation:] + names[:rotation]:
                result, elapsed = timed_run(plans[name], device, overrides.get(name))
                if iteration >= args.warmup:
                    rows[name]["samples_ms"].append(elapsed)
                del result

    print("\nbackend                 plan_ms   first_ms   median_ms    p10_ms    p90_ms")
    for name, row in rows.items():
        samples = sorted(row["samples_ms"])
        row["median_ms"] = statistics.median(samples)
        row["p10_ms"] = samples[int(0.1 * (len(samples) - 1))]
        row["p90_ms"] = samples[int(0.9 * (len(samples) - 1))]
        print(
            f"{name:22} {row['plan_ms']:9.3f} {row['first_call_ms']:10.3f} "
            f"{row['median_ms']:11.3f} {row['p10_ms']:9.3f} {row['p90_ms']:9.3f}"
        )
    speedup = rows["pytorch"]["median_ms"] / rows["flag_gems_quantum"]["median_ms"]
    print(f"\nBaseline / ours: {speedup:.3f}x (>1 means FlagGems-Quantum is faster)")
    if args.compare_generic_cx:
        cx_speedup = (
            rows["flag_gems_generic_cx"]["median_ms"]
            / rows["flag_gems_quantum"]["median_ms"]
        )
        print(f"Generic CX / specialized CX, same backend: {cx_speedup:.3f}x")
        print("Generic CX uses a matrix prepared outside timing; override setup is excluded.")
    print("Baseline keeps existing FlagQuantum optimizations; timing excludes planning.")
    print("First-call time is reported separately and may use existing compiler caches.")

    if args.output is not None:
        report = {
            "scope": "fq.run(prebuilt_plan), synchronized wall time, final state only",
            "distribution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "flagquantum_version": fq.__version__,
            "flag_gems_quantum_version": fgq.__version__,
            "qubits": args.qubits,
            "layers": None if corpus_workload else args.layers,
            "gate_count": len(program.instructions),
            "batch": args.batch,
            "dtype": "complex64",
            "cx_access_policy": "control_one_pairs_only",
            "h_implementation": "apply_h_inplace",
            "workload": args.workload,
            "workload_source": (
                "flagquantum.benchmarking.simulator_workload_corpus.build_workload"
                if corpus_workload else "make_circuit"
            ),
            "program_hash": program.content_hash,
            "opcode_counts": dict(sorted(Counter(gate.name for gate in program).items())),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "max_amplitude_error": max_error,
            "max_relative_l2_error": max_relative_l2_error,
            "max_state_norm_error": max_norm_error,
            "baseline_over_ours": speedup,
            "compare_generic_cx": args.compare_generic_cx,
            "generic_cx_matrix_preparation_excluded": args.compare_generic_cx,
            "results": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
