"""Compare dedicated gates to FlagQuantum's actual native single-gate dispatch.

Obtain the compiled execution segment from a normal Circuit.state() call, then
time that segment on an existing state. Do not force a reference implementation
or disable native dispatch. CUDA-graph timings include GPU parameter/matrix
construction and state application, but exclude initialization, compilation,
input transfers, Python dispatch and tracing. Cross-gate fusion is out of scope.
U3 uses synchronized wall time for both paths: the inspected native matrix
builder formats a GPU tensor in a log message, which cannot be graph-captured.
"""

import argparse
import cmath
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch
import triton
import triton.testing
from torch.utils._python_dispatch import TorchDispatchMode

import flagquantum as fq
from flagquantum.core.runtime_config import RuntimeConfig, runtime_config
from flagquantum.simulation.statevector import local

import flag_gems_quantum as fgq


GATES = ("h", "x", "z", "s", "t", "cx", "cz", "swap", "rx", "ry", "rz", "rzz", "u3", "phase",
         "crx", "cry", "crz")
TWO_QUBIT = {"cx", "cz", "swap", "rzz", "crx", "cry", "crz"}
PARAMETERS = {"rx": ("theta",), "ry": ("theta",), "rz": ("theta",),
              "crx": ("theta",), "cry": ("theta",), "crz": ("theta",),
              "rzz": ("theta",), "phase": ("theta",), "u3": ("theta", "phi", "lbd")}


def reference_matrix(gate, theta=0.0, phi=0.0, lbd=0.0):
    """CPU complex128 definitions, independent of either implementation."""
    if gate in {"crx", "cry", "crz"}:
        return torch.block_diag(torch.eye(2, dtype=torch.complex128),
                                reference_matrix(gate[1:], theta))
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    fixed = {
        "h": [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]],
        "x": [[0, 1], [1, 0]],
        "cx": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        "swap": [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        "rx": [[c, -1j * s], [-1j * s, c]],
        "ry": [[c, -s], [s, c]],
        "u3": [[c, -cmath.exp(1j * lbd) * s],
               [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lbd)) * c]],
    }
    if gate in fixed:
        return torch.tensor(fixed[gate], dtype=torch.complex128)
    negative, positive = cmath.exp(-0.5j * theta), cmath.exp(0.5j * theta)
    diagonal = {
        "z": [1, -1], "s": [1, 1j], "t": [1, cmath.exp(0.25j * math.pi)],
        "cz": [1, 1, 1, -1], "rz": [negative, positive],
        "rzz": [negative, positive, positive, negative],
        "phase": [1, cmath.exp(1j * theta)],
    }
    return torch.diag(torch.tensor(diagonal[gate], dtype=torch.complex128))


def reference_state(source, gate, bits, parameters):
    batch, size = source.shape
    n, arity = size.bit_length() - 1, len(bits)
    matrices = torch.stack([
        reference_matrix(gate, **{name: values[row].item() for name, values in parameters.items()})
        for row in range(batch)
    ])
    axes = tuple(n - bit for bit in bits)
    destinations = tuple(range(n + 1 - arity, n + 1))
    grouped = source.to(torch.complex128).reshape((batch,) + (2,) * n).movedim(axes, destinations)
    result = grouped.reshape(batch, -1, 1 << arity) @ matrices.transpose(-1, -2)
    return result.reshape(grouped.shape).movedim(destinations, axes).reshape(source.shape)


def native_call(state, gate, bits, parameters, config):
    """Reuse the exact program and bindings produced by normal native setup."""
    n = state.shape[-1].bit_length() - 1
    circuit = fq.Circuit(n, bsz=state.shape[0], inputs=state, device=state.device,
                         dtype=state.dtype, config=config)
    wires = tuple(n - 1 - bit for bit in bits)
    getattr(circuit, gate)(*wires, **parameters)
    execute = local._execute_statevector_program
    captured = []

    def observe(owner, program, output, bindings, **kwargs):
        captured.append((program, bindings))
        return execute(owner, program, output, bindings, **kwargs)

    # Observation only: the native setup and numerical result are unchanged.
    with patch.object(local, "_execute_statevector_program", observe):
        circuit.state(refresh=True)
    if len(captured) != 1:
        raise RuntimeError("Expected one native execution segment for a single gate")
    program, bindings = captured[0]
    return partial(execute, circuit, program, state, bindings, owns_input=False), [
        type(step).__name__ for step in program
    ]


def dedicated_call(state, gate, bits, parameters):
    return partial(getattr(fgq, f"apply_{gate}_inplace"), state, *bits, **parameters)


class TensorTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations[str(func)] += 1
        return func(*args, **(kwargs or {}))


def trace_native(call):
    """Record actual Python kernel entries and ATen ops outside the timed run."""
    entries = []

    def profile(frame, event, arg):
        module = frame.f_globals.get("__name__", "")
        if event == "call" and module.startswith("flagquantum.kernels.triton"):
            name = module + "." + frame.f_code.co_name
            if name not in entries:
                entries.append(name)

    previous = sys.getprofile()
    trace = TensorTrace()
    try:
        sys.setprofile(profile)
        with trace:
            actual = call()
    finally:
        sys.setprofile(previous)
    return actual, {"triton_entries": entries, "aten_operations": dict(trace.operations)}


def route_label(trace):
    entries, ops = trace["triton_entries"], trace["aten_operations"]
    if any("statevector_pauli_rotation" in name for name in entries):
        return "Triton Pauli rotation + Torch parameter preparation"
    if any("statevector_diagonal" in name for name in entries):
        return "Triton diagonal"
    if entries:
        return "Triton (see recorded entries)"
    if "aten.bmm.default" in ops:
        return "Torch bmm + layout operations"
    if "aten.flip.default" in ops:
        return "Torch flip/stack" if "aten.stack.default" in ops else "Torch flip"
    if "aten.transpose.int" in ops:
        return "Torch transpose/reshape"
    return "Torch (see recorded operations)"


def revision(path):
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True, timeout=10).stdout.strip()


def eager_samples(call, device, repeats):
    """Matched synchronized single-gate latency when graph capture is unavailable."""
    for _ in range(5):
        call()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        actual = call()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1e6)
        del actual
    return samples


def write_markdown(report, path):
    lines = [
        "# Dedicated gates versus FlagQuantum native single-gate execution", "",
        f"Device: {report['device_name']} ({report['device']}), complex64, forward only.",
        f"FlagQuantum revision: `{report['flagquantum_revision']}`.", "",
        "Each native path is taken from a normal `Circuit.state()` call; no native",
        "dispatch switch is disabled. GPU parameter/matrix preparation and state",
        "application are included. Initialization, compilation, input transfer,",
        "Python dispatch and tracing are excluded. No cross-gate fusion is measured.",
        "Both paths reuse buffers without cache flushing. Our gates update in place;",
        "the native path retains its normal output allocation and layout behavior.",
        f"Each case pools {report['rounds']} rounds of CUDA-graph replay samples, alternating order.",
    ]
    if any(row["gate"] == "u3" for row in report["results"]):
        lines += [
            "**U3 (marked *)** is the exception: both paths use synchronized wall time, including Python",
            "and synchronization overhead. The native `_ensure_batch_dim` formats a GPU tensor",
            "in an eagerly evaluated debug-log message, making CUDA Graph capture unavailable.",
            "This native behavior is retained. U3's ratio is not a kernel-only speedup.",
        ]
    lines += [
        "Speedup = FlagQuantum time / our time; values below 1 mean ours is slower.", "",
        "## Middle-bit summary", "",
        "One-qubit gates use bit n//2; two-qubit gates use (n//2, n//2-1). Bit 0 is the LSB.", "",
        "| Qubits | Batch | Gate | FlagQuantum path | FQ us | Ours us | Speedup | Range over tested bits |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        n, gate = row["qubits"], row["gate"]
        middle = [n // 2, n // 2 - 1] if gate in TWO_QUBIT else [n // 2]
        if row["bits"] != middle:
            continue
        ratios = [r["native_over_ours"] for r in report["results"]
                  if (r["qubits"], r["batch"], r["gate"]) == (n, row["batch"], gate)]
        times = row["timings"]
        label = gate.upper() + ("*" if row["timing_method"] == "synchronized_wall_time" else "")
        lines.append(f"| {n} | {row['batch']} | {label} | {row['native_route']} | "
                     f"{times['flagquantum']['median_us']:.3f} | {times['ours']['median_us']:.3f} | "
                     f"{row['native_over_ours']:.2f}x | {min(ratios):.2f}–{max(ratios):.2f}x |")
    lines += ["", "## Every measured bit layout", "",
              "| Qubits | Batch | Gate | Bits | FQ us | Ours us | Speedup | Max amplitude error |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in report["results"]:
        times = row["timings"]
        error = max(row["max_amplitude_error"].values())
        label = row["gate"].upper() + ("*" if row["timing_method"] == "synchronized_wall_time" else "")
        lines.append(f"| {row['qubits']} | {row['batch']} | {label} | {row['bits']} | "
                     f"{times['flagquantum']['median_us']:.3f} | {times['ours']['median_us']:.3f} | "
                     f"{row['native_over_ours']:.2f}x | {error:.2e} |")
    lines += ["", f"[Raw samples, parameters and route traces]({path.with_suffix('.json').name}).", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[20])
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--gates", choices=GATES, nargs="+", default=GATES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replay-ms", type=int, default=30)
    parser.add_argument("--eager-repeats", type=int, default=20,
                        help="Samples per round for U3's synchronized wall-time comparison")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.qubits) < 2 or min(args.batch) < 1 or min(args.rounds, args.replay_ms, args.eager_repeats) < 1:
        parser.error("qubits must be >= 2; batch, rounds and replay-ms must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("an available CUDA-interface device is required")
    config = RuntimeConfig(backend="pytorch", device=str(device), complex_dtype="complex64")
    report = {
        "complete": False,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "native single-gate segment vs dedicated operator; GPU time except tagged U3 wall time",
        "parameter_preparation": "GPU numerical preparation included; input transfers excluded",
        "state_initialization": "excluded", "cross_gate_fusion": "not measured",
        "python_overhead": "excluded except U3's matched wall-time comparison",
        "cache_flush": False, "backend": "pytorch",
        "u3_timing_exception": "matched synchronized wall time; native eager tensor-log formatting prevents graph capture",
        "eager_repeats_per_round": args.eager_repeats,
        "distribution_semantics": "single_device_fast_path", "scalability_claim_allowed": False,
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "dtype": "complex64", "gradient_method": "none; forward only",
        "torch_version": torch.__version__, "triton_version": triton.__version__,
        "triton_module": triton.__file__, "python_executable": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "controlled_rotation_access_policy": "all_target_pairs_with_control_selection",
        "flagquantum_revision": revision(Path(fq.__file__).resolve().parents[1]),
        "rounds": args.rounds, "replay_ms": args.replay_ms, "results": [],
    }
    generator = torch.Generator().manual_seed(2032)
    print(f"{report['device_name']} | {device} | complex64 | native dispatch vs dedicated", flush=True)
    print("gate    q  B bits        FQ_us    ours_us  FQ/ours  native path", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), torch.cuda.device(device), runtime_config(config):
        for n in args.qubits:
            for batch in args.batch:
                source = torch.randn((batch, 1 << n), dtype=torch.complex64, generator=generator)
                for gate in args.gates:
                    parameters_cpu = {
                        name: torch.tensor([base + 0.13 * row for row in range(batch)], dtype=torch.float32)
                        for name, base in zip(PARAMETERS.get(gate, ()), (0.37, -0.42, 0.63))
                    }
                    parameters = {name: value.to(device) for name, value in parameters_cpu.items()}
                    layouts = ([(0, n - 1), (n // 2, n // 2 - 1), (n - 1, 0)]
                               if gate in TWO_QUBIT else [(0,), (n // 2,), (n - 1,)])
                    for bits in dict.fromkeys(layouts):
                        method = "synchronized_wall_time" if gate == "u3" else "cuda_graph_gpu"
                        states = {name: source.to(device) for name in ("flagquantum", "ours")}
                        native, steps = native_call(states["flagquantum"], gate, bits, parameters, config)
                        calls = {"flagquantum": native,
                                 "ours": dedicated_call(states["ours"], gate, bits, parameters)}
                        expected = reference_state(source, gate, bits, parameters_cpu)
                        errors, aliases, trace = {}, {}, None
                        for name, call in calls.items():
                            states[name].copy_(source)
                            if name == "flagquantum":
                                actual, trace = trace_native(call)
                            else:
                                actual = call()
                            aliases[name] = actual.data_ptr() == states[name].data_ptr()
                            actual_cpu = actual.cpu().to(torch.complex128)
                            torch.testing.assert_close(actual_cpu, expected, rtol=1e-5, atol=1e-5)
                            errors[name] = (actual_cpu - expected).abs().max().item()
                        del actual, actual_cpu, expected
                        timings = {name: {"samples_us": [], "rounds_us": []} for name in calls}
                        for round_index in range(args.rounds):
                            names = list(calls)
                            if (len(report["results"]) + round_index) % 2:
                                names.reverse()
                            for name in names:
                                states[name].copy_(source)
                                if method == "synchronized_wall_time":
                                    samples = eager_samples(calls[name], device, args.eager_repeats)
                                else:
                                    samples = triton.testing.do_bench_cudagraph(
                                        calls[name], rep=args.replay_ms, return_mode="all")
                                    samples = [sample * 1000 for sample in samples]
                                timings[name]["rounds_us"].append(samples)
                                timings[name]["samples_us"].extend(samples)
                        for value in timings.values():
                            samples = sorted(value["samples_us"])
                            value.update(median_us=statistics.median(samples),
                                         p10_us=samples[int(0.1 * (len(samples) - 1))],
                                         p90_us=samples[int(0.9 * (len(samples) - 1))])
                        ratio = timings["flagquantum"]["median_us"] / timings["ours"]["median_us"]
                        route = route_label(trace)
                        report["results"].append({
                            "qubits": n, "batch": batch, "gate": gate, "bits": list(bits),
                            "timing_method": method,
                            "parameters": {name: value.tolist() for name, value in parameters_cpu.items()},
                            "native_program_steps": steps, "native_route": route, "native_trace": trace,
                            "output_aliases_input": aliases, "max_amplitude_error": errors,
                            "timings": timings, "native_over_ours": ratio,
                        })
                        print(f"{gate:5} {n:3} {batch:2} {str(bits):10} "
                              f"{timings['flagquantum']['median_us']:8.3f} "
                              f"{timings['ours']['median_us']:10.3f} {ratio:7.2f}x  {route}", flush=True)
                        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, args.output.with_suffix(".md"))
    print(f"All {len(report['results'])} cases passed. Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
