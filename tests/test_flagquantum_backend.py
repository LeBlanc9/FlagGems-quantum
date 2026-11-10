"""Small state-only integration slice; FlagQuantum is an optional test dependency."""

import importlib.util
import os
import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import Mock, patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

HAS_TRITON = importlib.util.find_spec("triton") is not None
HAS_FLAGQUANTUM = importlib.util.find_spec("flagquantum") is not None
DEVICE = torch.device(os.environ.get("QUANTUM_TEST_DEVICE", "cuda"))
HAS_CUDA = DEVICE.type == "cuda" and torch.cuda.is_available()

if HAS_TRITON:
    from flag_gems_quantum import initialize_basis_state
if HAS_TRITON and HAS_FLAGQUANTUM:
    import flagquantum as fq
    from flagquantum.core.ir import Instruction, MeasurementNode
    from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
    from flagquantum.ecosystem.extensions import ExtensionConfig
    from flagquantum.errors import CapabilityError

    from flag_gems_quantum.integrations import flagquantum as adapter


class _TensorServicesOnly(TorchDispatchMode):
    """Allow allocation, constant transfer and views, but no ATen arithmetic."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        allowed = {
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.empty_strided.default,
            torch.ops.aten.lift_fresh.default,
            torch.ops.aten._to_copy.default,
            torch.ops.aten.view_as_real.default,
        }
        if func not in allowed:
            raise AssertionError(f"unexpected tensor computation: {func}")
        return func(*args, **(kwargs or {}))


class BasisInitializationTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TRITON and HAS_CUDA, "Requires Triton and CUDA")
    def test_initialization_overwrites_both_components(self):
        for shape in ((2,), (3, 8), (2, 1024), (0, 8)):
            for basis_index in (0, shape[-1] - 1):
                with self.subTest(shape=shape, basis_index=basis_index):
                    out = torch.full(
                        shape, complex(3, 7), dtype=torch.complex64, device=DEVICE
                    )
                    expected = torch.zeros(shape, dtype=torch.complex64)
                    expected[..., basis_index] = 1
                    with _TensorServicesOnly():
                        actual = initialize_basis_state(out, basis_index)
                    self.assertIs(actual, out)
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

        tracked = torch.ones(8, dtype=torch.complex64, device=DEVICE, requires_grad=True)
        for enabled in (True, False):
            with torch.set_grad_enabled(enabled), _TensorServicesOnly():
                with self.assertRaisesRegex(ValueError, "forward"):
                    initialize_basis_state(tracked)
        torch.testing.assert_close(tracked.cpu(), torch.ones(8, dtype=torch.complex64))


@unittest.skipUnless(HAS_TRITON and HAS_FLAGQUANTUM, "Requires Triton and FlagQuantum")
class FlagQuantumBackendTests(unittest.TestCase):
    def test_unsupported_requests_fail_before_allocation(self):
        backend = adapter.FlagGemsQuantumBackend()
        backend.start(ExtensionConfig())
        program = fq.Circuit(2).h(0).cx(0, 1).to_ir()
        options = {
            "mode": "statevector",
            "world_size": 1,
            "require_gradients": False,
            "dtype": "complex64",
            "device": "cuda",
            "batch_size": 1,
            "target": "full_state",
        }
        cases = [
            (replace(program, instructions=(Instruction("custom", (0,), matrix=[[1, 0], [0, 1]]),)),
             options),
            (replace(program, instructions=(Instruction("x", (0,), matrix=[[1, 0], [0, 1]]),)),
             options),
            (fq.Circuit(2).rx(0, [0.1, 0.2]).to_ir(), options),
            (fq.Circuit(2).rz(0, torch.tensor(0.3, dtype=torch.float64)).to_ir(), options),
            (fq.Circuit(2).ry(0, torch.tensor(0.2, requires_grad=True)).to_ir(), options),
            (fq.Circuit(2).crx(0, 1, [0.1, 0.2]).to_ir(), options),
            (fq.Circuit(2).cry(1, 0, torch.tensor(0.3, dtype=torch.float64)).to_ir(), options),
            (fq.Circuit(2).crz(0, 1, torch.tensor(0.2, requires_grad=True)).to_ir(), options),
            (
                replace(program, measurements=(MeasurementNode("probabilities", (0, 1)),)),
                options,
            ),
            (program, {**options, "require_gradients": True}),
            (program, {**options, "dtype": "complex128"}),
            (program, {**options, "device": "cpu"}),
        ]
        cases.extend(
            (replace(program, instructions=(Instruction(
                name, (0,), {parameter: 0.1 for parameter in schema.parameters}
            ),)), options)
            for name, schema in OPERATOR_SCHEMAS.items() if schema.channel
        )
        cases.extend(
            (replace(program, instructions=(Instruction(
                name, tuple(range(schema.arity)),
                {parameter: 0.1 for parameter in schema.parameters}
            ),)), options)
            for name, schema in OPERATOR_SCHEMAS.items()
            if schema.unitary and name not in adapter._GATE_DISPATCH
        )
        with patch.object(
            adapter.torch, "empty",
            side_effect=AssertionError("allocated before preflight"),
        ):
            for candidate, settings in cases:
                with self.subTest(settings=settings), torch.no_grad():
                    with self.assertRaises(CapabilityError):
                        backend.execute(candidate, options=settings)
        backend.close()

    @unittest.skipUnless(HAS_CUDA, "Requires CUDA")
    def test_supported_gate_routes_against_cpu_matrices(self):
        from test_apply_matrix import reference
        from test_specialized_gates import GATES, gate_matrix

        schemas = {name: schema for name, schema in OPERATOR_SCHEMAS.items() if schema.unitary}
        self.assertEqual(set(adapter._GATE_DISPATCH), set(GATES))
        self.assertTrue(set(GATES).issubset(schemas))
        schemas = {name: schema for name, schema in schemas.items() if name in GATES}
        # Nonzero complex amplitudes in every control/target sector. Test each
        # gate separately so consecutive inverse gates cannot hide dispatch bugs.
        prefix = (fq.Circuit(3, bsz=3).h(0).ry(1, 0.41).rx(2, -0.63)
                  .phase(0, 0.27).to_ir())
        parameter_data = {
            "theta": torch.tensor([0.37, -0.42, 1.23], device=DEVICE),
            "phi": torch.tensor(0.61),
            "lbd": [-0.7, 0.91, 0.19],
        }
        gates = []
        for name, schema in schemas.items():
            self.assertEqual(adapter._GATE_DISPATCH[name][0], schema.parameters)
            wires = {1: (2,), 2: (2, 0), 3: (1, 2, 0)}[schema.arity]
            params = {key: parameter_data[key] for key in schema.parameters}
            gates.append(Instruction(name, wires, params))
            for alias in schema.aliases:
                self.assertEqual(Instruction(alias, wires, params).name, name)
        programs = [replace(prefix, instructions=prefix.instructions + (gate,)) for gate in gates]
        programs.append(replace(prefix, instructions=prefix.instructions + tuple(gates)))

        backend = adapter.FlagGemsQuantumBackend()
        backend.start(ExtensionConfig())
        self.addCleanup(backend.close)
        settings = {
            "mode": "statevector", "world_size": 1, "require_gradients": False,
            "dtype": "complex64", "device": str(DEVICE), "batch_size": 3,
            "target": "full_state",
        }
        execute = adapter.FlagGemsQuantumBackend.execute

        def guarded_execute(backend, program, *, options):
            with _TensorServicesOnly():
                return execute(backend, program, options=options)

        with adapter.use_backend(device=str(DEVICE)) as options:
            for program in programs:
                with self.subTest(gate=program.instructions[-1].name, length=len(program.instructions)):
                    expected = torch.zeros((3, 8), dtype=torch.complex128)
                    expected[:, 0] = 1
                    for gate in program:
                        values = {key: torch.as_tensor(value, dtype=torch.float32).cpu().reshape(-1).tolist()
                                  for key, value in gate.params.items()}
                        matrices = torch.stack([
                            gate_matrix(gate.name, **{key: value[0 if len(value) == 1 else b]
                                                     for key, value in values.items()})
                            for b in range(3)
                        ])
                        expected = reference(expected, matrices, tuple(2 - wire for wire in gate.wires))
                    # Direct IR retains every opcode; fq.run also exercises
                    # upstream compiler lowering and the public backend selection.
                    with _TensorServicesOnly():
                        actual = backend.execute(program, options=settings)
                    torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                                               rtol=1e-5, atol=1e-5)
                    plan = fq.plan(program, options=replace(options, batch_size=3))
                    with patch("flagquantum.simulation.statevector.local.state",
                               side_effect=AssertionError("host evolution")), patch.object(
                                   adapter.FlagGemsQuantumBackend, "execute", guarded_execute):
                        result = fq.run(plan)
                    torch.testing.assert_close(result.state.cpu().to(torch.complex128), expected,
                                               rtol=1e-5, atol=1e-5)
                    self.assertEqual(result.runtime["backend"], "flag_gems_quantum")

    @unittest.skipUnless(HAS_CUDA, "Requires CUDA")
    def test_state_only_execution_uses_owned_kernels(self):
        with adapter.use_backend(device=str(DEVICE)) as options:
            options = replace(options, batch_size=3)
            # The last case distinguishes control/target order; sorting wires
            # would incorrectly entangle qubits 0 and 2.
            cases = [
                (fq.Circuit(2, bsz=3).h(0).cx(0, 1).to_ir(), (0, 3)),
                (fq.Circuit(3, bsz=3).h(0).cx(0, 2).to_ir(), (0, 5)),
                (fq.Circuit(3, bsz=3).h(0).cx(2, 0).to_ir(), (0, 4)),
            ]
            with ExitStack() as stack:
                calls = [
                    stack.enter_context(patch.object(
                        adapter, "initialize_basis_state", wraps=adapter.initialize_basis_state
                    ))
                ]
                for gate in ("h", "cx"):
                    parameters, operator = adapter._GATE_DISPATCH[gate]
                    observed = Mock(wraps=operator)
                    stack.enter_context(patch.dict(
                        adapter._GATE_DISPATCH, {gate: (parameters, observed)}
                    ))
                    calls.append(observed)
                stack.enter_context(patch(
                    "flagquantum.simulation.statevector.local.state",
                    side_effect=AssertionError("host evolution"),
                ))
                for program, indices in cases:
                    plan = fq.plan(program, options=options)
                    with _TensorServicesOnly():
                        result = fq.run(plan)
                    expected = torch.zeros(
                        (3, 1 << program.n_wires), dtype=torch.complex64
                    )
                    expected[:, list(indices)] = 0.7071067811865476
                    torch.testing.assert_close(
                        result.state.cpu(), expected, rtol=1e-5, atol=1e-5
                    )
                    self.assertEqual(result.state.dtype, torch.complex64)
                    self.assertEqual(result.runtime["backend"], "flag_gems_quantum")
                    self.assertEqual(
                        result.runtime["distribution_semantics"], "single_device_fast_path"
                    )
                    self.assertEqual(result.measurements, ())
                self.assertEqual([call.call_count for call in calls], [3, 3, 3])

                # Public output requests must reach backend preflight via the
                # existing IR, before Runtime can run its measurement path.
                with self.assertRaisesRegex(CapabilityError, "final state"):
                    fq.run(cases[0][0], options=options, outputs=fq.probabilities())
                self.assertEqual(calls[0].call_count, 3)

    @unittest.skipUnless(HAS_CUDA, "Requires CUDA")
    def test_mixed_parameterized_circuit_uses_owned_kernels(self):
        from test_apply_matrix import reference
        from test_specialized_gates import gate_matrix

        execute = adapter.FlagGemsQuantumBackend.execute
        backend = adapter.FlagGemsQuantumBackend()
        backend.start(ExtensionConfig())
        settings = {
            "mode": "statevector", "world_size": 1, "require_gradients": False,
            "dtype": "complex64", "device": str(DEVICE), "batch_size": 3,
            "target": "full_state",
        }

        def guarded_execute(backend, program, *, options):
            # FlagQuantum deserializes and validates parameter tensors before
            # entering the backend. Guard our entire numerical execution path.
            with _TensorServicesOnly():
                return execute(backend, program, options=options)

        for angle in (0.7, [0.0, -0.4, 2.1], torch.tensor([0.2]),
                      torch.tensor([0.1, 0.4, -0.9], device=DEVICE)):
            circuit = (fq.Circuit(3, bsz=3).h(0).h(2).x(1).s(0).t(2)
                       .rx(2, angle).ry(0, angle).cx(2, 0).swap(2, 1)
                       .rz(1, angle).rzz(2, 0, angle).cz(0, 2)
                       .crx(2, 0, angle).cry(0, 1, angle).crz(1, 2, angle).z(2).h(1)
                       .u3(1, theta=0.37, phi=-0.42, lbd=0.63).phase(0, angle))
            program = circuit.to_ir()
            expected = torch.zeros((3, 8), dtype=torch.complex128)
            expected[:, 0] = 1
            for gate in program:
                theta = gate.params.get("theta", 0.0)
                values = torch.as_tensor(theta, dtype=torch.float32).cpu().reshape(-1).tolist()
                matrices = [gate_matrix(gate.name, value, phi=gate.params.get("phi"),
                                        lbd=gate.params.get("lbd")) for value in values]
                matrix = matrices[0] if len(matrices) == 1 else torch.stack(matrices)
                expected = reference(expected, matrix, tuple(2 - w for w in gate.wires))
            # Exercise every named route before compiler fusion can replace it.
            with _TensorServicesOnly():
                actual = backend.execute(program, options=settings)
            torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                                       rtol=1e-5, atol=1e-5)
            with adapter.use_backend(device=str(DEVICE)) as options:
                plan = fq.plan(program, options=replace(options, batch_size=3))
                with patch("flagquantum.simulation.statevector.local.state",
                           side_effect=AssertionError("host evolution")), patch.object(
                               adapter.FlagGemsQuantumBackend, "execute", guarded_execute):
                    result = fq.run(plan)
                torch.testing.assert_close(result.state.cpu().to(torch.complex128), expected,
                                           rtol=1e-5, atol=1e-5)
        backend.close()


if __name__ == "__main__":
    unittest.main()
