"""Dedicated gates against independent CPU complex128 matrix application."""

import cmath
import math
import unittest
from itertools import combinations, permutations

import torch

from test_apply_matrix import DEVICE, HAS_DEVICE, HAS_TRITON, reference
from test_flagquantum_backend import _TensorServicesOnly

if HAS_TRITON:
    import flag_gems_quantum as fgq


GATES = ("i", "h", "x", "y", "sx", "sxdg", "swap", "cx", "cy", "ccx", "cswap",
         "z", "s", "sdg", "t", "tdg", "cz", "rx", "ry", "rz", "rxx", "ryy", "rzz",
         "u1", "u2", "u3", "phase", "cphase", "crx", "cry", "crz")
ROTATIONS = {"rx", "ry", "rz", "rxx", "ryy", "rzz", "u1", "u2", "u3", "phase",
             "cphase", "crx", "cry", "crz"}
TWO_QUBIT = {"swap", "cx", "cy", "cz", "rxx", "ryy", "rzz", "cphase", "crx", "cry", "crz"}
THREE_QUBIT = {"ccx", "cswap"}


def gate_matrix(gate, theta=0.0, *, phi=None, lbd=None):
    """Offline high-precision reference, independent of GPU implementation."""
    if gate in {"cy", "crx", "cry", "crz"}:
        return torch.block_diag(torch.eye(2, dtype=torch.complex128),
                                gate_matrix(gate[1:], theta))
    if gate == "ccx":
        return torch.block_diag(torch.eye(6, dtype=torch.complex128), gate_matrix("x"))
    if gate == "cswap":
        return torch.block_diag(torch.eye(4, dtype=torch.complex128), gate_matrix("swap"))
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if gate in {"rxx", "ryy"}:
        pauli = gate_matrix(gate[1])
        return c * torch.eye(4, dtype=torch.complex128) - 1j * s * torch.kron(pauli, pauli)
    if gate == "i":
        matrix = [[1, 0], [0, 1]]
    elif gate == "h":
        matrix = [[1 / math.sqrt(2), 1 / math.sqrt(2)],
                  [1 / math.sqrt(2), -1 / math.sqrt(2)]]
    elif gate == "x":
        matrix = [[0, 1], [1, 0]]
    elif gate == "y":
        matrix = [[0, -1j], [1j, 0]]
    elif gate in {"sx", "sxdg"}:
        a = (1 + (1j if gate == "sx" else -1j)) / 2
        matrix = [[a, a.conjugate()], [a.conjugate(), a]]
    elif gate == "swap":
        matrix = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    elif gate == "cx":
        matrix = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    elif gate == "rx":
        matrix = [[c, -1j * s], [-1j * s, c]]
    elif gate == "ry":
        matrix = [[c, -s], [s, c]]
    elif gate in {"u2", "u3"}:
        phi = theta if phi is None else phi
        lbd = theta if lbd is None else lbd
        if gate == "u2":
            c = s = 1 / math.sqrt(2)
        matrix = [[c, -cmath.exp(1j * lbd) * s],
                  [cmath.exp(1j * phi) * s, cmath.exp(1j * (phi + lbd)) * c]]
    else:
        diagonals = {
            "z": [1, -1], "s": [1, 1j], "sdg": [1, -1j],
            "t": [1, cmath.exp(1j * math.pi / 4)],
            "tdg": [1, cmath.exp(-1j * math.pi / 4)],
            "cz": [1, 1, 1, -1],
            "phase": [1, cmath.exp(1j * theta)],
            "u1": [1, cmath.exp(1j * theta)],
            "cphase": [1, 1, 1, cmath.exp(1j * theta)],
            "rz": [cmath.exp(-0.5j * theta), cmath.exp(0.5j * theta)],
            "rzz": [cmath.exp(-0.5j * theta), cmath.exp(0.5j * theta),
                    cmath.exp(0.5j * theta), cmath.exp(-0.5j * theta)],
        }
        return torch.diag(torch.tensor(diagonals[gate], dtype=torch.complex128))
    return torch.tensor(matrix, dtype=torch.complex128)


def apply_gate(state, gate, bits, theta=None):
    if gate == "u3":
        return fgq.apply_u3_inplace(state, *bits, theta, theta, theta)
    if gate == "u2":
        return fgq.apply_u2_inplace(state, *bits, theta, theta)
    parameters = (theta,) if gate in ROTATIONS else ()
    return getattr(fgq, f"apply_{gate}_inplace")(state, *bits, *parameters)


@unittest.skipUnless(HAS_TRITON and HAS_DEVICE, "Requires Triton and an accelerator")
class SpecializedGateTests(unittest.TestCase):
    def test_swap_tiles_preserve_bits_and_storage(self):
        generator = torch.Generator().manual_seed(2049)
        # Exercise targets inside, across, and outside an 8-amplitude tile,
        # including partial tiles, offset storage and independent batch rows.
        for n in (2, 4, 5, 8, 9, 11):
            size = 1 << n
            for shape in ((size,), (3, size)):
                raw = torch.randint(-(1 << 63), (1 << 63) - 1, shape,
                                    dtype=torch.int64, generator=generator)
                raw.reshape(-1)[:4] = torch.tensor(
                    [0, -(1 << 63), 0x7FC12345, 0x80000000], dtype=torch.int64)
                source = raw.view(torch.complex64)
                batch = 1 if len(shape) == 1 else shape[0]
                storage = torch.full((source.numel() + 2,), 7 + 9j,
                                     dtype=torch.complex64, device=DEVICE)
                state = storage[1:-1].view(shape)
                for first, second in combinations(range(n), 2):
                    expected = raw.reshape((batch,) + (2,) * n).transpose(
                        n - first, n - second).reshape(shape)
                    for bits in ((first, second), (second, first)):
                        with self.subTest(shape=shape, bits=bits):
                            state.copy_(source)
                            with _TensorServicesOnly():
                                self.assertIs(fgq.apply_swap_inplace(state, *bits), state)
                            self.assertTrue(torch.equal(state.cpu().view(torch.int64), expected))
                            # SWAP twice is identity, including NaN payloads and signed zeros.
                            fgq.apply_swap_inplace(state, *bits)
                            self.assertTrue(torch.equal(state.cpu().view(torch.int64), raw))
                            self.assertEqual(storage[0].item(), 7 + 9j)
                            self.assertEqual(storage[-1].item(), 7 + 9j)

    def test_gates_against_cpu_matrices(self):
        generator = torch.Generator().manual_seed(2031)
        for gate in GATES:
            arity = 3 if gate in THREE_QUBIT else 2 if gate in TWO_QUBIT else 1
            for size in (1 << arity, 32, 2048):
                n = size.bit_length() - 1
                if arity == 1:
                    targets = {(0,), (n - 1,)}
                elif arity == 2:
                    targets = {(0, n - 1), (n - 1, 0), (n // 2, n // 2 - 1)}
                else:
                    targets = set(permutations((0, n // 2, n - 1)))
                for bits in sorted(targets):
                    for shape in ((size,), (3, size), (0, size)):
                        batch = shape[0] if len(shape) == 2 else 1
                        source = torch.randn(shape, dtype=torch.complex64, generator=generator)
                        angles = [None]
                        if gate in ROTATIONS:
                            angles = [0.73, torch.tensor(0.73), torch.tensor([0.73]),
                                      torch.tensor([0.0, -math.pi, 1e6][:batch])]
                        if gate in {"crx", "cry", "crz"}:
                            # A 2*pi controlled rotation retains a relative -1 phase.
                            angles.append(2 * math.pi)
                        for angle in angles:
                            with self.subTest(gate=gate, shape=shape, bits=bits, angle=angle):
                                # An offset contiguous view catches accidental writes outside state.
                                storage = torch.full((source.numel() + 2,), 7 + 9j,
                                                     dtype=torch.complex64, device=DEVICE)
                                state = storage[1:-1].view(shape)
                                state.copy_(source)
                                if gate in ROTATIONS:
                                    values = torch.as_tensor(angle, dtype=torch.float32).reshape(-1).tolist()
                                    matrix = gate_matrix(gate, values[0]) if len(values) == 1 else (
                                        torch.stack([gate_matrix(gate, value) for value in values])
                                        if values else torch.empty((0, 1 << arity, 1 << arity),
                                                                   dtype=torch.complex128)
                                    )
                                else:
                                    matrix = gate_matrix(gate)
                                expected = (reference(source.to(torch.complex128), matrix, bits)
                                            if batch else source.to(torch.complex128))
                                parameter = angle.to(DEVICE) if isinstance(angle, torch.Tensor) else angle
                                with _TensorServicesOnly():
                                    actual = apply_gate(state, gate, bits, parameter)
                                self.assertIs(actual, state)
                                torch.testing.assert_close(actual.cpu().to(torch.complex128), expected,
                                                           rtol=1e-5, atol=1e-5)
                                if gate in {"cx", "cy", "crx", "cry", "crz", "cphase", "cswap", "ccx"}:
                                    control_zero = ((torch.arange(size) >> bits[0]) & 1) == 0
                                    if gate == "ccx":
                                        control_zero |= ((torch.arange(size) >> bits[1]) & 1) == 0
                                    torch.testing.assert_close(actual.cpu()[..., control_zero],
                                                               source[..., control_zero], rtol=0, atol=0)
                                self.assertEqual(storage[0].item(), 7 + 9j)
                                self.assertEqual(storage[-1].item(), 7 + 9j)

    def test_legacy_diagonal_entry(self):
        source = torch.randn((3, 8), dtype=torch.complex64, device=DEVICE)
        theta = torch.tensor([0.0, -0.7, 2.1], device=DEVICE)
        for gate in ("z", "s", "sdg", "t", "tdg", "cz", "rz", "rzz", "crz",
                     "phase", "u1", "cphase"):
            bits = (0, 2) if gate in TWO_QUBIT else (1,)
            angle = theta if gate in ROTATIONS else None
            direct, legacy = source.clone(), source.clone()
            with self.subTest(gate=gate), _TensorServicesOnly():
                apply_gate(direct, gate, bits, angle)
                self.assertIs(fgq.apply_diagonal_inplace(legacy, gate, bits, angle), legacy)
            torch.testing.assert_close(legacy, direct, rtol=0, atol=0)

    def test_invalid_angles_and_targets(self):
        state = torch.ones((3, 8), dtype=torch.complex64, device=DEVICE)
        invalid = [
            (TypeError, lambda: fgq.apply_h_inplace(state.to(torch.complex128), 0)),
            (ValueError, lambda: fgq.apply_h_inplace(state[:, ::2], 0)),
            (ValueError, lambda: fgq.apply_h_inplace(state.conj(), 0)),
            (ValueError, lambda: fgq.apply_x_inplace(state, 3)),
            (TypeError, lambda: fgq.apply_x_inplace(state, True)),
            (ValueError, lambda: fgq.apply_swap_inplace(state, 1, 1)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "cz", (0,))),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "z", (0,), 0.5)),
            (ValueError, lambda: fgq.apply_crx_inplace(state, 1, 1, 0.5)),
            (TypeError, lambda: fgq.apply_crx_inplace(state, None, 0, 0.5)),
            (ValueError, lambda: fgq.apply_cry_inplace(state, 3, 0, 0.5)),
            (TypeError, lambda: fgq.apply_cry_inplace(state, True, 0, 0.5)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "crz", (0,), 0.5)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "crz", (1, 1), 0.5)),
            (ValueError, lambda: fgq.apply_ccx_inplace(state, 0, 0, 1)),
            (ValueError, lambda: fgq.apply_ccx_inplace(state, 0, 1, 0)),
            (ValueError, lambda: fgq.apply_cswap_inplace(state, 0, 1, 0)),
            (ValueError, lambda: fgq.apply_cswap_inplace(state, 0, 1, 3)),
            (ValueError, lambda: fgq.apply_cy_inplace(state, 1, 1)),
            (ValueError, lambda: fgq.apply_rxx_inplace(state, 1, 1, 0.5)),
            (ValueError, lambda: fgq.apply_ryy_inplace(state, 0, 3, 0.5)),
            (ValueError, lambda: fgq.apply_i_inplace(state, 3)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "cphase", (0,), 0.5)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "sdg", (0,), 0.5)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "tdg", (0,), 0.5)),
            (ValueError, lambda: fgq.apply_diagonal_inplace(state, "unknown", (0,))),
            (ValueError, lambda: fgq.apply_cz_inplace(state, 1, 1)),
            (ValueError, lambda: fgq.apply_crz_inplace(state, 1, 1, 0.5)),
            (ValueError, lambda: fgq.apply_cphase_inplace(state, 3, 1, 0.5)),
        ]
        for angle, error in [
            (None, TypeError), (True, TypeError), (float("nan"), ValueError),
            (float("inf"), ValueError), (1e40, ValueError),
            (torch.tensor(0.2), ValueError),
            (torch.ones(2, device=DEVICE), ValueError),
            (torch.ones((3, 1), device=DEVICE), ValueError),
            (torch.ones(3, dtype=torch.float64, device=DEVICE), TypeError),
            (torch.ones(6, device=DEVICE)[::2], ValueError),
            (torch.ones(3, device=DEVICE, requires_grad=True), ValueError),
            (torch.view_as_real(state).reshape(-1)[:3], ValueError),
        ]:
            for gate in ROTATIONS:
                bits = (0, 1) if gate in TWO_QUBIT else (0,)
                invalid.append((error, lambda a=angle, g=gate, b=bits: apply_gate(state, g, b, a)))
        original = state.clone()
        for error, call in invalid:
            with self.assertRaises(error):
                call()
        torch.testing.assert_close(state, original, rtol=0, atol=0)

        # Gate operators accept tracked inputs only when autograd is disabled.
        tracked = state.clone().requires_grad_()
        theta = torch.zeros(3, device=DEVICE, requires_grad=True)
        identity = torch.eye(2, dtype=torch.complex64, device=DEVICE, requires_grad=True)
        with torch.no_grad(), _TensorServicesOnly():
            fgq.apply_crx_inplace(tracked, 0, 1, theta)
            fgq.apply_matrix_1q(tracked, identity, 0, out=tracked)
        torch.testing.assert_close(tracked, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
