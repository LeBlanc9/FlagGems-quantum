"""Run with PYTHONPATH=src python -m unittest discover -s tests -v.

Set QUANTUM_TEST_DEVICE to select an accelerator (default: cuda).
"""

import importlib.util
import os
import unittest

import torch

HAS_TRITON = importlib.util.find_spec("triton") is not None
DEVICE = torch.device(os.environ.get("QUANTUM_TEST_DEVICE", "cuda"))
DEVICE_MODULE = torch.get_device_module(DEVICE)
HAS_DEVICE = DEVICE_MODULE.is_available()

if HAS_TRITON:
    from flag_gems_quantum import apply_matrix_1q, apply_matrix_2q


def reference(state, matrix, bits):
    """Apply the matrix by moving target axes, independently of kernel indexing."""
    size = state.shape[-1]
    n = size.bit_length() - 1
    batch = state.shape[0] if state.ndim == 2 else 1
    axes = tuple(n - bit for bit in bits)
    destinations = tuple(range(n + 1 - len(bits), n + 1))
    tensor = state.reshape((batch,) + (2,) * n)
    grouped = tensor.movedim(axes, destinations)
    result = grouped.reshape(batch, -1, 1 << len(bits)) @ matrix.transpose(-1, -2)
    return result.reshape(grouped.shape).movedim(destinations, axes).reshape(state.shape)


@unittest.skipUnless(HAS_TRITON, "Triton is not installed")
class InputValidationTests(unittest.TestCase):
    def test_invalid_inputs(self):
        state = torch.ones(8, dtype=torch.complex64)
        matrix = torch.eye(2, dtype=torch.complex64)
        cases = [
            (TypeError, "torch.Tensor", lambda: apply_matrix_1q([], matrix, 0)),
            (TypeError, "complex64", lambda: apply_matrix_1q(state.to(torch.complex128), matrix, 0)),
            (ValueError, "contiguous", lambda: apply_matrix_1q(state[::2], matrix, 0)),
            (ValueError, "power of two", lambda: apply_matrix_1q(state[:6], matrix, 0)),
            (TypeError, "integers", lambda: apply_matrix_1q(state, matrix, True)),
            (ValueError, "outside", lambda: apply_matrix_1q(state, matrix, -1)),
            (ValueError, "outside", lambda: apply_matrix_1q(state, matrix, 3)),
            (ValueError, "matrix must have shape", lambda: apply_matrix_1q(state, torch.eye(4, dtype=state.dtype), 0)),
            (ValueError, "same shape", lambda: apply_matrix_1q(state, matrix, 0, out=state[:4])),
            (ValueError, "resolved", lambda: apply_matrix_1q(state.conj(), matrix, 0)),
            (ValueError, "forward", lambda: apply_matrix_1q(state.clone().requires_grad_(), matrix, 0)),
            (ValueError, "accelerator", lambda: apply_matrix_1q(state, matrix, 0)),
            (ValueError, "distinct", lambda: apply_matrix_2q(state, torch.eye(4, dtype=state.dtype), 0, 0)),
        ]
        for error, message, call in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    call()


@unittest.skipUnless(HAS_TRITON and HAS_DEVICE, "Requires Triton and the target accelerator")
class MatrixApplicationTests(unittest.TestCase):
    def test_matrix_action_and_output_modes(self):
        generator = torch.Generator().manual_seed(42)
        for width, operator in ((2, apply_matrix_1q), (4, apply_matrix_2q)):
            for size in (width, 32, 2048):
                n = size.bit_length() - 1
                targets = [(bit,) for bit in (0, n - 1)] if width == 2 else [
                    (0, n - 1), (n - 1, 0), (0, 1)
                ]
                for bits in set(targets):
                    for shape in ((size,), (3, size)):
                        batch = shape[0] if len(shape) == 2 else 1
                        for batched_matrix in (False, True):
                            matrix_shape = (batch, width, width) if batched_matrix else (width, width)
                            state_cpu = torch.randn(shape, dtype=torch.complex64, generator=generator)
                            matrix_cpu = torch.randn(matrix_shape, dtype=torch.complex64, generator=generator)
                            expected = reference(state_cpu, matrix_cpu, bits)
                            matrix = matrix_cpu.to(DEVICE)
                            for mode in ("allocate", "out", "inplace"):
                                with self.subTest(width=width, shape=shape, bits=bits, batched=batched_matrix, mode=mode):
                                    state = state_cpu.to(DEVICE)
                                    out = None if mode == "allocate" else (
                                        state if mode == "inplace" else torch.empty_like(state)
                                    )
                                    result = operator(state, matrix, *bits, out=out)
                                    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-5)
                                    if out is not None:
                                        self.assertIs(result, out)
                                    if mode != "inplace":
                                        torch.testing.assert_close(state.cpu(), state_cpu, rtol=0, atol=0)

    def test_empty_batch(self):
        for width, operator, bits in (
            (2, apply_matrix_1q, (0,)),
            (4, apply_matrix_2q, (1, 0)),
        ):
            state = torch.empty((0, 8), dtype=torch.complex64, device=DEVICE)
            for shape in ((width, width), (0, width, width)):
                matrix = torch.empty(shape, dtype=state.dtype, device=DEVICE)
                self.assertEqual(operator(state, matrix, *bits).shape, state.shape)

    def test_unsafe_output_aliases(self):
        for width, operator, bits in (
            (2, apply_matrix_1q, (0,)),
            (4, apply_matrix_2q, (1, 0)),
        ):
            storage = torch.ones(17, dtype=torch.complex64, device=DEVICE)
            state = storage[:16]
            matrix = torch.eye(width, dtype=state.dtype, device=DEVICE)
            with self.assertRaisesRegex(ValueError, "overlap state"):
                operator(state, matrix, *bits, out=storage[1:])
            aliased_matrix = state[:width * width].view(width, width)
            with self.assertRaisesRegex(ValueError, "overlap matrix"):
                operator(state, aliased_matrix, *bits, out=state)


if __name__ == "__main__":
    unittest.main()
