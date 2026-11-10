"""Controlled-X permutation, bit ordering, and in-place ownership checks."""

import unittest

import torch

from test_apply_matrix import DEVICE, HAS_DEVICE, HAS_TRITON

if HAS_TRITON:
    from flag_gems_quantum import apply_cx_inplace


@unittest.skipUnless(HAS_TRITON, "Triton is not installed")
class CXValidationTests(unittest.TestCase):
    def test_invalid_inputs(self):
        state = torch.ones(8, dtype=torch.complex64)
        cases = [
            (TypeError, "torch.Tensor", [], 0, 1),
            (TypeError, "complex64", state.to(torch.complex128), 0, 1),
            (ValueError, "contiguous", state[::2], 0, 1),
            (ValueError, "resolved", state.conj(), 0, 1),
            (ValueError, "forward", state.clone().requires_grad_(), 0, 1),
            (ValueError, "shape", state.reshape(1, 2, 4), 0, 1),
            (ValueError, "power of two", state[:6], 0, 1),
            (ValueError, "power of two", state[:2], 0, 1),
            (TypeError, "integers", state, True, 1),
            (ValueError, "outside", state, -1, 0),
            (ValueError, "outside", state, 0, 3),
            (ValueError, "distinct", state, 1, 1),
            (ValueError, "accelerator", state, 0, 1),
        ]
        for error, message, candidate, control, target in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    apply_cx_inplace(candidate, control, target)


@unittest.skipUnless(HAS_TRITON and HAS_DEVICE, "Requires Triton and an accelerator")
class CXApplicationTests(unittest.TestCase):
    def test_permutation_and_involution(self):
        generator = torch.Generator().manual_seed(2026)
        for size in (4, 32, 4096):
            n = size.bit_length() - 1
            pairs = {(0, n - 1), (n - 1, 0), (n // 2, n // 2 - 1)}
            pairs.update((c, n - 1) for c in (1, 2, 3) if c < n - 1)
            if size == 32:
                pairs = {(c, t) for c in range(n) for t in range(n) if c != t}
            for shape in ((size,), (3, size), (0, size)):
                source = torch.randn(shape, dtype=torch.complex64, generator=generator)
                indices = torch.arange(size)
                for control, target in sorted(pairs):
                    with self.subTest(shape=shape, control=control, target=target):
                        # A CPU gather independently specifies the permutation.
                        mapped = indices ^ (((indices >> control) & 1) << target)
                        expected = source[..., mapped]
                        # A nonzero storage offset also checks pointer addressing.
                        storage = torch.empty(
                            source.numel() + 2, dtype=source.dtype, device=DEVICE
                        ).fill_(complex(13, 7))
                        state = storage[1:-1].view(shape)
                        state.copy_(source)
                        result = apply_cx_inplace(state, control, target)
                        self.assertIs(result, state)
                        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
                        apply_cx_inplace(state, control, target)
                        torch.testing.assert_close(state.cpu(), source, rtol=0, atol=0)
                        self.assertEqual(storage[0].item(), complex(13, 7))
                        self.assertEqual(storage[-1].item(), complex(13, 7))


if __name__ == "__main__":
    unittest.main()
