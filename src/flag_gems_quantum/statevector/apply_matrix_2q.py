"""Apply a row-major 4x4 matrix to groups of four complex64 amplitudes.

The group addressing and matrix action are adapted from FlagQuantum's
flagquantum/kernels/triton/statevector_gates.py (Apache-2.0).
"""

import torch
import triton
import triton.language as tl

from .utils import validate_matrix_inputs


@triton.jit
def apply_matrix_2q_kernel(
    state,
    matrix,
    output,
    SIZE: tl.constexpr,
    FIRST_TARGET_BIT: tl.constexpr,
    SECOND_TARGET_BIT: tl.constexpr,
    MATRIX_BATCHED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = group < SIZE // 4

    # Insert zero bits from low to high to enumerate each disjoint group once.
    low_bit = min(FIRST_TARGET_BIT, SECOND_TARGET_BIT)
    high_bit = max(FIRST_TARGET_BIT, SECOND_TARGET_BIT)
    lower = group & ((1 << low_bit) - 1)
    with_low_zero = ((group - lower) << 1) + lower
    lower = with_low_zero & ((1 << high_bit) - 1)
    index00 = batch * SIZE + ((with_low_zero - lower) << 1) + lower

    # Matrix basis order is |first, second>: 00, 01, 10, 11.
    index01 = index00 + (1 << SECOND_TARGET_BIT)
    index10 = index00 + (1 << FIRST_TARGET_BIT)
    index11 = index10 + (1 << SECOND_TARGET_BIT)
    x00_real = tl.load(state + 2 * index00, valid, 0.0)
    x00_imag = tl.load(state + 2 * index00 + 1, valid, 0.0)
    x01_real = tl.load(state + 2 * index01, valid, 0.0)
    x01_imag = tl.load(state + 2 * index01 + 1, valid, 0.0)
    x10_real = tl.load(state + 2 * index10, valid, 0.0)
    x10_imag = tl.load(state + 2 * index10 + 1, valid, 0.0)
    x11_real = tl.load(state + 2 * index11, valid, 0.0)
    x11_imag = tl.load(state + 2 * index11 + 1, valid, 0.0)

    values_real = (x00_real, x01_real, x10_real, x11_real)
    values_imag = (x00_imag, x01_imag, x10_imag, x11_imag)
    indices = (index00, index01, index10, index11)
    base = batch * 32 if MATRIX_BATCHED else 0

    # The full input group is loaded before stores, allowing exact state/output aliasing.
    for row in tl.static_range(4):
        result_real = tl.zeros((BLOCK,), dtype=tl.float32)
        result_imag = tl.zeros((BLOCK,), dtype=tl.float32)
        for column in tl.static_range(4):
            offset = base + 2 * (4 * row + column)
            coefficient_real = tl.load(matrix + offset)
            coefficient_imag = tl.load(matrix + offset + 1)
            result_real += coefficient_real * values_real[column] - coefficient_imag * values_imag[column]
            result_imag += coefficient_real * values_imag[column] + coefficient_imag * values_real[column]
        tl.store(output + 2 * indices[row], result_real, valid)
        tl.store(output + 2 * indices[row] + 1, result_imag, valid)


def apply_matrix_2q(state, matrix, first_target_bit, second_target_bit, out=None):
    """Apply a 4x4 matrix to a contiguous complex64 state on an accelerator.

    ``state`` has shape (N,) or (B, N), with N a power of two. ``matrix``
    has shape (4, 4) for a shared matrix or (B, 4, 4) for per-sample matrices
    (B=1 for an unbatched state). Bit 0 is the least significant index bit.
    The matrix basis is |first, second> in 00, 01, 10, 11 order, even when
    first_target_bit > second_target_bit. ``out`` must match state; pass
    ``out=state`` for an in-place update. Autograd is not supported.
    """
    size, batch = validate_matrix_inputs(
        state, matrix, (first_target_bit, second_target_bit), out
    )
    if out is None:
        out = torch.empty_like(state)
    if batch == 0:
        return out

    block = 256
    grid = (triton.cdiv(size // 4, block), batch)
    with torch.get_device_module(state.device).device(state.device):
        apply_matrix_2q_kernel[grid](
            torch.view_as_real(state),
            torch.view_as_real(matrix),
            torch.view_as_real(out),
            SIZE=size,
            FIRST_TARGET_BIT=first_target_bit,
            SECOND_TARGET_BIT=second_target_bit,
            MATRIX_BATCHED=matrix.ndim == 3,
            BLOCK=block,
            num_warps=4,
        )
    return out
