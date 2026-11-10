"""Apply controlled-X by swapping only the control-one amplitudes."""

import torch
import triton
import triton.language as tl

from .indexing import _insert_zero_bit
from .utils import validate_inplace_state


@triton.jit
def apply_cx_inplace_kernel(
    state,
    SIZE: tl.constexpr,
    CONTROL_BIT: tl.constexpr,
    TARGET_BIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = pair < SIZE // 4

    # Insert two zero bits, then set control=1. Every pair has exactly one
    # owner, so the swaps are safe in place across all programs. Use this same
    # addressing strategy for every control/target position.
    low_bit = min(CONTROL_BIT, TARGET_BIT)
    high_bit = max(CONTROL_BIT, TARGET_BIT)
    with_low_zero = _insert_zero_bit(pair, low_bit)
    index0 = _insert_zero_bit(with_low_zero, high_bit) + (1 << CONTROL_BIT)
    index0 += batch * SIZE
    index1 = index0 + (1 << TARGET_BIT)

    # CX only permutes data: move both FP32 components together as 64 raw bits.
    # This also avoids separate, strided stores to real and imaginary parts.
    packed = tl.cast(state, tl.pointer_type(tl.int64))
    value0 = tl.load(packed + index0, valid, 0)
    value1 = tl.load(packed + index1, valid, 0)
    tl.store(packed + index0, value1, valid)
    tl.store(packed + index1, value0, valid)


def apply_cx_inplace(
    state: torch.Tensor, control_bit: int, target_bit: int
) -> torch.Tensor:
    """Apply CX to a contiguous complex64 accelerator state, returning ``state``.

    The shape is ``(N,)`` or ``(B, N)``, with ``N`` a power of two >= 4.
    Bit 0 is the least significant state-index bit. Control and target order
    matters. No allocation or gate matrix is needed. Autograd is unsupported;
    gradient-tracked tensors require a ``torch.no_grad()`` context.
    """
    size, batch = validate_inplace_state(state, (control_bit, target_bit))
    if batch == 0:
        return state
    block = 256
    with torch.get_device_module(state.device).device(state.device):
        apply_cx_inplace_kernel[(triton.cdiv(size // 4, block), batch)](
            torch.view_as_real(state),
            SIZE=size,
            CONTROL_BIT=control_bit,
            TARGET_BIT=target_bit,
            BLOCK=block,
            num_warps=4,
        )
    return state
