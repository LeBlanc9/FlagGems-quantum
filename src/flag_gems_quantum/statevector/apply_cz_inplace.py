"""CZ on a contiguous complex64 state, with one owner per amplitude."""

import torch
import triton
import triton.language as tl

from .utils import validate_inplace_state


@triton.jit
def apply_cz_inplace_kernel(
    state, SIZE: tl.constexpr, CONTROL_BIT: tl.constexpr, TARGET_BIT: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = index < SIZE
    control_one = ((index >> CONTROL_BIT) & 1) != 0
    target_one = ((index >> TARGET_BIT) & 1) != 0
    active = control_one & target_one
    offset = 2 * (index + batch * SIZE)
    ar = tl.load(state + offset, valid, 0)
    ai = tl.load(state + offset + 1, valid, 0)
    yr, yi = -ar, -ai
    # Keep continuous reads/writes and preserve amplitudes outside the subspace.
    tl.store(state + offset, tl.where(active, yr, ar), valid)
    tl.store(state + offset + 1, tl.where(active, yi, ai), valid)


def apply_cz_inplace(state: torch.Tensor, control_bit: int, target_bit: int) -> torch.Tensor:
    """Apply CZ = diag(1, 1, 1, -1) with distinct LSB-numbered bits.

    State has shape (N,) or (B, N). Bits use LSB numbering.
    Returns the same buffer; forward execution only.
    """
    size, batch = validate_inplace_state(state, (control_bit, target_bit))
    if batch == 0:
        return state
    with torch.get_device_module(state.device).device(state.device):
        apply_cz_inplace_kernel[(triton.cdiv(size, 256), batch)](
            torch.view_as_real(state), SIZE=size, CONTROL_BIT=control_bit, TARGET_BIT=target_bit,
            BLOCK=256, num_warps=4,
        )
    return state
