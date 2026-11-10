"""I, X, CCX and CSWAP using one owner per exchanged amplitude pair."""

import torch
import triton
import triton.language as tl

from .indexing import _insert_zero_bit
from .utils import validate_inplace_state


@triton.jit
def apply_permutation_inplace_kernel(
    state, SIZE: tl.constexpr, FIRST: tl.constexpr, SECOND: tl.constexpr,
    SWAP: tl.constexpr, CONTROL_MASK: tl.constexpr, BLOCK: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    if SWAP:
        valid = pair < SIZE // 4
        low_bit, high_bit = min(FIRST, SECOND), max(FIRST, SECOND)
        base = _insert_zero_bit(pair, low_bit)
        base = _insert_zero_bit(base, high_bit) + batch * SIZE
        i, j = base + (1 << FIRST), base + (1 << SECOND)
    else:
        valid = pair < SIZE // 2
        i = _insert_zero_bit(pair, FIRST) + batch * SIZE
        j = i + (1 << FIRST)
    # Transport the two FP32 components together; this is not FP64 arithmetic.
    packed = tl.cast(state, tl.pointer_type(tl.int64))
    a, b = tl.load(packed + i, valid, 0), tl.load(packed + j, valid, 0)
    if CONTROL_MASK:
        # Same traversal for every control position; select packed original
        # values outside the control-one subspace, without any complex math.
        active = (i & CONTROL_MASK) == CONTROL_MASK
        out_i, out_j = tl.where(active, b, a), tl.where(active, a, b)
    else:
        out_i, out_j = b, a
    tl.store(packed + i, out_i, valid)
    tl.store(packed + j, out_j, valid)


def _apply(state, bits, controls=()):
    size, batch = validate_inplace_state(state, controls + bits)
    if batch == 0:
        return state
    swap = len(bits) == 2
    with torch.get_device_module(state.device).device(state.device):
        apply_permutation_inplace_kernel[(triton.cdiv(size >> len(bits), 256), batch)](
            torch.view_as_real(state), SIZE=size, FIRST=bits[0],
            SECOND=bits[-1], SWAP=swap, CONTROL_MASK=sum(1 << bit for bit in controls),
            BLOCK=256, num_warps=4,
        )
    return state


def apply_i_inplace(state: torch.Tensor, target_bit: int) -> torch.Tensor:
    """Validate an identity operation and return the state without a launch."""
    validate_inplace_state(state, (target_bit,))
    return state


def apply_x_inplace(state: torch.Tensor, target_bit: int) -> torch.Tensor:
    """Apply X by swapping target-zero/one amplitudes; bit 0 is the LSB."""
    return _apply(state, (target_bit,))


def apply_ccx_inplace(
    state: torch.Tensor, first_control_bit: int, second_control_bit: int, target_bit: int
) -> torch.Tensor:
    """Apply X only where both distinct control bits are one (Toffoli)."""
    return _apply(state, (target_bit,), (first_control_bit, second_control_bit))


def apply_cswap_inplace(
    state: torch.Tensor, control_bit: int, first_bit: int, second_bit: int
) -> torch.Tensor:
    """Swap two target bits only where the distinct control bit is one (Fredkin)."""
    return _apply(state, (first_bit, second_bit), (control_bit,))
