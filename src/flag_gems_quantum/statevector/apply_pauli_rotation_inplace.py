"""Matrix-free RXX/RYY, pairing indices that differ in both target bits."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .indexing import _insert_zero_bit
from .utils import validate_angle_for_state, validate_inplace_state


@triton.jit
def apply_pauli_rotation_inplace_kernel(
    state, theta, SIZE: tl.constexpr, FIRST: tl.constexpr, SECOND: tl.constexpr,
    GATE: tl.constexpr, ANGLE_TENSOR: tl.constexpr, ANGLE_BATCHED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(GATE == "rxx" or GATE == "ryy", "Unsupported Pauli rotation")
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = pair < SIZE // 2
    # An index with bit FIRST cleared owns the pair; its partner sets that bit.
    # This partitions the entire state without inter-program read/write races.
    index = _insert_zero_bit(pair, FIRST)
    partner = index ^ ((1 << FIRST) | (1 << SECOND))
    i, j = index + batch * SIZE, partner + batch * SIZE
    ar, ai = tl.load(state + 2 * i, valid, 0), tl.load(state + 2 * i + 1, valid, 0)
    br, bi = tl.load(state + 2 * j, valid, 0), tl.load(state + 2 * j + 1, valid, 0)
    if ANGLE_TENSOR:
        angle = tl.load(theta + (batch if ANGLE_BATCHED else 0))
    else:
        angle = theta.to(tl.float32)
    c, s = libdevice.cos(angle * 0.5), libdevice.sin(angle * 0.5)
    if GATE == "ryy":
        # YY maps 00 <-> -11 and 01 <-> 10; XX always has coefficient +1.
        s = s * tl.where(((index >> SECOND) & 1) != 0, 1.0, -1.0)
    yr, yi = c * ar + s * bi, c * ai - s * br
    zr, zi = c * br + s * ai, c * bi - s * ar
    tl.store(state + 2 * i, yr, valid)
    tl.store(state + 2 * i + 1, yi, valid)
    tl.store(state + 2 * j, zr, valid)
    tl.store(state + 2 * j + 1, zi, valid)


def _apply(state, first_bit, second_bit, theta, gate):
    size, batch = validate_inplace_state(state, (first_bit, second_bit))
    theta = validate_angle_for_state(theta, state)
    if batch == 0:
        return state
    tensor_angle = isinstance(theta, torch.Tensor)
    with torch.get_device_module(state.device).device(state.device):
        apply_pauli_rotation_inplace_kernel[(triton.cdiv(size // 2, 256), batch)](
            torch.view_as_real(state), theta, SIZE=size, FIRST=first_bit, SECOND=second_bit,
            GATE=gate, ANGLE_TENSOR=tensor_angle,
            ANGLE_BATCHED=tensor_angle and theta.numel() > 1, BLOCK=256, num_warps=4,
        )
    return state


def apply_rxx_inplace(state: torch.Tensor, first_bit: int, second_bit: int, theta) -> torch.Tensor:
    """Apply exp(-i theta X⊗X/2) with distinct LSB-numbered bits.

    State is contiguous complex64, shape (N,) or (B, N). The finite FP32 angle
    is a real scalar or a same-device tensor of shape (), (1,), or (B,).
    Returns the same state buffer; forward execution only.
    """
    return _apply(state, first_bit, second_bit, theta, "rxx")


def apply_ryy_inplace(state: torch.Tensor, first_bit: int, second_bit: int, theta) -> torch.Tensor:
    """Apply exp(-i theta Y⊗Y/2), with the same contract as apply_rxx_inplace."""
    return _apply(state, first_bit, second_bit, theta, "ryy")
