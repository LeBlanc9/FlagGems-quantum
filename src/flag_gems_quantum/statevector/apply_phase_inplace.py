"""PHASE on a contiguous complex64 state, with one owner per amplitude."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .utils import validate_angle_for_state, validate_inplace_state


@triton.jit
def apply_phase_inplace_kernel(
    state, theta, SIZE: tl.constexpr, TARGET_BIT: tl.constexpr,
    ANGLE_TENSOR: tl.constexpr, ANGLE_BATCHED: tl.constexpr, BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = index < SIZE
    target_one = ((index >> TARGET_BIT) & 1) != 0
    offset = 2 * (index + batch * SIZE)
    ar = tl.load(state + offset, valid, 0)
    ai = tl.load(state + offset + 1, valid, 0)
    if ANGLE_TENSOR:
        angle = tl.load(theta + (batch if ANGLE_BATCHED else 0))
    else:
        angle = theta.to(tl.float32)
    c, s = libdevice.cos(angle), libdevice.sin(angle)
    yr, yi = c * ar - s * ai, s * ar + c * ai
    yr, yi = tl.where(target_one, yr, ar), tl.where(target_one, yi, ai)
    tl.store(state + offset, yr, valid)
    tl.store(state + offset + 1, yi, valid)


def apply_phase_inplace(
    state: torch.Tensor, target_bit: int, theta
) -> torch.Tensor:
    """Apply Phase = diag(1, exp(i theta)) with LSB bit numbering.

    State is contiguous complex64, shape (N,) or (B, N). The finite FP32 angle
    is a real scalar or a same-device tensor of shape (), (1,), or (B,).
    Returns the same state buffer; forward execution only.
    """
    size, batch = validate_inplace_state(state, (target_bit,))
    theta = validate_angle_for_state(theta, state)
    if batch == 0:
        return state
    tensor_angle = isinstance(theta, torch.Tensor)
    with torch.get_device_module(state.device).device(state.device):
        apply_phase_inplace_kernel[(triton.cdiv(size, 256), batch)](
            torch.view_as_real(state), theta, SIZE=size, TARGET_BIT=target_bit,
            ANGLE_TENSOR=tensor_angle, ANGLE_BATCHED=tensor_angle and theta.numel() > 1,
            BLOCK=256, num_warps=4,
        )
    return state


def apply_u1_inplace(state: torch.Tensor, target_bit: int, theta) -> torch.Tensor:
    """U1 is exactly Phase; reuse its validation and kernel."""
    return apply_phase_inplace(state, target_bit, theta)
