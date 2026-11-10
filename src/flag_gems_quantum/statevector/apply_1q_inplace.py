"""Matrix-free single-qubit gates and controlled Y/RX/RY on disjoint pairs."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .indexing import _insert_zero_bit
from .utils import validate_angle_for_state, validate_inplace_state


@triton.jit
def _load_angle(value, batch, IS_TENSOR: tl.constexpr, BATCHED: tl.constexpr):
    if IS_TENSOR:
        return tl.load(value + (batch if BATCHED else 0))
    else:
        return value.to(tl.float32)


@triton.jit
def apply_1q_inplace_kernel(
    state, theta, phi, lbd, SIZE: tl.constexpr, BIT: tl.constexpr, GATE: tl.constexpr,
    ANGLE_TENSOR: tl.constexpr, ANGLE_BATCHED: tl.constexpr, BLOCK: tl.constexpr,
    PHI_TENSOR: tl.constexpr, PHI_BATCHED: tl.constexpr,
    LBD_TENSOR: tl.constexpr, LBD_BATCHED: tl.constexpr,
    CONTROL: tl.constexpr,
):
    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = pair < SIZE // 2
    i = _insert_zero_bit(pair, BIT) + batch * SIZE
    j = i + (1 << BIT)
    ar = tl.load(state + 2 * i, valid, 0)
    ai = tl.load(state + 2 * i + 1, valid, 0)
    br = tl.load(state + 2 * j, valid, 0)
    bi = tl.load(state + 2 * j + 1, valid, 0)
    if GATE == "h":
        yr, yi = (ar + br) * 0.7071067811865476, (ai + bi) * 0.7071067811865476
        zr, zi = (ar - br) * 0.7071067811865476, (ai - bi) * 0.7071067811865476
    elif GATE == "y":
        yr, yi = bi, -br
        zr, zi = -ai, ar
    elif GATE == "sx" or GATE == "sxdg":
        # SX = ((1+i) I + (1-i) X)/2, including its global phase.
        # Its adjoint conjugates both coefficients; RX(+/-pi/2) is not equal.
        sign = 1.0 if GATE == "sx" else -1.0
        yr = ((ar + br) + sign * (bi - ai)) * 0.5
        yi = ((ai + bi) + sign * (ar - br)) * 0.5
        zr = ((ar + br) + sign * (ai - bi)) * 0.5
        zi = ((ai + bi) + sign * (br - ar)) * 0.5
    elif GATE == "rx" or GATE == "ry" or GATE == "u3":
        angle = _load_angle(theta, batch, ANGLE_TENSOR, ANGLE_BATCHED)
        c = libdevice.cos(angle * 0.5)
        s = libdevice.sin(angle * 0.5)
        if GATE == "rx":
            yr, yi = c * ar + s * bi, c * ai - s * br
            zr, zi = c * br + s * ai, c * bi - s * ar
        elif GATE == "ry":
            yr, yi = c * ar - s * br, c * ai - s * bi
            zr, zi = s * ar + c * br, s * ai + c * bi
        elif GATE == "u3":
            p = _load_angle(phi, batch, PHI_TENSOR, PHI_BATCHED)
            l = _load_angle(lbd, batch, LBD_TENSOR, LBD_BATCHED)
            cp, sp = libdevice.cos(p), libdevice.sin(p)
            cl, sl = libdevice.cos(l), libdevice.sin(l)
            # U3 = [[c, -exp(i*l)*s], [exp(i*p)*s, exp(i*(p+l))*c]].
            lr, li = cl * br - sl * bi, sl * br + cl * bi
            yr, yi = c * ar - s * lr, c * ai - s * li
            tr, ti = s * ar + c * lr, s * ai + c * li
            zr, zi = cp * tr - sp * ti, sp * tr + cp * ti
    else:
        tl.static_assert(False, "Unsupported single-qubit gate")
    if CONTROL >= 0:
        # Traverse all target pairs uniformly; preserve control-zero values by
        # selection rather than creating sparse reads/writes on low controls.
        controlled = ((i >> CONTROL) & 1) != 0
        yr, yi = tl.where(controlled, yr, ar), tl.where(controlled, yi, ai)
        zr, zi = tl.where(controlled, zr, br), tl.where(controlled, zi, bi)
    tl.store(state + 2 * i, yr, valid)
    tl.store(state + 2 * i + 1, yi, valid)
    tl.store(state + 2 * j, zr, valid)
    tl.store(state + 2 * j + 1, zi, valid)


def _apply(state, bits, gate, theta=0.0, phi=0.0, lbd=0.0):
    size, batch = validate_inplace_state(state, bits)
    theta = validate_angle_for_state(theta, state)
    phi = validate_angle_for_state(phi, state)
    lbd = validate_angle_for_state(lbd, state)
    if batch == 0:
        return state
    tensor_angle = isinstance(theta, torch.Tensor)
    controlled = len(bits) == 2
    with torch.get_device_module(state.device).device(state.device):
        apply_1q_inplace_kernel[(triton.cdiv(size // 2, 256), batch)](
            torch.view_as_real(state), theta, phi, lbd, SIZE=size, BIT=bits[-1], GATE=gate,
            ANGLE_TENSOR=tensor_angle,
            ANGLE_BATCHED=tensor_angle and theta.numel() > 1,
            PHI_TENSOR=isinstance(phi, torch.Tensor),
            PHI_BATCHED=isinstance(phi, torch.Tensor) and phi.numel() > 1,
            LBD_TENSOR=isinstance(lbd, torch.Tensor),
            LBD_BATCHED=isinstance(lbd, torch.Tensor) and lbd.numel() > 1,
            CONTROL=bits[0] if controlled else -1,
            BLOCK=256, num_warps=4,
        )
    return state


def apply_h_inplace(state: torch.Tensor, target_bit: int) -> torch.Tensor:
    """Apply H to a contiguous complex64 state, using LSB bit numbering."""
    return _apply(state, (target_bit,), "h")


def apply_y_inplace(state: torch.Tensor, target_bit: int) -> torch.Tensor:
    """Apply Y = [[0, -i], [i, 0]], using LSB bit numbering."""
    return _apply(state, (target_bit,), "y")


def apply_cy_inplace(state: torch.Tensor, control_bit: int, target_bit: int) -> torch.Tensor:
    """Apply diag(I, Y), preserving control-zero amplitudes exactly."""
    return _apply(state, (control_bit, target_bit), "y")


def apply_sx_inplace(state: torch.Tensor, target_bit: int) -> torch.Tensor:
    """Apply the principal square root of X, retaining its full phase."""
    return _apply(state, (target_bit,), "sx")


def apply_sxdg_inplace(state: torch.Tensor, target_bit: int) -> torch.Tensor:
    """Apply the adjoint of SX."""
    return _apply(state, (target_bit,), "sxdg")


def apply_rx_inplace(state: torch.Tensor, target_bit: int, theta) -> torch.Tensor:
    """Apply exp(-i theta X/2); theta is a finite scalar or device FP32 tensor.

    Tensor angles have shape (), (1,), or (B,), for shared or per-sample angles.
    The state has shape (N,) or (B, N). Return the same tensor; no autograd.
    """
    return _apply(state, (target_bit,), "rx", theta)


def apply_ry_inplace(state: torch.Tensor, target_bit: int, theta) -> torch.Tensor:
    """Apply exp(-i theta Y/2), with the same contract as apply_rx_inplace."""
    return _apply(state, (target_bit,), "ry", theta)


def apply_crx_inplace(
    state: torch.Tensor, control_bit: int, target_bit: int, theta
) -> torch.Tensor:
    """Apply RX only where control=1, preserving the control-zero amplitudes.

    Bits use LSB numbering and must be distinct. State and angle contracts match
    apply_rx_inplace. This implements diag(I, RX), including its relative phase.
    """
    return _apply(state, (control_bit, target_bit), "rx", theta)


def apply_cry_inplace(
    state: torch.Tensor, control_bit: int, target_bit: int, theta
) -> torch.Tensor:
    """Apply diag(I, RY), with the same contract as apply_crx_inplace."""
    return _apply(state, (control_bit, target_bit), "ry", theta)


def apply_u3_inplace(state: torch.Tensor, target_bit: int, theta, phi, lbd) -> torch.Tensor:
    """Apply U3(theta, phi, lambda), including the phase convention of its matrix.

    Each angle independently follows the scalar/(1,)/(B,) FP32 contract of RX.
    This also handles U3 instructions emitted by FlagQuantum's compiler.
    """
    return _apply(state, (target_bit,), "u3", theta, phi, lbd)


def apply_u2_inplace(state: torch.Tensor, target_bit: int, phi, lbd) -> torch.Tensor:
    """Apply U2(phi, lambda) = U3(pi/2, phi, lambda)."""
    return _apply(state, (target_bit,), "u3", math.pi / 2, phi, lbd)
