"""Input validation for statevector matrix and dedicated gate operators."""

import math
from numbers import Real

import torch


def _validate_complex_tensor(tensor, name):
    """Check storage and forward-only properties shared by states and matrices."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.complex64:
        raise TypeError(f"{name} must have dtype torch.complex64")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError(f"{name} must have resolved conjugate and negative views")
    if torch.is_grad_enabled() and tensor.requires_grad:
        raise ValueError("Only forward execution is supported; use torch.no_grad()")


def _validate_state_structure(state, bits, *, name="state"):
    """Check state dimensions and bit ownership after tensor storage validation."""
    if state.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape (N,) or (B, N)")
    size = state.shape[-1]
    width = max(2, 1 << len(bits))
    if size < width or size & (size - 1):
        raise ValueError(f"State length must be a power of two >= {width}")
    for bit in bits:
        if not isinstance(bit, int) or isinstance(bit, bool):
            raise TypeError("Target bits must be Python integers")
        if not 0 <= bit < size.bit_length() - 1:
            raise ValueError("Target bit is outside the state")
    if len(set(bits)) != len(bits):
        raise ValueError("Target bits must be distinct")
    return size, state.shape[0] if state.ndim == 2 else 1


def _require_accelerator(tensor, name):
    if tensor.device.type in ("cpu", "meta"):
        raise ValueError(f"{name} must be on an accelerator supported by Triton")


def validate_matrix_inputs(state, matrix, target_bits, out):
    tensors = [("state", state), ("matrix", matrix)]
    if out is not None:
        tensors.append(("out", out))
    for name, tensor in tensors:
        _validate_complex_tensor(tensor, name)
        if tensor.device != state.device:
            raise ValueError(f"{name} must be on the same device as state")

    size, batch = _validate_state_structure(state, target_bits)
    width = 1 << len(target_bits)
    if matrix.shape not in ((width, width), (batch, width, width)):
        raise ValueError(
            f"matrix must have shape ({width}, {width}) or ({batch}, {width}, {width})"
        )
    if out is not None and out.shape != state.shape:
        raise ValueError("out must have the same shape as state")
    _require_accelerator(state, "state")

    # Contiguous tensors occupy one interval; only exact state/output aliasing
    # is safe because each program reads and writes a disjoint amplitude group.
    if out is not None and out.numel():
        out_start = out.data_ptr()
        out_end = out_start + out.numel() * out.element_size()
        for name, tensor in (("state", state), ("matrix", matrix)):
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            if out_start < end and start < out_end:
                if name == "matrix" or out_start != start:
                    raise ValueError(f"out must not overlap {name}, except exact state aliasing")
    return size, batch


def validate_inplace_state(state, bits):
    """Validate the common state contract for dedicated in-place gates."""
    _validate_complex_tensor(state, "state")
    size, batch = _validate_state_structure(state, bits)
    _require_accelerator(state, "state")
    return size, batch


def validate_basis_state(out, basis_index):
    """Validate initialization, which rejects tracked buffers even under no_grad."""
    _validate_complex_tensor(out, "out")
    if out.requires_grad:
        raise ValueError("basis initialization only supports forward execution")
    size, batch = _validate_state_structure(out, (), name="out")
    if not isinstance(basis_index, int) or isinstance(basis_index, bool):
        raise TypeError("basis_index must be a Python integer")
    if not 0 <= basis_index < size:
        raise ValueError("basis_index is outside the state")
    _require_accelerator(out, "out")
    return size, batch


def validate_angle_value(theta, batch):
    """Check FP32 angle values/metadata before a destination state is available.

    Tensor contents must be finite. Checking their values would require a
    numerical reduction and synchronization, so only metadata is checked here.
    This function imposes no device or state-storage relationship.
    """
    if isinstance(theta, torch.Tensor):
        if theta.dtype != torch.float32:
            raise TypeError("theta must have dtype torch.float32")
        if theta.layout != torch.strided or not theta.is_contiguous():
            raise ValueError("theta must be a contiguous strided tensor")
        if theta.is_conj() or theta.is_neg():
            raise ValueError("theta must have resolved conjugate and negative views")
        if theta.ndim > 1 or theta.numel() not in (1, batch):
            raise ValueError("theta must be scalar or have shape (1,) or (B,)")
        if torch.is_grad_enabled() and theta.requires_grad:
            raise ValueError("Only forward execution is supported; use torch.no_grad()")
        return theta
    if not isinstance(theta, Real) or isinstance(theta, bool):
        raise TypeError("theta must be a real scalar or FP32 tensor")
    theta = float(theta)
    if not math.isfinite(theta) or abs(theta) > 3.4028234663852886e38:
        raise ValueError("theta must be finite and representable in FP32")
    return theta


def validate_angle_for_state(theta, state):
    """Validate an angle against an already-validated state, including aliasing."""
    batch = state.shape[0] if state.ndim == 2 else 1
    theta = validate_angle_value(theta, batch)
    if isinstance(theta, torch.Tensor):
        if theta.device != state.device:
            raise ValueError("theta must be on the same device as state")
        start, end = state.data_ptr(), state.data_ptr() + state.numel() * state.element_size()
        angle_start = theta.data_ptr()
        angle_end = angle_start + theta.numel() * theta.element_size()
        if state.numel() and theta.numel() and start < angle_end and angle_start < end:
            raise ValueError("theta must not overlap state")
    return theta
