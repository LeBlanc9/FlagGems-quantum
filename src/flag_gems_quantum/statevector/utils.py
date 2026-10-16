"""Input validation shared by the matrix application operators."""

import torch


def validate_matrix_inputs(state, matrix, target_bits, out):
    tensors = [("state", state), ("matrix", matrix)]
    if out is not None:
        tensors.append(("out", out))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype != torch.complex64:
            raise TypeError(f"{name} must have dtype torch.complex64")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous strided tensor")
        if tensor.is_conj() or tensor.is_neg():
            raise ValueError(f"{name} must have resolved conjugate and negative views")
        if tensor.device != state.device:
            raise ValueError(f"{name} must be on the same device as state")
        if torch.is_grad_enabled() and tensor.requires_grad:
            raise ValueError("Only forward execution is supported; use torch.no_grad()")

    if state.ndim not in (1, 2):
        raise ValueError("state must have shape (N,) or (B, N)")
    size = state.shape[-1]
    width = 1 << len(target_bits)
    if size < width or size & (size - 1):
        raise ValueError(f"State length must be a power of two >= {width}")
    for bit in target_bits:
        if not isinstance(bit, int) or isinstance(bit, bool):
            raise TypeError("Target bits must be Python integers")
        if not 0 <= bit < size.bit_length() - 1:
            raise ValueError("Target bit is outside the state")
    if len(set(target_bits)) != len(target_bits):
        raise ValueError("Target bits must be distinct")

    batch = state.shape[0] if state.ndim == 2 else 1
    if matrix.shape not in ((width, width), (batch, width, width)):
        raise ValueError(
            f"matrix must have shape ({width}, {width}) or ({batch}, {width}, {width})"
        )
    if out is not None and out.shape != state.shape:
        raise ValueError("out must have the same shape as state")
    if state.device.type in ("cpu", "meta"):
        raise ValueError("state must be on an accelerator supported by Triton")

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
