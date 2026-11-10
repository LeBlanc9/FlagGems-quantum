"""Initialize a complex64 statevector without tensor arithmetic in PyTorch."""

import torch
import triton
import triton.language as tl

from .utils import validate_basis_state


@triton.jit
def initialize_basis_state_kernel(
    output,
    SIZE: tl.constexpr,
    BASIS_INDEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    offset = 2 * (batch * SIZE + index)
    valid = index < SIZE
    tl.store(output + offset, (index == BASIS_INDEX).to(tl.float32), valid)
    tl.store(output + offset + 1, 0.0, valid)


def initialize_basis_state(out: torch.Tensor, basis_index: int = 0) -> torch.Tensor:
    """Overwrite ``out`` with the same computational basis state in every row.

    ``out`` is a contiguous complex64 accelerator tensor of shape ``(N,)`` or
    ``(B, N)``, with ``N`` a power of two. The returned tensor is ``out`` itself.
    Gradient-tracked buffers are rejected, including under ``torch.no_grad()``.
    """
    size, batch = validate_basis_state(out, basis_index)
    if batch == 0:
        return out
    block = 256
    with torch.get_device_module(out.device).device(out.device):
        initialize_basis_state_kernel[(triton.cdiv(size, block), batch)](
            torch.view_as_real(out),
            SIZE=size,
            BASIS_INDEX=basis_index,
            BLOCK=block,
            num_warps=4,
        )
    return out
