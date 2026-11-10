"""SWAP with contiguous tile writes and no state-sized temporary buffer."""

import torch
import triton
import triton.language as tl

from .indexing import _insert_zero_bit
from .utils import validate_inplace_state


@triton.jit
def apply_swap_inplace_kernel(
    state, SIZE: tl.constexpr, LOW: tl.constexpr, HIGH: tl.constexpr,
    TILE: tl.constexpr, BLOCK: tl.constexpr,
):
    lane = tl.arange(0, BLOCK)
    index = tl.program_id(0).to(tl.int64) * BLOCK + lane
    batch = tl.program_id(1).to(tl.int64) * SIZE
    packed = tl.cast(state, tl.pointer_type(tl.int64))

    # Each program owns a complete set of tiles closed under SWAP. Gather from
    # loaded tiles so cross-thread exchanges finish before those tiles are stored.
    if (1 << HIGH) < TILE:
        # Both bits lie within each contiguous tile.
        different = ((lane >> LOW) ^ (lane >> HIGH)) & 1
        source = lane ^ (different << LOW) ^ (different << HIGH)
        values = tl.load(packed + batch + index, index < SIZE, 0)
        output = tl.gather(values, source, axis=0)
        tl.store(packed + batch + index, output, index < SIZE)
    elif (1 << LOW) < TILE:
        # The low bit is inside a tile; the high bit selects its partner tile.
        first = _insert_zero_bit(index, HIGH)
        second = first + (1 << HIGH)
        valid = index < SIZE // 2
        a = tl.load(packed + batch + first, valid, 0)
        b = tl.load(packed + batch + second, valid, 0)
        partner = lane ^ (1 << LOW)
        a_partner = tl.gather(a, partner, axis=0)
        b_partner = tl.gather(b, partner, axis=0)
        low_one = (lane & (1 << LOW)) != 0
        out_a = tl.where(low_one, b_partner, a)
        out_b = tl.where(low_one, b, a_partner)
        tl.store(packed + batch + first, out_a, valid)
        tl.store(packed + batch + second, out_b, valid)
    else:
        # Both bits select whole tiles. Only the 01/10 tiles need moving.
        base = _insert_zero_bit(_insert_zero_bit(index, LOW), HIGH)
        first, second = base + (1 << LOW), base + (1 << HIGH)
        valid = index < SIZE // 4
        a = tl.load(packed + batch + first, valid, 0)
        b = tl.load(packed + batch + second, valid, 0)
        tl.store(packed + batch + first, b, valid)
        tl.store(packed + batch + second, a, valid)


def apply_swap_inplace(state: torch.Tensor, first_bit: int, second_bit: int) -> torch.Tensor:
    """Exchange two state-index bits in place, writing contiguous blocks."""
    size, batch = validate_inplace_state(state, (first_bit, second_bit))
    if batch == 0:
        return state
    low, high = sorted((first_bit, second_bit))
    # Fixed 64-byte tiles, grouped into one program; no device-specific tuning.
    tile, block = 8, 256
    # Only bits outside a tile reduce the number of independently owned tiles.
    external_bits = int((1 << low) >= tile) + int((1 << high) >= tile)
    grid = (triton.cdiv(size >> external_bits, block), batch)
    with torch.get_device_module(state.device).device(state.device):
        apply_swap_inplace_kernel[grid](
            torch.view_as_real(state), SIZE=size, LOW=low, HIGH=high,
            TILE=tile, BLOCK=block, num_warps=4,
        )
    return state
