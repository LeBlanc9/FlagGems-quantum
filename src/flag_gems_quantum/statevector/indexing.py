"""Shared bit insertion for disjoint statevector amplitude groups."""

import triton
import triton.language as tl


@triton.jit
def _insert_zero_bit(index, BIT: tl.constexpr):
    # Open a gap at position BIT: keep the bits below it, shift everything from
    # BIT up by one, and leave a 0 in the freed slot. This maps a compacted
    # group counter to the amplitude index whose target bit is 0.
    #
    #   example BIT = 2
    #   bit position:   4   3   2   1   0
    #   index       :           a2  a1  a0        (a2.. = high, a1 a0 = low)
    #                          └high┘ └─low─┘
    #                             │
    #                             ▼  shift high up one, drop a 0 into bit BIT
    #   result      :   a3  a2   0   a1  a0
    #                  └─high──┘  ↑  └─low─┘
    #                          inserted zero
    #
    # Input indices are int64. For two target bits, call this with the lower bit
    # first, then the higher bit at its final position in the state index.
    lower = index & ((1 << BIT) - 1)    # low = bits [0, BIT)
    return ((index - lower) << 1) + lower  # clear low, shift high up, restore low
