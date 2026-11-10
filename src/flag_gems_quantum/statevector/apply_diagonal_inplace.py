"""Compatibility entry for callers using the older gate-name interface.

New code calls apply_z_inplace, apply_rz_inplace, etc. directly. Each gate's
formula and launch live in its own module; this file contains no GPU kernel.
"""

import torch

from .apply_cphase_inplace import apply_cphase_inplace
from .apply_crz_inplace import apply_crz_inplace
from .apply_cz_inplace import apply_cz_inplace
from .apply_phase_inplace import apply_phase_inplace, apply_u1_inplace
from .apply_rz_inplace import apply_rz_inplace
from .apply_rzz_inplace import apply_rzz_inplace
from .apply_s_inplace import apply_s_inplace
from .apply_sdg_inplace import apply_sdg_inplace
from .apply_t_inplace import apply_t_inplace
from .apply_tdg_inplace import apply_tdg_inplace
from .apply_z_inplace import apply_z_inplace


# Legacy argument contract only: (entry point, number of bits, accepts theta).
_ENTRIES = {
    "z": (apply_z_inplace, 1, False),
    "s": (apply_s_inplace, 1, False),
    "sdg": (apply_sdg_inplace, 1, False),
    "t": (apply_t_inplace, 1, False),
    "tdg": (apply_tdg_inplace, 1, False),
    "cz": (apply_cz_inplace, 2, False),
    "rz": (apply_rz_inplace, 1, True),
    "rzz": (apply_rzz_inplace, 2, True),
    "crz": (apply_crz_inplace, 2, True),
    "phase": (apply_phase_inplace, 1, True),
    "u1": (apply_u1_inplace, 1, True),
    "cphase": (apply_cphase_inplace, 2, True),
}


def apply_diagonal_inplace(state: torch.Tensor, gate: str, target_bits, theta=None) -> torch.Tensor:
    """Forward a legacy gate-name call to its dedicated operator.

    Bits use LSB numbering, with (control, target) order for controlled gates.
    The operator validates the complex64 state and any FP32 angle. New callers
    can use apply_z_inplace, apply_rz_inplace, etc. directly.
    """
    if gate not in _ENTRIES:
        raise ValueError(f"gate must be one of {', '.join(_ENTRIES)}")
    apply_gate, arity, accepts_theta = _ENTRIES[gate]
    bits = tuple(target_bits)
    if len(bits) != arity:
        raise ValueError(f"{gate} requires {arity} target bits")
    if accepts_theta:
        return apply_gate(state, *bits, theta)
    if theta is not None:
        raise ValueError(f"{gate} does not accept theta")
    return apply_gate(state, *bits)
