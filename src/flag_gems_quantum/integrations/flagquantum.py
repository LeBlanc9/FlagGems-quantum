"""An optional statevector backend for FlagQuantum's existing extension SDK.

This adapter owns the gate loop and wire mapping. All state initialization and
evolution use FlagGems-Quantum kernels. It accepts zero-initialized CircuitIR
programs and returns state tensors only; measurements are rejected before launch.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

import torch

from flagquantum import ExecutionOptions
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)
from flagquantum.ecosystem.extensions.admission import (
    admit_backend_extension,
    withdraw_backend,
)
from flagquantum.errors import CapabilityError

from .. import (
    apply_ccx_inplace,
    apply_cphase_inplace,
    apply_crx_inplace,
    apply_cry_inplace,
    apply_crz_inplace,
    apply_cswap_inplace,
    apply_cx_inplace,
    apply_cy_inplace,
    apply_cz_inplace,
    apply_h_inplace,
    apply_i_inplace,
    apply_phase_inplace,
    apply_rx_inplace,
    apply_rxx_inplace,
    apply_ry_inplace,
    apply_ryy_inplace,
    apply_rz_inplace,
    apply_rzz_inplace,
    apply_s_inplace,
    apply_sdg_inplace,
    apply_swap_inplace,
    apply_sx_inplace,
    apply_sxdg_inplace,
    apply_t_inplace,
    apply_tdg_inplace,
    apply_u1_inplace,
    apply_u2_inplace,
    apply_u3_inplace,
    apply_x_inplace,
    apply_y_inplace,
    apply_z_inplace,
    initialize_basis_state,
)
from ..statevector.utils import validate_angle_value


# One backend support table: parameter order and the operator itself.
# All operators accept (state, *LSB-numbered bits, *ordered parameters).
_GATE_DISPATCH = {
    "i": ((), apply_i_inplace),
    "x": ((), apply_x_inplace),
    "y": ((), apply_y_inplace),
    "z": ((), apply_z_inplace),
    "h": ((), apply_h_inplace),
    "s": ((), apply_s_inplace),
    "sdg": ((), apply_sdg_inplace),
    "t": ((), apply_t_inplace),
    "tdg": ((), apply_tdg_inplace),
    "sx": ((), apply_sx_inplace),
    "sxdg": ((), apply_sxdg_inplace),
    "rx": (("theta",), apply_rx_inplace),
    "ry": (("theta",), apply_ry_inplace),
    "rz": (("theta",), apply_rz_inplace),
    "phase": (("theta",), apply_phase_inplace),
    "u1": (("theta",), apply_u1_inplace),
    "u2": (("phi", "lbd"), apply_u2_inplace),
    "u3": (("theta", "phi", "lbd"), apply_u3_inplace),
    "cx": ((), apply_cx_inplace),
    "cy": ((), apply_cy_inplace),
    "cz": ((), apply_cz_inplace),
    "swap": ((), apply_swap_inplace),
    "crx": (("theta",), apply_crx_inplace),
    "cry": (("theta",), apply_cry_inplace),
    "crz": (("theta",), apply_crz_inplace),
    "cphase": (("theta",), apply_cphase_inplace),
    "rxx": (("theta",), apply_rxx_inplace),
    "ryy": (("theta",), apply_ryy_inplace),
    "rzz": (("theta",), apply_rzz_inplace),
    "ccx": ((), apply_ccx_inplace),
    "cswap": ((), apply_cswap_inplace),
}


def _check_angle(theta, batch_size):
    if isinstance(theta, (list, tuple)):
        if len(theta) not in (1, batch_size):
            raise ValueError("theta must have length 1 or batch_size")
        # Lists are input data; their conversion does not perform gate math.
        if any(isinstance(value, torch.Tensor) for value in theta):
            raise TypeError("theta sequences must contain real scalars")
        return [validate_angle_value(value, batch_size) for value in theta]
    if isinstance(theta, torch.Tensor):
        if theta.requires_grad:
            raise ValueError("gradient-tracked angles are unsupported")
        if theta.device.type not in ("cpu", "cuda"):
            raise ValueError("theta must be on CPU or a CUDA-interface device")
    return validate_angle_value(theta, batch_size)


class FlagGemsQuantumBackend:
    """Single-device, complex64, forward execution from the zero state.

    Use ``use_backend()`` to register this backend, obtain execution options,
    and clean up automatically. FlagQuantum is an optional dependency.
    """

    manifest = ExtensionManifest(
        name="flag_gems_quantum",
        version="0.1.0",
        kind="backend",
        capabilities=frozenset({"complex64", "cuda", "statevector"}),
    )
    backend_capabilities: ClassVar[dict[str, Any]] = {
        "tensor_backend": "flag_gems_quantum",
        "devices": ("cuda",),
        "preferred_device": "cuda",
        "dtypes": ("complex64",),
        "supports_autograd": False,
        "supports_distributed": False,
        "supports_statevector": True,
        "supports_density_matrix": False,
        "supports_mps": False,
    }

    def __init__(self) -> None:
        self.active = False

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        blockers = []
        if request.dtype not in (None, "complex64"):
            blockers.append("only complex64 is supported")
        if request.device_type not in (None, "cuda"):
            blockers.append("only the CUDA device interface is supported")
        if request.require_gradients:
            blockers.append("only forward execution is supported")
        return CapabilityResponse(
            not blockers, self.manifest.capabilities, tuple(blockers)
        )

    def start(self, config: ExtensionConfig) -> None:
        self.active = True

    def close(self) -> None:
        self.active = False

    def execute(self, program: CircuitIR, *, options: Mapping[str, Any]) -> torch.Tensor:
        """Validate the full request, then launch only owned numerical operators."""
        if not self.active:
            raise RuntimeError("FlagGemsQuantumBackend is closed")
        if not isinstance(program, CircuitIR):
            raise TypeError("program must be a FlagQuantum CircuitIR")
        if options["mode"] != "statevector" or options["world_size"] != 1:
            raise CapabilityError("only single-device statevector execution is supported")
        if options["require_gradients"]:
            raise CapabilityError("FlagGemsQuantumBackend does not support gradients")
        if program.dtype != "complex64" or options["dtype"] != "complex64":
            raise CapabilityError("FlagGemsQuantumBackend requires complex64")
        if (
            program.measurements
            or program.observables
            or options["target"] != "full_state"
        ):
            raise CapabilityError(
                "only final state output is supported; measurements are unavailable"
            )
        device = torch.device(options["device"])
        if device.type != "cuda":
            raise CapabilityError("FlagGemsQuantumBackend requires the CUDA device interface")
        batch_size = options["batch_size"]
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        angles = []
        for gate in program:
            entry = _GATE_DISPATCH.get(gate.name)
            if entry is None or gate.matrix is not None:
                raise CapabilityError(f"unsupported gate: {gate.name}")
            expected_params, _ = entry
            if set(gate.params) != set(expected_params):
                raise CapabilityError(f"unsupported parameters for {gate.name}")
            try:
                angles.append([_check_angle(gate.params[name], batch_size)
                               for name in expected_params])
            except (TypeError, ValueError) as exc:
                raise CapabilityError(f"unsupported angle for {gate.name}: {exc}") from exc

        # Transfer only parameter data. Trigonometry stays in our gate kernels.
        for gate_angles in angles:
            for index, angle in enumerate(gate_angles):
                if isinstance(angle, list):
                    gate_angles[index] = torch.tensor(angle, dtype=torch.float32, device=device)
                elif isinstance(angle, torch.Tensor):
                    gate_angles[index] = angle.to(device=device)

        state = torch.empty(
            (batch_size, 1 << program.n_wires), dtype=torch.complex64, device=device
        )
        initialize_basis_state(state)
        for gate, parameters in zip(program, angles):
            bits = tuple(program.n_wires - 1 - wire for wire in gate.wires)
            _, apply_gate = _GATE_DISPATCH[gate.name]
            apply_gate(state, *bits, *parameters)
        return state


@contextmanager
def use_backend(device: str = "cuda:0") -> Iterator[ExecutionOptions]:
    """Select this state-only backend for calls to ``fq.run`` inside the scope.

    The returned options select complex64 forward execution without backend
    fallback. Registration is removed on exit, including after an exception.

    Example:
        with use_backend() as options:
            result = fq.run(circuit, options=options)
    """
    handle, capabilities = admit_backend_extension(FlagGemsQuantumBackend())
    try:
        yield ExecutionOptions(
            backend=capabilities.name,
            device=device,
            mode="statevector",
            precision="complex64",
            require_gradients=False,
            allow_backend_fallback=False,
        )
    finally:
        try:
            handle.close()
        finally:
            withdraw_backend(capabilities.name)
