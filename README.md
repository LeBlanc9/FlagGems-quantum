# FlagGems-Quantum

Triton kernels for single-precision statevector simulation. Gate-level fusion
combines tensor operations into a kernel; gate-specific formulas avoid dense
matrix work for rotations, diagonal gates, and permutations.

PyTorch provides tensors, memory allocation, and device management. State
initialization and gate arithmetic run in this library's Triton kernels.

## Install

Requires Python 3.10+, PyTorch, and a Triton-compatible GPU. Install the PyTorch
and FlagTree/Triton versions appropriate for your hardware, then run:

```bash
python -m pip install -e .
```

## Apply gates directly

```python
import torch
import flag_gems_quantum as fgq

state = torch.empty(4, dtype=torch.complex64, device="cuda:0")
fgq.initialize_basis_state(state)
fgq.apply_h_inplace(state, 0)
fgq.apply_cx_inplace(state, 0, 1)
print(state)
# Bell state: approximately [0.7071, 0, 0, 0.7071]
```

States are contiguous `torch.complex64` tensors of shape `(2**n,)` or
`(batch, 2**n)`. Bit 0 is the least significant state-index bit.
Gate functions update and return the input state. Rotation angles accept finite
real scalars or contiguous FP32 tensors on the same device, with shape `()`,
`(1,)`, or `(batch,)`.

## Use with FlagQuantum

Make [FlagQuantum](https://github.com/flagos-ai/FlagQuantum) available in your
Python environment. For an adjacent source checkout, use `export PYTHONPATH=../FlagQuantum`.
The adapter registers through FlagQuantum's extension SDK; no upstream code
changes are needed.

```python
import flagquantum as fq
from flag_gems_quantum.integrations.flagquantum import use_backend

circuit = fq.Circuit(2).h(0).cx(0, 1)
with use_backend(device="cuda:0") as options:
    result = fq.run(circuit, options=options)
print(result.state)
# Bell state, with shape (1, 4)
```

Use `fq.run(..., options=options)` to select this backend. It starts from the
zero basis state and returns the final statevector, with optional batching on
one GPU. Measurement, noise, gradients, and double precision are not supported.
Unsupported gates raise an error without falling back to another backend.

## Supported gates

The library and adapter support these 31 gate names:

| Type | Gates |
| --- | --- |
| Fixed single-qubit | I, X, Y, Z, H, S, SDG, T, TDG, SX, SXDG |
| Parameterized single-qubit | RX, RY, RZ, Phase, U1, U2, U3 |
| Fixed two-qubit | CX, CY, CZ, SWAP |
| Parameterized two-qubit | CRX, CRY, CRZ, CPhase, RXX, RYY, RZZ |
| Three-qubit | CCX, CSWAP |

Direct calls use `apply_<gate>_inplace`, for example `apply_rzz_inplace(state, 1, 0, 0.3)`.
The standalone API also provides `apply_matrix_1q` and `apply_matrix_2q` for
2x2 and 4x4 complex64 matrices, including batched matrices and optional `out`.
These generic matrix operators are not exposed through the adapter.

## Test and benchmark

Run from the repository root with FlagQuantum available:

```bash
QUANTUM_TEST_DEVICE=cuda:0 python -m unittest discover -s tests -v

# Compare circuit execution with FlagQuantum's default backend.
python benchmarks/compare_flagquantum.py --device cuda:0 --qubits 20 --workload mixed

# Compare SWAP across all qubit pairs.
python benchmarks/compare_swap.py --device cuda:0 --qubits 20 --all-pairs --output /tmp/swap.json
```

See [benchmarks](benchmarks) for single-gate comparisons and
[the Bell example](examples/flagquantum_bell.py) for explicit backend registration.
