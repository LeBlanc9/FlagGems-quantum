# FlagGems-Quantum

Triton kernels for quantum simulation, starting with dense statevectors.

```text
src/flag_gems_quantum/statevector/
    apply_matrix_1q.py       Single-qubit 2x2 matrix application
    apply_matrix_2q.py       Two-qubit 4x4 matrix application
```

The two forward operators, `apply_matrix_1q` and `apply_matrix_2q`, accept
PyTorch tensors. Each operator and its Triton kernel (`apply_matrix_1q_kernel`
or `apply_matrix_2q_kernel`) live in the same file. The Python entry points
validate inputs, allocate output when needed, expose real/imaginary storage
without copying, and launch the kernels on the input device's current stream.

```python
from flag_gems_quantum import apply_matrix_1q, apply_matrix_2q

# state and matrices are contiguous torch.complex64 tensors on an accelerator.
result = apply_matrix_1q(state, matrix_2x2, target_bit=0)
result = apply_matrix_2q(result, matrix_4x4, first_target_bit=1, second_target_bit=0)
apply_matrix_1q(result, matrix_2x2, target_bit=0, out=result)  # in-place
```

- State shape: `(N,)` or `(B, N)`, where `N = 2**n`.
- Matrix shape: `(2, 2)` / `(4, 4)` shared across the batch, or `(B, 2, 2)` /
  `(B, 4, 4)` for per-sample matrices. An unbatched state has `B=1`.
- Inputs and optional `out` must be contiguous `torch.complex64` tensors on
  the same accelerator. `out` must match the state's shape. Exact state/output
  aliasing is supported; partial overlap and output/matrix overlap are rejected.
- Target bits start at the least significant state-index bit (bit 0). The
  two-qubit matrix basis is `|first, second>` in `00, 01, 10, 11` order,
  regardless of the numerical order of the target bits.
- Only forward execution is supported. Gradient-tracked tensors require a
  `torch.no_grad()` context. Conjugate/negative views must be resolved first.

The simulator remains responsible for circuit scheduling, gate fusion, and
qubit-to-bit mapping. Kernel launch parameters stay inside the operator.

Install a compatible FlagTree/Triton compiler for the target hardware before
importing kernel modules. The package does not choose a compiler distribution.

Install the package with `pip install -e .` after preparing the target's
PyTorch and compiler environment. More operators will be added as
implementations are developed; there are no placeholder operators.
GPU compilation, numerical execution, and performance remain unverified.

Run the input checks and numerical tests in the prepared environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
# For a non-default accelerator, e.g. a registered PyTorch backend:
# QUANTUM_TEST_DEVICE=musa PYTHONPATH=src python -m unittest discover -s tests -v
```
