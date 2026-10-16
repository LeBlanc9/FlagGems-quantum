# FlagGems-Quantum

Triton kernels for quantum simulation, starting with dense statevectors.

```text
src/flag_gems_quantum/statevector/
    apply_matrix_1q.py       Single-qubit 2x2 matrix application
    apply_matrix_2q.py       Two-qubit 4x4 matrix application
```

The current implementation contains two forward kernels, `apply_matrix_1q`
and `apply_matrix_2q`.
The caller supplies device buffers, launch dimensions, and scalar metadata.

The kernels use contiguous, interleaved float32 real/imaginary storage for
complex64 states and row-major matrices. They support shared and per-sample
matrices. Target bits are numbered from the least significant state-index bit
(bit 0). The two-qubit matrix basis is `|first, second>` in `00, 01, 10, 11`
order, regardless of the numerical order of the target bits. Qubit-to-bit
mapping, output allocation, launch configuration, stream selection, and
framework integration are the caller's responsibility.

Install a compatible FlagTree/Triton compiler for the target hardware before
importing kernel modules. The package does not choose a compiler distribution.

Only kernel source and minimal packaging are included. More operators will be
added as implementations are developed; there are no placeholder operators.
GPU compilation, numerical execution, and performance remain unverified.
