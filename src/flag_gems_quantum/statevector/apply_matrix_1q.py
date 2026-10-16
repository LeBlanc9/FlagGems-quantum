"""Apply a row-major 2x2 matrix to pairs of complex64 amplitudes."""

import triton
import triton.language as tl


@triton.jit
def apply_matrix_1q(
    state,
    matrix,
    output,
    SIZE: tl.constexpr,
    TARGET_BIT: tl.constexpr,
    MATRIX_BATCHED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(SIZE >= 2 and (SIZE & (SIZE - 1)) == 0, "SIZE must be a power of two >= 2")
    tl.static_assert(TARGET_BIT >= 0 and (1 << TARGET_BIT) < SIZE, "Invalid target bit")
    tl.static_assert(state.dtype.element_ty == tl.float32, "state must point to float32 components")
    tl.static_assert(matrix.dtype.element_ty == tl.float32, "matrix must point to float32 components")
    tl.static_assert(output.dtype.element_ty == tl.float32, "output must point to float32 components")

    pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    batch = tl.program_id(1).to(tl.int64)
    valid = pair < SIZE // 2

    # Insert a zero at the target bit; flipping it selects the paired amplitude.
    target_mask = 1 << TARGET_BIT
    lower = pair & (target_mask - 1)
    index0 = batch * SIZE + ((pair - lower) << 1) + lower
    index1 = index0 + target_mask

    x0_real = tl.load(state + 2 * index0, valid, 0.0)
    x0_imag = tl.load(state + 2 * index0 + 1, valid, 0.0)
    x1_real = tl.load(state + 2 * index1, valid, 0.0)
    x1_imag = tl.load(state + 2 * index1 + 1, valid, 0.0)

    # Four row-major complex coefficients occupy eight float32 components.
    base = batch * 8 if MATRIX_BATCHED else 0
    u00_real = tl.load(matrix + base)
    u00_imag = tl.load(matrix + base + 1)
    u01_real = tl.load(matrix + base + 2)
    u01_imag = tl.load(matrix + base + 3)
    u10_real = tl.load(matrix + base + 4)
    u10_imag = tl.load(matrix + base + 5)
    u11_real = tl.load(matrix + base + 6)
    u11_imag = tl.load(matrix + base + 7)

    y0_real = u00_real * x0_real - u00_imag * x0_imag + u01_real * x1_real - u01_imag * x1_imag
    y0_imag = u00_real * x0_imag + u00_imag * x0_real + u01_real * x1_imag + u01_imag * x1_real
    y1_real = u10_real * x0_real - u10_imag * x0_imag + u11_real * x1_real - u11_imag * x1_imag
    y1_imag = u10_real * x0_imag + u10_imag * x0_real + u11_real * x1_imag + u11_imag * x1_real

    tl.store(output + 2 * index0, y0_real, valid)
    tl.store(output + 2 * index0 + 1, y0_imag, valid)
    tl.store(output + 2 * index1, y1_real, valid)
    tl.store(output + 2 * index1 + 1, y1_imag, valid)
