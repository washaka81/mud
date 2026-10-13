use half::f16;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; 16],
}

extern "C" {
    // ── Pack.s: batch register ops ─────────────────────────────
    /// Convert N PackedRegister (u32, f16×2) to real/imag f32 arrays (scratch-free).
    /// n must be a multiple of 8 (caller's responsibility; leftover processed in Rust).
    pub fn registers_to_f32_batch(
        regs: *const u32,
        out_real: *mut f32,
        out_imag: *mut f32,
        n: usize,
    );
    /// Pack 8 real + 8 imag f32 values into 8 PackedRegister (f16×2 → u32).
    /// n must be a multiple of 8.
    pub fn f32_to_registers_batch(
        real: *const f32,
        imag: *const f32,
        out_regs: *mut u32,
        n: usize,
    );
    /// Compute sigmoid gate from imag component of N packed registers.
    /// gate = 1/(1+exp(-imag)). n must be a multiple of 8.
    pub fn gate_batch(
        regs: *const u32,
        out_gates: *mut f32,
        n: usize,
    );

    pub fn rms_norm_scale_asm(n: usize, x: *const f32, eps: f32) -> f32;
    pub fn ternary_gemv_avx2(
        n: usize,
        x: *const f32,
        weights: *const u32,
        out: *mut f32,
        scale: f32,
    );
    pub fn ternary_gemv_4rows_avx2(
        n: usize,
        x: *const f32,
        weights: *const u32,
        out: *mut f32,
        scale: f32,
        stride: usize,
    );
    /// 8 rows × shared `x` load (ELUT). Prefer over 4rows when n_out ≥ 8.
    pub fn ternary_gemv_8rows_avx2(
        n: usize,
        x: *const f32,
        weights: *const u32,
        out: *mut f32,
        scale: f32,
        stride: usize,
    );
    pub fn ternary_gemm_batch4_avx2(
        out_dim: usize,
        in_dim: usize,
        x_ptr: *const f32,
        w_ptr: *const u32,
        out_ptr: *mut f32,
        scales: *const f32,
    );
    pub fn dot_product_avx2(n: usize, a: *const f32, b: *const f32) -> f32;
    pub fn sum_squares_avx2(n: usize, x: *const f32) -> f32;
    pub fn q4_0_gemv_asm(n: usize, x: *const f32, weights: *const BlockQ4_0, out: *mut f32);
    pub fn silu_vectorial_avx2(n: usize, src: *const f32, dst: *mut f32);
    /// Backward ternary GEMV (grad_x accumulation):
    /// dx[j] += scale_gy * dequant(w[j]) per element.
    /// Uses same nibble-extraction as ternary_gemv_avx2 (vpbroadcastd + vpsrlvd + vpcmpeqd).
    /// L-08 NaN/Inf guard on vector + scalar tail.
    ///
    /// # Safety: all pointers valid for n elements. weights has n/8 u32 words.
    pub fn backward_ternary_gemv_dx_avx2(
        n: usize,
        scale_gy: f32,
        weights: *const u32,
        grad_x: *mut f32,
    );
    /// Backward ternary GEMV (8-row batched grad_x accumulation):
    /// dx[j] += sum_r(scale_gy[r] * dequant(w[r, j])) for r=0..7.
    /// Batched 8-row: loads grad_x[0:8] ONCE per word, accumulates all 8 rows.
    /// n MUST be a multiple of 8. L-08 NaN/Inf guard.
    ///
    /// # Safety: n is multiple of 8; scale_gy has 8 valid f32; weights has 8*stride u32 words;
    ///   grad_x has n valid f32 elements.
    pub fn backward_ternary_gemv_8rows_dx_avx2(
        n: usize,
        scale_gy: *const f32,
        weights: *const u32,
        grad_x: *mut f32,
        stride: usize,
    );
    /// JEPA EMA update: z_next[i] = alpha * y_norm[i] + (1-alpha) * z[i]
    /// Implemented as z + alpha*(y_norm - z) via vsubps + vfmadd231ps (2 instrs/loop).
    /// alpha is read from r8d (low 32 bits as float bits). NaN/Inf guard (L-08).
    /// # Safety: all pointers valid for n elements
    pub fn jepa_ou_avx2(n: usize, z: *const f32, y_norm: *const f32, z_next: *mut f32, alpha_bits: u32);
    /// Backward RMSNorm: grad_x[i] = grad_y[i]*weight[i] - y_unscaled[i]*mean
    /// Computes grad_y*weight + y_unscaled*(-mean) via vfmadd231ps (1 FMA + 1 mul).
    /// mean is passed as float bits in r8d; -mean computed via XOR with -0.0.
    /// L-08: NaN/Inf guard on scalar tail path.
    /// # Safety: all pointers valid for n elements; weight != null
    pub fn backward_rmsnorm_avx2(
        n: usize,
        grad_y: *const f32,
        y_unscaled: *const f32,
        weight: *const f32,
        mean_bits: u32,
        grad_x: *mut f32,
    );
    /// Backward RoPE rotation (split layout): dx0=dy0*cos+dy1*sin, dx1=dy1*cos-dy0*sin
    /// In-place on grad_y (split: first half = real, second = imaginary). L-08 NaN guard.
    /// # Safety: all pointers valid for n elements
    pub fn backward_rope_asm(n: usize, grad_y: *mut f32, cos: *const f32, sin: *const f32);
    /// mHC gate backward: d_v = gate * (1 - gate) * d_gate
    /// gate is sigmoid output, d_gate is upstream gradient, d_v is gradient w.r.t. v_jepa. L-08 NaN guard.
    /// # Safety: all pointers valid for n elements
    pub fn mhc_gate_backward_avx2(n: usize, gate: *const f32, grad_gate: *const f32, grad_v: *mut f32);
    // L-04: removed pext_unpack_ternary, ternary_gemv_lut, mamba_*, elut_gemv, slime_rmsnorm
    // (2-bit/i16/unwired orphans — see docs/sessions and GEMINI L-04)
    /// Argmax over vocab: returns index of max dot(regs, weights[row]).
    pub fn lm_head_avx2(
        vocab_size: usize,
        hidden: usize,
        regs: *const f32,
        weights: *const f32,
    ) -> usize;
    /// Full LM-head logits: out[v] = dot(regs, weights[v * hidden ..]).
    pub fn lm_head_logits_avx2(
        vocab_size: usize,
        hidden: usize,
        regs: *const f32,
        weights: *const f32,
        out_logits: *mut f32,
    );
    /// Adam step (see src/asm/adam_step.s for full arg list / bias-correction scalars).
    pub fn adam_step_avx2(
        n: usize,
        w: *mut f32,
        m: *mut f32,
        v: *mut f32,
        grads: *const f32,
        clip_coef: f32,
        wd: f32,
        b1: f32,
        b2: f32,
        lr_bc1: f32,
        inv_bc2: f32,
        eps: f32,
    );
    pub fn sgemm_abt_avx2(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32);

    /// Batch apply per-row PRQ scales to 8 f32 outputs simultaneously (AVX2).
    /// For each of 8 rows: out[i] *= clamp(scale[i], -1e8, 1e8) if finite, else 0.
    ///
    /// This fuses the scalar scale-application loop in `ternary_gemv_rowwise_submit`
    /// into a single AVX2 pass, leveraging packed f32 (8 values per YMM register).
    ///
    /// # Safety
    /// - `out` must have at least 8 valid f32 elements.
    /// - `scales` must have at least 8 valid f32 elements.
    pub fn apply_row_scales_8_avx2(out: *mut f32, scales: *const f32);

    /// Batch complex multiply: Y = X * W (8 complex numbers at once).
    ///
    /// For each of 8 elements:
    ///   Y_R[i] = X_R[i] * W_R[i] - X_I[i] * W_I[i]
    ///   Y_I[i] = X_R[i] * W_I[i] + X_I[i] * W_R[i]
    ///
    /// Uses AVX2 FMA for 2 FMA instructions per complex multiply.
    ///
    /// # Safety
    /// - `x_re`, `x_im`, `w_re`, `w_im` must have at least 8 valid f32 elements.
    /// - `y_re`, `y_im` must have at least 8 writable f32 elements.
    pub fn cmud_gauss_mul_batch_avx2(
        x_re: *const f32,
        x_im: *const f32,
        w_re: *const f32,
        w_im: *const f32,
        y_re: *mut f32,
        y_im: *mut f32,
    );
}

/// Dequantizes a row of Q4_0 weights to f32.
/// # Safety
/// The caller must ensure that `row` points to at least `n / 32` valid `BlockQ4_0` blocks
/// and that `out` has at least `n` elements.
pub unsafe fn dequantize_q4_0_row(row: *const BlockQ4_0, out: &mut [f32], n: usize) {
    let blocks = n / 32;
    for i in 0..blocks {
        let block = &*row.add(i);
        let mut d = block.d.to_f32();
        if d.is_nan() || d.is_infinite() {
            d = 0.0;
        }
        for j in 0..16 {
            let qs = block.qs[j];
            let low = (qs & 0x0F) as f32 - 8.0;
            let high = (qs >> 4) as f32 - 8.0;
            out[i * 32 + j] = low * d;
            out[i * 32 + j + 16] = high * d;
        }
    }
}

/// Fused Matrix-Vector multiplication for Q4_0 block quantization.
/// # Safety
/// The caller must ensure that `x`, `x_norm`, and `out` have sufficient length for
/// `n_in` and `n_out` sizes. `weights` must point to valid memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn q4_0_gemv_fused(
    n_in: usize,
    n_out: usize,
    x: &[f32],
    weights: *const BlockQ4_0,
    norm_w: *const f32,
    out: &mut [f32],
    eps: f32,
    x_norm: &mut [f32],
) {
    let scale = rms_norm_scale_asm(n_in, x.as_ptr(), eps);
    for (i, item) in x_norm.iter_mut().enumerate().take(n_in) {
        *item = x[i] * scale * (*norm_w.add(i));
    }

    let row_size_blocks = n_in / 32;
    for (i, item) in out.iter_mut().enumerate().take(n_out) {
        let weight_ptr = weights.add(i * row_size_blocks);
        let mut val = 0.0f32;
        q4_0_gemv_asm(n_in, x_norm.as_ptr(), weight_ptr, &mut val as *mut f32);
        *item = val;
    }
}

#[cfg(test)]
mod tests;

/// R-35: scalar 1..7 tail for ternary GEMV rows.
///
/// The ASM kernels only cover multiples of 8: `ternary_gemv_avx2`'s
/// `.leftover8` unconditionally processes a full 8-wide block (over-reading
/// `x` and accumulating garbage lanes when `n % 8 != 0`), while the 4/8-row
/// kernels silently DROP the remainder. This helper accumulates the exact
/// tail (`n_tail` in 1..=7, nibbles LSB-first per the ELUT convention) so
/// every wrapper below is bit-correct for any `n` with zero over-read.
///
/// Layout: canonical per-row padded (R-15) — row `r` owns `stride` u32 words
/// and the tail lives in word `n_main/8` at nibble positions `0..n_tail`.
/// `x` points at the tail activations (`x_base + n_main`).
///
/// # Safety
/// `x` valid for `n_tail` f32; each of the `rows` rows has `stride` valid
/// u32 words with `word_idx = n_main/8 < stride`; `out` valid for `rows` f32.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ternary_gemv_tail(
    rows: usize,
    n_main: usize,
    n_tail: usize,
    x: *const f32,
    weights: *const u32,
    stride: usize,
    out: *mut f32,
    scale: f32,
) {
    if n_tail == 0 {
        return;
    }
    // R-35: FFI-boundary contract — the tail word must exist in every row.
    debug_assert!(n_main.is_multiple_of(8));
    debug_assert!((1..8).contains(&n_tail));
    debug_assert!(n_main / 8 < stride);
    let word_idx = n_main / 8;
    for r in 0..rows {
        let w = *weights.add(r * stride + word_idx);
        let mut acc = 0.0f32;
        for j in 0..n_tail {
            let code = (w >> (4 * j)) & 0xF;
            // ELUT: 0x1 -> +1, 0xF -> -1, else 0 (same as the ASM compare).
            let v = match code {
                1 => 1.0f32,
                0xF => -1.0f32,
                _ => 0.0f32,
            };
            acc += v * *x.add(j);
        }
        *out.add(r) += scale * acc;
    }
}

#[inline(always)]
/// # Safety
/// The caller must ensure that the pointers point to valid memory and n, stride are correct.
/// R-35: `stride` is u32 words per row and MUST be `>= n.div_ceil(8)`
/// (canonical per-row padded layout); `x` valid for `n` f32. Any `n` is
/// exact — the 1..7 remainder is accumulated by [`ternary_gemv_tail`].
pub unsafe fn ternary_gemv_4rows(
    n: usize,
    x: *const f32,
    weights: *const u32,
    out: *mut f32,
    scale: f32,
    stride: usize,
) {
    let n_main = n & !7;
    ternary_gemv_4rows_avx2(n_main, x, weights, out, scale, stride);
    ternary_gemv_tail(4, n_main, n - n_main, x.add(n_main), weights, stride, out, scale);
}

#[inline(always)]
/// # Safety
/// Pointers must be valid for 8 output rows spaced by `stride` u32 words.
/// R-35: same contract as [`ternary_gemv_4rows`] — any `n` is exact via
/// [`ternary_gemv_tail`].
pub unsafe fn ternary_gemv_8rows(
    n: usize,
    x: *const f32,
    weights: *const u32,
    out: *mut f32,
    scale: f32,
    stride: usize,
) {
    let n_main = n & !7;
    ternary_gemv_8rows_avx2(n_main, x, weights, out, scale, stride);
    ternary_gemv_tail(8, n_main, n - n_main, x.add(n_main), weights, stride, out, scale);
}

#[inline(always)]
/// # Safety
/// The caller must ensure that the pointers point to valid memory and n is correct.
/// R-35: the row must hold `>= n.div_ceil(8)` u32 words; any `n` is exact —
/// the ASM kernel covers `n & !7` and [`ternary_gemv_tail`] the rest (the old
/// `.leftover8` path over-read `x` and accumulated garbage on remainders).
pub unsafe fn ternary_gemv(
    n: usize,
    x: *const f32,
    weights: *const u32,
    out: *mut f32,
    scale: f32,
) {
    let n_main = n & !7;
    ternary_gemv_avx2(n_main, x, weights, out, scale);
    // Single contiguous row: words 0..div_ceil(n,8) are adjacent.
    ternary_gemv_tail(1, n_main, n - n_main, x.add(n_main), weights, n.div_ceil(8).max(1), out, scale);
}

#[inline(always)]
/// # Safety
/// Pointers must be valid for the given m, n, k (C = A * Bᵀ style via sgemm_abt_avx2).
pub unsafe fn sgemm_abt(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    sgemm_abt_avx2(m, n, k, a, b, c);
}

// ── Pack.s wrappers ────────────────────────────────────────────

/// Batch-convert PackedRegister f16×2 → separate real/imag f32 arrays.
/// `n` is rounded down to the nearest multiple of 8; returns remainder count.
///
/// # Safety
/// Pointers `regs`, `out_real`, `out_imag` must be non-null and valid for at least `n` elements.
#[inline]
pub unsafe fn registers_to_f32_batch_into(
    regs: *const u32,
    out_real: *mut f32,
    out_imag: *mut f32,
    n: usize,
) -> usize {
    let n8 = n & !7;
    if n8 >= 8 {
        registers_to_f32_batch(regs, out_real, out_imag, n8);
    }
    n - n8
}

/// Batch-pack 8 real + 8 imag f32 → PackedRegister.
/// `n` is rounded down to nearest multiple of 8; returns remainder count.
///
/// # Safety
/// Pointers `real`, `imag`, `out_regs` must be non-null and valid for at least `n` elements.
#[inline]
pub unsafe fn f32_to_registers_batch_into(
    real: *const f32,
    imag: *const f32,
    out_regs: *mut u32,
    n: usize,
) -> usize {
    let n8 = n & !7;
    if n8 >= 8 {
        f32_to_registers_batch(real, imag, out_regs, n8);
    }
    n - n8
}

/// Batch-compute sigmoid gates from imag component of packed registers.
/// `n` is rounded down to nearest multiple of 8; returns remainder count.
///
/// # Safety
/// Pointers `regs` and `out_gates` must be non-null and valid for at least `n` elements.
#[inline]
pub unsafe fn gate_batch_into(
    regs: *const u32,
    out_gates: *mut f32,
    n: usize,
) -> usize {
    let n8 = n & !7;
    if n8 >= 8 {
        gate_batch(regs, out_gates, n8);
    }
    n - n8
}
