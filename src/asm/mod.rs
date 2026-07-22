use half::f16;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; 16],
}

extern "C" {
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
    pub fn apply_rope_asm(n: usize, x: *mut f32, cos: *const f32, sin: *const f32);
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

#[inline(always)]
/// # Safety
/// The caller must ensure that the pointers point to valid memory and n, stride are correct.
pub unsafe fn ternary_gemv_4rows(
    n: usize,
    x: *const f32,
    weights: *const u32,
    out: *mut f32,
    scale: f32,
    stride: usize,
) {
    ternary_gemv_4rows_avx2(n, x, weights, out, scale, stride);
}

#[inline(always)]
/// # Safety
/// Pointers must be valid for 8 output rows spaced by `stride` u32 words.
pub unsafe fn ternary_gemv_8rows(
    n: usize,
    x: *const f32,
    weights: *const u32,
    out: *mut f32,
    scale: f32,
    stride: usize,
) {
    ternary_gemv_8rows_avx2(n, x, weights, out, scale, stride);
}

#[inline(always)]
/// # Safety
/// The caller must ensure that the pointers point to valid memory and n is correct.
pub unsafe fn ternary_gemv(
    n: usize,
    x: *const f32,
    weights: *const u32,
    out: *mut f32,
    scale: f32,
) {
    ternary_gemv_avx2(n, x, weights, out, scale);
}

#[inline(always)]
/// # Safety
/// Pointers must be valid for the given m, n, k (C = A * Bᵀ style via sgemm_abt_avx2).
pub unsafe fn sgemm_abt(m: usize, n: usize, k: usize, a: *const f32, b: *const f32, c: *mut f32) {
    sgemm_abt_avx2(m, n, k, a, b, c);
}
