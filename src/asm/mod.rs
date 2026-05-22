use half::f16;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct BlockQ4_0 {
    pub d: f16,
    pub qs: [u8; 16],
}

extern "C" {
    pub fn q4_0_gemv_asm(n: usize, x: *const f32, weights: *const BlockQ4_0, out: *mut f32);
    pub fn rms_norm_scale_asm(n: usize, x: *const f32, eps: f32) -> f32;
    /// New optimized Ternary AVX2 kernel (Additions/Subtractions only)
    pub fn ternary_gemv_avx2(n: usize, x: *const f32, weights: *const u32, out: *mut f32, scale: f32);
    pub fn ternary_gemv_4rows_avx2(n: usize, x: *const f32, weights: *const u32, out: *mut f32, scale: f32, stride: usize);
    pub fn dot_product_avx2(n: usize, a: *const f32, b: *const f32) -> f32;
    pub fn sum_squares_avx2(n: usize, x: *const f32) -> f32;
}

pub unsafe fn dequantize_q4_0_row(row: *const BlockQ4_0, out: &mut [f32], n: usize) {
    let blocks = n / 32;
    for i in 0..blocks {
        let block = &*row.add(i);
        let mut d = block.d.to_f32();
        if d.is_nan() || d.is_infinite() { d = 0.0; }
        for j in 0..16 {
            let qs = block.qs[j];
            let low = (qs & 0x0F) as f32 - 8.0;
            let high = (qs >> 4) as f32 - 8.0;
            out[i * 32 + j] = low * d;
            out[i * 32 + j + 16] = high * d;
        }
    }
}

pub unsafe fn q4_0_gemv_fused(n_in: usize, n_out: usize, x: &[f32], weights: *const BlockQ4_0, norm_w: *const f32, out: &mut [f32], eps: f32) {
    let scale = rms_norm_scale_asm(n_in, x.as_ptr(), eps);
    let mut x_norm = vec![0.0f32; n_in];
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
