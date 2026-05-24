use std::sync::{Arc, Mutex, LazyLock};
use rayon::prelude::*;

static VK_CTX: LazyLock<Mutex<Option<Arc<crate::vulkan::VulkanContext>>>> =
    LazyLock::new(|| Mutex::new(None));

// Global lock for Vulkan submissions to ensure thread safety
static VK_SUBMIT_LOCK: Mutex<()> = Mutex::new(());

fn lazy_init_vulkan() -> Option<Arc<crate::vulkan::VulkanContext>> {
    let mut lock = VK_CTX.lock().unwrap();
    if lock.is_none() {
        if let Ok(ctx) = crate::vulkan::VulkanContext::new() {
            *lock = Some(Arc::new(ctx));
        }
    }
    lock.as_ref().cloned()
}

fn quantize_ternary(w: &[f32]) -> Vec<u32> {
    let gamma = w.iter().copied().map(|x| x.abs() as f64).sum::<f64>() / w.len() as f64;
    let scale = (gamma as f32).max(1e-7);
    let n = w.len();
    let mut packed = vec![0u32; n.div_ceil(16)];
    for i in 0..n {
        let val = (w[i] / scale).round().clamp(-1.0, 1.0) as i8;
        let bits = match val { 1 => 1u32, -1 => 2u32, _ => 0u32 };
        packed[i / 16] |= bits << ((i % 16) * 2);
    }
    packed
}

fn gemv_cpu(x: &[f32], w_packed: &[u32], y: &mut [f32], n_in: usize, n_out: usize, scale: f32) {
    let blocks_per_row = n_in / 16;
    let stride = blocks_per_row;
    let mut i = 0;
    unsafe {
        // Process 4 rows at a time using the 4-rows kernel
        while i + 4 <= n_out {
            let row_ptr = w_packed.as_ptr().add(i * stride);
            crate::asm::ternary_gemv_4rows_avx2(
                n_in, x.as_ptr(), row_ptr, y.as_mut_ptr().add(i), scale, stride,
            );
            i += 4;
        }
        // Handle remaining rows (1-3) with single-row kernel
        while i < n_out {
            let row_ptr = w_packed.as_ptr().add(i * stride);
            crate::asm::ternary_gemv_avx2(n_in, x.as_ptr(), row_ptr, &mut y[i], scale);
            i += 1;
        }
    }
}

fn gemv_transpose_cpu(dy: &[f32], w_packed: &[u32], dx: &mut [f32], n_in: usize, n_out: usize) {
    let blocks = n_in / 16;
    dx.fill(0.0);
    // Process each output row: for each row, unpack 16 weights at a time
    // and add dy_i * w_ij into dx[j]
    for i in 0..n_out {
        let dy_i = dy[i];
        if dy_i == 0.0 { continue; }
        let row_start = i * blocks;
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { gemv_transpose_avx2_row(dy_i, &w_packed[row_start..row_start + blocks], dx, n_in) }
                continue;
            }
        }
        // Fallback scalar
        for b in 0..blocks {
            let block = w_packed[row_start + b];
            let base = b * 16;
            for j in 0..16 {
                let bits = (block >> (j * 2)) & 3;
                let w = match bits { 1 => 1.0, 2 => -1.0, _ => 0.0 };
                dx[base + j] += dy_i * w;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn gemv_transpose_avx2_row(dy_i: f32, row_blocks: &[u32], dx: &mut [f32], n_in: usize) {
    use std::arch::x86_64::*;
    let dy_bcast = _mm256_set1_ps(dy_i);
    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(-1.0);
    let zero = _mm256_setzero_ps();
    let mask2bit = _mm256_set1_epi32(3);
    let shifts_low  = _mm256_set_epi32(14, 12, 10, 8, 6, 4, 2, 0);
    let shifts_high = _mm256_set_epi32(30, 28, 26, 24, 22, 20, 18, 16);

    for (b, &block) in row_blocks.iter().enumerate() {
        let base = b * 16;
        if base + 16 > n_in { break; }

        let w_vec = _mm256_set1_epi32(block as i32);

        // Low 8 weights (bits 0-15)
        let lo_bits = _mm256_srlv_epi32(w_vec, shifts_low);
        let lo_bits = _mm256_and_si256(lo_bits, mask2bit);

        let lo_mask1 = _mm256_cmpeq_epi32(lo_bits, _mm256_set1_epi32(1));
        let lo_mask2 = _mm256_cmpeq_epi32(lo_bits, _mm256_set1_epi32(2));
        let lo_w = _mm256_blendv_ps(zero, one, _mm256_castsi256_ps(lo_mask1));
        let lo_w = _mm256_blendv_ps(lo_w, two, _mm256_castsi256_ps(lo_mask2));

        // High 8 weights (bits 16-31)
        let hi_bits = _mm256_srlv_epi32(w_vec, shifts_high);
        let hi_bits = _mm256_and_si256(hi_bits, mask2bit);

        let hi_mask1 = _mm256_cmpeq_epi32(hi_bits, _mm256_set1_epi32(1));
        let hi_mask2 = _mm256_cmpeq_epi32(hi_bits, _mm256_set1_epi32(2));
        let hi_w = _mm256_blendv_ps(zero, one, _mm256_castsi256_ps(hi_mask1));
        let hi_w = _mm256_blendv_ps(hi_w, two, _mm256_castsi256_ps(hi_mask2));

        // Multiply by dy_i and add to dx
        let dx_lo = _mm256_loadu_ps(dx.as_ptr().add(base));
        let dx_hi = _mm256_loadu_ps(dx.as_ptr().add(base + 8));

        let contrib_lo = _mm256_mul_ps(dy_bcast, lo_w);
        let contrib_hi = _mm256_mul_ps(dy_bcast, hi_w);

        _mm256_storeu_ps(dx.as_mut_ptr().add(base), _mm256_add_ps(dx_lo, contrib_lo));
        _mm256_storeu_ps(dx.as_mut_ptr().add(base + 8), _mm256_add_ps(dx_hi, contrib_hi));
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn outer_product_avx2_row(dy_i: f32, x: &[f32], row: &mut [f32], n_in: usize) {
    use std::arch::x86_64::*;
    let dy_bcast = _mm256_set1_ps(dy_i);
    let mut offset = 0usize;
    while offset + 8 <= n_in {
        let x_chunk = _mm256_loadu_ps(x.as_ptr().add(offset));
        let dw_chunk = _mm256_loadu_ps(row.as_ptr().add(offset));
        let prod = _mm256_mul_ps(dy_bcast, x_chunk);
        _mm256_storeu_ps(row.as_mut_ptr().add(offset), _mm256_add_ps(dw_chunk, prod));
        offset += 8;
    }
    for j in offset..n_in {
        row[j] += dy_i * x[j];
    }
}

#[no_mangle]
/// # Safety
/// `w` must be a valid pointer to at least `w_len` f32 elements.
/// `out` must be a valid pointer to at least `ceil(w_len / 16)` u32 elements.
pub unsafe extern "C" fn vb_quantize(w: *const f32, w_len: u32, out: *mut u32) -> i32 {
    if w.is_null() || out.is_null() { return -1; }
    let w_slice = std::slice::from_raw_parts(w, w_len as usize);
    let packed = quantize_ternary(w_slice);
    let out_slice = std::slice::from_raw_parts_mut(out, packed.len());
    out_slice.copy_from_slice(&packed);
    0
}

#[no_mangle]
/// # Safety
/// Pointers must be valid and appropriately sized:
/// x: [batch_size, n_in]
/// w_packed: [n_out, n_in/16]
/// y: [batch_size, n_out]
pub unsafe extern "C" fn vb_gemm_forward(
    x: *const f32,
    w_packed: *const u32,
    y: *mut f32,
    batch_size: u32,
    n_in: u32,
    n_out: u32,
    scale: f32,
    use_vulkan: u8,
) -> i32 {
    if x.is_null() || w_packed.is_null() || y.is_null() { return -1; }
    let batch_size = batch_size as usize;
    let n_in = n_in as usize;
    let n_out = n_out as usize;
    if batch_size == 0 || n_in == 0 || n_out == 0 { return 0; }
    
    let x_slice = std::slice::from_raw_parts(x, batch_size * n_in);
    let w_len = (n_in.div_ceil(16)) * n_out;
    let w_slice = std::slice::from_raw_parts(w_packed, w_len);
    let y_slice = std::slice::from_raw_parts_mut(y, batch_size * n_out);

    if use_vulkan != 0 {
        if let Some(ctx) = lazy_init_vulkan() {
            let _submit_guard = VK_SUBMIT_LOCK.lock().unwrap();
            let key = format!("ptr_{:x}", w_packed as usize);
            if let Ok(()) = ctx.run_ternary_gemm_cached(
                &key, batch_size, n_in, n_out, 
                x_slice, w_packed, scale, y_slice
            ) {
                return 0;
            }
        }
    }

    // CPU Path: Parallelized over batch using Rayon
    y_slice.par_chunks_mut(n_out).enumerate().for_each(|(b, out_row)| {
        let x_row = &x_slice[b * n_in .. (b + 1) * n_in];
        gemv_cpu(x_row, w_slice, out_row, n_in, n_out, scale);
    });

    0
}

#[no_mangle]
/// # Safety
/// Pointers must be valid and appropriately sized.
pub unsafe extern "C" fn vb_gemm_backward_input(
    dy: *const f32,
    w_packed: *const u32,
    dx: *mut f32,
    batch_size: u32,
    n_in: u32,
    n_out: u32,
) -> i32 {
    if dy.is_null() || w_packed.is_null() || dx.is_null() { return -1; }
    let batch_size = batch_size as usize;
    let n_in = n_in as usize;
    let n_out = n_out as usize;
    if batch_size == 0 || n_in == 0 || n_out == 0 { return 0; }
    
    let dy_slice = std::slice::from_raw_parts(dy, batch_size * n_out);
    let w_len = (n_in.div_ceil(16)) * n_out;
    let w_slice = std::slice::from_raw_parts(w_packed, w_len);
    let dx_slice = std::slice::from_raw_parts_mut(dx, batch_size * n_in);

    // CPU Path: Parallelized over batch
    dx_slice.par_chunks_mut(n_in).enumerate().for_each(|(b, dx_row)| {
        let dy_row = &dy_slice[b * n_out .. (b + 1) * n_out];
        gemv_transpose_cpu(dy_row, w_slice, dx_row, n_in, n_out);
    });

    0
}

#[no_mangle]
/// # Safety
/// Pointers must be valid and appropriately sized.
pub unsafe extern "C" fn vb_gemm_outer_product(
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    batch_size: u32,
    n_out: u32,
    n_in: u32,
) -> i32 {
    if dy.is_null() || x.is_null() || dw.is_null() { return -1; }
    let batch_size = batch_size as usize;
    let n_out = n_out as usize;
    let n_in = n_in as usize;
    if batch_size == 0 || n_out == 0 || n_in == 0 { return 0; }
    
    let dy_slice = std::slice::from_raw_parts(dy, batch_size * n_out);
    let x_slice = std::slice::from_raw_parts(x, batch_size * n_in);
    let dw_slice = std::slice::from_raw_parts_mut(dw, n_out * n_in);

    // Accumulated outer product over batch
    dw_slice.par_chunks_mut(n_in).enumerate().for_each(|(i, dw_row)| {
        for b in 0..batch_size {
            let dy_val = dy_slice[b * n_out + i];
            if dy_val == 0.0 { continue; }
            let x_row = &x_slice[b * n_in .. (b + 1) * n_in];
            
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && n_in >= 8 {
                    unsafe { outer_product_avx2_row(dy_val, x_row, dw_row, n_in) }
                    continue;
                }
            }
            for j in 0..n_in {
                dw_row[j] += dy_val * x_row[j];
            }
        }
    });

    0
}

#[no_mangle]
/// # Safety
/// Vulkan initialization is generally safe but relies on system drivers.
pub unsafe extern "C" fn vb_init_vulkan() -> i32 {
    if lazy_init_vulkan().is_some() { 0 } else { -1 }
}

#[no_mangle]
/// Clears all cached Vulkan buffers (call when weights change between training steps).
pub unsafe extern "C" fn vb_clear_caches() {
    if let Some(ctx) = VK_CTX.lock().unwrap().as_ref() {
        ctx.buffer_cache.lock().clear();
        ctx.buffer_init.lock().clear();
        ctx.buffer_x_cache.lock().clear();
        ctx.buffer_y_cache.lock().clear();
    }
}
