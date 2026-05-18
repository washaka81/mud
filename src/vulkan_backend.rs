use std::sync::{Arc, Mutex, LazyLock};

static VK_CTX: LazyLock<Mutex<Option<Arc<crate::vulkan::VulkanContext>>>> =
    LazyLock::new(|| Mutex::new(crate::vulkan::VulkanContext::new().ok().map(Arc::new)));

fn quantize_ternary(w: &[f32]) -> Vec<u32> {
    let gamma = w.iter().copied().map(|x| x.abs() as f64).sum::<f64>() / w.len() as f64;
    let scale = (gamma as f32).max(1e-7);
    let n = w.len();
    let mut packed = vec![0u32; (n + 15) / 16];
    for i in 0..n {
        let val = (w[i] / scale).round().clamp(-1.0, 1.0) as i8;
        let bits = match val { 1 => 1u32, -1 => 2u32, _ => 0u32 };
        packed[i / 16] |= bits << ((i % 16) * 2);
    }
    packed
}

fn gemv_cpu(x: &[f32], w_packed: &[u32], y: &mut [f32], n_in: usize, n_out: usize, scale: f32) {
    let blocks_per_row = n_in / 16;
    unsafe {
        for i in 0..n_out {
            y[i] = 0.0;
            let row_ptr = w_packed.as_ptr().add(i * blocks_per_row);
            crate::asm::ternary_gemv_avx2(n_in, x.as_ptr(), row_ptr, &mut y[i], scale);
        }
    }
}

fn gemv_transpose_cpu(dy: &[f32], w_packed: &[u32], dx: &mut [f32], n_in: usize, n_out: usize) {
    let blocks_per_row = n_in / 16;
    for j in 0..n_in {
        let mut sum = 0.0f32;
        for i in 0..n_out {
            let block = w_packed[i * blocks_per_row + j / 16];
            let bits = (block >> ((j % 16) * 2)) & 3;
            let w_ij: f32 = match bits {
                1 => 1.0,
                2 => -1.0,
                _ => 0.0,
            };
            sum += w_ij * dy[i];
        }
        dx[j] = sum;
    }
}

fn outer_product(dy: &[f32], x: &[f32], dw: &mut [f32], n_out: usize, n_in: usize) {
    for i in 0..n_out {
        let dy_i = dy[i];
        let offset = i * n_in;
        for j in 0..n_in {
            dw[offset + j] += dy_i * x[j];
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn vb_quantize(w: *const f32, w_len: u32, out: *mut u32) -> i32 {
    let w_slice = std::slice::from_raw_parts(w, w_len as usize);
    let packed = quantize_ternary(w_slice);
    let out_slice = std::slice::from_raw_parts_mut(out, packed.len());
    out_slice.copy_from_slice(&packed);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vb_gemv_forward(
    x: *const f32,
    w_packed: *const u32,
    y: *mut f32,
    n_in: u32,
    n_out: u32,
    scale: f32,
    use_vulkan: u8,
) -> i32 {
    let n_in = n_in as usize;
    let n_out = n_out as usize;
    let x_slice = std::slice::from_raw_parts(x, n_in);
    let w_slice = std::slice::from_raw_parts(w_packed, (n_in / 16) * n_out);
    let y_slice = std::slice::from_raw_parts_mut(y, n_out);

    if use_vulkan != 0 {
        if let Some(ctx) = VK_CTX.lock().unwrap().as_ref() {
            return match ctx.run_ternary_gemv(n_in, n_out, x_slice, w_slice.as_ptr(), scale, y_slice) {
                Ok(()) => 0,
                Err(_) => {
                    gemv_cpu(x_slice, w_slice, y_slice, n_in, n_out, scale);
                    1
                }
            };
        }
    }
    gemv_cpu(x_slice, w_slice, y_slice, n_in, n_out, scale);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vb_gemv_backward_input(
    dy: *const f32,
    w_packed: *const u32,
    dx: *mut f32,
    n_in: u32,
    n_out: u32,
) -> i32 {
    let n_in = n_in as usize;
    let n_out = n_out as usize;
    let dy_slice = std::slice::from_raw_parts(dy, n_out);
    let w_slice = std::slice::from_raw_parts(w_packed, (n_in / 16) * n_out);
    let dx_slice = std::slice::from_raw_parts_mut(dx, n_in);
    gemv_transpose_cpu(dy_slice, w_slice, dx_slice, n_in, n_out);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vb_outer_product(
    dy: *const f32,
    x: *const f32,
    dw: *mut f32,
    n_out: u32,
    n_in: u32,
) -> i32 {
    let n_out = n_out as usize;
    let n_in = n_in as usize;
    let dy_slice = std::slice::from_raw_parts(dy, n_out);
    let x_slice = std::slice::from_raw_parts(x, n_in);
    let dw_slice = std::slice::from_raw_parts_mut(dw, n_out * n_in);
    outer_product(dy_slice, x_slice, dw_slice, n_out, n_in);
    0
}

#[no_mangle]
pub unsafe extern "C" fn vb_init_vulkan() -> i32 {
    match crate::vulkan::VulkanContext::new() {
        Ok(ctx) => {
            *VK_CTX.lock().unwrap() = Some(Arc::new(ctx));
            0
        }
        Err(_) => -1,
    }
}
