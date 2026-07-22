//! Muon optimizer — Newton-Schulz gradient orthogonalization.
//!
//! **L-01:** CPU path wired into QAT step via `OptimizerStrategy::Muon`.
//! **L-02:** Optional Vulkan path (`MUD_USE_VULKAN=1`) for large matrices;
//! falls back to CPU if ash is unavailable or matrix is small.
//! **Perf:** CPU NS uses AVX2 `sgemm_abt` for Gram + X·G (was scalar O(n³)).

use std::sync::{Mutex, OnceLock};

/// Minimum elements before trying GPU NS (dispatch overhead dominates tiny mats).
const VK_NS_MIN_ELEMENTS: usize = 32 * 32;

/// Shared ash context for NS only (lazy; None if Vulkan init fails).
static NS_ASH: OnceLock<Mutex<Option<crate::vulkan::ash_backend::AshContext>>> = OnceLock::new();

fn ns_ash_slot() -> &'static Mutex<Option<crate::vulkan::ash_backend::AshContext>> {
    NS_ASH.get_or_init(|| {
        let ctx = match crate::vulkan::ash_backend::AshContext::new() {
            Ok(c) if c.is_available() => Some(c),
            _ => None,
        };
        Mutex::new(ctx)
    })
}

fn vulkan_ns_enabled() -> bool {
    std::env::var("MUD_USE_VULKAN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Newton–Schulz iteration count (default 5; quick train / env can lower).
///
/// - `MUD_MUON_NS_ITERS=N` — hard override
/// - else if `MUD_TRAIN_MAX_CHUNKS` set → 1 (smoke-friendly)
/// - else 5
pub fn muon_ns_iters() -> usize {
    if let Ok(v) = std::env::var("MUD_MUON_NS_ITERS") {
        return v.parse::<usize>().unwrap_or(5).clamp(1, 16);
    }
    if std::env::var("MUD_TRAIN_MAX_CHUNKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&m| m > 0)
        .is_some()
    {
        return 1;
    }
    5
}

/// Try GPU Newton-Schulz on already-normalized `x` (rows×cols). Returns true on success.
fn try_vulkan_ns(x: &mut [f32], rows: usize, cols: usize, n_iters: usize) -> bool {
    if !vulkan_ns_enabled() || rows * cols < VK_NS_MIN_ELEMENTS {
        return false;
    }
    let Ok(mut guard) = ns_ash_slot().lock() else {
        return false;
    };
    let Some(ctx) = guard.as_mut() else {
        return false;
    };
    if !ctx.is_available() {
        return false;
    }
    // SAFETY: x len checked by caller; ctx owned exclusively by this mutex.
    unsafe {
        ctx.dispatch_newton_schulz_sync(x, rows, cols, n_iters)
            .is_ok()
    }
}

/// Newton-Schulz orthogonalization of the gradient matrix (hybrid CPU/GPU).
/// Replaces the gradient in-place with its orthogonalized version.
pub fn newton_schulz_orthogonalize(grad: &mut [f32], rows: usize, cols: usize, n_iters: usize) {
    if rows == 0 || cols == 0 || grad.is_empty() || n_iters == 0 {
        return;
    }
    if grad.len() != rows * cols {
        // Malformed layout — refuse rather than corrupt memory
        return;
    }

    let g_norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in grad.iter_mut() {
        *v /= g_norm;
    }

    let mut x = grad.to_vec();

    if !try_vulkan_ns(&mut x, rows, cols, n_iters) {
        newton_schulz_inner_cpu(&mut x, rows, cols, n_iters);
    }

    for (g, xi) in grad.iter_mut().zip(x.iter()) {
        *g = *xi * g_norm;
    }
}

/// CPU-only NS (for tests / parity checks). Same API as hybrid after normalization.
pub fn newton_schulz_orthogonalize_cpu(grad: &mut [f32], rows: usize, cols: usize, n_iters: usize) {
    if rows == 0 || cols == 0 || grad.is_empty() || n_iters == 0 {
        return;
    }
    if grad.len() != rows * cols {
        return;
    }
    let g_norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in grad.iter_mut() {
        *v /= g_norm;
    }
    let mut x = grad.to_vec();
    newton_schulz_inner_cpu(&mut x, rows, cols, n_iters);
    for (g, xi) in grad.iter_mut().zip(x.iter()) {
        *g = *xi * g_norm;
    }
}

fn newton_schulz_inner_cpu(x: &mut [f32], rows: usize, cols: usize, n_iters: usize) {
    let mut gram = vec![0.0f32; cols * cols];
    let mut next_x = vec![0.0f32; rows * cols];
    // X^T as [cols, rows] for Gram via sgemm_abt(C = A A^T)
    let mut xt = vec![0.0f32; cols * rows];
    for _ in 0..n_iters {
        muon_step_inner_fast(x, &mut gram, &mut next_x, &mut xt, rows, cols);
    }
}

/// One NS step: G = XᵀX (symmetric), X ← 1.5 X − 0.5 X G.
/// Uses AVX2 SGEMM when available; scalar fallback for tiny mats / no AVX2.
fn muon_step_inner_fast(
    x: &mut [f32],
    gram: &mut [f32],
    next_x: &mut [f32],
    xt: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(x.len(), rows * cols);
    debug_assert_eq!(gram.len(), cols * cols);
    debug_assert_eq!(next_x.len(), rows * cols);
    debug_assert_eq!(xt.len(), cols * rows);

    let use_avx = {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    };
    if use_avx && rows >= 8 && cols >= 8 {
        // xt[j, i] = x[i, j]  →  shape [cols, rows]
        for i in 0..rows {
            for j in 0..cols {
                xt[j * rows + i] = x[i * cols + j];
            }
        }
        // gram = xt * xt^T  →  [cols, cols]
        unsafe {
            forge_autograd::avx_math::sgemm_abt_avx2(cols, cols, rows, xt, xt, gram);
        }
        // next = x * gram  (gram symmetric ⇒ x * gram = sgemm_abt(x, gram))
        // C[i,j] = sum_p x[i,p] * gram[j,p] = sum_p x[i,p] * gram^T[p,j]
        // with symmetric gram, gram[j,p] = gram[p,j] ✓
        unsafe {
            forge_autograd::avx_math::sgemm_abt_avx2(rows, cols, cols, x, gram, next_x);
        }
        for i in 0..rows * cols {
            x[i] = 1.5 * x[i] - 0.5 * next_x[i];
        }
        return;
    }

    // Scalar fallback (tiny mats / no AVX2)
    for j in 0..cols {
        for k in 0..cols {
            let mut sum = 0.0f32;
            for i in 0..rows {
                sum += x[i * cols + j] * x[i * cols + k];
            }
            gram[j * cols + k] = sum;
        }
    }
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0f32;
            for k in 0..cols {
                sum += x[i * cols + k] * gram[k * cols + j];
            }
            next_x[i * cols + j] = 1.5 * x[i * cols + j] - 0.5 * sum;
        }
    }
    x.copy_from_slice(next_x);
}

/// Applies the Muon QAT step to a weight matrix.
pub fn muon_qat_step(shadow_w: &mut [f32], grad: &mut [f32], rows: usize, cols: usize, lr: f32) {
    newton_schulz_orthogonalize(grad, rows, cols, muon_ns_iters());
    for (w, g) in shadow_w.iter_mut().zip(grad.iter()) {
        *w -= lr * *g;
        *w = w.clamp(-1.0, 1.0);
    }
}

/// Applies Muon to a wide matrix by splitting it into smaller chunks of columns.
pub fn chunked_muon_step(
    grad: &mut [f32],
    rows: usize,
    cols: usize,
    chunk_cols: usize,
    ns_iters: usize,
) {
    if rows == 0 || cols == 0 || chunk_cols == 0 || grad.is_empty() {
        return;
    }

    let n_chunks = cols.div_ceil(chunk_cols);
    for chunk_idx in 0..n_chunks {
        let c_start = chunk_idx * chunk_cols;
        let c_end = (c_start + chunk_cols).min(cols);
        let current_chunk_cols = c_end - c_start;

        if current_chunk_cols == 0 {
            continue;
        }

        let mut chunk_buf = vec![0.0f32; rows * current_chunk_cols];
        for r in 0..rows {
            let src_start = r * cols + c_start;
            let dst_start = r * current_chunk_cols;
            chunk_buf[dst_start..dst_start + current_chunk_cols]
                .copy_from_slice(&grad[src_start..src_start + current_chunk_cols]);
        }

        newton_schulz_orthogonalize(&mut chunk_buf, rows, current_chunk_cols, ns_iters);

        for r in 0..rows {
            let src_start = r * current_chunk_cols;
            let dst_start = r * cols + c_start;
            grad[dst_start..dst_start + current_chunk_cols]
                .copy_from_slice(&chunk_buf[src_start..src_start + current_chunk_cols]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_muon_orthogonalize() {
        let rows = 4;
        let cols = 4;
        let mut grad = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        newton_schulz_orthogonalize(&mut grad, rows, cols, 5);

        for g in grad.iter() {
            assert!(g.is_finite(), "Gradients should remain finite");
        }
    }

    #[test]
    fn test_muon_edge_cases() {
        let mut empty = vec![];
        newton_schulz_orthogonalize(&mut empty, 0, 0, 5);
        assert!(empty.is_empty());

        let mut zero_grad = vec![0.0; 16];
        newton_schulz_orthogonalize(&mut zero_grad, 4, 4, 5);
        for g in zero_grad.iter() {
            assert_eq!(*g, 0.0);
        }
    }

    #[test]
    fn test_ns_cpu_explicit() {
        let rows = 8;
        let cols = 8;
        let mut grad: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        newton_schulz_orthogonalize_cpu(&mut grad, rows, cols, 3);
        assert!(grad.iter().all(|g| g.is_finite()));
        // Not all zeros for non-zero input
        assert!(grad.iter().any(|g| g.abs() > 1e-8));
    }

    #[test]
    fn test_ns_avx_large_finite() {
        let rows = 64usize;
        let cols = 64usize;
        let mut grad: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();
        newton_schulz_orthogonalize_cpu(&mut grad, rows, cols, 3);
        assert!(grad.iter().all(|g| g.is_finite()));
        assert!(grad.iter().any(|g| g.abs() > 1e-8));
    }

    #[test]
    fn test_ns_vulkan_matches_cpu_if_available() {
        // Force Vulkan attempt; skip quietly if no device.
        // Use a *local* AshContext so we don't pin a process-global GPU context
        // that can hang teardown under cargo test.
        std::env::set_var("MUD_USE_VULKAN", "1");
        let rows = 64usize;
        let cols = 64usize;
        let mut base: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
            .collect();

        let mut cpu = base.clone();
        newton_schulz_orthogonalize_cpu(&mut cpu, rows, cols, 3);

        let Ok(mut ctx) = crate::vulkan::ash_backend::AshContext::new() else {
            return;
        };
        if !ctx.is_available() {
            return;
        }

        let g_norm = base.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in base.iter_mut() {
            *v /= g_norm;
        }
        let mut x = base;
        let ok = unsafe {
            ctx.dispatch_newton_schulz_sync(&mut x, rows, cols, 3)
                .is_ok()
        };
        if !ok {
            return;
        }
        for v in x.iter_mut() {
            *v *= g_norm;
        }

        let max_delta = cpu
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta < 5e-2,
            "Vulkan NS vs CPU max_delta={max_delta} (expect < 5e-2 for f32 GPU)"
        );
        // Explicit drop before test ends
        drop(ctx);
    }
}
