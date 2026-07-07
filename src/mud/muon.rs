/// Newton-Schulz orthogonalization of the gradient matrix.
/// Replaces the gradient in-place with its orthogonalized version.
/// Complexity is O(iters * rows * cols * cols), so we parallelize over rows.
pub fn newton_schulz_orthogonalize(
    grad: &mut [f32],
    rows: usize,
    cols: usize,
    n_iters: usize,
) {
    if rows == 0 || cols == 0 || grad.is_empty() {
        return;
    }

    let g_norm = grad.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in grad.iter_mut() {
        *x /= g_norm;
    }

    let mut x = grad.to_vec();
    let mut tmp = vec![0.0f32; cols * cols];
    for _ in 0..n_iters {
        muon_step_inner(&mut x, &mut tmp, rows, cols);
    }

    for (g, xi) in grad.iter_mut().zip(x.iter()) {
        *g = xi * g_norm;
    }
}

fn muon_step_inner(x: &mut [f32], tmp: &mut [f32], rows: usize, cols: usize) {
    // tmp = X^T * X (cols x cols, serial)
    for j in 0..cols {
        for k in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                sum += x[i * cols + j] * x[i * cols + k];
            }
            tmp[j * cols + k] = sum;
        }
    }

    // next_x = 1.5 * X - 0.5 * X * tmp
    let mut next_x = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..cols {
                sum += x[i * cols + k] * tmp[k * cols + j];
            }
            next_x[i * cols + j] = 1.5 * x[i * cols + j] - 0.5 * sum;
        }
    }
    
    x.copy_from_slice(&next_x);
}

/// Applies the Muon QAT step to a weight matrix.
/// - Orthogonalizes the gradient
/// - Applies it as SGD
/// - Clamps back to [-1.0, 1.0] (P-15 Hot PRQ clamp)
pub fn muon_qat_step(
    shadow_w: &mut [f32],
    grad: &mut [f32],
    rows: usize,
    cols: usize,
    lr: f32,
) {
    newton_schulz_orthogonalize(grad, rows, cols, 5);
    for (w, g) in shadow_w.iter_mut().zip(grad.iter()) {
        *w -= lr * g;
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

        // Extract chunk into a dense matrix
        let mut chunk_buf = vec![0.0f32; rows * current_chunk_cols];
        for r in 0..rows {
            let src_start = r * cols + c_start;
            let dst_start = r * current_chunk_cols;
            chunk_buf[dst_start..dst_start + current_chunk_cols]
                .copy_from_slice(&grad[src_start..src_start + current_chunk_cols]);
        }

        // Apply Muon
        newton_schulz_orthogonalize(&mut chunk_buf, rows, current_chunk_cols, ns_iters);

        // Put back the chunk
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
            1.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0,
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
}
