// Rayon will be used in future optimization passes

/// Computes the top `r` left singular vectors of `grad` (size rows x cols) using Power Iteration.
/// Returns a flat matrix `P` of size `rows x r`.
pub fn compute_projection_matrix(grad: &[f32], rows: usize, cols: usize, r: usize, iters: usize) -> Vec<f32> {
    let mut p_matrix = vec![0.0f32; rows * r];
    let mut g_residual = grad.to_vec();

    for i in 0..r {
        // Initialize u randomly
        let mut u = vec![0.0f32; rows];
        let mut rng = 1337u32 + i as u32;
        for val in u.iter_mut() {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = (rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut v = vec![0.0f32; cols];
        for _ in 0..iters {
            // v = G^T * u
            for c in 0..cols {
                let mut sum = 0.0;
                for r_idx in 0..rows {
                    sum += g_residual[r_idx * cols + c] * u[r_idx];
                }
                v[c] = sum;
            }
            // Normalize v
            let v_norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for val in v.iter_mut() { *val /= v_norm; }

            // u = G * v
            for r_idx in 0..rows {
                let mut sum = 0.0;
                for c in 0..cols {
                    sum += g_residual[r_idx * cols + c] * v[c];
                }
                u[r_idx] = sum;
            }
            // Normalize u
            let u_norm = u.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for val in u.iter_mut() { *val /= u_norm; }
        }

        // Compute sigma
        let mut sigma = 0.0;
        for r_idx in 0..rows {
            for c in 0..cols {
                sigma += u[r_idx] * g_residual[r_idx * cols + c] * v[c];
            }
        }

        // Deflation: G_residual -= sigma * u * v^T
        for r_idx in 0..rows {
            for c in 0..cols {
                g_residual[r_idx * cols + c] -= sigma * u[r_idx] * v[c];
            }
        }

        // Store u in P matrix column i
        for r_idx in 0..rows {
            p_matrix[r_idx * r + i] = u[r_idx];
        }
    }

    p_matrix
}

/// Applies GaLore projection to the gradient.
/// G_low = P^T * G (size: r x cols)
/// Then it passes G_low through SGD (simply returning it).
/// Then G_updated = P * G_low (size: rows x cols)
pub fn galore_step(grad: &mut [f32], rows: usize, cols: usize, rank: usize) {
    if rows <= cols { return; } // Only implemented for tall matrices right now

    let p_matrix = compute_projection_matrix(grad, rows, cols, rank, 3); // 3 iters is usually enough for top approx
    
    // 1. G_low = P^T * G
    let mut g_low = vec![0.0f32; rank * cols];
    for c in 0..cols {
        for i in 0..rank {
            let mut sum = 0.0;
            for r_idx in 0..rows {
                // P is rows x r, so P[r_idx, i] = p_matrix[r_idx * rank + i]
                sum += p_matrix[r_idx * rank + i] * grad[r_idx * cols + c];
            }
            g_low[i * cols + c] = sum;
        }
    }

    // (Adam would go here on g_low, but we just pass through for STE compatibility phase 1)

    // 2. G_updated = P * G_low
    for r_idx in 0..rows {
        for c in 0..cols {
            let mut sum = 0.0;
            for i in 0..rank {
                sum += p_matrix[r_idx * rank + i] * g_low[i * cols + c];
            }
            grad[r_idx * cols + c] = sum;
        }
    }
}
