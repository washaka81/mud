use std::time::Instant;

fn gauss_jordan_inverse(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut augmented = vec![0.0; n * n * 2];

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented[i * (2 * n) + j] = matrix[i * n + j];
        }
        augmented[i * (2 * n) + n + i] = 1.0;
    }

    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        let mut max_val = augmented[i * (2 * n) + i].abs();
        for k in i + 1..n {
            let val = augmented[k * (2 * n) + i].abs();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }

        if max_val < 1e-10 {
            return None; // Singular matrix
        }

        // Swap rows if necessary
        if pivot_row != i {
            for j in 0..(2 * n) {
                augmented.swap(i * (2 * n) + j, pivot_row * (2 * n) + j);
            }
        }

        // Scale pivot row
        let pivot_val = augmented[i * (2 * n) + i];
        for j in 0..(2 * n) {
            augmented[i * (2 * n) + j] /= pivot_val;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = augmented[k * (2 * n) + i];
                for j in 0..(2 * n) {
                    augmented[k * (2 * n) + j] -= factor * augmented[i * (2 * n) + j];
                }
            }
        }
    }

    // Extract inverse
    let mut inverse = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inverse[i * n + j] = augmented[i * (2 * n) + n + j];
        }
    }

    Some(inverse)
}

fn multiply_matrices(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

fn main() {
    let sizes = [10, 50, 100, 200, 500]; // Diferentes tamaños para el benchmark

    // Simple LCG PRNG to avoid rand crate version issues
    let mut seed: u64 = 123456789;
    let mut next_f64 = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (seed >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
        (val * 20.0) - 10.0 // Range [-10.0, 10.0]
    };

    println!("============================================================");
    println!(" Benchmark de Precisión: Multiplicación de Matriz x Inversa ");
    println!("============================================================");

    for &n in &sizes {
        // Generate random matrix
        let mut matrix = vec![0.0; n * n];
        for val in &mut matrix {
            *val = next_f64();
        }

        let start_inv = Instant::now();
        if let Some(inverse) = gauss_jordan_inverse(&matrix, n) {
            let inv_time = start_inv.elapsed();

            let start_mul = Instant::now();
            let identity_approx = multiply_matrices(&matrix, &inverse, n);
            let mul_time = start_mul.elapsed();

            // Check precision (Maximum Absolute Error compared to Identity matrix)
            let mut max_error = 0.0f64;
            let mut mse = 0.0f64;

            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let actual = identity_approx[i * n + j];
                    let err = (expected - actual).abs();

                    if err > max_error {
                        max_error = err;
                    }
                    mse += err * err;
                }
            }
            mse /= (n * n) as f64;

            println!("Tamaño: {:<4}x{:<4} | Inv Time: {:<8?} | Mul Time: {:<8?} | Max Error: {:e} | MSE: {:e}", 
                n, n, inv_time, mul_time, max_error, mse);
        } else {
            println!(
                "Tamaño: {:<4}x{:<4} | Matriz Singular (No se pudo invertir)",
                n, n
            );
        }
    }
    println!("============================================================");
}
