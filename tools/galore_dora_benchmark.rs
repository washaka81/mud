use std::time::Instant;

fn main() {
    println!("🧪 GALORE & DORA NATIVE BENCHMARK");
    println!("---------------------------------");
    let rows = 4096;
    let cols = 4096;
    println!(
        "Allocating synthetic tensor: {}x{} ({:.2} MB)",
        rows,
        cols,
        (rows * cols * 4) as f32 / 1024.0 / 1024.0
    );

    // Simulate w_fp32
    let mut w_fp32 = vec![0.01f32; rows * cols];

    // Simulate x
    let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin() * 0.1).collect();

    // In order to call ghost_align_cpu, we need an instance of MudCorpusTrainer.
    // However, since it requires a real file, we can just extract the mathematical logic here
    // or we can test it directly if we modify it to be standalone.

    // Actually, I'll just run the logic directly here to measure throughput of the exact routine.
    let start = Instant::now();
    let iters = 10;

    for _ in 0..iters {
        let mut x_proj = vec![0.0f32; cols];
        let mut rng_galore = 999u32;
        let r_rank = 16.min(cols);

        let mut p_matrix = vec![0.0f32; cols * r_rank];
        for p_val in p_matrix.iter_mut() {
            rng_galore = rng_galore.wrapping_mul(1664525).wrapping_add(1013904223);
            *p_val = (rng_galore as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }

        let mut x_low_rank = vec![0.0f32; r_rank];
        for i in 0..r_rank {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += x[j] * p_matrix[j * r_rank + i];
            }
            x_low_rank[i] = sum;
        }
        for j in 0..cols {
            let mut sum = 0.0;
            for i in 0..r_rank {
                sum += x_low_rank[i] * p_matrix[j * r_rank + i];
            }
            x_proj[j] = sum / (r_rank as f32);
        }

        for r in 0..rows {
            let start = r * cols;
            let row_slice = &w_fp32[start..start + cols];

            let mut abs_sum = 0.0;
            for v in row_slice.iter() {
                abs_sum += v.abs();
            }
            let absmean = abs_sum / cols as f32;
            let mut scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
            let delta = 0.7 * absmean;

            let mut y_master = 0.0;
            for c in 0..cols {
                y_master += row_slice[c] * x[c];
            }

            let mut y_student_unscaled = 0.0;
            for c in 0..cols {
                let w_f = row_slice[c];
                if w_f > delta {
                    y_student_unscaled += x[c];
                } else if w_f < -delta {
                    y_student_unscaled -= x[c];
                }
            }

            let y_student = scale * y_student_unscaled;
            let err = y_student - y_master;

            if err.abs() > 1e-9 {
                let grad_scale = err * y_student_unscaled;
                scale -= 0.005 * grad_scale;

                let alpha = -0.001 * err * scale;
                // Direct decay and alpha apply
                for c in 0..cols {
                    w_fp32[start + c] = w_fp32[start + c] * (1.0 - 0.0001) + alpha * x_proj[c];
                }
            }
        }
    }
    let elapsed = start.elapsed().as_secs_f32();
    let per_iter = elapsed / (iters as f32);

    // (rows * cols) * 2 ops for student forward
    // (rows * cols) * 2 for master forward
    // (rows * cols) * 2 for gradient apply
    // + GaLore P matrix generation
    let gflops = ((rows * cols * 6 * iters) as f32) / elapsed / 1e9;

    println!("✅ RESULTS:");
    println!("   Total Time: {:.2} s", elapsed);
    println!("   Time/Iter:  {:.2} ms", per_iter * 1000.0);
    println!(
        "   Throughput: {:.2} GFLOPS (Effective QAT Healing)",
        gflops
    );
}
