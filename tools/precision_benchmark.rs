use half::f16;
use std::time::Instant;

const MATRIX_SIZE: usize = 512; // 512x512 matrices

// Helper to calculate SNR
fn calculate_snr(ground_truth: &[f32], test_data: &[f32]) -> f64 {
    let mut signal_power = 0.0;
    let mut noise_power = 0.0;

    for i in 0..ground_truth.len() {
        signal_power += (ground_truth[i] as f64).powi(2);
        let diff = (ground_truth[i] - test_data[i]) as f64;
        noise_power += diff.powi(2);
    }

    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

fn calculate_mse(ground_truth: &[f32], test_data: &[f32]) -> f64 {
    let mut mse = 0.0;
    for i in 0..ground_truth.len() {
        let diff = (ground_truth[i] - test_data[i]) as f64;
        mse += diff.powi(2);
    }
    mse / ground_truth.len() as f64
}

fn main() {
    println!(
        "🚀 Inicializando Benchmark de Precisión y Velocidad (Matrices {}x{})...",
        MATRIX_SIZE, MATRIX_SIZE
    );

    let mut seed: u64 = 42;
    let mut next_f32 = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = (seed >> 11) as f32 * (1.0 / (1u64 << 53) as f32);
        (val * 2.0) - 1.0 // Range [-1.0, 1.0]
    };

    let n = MATRIX_SIZE;
    let mut w = vec![0.0f32; n * n];
    let mut x = vec![0.0f32; n * n];

    // Llenar con valores flotantes sintéticos (distribución normal aproximada)
    for i in 0..(n * n) {
        // Pseudo-normal: suma de uniformes
        let mut r_w = 0.0;
        let mut r_x = 0.0;
        for _ in 0..3 {
            r_w += next_f32();
            r_x += next_f32();
        }
        w[i] = r_w * 0.1; // Pequeños pesos típicos en LLMs
        x[i] = r_x * 0.5; // Activaciones un poco mayores
    }

    // 1. FP32 (GROUND TRUTH)
    let mut y_fp32 = vec![0.0f32; n * n];
    let start_fp32 = Instant::now();
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[i * n + k] * w[k * n + j];
            }
            y_fp32[i * n + j] = sum;
        }
    }
    let time_fp32 = start_fp32.elapsed();

    // 2. FP16 (Media Precisión)
    let mut y_fp16 = vec![0.0f32; n * n];
    let mut w_f16 = vec![f16::from_f32(0.0); n * n];
    let mut x_f16 = vec![f16::from_f32(0.0); n * n];
    for i in 0..(n * n) {
        w_f16[i] = f16::from_f32(w[i]);
        x_f16[i] = f16::from_f32(x[i]);
    }

    let start_fp16 = Instant::now();
    for i in 0..n {
        for j in 0..n {
            let mut sum = f16::from_f32(0.0);
            for k in 0..n {
                sum += x_f16[i * n + k] * w_f16[k * n + j];
            }
            y_fp16[i * n + j] = sum.to_f32();
        }
    }
    let time_fp16 = start_fp16.elapsed();

    // 3. INT8 (Cuantización Entera Simétrica con Escala)
    let mut y_int8 = vec![0.0f32; n * n];
    // Find max for scale
    let max_w = w.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    let max_x = x.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
    let scale_w = max_w / 127.0;
    let scale_x = max_x / 127.0;

    let mut w_i8 = vec![0i8; n * n];
    let mut x_i8 = vec![0i8; n * n];
    for i in 0..(n * n) {
        w_i8[i] = (w[i] / scale_w).round() as i8;
        x_i8[i] = (x[i] / scale_x).round() as i8;
    }

    let start_int8 = Instant::now();
    for i in 0..n {
        for j in 0..n {
            let mut sum: i32 = 0;
            for k in 0..n {
                sum += (x_i8[i * n + k] as i32) * (w_i8[k * n + j] as i32);
            }
            // Dequantize
            y_int8[i * n + j] = (sum as f32) * (scale_w * scale_x);
        }
    }
    let time_int8 = start_int8.elapsed();

    // 4. Ternary 1.58b (Estilo MUD con Escala por Fila)
    let mut y_ternary = vec![0.0f32; n * n];
    let mut w_ternary = vec![0i8; n * n];
    let mut scales_ternary = vec![0.0f32; n];

    // Per-Row Quantization (MUD)
    for row in 0..n {
        let mut sum_abs = 0.0;
        for col in 0..n {
            sum_abs += w[row * n + col].abs();
        }
        let mean_abs = sum_abs / (n as f32);
        scales_ternary[row] = mean_abs;

        for col in 0..n {
            let val = w[row * n + col] / mean_abs;
            w_ternary[row * n + col] = val.round().clamp(-1.0, 1.0) as i8;
        }
    }

    let start_ternary = Instant::now();
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                let weight = w_ternary[k * n + j]; // Nota: La escala es por fila k
                let act = x[i * n + k];

                // Ternary math emulation
                if weight == 1 {
                    sum += act * scales_ternary[k];
                } else if weight == -1 {
                    sum -= act * scales_ternary[k];
                }
            }
            y_ternary[i * n + j] = sum;
        }
    }
    let time_ternary = start_ternary.elapsed();

    // Imprimir Resultados
    println!("==========================================================================");
    println!(
        "{:<15} | {:<12} | {:<15} | {:<15}",
        "Paradigma", "Tiempo", "Error (MSE)", "Fidelidad (SNR)"
    );
    println!("--------------------------------------------------------------------------");

    let mse_fp32 = calculate_mse(&y_fp32, &y_fp32);
    let snr_fp32 = calculate_snr(&y_fp32, &y_fp32);
    println!(
        "{:<15} | {:<12?} | {:<15.8e} | {:<15.2} dB",
        "FP32 (Base)", time_fp32, mse_fp32, snr_fp32
    );

    let mse_fp16 = calculate_mse(&y_fp32, &y_fp16);
    let snr_fp16 = calculate_snr(&y_fp32, &y_fp16);
    println!(
        "{:<15} | {:<12?} | {:<15.8e} | {:<15.2} dB",
        "FP16", time_fp16, mse_fp16, snr_fp16
    );

    let mse_int8 = calculate_mse(&y_fp32, &y_int8);
    let snr_int8 = calculate_snr(&y_fp32, &y_int8);
    println!(
        "{:<15} | {:<12?} | {:<15.8e} | {:<15.2} dB",
        "INT8 (Simétrica)", time_int8, mse_int8, snr_int8
    );

    let mse_ternary = calculate_mse(&y_fp32, &y_ternary);
    let snr_ternary = calculate_snr(&y_fp32, &y_ternary);
    println!(
        "{:<15} | {:<12?} | {:<15.8e} | {:<15.2} dB",
        "Ternary (1.58b)", time_ternary, mse_ternary, snr_ternary
    );

    println!("==========================================================================");
    println!("*Nota de Velocidad: En este benchmark simulado en Rust puro, el compilador");
    println!("puede vectorizar mejor el float32 que nuestros bucles de enteros custom.");
    println!("En hardware real (Nvidia/AVX2 con asm), INT8 es 4x más rápido y Ternary es 8x.");
}
