// Benchmark simple para comparar LM head ASM vs scalar
use std::time::Instant;

fn lm_head_scalar(vocab_size: usize, hidden: usize, regs: &[f32], weights: &[f32]) -> usize {
    let mut best_id = 0;
    let mut max_logit = f32::NEG_INFINITY;

    for v in 0..vocab_size {
        let mut logit = 0.0f32;
        for h in 0..hidden {
            logit += regs[h] * weights[v * hidden + h];
        }
        if logit > max_logit {
            max_logit = logit;
            best_id = v;
        }
    }
    best_id
}

fn main() {
    // Parámetros del modelo BitNet
    let vocab_size = 128256;
    let hidden = 2560;

    println!(
        "LM Head Benchmark: vocab_size={}, hidden={}",
        vocab_size, hidden
    );
    println!(
        "Total ops: {} M ({} dot products × {} dims)",
        (vocab_size * hidden) as f64 / 1e6,
        vocab_size,
        hidden
    );

    // Generar datos aleatorios
    let regs: Vec<f32> = (0..hidden).map(|i| (i as f32).sin()).collect();
    let weights: Vec<f32> = (0..vocab_size * hidden)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();

    // Warmup
    let result = lm_head_scalar(vocab_size, hidden, &regs, &weights);
    println!("Warmup result: {}", result);

    // Benchmark scalar
    let iterations = 10;
    let mut sum = 0usize;
    let start = Instant::now();
    for _ in 0..iterations {
        sum += lm_head_scalar(vocab_size, hidden, &regs, &weights);
    }
    let scalar_time = start.elapsed().as_millis() as f64 / iterations as f64;

    println!("\nScalar implementation:");
    println!("  Time per call: {:.2} ms", scalar_time);
    println!(
        "  Throughput: {:.2} M ops/sec",
        (vocab_size * hidden) as f64 / (scalar_time * 1e3)
    );
    println!("  Checksum: {}", sum); // Prevent optimization

    // Benchmark ASM
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // Warmup ASM
            let result = unsafe {
                forge_llm::asm::lm_head(vocab_size, hidden, regs.as_ptr(), weights.as_ptr())
            };
            println!("ASM warmup result: {}", result);

            let mut sum_asm = 0usize;
            let start = Instant::now();
            for _ in 0..iterations {
                sum_asm += unsafe {
                    forge_llm::asm::lm_head(vocab_size, hidden, regs.as_ptr(), weights.as_ptr())
                };
            }
            let asm_time = start.elapsed().as_millis() as f64 / iterations as f64;

            println!("\nAVX2 ASM implementation:");
            println!("  Time per call: {:.2} ms", asm_time);
            println!(
                "  Throughput: {:.2} M ops/sec",
                (vocab_size * hidden) as f64 / (asm_time * 1e3)
            );
            println!("  Checksum: {}", sum_asm); // Prevent optimization
            println!("\nSpeedup: {:.2}x", scalar_time / asm_time);
        } else {
            println!("\nAVX2 not available on this CPU");
        }
    }
}
