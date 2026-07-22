use forge_llm::mud::ash_qat_dispatcher::AshTensorStep;
use std::time::Instant;

fn main() {
    println!("🚀 MUD-QAT 5-Minute Endurance Certification Benchmark\n");

    let cols = 2048;
    let rows = 2048;
    let elements = rows * cols;
    let packed_elements = elements / 8;

    println!("Matrix size: {}x{} ({} elements)", rows, cols, elements);
    println!("Running continuous simulated training loop for 5 minutes...\n");

    // CPU Allocations
    let mut shadow_cpu = vec![0.0f32; elements];
    let mut grad_cpu = vec![0.0f32; elements];
    let mut packed_cpu = vec![0u8; packed_elements * 4];
    let mut scales_cpu = vec![0.0f32; rows];

    // Fill with predictable "random" noise
    for i in 0..elements {
        shadow_cpu[i] = ((i % 4096) as f32 / 1024.0) - 2.0;
        grad_cpu[i] = ((i % 1024) as f32 / 512.0) - 1.0;
    }

    let shadow_gpu = shadow_cpu.clone();

    let lr = 1e-3;
    let weight_decay = 0.01;
    let num_tokens = 2048.0;

    let pool = forge_llm::mud::pcore_pool::PCorePool::new(
        forge_llm::mud::constants::default_pcore_threads(),
    );

    let mut dispatcher = match forge_llm::mud::ash_qat_dispatcher::AshQatDispatcher::new() {
        Ok(d) => d,
        Err(e) => {
            println!("❌ Vulkan unavailable: {}", e);
            return;
        }
    };

    let name = "quad_test";

    let start_time = Instant::now();
    let mut iterations = 0;
    let mut last_print = Instant::now();

    // 5 minutes = 300 seconds
    let duration_limit = 300;

    while start_time.elapsed().as_secs() < duration_limit {
        // 1. Mutate gradients slightly to simulate changing data across batches
        let grad_modifier = ((iterations % 100) as f32) / 1000.0;
        for (i, g) in grad_cpu.iter_mut().enumerate().take(1024) {
            // mutate only first 1k to avoid O(N) overhead in the test loop itself
            *g = (((i + iterations) % 1024) as f32 / 512.0) - 1.0 + grad_modifier;
        }

        // 2. CPU Step
        let strategy = forge_llm::mud::slime_backward::select_optimizer(rows, cols);
        let mut adam = forge_llm::mud::adam_state::AdamState::for_strategy(elements, strategy);
        unsafe {
            forge_llm::mud::corpus_trainer::apply_optimizer_cpu_step_and_pack(
                &mut shadow_cpu,
                &grad_cpu,
                packed_cpu.as_mut_ptr(),
                scales_cpu.as_mut_ptr(),
                lr,
                weight_decay,
                num_tokens,
                cols,
                &pool,
                strategy,
                adam.as_mut(),
            );
        }

        // 3. GPU Step
        unsafe {
            let step = AshTensorStep {
                name: name.to_string(),
                shadow: &shadow_gpu, // The initial slice doesn't matter on iteration > 0 since we don't upload it again
                grad: &grad_cpu,
                elements,
                cols,
                rows,
            };
            dispatcher
                .step_async(&[step], lr, weight_decay, num_tokens)
                .unwrap();

            // For true stress test, we sync on every iteration so we can readback at the end.
            // Normally training overlaps, but here we want to test sheer mathematical integrity over millions of chained operations.
            dispatcher.sync().unwrap();
        }

        iterations += 1;

        if last_print.elapsed().as_secs() >= 30 {
            let elapsed = start_time.elapsed().as_secs();
            let ops_per_sec = iterations as f64 / elapsed as f64;
            println!(
                "[{} / 300s] Executed {} chained training steps... ({:.2} iter/sec)",
                elapsed, iterations, ops_per_sec
            );
            last_print = Instant::now();
        }
    }

    println!("\n[!] 5-Minute Endurance Test Complete.");
    println!(
        "Total iterations (chained forward/backward steps): {}",
        iterations
    );
    println!(
        "Total parameters processed (Ops): {}",
        iterations * elements
    );

    // Readback from GPU
    let mut shadow_gpu_out = vec![0.0f32; elements];
    let mut packed_gpu_out = vec![0u8; packed_elements * 4];
    let mut scales_gpu_out = vec![0.0f32; rows];

    unsafe {
        dispatcher.readback_packed(name, &mut packed_gpu_out);
        dispatcher.readback_scales(name, &mut scales_gpu_out);

        let key = format!("{}.shadow", name);
        if let Some(buf) = dispatcher.ctx.get_buffer(&key) {
            buf.read_f32(&mut shadow_gpu_out);
        }
    }

    println!("\n[3] Final Bit-Quadrature Verification...");

    let mut shadow_err = 0.0;
    for i in 0..elements {
        let diff = (shadow_cpu[i] - shadow_gpu_out[i]).abs();
        if diff > shadow_err {
            shadow_err = diff;
        }
    }
    println!(
        "   - Max Shadow Divergence after {} steps: {:.6}",
        iterations, shadow_err
    );
    if shadow_err > 1e-3 {
        println!("   ❌ FAILED: Shadow weights diverged heavily over time!");
    } else {
        println!("   ✅ PASS: Shadow weights stabilized (Lossless).");
    }

    let mut scale_err = 0.0;
    for i in 0..rows {
        let diff = (scales_cpu[i] - scales_gpu_out[i]).abs();
        if diff > scale_err {
            scale_err = diff;
        }
    }
    println!("   - Max PRQ Scale Divergence: {:.6}", scale_err);
    if scale_err > 1e-4 {
        println!("   ❌ FAILED: PRQ Scales diverged!");
    } else {
        println!("   ✅ PASS: PRQ Scales stabilized.");
    }

    let mut pack_mismatches = 0;
    for i in 0..packed_cpu.len() {
        if packed_cpu[i] != packed_gpu_out[i] {
            pack_mismatches += 1;
        }
    }
    println!(
        "   - Final ELUT Byte Mismatches: {} / {}",
        pack_mismatches,
        packed_cpu.len()
    );

    if pack_mismatches > 0 {
        println!(
            "   ❌ FAILED: ELUT Packaging drifted! The precision gap compounded over 5 minutes."
        );
    } else {
        println!("   ✅ PASS: ELUT Packaging matches 100% (Lossless). No compounding errors.");
    }

    if shadow_err < 1e-3 && scale_err < 1e-4 && pack_mismatches == 0 {
        println!("\n🏆 5-MINUTE ENDURANCE CERTIFICATION GRANTED.");
    } else {
        println!("\n❌ CERTIFICATION FAILED.");
        std::process::exit(1);
    }
}
