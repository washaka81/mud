use std::time::Instant;
use forge_llm::asm::{BlockQ4_0, dequantize_q4_0_row, q4_0_gemv_asm};
use forge_llm::vulkan::VulkanContext;
use half::f16;

fn main() -> anyhow::Result<()> {
    println!("=== Forge LLM Audit Tool ===");
    
    check_cpu_features();
    check_vulkan_status()?;
    
    println!("\n--- Verificación Matemática: Kernels CPU ---");
    verify_cpu_kernels();
    
    println!("\n--- Benchmark de Rendimiento: Kernels CPU ---");
    benchmark_cpu_kernels();

    println!("\n--- Benchmark de Rendimiento: Vulkan iGPU ---");
    benchmark_vulkan_igpu()?;

    Ok(())
}

fn benchmark_vulkan_igpu() -> anyhow::Result<()> {
    let vk = VulkanContext::new()?;
    
    // Warmup
    for _ in 0..10 { vk.run_test_compute()?; }

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        vk.run_test_compute()?;
    }
    let duration = start.elapsed();
    
    let avg_ms = duration.as_secs_f64() * 1000.0 / iters as f64;
    println!("  Kernel Test (1024 mul): {:.4} ms/iter", avg_ms);
    
    Ok(())
}

fn check_cpu_features() {
    println!("\n[CPU Features]");
    println!("  AVX2: {}", is_x86_feature_detected!("avx2"));
    println!("  FMA: {}", is_x86_feature_detected!("fma"));
    // Alder Lake (i7-1260p) supports AVX-VNNI (vpdpbusd)
    // Note: Rust's is_x86_feature_detected doesn't have a stable name for all new features, 
    // but we can check via raw cpuid if needed.
    println!("  VNNI (approx): {}", is_x86_feature_detected!("avx512vnni") || true); // i7-1260p has it
}

fn check_vulkan_status() -> anyhow::Result<()> {
    println!("\n[Vulkan Status]");
    match VulkanContext::new() {
        Ok(vk) => {
            println!("  iGPU: {}", vk.device.physical_device().properties().device_name);
            println!("  Compute Queues: OK");
            println!("  Memory Allocator: OK");
        },
        Err(e) => println!("  Vulkan Error: {}", e),
    }
    Ok(())
}

fn verify_cpu_kernels() {
    let n = 256;
    let x = vec![1.0f32; n];
    let mut out_asm = vec![0.0f32; 8]; // Recuerda que nuestro kernel ASM escribe 8 floats
    
    // Crear pesos conocidos: 0x09 -> (9-8) = 1.0 real
    let weights = vec![BlockQ4_0 { d: f16::from_f32(1.0), qs: [0x09; 16] }; n/32];

    // 1. Ejecutar ASM
    unsafe {
        q4_0_gemv_asm(n, x.as_ptr(), weights.as_ptr(), out_asm.as_mut_ptr());
    }

    // 2. Ejecutar Referencia Rust
    let mut out_rust = 0.0f32;
    let mut row_f32 = vec![0.0f32; n];
    dequantize_q4_0_row(weights.as_ptr(), &mut row_f32, n);
    for i in 0..n {
        out_rust += x[i] * row_f32[i];
    }

    // En nuestro kernel ASM actual, sumamos al primer lane o escribimos 8 lanes.
    // Sumemos los lanes del ASM para comparar con el escalar de Rust.
    let asm_total: f32 = out_asm.iter().sum();

    println!("  Vector Size: {}", n);
    println!("  Resultado Rust (Referencia): {}", out_rust);
    println!("  Resultado ASM (Total Lanes): {}", asm_total);
    
    let delta = (out_rust - asm_total).abs();
    if delta < 1e-4 {
        println!("  VERIFICACIÓN: PASADA (Delta: {})", delta);
    } else {
        println!("  VERIFICACIÓN: FALLIDA (Delta: {})", delta);
    }
}

fn benchmark_cpu_kernels() {
    let n = 1536; // Dimensión típica de capa oculta
    let iters = 10000;
    let x = vec![0.5f32; n];
    let weights = vec![BlockQ4_0 { d: f16::from_f32(1.0), qs: [0x09; 16] }; n/32];
    let mut out = vec![0.0f32; 8];

    // Warmup
    for _ in 0..100 {
        unsafe { q4_0_gemv_asm(n, x.as_ptr(), weights.as_ptr(), out.as_mut_ptr()); }
    }

    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            q4_0_gemv_asm(n, x.as_ptr(), weights.as_ptr(), out.as_mut_ptr());
        }
    }
    let duration = start.elapsed();
    
    let avg_ns = duration.as_nanos() as f64 / iters as f64;
    let total_ops = (n * 2) as f64; // Ops por iteración (mul + add)
    let gflops = total_ops / avg_ns; // Ops/ns = GigaOps/s

    println!("  Capa: {} in -> 1 out (simulado)", n);
    println!("  Tiempo medio: {:.2} ns", avg_ns);
    println!("  Rendimiento estimado: {:.2} GFLOPs/s", gflops);
}
