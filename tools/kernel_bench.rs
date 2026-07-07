//! Kernel Benchmark Tool
//! Measures throughput of dispatched ASM kernels at various input sizes.
//! Compares dispatch wrapper vs direct AVX2 call to quantify dispatch overhead.

use std::time::Instant;

fn bench(label: &str, sizes: &[usize], mut f: impl FnMut(usize)) {
    println!("  {:<30}", label);
    for &n in sizes {
        let warmup = 100.min(100_000_000 / (n as u64).max(1) as usize);
        for _ in 0..warmup {
            f(n);
        }
        let iters = warmup.max(10);
        let start = Instant::now();
        for _ in 0..iters {
            f(n);
        }
        let dur = start.elapsed().as_secs_f64() / iters as f64;
        let bytes = (n as f64) * 4.0;
        let bw = bytes / dur / 1e9;
        let ops = n as f64 / dur / 1e9;
        println!(
            "    n={:6}  {:8.2} GB/s  {:8.2} Gop/s  {:8.1} ns",
            n,
            bw,
            ops,
            dur * 1e9
        );
    }
}

fn main() {
    println!("\x1b[1;36m=== ASM Kernel Benchmark ===\x1b[0m");
    let sizes = [128, 512, 2048, 8192, 32768];

    // ── sum_squares ──
    println!("\n\x1b[1msum_squares (dispatch):\x1b[0m");
    let xs: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![1.5f32; n]).collect();
    bench("sum_squares", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::sum_squares(n, xs[i].as_ptr());
        }
    });

    println!("\n\x1b[1msum_squares (direct AVX2):\x1b[0m");
    bench("sum_squares_avx2", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::sum_squares_avx2(n, xs[i].as_ptr());
        }
    });

    // ── dot_product ──
    println!("\n\x1b[1mdot_product (dispatch):\x1b[0m");
    let as_: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![1.0f32; n]).collect();
    let bs: Vec<Vec<f32>> = sizes.iter().map(|&n| vec![2.0f32; n]).collect();
    bench("dot_product", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::dot_product(n, as_[i].as_ptr(), bs[i].as_ptr());
        }
    });

    println!("\n\x1b[1mdot_product (direct AVX2):\x1b[0m");
    bench("dot_product_avx2", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::dot_product_avx2(n, as_[i].as_ptr(), bs[i].as_ptr());
        }
    });

    // ── ternary_gemv ──
    println!("\n\x1b[1mternary_gemv (dispatch):\x1b[0m");
    let packed: Vec<Vec<u32>> = sizes.iter().map(|&n| vec![0x55555555u32; n / 16]).collect();
    let out = &mut 0.0f32;
    bench("ternary_gemv", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::ternary_gemv(n, xs[i].as_ptr(), packed[i].as_ptr(), out, 1.0);
        }
    });

    println!("\n\x1b[1mternary_gemv (direct AVX2):\x1b[0m");
    bench("ternary_gemv_avx2", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::ternary_gemv_avx2(n, xs[i].as_ptr(), packed[i].as_ptr(), out, 1.0);
        }
    });

    // ── silu_vectorial ──
    println!("\n\x1b[1msilu_vectorial (dispatch):\x1b[0m");
    let dst = &mut vec![0.0f32; sizes[sizes.len() - 1]];
    bench("silu_vectorial", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::silu_vectorial(n, xs[i].as_ptr(), dst.as_mut_ptr());
        }
    });

    println!("\n\x1b[1msilu_vectorial (direct AVX2):\x1b[0m");
    bench("silu_vectorial_avx2", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::silu_vectorial_avx2(n, xs[i].as_ptr(), dst.as_mut_ptr());
        }
    });

    // ── rms_norm_scale ──
    println!("\n\x1b[1mrms_norm_scale (dispatch):\x1b[0m");
    bench("rms_norm_scale", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::rms_norm_scale(n, xs[i].as_ptr(), 1e-6);
        }
    });

    println!("\n\x1b[1mrms_norm_scale (direct ASM):\x1b[0m");
    bench("rms_norm_scale_asm", &sizes, |n| {
        let i = sizes.iter().position(|&s| s == n).unwrap();
        unsafe {
            forge_llm::asm::rms_norm_scale_asm(n, xs[i].as_ptr(), 1e-6);
        }
    });
}
