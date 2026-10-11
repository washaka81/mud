use forge_autograd::avx_math::{axpy_avx2, dot_product_avx2, silu_avx2};
use forge_llm::asm;
use std::io::Write;

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn axpy_scalar(a: &mut [f32], alpha: f32, b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) {
        *x += alpha * y;
    }
}

fn silu_scalar(a: &[f32], out: &mut [f32]) {
    for (x, y) in a.iter().zip(out.iter_mut()) {
        *y = *x / (1.0 + (-*x).exp());
    }
}

fn sum_squares_scalar(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

fn assert_f32_eq(a: f32, b: f32, tol: f32, ctx: &str) -> f32 {
    let diff = (a - b).abs();
    if diff > tol {
        panic!(
            "Validation Failed: {} | AVX2: {}, Scalar: {} | Diff: {}",
            ctx, a, b, diff
        );
    }
    diff
}

fn assert_slice_eq(a: &[f32], b: &[f32], tol: f32, ctx: &str) -> f32 {
    let mut max_diff = 0.0f32;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tol {
            panic!(
                "Validation Failed: {} at index {} | AVX2: {}, Scalar: {} | Diff: {}",
                ctx, i, x, y, diff
            );
        }
    }
    max_diff
}

fn main() {
    println!("=== AVX2 Math Validator ===");
    let n = 2048 + 17; // non-multiple of 8 to test tail loops

    let vec_a: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.123).sin()).collect();
    let vec_b: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.321).cos()).collect();

    // --- forge_autograd module tests ---
    println!("[forge_autograd tests]");
    std::io::stdout().flush().unwrap();

    let dot_avx = unsafe { dot_product_avx2(&vec_a, &vec_b) };
    let dot_scl = dot_product_scalar(&vec_a, &vec_b);
    let err = assert_f32_eq(dot_avx, dot_scl, 1e-3, "forge_autograd::dot_product_avx2");
    println!(
        "✓ forge_autograd::dot_product_avx2 validated. (Error: {:.2e})",
        err
    );

    let mut axpy_avx = vec_a.clone();
    let mut axpy_scl = vec_a.clone();
    let alpha = 0.54321;
    unsafe {
        axpy_avx2(&mut axpy_avx, alpha, &vec_b);
    }
    axpy_scalar(&mut axpy_scl, alpha, &vec_b);
    let err = assert_slice_eq(&axpy_avx, &axpy_scl, 1e-5, "forge_autograd::axpy_avx2");
    println!(
        "✓ forge_autograd::axpy_avx2 validated. (Max Error: {:.2e})",
        err
    );

    let mut silu_avx = vec![0.0; n];
    let mut silu_scl = vec![0.0; n];
    unsafe {
        silu_avx2(&vec_a, &mut silu_avx);
    }
    silu_scalar(&vec_a, &mut silu_scl);
    let err = assert_slice_eq(&silu_avx, &silu_scl, 1e-3, "forge_autograd::silu_avx2");
    println!(
        "✓ forge_autograd::silu_avx2 validated. (Max Error: {:.2e})",
        err
    );

    // --- forge_llm::asm module tests ---
    println!("\n[forge_llm::asm module tests]");

    // ASM kernels assume N is a multiple of 8 or 16
    let n_asm = n & !7;

    let asm_dot = unsafe { asm::dot_product_avx2(n_asm, vec_a.as_ptr(), vec_b.as_ptr()) };
    let dot_scl_asm = dot_product_scalar(&vec_a[..n_asm], &vec_b[..n_asm]);
    let err = assert_f32_eq(asm_dot, dot_scl_asm, 1e-3, "asm::dot_product_avx2");
    println!("✓ asm::dot_product_avx2 validated. (Error: {:.2e})", err);

    let asm_sum_sq = unsafe { asm::sum_squares_avx2(n_asm, vec_a.as_ptr()) };
    let scl_sum_sq_asm = sum_squares_scalar(&vec_a[..n_asm]);
    let err = assert_f32_eq(asm_sum_sq, scl_sum_sq_asm, 1e-3, "asm::sum_squares_avx2");
    println!("✓ asm::sum_squares_avx2 validated. (Error: {:.2e})", err);

    let mut asm_silu = vec![0.0; n];
    unsafe {
        asm::silu_vectorial_avx2(n_asm, vec_a.as_ptr(), asm_silu.as_mut_ptr());
    }
    let err = assert_slice_eq(
        &asm_silu[..n_asm],
        &silu_scl[..n_asm],
        2e-3, // rcpps + NR path: slightly looser than divps
        "asm::silu_vectorial_avx2",
    );
    println!(
        "✓ asm::silu_vectorial_avx2 validated. (Max Error: {:.2e})",
        err
    );

    // LM head logits (FMA dual-acc) — used by main.rs inference
    {
        let hidden = 64usize;
        let vocab = 48usize;
        let regs: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.017).sin()).collect();
        let weights: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32) * 0.011).cos())
            .collect();
        let mut out = vec![0.0f32; vocab];
        unsafe {
            asm::lm_head_logits_avx2(
                vocab,
                hidden,
                regs.as_ptr(),
                weights.as_ptr(),
                out.as_mut_ptr(),
            );
        }
        let mut max_err = 0.0f32;
        for v in 0..vocab {
            let mut exp = 0.0f32;
            for j in 0..hidden {
                exp += regs[j] * weights[v * hidden + j];
            }
            max_err = max_err.max((out[v] - exp).abs());
        }
        if max_err > 1e-3 {
            panic!("lm_head_logits_avx2 max err {max_err}");
        }
        println!(
            "✓ asm::lm_head_logits_avx2 validated. (Max Error: {:.2e})",
            max_err
        );

        let best = unsafe { asm::lm_head_avx2(vocab, hidden, regs.as_ptr(), weights.as_ptr()) };
        let mut best_s = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (v, &ov) in out.iter().enumerate() {
            if ov > best_v {
                best_v = ov;
                best_s = v;
            }
        }
        assert_eq!(best, best_s, "argmax vs full logits mismatch");
        println!("✓ asm::lm_head_avx2 argmax matches logits row {best}");
    }

    // Ternary GEMV (ELUT 4-bit × FP32) — core of PCorePool GEMV path
    {
        let n_gemv = 256usize;
        let x: Vec<f32> = (0..n_gemv).map(|i| (i as f32 * 0.01) - 1.0).collect();
        // pack all +1 → nibble 0x1 repeated
        let mut packed = vec![0u32; n_gemv / 8];
        for p in &mut packed {
            let mut w = 0u32;
            for i in 0..8 {
                w |= 0x1u32 << (i * 4);
            }
            *p = w;
        }
        let mut out = 0.0f32;
        let scale = 0.5f32;
        unsafe {
            asm::ternary_gemv_avx2(n_gemv, x.as_ptr(), packed.as_ptr(), &mut out, scale);
        }
        let expected: f32 = x.iter().sum::<f32>() * scale;
        let err = (out - expected).abs();
        if err > 1e-3 {
            panic!("ternary_gemv_avx2 err {err}: {out} vs {expected}");
        }
        println!(
            "✓ asm::ternary_gemv_avx2 (ELUT×FP32) validated. (Error: {:.2e})",
            err
        );
    }

    println!("\n--- Compute stack reminder ---");
    println!("  LIVE GEMV: ternary_gemv* via PCorePool(8) in slime_forward");
    println!("  LIVE LM head: lm_head_logits_avx2 in main.rs");
    println!("  LIVE QAT step: shape-dispatch optimizers (L-01) + STE pack");
    println!("  ash Vulkan: infrastructure; GEMV dispatch not on engine critical path");

    println!("\nALL AVX2 & ASM MATH VALIDATED SUCCESSFULLY.");
}
