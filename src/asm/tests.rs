use half::f16;
use rand::RngExt;
use crate::asm::*;

#[test]
fn test_basic_gemv_regression() {
    let n = 32;
    let x = vec![1.0f32; n];
    let mut out = vec![0.0f32; 1]; // Only needs one float for the reduced result
    let weights = [BlockQ4_0 { d: f16::from_f32(1.0), qs: [0x99; 16] }];
    
    // The kernel adds to *out, so it must be initialized to 0.0
    unsafe {
        q4_0_gemv_asm(n, x.as_ptr(), weights.as_ptr(), out.as_mut_ptr());
    }
    
    println!("Basic GEMV Result: {:?}", out[0]);
    // 0x99 -> both nibbles are 9. 
    // 9 - 8 = 1.0 real value.
    // 32 elements * 1.0 * 1.0 = 32.0
    assert_eq!(out[0], 32.0);
}

#[test]
fn test_rms_norm_scale() {
    let n = 64;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let scale = unsafe { rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };
    
    let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
    let expected = 1.0 / ((sum_sq / n as f32) + 1e-6).sqrt();
    
    assert!((scale - expected).abs() < 1e-4, "rms_norm_scale: {} vs {}", scale, expected);
}

#[test]
fn test_sum_squares_avx2() {
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let result = unsafe { sum_squares_avx2(n, x.as_ptr()) };
    let expected: f32 = x.iter().map(|&v| v * v).sum();
    
    assert!((result - expected).abs() < 1e-2, "sum_squares: {} vs {}", result, expected);
}

#[test]
fn test_dot_product_avx2() {
    let n = 128;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7).collect();
    let result = unsafe { dot_product_avx2(n, a.as_ptr(), b.as_ptr()) };
    let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    
    assert!((result - expected).abs() < 1e-1, "dot_product: {} vs {}", result, expected);
}

/// Packs 16 ternary values (-1, 0, 1) into a u32 for testing.
fn pack_ternary_row(values: &[i8]) -> Vec<u32> {
    let n = values.len();
    let mut packed = vec![0u32; n.div_ceil(16)];
    for (i, &v) in values.iter().enumerate() {
        let bits = match v { 1 => 1u32, -1 => 2u32, _ => 0u32 };
        packed[i / 16] |= bits << ((i % 16) * 2);
    }
    packed
}

#[test]
fn test_ternary_gemv_avx2_vs_reference() {
    let n = 256;
    let mut rng = rand::rng();
    let x: Vec<f32> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();

    let mut raw_weights = Vec::with_capacity(n);
    for _ in 0..n {
        let w: i8 = rng.random_range(-1..=1);
        raw_weights.push(w);
    }
    let packed_w = pack_ternary_row(&raw_weights);

    let scale = 0.5f32;
    let mut out_asm = 0.0f32;
    unsafe { ternary_gemv_avx2(n, x.as_ptr(), packed_w.as_ptr(), &mut out_asm, scale); }

    let mut out_rust = 0.0f32;
    for i in 0..n {
        out_rust += x[i] * raw_weights[i] as f32;
    }
    out_rust *= scale;

    assert!(
        (out_rust - out_asm).abs() < 1e-4,
        "ternary_gemv delta: {} (rust) vs {} (asm), diff={}",
        out_rust, out_asm, (out_rust - out_asm).abs()
    );
}

#[test]
fn test_ternary_gemv_all_ones() {
    let n = 128;
    let x = vec![1.0f32; n];
    let raw = vec![1i8; n];
    let packed = pack_ternary_row(&raw);

    let mut out = 0.0f32;
    unsafe { ternary_gemv_avx2(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0); }

    assert!((out - n as f32).abs() < 1e-4, "all ones: expected {}, got {}", n, out);
}

#[test]
fn test_ternary_gemv_all_neg_ones() {
    let n = 128;
    let x = vec![1.0f32; n];
    let raw = vec![-1i8; n];
    let packed = pack_ternary_row(&raw);

    let mut out = 0.0f32;
    unsafe { ternary_gemv_avx2(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0); }

    assert!((out + n as f32).abs() < 1e-4, "all neg ones: expected {}, got {}", -(n as f32), out);
}

#[test]
fn test_ternary_gemv_all_zeros() {
    let n = 128;
    let x = vec![5.0f32; n];
    let raw = vec![0i8; n];
    let packed = pack_ternary_row(&raw);

    let mut out = 0.0f32;
    unsafe { ternary_gemv_avx2(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0); }

    assert!((out).abs() < 1e-6, "all zeros: expected 0, got {}", out);
}

#[test]
fn test_rms_norm_scale_constant_input() {
    let n = 64;
    let x = vec![3.0f32; n];
    let scale = unsafe { rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };

    let expected = 1.0 / ((9.0_f32) + 1e-6_f32).sqrt(); // mean_sq = 9.0
    assert!((scale - expected).abs() < 1e-5, "rms constant: {} vs {}", scale, expected);
}

#[test]
fn test_rms_norm_scale_zero_input() {
    let n = 64;
    let x = vec![0.0f32; n];
    let scale = unsafe { rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };

    let expected = 1.0 / (1e-6_f32).sqrt(); // only eps remains
    assert!((scale - expected).abs() < 1e-4, "rms zero: {} vs {}", scale, expected);
}

#[test]
fn test_sum_squares_edge_cases() {
    // Single element (kernel may process in batches, but should handle small sizes)
    let x = vec![4.0f32; 32];
    let result = unsafe { sum_squares_avx2(32, x.as_ptr()) };
    assert!((result - 32.0 * 16.0).abs() < 1e-3, "sum_squares 32 elements: {}", result);

    // Large values (check relative error within f32 precision)
    let x = vec![1000.0f32; 64];
    let result = unsafe { sum_squares_avx2(64, x.as_ptr()) };
    let expected = 64.0 * 1_000_000.0;
    let rel_err = (result - expected).abs() / expected;
    assert!(rel_err < 1e-4, "sum_squares large: {}, expected {}, rel_err {}", result, expected, rel_err);
}

#[test]
fn test_ternary_gemv_4rows_avx2() {
    let n = 256;
    let mut rng = rand::rng();
    let x: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let scale = 0.5f32;
    let stride = n / 16;
    
    let mut raw_weights = Vec::new();
    let mut packed_weights = Vec::new();
    
    for _ in 0..4 {
        let row: Vec<i8> = (0..n).map(|_| rng.random_range(-1..=1)).collect();
        packed_weights.extend(pack_ternary_row(&row));
        raw_weights.push(row);
    }

    let mut out_asm = vec![0.0f32; 4];
    unsafe {
        ternary_gemv_4rows_avx2(n, x.as_ptr(), packed_weights.as_ptr(), out_asm.as_mut_ptr(), scale, stride);
    }

    for i in 0..4 {
        let mut out_rust = 0.0f32;
        for j in 0..n {
            out_rust += x[j] * raw_weights[i][j] as f32;
        }
        out_rust *= scale;
        
        assert!((out_rust - out_asm[i]).abs() < 1e-4, "Row {} mismatch: rust {} vs asm {}", i, out_rust, out_asm[i]);
    }
}

#[test]
fn bench_ternary_gemv_comparison() {
    let n = 2048;
    let x = vec![1.0f32; n];
    let stride = n / 16;
    let scale = 1.0f32;
    
    let mut packed_weights = Vec::new();
    for _ in 0..1024 {
        packed_weights.extend(vec![0x55555555u32; stride]);
    }

    let mut out = vec![0.0f32; 1024];

    // Benchmark single-row
    let start_single = std::time::Instant::now();
    for i in 0..1024 {
        unsafe {
            ternary_gemv_avx2(n, x.as_ptr(), packed_weights.as_ptr().add(i * stride), &mut out[i], scale);
        }
    }
    let duration_single = start_single.elapsed();
    println!("Single-row GEMV (1024 rows): {:?}", duration_single);

    // Benchmark 4-rows
    out.fill(0.0);
    let start_multi = std::time::Instant::now();
    for i in (0..1024).step_by(4) {
        unsafe {
            ternary_gemv_4rows_avx2(n, x.as_ptr(), packed_weights.as_ptr().add(i * stride), out.as_mut_ptr().add(i), scale, stride);
        }
    }
    let duration_multi = start_multi.elapsed();
    println!("Multi-row GEMV (1024 rows): {:?}", duration_multi);
    
    let speedup = duration_single.as_secs_f64() / duration_multi.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);
}
