use crate::asm::*;
use half::f16;
use rand::RngExt;

#[test]
fn test_basic_gemv_regression() {
    let n = 32;
    let x = vec![1.0f32; n];
    let mut out = vec![0.0f32; 1]; // Only needs one float for the reduced result
    let weights = [BlockQ4_0 {
        d: f16::from_f32(1.0),
        qs: [0x99; 16],
    }];

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

    assert!(
        (scale - expected).abs() < 1e-4,
        "rms_norm_scale: {} vs {}",
        scale,
        expected
    );
}

#[test]
fn test_sum_squares_avx2() {
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
    let result = unsafe { sum_squares_avx2(n, x.as_ptr()) };
    let expected: f32 = x.iter().map(|&v| v * v).sum();

    assert!(
        (result - expected).abs() < 1e-2,
        "sum_squares: {} vs {}",
        result,
        expected
    );
}

#[test]
fn test_dot_product_avx2() {
    let n = 128;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.7).collect();
    let result = unsafe { dot_product_avx2(n, a.as_ptr(), b.as_ptr()) };
    let expected: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();

    assert!(
        (result - expected).abs() < 1e-1,
        "dot_product: {} vs {}",
        result,
        expected
    );
}

/// Packs 16 ternary values (-1, 0, 1) into a u32 for testing.
fn pack_ternary_row(values: &[i8]) -> Vec<u32> {
    let n = values.len();
    let mut packed = vec![0u32; n.div_ceil(8)];
    for (i, &v) in values.iter().enumerate() {
        let bits = match v {
            1 => 1u32,
            -1 => 15u32,
            _ => 0u32,
        };
        packed[i / 8] |= bits << ((i % 8) * 4);
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
    unsafe {
        ternary_gemv_avx2(n, x.as_ptr(), packed_w.as_ptr(), &mut out_asm, scale);
    }

    let mut out_rust = 0.0f32;
    for i in 0..n {
        out_rust += x[i] * raw_weights[i] as f32;
    }
    out_rust *= scale;

    assert!(
        (out_rust - out_asm).abs() < 1e-4,
        "ternary_gemv delta: {} (rust) vs {} (asm), diff={}",
        out_rust,
        out_asm,
        (out_rust - out_asm).abs()
    );
}

#[test]
fn test_ternary_gemv_all_ones() {
    let n = 128;
    let x = vec![1.0f32; n];
    let raw = vec![1i8; n];
    let packed = pack_ternary_row(&raw);

    let mut out = 0.0f32;
    unsafe {
        ternary_gemv_avx2(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0);
    }

    assert!(
        (out - n as f32).abs() < 1e-4,
        "all ones: expected {}, got {}",
        n,
        out
    );
}

#[test]
fn test_ternary_gemv_all_neg_ones() {
    let n = 128;
    let x = vec![1.0f32; n];
    let raw = vec![-1i8; n];
    let packed = pack_ternary_row(&raw);

    let mut out = 0.0f32;
    unsafe {
        ternary_gemv_avx2(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0);
    }

    assert!(
        (out + n as f32).abs() < 1e-4,
        "all neg ones: expected {}, got {}",
        -(n as f32),
        out
    );
}

#[test]
fn test_ternary_gemv_all_zeros() {
    let n = 128;
    let x = vec![5.0f32; n];
    let raw = vec![0i8; n];
    let packed = pack_ternary_row(&raw);

    let mut out = 0.0f32;
    unsafe {
        ternary_gemv_avx2(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0);
    }

    assert!((out).abs() < 1e-6, "all zeros: expected 0, got {}", out);
}

#[test]
fn test_rms_norm_scale_constant_input() {
    let n = 64;
    let x = vec![3.0f32; n];
    let scale = unsafe { rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };

    let expected = 1.0 / ((9.0_f32) + 1e-6_f32).sqrt(); // mean_sq = 9.0
    assert!(
        (scale - expected).abs() < 1e-5,
        "rms constant: {} vs {}",
        scale,
        expected
    );
}

#[test]
fn test_rms_norm_scale_zero_input() {
    let n = 64;
    let x = vec![0.0f32; n];
    let scale = unsafe { rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };

    let expected = 1.0 / (1e-6_f32).sqrt(); // only eps remains
    assert!(
        (scale - expected).abs() < 1e-4,
        "rms zero: {} vs {}",
        scale,
        expected
    );
}

#[test]
fn test_sum_squares_edge_cases() {
    // Single element (kernel may process in batches, but should handle small sizes)
    let x = [4.0f32; 32];
    let result = unsafe { sum_squares_avx2(32, x.as_ptr()) };
    assert!(
        (result - 32.0 * 16.0).abs() < 1e-3,
        "sum_squares 32 elements: {}",
        result
    );

    // Large values (check relative error within f32 precision)
    let x = [1000.0f32; 64];
    let result = unsafe { sum_squares_avx2(64, x.as_ptr()) };
    let expected = 64.0 * 1_000_000.0;
    let rel_err = (result - expected).abs() / expected;
    assert!(
        rel_err < 1e-4,
        "sum_squares large: {}, expected {}, rel_err {}",
        result,
        expected,
        rel_err
    );
}

#[test]
fn test_ternary_gemv_4rows_avx2() {
    let n = 256;
    let mut rng = rand::rng();
    let x: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let scale = 0.5f32;
    let stride = n / 8;

    let mut raw_weights = Vec::new();
    let mut packed_weights = Vec::new();

    for _ in 0..4 {
        let row: Vec<i8> = (0..n).map(|_| rng.random_range(-1..=1)).collect();
        packed_weights.extend(pack_ternary_row(&row));
        raw_weights.push(row);
    }

    let mut out_asm = vec![0.0f32; 4];
    unsafe {
        ternary_gemv_4rows(
            n,
            x.as_ptr(),
            packed_weights.as_ptr(),
            out_asm.as_mut_ptr(),
            scale,
            stride,
        );
    }

    for i in 0..4 {
        let mut out_rust = 0.0f32;
        for j in 0..n {
            out_rust += x[j] * raw_weights[i][j] as f32;
        }
        out_rust *= scale;

        assert!(
            (out_rust - out_asm[i]).abs() < 1e-4,
            "Row {} mismatch: rust {} vs asm {}",
            i,
            out_rust,
            out_asm[i]
        );
    }
}

#[test]
fn test_ternary_gemv_8rows_avx2() {
    let n = 256;
    let mut rng = rand::rng();
    let x: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let scale = 0.5f32;
    let stride = n / 8;

    let mut raw_weights = Vec::new();
    let mut packed_weights = Vec::new();

    for _ in 0..8 {
        let row: Vec<i8> = (0..n).map(|_| rng.random_range(-1..=1)).collect();
        packed_weights.extend(pack_ternary_row(&row));
        raw_weights.push(row);
    }

    let mut out_asm = vec![0.0f32; 8];
    unsafe {
        ternary_gemv_8rows(
            n,
            x.as_ptr(),
            packed_weights.as_ptr(),
            out_asm.as_mut_ptr(),
            scale,
            stride,
        );
    }

    for i in 0..8 {
        let mut out_rust = 0.0f32;
        for j in 0..n {
            out_rust += x[j] * raw_weights[i][j] as f32;
        }
        out_rust *= scale;

        assert!(
            (out_rust - out_asm[i]).abs() < 1e-4,
            "8row Row {} mismatch: rust {} vs asm {}",
            i,
            out_rust,
            out_asm[i]
        );
    }
}

#[test]
fn test_ternary_gemv_8rows_matches_4rows() {
    // Same matrix, 8-row vs two×4-row must agree (scale=1).
    let n: usize = 320; // not multiple of 64 — exercise 16+8 tails
    let mut rng = rand::rng();
    let x: Vec<f32> = (0..n).map(|_| rng.random_range(-2.0..2.0)).collect();
    let stride = n.div_ceil(8);
    let mut packed = Vec::new();
    for _ in 0..8 {
        let row: Vec<i8> = (0..n).map(|_| rng.random_range(-1..=1)).collect();
        let mut p = pack_ternary_row(&row);
        p.resize(stride, 0);
        packed.extend(p);
    }
    let mut out8 = vec![0.0f32; 8];
    let mut out4 = vec![0.0f32; 8];
    unsafe {
        ternary_gemv_8rows(
            n,
            x.as_ptr(),
            packed.as_ptr(),
            out8.as_mut_ptr(),
            1.0,
            stride,
        );
        ternary_gemv_4rows(
            n,
            x.as_ptr(),
            packed.as_ptr(),
            out4.as_mut_ptr(),
            1.0,
            stride,
        );
        ternary_gemv_4rows(
            n,
            x.as_ptr(),
            packed.as_ptr().add(4 * stride),
            out4.as_mut_ptr().add(4),
            1.0,
            stride,
        );
    }
    for i in 0..8 {
        assert!(
            (out8[i] - out4[i]).abs() < 1e-4,
            "row {i}: 8rows={} 4rows={}",
            out8[i],
            out4[i]
        );
    }
}

#[test]
fn bench_ternary_gemv_comparison() {
    let n = 2048;
    let x = vec![1.0f32; n];
    let stride = n / 8;
    let scale = 1.0f32;
    const N_ROWS: usize = 1024;

    let mut packed_weights = Vec::new();
    for _ in 0..N_ROWS {
        packed_weights.extend(vec![0x55555555u32; stride]);
    }

    let mut out = vec![0.0f32; N_ROWS];

    const REPS: usize = 32;
    // Warmup (bring turbo + L2 to steady state)
    for _ in 0..4 {
        for i in (0..N_ROWS).step_by(8) {
            unsafe {
                ternary_gemv_8rows(
                    n,
                    x.as_ptr(),
                    packed_weights.as_ptr().add(i * stride),
                    out.as_mut_ptr().add(i),
                    scale,
                    stride,
                );
            }
        }
    }

    let start_single = std::time::Instant::now();
    for _ in 0..REPS {
        for (i, out_val) in out.iter_mut().enumerate().take(N_ROWS) {
            unsafe {
                ternary_gemv(
                    n,
                    x.as_ptr(),
                    packed_weights.as_ptr().add(i * stride),
                    out_val,
                    scale,
                );
            }
        }
    }
    let duration_single = start_single.elapsed();
    println!("Single-row GEMV ({N_ROWS}×{REPS}): {duration_single:?}");

    let start_4 = std::time::Instant::now();
    for _ in 0..REPS {
        for i in (0..N_ROWS).step_by(4) {
            unsafe {
                ternary_gemv_4rows(
                    n,
                    x.as_ptr(),
                    packed_weights.as_ptr().add(i * stride),
                    out.as_mut_ptr().add(i),
                    scale,
                    stride,
                );
            }
        }
    }
    let duration_4 = start_4.elapsed();
    println!("4-row GEMV ({N_ROWS}×{REPS}): {duration_4:?}");

    let start_8 = std::time::Instant::now();
    for _ in 0..REPS {
        for i in (0..N_ROWS).step_by(8) {
            unsafe {
                ternary_gemv_8rows(
                    n,
                    x.as_ptr(),
                    packed_weights.as_ptr().add(i * stride),
                    out.as_mut_ptr().add(i),
                    scale,
                    stride,
                );
            }
        }
    }
    let duration_8 = start_8.elapsed();
    println!("8-row GEMV ({N_ROWS}×{REPS}): {duration_8:?}");

    let sp4 = duration_single.as_secs_f64() / duration_4.as_secs_f64();
    let sp8 = duration_single.as_secs_f64() / duration_8.as_secs_f64();
    let sp8v4 = duration_4.as_secs_f64() / duration_8.as_secs_f64();
    println!("Speedup 4r/1r={sp4:.2}x  8r/1r={sp8:.2}x  8r/4r={sp8v4:.2}x");
}

#[test]
#[ignore]
fn test_hadamard_transform_avx2() {
    let n = 1024;
    let mut rng = rand::rng();
    let x_asm: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut x_rust = x_asm.clone();

    // Reference FWHT
    fn fwht_ref(x: &mut [f32]) {
        let n = x.len();
        let mut s = 1;
        while s < n {
            for i in (0..n).step_by(s * 2) {
                for j in 0..s {
                    let a = x[i + j];
                    let b = x[i + j + s];
                    x[i + j] = a + b;
                    x[i + j + s] = a - b;
                }
            }
            s *= 2;
        }
    }

    fwht_ref(&mut x_rust);
    // {
    //     // hadamard_transform(n, x_asm.as_mut_ptr());
    // }

    for i in 0..n {
        assert!(
            (x_rust[i] - x_asm[i]).abs() < 1e-4,
            "Hadamard mismatch at {}: rust {} vs asm {}",
            i,
            x_rust[i],
            x_asm[i]
        );
    }
}

#[test]
fn test_ternary_gemm_batch4_avx2() {
    let out_dim = 8;
    let in_dim = 64;
    let mut rng = rand::rng();

    // 4 tokens of x, each of length in_dim
    let x_tokens: Vec<Vec<f32>> = (0..4)
        .map(|_| (0..in_dim).map(|_| rng.random_range(-2.0..2.0)).collect())
        .collect();
    let x_flat: Vec<f32> = x_tokens.iter().flat_map(|t| t.iter()).copied().collect();

    // out_dim rows of ternary weights, each of length in_dim
    let raw_rows: Vec<Vec<i8>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| rng.random_range(-1..=1)).collect())
        .collect();
    let packed: Vec<u32> = raw_rows.iter().flat_map(|r| pack_ternary_row(r)).collect();

    // Per-row scales
    let scales: Vec<f32> = (0..out_dim).map(|_| rng.random_range(0.5..2.0)).collect();

    let mut out_asm = vec![0.0f32; 4 * out_dim];
    unsafe {
        ternary_gemm_batch4_avx2(
            out_dim,
            in_dim,
            x_flat.as_ptr(),
            packed.as_ptr(),
            out_asm.as_mut_ptr(),
            scales.as_ptr(),
        );
    }

    // Reference: for each token and each row, compute dot product
    for token in 0..4 {
        for row in 0..out_dim {
            let mut expected = 0.0f32;
            for i in 0..in_dim {
                expected += x_tokens[token][i] * raw_rows[row][i] as f32;
            }
            expected *= scales[row];
            let got = out_asm[token * out_dim + row];
            assert!(
                (expected - got).abs() < 1e-4,
                "Token {} row {}: expected {} got {} diff {}",
                token,
                row,
                expected,
                got,
                (expected - got).abs()
            );
        }
    }
}

#[test]
fn test_ternary_gemm_batch4_known_pattern() {
    for &(out_dim, in_dim) in &[(1, 16), (1, 32), (2, 16), (2, 32)] {
        let scale = 1.0f32;

        let x_data: Vec<f32> = (0..4 * in_dim).map(|i| (i as f32) + 1.0).collect();
        let raw: Vec<i8> = vec![1i8; in_dim];
        let packed = pack_ternary_row(&raw);
        let packed_rows: Vec<u32> = packed
            .iter()
            .copied()
            .cycle()
            .take(out_dim * packed.len())
            .collect();
        let scales = vec![scale; out_dim];

        let mut out = vec![0.0f32; 4 * out_dim];
        unsafe {
            ternary_gemm_batch4_avx2(
                out_dim,
                in_dim,
                x_data.as_ptr(),
                packed_rows.as_ptr(),
                out.as_mut_ptr(),
                scales.as_ptr(),
            );
        }

        for token in 0..4 {
            let x_start = token * in_dim;
            let sum_x: f32 = (x_start..x_start + in_dim).map(|i| x_data[i]).sum();
            let expected = sum_x * scale;
            for row in 0..out_dim {
                let got = out[token * out_dim + row];
                assert!(
                    (expected - got).abs() < 1e-4,
                    "[out_dim={} in_dim={}] Token {} row {}: expected {} got {}",
                    out_dim,
                    in_dim,
                    token,
                    row,
                    expected,
                    got,
                );
            }
        }
    }
}

#[test]
fn test_lm_head_avx2_argmax() {
    let hidden = 64usize;
    let vocab = 32usize;
    let regs: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.01).collect();
    // weights[v * hidden + j]; make row 7 the best match (identical to regs)
    let mut weights = vec![0.0f32; vocab * hidden];
    for v in 0..vocab {
        for j in 0..hidden {
            weights[v * hidden + j] = if v == 7 { regs[j] } else { (v as f32) * 0.001 };
        }
    }
    let best = unsafe { lm_head_avx2(vocab, hidden, regs.as_ptr(), weights.as_ptr()) };
    assert_eq!(best, 7, "lm_head argmax expected row 7, got {best}");
}

#[test]
fn test_lm_head_logits_avx2_vs_scalar() {
    let hidden = 48usize;
    let vocab = 16usize;
    let mut rng = rand::rng();
    let regs: Vec<f32> = (0..hidden).map(|_| rng.random_range(-1.0..1.0)).collect();
    let weights: Vec<f32> = (0..vocab * hidden)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let mut out = vec![0.0f32; vocab];
    unsafe {
        lm_head_logits_avx2(
            vocab,
            hidden,
            regs.as_ptr(),
            weights.as_ptr(),
            out.as_mut_ptr(),
        );
    }
    for v in 0..vocab {
        let mut expected = 0.0f32;
        for j in 0..hidden {
            expected += regs[j] * weights[v * hidden + j];
        }
        assert!(
            (out[v] - expected).abs() < 1e-3,
            "logit[{v}]: asm {} vs scalar {}, delta {}",
            out[v],
            expected,
            (out[v] - expected).abs()
        );
    }
}

#[test]
fn test_silu_vectorial_smoke() {
    let n = 32usize;
    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.5).collect();
    let mut dst = vec![0.0f32; n];
    unsafe {
        silu_vectorial_avx2(n, src.as_ptr(), dst.as_mut_ptr());
    }
    for i in 0..n {
        let x = src[i];
        let expected = x / (1.0 + (-x).exp());
        let err = (dst[i] - expected).abs();
        assert!(
            err < 2e-3,
            "silu[{i}]: got {} expected {} err {err}",
            dst[i],
            expected
        );
    }
}

#[test]
fn test_sgemm_abt_avx2() {
    let m = 4;
    let n = 4;
    let k = 128;
    let mut rng = rand::rng();

    let a: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut c = vec![0.0f32; m * n];

    unsafe {
        sgemm_abt(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    }

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p];
            }
            expected[i * n + j] = sum;
        }
    }

    for i in 0..(m * n) {
        assert!(
            (c[i] - expected[i]).abs() < 1e-3,
            "sgemm_abt mismatch at {}: expected {} got {}",
            i,
            expected[i],
            c[i]
        );
    }
}

#[test]
#[ignore]
fn test_sgemm_avx2() {
    let m = 4;
    let n = 128;
    let k = 4;
    let mut rng = rand::rng();

    let a: Vec<f32> = (0..m * k).map(|_| rng.random_range(-1.0..1.0)).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let c = vec![0.0f32; m * n];

    // {
    //     // sgemm(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr());
    // }

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            expected[i * n + j] = sum;
        }
    }

    for i in 0..(m * n) {
        assert!(
            (c[i] - expected[i]).abs() < 1e-3,
            "sgemm mismatch at {}: expected {} got {}",
            i,
            expected[i],
            c[i]
        );
    }
}
