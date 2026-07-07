use crate::asm;
use crate::mud::routing::MudRouter;
use crate::mud::{dequantize_ternary_row, MudFile, MudTensorType};

// ============================================================
// MUD FORMAT ROUNDTRIP
// ============================================================

#[test]
fn test_mud_save_load_roundtrip() {
    let path = "/tmp/test_mud_roundtrip.mud";
    let _ = std::fs::remove_file(path);

    let data: Vec<u8> = (0..128).map(|i| if i < 64 { 0x11 } else { 0x22 }).collect();
    let tensor = crate::mud::MudTensor {
        name: "test.weight".into(),
        t_type: MudTensorType::Ternary2Bit,
        shape: vec![16, 16],
        data_ptr: data.as_ptr(),
        offset: 0,
        mmap: None,
        owned_data: Some(data.clone()),
    };

    let mut tensors = std::collections::HashMap::new();
    tensors.insert("test.weight".into(), tensor);

    let skill = crate::mud::MudSkill {
        name: "core".into(),
        tensors,
        metadata: std::collections::HashMap::new(),
    };

    let mut skills = std::collections::HashMap::new();
    skills.insert("core".into(), skill);

    let file = MudFile {
        mmap: None,
        skills,
        global_metadata: {
            let mut m = std::collections::HashMap::new();
            m.insert("hidden_size".into(), "512".into());
            m.insert("num_layers".into(), "2".into());
            m.insert("num_experts".into(), "8".into());
            m.insert("ffn_hidden".into(), "2048".into());
            m.insert("num_heads".into(), "8".into());
            m.insert("max_seq_len".into(), "4096".into());
            m
        },
    };

    file.save(path).expect("save should succeed");
    let loaded = MudFile::load(path).expect("load should succeed");

    let core = loaded.skills.get("core").expect("core skill");
    let t = core.tensors.get("test.weight").expect("tensor");
    assert_eq!(t.name, "test.weight");
    assert_eq!(t.t_type, MudTensorType::Ternary2Bit);
    assert_eq!(t.shape, vec![16, 16]);

    let _ = std::fs::remove_file(path);
}

// ============================================================
// DEQUANTIZE TERNARY ROW
// ============================================================

#[test]
fn test_dequantize_ternary_row_all_zeros() {
    let packed: Vec<u32> = vec![0u32; 4];
    let mut out = [0.0f32; 32];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut out, 32);
    }
    for v in out.iter() {
        assert_eq!(*v, 0.0, "all zeros failed");
    }
}

#[test]
fn test_dequantize_ternary_row_all_ones() {
    let packed: Vec<u32> = vec![0x1111_1111u32; 4];
    let mut out = [0.0f32; 32];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut out, 32);
    }
    for v in out.iter() {
        assert_eq!(*v, 1.0, "all ones failed");
    }
}

#[test]
fn test_dequantize_ternary_row_all_neg_ones() {
    let packed: Vec<u32> = vec![0xFFFF_FFFFu32; 4];
    let mut out = [0.0f32; 32];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut out, 32);
    }
    for v in out.iter() {
        assert_eq!(*v, -1.0, "all neg ones failed");
    }
}

#[test]
fn test_dequantize_ternary_row_mixed() {
    // 0x01F0: LSB first: bits[0..3]=0 (0->0), bits[4..7]=F (F->-1),
    //                bits[8..11]=1 (1->+1), bits[12..15]=0 (0->0)
    let packed = [0x01F0u32];
    let mut out = [0.0f32; 4];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut out, 4);
    }
    assert_eq!(out[0], 0.0, "bits 0-3=0 -> 0");
    assert_eq!(out[1], -1.0, "bits 4-7=F -> -1");
    assert_eq!(out[2], 1.0, "bits 8-11=1 -> +1");
    assert_eq!(out[3], 0.0, "bits 12-15=0 -> 0");
}

#[test]
fn test_dequantize_ternary_row_remainder() {
    let packed = [0x1111_1111u32, 0u32];
    let mut out = [0.0f32; 12];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut out, 12);
    }
    for (i, val) in out.iter().enumerate().take(8) {
        assert_eq!(*val, 1.0, "remainder full block idx {i}");
    }
    for (i, val) in out.iter().enumerate().skip(8) {
        assert_eq!(*val, 0.0, "remainder out of range idx {i}");
    }
}

// ============================================================
// TERNARY GEMV: ASM vs SCALAR REFERENCE
// ============================================================

fn scalar_ternary_gemv(x: &[f32], weights: &[u32], scale: f32, n: usize) -> f32 {
    let mut sum = 0.0f32;
    let u32_count = n / 16;
    let remainder = n % 16;

    for i in 0..u32_count {
        let val = weights[i];
        for j in 0..16 {
            let bits = (val >> (j * 2)) & 3;
            let w = match bits {
                1 => 1.0,
                2 => -1.0,
                _ => 0.0,
            };
            sum += w * x[i * 16 + j];
        }
    }

    if remainder > 0 {
        let val = weights[u32_count];
        for j in 0..remainder {
            let bits = (val >> (j * 2)) & 3;
            let w = match bits {
                1 => 1.0,
                2 => -1.0,
                _ => 0.0,
            };
            sum += w * x[u32_count * 16 + j];
        }
    }

    sum * scale
}

fn make_ternary_packed(values: &[i8]) -> Vec<u32> {
    let n = values.len();
    let u32_count = n.div_ceil(16);
    let mut packed = vec![0u32; u32_count];
    for i in 0..n {
        let code = match values[i] {
            1 => 1u32,
            -1 => 2u32,
            _ => 0u32,
        };
        packed[i / 16] |= code << ((i % 16) * 2);
    }
    packed
}

#[test]
fn test_ternary_gemv_asm_vs_scalar_random() {
    let n = 512;
    let mut x = vec![0.0f32; n];
    for (i, val) in x.iter_mut().enumerate() {
        *val = (i as f32 * 0.1).sin() * 3.0;
    }
    let mut ternary_vals = vec![0i8; n];
    for (i, val) in ternary_vals.iter_mut().enumerate() {
        *val = match i % 3 {
            0 => 1,
            1 => -1,
            _ => 0,
        };
    }
    let packed = make_ternary_packed(&ternary_vals);
    let scale = 0.5;

    let mut asm_out = 0.0f32;
    unsafe {
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut asm_out, scale);
    }
    let ref_out = scalar_ternary_gemv(&x, &packed, scale, n);

    let rel_err = (asm_out - ref_out).abs() / ref_out.abs().max(1e-10);
    assert!(
        rel_err < 1e-4,
        "ASM vs scalar GEMV relative error: {rel_err:.2e} (asm={asm_out}, ref={ref_out})"
    );
}

#[test]
fn test_ternary_gemv_asm_accumulate() {
    let n = 256;
    let x = vec![1.0f32; n];
    let packed = make_ternary_packed(&vec![1i8; n]);
    let scale = 1.0;

    let mut out = 42.0f32;
    unsafe {
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut out, scale);
    }
    let expected = n as f32 * scale;
    assert_eq!(out, expected, "ASM should overwrite existing value");
}

#[test]
fn test_ternary_gemv_4rows_consistency() {
    let n = 256;
    let x = vec![0.5f32; n];

    let mut single_results = [0.0f32; 4];
    for (row, res) in single_results.iter_mut().enumerate() {
        let packed = make_ternary_packed(&vec![1i8 - (row as i8 * 2).signum(); n]);
        unsafe {
            asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), res, 0.5);
        }
    }

    let mut quad_results = [0.0f32; 4];
    let stride = n / 16;
    let all_weights: Vec<u32> = (0..4)
        .flat_map(|row| make_ternary_packed(&vec![1i8 - (row as i8 * 2).signum(); n]))
        .collect();
    unsafe {
        asm::ternary_gemv_4rows(
            n,
            x.as_ptr(),
            all_weights.as_ptr(),
            quad_results.as_mut_ptr(),
            0.5,
            stride,
        );
    }

    for row in 0..4 {
        let rel_err =
            (quad_results[row] - single_results[row]).abs() / single_results[row].abs().max(1e-10);
        assert!(
            rel_err < 1e-4,
            "4rows row {row} mismatch: quad={} single={}",
            quad_results[row],
            single_results[row]
        );
    }
}

// ============================================================
// T-SAR QUANTIZATION ACCURACY
// ============================================================

#[test]
fn test_tsar_quantization_roundtrip() {
    let n = 512;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin() * 2.0).collect();

    let absmax = x.iter().map(|v| v.abs()).fold(1e-8f32, f32::max);
    let q_scale = 127.0 / absmax;
    let mut x_q = vec![0i8; n];
    for i in 0..n {
        x_q[i] = (x[i] * q_scale).round().clamp(-127.0, 127.0) as i8;
    }
    let inv_q_scale = absmax / 127.0;

    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f32;
    for i in 0..n {
        let reconstructed = x_q[i] as f32 * inv_q_scale;
        let err = (reconstructed - x[i]).abs();
        max_err = max_err.max(err);
        sum_sq_err += err * err;
    }
    let rmse = (sum_sq_err / n as f32).sqrt();

    assert!(
        max_err < inv_q_scale * 1.5,
        "T-SAR max quantization error too high: {max_err} > {}",
        inv_q_scale * 1.5
    );
    assert!(rmse < inv_q_scale * 0.5, "T-SAR RMSE too high: {rmse}");
}

#[test]
fn test_tsar_symmetry() {
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.5).collect();

    let absmax = x.iter().map(|v| v.abs()).fold(1e-8f32, f32::max);
    let q_scale = 127.0 / absmax;
    let mut x_q = vec![0i8; n];
    for i in 0..n {
        x_q[i] = (x[i] * q_scale).round().clamp(-127.0, 127.0) as i8;
    }

    for i in 0..n {
        let reconstructed = x_q[i] as f32 * (absmax / 127.0);
        let rel_err = (reconstructed - x[i]).abs() / x[i].abs().max(1e-10);
        if x[i].abs() > 1.0 {
            assert!(
                rel_err < 0.1,
                "T-SAR symmetry error at {i}: x={}, reconstructed={}, rel_err={rel_err:.4}",
                x[i],
                reconstructed
            );
        }
    }
}

#[test]
fn test_tsar_edge_cases() {
    let test_cases = [
        ("all zeros", vec![0.0f32; 64]),
        ("constant", vec![std::f32::consts::PI; 64]),
        ("small values", vec![1e-6f32; 64]),
        ("large values", vec![1e6f32; 64]),
        ("mixed signs", {
            let mut v = vec![0.0f32; 64];
            for (i, val) in v.iter_mut().enumerate() {
                *val = if i % 2 == 0 { 10.0 } else { -10.0 };
            }
            v
        }),
    ];

    for (name, x) in &test_cases {
        let absmax = x.iter().map(|v| v.abs()).fold(1e-8f32, f32::max);
        let q_scale = 127.0 / absmax;
        let mut x_q = vec![0i8; x.len()];
        for (i, val) in x_q.iter_mut().enumerate() {
            *val = (x[i] * q_scale).round().clamp(-127.0, 127.0) as i8;
        }

        for &v in &x_q {
            assert!(v >= -127, "T-SAR {name}: value {v} out of [-127, 127]");
        }
    }
}

// ============================================================
// PEXT UNPACK CORRECTNESS
// ============================================================

#[test]
fn test_pext_unpack_all_patterns() {
    let mut expected_vals = [0i8; 32];
    let mut packed: u64 = 0;
    for (i, val) in expected_vals.iter_mut().enumerate() {
        let bits = match i % 4 {
            0 => 0u64, // 00 -> 0
            1 => 1u64, // 01 -> +1
            2 => 2u64, // 10 -> -1
            _ => 3u64, // 11 -> 0
        };
        packed |= bits << (i * 2);
        *val = match bits {
            1 => 1i8,
            2 => -1i8,
            _ => 0i8,
        };
    }

    let mut out = [0i8; 32];
    eprintln!("DEBUG packed=0x{packed:016x}");
    for (i, val) in expected_vals.iter().enumerate().take(4) {
        let bits = (packed >> (i * 2)) & 3;
        let low = (packed >> (i * 2)) & 1;
        let high = (packed >> (i * 2 + 1)) & 1;
        eprintln!("DEBUG i={i} bits={bits} low={low} high={high} expected={val}");
    }
    unsafe {
        asm::pext_unpack_ternary(packed, out.as_mut_ptr());
    }

    for (i, val) in out.iter().enumerate() {
        assert_eq!(
            *val, expected_vals[i],
            "PEXT unpack idx {i}: expected={}, got={}",
            expected_vals[i], *val
        );
    }
}

#[test]
fn test_pext_unpack_all_ones() {
    let packed: u64 = 0x5555_5555_5555_5555u64;
    let mut out = [0i8; 32];
    unsafe {
        asm::pext_unpack_ternary(packed, out.as_mut_ptr());
    }
    eprintln!("PEXT all-ones out = {:?}", &out[..32]);
    let mut failures = Vec::new();
    for (i, &val) in out.iter().enumerate() {
        if val != 1 {
            failures.push((i, val));
        }
    }
    if !failures.is_empty() {
        eprintln!("PEXT all-ones failures: {failures:?}");
    }
    assert!(
        failures.is_empty(),
        "PEXT all ones had {} failures",
        failures.len()
    );
}

#[test]
fn test_pext_unpack_all_zeros() {
    let packed: u64 = 0u64;
    let mut out = [0i8; 32];
    unsafe {
        asm::pext_unpack_ternary(packed, out.as_mut_ptr());
    }
    for (i, val) in out.iter().enumerate() {
        assert_eq!(*val, 0, "PEXT all zeros idx {i}");
    }
}

// ============================================================
// RMSNORM CORRECTNESS
// ============================================================

#[test]
fn test_rms_norm_scale_known_values() {
    let n = 128;
    let x: Vec<f32> = vec![1.0f32; n];
    let eps = 1e-6;

    let scale = unsafe { asm::rms_norm_scale_asm(n, x.as_ptr(), eps) };

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let expected = 1.0 / ((sum_sq / n as f32) + eps).sqrt();

    let rel_err = (scale - expected).abs() / expected;
    assert!(
        rel_err < 1e-5,
        "RMSNorm scale error: {scale} vs {expected} (rel_err={rel_err:.2e})"
    );
}

#[test]
fn test_rms_norm_numerical_stability() {
    let n = 256;
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("zeros", vec![0.0f32; n]),
        ("tiny", vec![1e-20f32; n]),
        ("mixed_magnitude", {
            let mut v = vec![1.0f32; n];
            v[0] = 1e10;
            v[1] = -1e10;
            v
        }),
    ];

    for (name, x) in &cases {
        let scale = unsafe { asm::rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };
        assert!(
            scale.is_finite(),
            "RMSNorm {name}: scale not finite ({scale})"
        );
        assert!(
            scale > 0.0,
            "RMSNorm {name}: scale should be positive ({scale})"
        );
    }
}

// ============================================================
// DOT PRODUCT CORRECTNESS
// ============================================================
#[test]
fn test_dot_product_avx2_known() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b: Vec<f32> = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let expected: f32 = 1.0 * 8.0
        + 2.0 * 7.0
        + 3.0 * 6.0
        + 4.0 * 5.0
        + 5.0 * 4.0
        + 6.0 * 3.0
        + 7.0 * 2.0
        + 8.0 * 1.0;
    let result = unsafe { asm::dot_product_avx2(8, a.as_ptr(), b.as_ptr()) };
    let rel_err = (result - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-6,
        "dot product: {result} vs {expected} (rel_err={rel_err:.2e})"
    );
}

#[test]
fn test_dot_product_symmetry() {
    let n = 256;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.05).collect();

    let ab = unsafe { asm::dot_product_avx2(n, a.as_ptr(), b.as_ptr()) };
    let ba = unsafe { asm::dot_product_avx2(n, b.as_ptr(), a.as_ptr()) };

    let rel_err = (ab - ba).abs() / ab.abs().max(1e-10);
    assert!(rel_err < 1e-6, "dot product asymmetry: {ab} vs {ba}");
}

// ============================================================
// SiLU ACTIVATION
// ============================================================

#[test]
fn test_silu_known_values() {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let test_vals = [
        (-3.0, -3.0 / (1.0 + (3.0f32).exp())),
        (0.0, 0.0),
        (3.0, 3.0 / (1.0 + (-3.0f32).exp())),
    ];
    let src: Vec<f32> = test_vals.iter().map(|(x, _)| *x).collect();
    let mut dst = vec![0.0f32; test_vals.len()];

    unsafe {
        asm::silu_vectorial_avx2(test_vals.len(), src.as_ptr(), dst.as_mut_ptr());
    }

    for (i, (x, expected)) in test_vals.iter().enumerate() {
        let rel_err = (dst[i] - expected).abs() / expected.abs().max(1e-10);
        let got = dst[i];
        assert!(
            rel_err < 1e-5,
            "SiLU({}): {} vs {} (rel_err={:.2e})",
            x,
            got,
            expected,
            rel_err
        );
    }
}

// ============================================================
// ROUTING CORRECTNESS
// ============================================================

#[test]
fn test_router_route_in_place_distribution() {
    let num_experts = 8;
    let top_k = 2;
    let router = MudRouter::new(num_experts, top_k);

    let logits = vec![1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5];
    let mut indexed = Vec::with_capacity(num_experts);
    let mut results = Vec::with_capacity(8);

    let z_loss = router.route_in_place(&logits, &mut indexed, &mut results, None);

    assert!(!results.is_empty(), "route_in_place returned empty");
    assert!(results.len() <= top_k, "route_in_place returned too many");
    assert!(z_loss.is_finite(), "z_loss not finite: {z_loss}");

    let probs_sum: f32 = results.iter().map(|(_, p)| p).sum();
    let rel_err = (probs_sum - 1.0).abs();
    assert!(
        rel_err < 1e-5,
        "route_in_place probabilities sum to {probs_sum} != 1.0"
    );
}

#[test]
fn test_router_route_by_hash_deterministic() {
    let num_experts = 8;
    let top_k = 2;
    let router = MudRouter::new(num_experts, top_k);

    let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
    let mut r1 = Vec::new();
    let mut r2 = Vec::new();

    router.route_by_hash(&x, &mut r1);
    router.route_by_hash(&x, &mut r2);

    assert_eq!(r1, r2, "hash routing should be deterministic");
    assert_eq!(r1.len(), top_k, "hash routing should return exactly top_k");
}

#[test]
fn test_router_route_by_hash_caps_at_num_experts() {
    let num_experts = 3;
    let top_k = 8;
    let router = MudRouter::new(num_experts, top_k);

    let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.25).collect();
    let mut results = Vec::new();

    router.route_by_hash(&x, &mut results);

    assert_eq!(
        results.len(),
        num_experts,
        "hash routing should not exceed expert count"
    );
    let probs_sum: f32 = results.iter().map(|(_, p)| *p).sum();
    assert!(
        (probs_sum - 1.0).abs() < 1e-5,
        "hash routing probabilities sum to {probs_sum} != 1.0"
    );
}

#[test]
fn test_router_route_by_hash_zero_top_k() {
    let router = MudRouter::new(4, 0);
    let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let mut results = Vec::new();

    router.route_by_hash(&x, &mut results);

    assert!(results.is_empty(), "zero top_k should select no experts");
}

#[test]
fn test_router_q_head_stochastic() {
    let num_experts = 8;
    let top_k = 2;
    let router = MudRouter::new(num_experts, top_k);

    let logits = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
    let mut indexed = Vec::with_capacity(num_experts);
    let mut results = Vec::with_capacity(8);

    let z_loss = router.route_by_q_head(&logits, 0.1, 42, &mut indexed, &mut results);

    assert!(!results.is_empty(), "Q-head routing returned empty");
    assert!(results.len() <= top_k, "Q-head routing too many results");
    assert!(z_loss.is_finite(), "Q-head z_loss not finite: {z_loss}");
}

// ============================================================
// NaN / Inf STABILITY
// ============================================================

#[test]
fn test_ternary_gemv_nan_input() {
    let n = 64;
    let x: Vec<f32> = vec![f32::NAN; n];
    let packed = make_ternary_packed(&vec![1i8; n]);
    let mut out = 0.0f32;
    unsafe {
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0);
    }
    assert!(out.is_nan(), "GEMV NaN input should produce NaN output");
}

#[test]
fn test_ternary_gemv_inf_input() {
    let n = 64;
    let x: Vec<f32> = vec![f32::INFINITY; n];
    let packed = make_ternary_packed(&vec![1i8; n]);
    let mut out = 0.0f32;
    unsafe {
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut out, 1.0);
    }
    assert!(
        out.is_infinite(),
        "GEMV Inf input should produce Inf output"
    );
}

#[test]
fn test_rms_norm_nan_input() {
    let n = 128;
    let x: Vec<f32> = vec![f32::NAN; n];
    let scale = unsafe { asm::rms_norm_scale_asm(n, x.as_ptr(), 1e-6) };
    assert!(scale.is_nan(), "RMSNorm NaN input should produce NaN");
}

#[test]
fn test_dot_product_nan_input() {
    let n = 64;
    let a: Vec<f32> = vec![f32::NAN; n];
    let b: Vec<f32> = vec![1.0f32; n];
    let result = unsafe { asm::dot_product_avx2(n, a.as_ptr(), b.as_ptr()) };
    assert!(result.is_nan(), "dot product NaN input should produce NaN");
}

// ============================================================
// SUM SQUARES
// ============================================================

#[test]
fn test_sum_squares_known() {
    let x: Vec<f32> = (0..128).map(|i| i as f32 * 0.5).collect();
    let expected: f32 = x.iter().map(|v| v * v).sum();
    let result = unsafe { asm::sum_squares_avx2(128, x.as_ptr()) };
    let rel_err = (result - expected).abs() / expected.abs();
    assert!(
        rel_err < 1e-5,
        "sum_squares: {result} vs {expected} (rel_err={rel_err:.2e})"
    );
}

// ============================================================
// TERNARY GEMV SCALING
// ============================================================

#[test]
fn test_ternary_gemv_scale_effect() {
    let n = 128;
    let x: Vec<f32> = vec![2.0f32; n];
    let packed = make_ternary_packed(&vec![1i8; n]);

    let mut out1 = 0.0f32;
    let mut out2 = 0.0f32;
    unsafe {
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut out1, 1.0);
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut out2, 2.5);
    }

    let rel_err = (out2 - 2.5 * out1).abs() / (2.5 * out1).abs().max(1e-10);
    assert!(
        rel_err < 1e-5,
        "GEMV scale factor mismatch: {out2} vs {} (rel_err={rel_err:.2e})",
        2.5 * out1
    );
}

// ============================================================
// QUANTIZATION NOISE BOUND
// ============================================================

#[test]
fn test_ternary_vs_fp32_quantization_noise() {
    let n = 512;
    let x: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.3).cos() * 5.0).collect();

    let mut fp32_result = 0.0f32;
    let packed = make_ternary_packed(&vec![1i8; n]);
    unsafe {
        asm::ternary_gemv(n, x.as_ptr(), packed.as_ptr(), &mut fp32_result, 1.0);
    }

    let absmax = x.iter().map(|v| v.abs()).fold(1e-8f32, f32::max);
    let q_scale = 127.0 / absmax;
    let mut x_q = vec![0i8; n];
    for i in 0..n {
        x_q[i] = (x[i] * q_scale).round().clamp(-127.0, 127.0) as i8;
    }
    let inv_q_scale = absmax / 127.0;

    let blocks_64 = n / 32;
    let mut w_unpacked = vec![0i8; n];
    unsafe {
        for b in 0..blocks_64 {
            asm::pext_unpack_ternary(
                *(packed.as_ptr() as *const u64).add(b),
                w_unpacked.as_mut_ptr().add(b * 32),
            );
        }
    }

    let mut tsar_result = 0.0f32;
    unsafe {
        asm::ternary_gemv_lut_avx2(n, x_q.as_ptr(), w_unpacked.as_ptr(), &mut tsar_result, 1.0);
    }

    tsar_result *= inv_q_scale;

    let snr = fp32_result.abs() / (tsar_result - fp32_result).abs().max(1e-10);
    assert!(
        snr > 10.0 || fp32_result.abs() < 1.0,
        "T-SAR SNR too low: {snr:.1}dB (fp32={fp32_result}, tsar={tsar_result})"
    );
}

// ============================================================
// E2E SMOKE TESTS — Model loading & error handling
// ============================================================

#[test]
fn test_mud_load_nonexistent_file() {
    let result = MudFile::load("/tmp/nonexistent_model_xyz.mud");
    assert!(result.is_err(), "Should fail on nonexistent file");
}

#[test]
fn test_mud_load_invalid_magic() {
    std::fs::write("/tmp/test_bad_magic.mud", b"BADDATA").unwrap();
    let result = MudFile::load("/tmp/test_bad_magic.mud");
    assert!(result.is_err(), "Should fail on invalid magic");
    let _ = std::fs::remove_file("/tmp/test_bad_magic.mud");
}

#[test]
fn test_slime_workspace_creation_no_core() {
    let ws = crate::mud::slime::SlimeWorkspace::new(64, 128, 4, 2, 16, 64, 30, 128.0);
    assert_eq!(ws.registers.len(), 64);
    assert_eq!(ws.kv_cache.len(), 30 * 2 * 128 * 16);
    assert_eq!(ws.norm_i8.len(), 64);
    assert_eq!(ws.scores.len(), 128);
}

#[test]
fn test_real_model_loads_successfully() {
    let model_path = "models/bitnet-b1.58-2B-4T.mud";
    if !std::path::Path::new(model_path).exists() {
        eprintln!("Skipping real model load test: {model_path} not found");
        return;
    }
    let file = MudFile::load(model_path);
    assert!(file.is_ok(), "Should load real model file");
    let file = file.unwrap();
    assert!(file.skills.contains_key("core"), "Should have core skill");
}
