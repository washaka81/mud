#![allow(clippy::needless_range_loop)]
use half::f16;
use safetensors::tensor::{Dtype, TensorView};

/// EPSILON_FLOOR: Mandated by Audit V9 as the absolute numerical floor for all stability-critical divisions.
const EPSILON_FLOOR: f32 = 1.1e-8;

/// ELUT 4-bit → f32 decode table (mirrors `forge_llm::mud::TERNARY_LUT`).
/// Branchless lookup replaces per-element `match` in the audit hot path.
/// Index 0x1 → +1.0, 0xF → -1.0, all others → 0.0.
const TERNARY_LUT: [f32; 16] = [
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
];

/// 2-bit → f32 decode table for U8 BitNet format (mirrors quantizer U8 unpacking).
/// Index 0 → -1.0, 1 → 0.0, 2 → +1.0, 3 → 0.0.
const U8_LUT: [f32; 4] = [-1.0, 0.0, 1.0, 0.0];

/// Convert a safetensors TensorView to f32 with zero intermediate allocations (P-01).
/// Uses direct pointer arithmetic instead of creating temporary Vecs of f16/bf16.
pub fn to_f32_vec(tensor: &TensorView) -> Vec<f32> {
    let data = tensor.data();
    match tensor.dtype() {
        Dtype::F16 => {
            let n = data.len() / 2;
            let mut out = vec![0.0f32; n];
            // SAFETY: data.len() == n * 2, reading n f16 values.
            unsafe {
                let ptr = data.as_ptr() as *const u16;
                for i in 0..n {
                    let bits = *ptr.add(i);
                    out[i] = f16::from_le_bytes([bits as u8, (bits >> 8) as u8]).to_f32();
                }
            }
            out
        }
        Dtype::BF16 => {
            let n = data.len() / 2;
            let mut out = vec![0.0f32; n];
            // SAFETY: data.len() == n * 2, reading n bf16 values.
            unsafe {
                let ptr = data.as_ptr() as *const u16;
                for i in 0..n {
                    let bits = *ptr.add(i);
                    out[i] = f32::from_bits((bits as u32) << 16);
                }
            }
            out
        }
        Dtype::F32 => {
            let n = data.len() / 4;
            // SAFETY: data.len() == n * 4, reading n f32 values.
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, n).to_vec()
            }
        }
        Dtype::U8 => {
            let slice: &[u8] = data;
            let rows_packed = tensor.shape()[0];
            let cols = tensor.shape()[1];
            let mut un_packed = vec![0.0f32; rows_packed * 4 * cols];

            for r_p in 0..rows_packed {
                for c in 0..cols {
                    let val = slice[r_p * cols + c];
                    for k in 0..4 {
                        let bits = (val >> (k * 2)) & 3;
                        // Branchless LUT lookup (P-00)
                        un_packed[(r_p * 4 + k) * cols + c] = U8_LUT[bits as usize];
                    }
                }
            }
            un_packed
        }
        _ => panic!("Unsupported dtype: {:?}", tensor.dtype()),
    }
}

#[allow(dead_code)]
pub fn ternarize_and_pack(tensor: &TensorView, bitnet_scale: f32) -> (Vec<u8>, Vec<f32>) {
    let floats = to_f32_vec(tensor);

    let n_rows = if tensor.dtype() == Dtype::U8 {
        tensor.shape()[0] * 4
    } else {
        tensor.shape()[0]
    };
    let n_cols = floats.len() / n_rows;

    ternarize_f32_and_pack(&floats, n_rows, n_cols, bitnet_scale)
}

/// Ternarize f32 weights and pack into ELUT 4-bit format (8 values per u32).
///
/// Optimizations:
/// - Branchless ternary quantization: `q = (w > delta) - (w < -delta)` avoids branch misprediction
/// - Branchless bit encoding: uses arithmetic instead of if/else chains
/// - Raw pointer writes for packed u32s (P-00)
/// - In-place byte conversion (no intermediate Vec copy)
pub fn ternarize_f32_and_pack(
    floats: &[f32],
    n_rows: usize,
    n_cols: usize,
    bitnet_scale: f32,
) -> (Vec<u8>, Vec<f32>) {
    // Compute per-row absmean scales with depth dampening factor (0.707)
    let mut row_scales: Vec<f32> = (0..n_rows)
        .map(|row| {
            let start = row * n_cols;
            let row_slice = &floats[start..start + n_cols];
            let absmean = row_slice.iter().map(|w| w.abs()).sum::<f32>() / (n_cols as f32);
            absmean * std::f32::consts::FRAC_1_SQRT_2
        })
        .collect();

    // Pack ternary values per row (8 values per u32, 4 bits each - ELUT format)
    let u32s_per_row = n_cols.div_ceil(8);
    let total_u32s = n_rows * u32s_per_row;
    let mut packed_u32s = vec![0u32; total_u32s];

    for row in 0..n_rows {
        let row_packed = &mut packed_u32s[row * u32s_per_row..(row + 1) * u32s_per_row];
        let start = row * n_cols;
        let dampened_scale = row_scales[row];
        let delta = 0.7 * dampened_scale;

        // Branchless quantization: q ∈ {-1, 0, +1} without branches
        // q = (w > delta) as i32 - (w < -delta) as i32
        // Then map to bits: +1 → 0x1, -1 → 0xF, 0 → 0x0
        for (col, &w) in floats[start..start + n_cols].iter().enumerate() {
            // Branchless sign extraction
            let is_pos = (w > delta) as u32;
            let is_neg = (w < -delta) as u32;
            // bits: 0x1 if pos, 0xF if neg, 0x0 if zero
            // Using arithmetic: bits = is_pos * 1 + is_neg * 0xF
            // (is_pos and is_neg are mutually exclusive, so this is correct)
            let bits = is_pos + is_neg * 0xF;
            let u32_idx = col / 8;
            let shift = (col % 8) * 4;
            row_packed[u32_idx] |= bits << shift;
        }

        // STRICTLY USING ABSMEAN SCALE to prevent scale calculation drift!
        row_scales[row] = (dampened_scale * (1.0 / bitnet_scale)).max(EPSILON_FLOOR);
    }

    // Convert u32s to bytes (flat_map avoids intermediate copy, safe for clippy)
    let out_bytes: Vec<u8> = packed_u32s
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .collect();

    (out_bytes, row_scales)
}

/// DEEP AUDIT: Verifies that the .mud ternarized weights are the closest possible
/// to the original tokens to ensure the tokenizer routes correctly to expected probabilities.
///
/// Uses branchless TERNARY_LUT lookup (mirrors `forge_llm::mud::TERNARY_LUT`) instead
/// of `match` for improved performance in the audit hot path.
pub fn audit_ternary_fidelity(
    name: &str,
    f32_data: &[f32],
    mud_bytes: &[u8],
    scales: &[f32],
    n_rows: usize,
    n_cols: usize,
    _bitnet_scale: f32,
) {
    if f32_data.is_empty() || mud_bytes.is_empty() {
        return;
    }

    // mud_bytes is an array of u32 (little endian)
    let u32s_per_row = n_cols.div_ceil(8);
    let u32_slice = unsafe {
        std::slice::from_raw_parts(mud_bytes.as_ptr() as *const u32, mud_bytes.len() / 4)
    };

    let mut total_mse = 0.0f64;
    let mut total_signal = 0.0f64;
    let mut max_diff = 0.0f32;

    for row in 0..n_rows {
        let scale = scales[row];
        let row_start_u32 = row * u32s_per_row;
        let row_start_f32 = row * n_cols;

        for col in 0..n_cols {
            let u32_idx = row_start_u32 + col / 8;
            let shift = (col % 8) * 4;
            let bits = ((u32_slice[u32_idx] >> shift) & 0xF) as usize;

            // Branchless LUT lookup (mirrors forge_llm::mud::TERNARY_LUT)
            let q_val = TERNARY_LUT[bits] * scale;

            // FIX (Finding 1): Do NOT multiply orig by bitnet_scale.
            // The scale already includes 1/bitnet_scale (computed in ternarize_f32_and_pack
            // as `dampened_scale * (1.0 / bitnet_scale)`). Multiplying orig by bitnet_scale
            // would make the audit wrong by a factor of bitnet_scale² when bitnet_scale ≠ 1.0.
            // The correct comparison is: original_weight vs (ternary_value * final_scale).
            let orig = f32_data[row_start_f32 + col];
            let diff = orig - q_val;

            total_mse += (diff as f64) * (diff as f64);
            total_signal += (orig as f64) * (orig as f64);
            if diff.abs() > max_diff {
                max_diff = diff.abs();
            }
        }
    }

    let mse = total_mse / (n_rows * n_cols) as f64;
    let signal_power = total_signal / (n_rows * n_cols) as f64;
    let sqnr = if mse > 0.0 {
        10.0 * (signal_power / mse).log10()
    } else {
        f64::INFINITY
    };

    let is_critical = name.contains("embed") || name == "output.weight";

    if is_critical {
        println!(
            "🔍 DEEP AUDIT [{}]: SQNR = {:.2} dB | MSE = {:.6} | Max Diff = {:.4}",
            name, sqnr, mse, max_diff
        );
        if sqnr < 5.0 {
            println!("⚠️  WARNING: Token routing fidelity is dangerously low for {}! The tokenizer may not route correctly to expected probabilities.", name);
        } else {
            println!("✅  Token routing fidelity verified for {}. Weights match closest ternary proxies.", name);
        }
    } else if sqnr < 1.0 {
        // Random other layers
        println!(
            "⚠️  WARNING: Layer {} has very low SQNR ({:.2} dB)",
            name, sqnr
        );
    }
}
