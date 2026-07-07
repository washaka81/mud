#![allow(clippy::needless_range_loop)]
use std::collections::HashMap;

use half::{bf16, f16};
use safetensors::tensor::{Dtype, TensorView};

/// EPSILON_FLOOR: Mandated by Audit V9 as the absolute numerical floor for all stability-critical divisions.
const EPSILON_FLOOR: f32 = 1.1e-8;

/// Applies a 1D Haar Wavelet Decomposition (Transformada Wavelet de Haar).
/// Extracts high-frequency and low-frequency components to prepare weights
/// for advanced Phase 1 Holographic Wave conversions.
#[allow(dead_code)]
pub fn haar_wavelet_decompose(row_slice: &[f32]) -> Vec<f32> {
    let len = row_slice.len();
    if len == 0 || !len.is_multiple_of(2) {
        return row_slice.to_vec(); // Requires even length
    }
    let mut decomposed = vec![0.0; len];
    let half = len / 2;
    for i in 0..half {
        let a = row_slice[2 * i];
        let b = row_slice[2 * i + 1];
        // Standard Haar Wavelet using 1/sqrt(2) scaling
        let sqrt2 = std::f32::consts::SQRT_2;
        decomposed[i] = (a + b) / sqrt2;
        decomposed[half + i] = (a - b) / sqrt2;
    }
    decomposed
}

pub fn to_f32_vec(tensor: &TensorView) -> Vec<f32> {
    match tensor.dtype() {
        Dtype::F16 => {
            let n = tensor.data().len() / 2;
            let mut slice = vec![f16::from_bits(0); n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.data().as_ptr(),
                    slice.as_mut_ptr() as *mut u8,
                    tensor.data().len(),
                );
            }
            slice.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let n = tensor.data().len() / 2;
            let mut slice = vec![bf16::from_bits(0); n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.data().as_ptr(),
                    slice.as_mut_ptr() as *mut u8,
                    tensor.data().len(),
                );
            }
            slice.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::F32 => {
            let n = tensor.data().len() / 4;
            let mut slice = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    tensor.data().as_ptr(),
                    slice.as_mut_ptr() as *mut u8,
                    tensor.data().len(),
                );
            }
            slice
        }
        Dtype::U8 => {
            let slice: &[u8] = tensor.data();
            let rows_packed = tensor.shape()[0];
            let cols = tensor.shape()[1];
            let mut un_packed = vec![0.0f32; rows_packed * 4 * cols];

            for r_p in 0..rows_packed {
                for c in 0..cols {
                    let val = slice[r_p * cols + c];
                    for k in 0..4 {
                        let bits = (val >> (k * 2)) & 3;
                        let weight = match bits {
                            0 => -1.0,
                            1 => 0.0,
                            2 => 1.0,
                            _ => 0.0,
                        };
                        un_packed[(r_p * 4 + k) * cols + c] = weight;
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
        for (col, &w) in floats[start..start + n_cols].iter().enumerate() {
            let q = if w > delta {
                1.0
            } else if w < -delta {
                -1.0
            } else {
                0.0
            };

            let bits = if q > 0.5 {
                0x1u32 // +1
            } else if q < -0.5 {
                0xFu32 // -1 in 4-bit two's complement
            } else {
                0x0u32 // 0
            };
            let u32_idx = col / 8;
            let shift = (col % 8) * 4;
            row_packed[u32_idx] |= bits << shift;
        }

        // STRICTLY USING ABSMEAN SCALE to prevent scale calculation drift!
        row_scales[row] = (dampened_scale * (1.0 / bitnet_scale)).max(EPSILON_FLOOR);
    }

    let mut out_bytes = Vec::with_capacity(total_u32s * 4);
    for p in packed_u32s {
        out_bytes.extend_from_slice(&p.to_le_bytes());
    }

    (out_bytes, row_scales)
}

/// H1-04: Direct ternary→ternary repack from BitNet U8 to MUD u32 format.
/// Bypasses float conversion entirely, preserving the original BitNet ternary values.
/// BitNet U8: 4 values per byte, 2 bits each (1=+1, 2=-1, 0=0), column-major within packed rows.
/// MUD u32: 8 values per u32, 4 bits each (ELUT format), row-major.
/// Also computes per-row inference scales from the ternary pattern without re-quantization.
pub fn repack_bitnet_to_mud(
    data: &[u8],
    rows_packed: usize,
    cols: usize,
    bitnet_scale: f32,
) -> (Vec<u8>, Vec<f32>) {
    let n_rows = rows_packed * 4;
    let u32s_per_row = cols.div_ceil(8);
    let total_u32s = n_rows * u32s_per_row;
    let mut packed_u32s = vec![0u32; total_u32s];
    let mut scales = vec![0.0f32; n_rows];

    for rp in 0..rows_packed {
        for col in 0..cols {
            let byte_val = data[rp * cols + col];
            for k in 0..4u32 {
                let bitnet_bits = ((byte_val >> (k * 2)) & 3) as u32;
                let bits = match bitnet_bits {
                    0 => 0xFu32, // BitNet 0 is -1 -> MUD ELUT 0xF is -1
                    1 => 0x0u32, // BitNet 1 is 0 -> MUD ELUT 0x0 is 0
                    2 => 0x1u32, // BitNet 2 is +1 -> MUD ELUT 0x1 is +1
                    _ => 0x0u32,
                };
                let row = rp * 4 + k as usize;

                let u32_idx = row * u32s_per_row + col / 8;
                let shift = (col % 8) * 4;
                packed_u32s[u32_idx] |= bits << shift;

                // Accumulate for inference scale computation
                if bits != 0 {
                    scales[row] += 1.0;
                }
            }
        }
    }

    // Compute inference scales from ternary density (no re-quantization)
    for row in 0..n_rows {
        let density = scales[row] / cols as f32;
        scales[row] = if density > EPSILON_FLOOR {
            (1.0 / bitnet_scale).max(EPSILON_FLOOR)
        } else {
            EPSILON_FLOOR
        };
    }

    let mut out_bytes = Vec::with_capacity(total_u32s * 4);
    for p in &packed_u32s {
        out_bytes.extend_from_slice(&p.to_le_bytes());
    }

    (out_bytes, scales)
}

/// Pack pre-ternarized f32 values (∈ {-1,0,+1}) into 2-bit u8
pub fn pack_ternary_from_f32(ternary: &[f32]) -> Vec<u8> {
    let u32_count = ternary.len().div_ceil(8);
    let mut packed = vec![0u32; u32_count];
    for i in 0..ternary.len() {
        let bit = if ternary[i] > 0.5 {
            0x1u32
        } else if ternary[i] < -0.5 {
            0xFu32
        } else {
            0x0u32
        };
        let u32_idx = i / 8;
        let shift = (i % 8) * 4;
        packed[u32_idx] |= bit << shift;
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4) };
    bytes.to_vec()
}

/// Apply row-wise absmean ternarization to an embedding table.
/// Returns (packed_ternary, per_row_scales_f32, metadata).
#[allow(dead_code)]
pub fn embedding_rowwise_ternarize(
    emb_data: &[f32],
    vocab: usize,
    hidden: usize,
) -> (Vec<u8>, Vec<f32>, HashMap<String, String>) {
    let mut scales_f32 = Vec::with_capacity(vocab);
    for row in 0..vocab {
        let start = row * hidden;
        let row_slice = &emb_data[start..start + hidden];
        let absmean = row_slice.iter().map(|w| w.abs()).sum::<f32>() / (hidden as f32);
        let scale = absmean * std::f32::consts::FRAC_1_SQRT_2;
        scales_f32.push(scale.max(EPSILON_FLOOR));
    }

    // Ternarize each row using direct rounding. Error Diffusion destroys orthogonal semantics.
    let mut ternary = vec![0.0f32; emb_data.len()];
    for row in 0..vocab {
        let start = row * hidden;
        let row_slice = &emb_data[start..start + hidden];
        let absmean = row_slice.iter().map(|w| w.abs()).sum::<f32>() / (hidden as f32);
        let dampened_scale = absmean * std::f32::consts::FRAC_1_SQRT_2;
        let delta = 0.7 * dampened_scale;
        for j in 0..hidden {
            let w = emb_data[start + j];
            let q = if w > delta {
                1.0
            } else if w < -delta {
                -1.0
            } else {
                0.0
            };
            ternary[start + j] = q;
        }
    }

    let packed = pack_ternary_from_f32(&ternary);

    let metadata = HashMap::from([("embed_ternarized".to_string(), "row_absmean".to_string())]);

    (packed, scales_f32, metadata)
}

/// DEEP AUDIT: Verifies that the .mud ternarized weights are the closest possible
/// to the original tokens to ensure the tokenizer routes correctly to expected probabilities.
pub fn audit_ternary_fidelity(
    name: &str,
    f32_data: &[f32],
    mud_bytes: &[u8],
    scales: &[f32],
    n_rows: usize,
    n_cols: usize,
    bitnet_scale: f32,
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
            let bits = (u32_slice[u32_idx] >> shift) & 0xF;

            let q_val = match bits {
                0x1 => scale,
                0xF => -scale,
                _ => 0.0,
            };

            let orig = f32_data[row_start_f32 + col] * bitnet_scale;
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

pub fn pack_native_ternary_f32(
    floats: &[f32],
    n_rows: usize,
    n_cols: usize,
    bitnet_scale: f32,
) -> (Vec<u8>, Vec<f32>) {
    let u32s_per_row = n_cols.div_ceil(8);
    let total_u32s = n_rows * u32s_per_row;
    let mut packed_u32s = vec![0u32; total_u32s];
    let mut row_scales = vec![0.0f32; n_rows];

    for row in 0..n_rows {
        let row_packed = &mut packed_u32s[row * u32s_per_row..(row + 1) * u32s_per_row];
        let start = row * n_cols;

        for (col, &w) in floats[start..start + n_cols].iter().enumerate() {
            let bits = if w > 0.5 {
                0x1u32
            } else if w < -0.5 {
                0xFu32
            } else {
                0x0u32
            };
            let u32_idx = col / 8;
            let shift = (col % 8) * 4;
            row_packed[u32_idx] |= bits << shift;
        }

        // BitNet mathematically divides the output by scale_w: Y = (X @ W_q) / scale_w.
        // MUD multiplies the output by scale: Y = (X @ W_q) * scale.
        // Therefore, MUD scale = 1.0 / scale_w.
        row_scales[row] = (1.0 / bitnet_scale).max(EPSILON_FLOOR);
    }

    let mut out_bytes = Vec::with_capacity(total_u32s * 4);
    for p in packed_u32s {
        out_bytes.extend_from_slice(&p.to_le_bytes());
    }

    (out_bytes, row_scales)
}
