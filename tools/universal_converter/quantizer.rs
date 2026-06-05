#![allow(clippy::needless_range_loop)]
use rayon::prelude::*;
use std::collections::HashMap;

use half::{bf16, f16};
use safetensors::tensor::{Dtype, TensorView};

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    for (a, b) in v1.iter().zip(v2.iter()) {
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    dot / (norm1.sqrt() * norm2.sqrt())
}

fn holographic_scale_search(row_slice: &[f32], absmean: f32) -> f32 {
    if absmean < 1e-8 { return 1e-8; }
    let initial_scale = absmean * 0.707;
    let mut best_scale = initial_scale;
    let mut best_score = -1000.0;

    let mut q_buf = vec![0.0f32; row_slice.len()];
    
    // Holographic Wave Distillation: 1D grid search to maximize Phase (Cosine) Similarity
    // while enforcing the MUD 26.0% Sparsity boundary.
    for step in 0..100 {
        let factor = 0.5 + (1.5 * (step as f32) / 99.0); // Search between 0.5x and 2.0x
        let scale = initial_scale * factor;
        
        let mut non_zeros = 0;
        for (i, &w) in row_slice.iter().enumerate() {
            let q = (w / scale).round().clamp(-1.0, 1.0);
            q_buf[i] = q;
            if q != 0.0 {
                non_zeros += 1;
            }
        }
        
        let sparsity = 1.0 - (non_zeros as f32 / row_slice.len() as f32);
        let sim = cosine_similarity(row_slice, &q_buf);
        
        // Combine Holographic Distillation with Mathematical Boundaries
        let sparsity_penalty = (sparsity - 0.26).abs() * 2.0;
        let score = sim - sparsity_penalty;

        if score > best_score {
            best_score = score;
            best_scale = scale;
        }
    }
    best_scale.max(1e-8)
}

pub fn to_f32_vec(tensor: &TensorView) -> Vec<f32> {
    match tensor.dtype() {
        Dtype::F16 => {
            let slice: &[f16] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f16,
                    tensor.data().len() / 2,
                )
            };
            slice.par_iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let slice: &[bf16] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const bf16,
                    tensor.data().len() / 2,
                )
            };
            slice.par_iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::F32 => {
            let slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f32,
                    tensor.data().len() / 4,
                )
            };
            slice.to_vec()
        }
        Dtype::U8 => {
            let slice: &[u8] = tensor.data();
            let n = tensor.shape()[0];
            let m = tensor.shape()[1]; // assuming 2D for linear layers
            let mut un_packed = vec![0.0f32; n * 4 * m];
            for i in 0..n {
                for j in 0..m {
                    let mut v = slice[i * m + j] as i32;
                    for k in 0..4 {
                        let w = (v % 3) - 1;
                        un_packed[(4 * i + k) * m + j] = w as f32;
                        v /= 3;
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

pub fn ternarize_f32_and_pack(floats: &[f32], n_rows: usize, n_cols: usize, bitnet_scale: f32) -> (Vec<u8>, Vec<f32>) {
    // Compute per-row absmean scales with depth dampening factor (0.707)
    let mut row_scales: Vec<f32> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let start = row * n_cols;
            let row_slice = &floats[start..start + n_cols];
            let absmean = row_slice.iter().map(|v| v.abs()).sum::<f32>() / n_cols as f32;
            holographic_scale_search(row_slice, absmean)
        })
        .collect();

    // Pack ternary values per row (16 values per u32, 2 bits each)
    let u32s_per_row = n_cols.div_ceil(16);
    let total_u32s = n_rows * u32s_per_row;
    let mut packed_u32s = vec![0u32; total_u32s];

    packed_u32s
        .par_chunks_mut(u32s_per_row)
        .zip(row_scales.par_iter_mut())
        .enumerate()
        .for_each(|(row, (row_packed, scale_ref))| {
            let start = row * n_cols;
            let scale_round = *scale_ref;
            let mut var_w = 0.0;
            let mut non_zeros = 0;
            
            for (col, &w) in floats[start..start + n_cols].iter().enumerate() {
                var_w += w * w;
                let q = (w / scale_round).round().clamp(-1.0, 1.0);
                if q != 0.0 { non_zeros += 1; }
                
                let bits = if q > 0.5 {
                    1u32
                } else if q < -0.5 {
                    2u32
                } else {
                    0u32
                };
                let u32_idx = col / 16;
                let shift = (col % 16) * 2;
                row_packed[u32_idx] |= bits << shift;
            }
            
            // Variance matching inference scale
            var_w /= n_cols as f32;
            let var_q = non_zeros as f32 / n_cols as f32;
            let scale_inf = if var_q > 0.0 {
                (var_w / var_q).sqrt()
            } else {
                0.0
            };
            
            // Overwrite the scale array with the inference scale, then scale it by BitNet's weight_scale!
            *scale_ref = scale_inf * bitnet_scale;
        });

    let mut out_bytes = Vec::with_capacity(total_u32s * 4);
    for p in packed_u32s {
        out_bytes.extend_from_slice(&p.to_le_bytes());
    }

    (out_bytes, row_scales)
}

/// Pack pre-ternarized f32 values (∈ {-1,0,+1}) into 2-bit u8
pub fn pack_ternary_from_f32(ternary: &[f32]) -> Vec<u8> {
    let u32_count = ternary.len().div_ceil(16);
    let mut packed = vec![0u32; u32_count];
    for i in 0..ternary.len() {
        let bit = if ternary[i] > 0.5 {
            1u32
        } else if ternary[i] < -0.5 {
            2u32
        } else {
            0u32
        };
        let u32_idx = i / 16;
        let shift = (i % 16) * 2;
        packed[u32_idx] |= bit << shift;
    }
    let bytes =
        unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4) };
    bytes.to_vec()
}

/// Apply row-wise absmean ternarization to an embedding table.
/// Returns (packed_ternary, per_row_scales_f32, metadata).
pub fn embedding_rowwise_ternarize(
    emb_data: &[f32],
    vocab: usize,
    hidden: usize,
) -> (Vec<u8>, Vec<f32>, HashMap<String, String>) {
    let mut scales_f32 = Vec::with_capacity(vocab);
    for row in 0..vocab {
        let start = row * hidden;
        let row_slice = &emb_data[start..start + hidden];
        let absmean = row_slice.iter().map(|v| v.abs()).sum::<f32>() / hidden as f32;
        scales_f32.push(holographic_scale_search(row_slice, absmean));
    }

    // Ternarize each row using direct rounding. Error Diffusion destroys orthogonal semantics.
    let mut ternary = vec![0.0f32; emb_data.len()];
    for row in 0..vocab {
        let s = scales_f32[row];
        let start = row * hidden;
        for j in 0..hidden {
            let w = emb_data[start + j];
            let q = (w / s).round().clamp(-1.0, 1.0);
            ternary[start + j] = q;
        }
    }

    let packed = pack_ternary_from_f32(&ternary);

    let metadata = HashMap::from([("embed_ternarized".to_string(), "row_absmean".to_string())]);

    (packed, scales_f32, metadata)
}

pub fn convert_to_f32_bytes(tensor: &TensorView) -> Vec<u8> {
    let floats: Vec<f32> = match tensor.dtype() {
        Dtype::F16 => {
            let slice: &[f16] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f16,
                    tensor.data().len() / 2,
                )
            };
            slice.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let slice: &[bf16] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const bf16,
                    tensor.data().len() / 2,
                )
            };
            slice.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::F32 => return tensor.data().to_vec(),
        Dtype::U8 => {
            let slice: &[u8] = tensor.data();
            let n = tensor.shape()[0];
            let m = tensor.shape()[1]; // assuming 2D for linear layers
            let mut un_packed = vec![0.0f32; n * 4 * m];
            for i in 0..n {
                for j in 0..m {
                    let mut v = slice[i * m + j] as i32;
                    for k in 0..4 {
                        let w = (v % 3) - 1;
                        un_packed[(4 * i + k) * m + j] = w as f32;
                        v /= 3;
                    }
                }
            }
            un_packed
        }
        _ => panic!("Unsupported dtype: {:?}", tensor.dtype()),
    };

    let mut out = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}
