use std::collections::HashMap;
use rayon::prelude::*;

use safetensors::tensor::{TensorView, Dtype};
use half::{f16, bf16};

pub fn ternarize_and_pack(tensor: &TensorView, dampening_factor: f32) -> (Vec<u8>, f32) {
    let floats: Vec<f32> = match tensor.dtype() {
        Dtype::F16 => {
            let slice: &[f16] = unsafe { std::slice::from_raw_parts(tensor.data().as_ptr() as *const f16, tensor.data().len() / 2) };
            slice.par_iter().map(|&x| x.to_f32()).collect()
        },
        Dtype::BF16 => {
            let slice: &[bf16] = unsafe { std::slice::from_raw_parts(tensor.data().as_ptr() as *const bf16, tensor.data().len() / 2) };
            slice.par_iter().map(|&x| x.to_f32()).collect()
        },
        Dtype::F32 => {
            let slice: &[f32] = unsafe { std::slice::from_raw_parts(tensor.data().as_ptr() as *const f32, tensor.data().len() / 4) };
            slice.to_vec()
        },
        _ => panic!("Unsupported dtype: {:?}", tensor.dtype()),
    };

    let max_abs = floats.par_iter().map(|x| x.abs()).reduce(|| 0.0f32, f32::max);
    
    let optimal_scale = if max_abs > 1e-6 {
        let scales_to_test: Vec<f32> = (1..=100).map(|i| (i as f32) / 100.0 * max_abs).collect();
        
        let best_candidate = scales_to_test.par_iter().map(|&tau| {
            if tau < 1e-7 { return (1.0, f32::MAX); }
            let s = 1.0 / tau;
            let error: f32 = floats.par_iter().map(|&w| {
                let q = (w * s).round().clamp(-1.0, 1.0);
                let diff = w - (q * tau);
                diff * diff
            }).sum();
            (s, error)
        }).min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        best_candidate.map(|(s, _)| s).unwrap_or(0.0)
    } else {
        0.0
    };

    let scale = optimal_scale * dampening_factor;
    let inv_scale = if scale > 1e-8 { 1.0 / scale } else { 1.0 };

    let packed_u32s: Vec<u32> = floats.par_chunks(16).map(|chunk| {
        let mut packed = 0u32;
        for (i, &w) in chunk.iter().enumerate() {
            let scaled = w * scale;
            let rounded = scaled.round();
            let ternary_bits = if rounded > 0.5 { 1 } else if rounded < -0.5 { 2 } else { 0 };
            packed |= ternary_bits << (i * 2);
        }
        packed
    }).collect();

    let mut out_bytes = Vec::with_capacity(packed_u32s.len() * 4);
    for p in packed_u32s {
        out_bytes.extend_from_slice(&p.to_le_bytes());
    }
    (out_bytes, inv_scale)
}

/// Pack pre-ternarized f32 values (∈ {-1,0,+1}) into 2-bit u8
pub fn pack_ternary_from_f32(ternary: &[f32]) -> Vec<u8> {
    let u32_count = ternary.len().div_ceil(16);
    let mut packed = vec![0u32; u32_count];
    for i in 0..ternary.len() {
        let bit = if ternary[i] > 0.5 { 1u32 } else if ternary[i] < -0.5 { 2u32 } else { 0u32 };
        let u32_idx = i / 16;
        let shift = (i % 16) * 2;
        packed[u32_idx] |= bit << shift;
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
    };
    bytes.to_vec()
}

/// Apply row-wise absmean ternarization to an embedding table.
/// Returns (packed_ternary, per_row_scales_f32, metadata).
pub fn embedding_rowwise_ternarize(emb_data: &[f32], vocab: usize, hidden: usize) -> (Vec<u8>, Vec<f32>, HashMap<String, String>) {
    let mut scales_f32 = Vec::with_capacity(vocab);
    for row in 0..vocab {
        let start = row * hidden;
        let row_slice = &emb_data[start..start + hidden];
        let absmean = row_slice.iter().map(|v| v.abs()).sum::<f32>() / hidden as f32;
        scales_f32.push(absmean.max(1e-10));
    }

    // Ternarize each row: data[row][j] = clamp(round(orig[j] / scale[row]), -1, +1)
    let mut ternary = vec![0.0f32; emb_data.len()];
    for row in 0..vocab {
        let s = scales_f32[row];
        let start = row * hidden;
        for j in 0..hidden {
            ternary[start + j] = (emb_data[start + j] / s).round().clamp(-1.0, 1.0);
        }
    }

    let packed = pack_ternary_from_f32(&ternary);

    let metadata = HashMap::from([
        ("embed_ternarized".to_string(), "row_absmean".to_string()),
    ]);

    (packed, scales_f32, metadata)
}

pub fn convert_to_f32_bytes(tensor: &TensorView) -> Vec<u8> {
    let floats: Vec<f32> = match tensor.dtype() {
        Dtype::F16 => {
            let slice: &[f16] = unsafe { std::slice::from_raw_parts(tensor.data().as_ptr() as *const f16, tensor.data().len() / 2) };
            slice.iter().map(|&x| x.to_f32()).collect()
        },
        Dtype::BF16 => {
            let slice: &[bf16] = unsafe { std::slice::from_raw_parts(tensor.data().as_ptr() as *const bf16, tensor.data().len() / 2) };
            slice.iter().map(|&x| x.to_f32()).collect()
        },
        Dtype::F32 => return tensor.data().to_vec(),
        _ => panic!("Unsupported dtype: {:?}", tensor.dtype()),
    };
    
    let mut out = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}
