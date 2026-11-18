//! GEMV parity: real `.mud` ternary tensor vs scalar dequantized reference.
//!
//! Compares, for one tensor of a real model:
//!   - `asm::ternary_gemv` (single row, PRQ scale applied)
//!   - `asm::ternary_gemv_8rows` (8 rows, scale=1) + per-row PRQ scale
//!
//! against a scalar `sum(x[i] * dequant(w)[i])` reference.
//!
//! Reports max abs error, RMS error, cosine similarity and top-1 row.
//!
//! Usage:
//!   cargo run --release --bin gemv_parity -- models/qwen_coder_0.5b_normfix.mud blk.0.attn_q.weight
//!   cargo run --release --bin gemv_parity -- models/ale.mud blk.0.attn_q.weight

use forge_llm::asm;
use forge_llm::mud::slime_backward::unpack_ternary2bit_to_f32;
use forge_llm::mud::{MudFile, MudTensorType};
use std::process::ExitCode;

fn ptr_of(t: &forge_llm::mud::MudTensor) -> *const u8 {
    t.owned_data
        .as_ref()
        .map(|d| d.as_ptr())
        .unwrap_or(t.data_ptr)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(model_path) = args.next() else {
        eprintln!("Usage: gemv_parity <model.mud> <tensor_name>");
        return ExitCode::from(2);
    };
    let tensor_name = args
        .next()
        .unwrap_or_else(|| "blk.0.attn_q.weight".to_string());

    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::from(1);
        }
    };
    let Some(core) = mud.skills.get("core") else {
        eprintln!("missing core skill");
        return ExitCode::from(1);
    };
    let Some(w) = core.tensors.get(&tensor_name) else {
        eprintln!("missing tensor {tensor_name}");
        return ExitCode::from(1);
    };
    if w.t_type != MudTensorType::Ternary2Bit {
        eprintln!("tensor is {:?}, expected Ternary2Bit", w.t_type);
        return ExitCode::from(1);
    }
    let scale_name = tensor_name.replace(".weight", ".prq_scale");
    let Some(s) = core.tensors.get(&scale_name) else {
        eprintln!("missing scales {scale_name}");
        return ExitCode::from(1);
    };
    let (n_out, n_in) = match w.shape.as_slice() {
        [r, c] => (*r, *c),
        other => {
            eprintln!("expected rank-2 tensor, got {other:?}");
            return ExitCode::from(1);
        }
    };
    let row_u32s = n_in.div_ceil(8);
    let packed_len = n_out * row_u32s * 4;
    let packed: &[u8] = unsafe { std::slice::from_raw_parts(ptr_of(w), packed_len) };
    let scales: &[f32] =
        unsafe { std::slice::from_raw_parts(ptr_of(s) as *const f32, n_out) };

    // Deterministic smooth input (same vector for every row).
    let x: Vec<f32> = (0..n_in)
        .map(|i| {
            let t = i as f32 * 0.017;
            t.sin() * 0.7 + (i % 7) as f32 * 0.01
        })
        .collect();

    // Reference: dequantize all rows and dot with x.
    let mut dequant = vec![0.0f32; n_out * n_in];
    unpack_ternary2bit_to_f32(packed, scales, n_in, &mut dequant);
    let mut reference = vec![0.0f32; n_out];
    for r in 0..n_out {
        let row = &dequant[r * n_in..(r + 1) * n_in];
        reference[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }

    // Single-row ASM path with PRQ scale.
    let mut asm1 = vec![0.0f32; n_out];
    unsafe {
        for (r, &scale) in scales.iter().enumerate().take(n_out) {
            asm::ternary_gemv(
                n_in,
                x.as_ptr(),
                packed.as_ptr().cast::<u32>().add(r * row_u32s),
                asm1.as_mut_ptr().add(r),
                scale,
            );
        }
    }

    // 8-row ASM path (scale=1) + per-row scale.
    let mut asm8 = vec![0.0f32; n_out];
    unsafe {
        let mut r = 0;
        while r + 8 <= n_out {
            asm::ternary_gemv_8rows(
                n_in,
                x.as_ptr(),
                packed.as_ptr().cast::<u32>().add(r * row_u32s),
                asm8.as_mut_ptr().add(r),
                1.0,
                row_u32s,
            );
            for k in 0..8 {
                asm8[r + k] *= scales[r + k];
            }
            r += 8;
        }
        while r < n_out {
            asm::ternary_gemv(
                n_in,
                x.as_ptr(),
                packed.as_ptr().cast::<u32>().add(r * row_u32s),
                asm8.as_mut_ptr().add(r),
                scales[r],
            );
            r += 1;
        }
    }

    let report = |label: &str, got: &[f32]| {
        let mut max_abs = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (r, &g) in got.iter().enumerate() {
            let e = (g - reference[r]).abs();
            max_abs = max_abs.max(e);
            sum_sq += (e as f64) * (e as f64);
            ref_sq += (reference[r] as f64) * (reference[r] as f64);
        }
        let rms = (sum_sq / got.len() as f64).sqrt();
        let rel = rms / (ref_sq / got.len() as f64).sqrt().max(1e-30);
        let cos = cosine(got, &reference);
        println!(
            "  {label:12} max_abs={max_abs:.6e} rms={rms:.6e} rel_rms={rel:.6e} cosine={cos:.9}"
        );
    };

    println!("[gemv-parity] {tensor_name}: n_out={n_out} n_in={n_in} row_u32s={row_u32s}");
    println!("  reference   top1_row={}", argmax(&reference));
    println!("  asm1        top1_row={}", argmax(&asm1));
    println!("  asm8        top1_row={}", argmax(&asm8));
    report("asm1_vs_ref", &asm1);
    report("asm8_vs_ref", &asm8);
    let cos18 = cosine(&asm1, &asm8);
    println!("  asm1_vs_asm8 cosine={cos18:.9}");
    println!(
        "  first8_ref   {:?}",
        &reference[..8.min(reference.len())]
    );
    println!("  first8_asm1  {:?}", &asm1[..8.min(asm1.len())]);
    println!("  first8_asm8  {:?}", &asm8[..8.min(asm8.len())]);

    ExitCode::SUCCESS
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}
