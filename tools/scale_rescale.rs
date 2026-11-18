//! Multiply every PRQ scale by a constant and save a new `.mud`.
//!
//! Diagnostic for the ternary scale-bias experiments in
//! `docs/research/MUD_QUALITY_FORMULAS_2026-11-07.md`:
//! the current converter uses `s = absmean/√2`, while BitNet b1.58 uses
//! `s = absmean` and the least-squares optimum is higher still. This tool
//! changes only the scales (not the trit codes), so it isolates the effect of
//! the reconstruction magnitude.
//!
//! Usage:
//!   cargo run --release --bin scale_rescale -- in.mud out.mud 1.4142
//!   cargo run --release --bin scale_rescale -- in.mud out.mud 1.889

use forge_llm::mud::{MudFile, MudTensorType};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("Usage: scale_rescale <in.mud> <out.mud> <factor>");
        return ExitCode::from(2);
    }
    let factor: f32 = match args[2].parse::<f32>() {
        Ok(v) if v.is_finite() && v > 0.0 => v,
        _ => {
            eprintln!("factor must be finite and > 0");
            return ExitCode::from(2);
        }
    };
    let mut mud = match MudFile::load(&args[0]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::from(1);
        }
    };
    let Some(core) = mud.skills.get_mut("core") else {
        eprintln!("missing core skill");
        return ExitCode::from(1);
    };

    let mut count = 0usize;
    let mut min_before = f32::INFINITY;
    let mut max_before = f32::NEG_INFINITY;
    let mut sum_before = 0.0f64;
    let mut min_after = f32::INFINITY;
    let mut max_after = f32::NEG_INFINITY;
    let mut sum_after = 0.0f64;

    for (name, t) in core.tensors.iter_mut() {
        if !name.ends_with(".prq_scale") || t.t_type != MudTensorType::Float32 {
            continue;
        }
        let n = t.shape.iter().product::<usize>();
        if n == 0 {
            continue;
        }
        // Read the current bytes from owned_data or the mmap-backed pointer.
        let src_ptr = t
            .owned_data
            .as_deref()
            .map(|d| d.as_ptr())
            .unwrap_or(t.data_ptr);
        if src_ptr.is_null() {
            continue;
        }
        let mut out = Vec::with_capacity(n * 4);
        for i in 0..n {
            // SAFETY: Float32 scale tensor; `checked_data_size()` is n*4 and
            // `i < n`; read_unaligned avoids alignment assumptions on &[u8].
            let v = unsafe { std::ptr::read_unaligned(src_ptr.add(i * 4) as *const f32) };
            if v.is_finite() {
                min_before = min_before.min(v);
                max_before = max_before.max(v);
                sum_before += v as f64;
            }
            let nv = v * factor;
            if nv.is_finite() {
                min_after = min_after.min(nv);
                max_after = max_after.max(nv);
                sum_after += nv as f64;
            }
            out.extend_from_slice(&nv.to_le_bytes());
        }
        t.owned_data = Some(out);
        t.data_ptr = std::ptr::null();
        t.data_base = 0;
        count += 1;
    }

    if count == 0 {
        eprintln!("no PRQ scale tensors found");
        return ExitCode::from(1);
    }

    let n_vals: usize = core
        .tensors
        .iter()
        .filter(|(name, t)| name.ends_with(".prq_scale") && t.t_type == MudTensorType::Float32)
        .map(|(_, t)| t.shape.iter().product::<usize>())
        .sum();

    if let Err(e) = mud.save(&args[1]) {
        eprintln!("save failed: {e}");
        return ExitCode::from(1);
    }
    println!(
        "[scale-rescale] factor={factor:.6} tensors={count} values={n_vals} \
         before[min={min_before:.6} max={max_before:.6} mean={:.6}] \
         after[min={min_after:.6} max={max_after:.6} mean={:.6}] -> {}",
        sum_before / n_vals.max(1) as f64,
        sum_after / n_vals.max(1) as f64,
        args[1]
    );
    ExitCode::SUCCESS
}
