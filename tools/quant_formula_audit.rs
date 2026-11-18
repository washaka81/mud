//! Quantization-formula audit for `.mud` weights.
//!
//! Computes, per sampled row, the statistics and reconstruction errors of the
//! ternary formulas discussed in `docs/research/MUD_QUALITY_FORMULAS_2026-11-08.md`:
//!
//! * current converter/checkpoint: `s = absmean/√2`, `δ = 0.7·s`
//! * BitNet b1.58: `γ = absmean`, `s = γ`, `δ = 0.5·γ`
//! * least-squares optimal scale for the codes: `s* = Σ T·W / Σ |T|`
//! * deadzone sweep with scale re-fit (`0.4 … 0.8`)
//!
//! For Ternary2Bit tensors (already quantized) it reports the stored scale and
//! code statistics; there is no full-precision reference to compute MSE against.
//!
//! Usage:
//!   cargo run --release --bin quant_formula_audit -- models/qwen_coder_0.5b.mud
//!   cargo run --release --bin quant_formula_audit -- models/qwen_coder_0.5b.mud \
//!       --tensors token_embd.weight,blk.0.attn_q.weight --max-rows 64 --json

use forge_llm::mud::{MudFile, MudTensor, MudTensorType};
use std::process::ExitCode;

const DEFAULT_TENSORS: &[&str] = &[
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
    "blk.0.attn_q.weight",
    "blk.0.attn_output.weight",
    "blk.0.ffn_up.weight",
    "blk.0.ffn_down.weight",
];

#[derive(Default, Clone, Copy)]
struct RowStats {
    n: usize,
    gamma: f64,
    sigma: f64,
    kurtosis: f64,
    s_cur: f64,
    delta_cur: f64,
    mse_cur: f64,
    s_opt_cur: f64,
    mse_opt_cur: f64,
    snr_cur: f64,
    snr_opt_cur: f64,
    nonzero_cur: f64,
    s_bitnet: f64,
    delta_bitnet: f64,
    mse_bitnet: f64,
    s_opt_bitnet: f64,
    mse_opt_bitnet: f64,
    snr_opt_bitnet: f64,
    nonzero_bitnet: f64,
    best_deadzone: f64,
    best_deadzone_mse: f64,
    // Extensive sums for correct global NMSE/SNR (do not divide in `scale`).
    sig_energy: f64,
    err_cur: f64,
    err_opt_cur: f64,
    err_bitnet: f64,
    err_opt_bitnet: f64,
}

impl RowStats {
    fn add(&mut self, o: &RowStats) {
        self.n += o.n;
        self.gamma += o.gamma;
        self.sigma += o.sigma;
        self.kurtosis += o.kurtosis;
        self.s_cur += o.s_cur;
        self.delta_cur += o.delta_cur;
        self.mse_cur += o.mse_cur;
        self.s_opt_cur += o.s_opt_cur;
        self.mse_opt_cur += o.mse_opt_cur;
        self.snr_cur += o.snr_cur;
        self.snr_opt_cur += o.snr_opt_cur;
        self.nonzero_cur += o.nonzero_cur;
        self.s_bitnet += o.s_bitnet;
        self.delta_bitnet += o.delta_bitnet;
        self.mse_bitnet += o.mse_bitnet;
        self.s_opt_bitnet += o.s_opt_bitnet;
        self.mse_opt_bitnet += o.mse_opt_bitnet;
        self.snr_opt_bitnet += o.snr_opt_bitnet;
        self.nonzero_bitnet += o.nonzero_bitnet;
        self.best_deadzone += o.best_deadzone;
        self.best_deadzone_mse += o.best_deadzone_mse;
        self.sig_energy += o.sig_energy;
        self.err_cur += o.err_cur;
        self.err_opt_cur += o.err_opt_cur;
        self.err_bitnet += o.err_bitnet;
        self.err_opt_bitnet += o.err_opt_bitnet;
    }

    fn scale(&self, rows: usize) -> RowStats {
        let r = rows.max(1) as f64;
        RowStats {
            n: self.n,
            gamma: self.gamma / r,
            sigma: self.sigma / r,
            kurtosis: self.kurtosis / r,
            s_cur: self.s_cur / r,
            delta_cur: self.delta_cur / r,
            mse_cur: self.mse_cur / r,
            s_opt_cur: self.s_opt_cur / r,
            mse_opt_cur: self.mse_opt_cur / r,
            snr_cur: self.snr_cur / r,
            snr_opt_cur: self.snr_opt_cur / r,
            nonzero_cur: self.nonzero_cur / r,
            s_bitnet: self.s_bitnet / r,
            delta_bitnet: self.delta_bitnet / r,
            mse_bitnet: self.mse_bitnet / r,
            s_opt_bitnet: self.s_opt_bitnet / r,
            mse_opt_bitnet: self.mse_opt_bitnet / r,
            snr_opt_bitnet: self.snr_opt_bitnet / r,
            nonzero_bitnet: self.nonzero_bitnet / r,
            best_deadzone: self.best_deadzone / r,
            best_deadzone_mse: self.best_deadzone_mse / r,
            // Extensive energies stay summed across rows.
            sig_energy: self.sig_energy,
            err_cur: self.err_cur,
            err_opt_cur: self.err_opt_cur,
            err_bitnet: self.err_bitnet,
            err_opt_bitnet: self.err_opt_bitnet,
        }
    }
}

/// Read one f32 from a tensor's live storage without assuming alignment.
///
/// # Safety
/// `t` must be a Float32 tensor whose `checked_data_size()` covers `idx*4+4`.
unsafe fn f32_at(t: &MudTensor, idx: usize) -> f32 {
    let p = t
        .owned_data
        .as_deref()
        .map(|d| d.as_ptr())
        .unwrap_or(t.data_ptr);
    unsafe { std::ptr::read_unaligned(p.add(idx * 4) as *const f32) }
}

/// Read one u32 from a ternary tensor's packed storage.
///
/// # Safety
/// `t` must be a Ternary2Bit tensor whose `checked_data_size()` covers `idx*4+4`.
unsafe fn u32_at(t: &MudTensor, idx: usize) -> u32 {
    let p = t
        .owned_data
        .as_deref()
        .map(|d| d.as_ptr())
        .unwrap_or(t.data_ptr);
    unsafe { std::ptr::read_unaligned(p.add(idx * 4) as *const u32) }
}

fn tensor_f32(t: &MudTensor) -> bool {
    t.t_type == MudTensorType::Float32
}

fn row_f32(t: &MudTensor, row: usize, cols: usize) -> Vec<f32> {
    (0..cols)
        // SAFETY: caller checked shape [rows, cols] and the tensor is Float32;
        // `row*cols + c < rows*cols`, covered by checked_data_size.
        .map(|c| unsafe { f32_at(t, row * cols + c) })
        .collect()
}

fn row_ternary(t: &MudTensor, scale: f32, row: usize, cols: usize) -> Vec<f32> {
    let row_words = cols.div_ceil(8);
    let mut out = Vec::with_capacity(cols);
    for c in 0..cols {
        // SAFETY: caller checked the packed layout rows*div_ceil(cols,8) u32 and
        // `row < rows`, `c < cols`.
        let w = unsafe { u32_at(t, row * row_words + c / 8) };
        let code = ((w >> ((c % 8) * 4)) & 0xF) as usize;
        let v = crate::ternary_code_to_f32(code);
        out.push(v * scale);
    }
    out
}

// Small local copy so the tool does not depend on a private helper.
fn ternary_code_to_f32(code: usize) -> f32 {
    match code {
        1 => 1.0,
        0xF => -1.0,
        _ => 0.0,
    }
}

fn quantize_row(w: &[f32], s: f64, delta_ratio: f64) -> (Vec<i8>, usize) {
    let d = delta_ratio * s;
    let mut codes = Vec::with_capacity(w.len());
    let mut nz = 0usize;
    for &v in w {
        let v = v as f64;
        let c = if v > d {
            1i8
        } else if v < -d {
            -1i8
        } else {
            0i8
        };
        if c != 0 {
            nz += 1;
        }
        codes.push(c);
    }
    (codes, nz)
}

fn optimal_scale(w: &[f32], codes: &[i8]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&v, &c) in w.iter().zip(codes.iter()) {
        let c = c as f64;
        num += c * v as f64;
        den += c.abs();
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

fn mse(w: &[f32], codes: &[i8], s: f64) -> f64 {
    let mut e = 0.0f64;
    for (&v, &c) in w.iter().zip(codes.iter()) {
        let d = v as f64 - s * c as f64;
        e += d * d;
    }
    e / w.len().max(1) as f64
}

fn db(sig: f64, err: f64) -> f64 {
    if err > 0.0 && sig > 0.0 {
        10.0 * (sig / err).log10()
    } else if sig > 0.0 {
        120.0
    } else {
        0.0
    }
}

fn snr_db(w: &[f32], codes: &[i8], s: f64) -> f64 {
    let mut sig = 0.0f64;
    let mut err = 0.0f64;
    for (&v, &c) in w.iter().zip(codes.iter()) {
        let v = v as f64;
        let d = v - s * c as f64;
        sig += v * v;
        err += d * d;
    }
    if err > 0.0 && sig > 0.0 {
        10.0 * (sig / err).log10()
    } else if sig > 0.0 {
        120.0
    } else {
        0.0
    }
}

fn analyze_row(w: &[f32]) -> RowStats {
    let n = w.len().max(1) as f64;
    let gamma = w.iter().map(|v| (*v as f64).abs()).sum::<f64>() / n;
    let mean = w.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = w
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let sigma = var.sqrt();
    let m4 = w
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d * d * d
        })
        .sum::<f64>()
        / n;
    let kurtosis = if var > 1e-30 { m4 / (var * var) - 3.0 } else { 0.0 };

    // Current formula: s = absmean/√2, δ = 0.7·s.
    let s_cur = gamma * std::f64::consts::FRAC_1_SQRT_2;
    let delta_cur = 0.7;
    let (codes_cur, nz_cur) = quantize_row(w, s_cur, delta_cur);
    let mse_cur = mse(w, &codes_cur, s_cur);
    let s_opt_cur = optimal_scale(w, &codes_cur);
    let mse_opt_cur = mse(w, &codes_cur, s_opt_cur);

    // BitNet b1.58: s = absmean, δ = 0.5·s.
    let s_bit = gamma;
    let (codes_bit, nz_bit) = quantize_row(w, s_bit, 0.5);
    let mse_bit = mse(w, &codes_bit, s_bit);
    let s_opt_bit = optimal_scale(w, &codes_bit);
    let mse_opt_bit = mse(w, &codes_bit, s_opt_bit);

    // Deadzone sweep, re-fitting the scale for each code set.
    let mut best_dz = 0.5;
    let mut best_mse = f64::INFINITY;
    for &dz in &[0.4, 0.5, 0.6, 0.7, 0.8] {
        let (codes, _) = quantize_row(w, s_cur, dz);
        let s_opt = optimal_scale(w, &codes);
        let e = mse(w, &codes, s_opt);
        if e < best_mse {
            best_mse = e;
            best_dz = dz;
        }
    }

    RowStats {
        n: w.len(),
        gamma,
        sigma,
        kurtosis,
        s_cur,
        delta_cur: delta_cur * s_cur,
        mse_cur,
        s_opt_cur,
        mse_opt_cur,
        snr_cur: snr_db(w, &codes_cur, s_cur),
        snr_opt_cur: snr_db(w, &codes_cur, s_opt_cur),
        nonzero_cur: nz_cur as f64 / n,
        s_bitnet: s_bit,
        delta_bitnet: 0.5 * s_bit,
        mse_bitnet: mse_bit,
        s_opt_bitnet: s_opt_bit,
        mse_opt_bitnet: mse_opt_bit,
        snr_opt_bitnet: snr_db(w, &codes_bit, s_opt_bit),
        nonzero_bitnet: nz_bit as f64 / n,
        best_deadzone: best_dz,
        best_deadzone_mse: best_mse,
        sig_energy: w.iter().map(|v| (*v as f64) * (*v as f64)).sum(),
        err_cur: mse_cur * n,
        err_opt_cur: mse_opt_cur * n,
        err_bitnet: mse_bit * n,
        err_opt_bitnet: mse_opt_bit * n,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "Usage: quant_formula_audit <model.mud> [--tensors a,b,c] [--max-rows N] [--json]"
        );
        return ExitCode::from(2);
    }
    let model_path = &args[0];
    let mut tensors: Vec<String> = DEFAULT_TENSORS.iter().map(|s| s.to_string()).collect();
    let mut max_rows = 128usize;
    let mut json = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tensors" => {
                if let Some(v) = args.get(i + 1) {
                    tensors = v.split(',').map(|s| s.trim().to_string()).collect();
                    i += 1;
                }
            }
            "--max-rows" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_rows = v;
                    i += 1;
                }
            }
            "--json" => json = true,
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    let mud = match MudFile::load(model_path) {
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

    println!("[quant-formula] model={model_path} max_rows={max_rows}");
    println!(
        "formula: current s=absmean/sqrt2, delta=0.7*s | BitNet s=absmean, delta=0.5*s | s*=(sum T*W)/(sum |T|)"
    );

    let mut global = RowStats::default();
    let mut global_rows = 0usize;

    for name in &tensors {
        let Some(t) = core.tensors.get(name) else {
            println!("  {name:38} : MISSING");
            continue;
        };
        let shape = &t.shape;
        if shape.len() != 2 {
            println!("  {name:38} : skip (rank {})", shape.len());
            continue;
        }
        let rows = shape[0];
        let cols = shape[1];
        let sample_rows = rows.min(max_rows.max(1));

        if tensor_f32(t) {
            let mut agg = RowStats::default();
            for r in 0..sample_rows {
                let w = row_f32(t, r, cols);
                agg.add(&analyze_row(&w));
            }
            global.add(&agg);
            global_rows += sample_rows;
            let m = agg.scale(sample_rows);
            println!(
                "  {name:38} F32  rows={sample_rows}/{rows} cols={cols} kurt={:.2} s_cur={:.5} s*={:.5} ratio={:.3} mse_cur={:.3e} mse*={:.3e} nmse_cur={:.4} nmse*={:.4} snr_cur={:.2}dB snr*={:.2}dB nz_cur={:.2} nz_bit={:.2} best_dz={:.1}",
                m.kurtosis,
                m.s_cur,
                m.s_opt_cur,
                if m.s_cur > 0.0 { m.s_opt_cur / m.s_cur } else { 0.0 },
                m.mse_cur,
                m.mse_opt_cur,
                m.err_cur / m.sig_energy.max(1e-30),
                m.err_opt_cur / m.sig_energy.max(1e-30),
                db(m.sig_energy, m.err_cur),
                db(m.sig_energy, m.err_opt_cur),
                m.nonzero_cur,
                m.nonzero_bitnet,
                m.best_deadzone,
            );
        } else if t.t_type == MudTensorType::Ternary2Bit {
            let scale_name = name.replace(".weight", ".prq_scale");
            let Some(st) = core.tensors.get(&scale_name) else {
                println!("  {name:38} : ternary, missing {scale_name}");
                continue;
            };
            let sample_rows = rows.min(max_rows.max(1));
            let mut nz = 0.0f64;
            let mut scale_sum = 0.0f64;
            let mut w_sq = 0.0f64;
            let mut count = 0usize;
            for r in 0..sample_rows {
                // SAFETY: Float32 scale tensor, checked shape [rows].
                let s = unsafe { f32_at(st, r.min(st.shape[0].saturating_sub(1))) };
                scale_sum += s as f64;
                let w = row_ternary(t, s, r, cols);
                for v in &w {
                    if *v != 0.0 {
                        nz += 1.0;
                    }
                    w_sq += (*v as f64) * (*v as f64);
                    count += 1;
                }
            }
            let n = sample_rows.max(1) as f64;
            println!(
                "  {name:38} T2B  rows={sample_rows}/{rows} cols={cols} scale_mean={:.5} w_rms={:.5} nonzero={:.2} (dequantized; no F32 reference)",
                scale_sum / n,
                (w_sq / count.max(1) as f64).sqrt(),
                nz / count.max(1) as f64,
            );
        } else {
            println!("  {name:38} : skip type {:?}", t.t_type);
        }
    }

    if global_rows > 0 {
        let g = global.scale(global_rows);
        println!(
            "\nGLOBAL F32 sample rows={global_rows} kurt={:.2} s_cur={:.5} s*_cur={:.5} ratio={:.3} mse_cur={:.3e} mse*_cur={:.3e} mse_bitnet={:.3e} mse*_bitnet={:.3e} nmse_cur={:.4} nmse*_cur={:.4} snr_cur={:.2}dB snr*_cur={:.2}dB nz_cur={:.2} nz_bitnet={:.2} best_dz={:.1}",
            g.kurtosis,
            g.s_cur,
            g.s_opt_cur,
            if g.s_cur > 0.0 { g.s_opt_cur / g.s_cur } else { 0.0 },
            g.mse_cur,
            g.mse_opt_cur,
            g.mse_bitnet,
            g.mse_opt_bitnet,
            g.err_cur / g.sig_energy.max(1e-30),
            g.err_opt_cur / g.sig_energy.max(1e-30),
            db(g.sig_energy, g.err_cur),
            db(g.sig_energy, g.err_opt_cur),
            g.nonzero_cur,
            g.nonzero_bitnet,
            g.best_deadzone,
        );

        if json {
            println!(
                "{{\"model\":\"{}\",\"rows\":{},\"kurtosis\":{:.6},\"s_cur\":{:.6},\"s_opt_cur\":{:.6},\"scale_ratio\":{:.6},\"mse_cur\":{:.6e},\"mse_opt_cur\":{:.6e},\"mse_bitnet\":{:.6e},\"mse_opt_bitnet\":{:.6e},\"nmse_cur\":{:.6},\"nmse_opt_cur\":{:.6},\"snr_cur_db\":{:.4},\"snr_opt_cur_db\":{:.4},\"nonzero_cur\":{:.6},\"nonzero_bitnet\":{:.6},\"best_deadzone\":{:.2}}}",
                model_path.replace('\\', "\\\\").replace('"', "\\\""),
                global_rows,
                g.kurtosis,
                g.s_cur,
                g.s_opt_cur,
                if g.s_cur > 0.0 { g.s_opt_cur / g.s_cur } else { 0.0 },
                g.mse_cur,
                g.mse_opt_cur,
                g.mse_bitnet,
                g.mse_opt_bitnet,
                g.err_cur / g.sig_energy.max(1e-30),
                g.err_opt_cur / g.sig_energy.max(1e-30),
                db(g.sig_energy, g.err_cur),
                db(g.sig_energy, g.err_opt_cur),
                g.nonzero_cur,
                g.nonzero_bitnet,
                g.best_deadzone,
            );
        }
    }

    ExitCode::SUCCESS
}
