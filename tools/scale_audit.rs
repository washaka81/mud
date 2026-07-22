//! scale_audit — P1.1: compare ternary-dequantized .mud weights vs BF16 source RMS.
//!
//! Detects vocabulary-collapse root cause: if the .mud dequantized weight RMS is
//! wildly different (≫ or ≪) from the original BF16 source, the PRQ scale / ternary
//! magnitude is wrong and inference logits will be flat.
//!
//! Usage: scale_audit <model.mud> <source_dir_or.safetensors>

use forge_llm::mud::{dequantize_ternary_row, MudFile, MudTensorType};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

fn source_f32(st: &SafeTensors, name: &str) -> Option<Vec<f32>> {
    let t = st.tensor(name).ok()?;
    let raw = t.data();
    let out = match t.dtype() {
        safetensors::tensor::Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        safetensors::tensor::Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        safetensors::tensor::Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => return None,
    };
    Some(out)
}

fn mud_to_hf(mud_name: &str) -> Option<String> {
    let rest = mud_name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let l = &rest[..dot];
    let tail = &rest[dot + 1..];
    let hf_tail = match tail {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{l}.{hf_tail}"))
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.mud> <source_dir_or.safetensors>", args[0]);
        std::process::exit(2);
    }
    let mud_path = &args[1];
    let src_path = std::path::PathBuf::from(&args[2]);

    let st_file = if src_path.is_dir() {
        src_path.join("model.safetensors")
    } else {
        src_path
    };
    let file = File::open(&st_file)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mud = MudFile::load(mud_path)?;
    let core = mud
        .skills
        .get("core")
        .ok_or_else(|| anyhow::anyhow!("no core skill"))?;

    println!(
        "{:<32} {:>12} {:>12} {:>8}",
        "tensor", "mud_rms", "src_rms", "ratio"
    );
    println!("{}", "-".repeat(68));

    let probe = ["attn_q", "attn_v", "ffn_up", "ffn_down"];
    let layers = [0usize, 15, 29];
    let mut ratios = vec![];

    for &l in &layers {
        for p in &probe {
            let mud_name = format!("blk.{l}.{p}.weight");
            let Some(t) = core.tensors.get(&mud_name) else {
                continue;
            };
            if t.t_type != MudTensorType::Ternary2Bit {
                continue;
            }
            let rows = t.shape[0];
            let cols = t.shape[1];
            let u32s_per_row = cols.div_ceil(8);
            let mut deq = vec![0.0f32; rows * cols];
            let scale_t = core.tensors.get(&format!("blk.{l}.{p}.prq_scale"));
            unsafe {
                for r in 0..rows {
                    dequantize_ternary_row(
                        (t.data_ptr as *const u32).add(r * u32s_per_row),
                        &mut deq[r * cols..(r + 1) * cols],
                        cols,
                    );
                    if let Some(st_t) = scale_t {
                        let s = *(st_t.data_ptr as *const f32).add(r);
                        for c in 0..cols {
                            deq[r * cols + c] *= s;
                        }
                    }
                }
            }
            let mud_rms = rms(&deq);

            let Some(hf) = mud_to_hf(&mud_name) else {
                continue;
            };
            let Some(src) = source_f32(&st, &hf) else {
                println!("{mud_name:<32} {mud_rms:>12.6} {:>12} (src missing)", "-");
                continue;
            };
            let src_rms = rms(&src);
            let ratio = if src_rms > 0.0 {
                mud_rms / src_rms
            } else {
                0.0
            };
            ratios.push(ratio);
            println!("{mud_name:<32} {mud_rms:>12.6} {src_rms:>12.6} {ratio:>8.3}");
        }
    }

    if !ratios.is_empty() {
        let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
        println!("{}", "-".repeat(68));
        println!("mean mud/src RMS ratio = {mean:.3}");
        if !(0.3..=3.0).contains(&mean) {
            println!(
                "VERDICT: SCALE BROKEN (ratio {mean:.3} outside [0.3, 3.0]) — dequant magnitude wrong."
            );
        } else {
            println!("VERDICT: scale within tolerance.");
        }
    }
    Ok(())
}
