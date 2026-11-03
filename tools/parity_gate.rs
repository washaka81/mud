//! parity_gate.rs — Numerical parity gate vs reference GGUF/Safetensors
//! R-50: Gate de Paridad Numérica

use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, Color, Table};
use forge_llm::mud::{dequantize_ternary_row, MudFile, MudTensorType};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::fs::File;

const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";
const YELLOW: &str = "\x1b[33m";

fn dequantize_tensor_prq(
    t: &forge_llm::mud::MudTensor,
    scales: Option<&forge_llm::mud::MudTensor>,
) -> Vec<f32> {
    let elements: usize = t.shape.iter().product();
    let mut data = vec![0.0f32; elements];
    unsafe {
        if t.t_type == MudTensorType::Ternary2Bit {
            let cols = *t.shape.last().unwrap_or(&elements);
            let rows = elements / cols;
            for r in 0..rows {
                let ptr = (t.data_ptr as *const u32).add(r * cols / 8);
                dequantize_ternary_row(ptr, &mut data[r * cols..(r + 1) * cols], cols);
            }
            if let Some(st) = scales {
                let sp = st.data_ptr as *const f32;
                let is_scalar = st.shape.iter().product::<usize>() == 1;
                for r in 0..rows {
                    let s = if is_scalar { *sp } else { *sp.add(r) };
                    for j in 0..cols {
                        data[r * cols + j] *= s;
                    }
                }
            }
        } else if t.t_type == MudTensorType::Float32 {
            std::ptr::copy_nonoverlapping(t.data_ptr as *const f32, data.as_mut_ptr(), elements);
        } else {
            // Try to copy as f32 anyway
            std::ptr::copy_nonoverlapping(t.data_ptr as *const f32, data.as_mut_ptr(), elements.min(t.shape.iter().product()));
        }
    }
    data
}

fn convert_view_to_f32(tensor: &TensorView) -> Vec<f32> {
    match tensor.dtype() {
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
        Dtype::F32 => {
            let slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f32,
                    tensor.data().len() / 4,
                )
            };
            slice.to_vec()
        }
        _ => tensor.data().iter().map(|&b| b as f32).collect(),
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let len = a.len().min(b.len());
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na.sqrt() * nb.sqrt() < 1e-12 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn main() -> anyhow::Result<()> {
    println!(
        "{}========================================================{}",
        BOLD, RESET
    );
    println!(
        "{}🧮   MUD PARITY GATE — Numerical Parity vs Reference      🧮{}",
        BOLD, RESET
    );
    println!(
        "{}========================================================{}",
        BOLD, RESET
    );

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("{}Usage:{} {} <ref.safetensors|ref.gguf> <model.mud> [--threshold 0.99]", BOLD, RESET, args[0]);
        std::process::exit(1);
    }
    let ref_path = &args[1];
    let mud_path = &args[2];

    println!("{}   📂 Reference: {}{}", CYAN, ref_path, RESET);
    println!("{}   📂 MUD Model: {}{}", CYAN, mud_path, RESET);

    // Load MUD
    let mud = MudFile::load(mud_path)?;
    println!("{}   ✅ MUD loaded{}", GREEN, RESET);

    // Load reference (safetensors for now)
    let file = File::open(ref_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let safe_tensors = SafeTensors::deserialize(&mmap)?;
    println!("{}   ✅ Reference loaded ({} tensors){}", GREEN, safe_tensors.tensors().len(), RESET);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            Cell::new("Tensor"),
            Cell::new("Cosine"),
            Cell::new("MSE"),
            Cell::new("Status"),
        ]);

    let mut total_cos = 0.0f32;
    let mut count = 0;
    let mut passed = 0;

    // Check a few key tensors
    for (name, view) in safe_tensors.tensors() {
        if name.contains("embed") || name.contains("lm_head") || name.contains("output") {
            let ref_data = convert_view_to_f32(&view);
            // Try to find matching tensor in MUD
            // Look up by various mappings
            let mud_name = if name.contains("embed_tokens") {
                "token_embd.weight"
            } else if name.contains("lm_head") || name == "lm_head.weight" {
                "output.weight"
            } else {
                continue;
            };
            if let Some(mt) = mud.skills.values().next().and_then(|s| s.tensors.get(mud_name)) {
                let scale_name = mud_name.replace(".weight", ".prq_scale");
                let scales = mud.skills.values().next().and_then(|s| s.tensors.get(&scale_name));
                let mud_data = dequantize_tensor_prq(mt, scales);
                let c = cosine_sim(&ref_data, &mud_data);
                let mut mse = 0.0f32;
                let len = ref_data.len().min(mud_data.len());
                for i in 0..len {
                    let d = ref_data[i] - mud_data[i];
                    mse += d * d;
                }
                mse /= len as f32;
                total_cos += c;
                count += 1;
                if c > 0.99 {
                    passed += 1;
                    table.add_row(vec![
                        Cell::new(mud_name),
                        Cell::new(format!("{:.6}", c)).fg(Color::Green),
                        Cell::new(format!("{:.6e}", mse)),
                        Cell::new("PASS").fg(Color::Green),
                    ]);
                } else {
                    table.add_row(vec![
                        Cell::new(mud_name),
                        Cell::new(format!("{:.6}", c)).fg(Color::Red),
                        Cell::new(format!("{:.6e}", mse)),
                        Cell::new("FAIL").fg(Color::Red),
                    ]);
                }
            }
        }
    }

    println!("{}", table);
    if count > 0 {
        println!("\n{}Mean cosine: {:.6} | Passed: {}/{}{}", 
            if passed == count { GREEN } else { YELLOW }, total_cos / count as f32, passed, count, RESET);
    }
    if passed == count && count > 0 {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
