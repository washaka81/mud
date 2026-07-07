use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/core_skills.mud");
    println!("=== MUD Weight & Sigma Audit: {} ===", model_path);

    let model = MudFile::load(model_path)?;
    let core = model.skills.get("core").expect("No core skill");

    for (name, tensor) in &core.tensors {
        if tensor.t_type == forge_llm::mud::MudTensorType::Ternary2Bit {
            // Read packed u32 values
            let n_elements = tensor.shape.iter().copied().product::<usize>();
            let n_u32 = n_elements.div_ceil(8);
            let data_ptr = tensor.data_ptr as *const u32;
            let packed_data = unsafe { std::slice::from_raw_parts(data_ptr, n_u32) };

            let mut counts = [0usize; 3]; // 0: 0, 1: +1, 2: -1
            let mut element_count = 0;
            for &val in packed_data {
                for i in 0..8 {
                    if element_count >= n_elements {
                        break;
                    }
                    let bits = (val >> (i * 4)) & 0xF;
                    if bits == 0x1 {
                        counts[1] += 1;
                    } else if bits == 0xF {
                        counts[2] += 1;
                    } else {
                        counts[0] += 1;
                    }
                    element_count += 1;
                }
            }

            let total = counts[0] + counts[1] + counts[2];
            let fill_rate = (counts[1] + counts[2]) as f32 / total as f32;

            let n = total as f32;
            let mean = (counts[1] as f32 - counts[2] as f32) / n;
            let variance = (counts[1] as f32 * (1.0 - mean).powi(2)
                + counts[2] as f32 * (-1.0 - mean).powi(2)
                + counts[0] as f32 * (0.0 - mean).powi(2))
                / n;
            let sigma = variance.sqrt();

            println!(
                "{:<40} | Sigma: {:.4} | Fill: {:.1}% | Pos: {:>5} | Neg: {:>5} | Zero: {:>5}",
                name,
                sigma,
                fill_rate * 100.0,
                counts[1],
                counts[2],
                counts[0]
            );
        }
    }

    Ok(())
}
