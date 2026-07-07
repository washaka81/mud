use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "models/core_skills.mud"
    };
    println!("=== MUD ADVANCED STATISTICAL AUDIT: {} ===", model_path);

    let model = MudFile::load(model_path)?;
    let core = model.skills.get("core").expect("No core skill found");

    for (name, tensor) in &core.tensors {
        if tensor.t_type == forge_llm::mud::MudTensorType::Ternary2Bit {
            let n_elements = tensor.shape.iter().copied().product::<usize>();
            let n_u32 = n_elements.div_ceil(16);
            let data_ptr = tensor.data_ptr as *const u32;
            let packed_data = unsafe { std::slice::from_raw_parts(data_ptr, n_u32) };

            let mut counts = [0usize; 3]; // 0: 0, 1: +1, 2: -1
            for &val in packed_data {
                for i in 0..16 {
                    let bits = (val >> (i * 2)) & 3;
                    if bits == 1 {
                        counts[1] += 1;
                    } else if bits == 2 {
                        counts[2] += 1;
                    } else {
                        counts[0] += 1;
                    }
                }
            }

            let n = (counts[0] + counts[1] + counts[2]) as f32;

            // 1. Mean (Expectation) - Should be close to 0 for balanced weights
            let mean = (counts[1] as f32 - counts[2] as f32) / n;

            // 2. Variance and Sigma
            let variance = (counts[1] as f32 * (1.0 - mean).powi(2)
                + counts[2] as f32 * (-1.0 - mean).powi(2)
                + counts[0] as f32 * (0.0 - mean).powi(2))
                / n;
            let sigma = variance.sqrt();

            // 2.5. Spectral Norm (Largest Singular Value) - ODA-02
            // Estimated via 10 power iterations on the dequantized matrix (W = Q * S)
            let rows = tensor.shape[0];
            let cols = tensor.shape[1];
            let mut u = vec![0.0f32; rows];
            u.fill(0.5);

            // Get scales if available
            let mut row_scales = vec![1.0f32; rows];
            let scale_names = vec![
                format!("{}.prq_scale", name),
                name.replace(".weight", ".prq_scale"),
                name.replace(".weight", ".scale"),
            ];

            for s_name in scale_names {
                if let Some(scale_tensor) = core.tensors.get(&s_name) {
                    if scale_tensor.t_type == forge_llm::mud::MudTensorType::Float32 {
                        let n_scales = scale_tensor.shape.iter().product::<usize>();
                        if let Some(data) = &scale_tensor.owned_data {
                            for (i, chunk) in data.chunks_exact(4).enumerate().take(n_scales) {
                                row_scales[i] =
                                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                            }
                        } else if !scale_tensor.data_ptr.is_null() {
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    scale_tensor.data_ptr as *const f32,
                                    row_scales.as_mut_ptr(),
                                    n_scales,
                                );
                            }
                        }
                        break;
                    }
                }
            }

            let mut spectral_norm = 0.0;
            for _ in 0..10 {
                let mut v = vec![0.0f32; cols];
                // v = W^T * u = (Q * S)^T * u = S * Q^T * u
                for r in 0..rows {
                    let mut row_f32 = vec![0.0f32; cols];
                    unsafe {
                        let row_ptr = (tensor.data_ptr as *const u32).add(r * (cols.div_ceil(16)));
                        forge_llm::mud::dequantize_ternary_row(row_ptr, &mut row_f32, cols);
                    }
                    let u_val = u[r] * row_scales[r];
                    for c in 0..cols {
                        v[c] += row_f32[c] * u_val;
                    }
                }
                let v_norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-8);
                for x in &mut v {
                    *x /= v_norm;
                }

                let mut next_u = vec![0.0f32; rows];
                // u = W * v = Q * S * v
                for r in 0..rows {
                    let mut row_f32 = vec![0.0f32; cols];
                    unsafe {
                        let row_ptr = (tensor.data_ptr as *const u32).add(r * (cols.div_ceil(16)));
                        forge_llm::mud::dequantize_ternary_row(row_ptr, &mut row_f32, cols);
                    }
                    let mut sum = 0.0;
                    for c in 0..cols {
                        sum += row_f32[c] * v[c];
                    }
                    next_u[r] = sum * row_scales[r];
                }
                spectral_norm = (next_u.iter().map(|x| x * x).sum::<f32>()).sqrt();
                for r in 0..rows {
                    u[r] = next_u[r] / spectral_norm.max(1e-8);
                }
            }

            // 3. Skewness (Asimetría) — Measures lack of symmetry
            // Standardized 3rd moment; guard against sigma=0
            let skewness = if sigma > 0.0 {
                (counts[1] as f32 * (1.0 - mean).powi(3)
                    + counts[2] as f32 * (-1.0 - mean).powi(3)
                    + counts[0] as f32 * (0.0 - mean).powi(3))
                    / (n * sigma.powi(3))
            } else {
                0.0
            };

            // 4. Kurtosis (Curtosis) — Measures thickness of tails (outliers)
            // Standardized 4th moment - 3 (Excess Kurtosis); guard against sigma=0
            let kurtosis = if sigma > 0.0 {
                (counts[1] as f32 * (1.0 - mean).powi(4)
                    + counts[2] as f32 * (-1.0 - mean).powi(4)
                    + counts[0] as f32 * (0.0 - mean).powi(4))
                    / (n * sigma.powi(4))
                    - 3.0
            } else {
                0.0
            };

            println!(
                "{:<35} | Sigma: {:.4} | Spec: {:>6.2} | AvgS: {:.4}",
                name,
                sigma,
                spectral_norm,
                row_scales.iter().sum::<f32>() / rows as f32
            );
            println!(
                "{:<35} | Skew: {:>6.2} | Kurt: {:>6.2} | Mean: {:>6.3}",
                "", skewness, kurtosis, mean
            );

            // Interpretation based on 365 Data Science principles
            if kurtosis > 1.0 {
                print!("  [Leptokurtic: Heavy Tails] ");
            }
            if skewness.abs() > 0.5 {
                print!("  [High Skewness: Asymmetric] ");
            }
            if mean.abs() > 0.1 {
                print!("  [Bias Detected] ");
            }
            if kurtosis.abs() > 0.1 || skewness.abs() > 0.1 || mean.abs() > 0.1 {
                println!();
            }
        }
    }

    Ok(())
}
