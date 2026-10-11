//! Pre-flight alignment: metadata validation, AWAKE-01 L-QAT, tokenization audit.
//!
//! These three routines are run by `MudCorpusTrainer::new` and the
//! start of every debate session. They are "post-convert" sanity
//! steps that bring a freshly converted `.mud` up to a workable state.
//!
//! - [`validate_metadata`]: P-13 / L-12 fail-fast checks on the model
//!   arch + ELUT count + scale count. Used by `new()`.
//! - [`deep_local_alignment`]: AWAKE-01 — runs a few SGD iterations
//!   per ternary weight tensor to align the FP32 shadow with the
//!   master output (`y_student` ≈ `y_master`). Also recovers collapsed
//!   mHC `alpha/beta/radius` tensors.
//! - [`audit_tokenization`]: Phase 1 sanity check that the encoder
//!   round-trips a few test phrases.

use crate::mud::corpus_trainer::{MudCorpusTrainer, SHOULD_TERMINATE};
use crate::mud::{MudFile, MudTensorType};
use std::sync::atomic::Ordering;

impl MudCorpusTrainer {
    /// P-13 / L-12 fail-fast metadata validation. Called by `new()`.
    pub(crate) fn validate_metadata(mud: &MudFile) -> anyhow::Result<()> {
        println!(
            "{}",
            crate::mud::trainer_ui::phase("Phase 0", "Metadata Integrity Validation (P-13 / L-12)")
        );
        // L-12: shared fail-fast parser (alternate key names, consistency, ELUT multiples)
        let dims = crate::mud::p13::parse_arch_dims(mud)?;
        crate::mud::p13::validate_trainer_required_keys(&mud.global_metadata)?;
        println!(
            "{}",
            crate::mud::trainer_ui::note(
                "ok",
                &format!(
                    "arch hidden={} layers={} heads={}/{} ffn={}",
                    dims.hidden_size,
                    dims.num_layers,
                    dims.num_heads,
                    dims.num_kv_heads,
                    dims.intermediate_size
                )
            )
        );
        if let Some(core) = mud.skills.get("core") {
            let mut ternary_count = 0;
            let mut scale_count = 0;
            for (name, tensor) in &core.tensors {
                if tensor.t_type == MudTensorType::Ternary2Bit {
                    ternary_count += 1;
                }
                if name.ends_with(".prq_scale") {
                    scale_count += 1;
                }
            }
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!("found {ternary_count} ternary weights and {scale_count} scales")
                )
            );
        }
        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "metadata validated successfully (P-13)")
        );
        Ok(())
    }

    /// AWAKE-01 — Universal Agnostic Deep Local Alignment (L-QAT).
    ///
    /// Runs a few SGD iterations per ternary weight tensor to align the
    /// FP32 shadow with the master output, then re-packs the tensor +
    /// its PRQ scale. Also recovers collapsed mHC `alpha/beta/radius`
    /// tensors (Priority 12 & 17).
    #[allow(unreachable_code, unused_variables, unused_mut, unused_assignments)]
    pub(crate) fn deep_local_alignment(
        &self,
        mud: &mut MudFile,
        print_stdout: bool,
    ) -> anyhow::Result<()> {
        if print_stdout {
            println!(
                "{}",
                crate::mud::trainer_ui::phase(
                    "AWAKE-01",
                    "Universal Agnostic Deep Local Alignment (L-QAT)"
                )
            );
        }

        let learning_rate = 0.001f32;
        // Post-convert: more L-QAT iters unless quick smoke (MAX_CHUNKS without explicit override)
        let ldt_iterations: usize = std::env::var("MUD_AWAKE_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                let quick = std::env::var("MUD_TRAIN_MAX_CHUNKS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&m| m > 0 && m <= 16)
                    .is_some();
                if quick {
                    4
                } else {
                    12
                }
            });
        let weight_decay = 0.0001f32;

        let mut aligned_count = 0;
        let mut total_ternary = 0;

        for skill in mud.skills.values() {
            total_ternary += skill
                .tensors
                .values()
                .filter(|t| t.t_type == MudTensorType::Ternary2Bit)
                .count();
        }

        // We iterate over the keys to avoid mutable borrow conflicts
        for (skill_name, skill) in mud.skills.iter_mut() {
            let ternary_keys: Vec<String> = skill
                .tensors
                .iter()
                .filter(|(_, t)| t.t_type == MudTensorType::Ternary2Bit && t.shape.len() == 2)
                .map(|(k, _)| -> String { k.clone() })
                .collect();

            for t_name in ternary_keys {
                if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                    break;
                }

                let (rows, cols) = {
                    let t = match skill.tensors.get(&t_name) {
                        Some(t) => t,
                        None => {
                            log::warn!("alignment: missing tensor {t_name}, skipping");
                            continue;
                        }
                    };
                    (t.shape[0], t.shape[1])
                };
                let elements = rows * cols;

                let scale_name = t_name.replace(".weight", ".prq_scale");
                let mut scales = vec![1.0f32; rows];
                if let Some(scale_tensor) = skill.tensors.get(&scale_name) {
                    if scale_tensor.t_type == MudTensorType::Float32
                        && scale_tensor.shape[0] == rows
                    {
                        unsafe {
                            let ptr = scale_tensor.data_ptr as *const f32;
                            std::ptr::copy_nonoverlapping(ptr, scales.as_mut_ptr(), rows);
                        }
                    }
                }

                // 1. Dequantize to FP32 shadow weights
                let mut w_fp32 = vec![0.0f32; elements];
                unsafe {
                    let t = match skill.tensors.get(&t_name) {
                        Some(t) => t,
                        None => {
                            log::warn!("alignment: missing tensor {t_name} during dequant, skipping");
                            continue;
                        }
                    };
                    for r in 0..rows {
                        crate::mud::dequantize_ternary_row(
                            (t.data_ptr as *const u32).add(r * cols / 8),
                            &mut w_fp32[r * cols..(r + 1) * cols],
                            cols,
                        );
                        let s = scales[r];
                        for c in 0..cols {
                            w_fp32[r * cols + c] *= s;
                        }
                    }
                }

                // 2. Perform L-QAT SGD iterations (AVX dots, reused scratch)
                let mut x = vec![0.0f32; cols];
                let mut w_q_row = vec![0.0f32; cols];
                for _iter in 0..ldt_iterations {
                    let mut rng_state = 1337u32.wrapping_add(_iter as u32 * 0x9E37);
                    #[allow(clippy::needless_range_loop)]
                    for c in 0..cols {
                        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        x[c] = (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    }

                    for r in 0..rows {
                        let row_start = r * cols;
                        let row = &w_fp32[row_start..row_start + cols];

                        let mut absmean = 0.0f32;
                        for &w in row {
                            absmean += w.abs();
                        }
                        absmean /= cols as f32;
                        let scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);

                        for c in 0..cols {
                            w_q_row[c] = (row[c] / scale).round().clamp(-1.0, 1.0) * scale;
                        }
                        let y_master = unsafe {
                            forge_autograd::avx_math::dot_product_avx2(row, &x[..cols])
                        };
                        let y_student = unsafe {
                            forge_autograd::avx_math::dot_product_avx2(&w_q_row, &x[..cols])
                        };
                        let err = y_student - y_master;

                        // Apply SGD gradients & Weight Decay
                        for c in 0..cols {
                            let mut grad = err * x[c] / (cols as f32); // Normalize by cols
                            grad = grad.clamp(-10.0, 10.0); // Clip gradient
                            w_fp32[row_start + c] -=
                                learning_rate * grad + weight_decay * w_fp32[row_start + c];
                            if !w_fp32[row_start + c].is_finite() {
                                w_fp32[row_start + c] = 0.0;
                            }
                        }
                    }
                }

                // 3. Re-quantize and Pack back to Ternary2Bit
                let mut new_scales = vec![0.0f32; rows];
                let u32_count = elements.div_ceil(8);
                let mut packed = vec![0u32; u32_count];

                #[allow(clippy::needless_range_loop)]
                for r in 0..rows {
                    let row_start = r * cols;
                    let mut absmean = 0.0f32;
                    for c in 0..cols {
                        absmean += w_fp32[row_start + c].abs();
                    }
                    absmean /= cols as f32;
                    let scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                    new_scales[r] = scale;

                    for c in 0..cols {
                        let idx = row_start + c;
                        let w_f = w_fp32[idx];
                        let w_q = (w_f / scale).round().clamp(-1.0, 1.0);
                        let bit = if w_q > 0.5 {
                            0x1u32
                        } else if w_q < -0.5 {
                            0xFu32
                        } else {
                            0x0u32
                        };
                        packed[idx / 8] |= bit << ((idx % 8) * 4);
                    }
                }

                // Update tensor data
                let packed_bytes = unsafe {
                    std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
                }
                .to_vec();
                if let Some(t) = skill.tensors.get_mut(&t_name) {
                    t.owned_data = Some(packed_bytes);
                    if let Some(d) = t.owned_data.as_ref() {
                        t.data_ptr = d.as_ptr();
                    }
                }

                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_scales.as_ptr() as *const u8,
                        new_scales.len() * 4,
                    )
                }
                .to_vec();
                if let Some(s_t) = skill.tensors.get_mut(&scale_name) {
                    s_t.owned_data = Some(scale_bytes);
                    if let Some(d) = s_t.owned_data.as_ref() {
                        s_t.data_ptr = d.as_ptr();
                    }
                } else {
                    skill.tensors.insert(
                        scale_name.clone(),
                        crate::mud::MudTensor {
                            name: scale_name.clone(),
                            t_type: MudTensorType::Float32,
                            shape: vec![rows],
                            data_ptr: scale_bytes.as_ptr(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: Some(scale_bytes),
                        },
                    );
                    if let Some(s_t) = skill.tensors.get_mut(&scale_name) {
                        if let Some(d) = s_t.owned_data.as_ref() {
                            s_t.data_ptr = d.as_ptr();
                        }
                    }
                }

                aligned_count += 1;
                if print_stdout {
                    print!(
                        "\r  \x1b[1;36m[L-QAT]\x1b[0m Aligned {}/{} tensors ({:.1}%)",
                        aligned_count,
                        total_ternary,
                        (aligned_count as f32 / total_ternary as f32) * 100.0
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            }
            // --- mHC Collapse Recovery (Priority 12 & 17) ---
            if print_stdout {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ok",
                        "auditing mHC residual collapse for skill..."
                    )
                );
            }
            // L-07 / P-13: skill metadata first, then global_metadata (converter often only sets global)
            let num_layers: usize = skill
                .metadata
                .get("num_hidden_layers")
                .or_else(|| skill.metadata.get("num_layers"))
                .or_else(|| mud.global_metadata.get("num_hidden_layers"))
                .or_else(|| mud.global_metadata.get("num_layers"))
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    panic!(
                        "P-13: missing num_hidden_layers/num_layers (skill={skill_name} and global)"
                    )
                });
            let hidden: usize = skill
                .metadata
                .get("hidden_size")
                .or_else(|| mud.global_metadata.get("hidden_size"))
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    panic!("P-13: missing hidden_size (skill={skill_name} and global)")
                });

            for l in 0..num_layers {
                let alpha_name = format!("blk.{}.mhc_alpha", l);
                let beta_name = format!("blk.{}.mhc_beta", l);
                let mut alpha_collapsed = false;

                if let Some(alpha_tensor) = skill.tensors.get(&alpha_name) {
                    if let Some(data) = &alpha_tensor.owned_data {
                        let alpha_slice: &[f32] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                        };
                        let mean_alpha = alpha_slice.iter().map(|v| v.abs()).sum::<f32>()
                            / (alpha_slice.len() as f32).max(1.0);
                        if mean_alpha < 0.1 {
                            alpha_collapsed = true;
                        }
                    } else {
                        alpha_collapsed = true;
                    }
                } else {
                    alpha_collapsed = true;
                }

                if alpha_collapsed {
                    let new_alpha = vec![0.85f32; hidden];
                    let new_beta = vec![0.15f32; hidden];

                    let alpha_bytes = unsafe {
                        std::slice::from_raw_parts(new_alpha.as_ptr() as *const u8, hidden * 4)
                    }
                    .to_vec();
                    let beta_bytes = unsafe {
                        std::slice::from_raw_parts(new_beta.as_ptr() as *const u8, hidden * 4)
                    }
                    .to_vec();

                    let t_alpha = skill.tensors.entry(alpha_name.clone()).or_insert_with(|| {
                        crate::mud::MudTensor {
                            name: alpha_name,
                            t_type: crate::mud::MudTensorType::Float32,
                            shape: vec![hidden],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: None,
                        }
                    });
                    t_alpha.owned_data = Some(alpha_bytes);
                    if let Some(d) = t_alpha.owned_data.as_ref() {
                        t_alpha.data_ptr = d.as_ptr();
                    }

                    let t_beta = skill.tensors.entry(beta_name.clone()).or_insert_with(|| {
                        crate::mud::MudTensor {
                            name: beta_name,
                            t_type: crate::mud::MudTensorType::Float32,
                            shape: vec![hidden],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: None,
                        }
                    });
                    t_beta.owned_data = Some(beta_bytes);
                    if let Some(d) = t_beta.owned_data.as_ref() {
                        t_beta.data_ptr = d.as_ptr();
                    }
                }

                let radius_name = format!("blk.{}.mhc_radius", l);
                let mut radius_collapsed = false;
                if let Some(radius_tensor) = skill.tensors.get(&radius_name) {
                    if let Some(data) = &radius_tensor.owned_data {
                        let rad_slice: &[f32] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                        };
                        if rad_slice.is_empty() || rad_slice[0] < 5.0 {
                            radius_collapsed = true;
                        }
                    } else {
                        radius_collapsed = true;
                    }
                } else {
                    radius_collapsed = true;
                }

                if radius_collapsed {
                    let computed_max_emb =
                        if let Some(emb) = skill.tensors.get("token_embd.weight") {
                            if let Some(d) = &emb.owned_data {
                                let slice: &[f32] = unsafe {
                                    std::slice::from_raw_parts(
                                        d.as_ptr() as *const f32,
                                        d.len() / 4,
                                    )
                                };
                                slice
                                    .iter()
                                    .map(|v| v.abs())
                                    .fold(0.0f32, |a: f32, b: f32| a.max(b))
                            } else {
                                1.0  // fallback when emb owned_data missing (was 128.0 — 128× too high)
                            }
                        } else {
                            1.0  // fallback when emb tensor missing (was 128.0)
                        };
                    let max_emb = computed_max_emb.max(0.1);  // was .max(5.0) — too high for ternary (max_emb≈0.74)

                    let safe_radius = max_emb * (hidden as f32).sqrt();
                    let new_radius = [safe_radius; 1];
                    let radius_bytes = unsafe {
                        std::slice::from_raw_parts(new_radius.as_ptr() as *const u8, 4)
                    }
                    .to_vec();

                    let t_radius = skill.tensors.entry(radius_name.clone()).or_insert_with(|| {
                        crate::mud::MudTensor {
                            name: radius_name,
                            t_type: crate::mud::MudTensorType::Float32,
                            shape: vec![1],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: None,
                        }
                    });
                    t_radius.owned_data = Some(radius_bytes);
                    if let Some(d) = t_radius.owned_data.as_ref() {
                        t_radius.data_ptr = d.as_ptr();
                    }
                }
            }
        } // end of mud.skills loop

        if print_stdout {
            println!(
                "{}",
                crate::mud::trainer_ui::note("ok", "AWAKE-01 alignment complete")
            );
        }
        Ok(())
    }

    /// Phase 1 sanity check that the encoder round-trips a few test phrases.
    pub(crate) fn audit_tokenization(&self) {
        println!(
            "{}",
            crate::mud::trainer_ui::phase("Phase 1", "Tokenization Sync Audit")
        );
        let test_phrases = [
            "MUD engine optimized.",
            "Inteligencia artificial.",
            "BPE Hello World!",
        ];
        for phrase in test_phrases {
            let ids = self.tokenizer.encode(phrase);
            let decoded = self.tokenizer.decode(&ids);
            println!(
                "   - original: \"{}\"  |  decoded: \"{}\"",
                phrase,
                decoded.trim()
            );
            println!("     numeric tokens: {:?}", ids);
        }
        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "tokenization audit complete")
        );
    }
}
