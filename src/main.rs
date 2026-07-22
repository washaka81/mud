use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::expert_bus::ExpertScratch;
use forge_llm::mud::slime::SlimeWorkspace;
use forge_llm::mud::slime_forward::{
    apply_output_norm, evaluate_slime_block, evaluate_slime_block_moe, SlimeLayer,
};
use forge_llm::mud::MudFile;
use rand::RngExt;
use std::env;
use std::io::{self, Write};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: forge_llm <model.mud>");
        return Ok(());
    }
    let model_path = &args[1];
    println!("Loading model: {}", model_path);
    let mud = MudFile::load(model_path)?;

    let core = mud
        .skills
        .get("core")
        .ok_or_else(|| anyhow::anyhow!("Missing core skill"))?;

    let hidden_size = mud
        .global_metadata
        .get("hidden_size")
        .and_then(|v| v.parse().ok())
        .expect("Missing hidden_size in metadata");
    let num_layers = mud
        .global_metadata
        .get("num_hidden_layers")
        .or_else(|| mud.global_metadata.get("num_layers"))
        .and_then(|v| v.parse().ok())
        .expect("Missing num_layers in metadata");
    let n_heads = mud
        .global_metadata
        .get("num_attention_heads")
        .or_else(|| mud.global_metadata.get("num_heads"))
        .and_then(|v| v.parse().ok())
        .expect("Missing num_heads in metadata");
    let n_kv_heads = mud
        .global_metadata
        .get("num_key_value_heads")
        .or_else(|| mud.global_metadata.get("num_kv_heads"))
        .and_then(|v| v.parse().ok())
        .expect("Missing num_kv_heads in metadata");
    let ffn_mid = mud
        .global_metadata
        .get("intermediate_size")
        .or_else(|| mud.global_metadata.get("ffn_hidden"))
        .and_then(|v| v.parse().ok())
        .expect("Missing ffn_mid in metadata");
    let vocab_size = mud
        .global_metadata
        .get("vocab_size")
        .and_then(|v| v.parse().ok())
        .expect("Missing vocab_size in metadata");
    let rms_norm_eps = mud
        .global_metadata
        .get("rms_norm_eps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5);
    let max_pos = mud
        .global_metadata
        .get("max_position_embeddings")
        .and_then(|v| v.parse().ok())
        .expect("Missing max_position_embeddings in metadata");
    let rope_theta = mud
        .global_metadata
        .get("rope.freq_base")
        .or_else(|| mud.global_metadata.get("rope_theta"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(10000.0);

    let mut layers: Vec<SlimeLayer> = Vec::new();
    for blk in 0..num_layers {
        let prefix = format!("blk.{}.", blk);
        let t = |name: &str| -> *const u8 {
            core.tensors
                .get(&format!("{}{}.weight", prefix, name))
                .map(|t| t.data_ptr)
                .unwrap_or(std::ptr::null())
        };
        let ts = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}.prq_scale", prefix, name))
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null())
        };
        let tn = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}.weight", prefix, name))
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null())
        };
        // Llama/Smol: blk.N.norm; Qwen3/Bonsai: blk.N.ffn_norm
        let ffn_norm_w = {
            let a = tn("ffn_norm");
            if !a.is_null() {
                a
            } else {
                tn("norm")
            }
        };
        let ffn_names = forge_llm::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);
        layers.push(SlimeLayer {
            q_w: t("attn_q"),
            k_w: t("attn_k"),
            v_w: t("attn_v"),
            o_w: t("attn_output"),
            q_scales: ts("attn_q"),
            k_scales: ts("attn_k"),
            v_scales: ts("attn_v"),
            o_scales: ts("attn_output"),
            ffn_up_w: t(&ffn_names.up),
            ffn_gate_w: t(&ffn_names.gate),
            ffn_down_w: t(&ffn_names.down),
            ffn_up_scales: ts(&ffn_names.up),
            ffn_gate_scales: ts(&ffn_names.gate),
            ffn_down_scales: ts(&ffn_names.down),
            attn_norm_w: tn("attn_norm"),
            ffn_norm_w,
            attn_sub_norm_w: tn("attn_sub_norm"),
            ffn_sub_norm_w: tn("ffn_sub_norm"),
            // Qwen3: blk.N.attn_q_norm / attn_k_norm (head_dim); Llama: absent → null
            q_norm_w: tn("attn_q_norm"),
            k_norm_w: tn("attn_k_norm"),
            mhc_alpha_w: tn("mhc_alpha"),
            mhc_beta_w: tn("mhc_beta"),
            mhc_radius_w: tn("mhc_radius"),
            n_kv_heads,
            ffn_mid,
            rope_theta,
        });
    }

    let tokens_str = mud
        .global_metadata
        .get("tokenizer.tokens")
        .map(|s| s.as_str())
        .unwrap_or("");
    let merges_str = mud
        .global_metadata
        .get("tokenizer.merges")
        .map(|s| s.as_str())
        .unwrap_or("");
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    // Setup embedding and output weights.
    // SAFETY: token_embd.weight is Ternary2Bit (ELUT 4-bit nibble packed).
    // Reading it as *const f32 would treat packed ternary bits as IEEE floats — wrong.
    // We must dequantize via unpack_ternary2bit_to_f32 using the PRQ scales.
    let emb_tensor = core
        .tensors
        .get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing token_embd.weight"))?;
    let mut emb_f32 = vec![0.0f32; vocab_size * hidden_size];
    let emb_rms_sum_local = if emb_tensor.t_type == forge_llm::mud::MudTensorType::Float32 {
        let emb_data_ptr = emb_tensor
            .owned_data
            .as_ref()
            .map(|d| d.as_ptr() as *const f32)
            .unwrap_or(emb_tensor.data_ptr as *const f32);
        let emb_f32_slice =
            unsafe { std::slice::from_raw_parts(emb_data_ptr, vocab_size * hidden_size) };
        emb_f32.copy_from_slice(emb_f32_slice);
        emb_f32_slice.iter().map(|&x| x * x).sum::<f32>() / (hidden_size as f32)
    } else {
        let emb_scales_tensor = core
            .tensors
            .get("token_embd.prq_scale")
            .ok_or_else(|| anyhow::anyhow!("Missing token_embd.prq_scale"))?;
        let emb_packed_ptr = emb_tensor
            .owned_data
            .as_ref()
            .map(|d| d.as_ptr())
            .unwrap_or(emb_tensor.data_ptr);
        let emb_byte_len = emb_tensor
            .owned_data
            .as_ref()
            .map(|d| d.len())
            .unwrap_or_else(|| {
                let n: usize = vocab_size * hidden_size;
                n.div_ceil(8) * 4
            });
        let emb_packed: &[u8] = unsafe { std::slice::from_raw_parts(emb_packed_ptr, emb_byte_len) };
        let emb_scales_ptr = emb_scales_tensor
            .owned_data
            .as_ref()
            .map(|d| d.as_ptr() as *const f32)
            .unwrap_or(emb_scales_tensor.data_ptr as *const f32);
        let emb_scales: &[f32] = unsafe { std::slice::from_raw_parts(emb_scales_ptr, vocab_size) };

        forge_llm::mud::slime_backward::unpack_ternary2bit_to_f32(
            emb_packed,
            emb_scales,
            hidden_size,
            &mut emb_f32,
        );
        // Compute RMS over dequantized values (not PRQ scales) — same formula as Float32 branch
        emb_f32.iter().map(|&x| x * x).sum::<f32>() / (hidden_size as f32)
    };

    // Output projection: reuse the embedding (tied weights) or load separate output.weight
    let out_tensor = core.tensors.get("output.weight").unwrap_or(emb_tensor);
    let tied_output = std::ptr::eq(out_tensor as *const _, emb_tensor as *const _);
    let mut out_f32_owned: Option<Vec<f32>> = if tied_output {
        None
    } else {
        let mut out_dq = vec![0.0f32; vocab_size * hidden_size];
        if out_tensor.t_type == forge_llm::mud::MudTensorType::Float32 {
            let out_data_ptr = out_tensor
                .owned_data
                .as_ref()
                .map(|d| d.as_ptr() as *const f32)
                .unwrap_or(out_tensor.data_ptr as *const f32);
            let out_f32_slice =
                unsafe { std::slice::from_raw_parts(out_data_ptr, vocab_size * hidden_size) };
            out_dq.copy_from_slice(out_f32_slice);
        } else {
            let out_scales_tensor = core
                .tensors
                .get("output.prq_scale")
                .ok_or_else(|| anyhow::anyhow!("Missing output.prq_scale"))?;
            let out_packed_ptr = out_tensor
                .owned_data
                .as_ref()
                .map(|d| d.as_ptr())
                .unwrap_or(out_tensor.data_ptr);
            let out_byte_len = out_tensor
                .owned_data
                .as_ref()
                .map(|d| d.len())
                .unwrap_or_else(|| (vocab_size * hidden_size).div_ceil(8) * 4);
            let out_packed: &[u8] =
                unsafe { std::slice::from_raw_parts(out_packed_ptr, out_byte_len) };
            let out_scales_ptr = out_scales_tensor
                .owned_data
                .as_ref()
                .map(|d| d.as_ptr() as *const f32)
                .unwrap_or(out_scales_tensor.data_ptr as *const f32);
            let out_scales: &[f32] =
                unsafe { std::slice::from_raw_parts(out_scales_ptr, vocab_size) };
            forge_llm::mud::slime_backward::unpack_ternary2bit_to_f32(
                out_packed,
                out_scales,
                hidden_size,
                &mut out_dq,
            );
        }
        Some(out_dq)
    };

    // Calibration for ternary / tied-untied emb:
    // - Aggressive emb×(1/rms) was blowing residual magnitude vs ternary W scales → garbage.
    // - Default: NO emb scale_up for Ternary2Bit (weights already PRQ-scaled).
    // - Optional: MUD_INFER_SCALE_UP=auto|N  (auto = clamp 1/rms to [1,8])
    // - LM-head: if separate output.weight (converter "untie" copy), apply 1/√H logit scale
    //   (float convert path already did this; ternary path historically skipped it).
    let emb_rms = (emb_rms_sum_local / vocab_size as f32)
        .sqrt()
        .max(forge_llm::mud::constants::EPSILON_FLOOR);
    let emb_is_ternary = emb_tensor.t_type == forge_llm::mud::MudTensorType::Ternary2Bit;
    // FP32 emb (HF convert): leave scale_up=1 — already calibrated.
    // Ternary emb: also default 1; use MUD_INFER_SCALE_UP=auto only if needed.
    let scale_up = match std::env::var("MUD_INFER_SCALE_UP")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "auto" => (1.0 / emb_rms).clamp(1.0, 8.0),
        s if !s.is_empty() && s != "0" && s != "off" && s != "none" => {
            s.parse::<f32>().unwrap_or(1.0).clamp(0.1, 128.0)
        }
        _ => 1.0,
    };
    let logit_scale = if !tied_output {
        // Untied duplicate of emb → match HF/convert float path
        1.0 / (hidden_size as f32).sqrt()
    } else {
        1.0
    };
    eprintln!(
        "[Inference] emb_rms={:.4e} scale_up={:.3} logit_scale={:.4} tied={} ternary={}",
        emb_rms, scale_up, logit_scale, tied_output, emb_is_ternary
    );
    if (scale_up - 1.0).abs() > 1e-6 {
        for v in emb_f32.iter_mut() {
            *v *= scale_up;
        }
    }
    if let Some(ref mut out) = out_f32_owned {
        let s = scale_up * logit_scale;
        if (s - 1.0).abs() > 1e-6 {
            for v in out.iter_mut() {
                *v *= s;
            }
        }
    }

    let emb_slice: &[f32] = &emb_f32;
    // Tied: LM-head rows == emb; untied: separate buffer (optionally logit-scaled)
    let out_slice_ternary: &[f32] = out_f32_owned.as_deref().unwrap_or(emb_slice);

    let out_norm_tensor = core
        .tensors
        .get("output_norm.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing output_norm.weight"))?;
    let out_norm_ptr = out_norm_tensor
        .owned_data
        .as_ref()
        .map(|d| d.as_ptr())
        .unwrap_or(out_norm_tensor.data_ptr);
    let out_norm_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(out_norm_ptr as *const f32, hidden_size) };

    let head_dim = hidden_size / n_heads;

    // Dynamically compute max_emb if missing to prevent mHC geometric prison
    let computed_max_emb = emb_slice
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, |a, b| a.max(b));
    let max_emb = mud
        .global_metadata
        .get("max_emb")
        .and_then(|v| v.parse().ok())
        .unwrap_or(computed_max_emb);

    let mut ws = SlimeWorkspace::new(
        hidden_size,
        max_pos,
        n_heads,
        n_kv_heads,
        head_dim,
        ffn_mid,
        num_layers,
        max_emb,
    );

    // MoE buses (multi-expert when present; single expert.0 → dense-compatible)
    let top_k = forge_llm::mud::moe_load::default_top_k();
    let moe_buses =
        forge_llm::mud::moe_load::load_model_buses(&mud, num_layers, hidden_size, ffn_mid, top_k);
    let multi = forge_llm::mud::moe_load::model_has_multi_expert(&moe_buses);
    let mut moe_scratch = ExpertScratch::new(hidden_size, ffn_mid, top_k.max(8));
    if multi {
        println!(
            "[MoE] Multi-expert FFN active (top_k={top_k}). Clone: MUD_MOE_CLONE=N for dense→MoE tests."
        );
    } else if let Some(eid) = forge_llm::mud::moe_load::train_expert_id() {
        println!("[MoE] Dense FFN uses expert.{eid} (MUD_TRAIN_EXPERT)");
    }

    println!("Model loaded successfully. Starting MUD Inference CLI...");
    println!("Type 'quit' or 'exit' to stop.");

    // Pre-allocated LM-head buffers (P-01: no per-token Vec in generation hot path)
    let mut logits = vec![0.0f32; vocab_size];
    let mut reg_f32 = vec![0.0f32; hidden_size];

    let stdin = io::stdin();

    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut input = String::new();
        let bytes_read = stdin.read_line(&mut input)?;
        if bytes_read == 0 {
            break;
        } // EOF reached
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        // Chat template: prefer model-native specials (SmolLM / ChatML), fall back to generic.
        let prompt = if tokenizer.special_tokens.contains_key("<|im_start|>") {
            format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                input
            )
        } else if tokenizer.special_tokens.contains_key("<|user|>") {
            format!("<|user|>\n{}<|end|>\n<|assistant|>\n", input)
        } else {
            // Plain continuation for base models without chat specials
            format!("{}\n", input)
        };
        let mut tokens = tokenizer.encode(&prompt);
        let eos_id = mud
            .global_metadata
            .get("eos_token_id")
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| tokenizer.special_tokens.get("<|im_end|>").copied())
            .or_else(|| tokenizer.special_tokens.get("<|endoftext|>").copied())
            .unwrap_or(u32::MAX);

        ws.kv_cache.fill(0.0);
        ws.v_cache.fill(0.0);
        ws.jepa_mu.fill(0.0);
        ws.jepa_inv_sigma.fill(0.0);
        ws.jepa_var_ema.fill(0.0);

        print!("AI: ");
        io::stdout().flush()?;

        let max_gen = std::env::var("MUD_INFER_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128usize)
            .clamp(8, 1024);
        let mut generated = 0;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut current_pos = 0;

        // Context processing
        while current_pos < tokens.len() {
            ws.clear_registers();
            let tid = tokens[current_pos] as usize;
            let emb_start = tid * hidden_size;

            for (i, v) in emb_slice[emb_start..emb_start + hidden_size]
                .iter()
                .enumerate()
            {
                forge_llm::mud::slime::SlimeRegister::init_from_embed(
                    &mut ws.registers[i],
                    &mut ws.jepa_z,
                    i,
                    hidden_size,
                    num_layers,
                    *v,
                    current_pos == 0,
                );
            }

            for (l_idx, layer) in layers.iter().enumerate() {
                let use_moe = moe_buses
                    .get(l_idx)
                    .and_then(|b| b.as_ref())
                    .map(|b| b.mounted_count() > 1)
                    .unwrap_or(false);
                if use_moe {
                    let bus = moe_buses[l_idx].as_ref().unwrap();
                    evaluate_slime_block_moe(
                        layer,
                        l_idx,
                        &mut ws,
                        current_pos,
                        rms_norm_eps,
                        None,
                        Some(bus),
                        Some(&mut moe_scratch),
                    );
                } else {
                    evaluate_slime_block(layer, l_idx, &mut ws, current_pos, rms_norm_eps, None);
                }
            }

            current_pos += 1;
        }

        // Auto-regressive generation loop
        while generated < max_gen {
            apply_output_norm(&mut ws, out_norm_slice.as_ptr(), rms_norm_eps);

            // Pack registers → flat f32 once, then full LM-head via AVX2 FMA kernel
            for (i, r) in ws.registers.iter().enumerate().take(hidden_size) {
                reg_f32[i] = r.matmul_accum;
            }
            // Optional C-MUD complex reasoning pass (research §3): no-op unless
            // MUD_CMUD_THINK=1. Acts on the hidden state before the LM-head.
            forge_llm::mud::cmud::maybe_think_collapse_rms_scaled(&mut reg_f32);
            unsafe {
                forge_llm::asm::lm_head_logits_avx2(
                    vocab_size,
                    hidden_size,
                    reg_f32.as_ptr(),
                    out_slice_ternary.as_ptr(),
                    logits.as_mut_ptr(),
                );
            }

            // DC Bias Removal
            let logit_mean = logits.iter().sum::<f32>() / vocab_size as f32;
            for l in logits.iter_mut() {
                *l -= logit_mean;
            }

            // Thermodynamic rescaling — only when logits have real dynamic range.
            // Boosting max→15 when peak < 0.05 amplifies pure noise → gibberish loops.
            let shifted_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min_peak = std::env::var("MUD_INFER_MIN_PEAK")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.05f32);
            let target_peak = std::env::var("MUD_INFER_TARGET_PEAK")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8.0f32);
            if shifted_max > min_peak {
                let boost = (target_peak / shifted_max).clamp(0.5, 16.0);
                for l in logits.iter_mut() {
                    *l *= boost;
                }
            } else if generated == 0 {
                eprintln!(
                    "[Inference] weak logits (max={:.4} < {:.2}) — skip peak boost (noise guard)",
                    shifted_max, min_peak
                );
            }

            // Repetition penalty (stronger on recent window to break ternary loops)
            let rep = std::env::var("MUD_INFER_REP_PENALTY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.25f32);
            let window = generated_tokens.len().saturating_sub(64);
            for &prev_token in &generated_tokens[window..] {
                let i = prev_token as usize;
                if i >= logits.len() {
                    continue;
                }
                if logits[i] > 0.0 {
                    logits[i] /= rep;
                } else {
                    logits[i] *= rep;
                }
            }

            // Sampling: greedy (MUD_INFER_GREEDY=1) or Top-P 0.95
            let greedy = std::env::var("MUD_INFER_GREEDY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);

            let best_idx = if greedy {
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();

                let mut probs: Vec<(usize, f32)> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i, (l - max_l).exp() / sum_exp))
                    .collect();

                probs.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut cumsum = 0.0;
                let mut top_p_probs = Vec::new();
                for p in probs {
                    cumsum += p.1;
                    top_p_probs.push(p);
                    if cumsum >= 0.95 {
                        break;
                    }
                }

                let top_p_sum: f32 = top_p_probs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-12);
                let mut rng = rand::rng();
                let mut r: f32 = rng.random();

                let mut pick = top_p_probs.last().map(|(i, _)| *i).unwrap_or(0);
                for &(idx, p) in &top_p_probs {
                    let norm_p = p / top_p_sum;
                    if r < norm_p {
                        pick = idx;
                        break;
                    }
                    r -= norm_p;
                }
                pick
            };

            let next_token = best_idx as u32;
            tokens.push(next_token);
            generated_tokens.push(next_token);

            let decoded_piece = tokenizer.decode(&[next_token]);
            print!("{}", decoded_piece);
            io::stdout().flush()?;

            // Stop on model-native EOS / chat end markers (not hard-coded Llama 128001).
            if next_token == eos_id
                || decoded_piece.contains("<|end|>")
                || decoded_piece.contains("<|endoftext|>")
                || decoded_piece.contains("<|im_end|>")
                || decoded_piece.contains("<|eot_id|>")
            {
                break;
            }

            ws.clear_registers();
            let emb_start = best_idx * hidden_size;
            for (i, v) in emb_slice[emb_start..emb_start + hidden_size]
                .iter()
                .enumerate()
            {
                forge_llm::mud::slime::SlimeRegister::init_from_embed(
                    &mut ws.registers[i],
                    &mut ws.jepa_z,
                    i,
                    hidden_size,
                    num_layers,
                    *v,
                    false,
                );
            }

            for (l_idx, layer) in layers.iter().enumerate() {
                let use_moe = moe_buses
                    .get(l_idx)
                    .and_then(|b| b.as_ref())
                    .map(|b| b.mounted_count() > 1)
                    .unwrap_or(false);
                if use_moe {
                    let bus = moe_buses[l_idx].as_ref().unwrap();
                    evaluate_slime_block_moe(
                        layer,
                        l_idx,
                        &mut ws,
                        current_pos,
                        rms_norm_eps,
                        None,
                        Some(bus),
                        Some(&mut moe_scratch),
                    );
                } else {
                    evaluate_slime_block(layer, l_idx, &mut ws, current_pos, rms_norm_eps, None);
                }
            }

            current_pos += 1;
            generated += 1;
        }
        println!();
    }

    Ok(())
}
