use forge_llm::mud::expert_bus::ExpertScratch;
use forge_llm::mud::fenwick::FenwickF32;
use forge_llm::mud::slime::SlimeWorkspace;
use forge_llm::mud::slime_forward::{
    apply_output_norm, evaluate_slime_block, evaluate_slime_block_moe, SlimeLayer,
};
use forge_llm::mud::{MudFile, MudTensorType};
use rand::RngExt;
use std::collections::HashSet;
use std::env;
use std::io::{self, Write};

/// ANSI color codes for chat output
const COLOR_USER: &str = "\x1b[96m";   // bright cyan
const COLOR_AI: &str = "\x1b[92m";     // bright green
const COLOR_DIM: &str = "\x1b[90m";    // bright black (gray)
const COLOR_RESET: &str = "\x1b[0m";

/// Debug flag: enables verbose logits/top-10/reg_f32 output
fn debug_mode() -> bool {
    std::env::var("MUD_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Format a chat prompt with system prompt and role tags (Llama-style)
fn format_chat_prompt(system_prompt: &str, user_input: &str, turn: usize) -> String {
    if system_prompt.is_empty() {
        format!("<|im_start|>user\n{user_input}<|im_end|><|im_start|>assistant\n")
    } else if turn == 0 {
        format!(
            "<|im_start|>system\n{system_prompt}<|im_end|><|im_start|>user\n{user_input}<|im_end|><|im_start|>assistant\n"
        )
    } else {
        format!("<|im_start|>user\n{user_input}<|im_end|><|im_start|>assistant\n")
    }
}

fn main() -> anyhow::Result<()> {
    // SEC-H6 (2026-07-27): initialize structured logger so log::warn!/log::error!
    // from library code (NaN guards, GGUF warnings, etc.) are visible on stderr.
    // Default level = warn; set RUST_LOG=info for verbose output.
    let _ = env_logger::Builder::from_env(env_logger::Env::default().filter_or("RUST_LOG", "warn"))
        .format_timestamp_millis()
        .try_init();
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

    // SPRINT-4: typed-error-aware metadata extraction. P-17 fail-fast is
    // preserved (panic on missing) but the panic message now includes a
    // remediation hint (re-convert the .mud) instead of a bare field name.
    use forge_llm::mud::error::expect_metadata;
    let hidden_size = expect_metadata(
        "hidden_size",
        mud.global_metadata
            .get("hidden_size")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("missing")),
    );
    let num_layers = expect_metadata(
        "num_layers",
        mud.global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("missing")),
    );
    let n_heads = expect_metadata(
        "num_heads",
        mud.global_metadata
            .get("num_attention_heads")
            .or_else(|| mud.global_metadata.get("num_heads"))
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("missing")),
    );
    let n_kv_heads = expect_metadata(
        "num_kv_heads",
        mud.global_metadata
            .get("num_key_value_heads")
            .or_else(|| mud.global_metadata.get("num_kv_heads"))
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("missing")),
    );
    let ffn_mid = expect_metadata(
        "ffn_mid",
        mud.global_metadata
            .get("intermediate_size")
            .or_else(|| mud.global_metadata.get("ffn_hidden"))
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("missing")),
    );
    let vocab_size = expect_metadata(
        "vocab_size",
        mud.global_metadata
            .get("vocab_size")
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("missing")),
    );
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
        // tc: lookup for non-`.weight` tensors (mHC alpha/beta/radius).
        // FIX (2026-07-30): mHC tensors are named `blk.N.mhc_alpha`, not
        // `blk.N.mhc_alpha.weight`. Using `tn` silently returned null.
        let tc = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}", prefix, name))
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
        let head_d = hidden_size / n_heads;
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
            mhc_alpha_w: tc("mhc_alpha"),
            mhc_beta_w: tc("mhc_beta"),
            mhc_alpha_shadow: None,
            mhc_beta_shadow: None,
            attn_norm_shadow: None,
            ffn_norm_shadow: None,
            mhc_radius_w: tc("mhc_radius"),
            n_kv_heads,
            ffn_mid,
            rope_theta,
            rope_inv_freq: forge_llm::mud::slime_forward::compute_rope_inv_freq(rope_theta, head_d),
            rope_cache: parking_lot::Mutex::new(forge_llm::mud::slime_forward::RopeCache::new()),
            is_f32_weights: core.tensors
                .get(&format!("blk.{}.attn_q.weight", 0))
                .map(|t| t.t_type == MudTensorType::Float32)
                .unwrap_or(false),
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
    let tokenizer = forge_llm::mud::smart_tokenizer::SmartTokenizer::from_mud_metadata(tokens_str, merges_str, &mud.global_metadata);

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
        // SAFETY: Float32 `token_embd.weight`; `MudFile::load` ran
        // `validate_model_shapes` (R-30), so its shape is [vocab,hidden] and
        // the backing (owned_data or mmap-validated `data_ptr`) holds
        // `vocab_size * hidden_size` f32.
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
        // SAFETY: `emb_byte_len` is the owned length or the canonical packed
        // size for the R-30-validated [vocab,hidden] shape; the pointer is the
        // tensor's own backing.
        let emb_packed: &[u8] = unsafe { std::slice::from_raw_parts(emb_packed_ptr, emb_byte_len) };
        let emb_scales_ptr = emb_scales_tensor
            .owned_data
            .as_ref()
            .map(|d| d.as_ptr() as *const f32)
            .unwrap_or(emb_scales_tensor.data_ptr as *const f32);
        // SAFETY: `token_embd.prq_scale` shape == [vocab] (R-30); its backing
        // holds `vocab_size` f32.
        let emb_scales: &[f32] = unsafe { std::slice::from_raw_parts(emb_scales_ptr, vocab_size) };

        forge_llm::mud::slime_backward::unpack_ternary2bit_to_f32(
            emb_packed,
            emb_scales,
            hidden_size,
            &mut emb_f32,
        );
        let mut hist = std::collections::HashMap::new();
        for &v in emb_f32.iter().take(100000) {
            if v != 0.0 {
                let v_scaled = v / emb_scales[0]; // roughly
                *hist.entry(v_scaled.round() as i32).or_insert(0) += 1;
            }
        }
        if debug_mode() {
            println!("{}Emb hist: {:?}{}", COLOR_DIM, hist, COLOR_RESET);
        }
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
            // SAFETY: Float32 `output.weight`; R-30 validated shape
            // [vocab,hidden]; backing holds `vocab_size * hidden_size` f32.
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
            // SAFETY: `out_byte_len` = owned length or canonical packed size
            // for the R-30-validated shape; pointer is the tensor's backing.
            let out_packed: &[u8] =
                unsafe { std::slice::from_raw_parts(out_packed_ptr, out_byte_len) };
            let out_scales_ptr = out_scales_tensor
                .owned_data
                .as_ref()
                .map(|d| d.as_ptr() as *const f32)
                .unwrap_or(out_scales_tensor.data_ptr as *const f32);
            // SAFETY: `output.prq_scale` shape == [vocab] (R-30); backing holds
            // `vocab_size` f32.
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

    let mut hist = std::collections::HashMap::new();
    for &v in emb_f32.iter().take(10000) {
        if v.abs() > 1e-5 {
            let v_scaled = v / 0.02709961; // roughly
            *hist.entry(v_scaled.round() as i32).or_insert(0) += 1;
        }
    }
    if debug_mode() {
        println!("{}Emb hist (first 10k): {:?}{}", COLOR_DIM, hist, COLOR_RESET);
    }

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
    // SAFETY: `output_norm.weight` shape == [hidden] (R-30); backing holds
    // `hidden_size` f32.
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
    println!("Type 'quit' or 'exit' to stop.  (MUD_DEBUG=1 for logits/top-10, MUD_SYSTEM_PROMPT=\"...\")");

    // Pre-allocated LM-head buffers (P-01: no per-token Vec in generation hot path)
    let mut logits = vec![0.0f32; vocab_size];
    let mut reg_f32 = vec![0.0f32; hidden_size];
    let mut scratch_imag = vec![0.0f32; hidden_size];

    // Fenwick tree for O(log n) top-P sampling (avoids full Vec sort per token)
    let mut fenwick_logits = FenwickF32::with_capacity(vocab_size);

    // System prompt (optional)
    let system_prompt = std::env::var("MUD_SYSTEM_PROMPT").unwrap_or_default();

    // Temperature for sampling (default 0.8)
    let temperature = std::env::var("MUD_INFER_TEMPERATURE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|t| *t > 0.0)
        .unwrap_or(0.8);

    // Top-P nucleus sampling threshold (default 0.95)
    let top_p = std::env::var("MUD_INFER_TOP_P")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|p| *p > 0.0 && *p <= 1.0)
        .unwrap_or(0.95);

    // Repetition penalty (default 1.25)
    let rep_penalty = std::env::var("MUD_INFER_REP_PENALTY")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(1.25);

    let stdin = io::stdin();
    let debug = debug_mode();

    // Conversation state (persisted across turns for context memory)
    let mut turn_count: usize = 0;
    let mut kv_pos: usize = 0;  // persistent KV cache position across turns
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut rep_window: HashSet<u32> = HashSet::with_capacity(128);

    loop {
        print!("\n{}You: {}{}", COLOR_USER, COLOR_RESET, COLOR_RESET);
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

        // Format prompt with chat template (system prompt on first turn)
        let prompt = format_chat_prompt(&system_prompt, input, turn_count);
        let mut tokens = tokenizer.encode(&prompt);
        if debug_mode() {
            println!("{}Tokens for '{}': {:?}{}", COLOR_DIM, input, tokens, COLOR_RESET);
        }
        let eos_id = mud
            .global_metadata
            .get("eos_token_id")
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| tokenizer.special_id("<|im_end|>"))
            .or_else(|| tokenizer.special_id("<|endoftext|>"))
            .unwrap_or(u32::MAX);

        // Conversation history: only clear KV cache on first turn or when
        // approaching max_pos (context window management)
        let total_expected = kv_pos + tokens.len() + 256; // 256 = max_gen buffer
        if turn_count == 0 || total_expected > max_pos {
            ws.kv_cache.fill(0.0);
            ws.v_cache.fill(0.0);
            ws.jepa_mu.fill(0.0);
            ws.jepa_inv_sigma.fill(0.0);
            ws.jepa_var_ema.fill(0.0);
            kv_pos = 0;
            generated_tokens.clear();
            rep_window.clear();
        }

        print!("{}AI: {}", COLOR_AI, COLOR_RESET);
        io::stdout().flush()?;

        let max_gen = std::env::var("MUD_INFER_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128usize)
            .clamp(8, 1024);
        let mut generated = 0;
        let mut current_pos = kv_pos;

        // Context processing
        while current_pos < tokens.len() {
            ws.clear_registers();
            // P-13: clamp tid to valid vocab range to prevent OOB on `emb_slice[emb_start..]`.
            // The token may come from user prompt (untrusted) or be a non-vocab id
            // (e.g., pad/EOS) outside `emb_slice`'s allocated vocab_size.
            let vocab_len = vocab_size.max(1);
            let tid = (tokens[current_pos] as usize).min(vocab_len - 1);
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
                        None,
                    );
                } else {
                    evaluate_slime_block(layer, l_idx, &mut ws, current_pos, rms_norm_eps, None, None);
                }
            }

            current_pos += 1;
        }

        // Auto-regressive generation loop
        while generated < max_gen {
            apply_output_norm(&mut ws, out_norm_slice.as_ptr(), rms_norm_eps);

            // Pack registers → flat f32 once via AVX2 batch (P4), then full LM-head via AVX2 FMA kernel
            forge_llm::mud::slime::SlimeRegister::registers_to_f32_into(
                &ws.registers[..hidden_size],
                &mut reg_f32,
                &mut scratch_imag,
            );
            // Optional C-MUD complex reasoning pass (research §3): no-op unless
            // MUD_CMUD_THINK=1. Acts on the hidden state before the LM-head.
            forge_llm::mud::cmud::maybe_think_collapse_rms_scaled(&mut reg_f32);
            if debug && generated == 0 {
                println!("{}reg_f32[0..5]={:?}{}", COLOR_DIM, &reg_f32[..5], COLOR_RESET);
            }
            // SAFETY: `reg_f32`/`logits` were pre-allocated with `hidden_size`
            // and `vocab_size` elements (above); `out_slice_ternary` is either
            // `emb_f32` or the untied `out_f32_owned` Vec, both sized
            // `vocab_size * hidden_size` and alive for the whole generation
            // loop. `lm_head_logits_avx2` reads `weights[v*hidden_size..]` for
            // `v < vocab_size` and writes `logits[..vocab_size]`.
            unsafe {
                forge_llm::asm::lm_head_logits_avx2(
                    vocab_size,
                    hidden_size,
                    reg_f32.as_ptr(),
                    out_slice_ternary.as_ptr(),
                    logits.as_mut_ptr(),
                );
            }
            if debug && generated == 0 {
                let lmin = logits.iter().cloned().fold(f32::INFINITY, f32::min);
                let lmax = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let lmean = logits.iter().sum::<f32>() / vocab_size as f32;
                let lstd = (logits.iter().map(|l| (l - lmean).powi(2)).sum::<f32>() / vocab_size as f32).sqrt();
                let mut top5: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i,&l)| (i,l)).collect();
                top5.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
                top5.truncate(10);
                let top5_decoded: Vec<String> = top5.iter().map(|(i,v)| format!("{}('{}')={:.4}", i, tokenizer.decode(&[*i as u32]), v)).collect();
                println!("{}LOGITS: min={:.4} max={:.4} mean={:.4} std={:.4}{}", COLOR_DIM, lmin, lmax, lmean, lstd, COLOR_RESET);
                println!("{}TOP-10: {}{}", COLOR_DIM, top5_decoded.join(" | "), COLOR_RESET);
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

            // Repetition penalty via HashSet — O(1) lookup instead of O(n) scan
            // Stronger on recent window to break ternary loops
            if rep_penalty > 1.0 {
                let window_start = generated_tokens.len().saturating_sub(64);
                rep_window.clear();
                for &t in &generated_tokens[window_start..] {
                    rep_window.insert(t);
                }
                for &tid in &rep_window {
                    let i = tid as usize;
                    if i >= logits.len() {
                        continue;
                    }
                    if logits[i] > 0.0 {
                        logits[i] /= rep_penalty;
                    } else {
                        logits[i] *= rep_penalty;
                    }
                }
            }

            // Sampling: greedy (MUD_INFER_GREEDY=1) or Top-P nucleus with temperature
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
                // Temperature scaling
                let inv_temp = 1.0 / temperature.max(1e-6);

                // Build softmax probabilities with temperature
                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits.iter().map(|&l| ((l - max_l) * inv_temp).exp()).sum();

                // Use Fenwick tree for O(log n) cumulative probability sampling
                fenwick_logits.reset_zeroed();
                for (i, &l) in logits.iter().enumerate() {
                    let p = ((l - max_l) * inv_temp).exp() / sum_exp;
                    fenwick_logits.update(i, p);
                }

                // Top-P nucleus: find the threshold via Fenwick prefix sum
                let mut rng = rand::rng();
                let target = rng.random::<f32>() * top_p;

                // Binary search for the top-P cutoff using Fenwick prefix sums
                let mut lo = 0usize;
                let mut hi = vocab_size;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    let cum = fenwick_logits.prefix_sum(mid);
                    if cum < target {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                // Sample within the nucleus [0, lo]
                let nucleus_end = lo.max(1).min(vocab_size);
                let nucleus_sum = fenwick_logits.prefix_sum(nucleus_end);
                if nucleus_sum < 1e-12 {
                    // Fallback: greedy if nucleus is degenerate
                    logits
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                } else {
                    let r = rng.random::<f32>() * nucleus_sum;
                    // Find k-th via Fenwick walk (safe method)
                    fenwick_logits.find_kth(r)
                }
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
                        None,
                    );
                } else {
                    evaluate_slime_block(layer, l_idx, &mut ws, current_pos, rms_norm_eps, None, None);
                }
            }

            current_pos += 1;
            generated += 1;
        }
        println!();

        // Persist KV cache position for next turn (conversation history)
        kv_pos = current_pos;
        turn_count += 1;
    }

    Ok(())
}
