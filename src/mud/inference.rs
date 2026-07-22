//! Reusable single-sequence forward for inference + sanity tests.
//! Faithful port of the working context-forward path from `src/main.rs`
//! (default `MUD_INFER_SCALE_UP`, tied-embedding handling). Kept separate so
//! the CLI and unit tests share one implementation.

use crate::asm;
use crate::mud::expert_bus::ExpertScratch;
use crate::mud::moe_load;
use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_backward;
use crate::mud::slime_forward::{
    apply_output_norm, evaluate_slime_block, evaluate_slime_block_moe, SlimeLayer,
};
use crate::mud::MudFile;
use crate::mud::MudTensorType;

/// Run the full context forward over `tokens` and return logits for the last
/// position (vocab-sized `f32` vector). Returns an error if the model is missing
/// required tensors/metadata.
pub fn forward_last_logits(mud: &MudFile, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
    let mut report = None;
    forward_last_logits_inner(mud, tokens, &mut report)
}

/// Like [`forward_last_logits`] but also returns the C-MUD thinking-loop report when
/// `MUD_CMUD_THINK=1` is set (otherwise the report is default/zero).
pub fn forward_last_logits_cmud(
    mud: &MudFile,
    tokens: &[u32],
) -> anyhow::Result<(Vec<f32>, crate::mud::cmud::CmudThinkReport)> {
    let mut report = None;
    let logits = forward_last_logits_inner(mud, tokens, &mut report)?;
    Ok((logits, report.unwrap_or_default()))
}

/// One analytic training step for the complex-reasoning pass (research §4, #4 entrenable).
///
/// Runs the REAL model forward to the pre-CMUD hidden, applies `K` trainable think steps
/// (recording a tape), computes next-token cross-entropy through the EXACT LM head, and returns
/// the analytic gradient w.r.t. `CmudLayerParams` via [`cmud_backward`]. The head weight is the
/// same matrix `asm::lm_head_logits_avx2` uses, so the loss/gradient are identical to production.
///
/// `params` are taken from `MUD_CMUD_PARAMS` if set, else identity defaults. Production f32 path
/// is untouched; C-MUD stays opt-in. Caller (e.g. `cmud_train`) owns the optimizer (Adam).
// The 3-tuple (loss, grad, params) is intentionally returned whole; define an alias to satisfy
// clippy::type_complexity without obscuring the call sites.
#[allow(clippy::type_complexity)]
pub fn cmud_training_forward(
    mud: &MudFile,
    ctx: &[u32],
    target: u32,
) -> anyhow::Result<(f32, crate::mud::cmud::CmudLayerParamsGrad, crate::mud::cmud::CmudLayerParams)>
{
    use crate::mud::cmud::{
        cmud_backward, CmudLayerParams, CmudLayerParamsGrad, ThinkingState, DEFAULT_THINK_ITERS,
        CMUD_RADIUS_RMS_FACTOR,
    };
    let (_logits, reg_pre, head_w, hidden, vocab) = forward_last_hidden_and_head(mud, ctx, &mut None)?;

    // Load (or default) the trainable params.
    let params = match std::env::var("MUD_CMUD_PARAMS") {
        Ok(path) if !path.is_empty() => CmudLayerParams::load_json(std::path::Path::new(&path))
            .unwrap_or_else(|_| CmudLayerParams::from_defaults(hidden)),
        _ => CmudLayerParams::from_defaults(hidden),
    };

    // Radius auto-scaled to hidden RMS (matches `maybe_think_collapse_rms_scaled`).
    let rms = (reg_pre.iter().map(|v| v * v).sum::<f32>() / hidden.max(1) as f32).max(1e-12).sqrt();
    let radius = CMUD_RADIUS_RMS_FACTOR * rms;

    let mut st = ThinkingState::from_real(&reg_pre, radius);
    let mut tapes = Vec::new();
    for _ in 0..DEFAULT_THINK_ITERS {
        st.think_step_trainable_record(&params, &mut tapes);
    }
    let mut reg_post = vec![0.0f32; hidden];
    st.collapse_to_real(&mut reg_post);

    // Exact LM head: logits[v] = Σ_h reg_post[h] * head_w[v*hidden + h].
    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        let row = &head_w[v * hidden..v * hidden + hidden];
        let mut s = 0.0f32;
        for h in 0..hidden {
            s += reg_post[h] * row[h];
        }
        logits[v] = s;
    }

    // Cross-entropy gradient: g_logits = softmax(logits) - onehot(target).
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits.iter() {
        sum += (l - max).exp();
    }
    let loss = match logits.get(target as usize) {
        Some(&lv) => -(lv - max) + sum.ln(),
        None => 0.0,
    };
    let mut g_logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        g_logits[v] = (logits[v] - max).exp() / sum;
    }
    if (target as usize) < vocab {
        g_logits[target as usize] -= 1.0;
    }

    // ∂L/∂reg_post = head_w^T · g_logits.
    let mut grad_reg = vec![0.0f32; hidden];
    for v in 0..vocab {
        let gv = g_logits[v];
        if gv == 0.0 {
            continue;
        }
        let row = &head_w[v * hidden..v * hidden + hidden];
        for h in 0..hidden {
            grad_reg[h] += gv * row[h];
        }
    }

    let mut grad = CmudLayerParamsGrad::default();
    // `cmud_backward` expects `∂L/∂mixed` (pre-collapse residual output of the last step). `grad_reg`
    // is `∂L/∂reg_post` where `reg_post = |mixed|`, so chain through the collapse: `∂L/∂mixed =
    // (∂L/∂|mixed|)·sign(mixed)`. `sign(mixed)` is reconstructed from the last recorded tape (the
    // collapse is |·| on a purely-real value; `reg_post` alone loses the sign).
    let last = tapes.last().expect("cmud_training_forward records ≥1 think step");
    let mut grad_mixed = grad_reg;
    for (gm, (hb, ai)) in grad_mixed
        .iter_mut()
        .zip(last.h_before.iter().zip(last.attn.iter()))
    {
        let mixed = hb + last.alpha * (ai - hb);
        *gm *= mixed.signum();
    }
    cmud_backward(&tapes, &grad_mixed, &params, &mut grad);
    Ok((loss, grad, params))
}

/// Full forward, returning the final logits PLUS the pre-CMUD hidden (`reg_pre`, post output-norm)
/// and the exact LM-head weight matrix (`head_w`). `head_w` is laid out row-major `vocab×hidden`
/// and already includes the same scaling `asm::lm_head_logits_avx2` receives, so a training
/// backward using it is numerically identical to the production head.
#[allow(clippy::type_complexity)]
pub fn forward_last_hidden_and_head(
    mud: &MudFile,
    tokens: &[u32],
    cmud_capture: &mut Option<crate::mud::cmud::CmudThinkReport>,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, usize, usize)> {
    let core = mud
        .skills
        .get("core")
        .ok_or_else(|| anyhow::anyhow!("Missing core skill"))?;

    let m = |k: &str| -> usize {
        mud.global_metadata
            .get(k)
            .and_then(|v| v.parse().ok())
            .expect("missing metadata")
    };
    let hidden_size = m("hidden_size");
    let num_layers = mud
        .global_metadata
        .get("num_hidden_layers")
        .or_else(|| mud.global_metadata.get("num_layers"))
        .and_then(|v| v.parse().ok())
        .expect("Missing num_layers");
    let n_heads = mud
        .global_metadata
        .get("num_attention_heads")
        .or_else(|| mud.global_metadata.get("num_heads"))
        .and_then(|v| v.parse().ok())
        .expect("Missing num_heads");
    let n_kv_heads = mud
        .global_metadata
        .get("num_key_value_heads")
        .or_else(|| mud.global_metadata.get("num_kv_heads"))
        .and_then(|v| v.parse().ok())
        .expect("Missing num_kv_heads");
    let ffn_mid = mud
        .global_metadata
        .get("intermediate_size")
        .or_else(|| mud.global_metadata.get("ffn_hidden"))
        .and_then(|v| v.parse().ok())
        .expect("Missing ffn_mid");
    let vocab_size = m("vocab_size");
    let rms_norm_eps = mud
        .global_metadata
        .get("rms_norm_eps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5);
    let max_pos = m("max_position_embeddings");
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
                .map(|t| {
                    t.owned_data
                        .as_ref()
                        .map(|d| d.as_ptr())
                        .unwrap_or(t.data_ptr)
                })
                .unwrap_or(std::ptr::null())
        };
        let ts = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}.prq_scale", prefix, name))
                .map(|t| {
                    t.owned_data
                        .as_ref()
                        .map(|d| d.as_ptr() as *const f32)
                        .unwrap_or(t.data_ptr as *const f32)
                })
                .unwrap_or(std::ptr::null())
        };
        let tn = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}.weight", prefix, name))
                .map(|t| {
                    t.owned_data
                        .as_ref()
                        .map(|d| d.as_ptr() as *const f32)
                        .unwrap_or(t.data_ptr as *const f32)
                })
                .unwrap_or(std::ptr::null())
        };
        let ffn_norm_w = {
            let a = tn("ffn_norm");
            if !a.is_null() {
                a
            } else {
                tn("norm")
            }
        };
        let ffn_names = moe_load::dense_ffn_names_for_train(&core.tensors, blk);
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

    let emb_tensor = core
        .tensors
        .get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("Missing token_embd.weight"))?;
    let mut emb_f32 = vec![0.0f32; vocab_size * hidden_size];
    let emb_rms_sum_local = if emb_tensor.t_type == MudTensorType::Float32 {
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
        slime_backward::unpack_ternary2bit_to_f32(
            emb_packed,
            emb_scales,
            hidden_size,
            &mut emb_f32,
        );
        emb_f32.iter().map(|&x| x * x).sum::<f32>() / (hidden_size as f32)
    };

    let out_tensor = core.tensors.get("output.weight").unwrap_or(emb_tensor);
    let tied_output = std::ptr::eq(out_tensor as *const _, emb_tensor as *const _);
    let mut out_f32_owned: Option<Vec<f32>> = if tied_output {
        None
    } else {
        let mut out_dq = vec![0.0f32; vocab_size * hidden_size];
        if out_tensor.t_type == MudTensorType::Float32 {
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
            slime_backward::unpack_ternary2bit_to_f32(
                out_packed,
                out_scales,
                hidden_size,
                &mut out_dq,
            );
        }
        Some(out_dq)
    };

    let _emb_rms = (emb_rms_sum_local / vocab_size as f32)
        .sqrt()
        .max(crate::mud::constants::EPSILON_FLOOR);
    let emb_is_ternary = emb_tensor.t_type == MudTensorType::Ternary2Bit;
    let _ = emb_is_ternary;
    let scale_up = 1.0f32; // default MUD_INFER_SCALE_UP behavior
    let logit_scale = if !tied_output {
        1.0 / (hidden_size as f32).sqrt()
    } else {
        1.0
    };
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
    let out_slice_ternary: &[f32] = out_f32_owned.as_deref().unwrap_or(emb_slice);
    // Effective LM-head weight matrix (already scaled exactly as `asm::lm_head_logits_avx2`
    // receives it), owned for the training backward. `out_slice_ternary` already points at the
    // right matrix (tied embedding or dequant `output.weight`), so copy it — no move of `out_f32_owned`.
    let head_w: Vec<f32> = out_slice_ternary.to_vec();

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

    let top_k = moe_load::default_top_k();
    let moe_buses = moe_load::load_model_buses(mud, num_layers, hidden_size, ffn_mid, top_k);
    let _multi = moe_load::model_has_multi_expert(&moe_buses);
    let mut moe_scratch = ExpertScratch::new(hidden_size, ffn_mid, top_k.max(8));

    ws.kv_cache.fill(0.0);
    ws.v_cache.fill(0.0);
    ws.jepa_mu.fill(0.0);
    ws.jepa_inv_sigma.fill(0.0);
    ws.jepa_var_ema.fill(0.0);

    let vocab_size = if hidden_size > 0 { emb_slice.len() / hidden_size } else { 0 };
    let mut current_pos = 0;
    while current_pos < tokens.len() {
        ws.clear_registers();
        let tid = (tokens[current_pos] as usize).min(vocab_size.saturating_sub(1));
        let emb_start = tid * hidden_size;
        for (i, v) in emb_slice[emb_start..emb_start + hidden_size]
            .iter()
            .enumerate()
        {
            crate::mud::slime::SlimeRegister::init_from_embed(
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

    apply_output_norm(&mut ws, out_norm_slice.as_ptr(), rms_norm_eps);
    let mut logits = vec![0.0f32; vocab_size];
    let mut reg_f32 = vec![0.0f32; hidden_size];
    for (i, r) in ws.registers.iter().enumerate().take(hidden_size) {
        reg_f32[i] = r.matmul_accum;
    }
    // Optional C-MUD complex reasoning pass (research §3): no-op unless MUD_CMUD_THINK=1.
    // Radius auto-scaled to hidden RMS inside the helper.
    {
        let rep = crate::mud::cmud::maybe_think_collapse_rms_scaled(&mut reg_f32);
        *cmud_capture = Some(rep);
    }
    unsafe {
        asm::lm_head_logits_avx2(
            vocab_size,
            hidden_size,
            reg_f32.as_ptr(),
            out_slice_ternary.as_ptr(),
            logits.as_mut_ptr(),
        );
    }
    Ok((logits, reg_f32, head_w, hidden_size, vocab_size))
}

/// Thin wrapper keeping the legacy `(mud, tokens) -> logits` signature used by the rest of the
/// crate; see [`forward_last_hidden_and_head`] for the training variant.
fn forward_last_logits_inner(
    mud: &MudFile,
    tokens: &[u32],
    cmud_capture: &mut Option<crate::mud::cmud::CmudThinkReport>,
) -> anyhow::Result<Vec<f32>> {
    let (logits, _, _, _, _) = forward_last_hidden_and_head(mud, tokens, cmud_capture)?;
    Ok(logits)
}

/// Lightweight health gate (T0.3): a model whose forward already collapses to
/// token-0 across diverse prompts cannot be repaired by the training circuit.
/// Returns `true` when the model is collapsed (token-0 is the argmax for more
/// than half of the probe prompts). A load/forward error is treated as
/// collapsed so the circuit refuses to run rather than panic mid-loop.
pub fn model_logits_collapsed(mud: &MudFile) -> bool {
    let tokens = mud
        .global_metadata
        .get("tokenizer.tokens")
        .cloned()
        .unwrap_or_default();
    let merges = mud
        .global_metadata
        .get("tokenizer.merges")
        .cloned()
        .unwrap_or_default();
    let tk = crate::model::tokenizer::Tokenizer::from_mud_metadata(&tokens, &merges);
    let prompts = ["Hello world", "The cat sat", "In 2024 the", "A B C D"];
    let mut dominance = 0usize;
    let mut total = 0usize;
    for p in prompts {
        let ids = tk.encode_simple(p);
        if ids.is_empty() {
            continue;
        }
        let lg = match forward_last_logits(mud, &ids) {
            Ok(l) => l,
            Err(_) => return true,
        };
        let argmax = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        if argmax == 0 {
            dominance += 1;
        }
        total += 1;
    }
    total > 0 && dominance > total / 2
}

/// Readable, validated audit of the C-MUD complex-reasoning path (research §3):
/// runs a real forward with `MUD_CMUD_THINK=1` over probe prompts and checks that
/// logits stay finite/non-degenerate and the Hermitian ball + phase-lock hold.
pub struct CmudAudit {
    pub forward_ok: bool,
    pub logits_finite: bool,
    /// Smallest dynamic range (max−min) across prompts; must be > 0.
    pub logit_range_min: f32,
    /// `true` when token-0 is the argmax for > half of the prompts (collapse smell).
    pub token0_dominant: bool,
    /// Thinking-loop metrics from the last prompt.
    pub think: crate::mud::cmud::CmudThinkReport,
    /// Spectral-collapse health of the complex manifold (research §3.3, Cauchy/free-prob).
    pub spectral: crate::mud::cmud::SpectralHealth,
}

impl CmudAudit {
    /// All gates green ⇒ the complex reasoning path is safe to use.
    pub fn healthy(&self) -> bool {
        self.forward_ok
            && self.logits_finite
            && self.logit_range_min > 0.0
            && !self.token0_dominant
            && self.think.max_herm_norm <= self.think.radius * 1.01
            && !self.spectral.collapsed
    }

    /// Human-readable metric lines for audit tools.
    pub fn summary_lines(&self) -> Vec<String> {
        let t = &self.think;
        let s = &self.spectral;
        vec![
            format!("  forward_ok      : {}", self.forward_ok),
            format!("  logits_finite   : {}", self.logits_finite),
            format!("  logit_range_min : {:.4}  (must be > 0)", self.logit_range_min),
            format!("  token0_dominant : {}", self.token0_dominant),
            format!("  think τ steps   : {}", t.steps),
            format!("  phase_locked    : {}", t.phase_locked),
            format!(
                "  herm norm max   : {:.4} / radius {:.4}  (ball respected: {})",
                t.max_herm_norm,
                t.radius,
                t.max_herm_norm <= t.radius * 1.01
            ),
            format!(
                "  spectral        : mag_spread={:.4} phase_R={:.4} cauchy|G(2)|={:.4} collapsed={}",
                s.spread_mag, s.circular_phase_r, s.cauchy_mag_at_2, s.collapsed
            ),
        ]
    }
}

/// Run the C-MUD audit over the standard probe prompts. Temporarily enables
/// `MUD_CMUD_THINK=1` and restores the previous env value afterwards.
pub fn cmud_audit(mud: &MudFile) -> CmudAudit {
    use crate::mud::cmud::CmudThinkReport;
    let prompts = ["Hello world", "The cat sat", "In 2024 the", "A B C D"];
    let tokens = mud
        .global_metadata
        .get("tokenizer.tokens")
        .cloned()
        .unwrap_or_default();
    let merges = mud
        .global_metadata
        .get("tokenizer.merges")
        .cloned()
        .unwrap_or_default();
    let tk = crate::model::tokenizer::Tokenizer::from_mud_metadata(&tokens, &merges);

    let prev = std::env::var("MUD_CMUD_THINK").ok();
    unsafe {
        std::env::set_var("MUD_CMUD_THINK", "1");
    }

    let mut out = CmudAudit {
        forward_ok: true,
        logits_finite: true,
        logit_range_min: f32::INFINITY,
        token0_dominant: false,
        think: CmudThinkReport::default(),
        spectral: crate::mud::cmud::SpectralHealth::default(),
    };
    let mut tok0 = 0usize;
    let mut total = 0usize;
    let mut last_report = CmudThinkReport::default();

    for p in prompts {
        let ids = tk.encode_simple(p);
        if ids.is_empty() {
            continue;
        }
        match forward_last_logits_cmud(mud, &ids) {
            Ok((lg, rep)) => {
                last_report = rep;
                out.spectral = rep.spectral;
                total += 1;
                if !lg.iter().all(|l| l.is_finite()) {
                    out.logits_finite = false;
                }
                let max = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = lg.iter().cloned().fold(f32::INFINITY, f32::min);
                out.logit_range_min = out.logit_range_min.min(max - min);
                let am = lg
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if am == 0 {
                    tok0 += 1;
                }
            }
            Err(_) => out.forward_ok = false,
        }
    }

    if out.logit_range_min == f32::INFINITY {
        out.logit_range_min = 0.0;
    }
    out.token0_dominant = total > 0 && tok0 > total / 2;
    out.think = last_report;

    match prev {
        Some(v) => unsafe {
            std::env::set_var("MUD_CMUD_THINK", v);
        },
        None => unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        },
    }
    out
}

/// Quality probe: compares a plain forward (baseline) vs the C-MUD reasoning
/// forward (with `MUD_CMUD_THINK=1`) on a probe prompt, reporting how much the
/// complex thinking path shifts the output distribution.
pub struct CmudCompare {
    pub baseline_argmax: usize,
    pub cmud_argmax: usize,
    pub argmax_changed: bool,
    pub logit_l2: f32,
    pub baseline_entropy: f32,
    pub cmud_entropy: f32,
}

fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    if sum <= 0.0 {
        return 0.0;
    }
    -logits
        .iter()
        .map(|&l| {
            let p = (l - max).exp() / sum;
            if p > 0.0 {
                p * p.ln()
            } else {
                0.0
            }
        })
        .sum::<f32>()
}

fn argmax_index(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

pub fn cmud_compare(mud: &MudFile) -> CmudCompare {
    let prompt = "The quick brown fox";
    let tokens = mud
        .global_metadata
        .get("tokenizer.tokens")
        .cloned()
        .unwrap_or_default();
    let merges = mud
        .global_metadata
        .get("tokenizer.merges")
        .cloned()
        .unwrap_or_default();
    let tk = crate::model::tokenizer::Tokenizer::from_mud_metadata(&tokens, &merges);
    let ids = tk.encode_simple(prompt);

    let base = forward_last_logits(mud, &ids).unwrap_or_default();

    let prev = std::env::var("MUD_CMUD_THINK").ok();
    unsafe {
        std::env::set_var("MUD_CMUD_THINK", "1");
    }
    let (cmud, _) = forward_last_logits_cmud(mud, &ids).unwrap_or_default();
    match prev {
        Some(v) => unsafe {
            std::env::set_var("MUD_CMUD_THINK", v);
        },
        None => unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        },
    }

    let n = base.len().min(cmud.len());
    let mut l2 = 0.0f32;
    for i in 0..n {
        let d = base[i] - cmud[i];
        l2 += d * d;
    }
    let ba = argmax_index(&base);
    let ca = argmax_index(&cmud);
    CmudCompare {
        baseline_argmax: ba,
        cmud_argmax: ca,
        argmax_changed: ba != ca,
        logit_l2: l2.sqrt(),
        baseline_entropy: softmax_entropy(&base),
        cmud_entropy: softmax_entropy(&cmud),
    }
}

#[cfg(test)]
mod tests {
    use crate::model::tokenizer::Tokenizer;
    use crate::mud::MudFile;
    use std::path::Path;

    fn load_model() -> Option<(MudFile, Tokenizer)> {
        let path = "models/smollm2.mud";
        if !Path::new(path).exists() {
            return None;
        }
        let mud = MudFile::load(path).ok()?;
        let tokens = mud
            .global_metadata
            .get("tokenizer.tokens")
            .cloned()
            .unwrap_or_default();
        let merges = mud
            .global_metadata
            .get("tokenizer.merges")
            .cloned()
            .unwrap_or_default();
        let tk = Tokenizer::from_mud_metadata(&tokens, &merges);
        Some((mud, tk))
    }

    #[test]
    fn forward_sanity() {
        let Some((mud, tk)) = load_model() else {
            return;
        };
        let prompt = "The quick brown fox";
        let ids = tk.encode_simple(prompt);
        assert!(!ids.is_empty(), "prompt must tokenize");
        let logits = crate::mud::inference::forward_last_logits(&mud, &ids).expect("forward");
        let vocab = mud
            .global_metadata
            .get("vocab_size")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        assert_eq!(logits.len(), vocab);

        // 1) finite logits
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "logits must be finite (no NaN/Inf)"
        );

        // 2) not collapsed to a single token (no token-0 dominance across prompts)
        let mut dominance = 0usize;
        let mut total = 0usize;
        for p in ["Hello world", "The cat sat", "In 2024 the", "A B C D"] {
            let ids = tk.encode_simple(p);
            if ids.is_empty() {
                continue;
            }
            let lg = crate::mud::inference::forward_last_logits(&mud, &ids).expect("forward");
            let argmax = lg
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();
            if argmax == 0 {
                dominance += 1;
            }
            total += 1;
        }
        assert!(total > 0);
        assert!(
            dominance <= total / 2,
            "token-0 dominates {dominance}/{total} prompts (collapsed logits)"
        );

        // 3) non-trivial distribution (entropy > 0): not a single spike, not uniform.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_l = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max_l > min_l, "logits must have dynamic range (entropy>0)");
        // softmax entropy over logits
        let max_l = max_l.max(min_l);
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let entropy = -logits
            .iter()
            .map(|&l| {
                let p = (l - max_l).exp() / sum_exp;
                if p > 0.0 {
                    p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f32>();
        assert!(entropy > 0.0, "logit distribution must have entropy > 0");
        assert!(
            entropy < (vocab as f32).ln(),
            "logits must not be uniform (entropy < log(vocab))"
        );
    }

    #[test]
    fn model_logits_not_collapsed() {
        let Some((mud, _tk)) = load_model() else {
            return;
        };
        assert!(
            !crate::mud::inference::model_logits_collapsed(&mud),
            "healthy smollm2.mud must NOT trip the collapse gate"
        );
    }

    #[test]
    fn cmud_audit_runs_and_reports() {
        let Some((mud, _tk)) = load_model() else {
            return;
        };
        let cka = crate::mud::inference::cmud_audit(&mud);
        // Forward + audit integration must be clean. (The thinking-loop itself is
        // validated directly by cmud::tests::test_maybe_think_collapse_report_respects_ball;
        // here we avoid asserting steps>0 because the harness runs env-setters in parallel.)
        assert!(cka.forward_ok, "C-MUD forward must succeed");
        assert!(cka.logits_finite, "C-MUD logits must be finite");
        assert!(
            cka.think.max_herm_norm <= cka.think.radius * 1.01,
            "Hermitian ball must be respected"
        );
    }

    #[test]
    fn cmud_compare_runs() {
        let Some((mud, _tk)) = load_model() else {
            return;
        };
        let c = crate::mud::inference::cmud_compare(&mud);
        assert!(c.logit_l2.is_finite(), "logit L2 delta must be finite");
        assert!(c.baseline_entropy >= 0.0 && c.cmud_entropy >= 0.0);
    }

    /// Regression guard for the over-sharp C-MUD bug: the complex reasoning path must NOT
    /// collapse the output to a single token (entropy≈0) nor wash it out to uniform (entropy≈max).
    /// It must also preserve logit magnitude (bounded L2 vs baseline).
    #[test]
    fn cmud_compare_not_degenerate() {
        let Some((mud, _tk)) = load_model() else {
            return;
        };
        let c = crate::mud::inference::cmud_compare(&mud);
        assert!(c.cmud_entropy > 0.05, "C-MUD must not over-sharpen (entropy≈0)");
        // vocab ~49k -> max entropy ~10.8; reject a washed-out near-uniform output
        assert!(
            c.cmud_entropy < 9.0,
            "C-MUD must not wash out to uniform (entropy≈max)"
        );
        assert!(
            c.logit_l2 < 20000.0,
            "C-MUD must preserve logit magnitude (no blow-up)"
        );
    }

    #[test]
    fn cmud_think_forward_smoke() {
        let Some((mud, tk)) = load_model() else {
            return;
        };
        // Exercise the full forward with the C-MUD complex reasoning pass enabled.
        unsafe {
            std::env::set_var("MUD_CMUD_THINK", "1");
        }
        let prompt = "The quick brown fox";
        let ids = tk.encode_simple(prompt);
        assert!(!ids.is_empty(), "prompt must tokenize");
        let logits = crate::mud::inference::forward_last_logits(&mud, &ids).expect("forward");
        unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        }
        // reasoning pass must keep logits finite and non-degenerate
        assert!(
            logits.iter().all(|l| l.is_finite()),
            "C-MUD reasoning logits must be finite"
        );
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_l = logits.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max_l > min_l, "C-MUD logits must keep dynamic range");
        assert_eq!(logits.len(), mud.global_metadata.get("vocab_size").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0));
    }
}
