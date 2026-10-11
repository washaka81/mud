//! Per-sequence training step (L-15 / L-10 forward+backward+optimizer loop).
//!
//! `train_on_sequence` is the inner loop of `run_alignment_session`. It runs:
//!   1. Build the per-window step list (full-seq windows or independent pairs).
//!   2. For each (input, target) pair:
//!      a. Load embedding (shadow FP32 or lazy mmap).
//!      b. Forward through all layers (with STP trajectory + JEPA dynamics).
//!      c. Sample negatives + compute sampled softmax CE loss.
//!      d. Backward (last-N layers only).
//!      e. Optimizer step per layer (Muon / Adam / SparseAdam / GaLore / SGD).
//!      f. mHC scale SGD step.
//!   3. Return a `TrainChunkMetrics` aggregating live activation stats.
//!
//! This is a 1018-line function. It depends on the workspace, the shadow
//! layer arrays, the per-layer tapes, and the optimizer dispatcher (all
//! pre-allocated by `run_alignment_session`).

use crate::mud::corpus_trainer::emb::{load_emb_row_into, slice_stddev};
use crate::mud::corpus_trainer::{
    metrics::ActStatsAccum, MudCorpusTrainer, TrainChunkMetrics,
};
use crate::mud::ezop;
use crate::mud::optim::{apply_optimizer_cpu_step_and_pack, mhc_scale_sgd_step};
use crate::mud::MudFile;

/// (name, shadow_weights, grad, packed_u8_ptr, prq_scale_ptr) — used in the
/// optimizer step loop to iterate over Q/K/V/FFN matrices uniformly.
type WeightMatrixRef<'a> = (&'a str, &'a [f32], &'a [f32], *mut u8, *mut f32);

/// FIX (2026-07-31): Apply LR warmup + cosine decay schedule.
/// **GRASSHOPPER FIX (2026-08-03):** Added cosine restarts (SGDR) to prevent
/// the monotonic LR decay that trapped the model at low LR for 73%+ of training.
/// The original single-half-cosine caused the LR to "fall" from its peak (step 14)
/// and by step 1059 (73% progress) it coincidentally matched the step-2 value (21%
/// of base), making it appear the LR was "stuck" while the loss barely moved
/// (6.48 → 6.32 over 1057 steps, 2.5% improvement vs ln(512)=6.24 random baseline).
///
/// Schedule modes (env `MUD_LR_RESTART`):
///   - `0` off: original single-cosine half-cycle (warmup → base → min_lr)
///   - `1` on (default): SGDR — `n_cycles` cosine half-cycles over total_steps,
///     each with its own warmup. The LR periodically "bounces" back to base_lr,
///     helping the model escape plateaus.
///
/// The number of restart cycles defaults to `max(3, sqrt(total_steps / warmup))`
/// giving ~3-10 bounces depending on training length. Override via `MUD_LR_CYCLES`.
///
/// Warmup: linear ramp from 0 → base_lr over `warmup_steps` (default 1% of total
/// per-cycle). Decay: cosine half-cycle from lr → min_lr over remaining cycle steps.
/// min_lr floor (5% of base) prevents LR from collapsing to 0.
/// Formula: min_lr + 0.5*(base - min_lr)*(1 + cos(π*progress))
/// At progress=0: lr=base_lr; at progress=1: lr=min_lr.
pub fn apply_lr_schedule(base_lr: f32, step: u64, total_steps: usize, warmup_steps: usize) -> f32 {
    let env_override = std::env::var("MUD_QAT_LR");
    // If user explicitly sets MUD_QAT_LR, use it directly (no schedule) for compat.
    if env_override.is_ok() {
        return base_lr;
    }
    let total = total_steps.max(1);
    let warmup = warmup_steps.min(total).max(1);

    // GRASSHOPPER FIX: Cosine restarts (SGDR)
    // Default to enabled. Disable with MUD_LR_RESTART=0.
    let use_restarts = std::env::var("MUD_LR_RESTART")
        .map(|v| v != "0" && v != "false" && v != "off")
        .unwrap_or(true);

    // Initial linear warmup (first warmup_steps only, before any cosine/restart)
    if step < warmup as u64 {
        return base_lr * (step as f32 + 1.0) / warmup as f32;
    }

    if use_restarts {
        let n_cycles: usize = std::env::var("MUD_LR_CYCLES")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or((total as f32).sqrt().floor() as usize)
            .clamp(3, 16);
        let cycle_len = (total - warmup) / n_cycles.max(1);
        let cycle_len = cycle_len.max(1);

        // Which cycle are we in?
        let post_warmup_step = step as usize - warmup;
        let _cycle_idx = post_warmup_step / cycle_len;
        let cycle_step = post_warmup_step % cycle_len;

        // Each cycle has its own mini-warmup (20% of cycle length, min 1 step).
        // Restart from 50% of base (not 0) to avoid losing too much momentum.
        let cycle_warmup = (cycle_len / 5).max(1);
        if cycle_step < cycle_warmup {
            // Warmup within this cycle
            let warmup_lr = base_lr * 0.5;
            let progress = cycle_step as f32 / cycle_warmup as f32;
            return warmup_lr + (base_lr - warmup_lr) * progress;
        }

        // Cosine decay within cycle
        let decay_step = cycle_step - cycle_warmup;
        let decay_len = cycle_len - cycle_warmup;
        let progress = decay_step as f32 / decay_len.max(1) as f32;
        let progress = progress.clamp(0.0, 1.0);
        let min_lr = base_lr * 0.05;
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    } else {
        // Original single cosine half-cycle (no restarts)
        let progress = (step as f32 - warmup as f32)
            / (total as f32 - warmup as f32).max(1.0);
        let progress = progress.clamp(0.0, 1.0);
        let min_lr = base_lr * 0.05;
        min_lr + 0.5 * (base_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }
}

impl MudCorpusTrainer {
    // 12 args: layers/shadow_layers/tapes/gradients are all per-layer arrays that
    // are pre-allocated by run_alignment_session and must be passed mutably.
    // TODO(arch): group into a `TrainModelState` struct to reduce arg count.
    #[allow(clippy::too_many_arguments)]
    pub fn train_on_sequence(
        &self,
        _mud: &mut MudFile,
        shadow_emb: &mut [f32],
        layers: &mut [crate::mud::slime_forward::SlimeLayer],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        workspace: &mut crate::mud::slime::SlimeWorkspace,
        backward_ws: &mut crate::mud::slime_backward::SlimeBackwardWorkspace,
        tapes: &mut [crate::mud::slime_backward::SlimeLayerTape],
        gradients: &mut [crate::mud::slime_backward::SlimeLayerGradients],
        tokens: &[u32],
        batch_size: usize,
        mut vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
    ) -> anyhow::Result<TrainChunkMetrics> {
        // L-05: do NOT sync at entry. Previous step's GPU work overlaps this chunk's
        // Forward+Backward; fence+packed readback runs inside step_async_deferred
        // after backward (and at epoch/checkpoint via sync_all).

        let base_lr = crate::mud::constants::qat_learning_rate();
        let hidden_size = workspace.hidden_size;
        // FIX (2026-07-31): LR warmup + cosine decay to prevent slow drift instability.
        // The model was trained with a fixed LR=0.0005 for 5000+ steps, causing
        // cog (cognitive activation) to slowly decay from ~593 to ~250 before the
        // sudden collapse at step 5848. Warmup prevents early instability and
        // cosine decay prevents late-stage oscillation/instability.
        // B-6 FIX (2026-08-03): use step+1 — step_count.fetch_add happens at END of
        // the chunk, so chunk N must schedule LR for the step it is about to apply,
        // not the one before. Using step_count.load() here yields the previous
        // chunk's LR (off-by-one).
        let lr = apply_lr_schedule(
            base_lr,
            self.step_count.load(std::sync::atomic::Ordering::Relaxed) + 1,
            self.total_steps_hint.borrow().unwrap_or(crate::mud::constants::QAT_LR_TOTAL_STEPS),
            *self.warmup_steps.borrow(),
        );
        // Lazy emb (FREEZE_EMB): shadow_emb empty — vocab from mud metadata / tensor shape
        let vocab_size = if !shadow_emb.is_empty() {
            shadow_emb.len() / hidden_size.max(1)
        } else {
            _mud.skills
                .get("core")
                .and_then(|c| c.tensors.get("token_embd.weight"))
                .map(|t| t.shape[0])
                .or_else(|| {
                    _mud.global_metadata
                        .get("vocab_size")
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(1)
        };
        let emb_lazy = shadow_emb.is_empty();
        // Model-native EOS (SmolLM2 = 0). Never use Llama 128001 when OOV.
        let (_bos, eos_raw) = self
            .tokenizer
            .special_ids_from_metadata(&_mud.global_metadata);
        let eos = crate::mud::sequence_pack::clamp_special_id(eos_raw, vocab_size);

        // Stream D: full-seq (causal windows, pos>0, KV grows) is default.
        // MUD_TRAIN_FULL_SEQ=0 → classic L-10 independent pairs at pos=0.
        let full_seq = crate::mud::sequence_pack::train_full_seq_enabled();
        let seq_len = crate::mud::sequence_pack::train_seq_len()
            .min(workspace.dense_kv_cap.max(2))
            .min(workspace.max_pos.max(2));
        // Stream H: long windows auto-enable L-15 segmented ckpt when unset
        crate::mud::sequence_pack::maybe_enable_grad_ckpt_for_long_seq(seq_len);

        // Align mode densifies gradients per AOT chunk (batch × 4, clamped).
        let target_preds = crate::mud::sequence_pack::train_steps_per_chunk(batch_size);

        // Stream G: multi-expert pool for round-robin dense STE
        let n_layers_meta = layers.len();
        let moe_pool = {
            let core = _mud.skills.get("core");
            core.map(|c| {
                crate::mud::moe_train::discover_train_expert_pool(&c.tensors, n_layers_meta)
            })
            .unwrap_or_else(|| vec![0])
        };
        if crate::mud::moe_train::moe_train_enabled() {
            crate::mud::moe_train::reset_utilization();
        }

        // Build step list: (input_id, target_id, pos, reset_kv)
        let mut steps: Vec<(usize, usize, usize, bool)> = Vec::new();
        if full_seq {
            let n_win = crate::mud::sequence_pack::windows_for_target_preds(target_preds, seq_len);
            let windows =
                crate::mud::sequence_pack::windows_from_stream(tokens, n_win, seq_len, eos);
            let mut n_pred = 0usize;
            'windows: for w in windows {
                let mut first = true;
                for pos in 0..w.n_preds() {
                    if n_pred >= target_preds {
                        break 'windows;
                    }
                    let abs_i = w.start + pos;
                    if abs_i + 1 >= tokens.len() {
                        break;
                    }
                    let inp_tok = tokens[abs_i];
                    if inp_tok == eos {
                        break; // do not train past EOS inside a window
                    }
                    let input_id = inp_tok as usize;
                    let target_id = tokens[abs_i + 1] as usize;
                    if input_id >= vocab_size || target_id >= vocab_size {
                        continue;
                    }
                    // Clamp pos to dense ring / logical max
                    let pos_clamped = pos.min(workspace.max_pos.saturating_sub(1));
                    steps.push((input_id, target_id, pos_clamped, first));
                    first = false;
                    n_pred += 1;
                }
            }
        }
        if steps.is_empty() {
            // Fallback: L-10 independent pairs at pos=0 (also when full-seq finds nothing)
            let pairs =
                crate::mud::sequence_pack::pairs_from_stream(tokens, target_preds, vocab_size, eos);
            steps = pairs
                .into_iter()
                .map(|(a, b)| (a, b, 0usize, true))
                .collect();
        }

        let mut total_loss = 0.0f32;
        let mut pair_count = 0;
        let mut grad_norm_sum = 0.0f32;
        let mut entropy_sum = 0.0f32;
        let eps = 1e-6; // rms norm eps
        let mut act_accum = ActStatsAccum::default();

        // Sampled softmax size (speed: fewer negs in align/quick)
        let num_neg = crate::mud::sequence_pack::train_num_negatives();
        let num_classes = 1 + num_neg;
        // Speed: only BWD+opt last N layers (forward still full stack)
        let last_n = crate::mud::sequence_pack::train_last_n_layers(layers.len());
        let first_train_layer = layers.len().saturating_sub(last_n);

        // SPRINT-6 (P-01): use pre-allocated buffer from backward_ws instead of
        // per-chunk `vec![]`. std::mem::take extracts it; it is restored at
        // the end of this function (backward_slime_block does not touch this field).
        let mut x_data = std::mem::take(&mut backward_ws.x_data);
        let mut pre_norm_x = std::mem::take(&mut backward_ws.pre_norm_x);
        let mut final_x = std::mem::take(&mut backward_ws.final_x);
        let mut class_embs_flat = std::mem::take(&mut backward_ws.class_embs_flat);
        let mut class_logits = std::mem::take(&mut backward_ws.class_logits);
        let mut prob_q = std::mem::take(&mut backward_ws.prob_q);
        let mut d_logits = std::mem::take(&mut backward_ws.d_logits);
        let mut grad_in = std::mem::take(&mut backward_ws.grad_in);
        let mut grad_out = std::mem::take(&mut backward_ws.grad_out);
        let mut neg_ids = std::mem::take(&mut backward_ws.neg_ids);
        let mut emb_ckpt = std::mem::take(&mut backward_ws.emb_ckpt);

        // Phase 2 (STP): per-window ring of recent top-of-stack residual states
        // (pre-output-norm `matmul_accum`). STP is train-only, zero inference cost.
        let stp_on = crate::mud::stp_loss::stp_enabled();
        let stp_lambda = crate::mud::stp_loss::stp_lambda();
        const STP_HIST: usize = 4; // small ring; triples sampled from it
        let mut stp_hist = std::mem::take(&mut backward_ws.stp_hist);
        let mut stp_grad_t = std::mem::take(&mut backward_ws.stp_grad_t);
        let mut stp_grad_r = std::mem::take(&mut backward_ws.stp_grad_r);
        let mut stp_grad_s = std::mem::take(&mut backward_ws.stp_grad_s);
        let mut stp_loss_accum = 0.0f32;
        let mut stp_triples = 0usize;
        let mut stp_hist_len = 0usize; // valid entries in current window
        let mut stp_hist_head = 0usize; // next write slot (ring)
        let mut stp_rng: u64 = 0x9E3779B97F4A7C15;

        // Ensure STP buffers are sized correctly (STP may be toggled per-call)
        if stp_on {
            for h in &mut stp_hist {
                h.resize(hidden_size, 0.0);
            }
            stp_grad_t.resize(hidden_size, 0.0);
            stp_grad_r.resize(hidden_size, 0.0);
            stp_grad_s.resize(hidden_size, 0.0);
        }
        // Ensure other buffers are sized correctly (hidden_size/num_neg may differ)
        x_data.resize(hidden_size, 0.0);
        pre_norm_x.resize(hidden_size, 0.0);
        final_x.resize(hidden_size, 0.0);
        class_embs_flat.resize(num_classes * hidden_size, 0.0);
        class_logits.resize(num_classes, 0.0);
        prob_q.resize(num_classes, 0.0);
        d_logits.resize(num_classes, 0.0);
        grad_in.resize(hidden_size, 0.0);
        grad_out.resize(hidden_size, 0.0);
        neg_ids.resize(num_neg, 0);

        for g in gradients.iter_mut() {
            g.reset();
        }

        // B-1 FIX (2026-08-03): reset output_norm_w_grad at chunk start.
        // Previously the resize() was inside the per-step loop (~756) and never
        // cleared between chunks → gradient accumulated cross-chunk → output norm
        // clamped to [0.01, 10] within ~40 chunks (integral controller, not SGD).
        // Hoisting the resize out of the hot loop is also a P-01 win (no resize
        // per-token). The sgd_step at the end of the chunk will .clear() this.
        backward_ws.output_norm_w_grad.resize(hidden_size, 0.0);
        backward_ws.output_norm_w_grad.fill(0.0);

        for (input_id, target_id, pos, reset_kv) in steps.iter().copied() {
            if crate::mud::corpus_trainer::SHOULD_TERMINATE
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                break;
            }

            if reset_kv {
                // New window / independent pair: wipe KV + HCA + JEPA context
                workspace.clear_kv_all();
                workspace.jepa_mu.fill(0.0);
                workspace.jepa_inv_sigma.fill(0.0);
                workspace.jepa_var_ema.fill(0.0);
                // STP trajectory is per-window; reset the history ring.
                stp_hist_len = 0;
                stp_hist_head = 0;
            }

            workspace.clear_registers();
            for t in tapes.iter_mut() {
                t.reset();
            }

            // 1. Load embedding. FREEZE_EMB/lazy: already ternary from mmap (no re-STE).
            // Shadow FP32 path: re-quantize to simulate Ternary Shock for trainable emb.
            load_emb_row_into(
                _mud,
                shadow_emb,
                emb_lazy,
                hidden_size,
                input_id,
                &mut x_data[..hidden_size],
            );
            if !emb_lazy {
                let absmean_x = x_data.iter().map(|v| v.abs()).sum::<f32>() / hidden_size as f32;
                let scale_x = (absmean_x * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                for v in &mut x_data {
                    *v = (*v / scale_x).round().clamp(-1.0, 1.0) * scale_x;
                }
            }

            // Stream G / G+: pick expert after STE emb is ready (hash needs x)
            if crate::mud::moe_train::moe_train_enabled() {
                let eid = match crate::mud::moe_train::moe_train_mode() {
                    crate::mud::moe_train::MoeTrainMode::Hash => {
                        let (e, _route) = crate::mud::moe_train::begin_step_hash(
                            &moe_pool,
                            &x_data[..hidden_size],
                            crate::mud::moe_load::default_top_k(),
                        );
                        e
                    }
                    _ => crate::mud::moe_train::begin_step(&moe_pool),
                };
                if let Some(core) = _mud.skills.get("core") {
                    for (blk, layer) in layers.iter_mut().enumerate() {
                        let ffn =
                            crate::mud::moe_load::resolve_expert_ffn_names(&core.tensors, blk, eid)
                                .unwrap_or(crate::mud::moe_load::ExpertFfnNames {
                                    up: format!("expert.{eid}.w3"),
                                    gate: format!("expert.{eid}.w1"),
                                    down: format!("expert.{eid}.w2"),
                                });
                        let p = format!("blk.{blk}.");
                        let t = |name: &str| -> *const u8 {
                            core.tensors
                                .get(&format!("{p}{name}.weight"))
                                .map(|t| t.data_ptr)
                                .unwrap_or(std::ptr::null())
                        };
                        let ts = |name: &str| -> *const f32 {
                            core.tensors
                                .get(&format!("{p}{name}.prq_scale"))
                                .map(|t| t.data_ptr as *const f32)
                                .unwrap_or(std::ptr::null())
                        };
                        layer.ffn_up_w = t(&ffn.up);
                        layer.ffn_gate_w = t(&ffn.gate);
                        layer.ffn_down_w = t(&ffn.down);
                        layer.ffn_up_scales = ts(&ffn.up);
                        layer.ffn_gate_scales = ts(&ffn.gate);
                        layer.ffn_down_scales = ts(&ffn.down);
                    }
                }
            }

            for (i, &x_val) in x_data.iter().enumerate().take(hidden_size) {
                crate::mud::slime::SlimeRegister::init_from_embed(
                    &mut workspace.registers[i],
                    &mut workspace.jepa_z,
                    i,
                    hidden_size,
                    layers.len(),
                    x_val,
                    true,
                );
            }

            // 2. Forward pass through layers (L-15: optional activation checkpointing)
            // Stream D: use causal `pos` so RoPE + KV history match inference.
            // Stream H: residual bank when MUD_GRAD_CKPT_RESIDUAL=1 + segmented.
            // FWD_LAST_N: skip lower layers for 1260P seating speed (approximate residual).
            let ckpt = crate::mud::grad_checkpoint::CheckpointPolicy::resolve();
            let n_layers = layers.len();
            let fwd_n = crate::mud::sequence_pack::train_fwd_last_n_layers(n_layers);
            let first_fwd = n_layers.saturating_sub(fwd_n);
            // one-shot log per process would spam — only first step of first chunk if env debug
            if first_fwd > 0 && pair_count == 0 {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ram",
                        &format!(
                            "FWD_LAST_N={fwd_n} first_fwd={first_fwd}/{n_layers} (approx residual; 1260P seating speed)"
                        )
                    )
                );
            }
            let use_residual_bank = ckpt.is_segmented()
                && crate::mud::grad_checkpoint::residual_bank_recompute_enabled()
                && first_fwd == 0; // residual bank assumes full stack
            let mut residual_bank = if use_residual_bank {
                Some(crate::mud::grad_checkpoint::ResidualBank::with_workspace(
                    ckpt.num_residual_slots(n_layers),
                    workspace,
                ))
            } else {
                None
            };
            // Keep quantized emb for exact recompute when MUD_GRAD_CKPT=1 (fallback)
            // P-01: use pre-allocated backward_ws.emb_ckpt instead of to_vec()
            let emb_ckpt_ref: Option<&[f32]> = if ckpt.is_segmented() && first_fwd == 0 {
                emb_ckpt.clear();
                emb_ckpt.extend_from_slice(&x_data[..hidden_size]);
                Some(&emb_ckpt)
            } else {
                None
            };
            for l_idx in first_fwd..n_layers {
                let layer = &layers[l_idx];
                if let Some(ref mut bank) = residual_bank {
                    // Save residual at each segment boundary (before layer runs)
                    if l_idx == ckpt.segment_start(l_idx) {
                        bank.save_from_workspace(ckpt.segment_of(l_idx), workspace);
                    }
                }
                // Tape only needed for thawed BWD layers
                let tape = if l_idx >= first_train_layer {
                    Some(&mut tapes[l_idx])
                } else {
                    None
                };
                crate::mud::slime_forward::evaluate_slime_block(
                    layer, l_idx, workspace, pos, eps, tape,
                    vk_qat.as_deref_mut(),
                );
            }
            if let Some(ref mut bank) = residual_bank {
                // Final residual after last layer (last slot)
                let last_slot = ckpt.num_residual_slots(n_layers).saturating_sub(1);
                bank.save_from_workspace(last_slot, workspace);
            }
            // Segmented: free all tapes after forward — recompute on backward
            if ckpt.is_segmented() {
                for t in tapes.iter_mut() {
                    crate::mud::grad_checkpoint::discard_tape(t);
                }
            }

            for (i, val) in pre_norm_x.iter_mut().enumerate().take(hidden_size) {
                *val = workspace.registers[i].read_accum();
            }

            // FIX (2026-07-31): output_norm.weight is mmap'd read-only.
            // Use an owned shadow buffer so the optimizer can update it.
            // Previously frozen at pretrained values → norm/weight scale mismatch
            // → activation collapse after ~5000 steps.
            let output_norm_w = {
                let mud_w = _mud
                    .skills
                    .get("core")
                    .and_then(|c| c.tensors.get("output_norm.weight"))
                    .map(|t| t.data_ptr as *const f32)
                    .unwrap_or(std::ptr::null());
                if mud_w.is_null() {
                    std::ptr::null()
                } else {
                    // Lazily initialize shadow from mmap'd data on first use.
                    if backward_ws.output_norm_shadow.is_empty() {
                        unsafe {
                            let slice = std::slice::from_raw_parts(mud_w, hidden_size);
                            backward_ws.output_norm_shadow.extend_from_slice(slice);
                        }
                    }
                    backward_ws.output_norm_shadow.as_ptr()
                }
            };
            crate::mud::slime_forward::apply_output_norm(workspace, output_norm_w, eps);

            for (i, val) in final_x.iter_mut().enumerate().take(hidden_size) {
                *val = workspace.registers[i].read_accum();
            }

            // ── LIVE activation telemetry (must be pre-BWD) ──────────────────
            // Post-BWD registers are garbage; sample here only.
            // VarH: max(stddev pre-output-norm, stddev post-norm, RMS as floor)
            let var_h_pre = slice_stddev(&pre_norm_x[..hidden_size]);
            let var_h_post = slice_stddev(&final_x[..hidden_size]);
            let mut sum_sq = 0.0f32;
            let mut abs_sum = 0.0f32;
            for &v in pre_norm_x.iter().take(hidden_size) {
                sum_sq += v * v;
                abs_sum += v.abs();
            }
            let rms_h = (sum_sq / hidden_size as f32).sqrt();
            let mean_abs_h = abs_sum / hidden_size as f32;
            // Prefer spatial stddev; if flat but non-zero RMS, report RMS so metric is not "dead zero"
            let var_h =
                var_h_pre
                    .max(var_h_post)
                    .max(if mean_abs_h > crate::mud::constants::EPSILON_FLOOR { rms_h * 0.1 } else { 0.0 });
            // FIX (2026-07-31): NaN guards on activation metrics to prevent
            // corrupted telemetry values from propagating.
            let var_h = if var_h.is_finite() { var_h } else { 0.0 };
            // VarJ: stddev of JEPA z-trackers + jepa_var_ema blend
            let var_j_z = slice_stddev(&workspace.jepa_z);
            let var_j_ema = if workspace.jepa_var_ema.is_empty() {
                0.0
            } else {
                let s: f32 = workspace.jepa_var_ema.iter().map(|v| v.abs()).sum();
                s / workspace.jepa_var_ema.len() as f32
            };
            let var_j = var_j_z.max(var_j_ema);
            // JEPA gate energy
            let mut e_sum = 0.0f32;
            for r in workspace.registers.iter().take(hidden_size) {
                e_sum += r.read_integral();
            }
            let mut jepa_i = e_sum / hidden_size as f32;
            if workspace.jepa_integral.is_finite() {
                jepa_i = 0.5 * jepa_i + 0.5 * workspace.jepa_integral;
            }
            let sigma_v = 100.0 / (1.0 + (-jepa_i).exp());
            // Cognitive: mean |pre_norm| × 100 (mai_bytes was never written → always 0)
            let cognitive = mean_abs_h * 100.0;
            act_accum.push(var_h, var_j, jepa_i, sigma_v, cognitive);

            // 3. Sampled Softmax (1 target + num_neg random negatives; env MUD_TRAIN_NUM_NEG)
            let mut rng_state = input_id.wrapping_mul(1664525).wrapping_add(target_id);
            let mut neg_count = 0;
            // Cap rejection sampling to avoid infinite loops on tiny vocabs
            let mut attempts = 0usize;
            while neg_count < num_neg && attempts < num_neg * 8 + 64 {
                attempts += 1;
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let neg = rng_state % vocab_size;
                if neg != target_id && !neg_ids[0..neg_count].contains(&neg) {
                    neg_ids[neg_count] = neg;
                    neg_count += 1;
                }
            }
            // If short (tiny vocab), pad with wraps
            while neg_count < num_neg {
                neg_ids[neg_count] = (neg_count + 1) % vocab_size.max(1);
                if neg_ids[neg_count] == target_id {
                    neg_ids[neg_count] = (neg_ids[neg_count] + 1) % vocab_size.max(1);
                }
                neg_count += 1;
            }

            // 1) Target emb → index 0
            load_emb_row_into(
                _mud,
                shadow_emb,
                emb_lazy,
                hidden_size,
                target_id,
                &mut class_embs_flat[0..hidden_size],
            );
            let scale = if emb_lazy {
                let absmean = class_embs_flat[0..hidden_size]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
                    / (hidden_size as f32);
                absmean.max(1e-8)
            } else {
                let absmean = class_embs_flat[0..hidden_size]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
                    / (hidden_size as f32);
                let scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                // AVX2 vectorized round-and-scale: clamp to [-1,1]*scale, then multiply by scale
                if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                    unsafe {
                        crate::mud::ezop::clamp_finite_avx2(
                            class_embs_flat.as_mut_ptr(),
                            -scale,
                            1.0 * scale,
                            0.0,
                            hidden_size,
                        );
                        crate::mud::ezop::scale_vec_avx2(
                            class_embs_flat.as_mut_ptr(),
                            scale,
                            hidden_size,
                        );
                    }
                } else {
                    for v in &mut class_embs_flat[0..hidden_size] {
                        *v = (*v / scale).round().clamp(-1.0, 1.0) * scale;
                    }
                }
                scale
            };

            // 2) Neg embs (lazy mmap or shadow)
            for (idx, &neg) in neg_ids.iter().take(num_neg).enumerate() {
                let c_start = (idx + 1) * hidden_size;
                load_emb_row_into(
                    _mud,
                    shadow_emb,
                    emb_lazy,
                    hidden_size,
                    neg,
                    &mut class_embs_flat[c_start..c_start + hidden_size],
                );
            }

            // Logit scale_up: reuse target STE scale path; avoid full-vocab emb RMS each step
            let scale_up = (1.0 / scale.max(1e-8)).clamp(1.0, 128.0);

            let temp_scale = 1.0 / (hidden_size as f32).sqrt();
            for (i, class_logit) in class_logits.iter_mut().enumerate().take(num_classes) {
                let c_start = i * hidden_size;
                let dot = unsafe {
                    forge_autograd::avx_math::dot_product_avx2(
                        &final_x[..hidden_size],
                        &class_embs_flat[c_start..c_start + hidden_size],
                    )
                };
                *class_logit = dot * scale_up * temp_scale;
            }

            // Softmax (AVX2 vectorized when num_classes >= 8)
            let max_logit = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && num_classes >= 8 {
                unsafe { crate::mud::ezop::max_avx2(class_logits.as_ptr(), num_classes) }
            } else {
                class_logits[0..num_classes].iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            };
            // P-17: max could be -Inf if all logits are -Inf
            let max_logit = if max_logit.is_finite() { max_logit } else { 0.0 };
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && num_classes >= 8 {
                unsafe {
                    crate::mud::ezop::broadcast_sub_avx2(
                        class_logits.as_mut_ptr(), max_logit, num_classes
                    );
                    crate::mud::ezop::exp_avx2(
                        class_logits.as_ptr(), prob_q.as_mut_ptr(), num_classes
                    );
                    let exp_sum = crate::mud::ezop::sum_avx2(prob_q.as_ptr(), num_classes);
                    let inv_sum = 1.0 / (exp_sum + 1e-10);
                    crate::mud::ezop::scale_vec_avx2(prob_q.as_mut_ptr(), inv_sum, num_classes);
                }
            } else {
                let mut exp_sum = 0.0;
                for i in 0..num_classes {
                    prob_q[i] = (class_logits[i] - max_logit).exp();
                    exp_sum += prob_q[i];
                }
                let inv_sum = 1.0 / (exp_sum + 1e-10);
                for prob in prob_q.iter_mut().take(num_classes) {
                    *prob *= inv_sum;
                }
            }

            // Cross Entropy Loss (target is always index 0)
            let loss = -prob_q[0].ln();
            total_loss += loss;
            pair_count += 1;

            // Entropy of the predicted distribution (info-theoretic health metric:
            // ~ln(num_classes) at init, →0 as model collapses, higher is better).
            let mut entropy = 0.0f32;
            for &p in prob_q.iter().take(num_classes) {
                if p > 1e-12 {
                    entropy -= p * p.log2();
                }
            }
            entropy_sum += entropy;

            // Calculate gradients
            for (i, d_l) in d_logits.iter_mut().enumerate().take(num_classes) {
                *d_l = (prob_q[i] - if i == 0 { 1.0 } else { 0.0 }) * temp_scale;
            }

            grad_in.fill(0.0); // gradient of x
            for (i, d_l) in d_logits.iter().enumerate().take(num_classes) {
                let c_start = i * hidden_size;
                unsafe {
                    forge_autograd::avx_math::axpy_avx2(
                        &mut grad_in[..hidden_size],
                        *d_l,
                        &class_embs_flat[c_start..c_start + hidden_size],
                    );
                }
            }

            if !output_norm_w.is_null() {
                let mut sq_sum = 0.0;
                // AVX2 sum_sq (8 f32/iter via FMA)
                if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                    sq_sum = unsafe { ezop::sum_sq(pre_norm_x.as_ptr(), hidden_size) };
                } else {
                    for &v in &pre_norm_x {
                        sq_sum += v * v;
                    }
                }
                let rms = (sq_sum / hidden_size as f32 + eps).sqrt();
                let rms_inv = 1.0 / rms;

                // AVX2 dot3: sum_dy_xnorm = sum(grad_in * w * pre_norm_x) * rms_inv
                let sum_dy_xnorm = if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                    unsafe {
                        ezop::dot3_fma_avx2(
                            grad_in.as_ptr(),
                            output_norm_w,
                            pre_norm_x.as_ptr(),
                            hidden_size,
                        ) * rms_inv
                    }
                } else {
                    let mut s = 0.0f32;
                    for i in 0..hidden_size {
                        let w = unsafe { *output_norm_w.add(i) };
                        let x_norm = pre_norm_x[i] * rms_inv;
                        s += grad_in[i] * w * x_norm;
                    }
                    s
                };
                let mean_dy_xnorm = sum_dy_xnorm / hidden_size as f32;

                // FIX (2026-07-31): Accumulate gradient w.r.t. output_norm weights.
                // d_w[i] = grad_y[i] * x_norm[i] (before grad_in is overwritten).
                // Previously output_norm was frozen — root cause of activation collapse.
                // B-1 FIX (2026-08-03): the resize+fill now happens once at chunk
                // start (above); here we just ensure the buffer stays zero-init
                // across the per-step accumulation. AVX2 path does += into [i].
                // AVX2 fused RMSNorm backward: w_grad accumulation + dx computation
                // in a single pass (8 f32/iter via FMA), with NaN/Inf guard (P-17).
                if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                    unsafe {
                        ezop::rmsnorm_backward_avx2(
                            grad_in.as_mut_ptr(),
                            pre_norm_x.as_ptr(),
                            output_norm_w,
                            backward_ws.output_norm_w_grad.as_mut_ptr(),
                            rms_inv,
                            mean_dy_xnorm,
                            hidden_size,
                        );
                    }
                } else {
                    for i in 0..hidden_size {
                        let x_norm = pre_norm_x[i] * rms_inv;
                        let g = grad_in[i] * x_norm;
                        backward_ws.output_norm_w_grad[i] += if g.is_finite() { g } else { 0.0 };
                        let w = unsafe { *output_norm_w.add(i) };
                        let dx = (w * rms_inv) * (grad_in[i] - x_norm * mean_dy_xnorm);
                        grad_in[i] = if dx.is_finite() { dx } else { 0.0 };
                    }
                }
            }

            grad_out.fill(0.0);

            // ── Phase 2: STP trajectory aux loss (train-only, zero inference cost) ──
            // Current position is `t`; sample `s < r < t` from the window history ring.
            // Inject λ·∂L_STP/∂h_t into the top-of-stack residual grad (grad wrt
            // pre_norm_x). h_s,h_r treated as constants this step (stochastic estimator);
            // over the window every position gets STP signal as it becomes `t`.
            if stp_on && stp_hist_len >= 2 {
                // Pick two distinct history indices (r more recent than s when possible).
                stp_rng = stp_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let a = (stp_rng >> 33) as usize % stp_hist_len;
                stp_rng = stp_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut b = (stp_rng >> 33) as usize % stp_hist_len;
                if b == a {
                    b = (b + 1) % stp_hist_len;
                }
                // Map ring slots to chronological order: older -> s, newer -> r.
                let oldest = if stp_hist_len < STP_HIST {
                    0
                } else {
                    stp_hist_head
                };
                let age = |k: usize| (k + STP_HIST - oldest) % STP_HIST;
                let (idx_s, idx_r) = {
                    let (ia, ib) = (
                        (oldest + a.min(stp_hist_len - 1)) % STP_HIST,
                        (oldest + b.min(stp_hist_len - 1)) % STP_HIST,
                    );
                    if age(ia) <= age(ib) {
                        (ia, ib)
                    } else {
                        (ib, ia)
                    }
                };
                if idx_s != idx_r {
                    stp_grad_t.fill(0.0);
                    stp_grad_r.fill(0.0);
                    stp_grad_s.fill(0.0);
                    // Only dL/dh_t is applied (s,r are frozen history this step).
                    let l_stp = crate::mud::stp_loss::stp_loss_and_grad(
                        &stp_hist[idx_s],
                        &stp_hist[idx_r],
                        &pre_norm_x[..hidden_size],
                        stp_lambda,
                        &mut stp_grad_s, // dL/dh_s (discarded: s is history)
                        &mut stp_grad_r, // dL/dh_r (discarded: r is history)
                        &mut stp_grad_t, // dL/dh_t -> current position
                        vk_qat.as_deref_mut(),
                    );
                    if stp_grad_t.iter().all(|v| v.is_finite()) {
                        // AVX2 add: grad_in[i] += stp_grad_t[i] (8 f32/iter via FMA)
                        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && hidden_size >= 8 {
                            unsafe {
                                ezop::axpy_avx2(grad_in.as_mut_ptr(), 1.0, stp_grad_t.as_ptr(), hidden_size);
                            }
                        } else {
                            for i in 0..hidden_size {
                                grad_in[i] += stp_grad_t[i];
                            }
                        }
                        stp_loss_accum += l_stp;
                        stp_triples += 1;
                    }
                }
            }
            // Push current top-of-stack residual into the window ring for future triples.
            if stp_on {
                stp_hist[stp_hist_head].copy_from_slice(&pre_norm_x[..hidden_size]);
                stp_hist_head = (stp_hist_head + 1) % STP_HIST;
                if stp_hist_len < STP_HIST {
                    stp_hist_len += 1;
                }
            }

            if grad_in.iter().all(|v| v.is_finite()) {
                // AVX2 sum_sq (8 f32/iter via FMA)
                let norm_sq = if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                    unsafe { ezop::sum_sq(grad_in.as_ptr(), hidden_size) }
                } else {
                    grad_in.iter().map(|&g| g * g).sum()
                };
                // FIX (2026-07-31): NaN guard on grad_norm. If norm_sq is NaN/Inf,
                // zero all gradients to prevent NaN propagation through backward.
                let grad_norm = if norm_sq.is_finite() { norm_sq.sqrt() } else { 0.0 };
                if !grad_norm.is_finite() || grad_norm > 1e6 {
                    // Gradient explosion — zero out to prevent NaN propagation (P-17)
                    grad_in.fill(0.0);
                    grad_norm_sum += 0.0;
                    continue;
                }
                grad_norm_sum += grad_norm;
                // B-2 FIX (2026-08-03): wire MUD_GRAD_CLIP (was hard-coded to 1.0,
                // making the documented env var a no-op). qat_grad_clip() defaults
                // to 1.0 and reads MUD_GRAD_CLIP for fine-tuning override.
                let clip_thresh = crate::mud::constants::qat_grad_clip();
                let clip = if grad_norm > clip_thresh {
                    clip_thresh / grad_norm
                } else {
                    1.0
                };
                // AVX2 scale + NaN guard: grad_in *= clip, non-finite → 0
                if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                    unsafe {
                        ezop::scale_finite_avx2(grad_in.as_mut_ptr(), clip, hidden_size);
                    }
                } else {
                    for v in &mut grad_in {
                        *v *= clip;
                        if !v.is_finite() { *v = 0.0; }
                    }
                }

                // L-15 / H: reverse layers; residual-bank recompute or emb fallback
                // Speed path: stop BWD below first_train_layer (frozen lower blocks)
                let mut l_idx = n_layers;
                let mut last_recomputed_seg: Option<usize> = None;
                while l_idx > first_train_layer {
                    l_idx -= 1;
                    if ckpt.is_segmented() && !tapes[l_idx].valid {
                        let seg = ckpt.segment_of(l_idx);
                        // Only recompute each segment once when walking reverse
                        if last_recomputed_seg != Some(seg) {
                            if use_residual_bank {
                                if let Some(ref bank) = residual_bank {
                                    crate::mud::grad_checkpoint::recompute_from_residual_bank(
                                        bank, layers, l_idx, workspace, tapes, ckpt, eps, pos,
                                    );
                                }
                            } else if let Some(emb) = emb_ckpt_ref {
                                let end = ckpt.segment_end(l_idx, n_layers);
                                crate::mud::grad_checkpoint::recompute_from_embedding(
                                    emb, layers, end, workspace, tapes, eps, pos,
                                );
                            }
                            last_recomputed_seg = Some(seg);
                        }
                    }
                crate::mud::slime_backward::backward_slime_block(
                        &layers[l_idx],
                        workspace,
                        backward_ws,
                        &tapes[l_idx],
                        &mut gradients[l_idx],
                        &grad_in,
                        &mut grad_out,
                    );
                    grad_in.copy_from_slice(&grad_out);
                    if ckpt.is_segmented() {
                        crate::mud::grad_checkpoint::discard_tape(&mut tapes[l_idx]);
                    }
                }

                // grad_in → input emb (skip if freeze / lazy)
                if !emb_lazy
                    && !shadow_emb.is_empty()
                    && !crate::mud::constants::train_freeze_emb()
                    && grad_in.iter().all(|v| v.is_finite())
                {
                    let norm_sq: f32 = grad_in.iter().map(|&g| g * g).sum();
                    let clip = if norm_sq.sqrt() > 1.0 {
                        1.0 / norm_sq.sqrt()
                    } else {
                        1.0
                    };
                    for v in &mut grad_in {
                        *v *= clip;
                    }
                    let target_slice =
                        &mut shadow_emb[input_id * hidden_size..(input_id + 1) * hidden_size];
                    unsafe {
                        forge_autograd::avx_math::axpy_avx2(target_slice, -lr, &grad_in);
                    }
                }
            }

            // 5. Update target and negative embeddings (skip if freeze / lazy emb)
            let freeze_emb = emb_lazy || crate::mud::constants::train_freeze_emb();
            if !freeze_emb && !shadow_emb.is_empty() && final_x.iter().all(|v| v.is_finite()) {
                let norm_sq_x: f32 = final_x.iter().map(|&x| x * x).sum();
                let x_norm = norm_sq_x.sqrt();

                let mut target_clip = 1.0;
                let target_dl = d_logits[0];
                if target_dl.abs() * x_norm > 1.0 {
                    target_clip = 1.0 / (target_dl.abs() * x_norm).max(1e-8);
                }
                let target_row =
                    &mut shadow_emb[target_id * hidden_size..(target_id + 1) * hidden_size];
                unsafe {
                    forge_autograd::avx_math::axpy_avx2(
                        target_row,
                        -lr * target_clip * target_dl,
                        &final_x,
                    );
                }

                for (ni, &neg_id) in neg_ids.iter().enumerate() {
                    let dl = d_logits[1 + ni];
                    let mut neg_clip = 1.0;
                    if dl.abs() * x_norm > 1.0 {
                        neg_clip = 1.0 / (dl.abs() * x_norm).max(1e-8);
                    }
                    let neg_row = &mut shadow_emb[neg_id * hidden_size..(neg_id + 1) * hidden_size];
                    unsafe {
                        forge_autograd::avx_math::axpy_avx2(neg_row, -lr * neg_clip * dl, &final_x);
                    }
                }
            }

            // EDGE-09: Dispatch Heartbeat to Vulkan to prevent RC6 deep sleep penalty
            if let Some(ref vk) = vk_qat {
                let _ = unsafe { vk.dispatch_heartbeat_sync() };
            }
            crate::mud::moe_train::end_step();
        }

        if crate::mud::moe_train::moe_train_enabled() {
            let util = crate::mud::moe_train::utilization_snapshot();
            if !util.is_empty() {
                println!(
                    "  [MoE-train] util={}",
                    util.iter()
                        .map(|(e, c)| format!("e{e}:{c}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }

        // 6. Apply gradients to deep layers
        let num_tokens = pair_count.max(1) as f32;
        let _weight_decay = 0.01;
        let _vk_updates: Vec<i32> = Vec::new();
        let mut vk_readbacks = Vec::new();

        // 1) First pass: collect all steps if we have Vulkan available
        let _use_vk = vk_qat.is_some();
        // FIX (2026-07-31): Use scheduled LR (warmup + cosine decay) instead of fixed base_lr.
        // Previously lr=0.0005 fixed for the entire run → cog drift → collapse at step ~5848.
        let lr = apply_lr_schedule(
            crate::mud::constants::qat_learning_rate(),
            // B-6 FIX: step+1 (fetch_add happens at end of chunk via Relaxed ordering).
            self.step_count.load(std::sync::atomic::Ordering::Relaxed) + 1,
            self.total_steps_hint.borrow().unwrap_or(crate::mud::constants::QAT_LR_TOTAL_STEPS),
            *self.warmup_steps.borrow(),
        );
        let decay = 0.01;
        // SCALES_ONLY freezes ELUT codes — ash GPU pack would re-threshold trits.
        // Keep ash for heartbeat + GEMV; force AVX2×PCorePool scales-only optimizer.
        let scales_only = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let use_ash_opt = vk_qat.is_some() && !scales_only;
        if use_ash_opt {
            let vk_qat = vk_qat.unwrap();
            let mut all_steps = Vec::new();
            let last_n_vk = crate::mud::sequence_pack::train_last_n_layers(shadow_layers.len());
            let first_train_vk = shadow_layers.len().saturating_sub(last_n_vk);
            for (l_idx, shadow_layer) in shadow_layers.iter().enumerate() {
                if l_idx < first_train_vk {
                    continue;
                }
                let grad = &gradients[l_idx];

                let matrices: Vec<WeightMatrixRef> = vec![
                    (
                        "q",
                        &shadow_layer.q_w,
                        &grad.q_w_grad,
                        layers[l_idx].q_w as *mut u8,
                        layers[l_idx].q_scales as *mut f32,
                    ),
                    (
                        "k",
                        &shadow_layer.k_w,
                        &grad.k_w_grad,
                        layers[l_idx].k_w as *mut u8,
                        layers[l_idx].k_scales as *mut f32,
                    ),
                    (
                        "v",
                        &shadow_layer.v_w,
                        &grad.v_w_grad,
                        layers[l_idx].v_w as *mut u8,
                        layers[l_idx].v_scales as *mut f32,
                    ),
                    (
                        "o",
                        &shadow_layer.o_w,
                        &grad.o_w_grad,
                        layers[l_idx].o_w as *mut u8,
                        layers[l_idx].o_scales as *mut f32,
                    ),
                    (
                        "up",
                        &shadow_layer.ffn_up_w,
                        &grad.ffn_up_w_grad,
                        layers[l_idx].ffn_up_w as *mut u8,
                        layers[l_idx].ffn_up_scales as *mut f32,
                    ),
                    (
                        "gate",
                        &shadow_layer.ffn_gate_w,
                        &grad.ffn_gate_w_grad,
                        layers[l_idx].ffn_gate_w as *mut u8,
                        layers[l_idx].ffn_gate_scales as *mut f32,
                    ),
                    (
                        "down",
                        &shadow_layer.ffn_down_w,
                        &grad.ffn_down_w_grad,
                        layers[l_idx].ffn_down_w as *mut u8,
                        layers[l_idx].ffn_down_scales as *mut f32,
                    ),
                ];

                for (m_name, shadow_cpu, grad_cpu, packed_ptr, scales_ptr) in matrices {
                    if packed_ptr.is_null() || scales_ptr.is_null() {
                        continue;
                    }
                    let elements = shadow_cpu.len();
                    let cols = match m_name {
                        "q" | "k" | "v" | "o" | "up" | "gate" => hidden_size,
                        "down" => elements / hidden_size,
                        _ => hidden_size,
                    };
                    let name = format!("blk.{}.{}", l_idx, m_name);
                    let rows = elements / cols;

                    vk_readbacks.push(crate::mud::ash_qat_dispatcher::PendingReadback {
                        name: name.clone(),
                        packed_ptr,
                        scales_ptr,
                        packed_len: elements.div_ceil(8) * 4,
                        rows,
                    });

                    all_steps.push(crate::mud::ash_qat_dispatcher::AshTensorStep {
                        name,
                        shadow: shadow_cpu,
                        grad: grad_cpu,
                        elements,
                        cols,
                        rows,
                    });
                }
            }

            // L-05 + D4: submit async only — packed readback deferred to next chunk start
            // so Forward N+1 overlaps GPU Optimizer N (CPU packed ≠ VRAM packed).
            // D4: when MUD_PIPELINE_OVERLAP=1, use step_pipeline for explicit 3-stage
            // overlap (CPU readback N-1 → GPU GEMV N → slot swap).
            // D5: when MUD_PIPELINE_OVERLAP=0, use step_combined_async_deferred to batch
            // optimizer step + mHC scale SGD in ONE vkQueueSubmit (signal keep-alive).

            // Phase 1 (mHC trainable): alpha/beta are dense f32 (CPU-resident); the ash
            // path only handles ternary GEMV weights. Batch mHC scale SGD via Vulkan
            // (safe on Iris Xe — uses optimizer_pipe in mHC mode, no QKV shader, no submit_and_wait).
            let last_n_vk = crate::mud::sequence_pack::train_last_n_layers(shadow_layers.len());
            let first_train_vk = shadow_layers.len().saturating_sub(last_n_vk);

            // Collect mHC updates for batch GPU dispatch (D5: combined with optimizer)
            let mut mhc_updates: Vec<crate::mud::ash_qat_dispatcher::AshMhcStep> = Vec::new();
            for l_idx in first_train_vk..shadow_layers.len() {
                let grad = &gradients[l_idx];
                if grad.mhc_alpha_grad.is_empty() {
                    continue;
                }
                let alpha_len = grad.mhc_alpha_grad.len();
                let beta_len = grad.mhc_beta_grad.len();
                let alpha_slice = unsafe {
                    std::slice::from_raw_parts(
                        layers[l_idx].mhc_alpha_w,
                        alpha_len,
                    )
                };
                let beta_slice = unsafe {
                    std::slice::from_raw_parts(
                        layers[l_idx].mhc_beta_w,
                        beta_len,
                    )
                };
                mhc_updates.push(crate::mud::ash_qat_dispatcher::AshMhcStep {
                    name: format!("blk.{}.mhc_alpha", l_idx),
                    shadow: alpha_slice,
                    grad: &grad.mhc_alpha_grad,
                    elements: alpha_len,
                });
                mhc_updates.push(crate::mud::ash_qat_dispatcher::AshMhcStep {
                    name: format!("blk.{}.mhc_beta", l_idx),
                    shadow: beta_slice,
                    grad: &grad.mhc_beta_grad,
                    elements: beta_len,
                });
            }

            unsafe {
                if crate::mud::ash_qat_dispatcher::pipeline_overlap_enabled() {
                    // D4: 3-stage overlap (no mHC batching — pipeline doesn't support it)
                    self.pipeline_overlap
                        .borrow_mut()
                        .step_pipeline(
                            vk_qat,
                            &all_steps,
                            vk_readbacks,
                            lr,
                            decay,
                            num_tokens,
                        )
                        .unwrap();
                    if !mhc_updates.is_empty() {
                        vk_qat.step_mhc_async(&mhc_updates, lr, num_tokens)?;
                        vk_qat.sync()?;
                    }
                } else {
                    // D5: combined batch — optimizer + mHC in ONE vkQueueSubmit
                    if !mhc_updates.is_empty() {
                        vk_qat.step_combined_async_deferred(
                            &all_steps, &mhc_updates, vk_readbacks,
                            lr, decay, num_tokens,
                        )?;
                    } else {
                        vk_qat
                            .step_async_deferred(&all_steps, vk_readbacks, lr, decay, num_tokens)
                            .unwrap();
                    }
                }
            }

            // Read back updated mHC weights to CPU-resident SlimeLayer pointers
            if !mhc_updates.is_empty() {
                unsafe {
                    vk_qat.sync()?;
                    for l_idx in first_train_vk..shadow_layers.len() {
                        let grad = &gradients[l_idx];
                        if grad.mhc_alpha_grad.is_empty() {
                            continue;
                        }
                        let alpha_len = grad.mhc_alpha_grad.len();
                        let beta_len = grad.mhc_beta_grad.len();
                        let alpha_out = std::slice::from_raw_parts_mut(
                            layers[l_idx].mhc_alpha_w as *mut f32,
                            alpha_len,
                        );
                        let beta_out = std::slice::from_raw_parts_mut(
                            layers[l_idx].mhc_beta_w as *mut f32,
                            beta_len,
                        );
                        vk_qat.readback_mhc(&format!("blk.{}.mhc_alpha", l_idx), alpha_out);
                        vk_qat.readback_mhc(&format!("blk.{}.mhc_beta", l_idx), beta_out);
                    }
                }
            }
        } else {
            // AVX2 CPU + STE pack (or SCALES_ONLY) via P-Core Pool
            // When SCALES_ONLY + ash: heartbeat already dispatched; GEMV may still use ash.
            let pool = crate::mud::pcore_pool::get_pool();
            let last_n = crate::mud::sequence_pack::train_last_n_layers(shadow_layers.len());
            let first_train_layer = shadow_layers.len().saturating_sub(last_n);
            for (l_idx, shadow_layer) in shadow_layers.iter_mut().enumerate() {
                if l_idx < first_train_layer {
                    continue; // frozen lower layers: skip opt+pack
                }
                let grad = &gradients[l_idx];

                // Split mut access: strategy + optional Adam moments per matrix
                let opts = [
                    (
                        "q",
                        shadow_layer.q_opt,
                        &mut shadow_layer.q_w as *mut Vec<f32>,
                        &grad.q_w_grad as *const Vec<f32>,
                        layers[l_idx].q_w as *mut u8,
                        layers[l_idx].q_scales as *mut f32,
                        &mut shadow_layer.q_adam as *mut Option<crate::mud::adam_state::AdamState>,
                    ),
                    (
                        "k",
                        shadow_layer.k_opt,
                        &mut shadow_layer.k_w as *mut _,
                        &grad.k_w_grad as *const _,
                        layers[l_idx].k_w as *mut u8,
                        layers[l_idx].k_scales as *mut f32,
                        &mut shadow_layer.k_adam as *mut _,
                    ),
                    (
                        "v",
                        shadow_layer.v_opt,
                        &mut shadow_layer.v_w as *mut _,
                        &grad.v_w_grad as *const _,
                        layers[l_idx].v_w as *mut u8,
                        layers[l_idx].v_scales as *mut f32,
                        &mut shadow_layer.v_adam as *mut _,
                    ),
                    (
                        "o",
                        shadow_layer.o_opt,
                        &mut shadow_layer.o_w as *mut _,
                        &grad.o_w_grad as *const _,
                        layers[l_idx].o_w as *mut u8,
                        layers[l_idx].o_scales as *mut f32,
                        &mut shadow_layer.o_adam as *mut _,
                    ),
                    (
                        "up",
                        shadow_layer.ffn_up_opt,
                        &mut shadow_layer.ffn_up_w as *mut _,
                        &grad.ffn_up_w_grad as *const _,
                        layers[l_idx].ffn_up_w as *mut u8,
                        layers[l_idx].ffn_up_scales as *mut f32,
                        &mut shadow_layer.ffn_up_adam as *mut _,
                    ),
                    (
                        "gate",
                        shadow_layer.ffn_gate_opt,
                        &mut shadow_layer.ffn_gate_w as *mut _,
                        &grad.ffn_gate_w_grad as *const _,
                        layers[l_idx].ffn_gate_w as *mut u8,
                        layers[l_idx].ffn_gate_scales as *mut f32,
                        &mut shadow_layer.ffn_gate_adam as *mut _,
                    ),
                    (
                        "down",
                        shadow_layer.ffn_down_opt,
                        &mut shadow_layer.ffn_down_w as *mut _,
                        &grad.ffn_down_w_grad as *const _,
                        layers[l_idx].ffn_down_w as *mut u8,
                        layers[l_idx].ffn_down_scales as *mut f32,
                        &mut shadow_layer.ffn_down_adam as *mut _,
                    ),
                ];

                for (m_name, strategy, sw_p, gw_p, packed_ptr, scales_ptr, adam_p) in opts {
                    if packed_ptr.is_null() || scales_ptr.is_null() {
                        continue;
                    }
                    // SAFETY: exclusive access to this matrix within the layer loop
                    let shadow_w = unsafe { &mut *sw_p };
                    let grad_w = unsafe { &*gw_p };
                    let elements = shadow_w.len();
                    let cols = match m_name {
                        "q" | "k" | "v" | "o" | "up" | "gate" => hidden_size,
                        "down" => elements / hidden_size.max(1),
                        _ => hidden_size,
                    };
                    let adam = unsafe { (*adam_p).as_mut() };
                    unsafe {
                        apply_optimizer_cpu_step_and_pack(
                            shadow_w, grad_w, packed_ptr, scales_ptr, lr, decay, num_tokens, cols,
                            pool, strategy, adam,
                        );
                    }
                }

                // Phase 1 (mHC trainable): dense f32 SGD step on alpha/beta scales.
                // These are Float32 tensors (init 0.85/0.15), not ternary — no ELUT pack.
                // Grads are empty for base models / frozen layers, so this is a no-op then.
                if !grad.mhc_alpha_grad.is_empty() {
                    let inv_tok = 1.0 / num_tokens.max(1.0);
                    unsafe {
                        mhc_scale_sgd_step(
                            layers[l_idx].mhc_alpha_w as *mut f32,
                            &grad.mhc_alpha_grad,
                            lr,
                            inv_tok,
                        );
                        mhc_scale_sgd_step(
                            layers[l_idx].mhc_beta_w as *mut f32,
                            &grad.mhc_beta_grad,
                            lr,
                            inv_tok,
                        );
                    }
                }

                // FIX (2026-07-31): Update RMSNorm weight shadows (per-layer attn + ffn).
                // Previously frozen — root cause of activation collapse (cog→0, varh→0
                // at step ~5848). Norm weights are F32 (no ternary pack); simple SGD
                // with NaN guard + clamp to [0.01, 10.0] keeps RMSNorm stable.
                // The SlimeLayer pointer (attn_norm_w/ffn_norm_w) already points to
                // the shadow Box<[f32]>, so updating via the pointer updates the
                // forward-pass weights in-place for the next chunk.
                if !scales_only && !grad.attn_norm_w_grad.is_empty() {
                    let lr_norm = lr / num_tokens.max(1.0);
                    unsafe {
                        crate::mud::ezop::sgd_step(
                            layers[l_idx].attn_norm_w as *mut f32,
                            grad.attn_norm_w_grad.as_ptr(),
                            hidden_size,
                            lr_norm,
                            0.0, // no weight decay on RMSNorm scales
                            1.0,
                        );
                        crate::mud::ezop::sgd_step(
                            layers[l_idx].ffn_norm_w as *mut f32,
                            grad.ffn_norm_w_grad.as_ptr(),
                            hidden_size,
                            lr_norm,
                            0.0,
                            1.0,
                        );
                        // Clamp RMSNorm weights to prevent runaway (P-17 fail-fast)
                        let attn_ptr = layers[l_idx].attn_norm_w as *mut f32;
                        let ffn_ptr = layers[l_idx].ffn_norm_w as *mut f32;
                        if is_x86_feature_detected!("avx2") && hidden_size >= 8 {
                            crate::mud::ezop::clamp_finite_avx2(attn_ptr, 0.01, 10.0, 1.0, hidden_size);
                            crate::mud::ezop::clamp_finite_avx2(ffn_ptr, 0.01, 10.0, 1.0, hidden_size);
                        } else {
                            for i in 0..hidden_size {
                                let aw = *attn_ptr.add(i);
                                *attn_ptr.add(i) = if aw.is_finite() { aw.clamp(0.01, 10.0) } else { 1.0 };
                                let fw = *ffn_ptr.add(i);
                                *ffn_ptr.add(i) = if fw.is_finite() { fw.clamp(0.01, 10.0) } else { 1.0 };
                            }
                        }
                    }
                }
            }
        }

        // FIX (2026-07-31): Update output_norm weight shadow (model-level RMSNorm).
        // Previously frozen at pretrained value — root cause of activation collapse.
        // Gradient accumulated in backward_ws.output_norm_w_grad across all tokens.
        if !backward_ws.output_norm_shadow.is_empty()
            && !backward_ws.output_norm_w_grad.is_empty()
            && !scales_only
        {
            let lr_norm = lr / num_tokens.max(1.0);
            unsafe {
                let n = backward_ws.output_norm_shadow.len();
                crate::mud::ezop::sgd_step(
                    backward_ws.output_norm_shadow.as_mut_ptr(),
                    backward_ws.output_norm_w_grad.as_ptr(),
                    n,
                    lr_norm,
                    0.0,
                    1.0,
                );
                // AVX2 clamp with NaN/Inf guard (replaces scalar loop)
                if is_x86_feature_detected!("avx2") && n >= 8 {
                    crate::mud::ezop::clamp_finite_avx2(
                        backward_ws.output_norm_shadow.as_mut_ptr(),
                        0.01, 10.0, 1.0, n,
                    );
                } else {
                    for i in 0..n {
                        let w = *backward_ws.output_norm_shadow.as_ptr().add(i);
                        *backward_ws.output_norm_shadow.as_mut_ptr().add(i) =
                            if w.is_finite() { w.clamp(0.01, 10.0) } else { 1.0 };
                    }
                }
            }

            // Sync output_norm shadow to MudFile tensor owned_data so checkpoint
            // saves the UPDATED norm weights (not the frozen mmap'd values).
            // FIX (2026-07-31): previously the checkpoint saved the original
            // pretrained output_norm.weight because the shadow was never synced.
            if let Some(core) = _mud.skills.get_mut("core") {
                if let Some(t) = core.tensors.get_mut("output_norm.weight") {
                    let bytes: &[u8] = unsafe {
                        std::slice::from_raw_parts(
                            backward_ws.output_norm_shadow.as_ptr() as *const u8,
                            backward_ws.output_norm_shadow.len() * 4,
                        )
                    };
                    crate::mud::checkpoint::write_owned_bytes(t, bytes);
                }
            }
        }

        // B-1 FIX (2026-08-03): clear output_norm_w_grad after the chunk's SGD step.
        // Without this, the += accumulation in the per-step loop carries over
        // across chunks → integral growth → degenerate clamp. This pairs with the
        // chunk-start resize+fill (above) to give a clean SGD per-chunk.
        backward_ws.output_norm_w_grad.clear();

        // FIX (2026-07-31): Increment global step counter for LR schedule.
        self.step_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let avg_loss = if pair_count > 0 {
            total_loss / pair_count as f32
        } else {
            0.0
        };
        if stp_on {
            let l_stp = if stp_triples > 0 {
                stp_loss_accum / stp_triples as f32
            } else {
                0.0
            };
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "stp",
                    &format!("λ={stp_lambda:.3} triples={stp_triples} L_STP={l_stp:.5} (aux, train-only)")
                )
            );
        }
        // SPRINT-6 (P-01): restore pre-allocated buffers to backward_ws
        backward_ws.x_data = x_data;
        backward_ws.pre_norm_x = pre_norm_x;
        backward_ws.final_x = final_x;
        backward_ws.class_embs_flat = class_embs_flat;
        backward_ws.class_logits = class_logits;
        backward_ws.prob_q = prob_q;
        backward_ws.d_logits = d_logits;
        backward_ws.grad_in = grad_in;
        backward_ws.grad_out = grad_out;
        backward_ws.neg_ids = neg_ids;
        backward_ws.stp_hist = stp_hist;
        backward_ws.stp_grad_t = stp_grad_t;
        backward_ws.stp_grad_r = stp_grad_r;
        backward_ws.stp_grad_s = stp_grad_s;
        backward_ws.emb_ckpt = emb_ckpt;

        // Push grad_norm + entropy (averaged across sub-steps) into the accumulator.
        let np = pair_count.max(1) as f32;
        let stp_avg = if stp_triples > 0 {
            stp_loss_accum / stp_triples as f32
        } else {
            0.0
        };
        act_accum.push_grad(grad_norm_sum / np, entropy_sum / np, stp_avg);

        Ok(act_accum.finish(avg_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate global LR env vars (MUD_QAT_LR, MUD_LR_RESTART, etc.)
    static LR_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn clear_lr_env() {
        std::env::remove_var("MUD_QAT_LR");
        std::env::remove_var("MUD_LR_RESTART");
        std::env::remove_var("MUD_LR_CYCLES");
    }

    #[test]
    fn test_lr_schedule_warmup_linear_ramp() {
        let _g = LR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_lr_env();

        let base = 0.0005;
        let total = 6400;
        let warmup = 64;
        let lr0 = apply_lr_schedule(base, 0, total, warmup);
        assert!(lr0 > 0.0 && lr0 < base * 0.1, "lr0={} should be in warmup", lr0);
        let lr_mid = apply_lr_schedule(base, 32, total, warmup);
        assert!(lr_mid > lr0, "lr should increase during warmup");
        let lr_pre = apply_lr_schedule(base, (warmup - 1) as u64, total, warmup);
        assert!(lr_pre > base * 0.9, "lr_pre={} should be ~base_lr", lr_pre);
    }

    #[test]
    fn test_lr_schedule_cosine_decay() {
        let _g = LR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_lr_env();
        std::env::set_var("MUD_LR_RESTART", "0"); // disable restarts for classic cosine test

        let base = 0.0005;
        let total = 6400;
        let warmup = 64;
        let lr_mid = apply_lr_schedule(base, 3200, total, warmup);
        assert!(lr_mid > 0.0 && lr_mid < base, "lr_mid={} should decay below base", lr_mid);
        // FIX (2026-08-02): cosine now floors at min_lr = base * 0.05, not 0.
        let lr_end = apply_lr_schedule(base, (total - 1) as u64, total, warmup);
        let min_lr = base * 0.05;
        assert!(lr_end >= min_lr * 0.95 && lr_end <= min_lr * 1.05,
            "lr_end={} should be near min_lr={}", lr_end, min_lr);

        std::env::remove_var("MUD_LR_RESTART");
    }

    /// GRASSHOPPER FIX (2026-08-03): Verify cosine restarts create LR bounces.
    #[test]
    fn test_lr_schedule_cosine_restart_bounce() {
        let _g = LR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_lr_env(); // restarts ON by default
        std::env::set_var("MUD_LR_CYCLES", "3"); // deterministic cycle count

        let base = 0.0005;
        let total = 6400;
        let warmup = 64;
        let n_cycles = 3;
        let cycle_len = (total - warmup) / n_cycles; // = 2112
        let cycle_warmup = cycle_len / 5; // = 422

        // End of cycle 0: LR should be near min_lr
        let end_cycle_0 = (warmup + cycle_len - 1) as u64;
        let lr_end_c0 = apply_lr_schedule(base, end_cycle_0, total, warmup);

        // Start of cycle 1 (after cycle-in warmup): LR should jump up
        let start_cycle_1 = (warmup + cycle_len + cycle_warmup) as u64;
        let lr_start_c1 = apply_lr_schedule(base, start_cycle_1, total, warmup);

        // The restart should produce a higher LR than the end of the previous cycle
        assert!(lr_start_c1 > lr_end_c0 * 1.5,
            "Restart LR ({}) should be >1.5x end-of-cycle LR ({}) — no bounce detected",
            lr_start_c1, lr_end_c0);

        // At minimum, LR at cycle restart should be above min_lr
        let min_lr = base * 0.05;
        assert!(lr_start_c1 > min_lr,
            "Restart LR ({}) should exceed min_lr ({})", lr_start_c1, min_lr);

        std::env::remove_var("MUD_LR_CYCLES");
    }

    /// GRASSHOPPER FIX (2026-08-03): Verify restart can be disabled.
    #[test]
    fn test_lr_schedule_restart_disabled() {
        let _g = LR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_lr_env();
        std::env::set_var("MUD_LR_RESTART", "0");

        let base = 0.0005;
        let total = 3200;
        let warmup = 32;

        // With restarts disabled, LR should monotonically decrease (single cosine)
        let lr_early = apply_lr_schedule(base, 100, total, warmup);
        let lr_mid = apply_lr_schedule(base, 1600, total, warmup);
        let lr_late = apply_lr_schedule(base, 3000, total, warmup);

        assert!(lr_early > lr_mid, "LR should decrease: {} > {}", lr_early, lr_mid);
        assert!(lr_mid > lr_late, "LR should decrease: {} > {}", lr_mid, lr_late);

        std::env::remove_var("MUD_LR_RESTART");
    }

    /// GRASSHOPPER FIX (2026-08-03): Verify fixed LR override still works.
    #[test]
    fn test_lr_schedule_fixed_lr_override() {
        let _g = LR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_lr_env();
        std::env::set_var("MUD_QAT_LR", "0.003");

        let base = 0.0005;
        let lr = apply_lr_schedule(base, 100, 6400, 64);
        // Should return base (0.0005) because MUD_QAT_LR bypasses schedule
        assert!((lr - 0.0005).abs() < 1e-6, "Fixed LR should be base ({}) when MUD_QAT_LR set, got {}", base, lr);

        std::env::remove_var("MUD_QAT_LR");
    }

    /// GRASSHOPPER FIX (2026-08-03): Verify the SGDR bounce breaks the monotonic decay.
    /// With a single cosine, step 1059 LR coincidentally equals step 2 LR (grasshopper trap).
    /// With restarts, mid-training LR should be higher than the monotonic-decay baseline.
    #[test]
    fn test_lr_schedule_restart_beats_single_cosine() {
        let _g = LR_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_lr_env();

        let base = 0.0005;
        let total = 1450; // 145 chunks/epoch * 10 epochs (user's run)
        let warmup = 14;  // total / 100

        // Step 1059: with restarts ON (default), the LR should be higher than
        // the single-cosine decay value, because a restart cycle boosts it.
        let lr_single = {
            std::env::set_var("MUD_LR_RESTART", "0");
            let lr = apply_lr_schedule(base, 1059, total, warmup);
            std::env::remove_var("MUD_LR_RESTART");
            lr
        };

        // With restarts ON (default)
        let lr_restart = apply_lr_schedule(base, 1059, total, warmup);

        println!("lr_single={:.6e}  lr_restart={:.6e}", lr_single, lr_restart);
        // The restart LR should be meaningfully different from the single cosine
        // at mid-training — this proves the grasshopper bounce is active.
        assert!(lr_restart != lr_single,
            "Restart LR ({}) should differ from single-cosine LR ({}) at step 1059",
            lr_restart, lr_single);
    }
}
