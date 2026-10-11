//! RLVR debate training session: DocumentDebate / ProfessorStudent / Math / TicTacToe.
//!
//! `run_debate_session` is the high-level wrapper that:
//!   1. Runs AWAKE-01 alignment (via `alignment::deep_local_alignment`).
//!   2. Materializes the model so every `data_ptr` is backed by owned memory.
//!   3. Builds the `SlimeLayer` and `SlimeLayerShadowF32` arrays.
//!   4. Spawns the `DebateArena::run_game` with the chosen `ArenaGame` factory.
//!   5. If `MUD_DEBATE_LEARN=1` (default ON inside the circuit), persists the
//!      trained shadow weights back to the `.mud` and verifies the writeback
//!      via `hash_trained_weights` (no-op guard).
//!
//! The function is **the** entry point for all debate-based training phases
//! (`MUD_DEBATE_MODE ∈ {debate, games, professor}`).

use crate::mud::corpus_trainer::MudCorpusTrainer;
use crate::mud::{MudFile, MudTensorType};

impl MudCorpusTrainer {
    /// Run a single RLVR debate session and (optionally) write the trained
    /// shadow layers back to the model file.
    pub fn run_debate_session(
        &mut self,
        sender: Option<std::sync::mpsc::Sender<String>>,
        stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<()> {
        let print_stdout = sender.is_none();
        // Wrap the sender so EVERY arena/debate event is also persisted to
        // logs/circuit.log (timestamped) in addition to reaching any live TUI.
        let log_sender: Option<std::sync::mpsc::Sender<String>> = match &sender {
            Some(tx_out) => {
                let (tx_in, rx_in) = std::sync::mpsc::channel::<String>();
                let tx_out = tx_out.clone();
                std::thread::spawn(move || {
                    while let Ok(msg) = rx_in.recv() {
                        let _ = crate::mud::trainer_ui::circuit_event("arena", &msg);
                        if tx_out.send(msg).is_err() {
                            break;
                        }
                    }
                });
                Some(tx_in)
            }
            None => None,
        };
        if let Some(tx) = &log_sender {
            let _ = tx.send("⚔️ Starting MUD Debate Arena Session...".to_string());
        }
        if print_stdout {
            println!("⚔️ Starting MUD Debate Arena Session...");
        }
        let mut mud = MudFile::load(&self.model_path)?;
        self.deep_local_alignment(&mut mud, print_stdout)?;
        // Materialize ALL tensors into owned buffers so every `data_ptr` used
        // below (emb, norms, weights, scales) points at live backing memory.
        // Without this, a tensor left on a dropped mmap (or with `data_ptr`
        // reset by ecc_verify) yields a dangling/null pointer → reads as zeros
        // → "Dead RMSNorm" panic on the first forward (norm weights ~0).
        mud.materialize_writable();
        // Capture input weight hash (post deep_local_alignment) before the debate trains.
        let input_weights_hash = Self::hash_trained_weights(&mud);

        let hidden = mud
            .global_metadata
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing hidden_size");
        let n_layers = mud
            .global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_layers");
        let n_heads = mud
            .global_metadata
            .get("num_attention_heads")
            .or_else(|| mud.global_metadata.get("num_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_heads");
        let n_kv_heads = mud
            .global_metadata
            .get("num_key_value_heads")
            .or_else(|| mud.global_metadata.get("num_kv_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_kv_heads");
        let ffn_mid = mud
            .global_metadata
            .get("intermediate_size")
            .or_else(|| mud.global_metadata.get("ffn_hidden"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing ffn_mid");
        let max_pos = mud
            .global_metadata
            .get("max_position_embeddings")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing max_position_embeddings");
        let rope_theta: f32 = mud
            .global_metadata
            .get("rope_theta")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000.0);
        let core = mud
            .skills
            .get_mut("core")
            .ok_or_else(|| anyhow::anyhow!("No core skill"))?;
        let vocab_size = core
            .tensors
            .get("token_embd.weight")
            .map(|t| t.shape[0])
            .expect("Missing token_embd.weight");

        let mut emb = vec![0.0; vocab_size * hidden];
        let emb_tensor = core.tensors.get("token_embd.weight").unwrap();
        unsafe {
            let cols = hidden;
            if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                for r in 0..vocab_size {
                    crate::mud::dequantize_ternary_row(
                        (emb_tensor.data_ptr as *const u32).add(r * cols / 8),
                        &mut emb[r * cols..(r + 1) * cols],
                        cols,
                    );
                }
                if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                    if scale_tensor.t_type == MudTensorType::Float32 {
                        let scale_data = std::slice::from_raw_parts(
                            scale_tensor.data_ptr as *const f32,
                            vocab_size,
                        );
                        for r in 0..vocab_size {
                            let s = scale_data[r];
                            for c in 0..cols {
                                emb[r * cols + c] *= s;
                            }
                        }
                    }
                }
            } else {
                std::ptr::copy_nonoverlapping(
                    emb_tensor.data_ptr as *const f32,
                    emb.as_mut_ptr(),
                    vocab_size * hidden,
                );
            }
        }

        let computed_max_emb = emb.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
        let max_emb = mud
            .global_metadata
            .get("max_emb")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(computed_max_emb);

        let mut out_f32 = vec![0.0f32; vocab_size * hidden];
        let output_weight;
        if let Some(t) = core.tensors.get("output.weight") {
            unsafe {
                let cols = hidden;
                if t.t_type == MudTensorType::Ternary2Bit {
                    for r in 0..vocab_size {
                        crate::mud::dequantize_ternary_row(
                            (t.data_ptr as *const u32).add(r * cols / 8),
                            &mut out_f32[r * cols..(r + 1) * cols],
                            cols,
                        );
                    }
                    if let Some(scale_tensor) = core.tensors.get("output.prq_scale") {
                        if scale_tensor.t_type == MudTensorType::Float32 {
                            let scale_data = std::slice::from_raw_parts(
                                scale_tensor.data_ptr as *const f32,
                                vocab_size,
                            );
                            for r in 0..vocab_size {
                                let s = scale_data[r];
                                for c in 0..cols {
                                    out_f32[r * cols + c] *= s;
                                }
                            }
                        }
                    }
                    output_weight = out_f32.as_ptr();
                } else {
                    output_weight = t.data_ptr as *const f32;
                }
            }
        } else {
            // Tied weights fallback
            output_weight = emb.as_ptr();
        }

        let mut output_norm_w = std::ptr::null();
        if let Some(t) = core.tensors.get("output_norm.weight") {
            output_norm_w = t.data_ptr as *const f32;
        }

        let document = "La computación ternaria (1.58-bit) como MUD y BitNet, promete revolucionar la IA al eliminar las costosas multiplicaciones de punto flotante en la inferencia profunda. Sin embargo, su precisión en razonamiento matemático aún se considera un desafío abierto.";
        let debate_mode = std::env::var("MUD_DEBATE_MODE")
            .unwrap_or_else(|_| "debate".to_string())
            .to_lowercase();
        let game = crate::mud::arena_games::DocumentDebate::new(
            "El futuro de la Computación Ternaria en IA",
            document,
            10,
        );
        // Rotating exercise index for Professor-Student (each infinite cycle
        // poses a different local exercise — no API, fully deterministic).
        let ex_idx = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut layers: Vec<crate::mud::slime_forward::SlimeLayer> = Vec::new();
        for blk in 0..n_layers {
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
            // tc: for non-`.weight` tensors (mHC alpha/beta/radius).
            // FIX (2026-07-30): mHC tensors are `blk.N.mhc_alpha`, not `.weight`.
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
            // w3=up, w1=gate, w2=down (or up/gate alt); honors MUD_TRAIN_EXPERT
            let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);

            // PERF-1: precompute RoPE inv_freq for this layer.
            let head_d = hidden / n_heads;
            layers.push(crate::mud::slime_forward::SlimeLayer {
                q_w: t("attn_q"),
                k_w: t("attn_k"),
                v_w: t("attn_v"),
                o_w: t("attn_output"),
                q_scales: ts("attn_q"),
                k_scales: ts("attn_k"),
                v_scales: ts("attn_v"),
                o_scales: ts("attn_output"),
                ffn_up_w: t(&ffn.up),
                ffn_gate_w: t(&ffn.gate),
                ffn_down_w: t(&ffn.down),
                ffn_up_scales: ts(&ffn.up),
                ffn_gate_scales: ts(&ffn.gate),
                ffn_down_scales: ts(&ffn.down),
                attn_norm_w: tn("attn_norm"),
                ffn_norm_w,
                attn_sub_norm_w: tn("attn_sub_norm"),
                ffn_sub_norm_w: tn("ffn_sub_norm"),
                q_norm_w: tn("attn_q_norm"),
                k_norm_w: tn("attn_k_norm"),
                mhc_alpha_w: tc("mhc_alpha"),
                mhc_beta_w: tc("mhc_beta"),
                mhc_alpha_shadow: None,
                mhc_beta_shadow: None,
                attn_norm_shadow: None,
                ffn_norm_shadow: None,
                mhc_radius_w: tn("mhc_radius"),
                n_kv_heads,
                ffn_mid,
                rope_theta,
                rope_inv_freq: crate::mud::slime_forward::compute_rope_inv_freq(rope_theta, head_d),
                rope_cache: parking_lot::Mutex::new(crate::mud::slime_forward::RopeCache::new()),
                is_f32_weights: core
                    .tensors
                    .get(&format!("blk.{}.attn_q.weight", 0))
                    .map(|t| t.t_type == MudTensorType::Float32)
                    .unwrap_or(false),
            });
        }

        let mut arena = crate::mud::debate_trainer::DebateArena::new(
            crate::model::tokenizer::Tokenizer::from_mud_metadata(
                mud.global_metadata
                    .get("tokenizer.tokens")
                    .map(|s| s.as_str())
                    .unwrap_or(""),
                mud.global_metadata
                    .get("tokenizer.merges")
                    .map(|s| s.as_str())
                    .unwrap_or(""),
            ),
            max_pos,
            3600,
            hidden,
            n_layers,
            n_heads,
            n_kv_heads,
            ffn_mid,
            max_emb,
            vocab_size,
            output_weight,
            output_norm_w,
        );
        arena.sender = log_sender.clone();

        let mut shadow_layers = Vec::with_capacity(n_layers);
        for layer in layers.iter().take(n_layers) {
            let head_dim = hidden / n_heads;
            let mut shadow = crate::mud::slime_backward::SlimeLayerShadowF32::new(
                hidden, ffn_mid, n_kv_heads, head_dim,
            );

            // Dequantize from layers to shadow and apply scales
            unsafe {
                let dequant_with_scale =
                    |packed: *const u32, scales: *const f32, out: &mut [f32], cols: usize| {
                        if packed.is_null() || scales.is_null() {
                            return;
                        }
                        let rows = out.len() / cols;
                        for r in 0..rows {
                            let start = r * cols;
                            crate::mud::dequantize_ternary_row(
                                packed.add(r * cols / 8),
                                &mut out[start..start + cols],
                                cols,
                            );
                            let s = *scales.add(r);
                            for c in 0..cols {
                                out[start + c] *= s;
                            }
                        }
                    };

                dequant_with_scale(
                    layer.q_w as *const u32,
                    layer.q_scales,
                    &mut shadow.q_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.k_w as *const u32,
                    layer.k_scales,
                    &mut shadow.k_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.v_w as *const u32,
                    layer.v_scales,
                    &mut shadow.v_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.o_w as *const u32,
                    layer.o_scales,
                    &mut shadow.o_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.ffn_up_w as *const u32,
                    layer.ffn_up_scales,
                    &mut shadow.ffn_up_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.ffn_gate_w as *const u32,
                    layer.ffn_gate_scales,
                    &mut shadow.ffn_gate_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.ffn_down_w as *const u32,
                    layer.ffn_down_scales,
                    &mut shadow.ffn_down_w,
                    ffn_mid,
                );
            }
            shadow_layers.push(shadow);
        }

        let _qat_opt: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher> = None;
        let mut emb = vec![0.0; vocab_size * hidden];
        let emb_tensor = core.tensors.get("token_embd.weight").unwrap();
        unsafe {
            let cols = hidden;
            if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                for r in 0..vocab_size {
                    crate::mud::dequantize_ternary_row(
                        (emb_tensor.data_ptr as *const u32).add(r * cols / 8),
                        &mut emb[r * cols..(r + 1) * cols],
                        cols,
                    );
                }
                if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                    if scale_tensor.t_type == MudTensorType::Float32 {
                        let scale_data = std::slice::from_raw_parts(
                            scale_tensor.data_ptr as *const f32,
                            vocab_size,
                        );
                        for r in 0..vocab_size {
                            let s = scale_data[r];
                            for c in 0..cols {
                                emb[r * cols + c] *= s;
                            }
                        }
                    }
                }
            } else {
                std::ptr::copy_nonoverlapping(
                    emb_tensor.data_ptr as *const f32,
                    emb.as_mut_ptr(),
                    vocab_size * hidden,
                );
            }
        }

        // Use global pool (sized pre-pin). Local new() after pin can collapse to 1 thread.
        let pool = crate::mud::pcore_pool::get_pool();
        let topic = "El futuro de la Computación Ternaria en IA".to_string();
        let doc = document.to_string();
        let max_t = game.max_turns();
        let factory = move || {
            match debate_mode.as_str() {
                "professor" => {
                    let i = ex_idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Box::new(crate::mud::arena_games::ProfessorStudent::new(3, i))
                        as Box<dyn crate::mud::arena_games::ArenaGame>
                }
                "games" => {
                    // Verifiable seed-survival games (no API): Math / TicTacToe.
                    // Rotate so the model practices different checkable tasks.
                    let pick = ex_idx.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if pick.is_multiple_of(2) {
                        Box::new(crate::mud::arena_games::MathChallenge::random())
                            as Box<dyn crate::mud::arena_games::ArenaGame>
                    } else {
                        Box::new(crate::mud::arena_games::TicTacToe::new())
                            as Box<dyn crate::mud::arena_games::ArenaGame>
                    }
                }
                _ => Box::new(crate::mud::arena_games::DocumentDebate::new(
                    &topic, &doc, max_t,
                )) as Box<dyn crate::mud::arena_games::ArenaGame>,
            }
        };
        arena = arena.with_stop_flag(stop_flag);
        arena.run_game(
            factory,
            &mut layers,
            &mut shadow_layers,
            &emb,
            vocab_size,
            pool,
        )?;

        // Fase 3: writeback survivors (here: the single arena pair) to the .mud.
        let learn_on = std::env::var("MUD_DEBATE_LEARN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        // Quantize back and save (Fase 3: only persist if MUD_DEBATE_LEARN=1).
        // The manual writeback below already mutates `core` in-place (no second
        // mutable borrow of `mud`), so it is gated instead of sync_shadow_to_mud.
        let writeback = learn_on;
        if writeback {
            if print_stdout {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ok",
                        "saving trained shadow layers back to MUD..."
                    )
                );
            } else if let Some(tx) = &log_sender {
                let _ = tx.send(crate::mud::trainer_ui::note(
                    "ok",
                    "saving trained shadow layers back to MUD...",
                ));
            }
        }
        if writeback {
            // C1/C2 fix (2026-07-20): reuse sync_shadow_to_mud instead of the manual
            // pack that (a) skipped the PRQ scale (s = absmean·√½) → lost all weight
            // magnitude (any |w|<0.5 became 0, large w became ±1), and (b) never wrote
            // the *.prq_scale tensors → stale/inflated scales persisted → vocab collapse.
            // Emb is frozen (default) so an empty shadow slices the skip branch.
            let mut empty_emb: Vec<f32> = Vec::new();
            self.sync_shadow_to_mud(&mut mud, &mut empty_emb, &mut shadow_layers, &layers, None, true);
            mud.save(&self.model_path)?;
            // Authoritative no-op check vs the captured pre-debate input hash.
            let out_hash = Self::hash_trained_weights(&mud);
            if out_hash == input_weights_hash {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        "⚠ DEBATE WRITE-BACK IS A NO-OP — trained weights byte-identical to input (hash match). The .mud did not change; debate learning produced no weight delta."
                    )
                );
            } else {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ok",
                        &format!(
                            "✓ debate write-back persisted (weights hash {:#018x} → {:#018x})",
                            input_weights_hash, out_hash
                        )
                    )
                );
            }
        }

        Ok(())
    }
}
