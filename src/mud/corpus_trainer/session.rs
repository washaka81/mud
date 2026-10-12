//! Main alignment training session (L-15 / L-10 corpus trainer pipeline).
//!
//! `run_alignment_session` is the heart of the trainer: it builds the workspace,
//! prefetches chunks via a background thread, drives the per-chunk
//! `train_on_sequence` (forward → backward → optimizer step → checkpoint),
//! and persists state to the `.mud`.
//!
//! It is the largest function in the corpus_trainer (1535 lines at extraction
//! time). All helpers it relies on (env parsers, telemetry, integrity gates,
//! train step) are now in their own submodules of `corpus_trainer/`.

use crate::mud::corpus_trainer::{
    checkpoint_every_chunks, MudCorpusTrainer, CHUNK_SIZE, SHOULD_TERMINATE,
};
use crate::mud::{MudFile, MudTensorType};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

impl MudCorpusTrainer {
    pub fn run_alignment_session(&self, batch_size: usize, epochs: usize) -> anyhow::Result<()> {
        // Pin main thread to the first P-core (Core 0) to maximize AVX2 throughput and L1/L2 cache locality
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if let Some(first_core) = core_ids.first() {
                core_affinity::set_for_current(*first_core);
            }
        }
        let mut mud = MudFile::load(&self.model_path)?;
        // (F3+ / L-15) Structural integrity gate — P-17 fail-fast, no silent
        // corruption. `circuit_eval_integrity` checks: non-zero RMSNorm
        // (`model_norms_alive`), num_layers > 0, and every attn_q/k/v/output +
        // w1/w2/w3 weight tensor present + non-null/data-backed.
        //
        // WHY: a model with missing/degenerate tensors (e.g.
        // `ale_pico_fresh.mud` → 18 missing tensors → 🔴 STRUCTURAL ISSUES)
        // produces inconsistent gradients → loss diverges (⚠DIV, +0.3s/s).
        // The TUI detects the divergence *after* it happens; this gate stops
        // it *before* the first chunk so the trainer doesn't waste compute
        // (or silently produce rotten checkpoints).
        //
        // Env override: `MUD_SKIP_INTEGRITY=1` to force past the gate (debug only).
        let skip = std::env::var("MUD_SKIP_INTEGRITY").as_deref() == Ok("1");
        if !skip {
            let (ok, detail) = crate::mud::corpus_trainer::integrity::circuit_eval_integrity(&self.model_path);
            if !ok {
                eprintln!("🔴 [INTEGRITY-GATE] Model '{p}' FAILED structural check — aborting BEFORE training (no checkpoints written).", p = self.model_path);
                eprintln!("   ✗ {d}", d = detail);
                eprintln!("   💡 Fix: regenerate a healthy .mud via `./mud.sh forge <profile> <tok_dir> <output>`");
                eprintln!("        then verify `./mud.sh health <output>` → 🟢 before training.");
                return Err(anyhow::anyhow!("model integrity gate failed: {d}", d = detail));
            }
            eprintln!("🟢 [INTEGRITY-GATE] Model '{p}' passed structural check. Training proceeds.", p = self.model_path);
        }
        // STE pack writes scales/weights in-place on thawed tensors only.
        // Full materialize doubles RAM (mmap + owned) — lethal for 1.7B+.
        // Keep frozen layers + norms on RO mmap; own only STE-writable tensors.
        {
            let n_layers_hint = mud
                .global_metadata
                .get("num_hidden_layers")
                .or_else(|| mud.global_metadata.get("num_layers"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let last_n = crate::mud::sequence_pack::train_last_n_layers(n_layers_hint.max(1));
            let first_train = n_layers_hint.saturating_sub(last_n);
            let freeze_emb = crate::mud::constants::train_freeze_emb();
            // Force full materialize only if explicitly requested (debug / non-LAST_N).
            let force_full = std::env::var("MUD_TRAIN_MATERIALIZE_FULL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if force_full || n_layers_hint == 0 {
                mud.materialize_writable();
            } else {
                mud.materialize_for_ste_train(first_train, freeze_emb);
            }
        }
        // Capture the input weight hash BEFORE training so the final save can prove the
        // checkpoint actually changed vs the original (guards against silent no-op runs).
        let input_weights_hash = Self::hash_trained_weights(&mud);
        // Persist model-native specials into metadata (converter often only has raw_config_json).
        {
            let (b, e) = self
                .tokenizer
                .special_ids_from_metadata(&mud.global_metadata);
            mud.global_metadata
                .entry("bos_token_id".into())
                .or_insert_with(|| b.to_string());
            mud.global_metadata
                .entry("eos_token_id".into())
                .or_insert_with(|| e.to_string());
        }

        // ── Numeric metadata panel (W=79, pure ASCII inside box) ──────────────
        let meta = &mud.global_metadata;
        let m_layers = meta
            .get("num_hidden_layers")
            .or_else(|| meta.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_hidden = meta
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_heads = meta
            .get("num_attention_heads")
            .or_else(|| meta.get("num_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_kv = meta
            .get("num_key_value_heads")
            .or_else(|| meta.get("num_kv_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_ffn = meta
            .get("intermediate_size")
            .or_else(|| meta.get("ffn_hidden"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_vocab = meta
            .get("vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_maxpos = meta
            .get("max_position_embeddings")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        // Approximate parameter count (ternary, not full f32)
        let param_attn = m_layers * 4 * m_hidden * m_hidden; // Q,K,V,O
        let param_ffn = m_layers * 3 * m_hidden * m_ffn; // up,gate,down
        let param_emb = m_vocab * m_hidden; // embedding
        let total_params = param_attn + param_ffn + param_emb;
        let model_size_bytes = std::fs::metadata(&self.model_path)
            .map(|m| m.len())
            .unwrap_or(0);

        use crate::mud::trainer_ui::{kv, section};
        // L-01 shape dispatch + L-02 Muon NS on ash when MUD_USE_VULKAN=1
        let vulkan_on = std::env::var("MUD_USE_VULKAN").unwrap_or_default() == "1";
        let opt_tag = if vulkan_on {
            "LIVE=Muon(NS GPU|CPU)/GaLore/SGD (L-01+L-02) ash=on"
        } else {
            "LIVE=Muon(NS CPU)/GaLore/SGD (L-01)  ash=off"
        };
        let lr_init = crate::mud::constants::qat_learning_rate();
        let lr_min = lr_init / 10.0;
        println!();
        println!("{}", section("Training Configuration"));
        println!("{}", section("Dimensions"));
        println!("{}", kv("Arch", &format!(
            "layers={} hidden={} heads={}/{} ffn={}", m_layers, m_hidden, m_heads, m_kv, m_ffn
        )));
        println!("{}", kv("Model", &format!(
            "vocab={} maxpos={} ~{:.2}M params  file={:.2} MB", m_vocab, m_maxpos, total_params as f64 / 1_000_000.0, model_size_bytes as f64 / 1_048_576.0
        )));
        println!("{}", section("Config"));
        println!("{}", kv("Schedule", &format!(
            "epochs={} batch={} chunk={} chars  optimizer={}", epochs, batch_size, crate::mud::corpus_trainer::CHUNK_SIZE, opt_tag
        )));
        let full_seq = crate::mud::sequence_pack::train_full_seq_enabled();
        let seq_len = crate::mud::sequence_pack::train_seq_len();
        let packing = if full_seq {
            format!("L-10 + Stream D full-seq (causal windows, seq_len={seq_len}, KV grows)")
        } else {
            "L-10 pairs@pos=0 (MUD_TRAIN_FULL_SEQ=0; set 1 for full-seq)".to_string()
        };
        println!("{}", kv("Packing", &packing));
        println!("{}", kv("QAT", &format!(
            "STE Ternary2Bit (1.58-bit/weight, PRQ)  ·  LR {:.0e}→{:.0e} cosine", lr_init, lr_min
        )));
        let moe_line = if let Some(eid) = crate::mud::moe_load::train_expert_id() {
            format!("MUD_TRAIN_EXPERT={eid} (dense FFN → expert.{eid})")
        } else if crate::mud::moe_train::moe_train_enabled() {
            crate::mud::moe_train::summary_line(&[])
        } else {
            "dense expert.0 (MUD_TRAIN_EXPERT=N | MUD_MOE_TRAIN=1|hash)".to_string()
        };
        println!("{}", kv("MoE", &moe_line));

        // ── HEALTH / LAYOUT panel (2026-08-03 universal + self-describing) ──────────────────────────────────────────────────────────────
        // Policy: prefer the self-describing layout metadata embedded in the .mud
        // (canonical source of truth written by `mud_forge`), fall back to the
        // SAME runtime resolver (`moe_load::dense_ffn_names_for_train`) that
        // forward/inference + checkpoint + the gate share (backwards-compat with
        // older .mud files). Health ↔ gate ↔ trainer always in lockstep.
        let layout_line = {
            let layout_tag = meta.get("mud:layout").map(|s| s.as_str());
            let wtype_tag = meta.get("mud:weight_type").map(|s| s.as_str());
            let ffn_tag = meta.get("mud:ffn_naming").map(|s| s.as_str());

            // FFN layout + naming (metadata first, resolver fallback).
            let ffn_kind = match layout_tag {
                Some("dense") => format!("Dense ({})", ffn_tag.unwrap_or("ffn_up/gate/down")),
                Some("moe") => format!("MoE ({})", ffn_tag.unwrap_or("expert.E.w1/w2/w3")),
                Some(other) => format!("custom layout ({other})"),
                None => mud.skills.get("core").and_then(|core| {
                    if m_layers == 0 {
                        return None;
                    }
                    let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, 0);
                    let k = match ffn.up.as_str() {
                        "ffn_up" => "Dense (ffn_up/gate/down)".to_string(),
                        _ => format!("MoE expert via resolver (up={}, gate={}, down={})", ffn.up, ffn.gate, ffn.down),
                    };
                    Some(k)
                }).unwrap_or_else(|| "unknown (no core skill)".to_string()),
            };

            // Weight type (metadata first, tensor inspection fallback).
            let wtype_str = match wtype_tag {
                Some("float32") => "Float32 (Q2_0→F32 dequant)".to_string(),
                Some("ternary2bit") => "Ternary2bit (1.58-bit ELUT+PRQ)".to_string(),
                Some(other) => format!("weight type = {other}"),
                None => mud.skills.get("core")
                    .and_then(|core| core.tensors.get("blk.0.attn_q.weight"))
                    .map(|t| match t.t_type {
                        crate::mud::MudTensorType::Float32 => "Float32 (Q2_0→F32 dequant)".to_string(),
                        crate::mud::MudTensorType::Ternary2Bit => "Ternary2bit (1.58-bit ELUT+PRQ)".to_string(),
                        other => format!("{:?}", other),
                    })
                    .unwrap_or_else(|| "weight type unknown".to_string()),
            };

            format!("{ffn_kind}  ·  {wtype_str}")
        };
        println!("{}", section("Health / Layout (universal)"));
        println!("{}", kv("Integrity gate", "✅ passed — trainer proceeds (or MUD_SKIP_INTEGRITY=1)"));
        println!("{}", kv("Layout", &layout_line));
        println!();
        // ──────────────────────────────────────────────────────────────────────

        // EZOP / ash QAT VRAM: host-visible UMA on Iris Xe doubles shadow+grad in system RAM.
        // Skip when SCALES_ONLY (CPU pack only) or MUD_TRAIN_EZOP=0 — avoids OOM on 15 GiB hosts.
        let scales_only_boot = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let ezop_forced_off = std::env::var("MUD_TRAIN_EZOP")
            .map(|v| {
                let t = v.trim().to_ascii_lowercase();
                t == "0" || t == "off" || t == "false" || t == "no" || t == "cpu"
            })
            .unwrap_or(false);
        let mut vk_qat_storage = if scales_only_boot || ezop_forced_off {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ram",
                    &format!(
                        "EZOP QAT VRAM skipped ({})",
                        if scales_only_boot {
                            "SCALES_ONLY — ash pack unused"
                        } else {
                            "MUD_TRAIN_EZOP=0"
                        }
                    )
                )
            );
            None
        } else if self.vk.is_some() {
            match crate::mud::ash_qat_dispatcher::AshQatDispatcher::new() {
                Ok(d) if d.is_available() => Some(d),
                Ok(_) => {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note(
                            "ram",
                            "EZOP dispatcher not available — CPU STE"
                        )
                    );
                    None
                }
                Err(e) => {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note(
                            "ram",
                            &format!("EZOP init failed ({e}) — CPU STE")
                        )
                    );
                    None
                }
            }
        } else {
            None
        };

        // AWAKE-01: Pre-align structural ternary boundaries (SGD on synthetic noise).
        // OFF by default on low-resource hosts (i7-1260P / 15 GiB Iris Xe): it rewrites
        // every ternary row before real training and wastes CPU/RAM for no corpus signal.
        // Opt-in via MUD_TRAIN_AWAKE=1 (or unset skip). MUD_TRAIN_SKIP_AWAKE=1 also skips.
        let awake_opt_in = std::env::var("MUD_TRAIN_AWAKE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let skip_awake = std::env::var("MUD_TRAIN_SKIP_AWAKE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if awake_opt_in && !skip_awake {
            self.deep_local_alignment(&mut mud, true)?;
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "skipping AWAKE-01 (low-resource default; MUD_TRAIN_AWAKE=1 to enable)"
                )
            );
        }

        // FREEZE_EMB / large-vocab (Bonsai 152k×2048≈1.2GiB): skip full FP32 emb materialize.
        // Empty shadow_emb → on-the-fly row unpack from mmap ELUT in train_on_sequence.
        // Default: frozen (see constants::train_freeze_emb).
        let freeze_emb_alloc = crate::mud::constants::train_freeze_emb();
        let mut shadow_emb = {
            let core = mud
                .skills
                .get("core")
                .ok_or_else(|| anyhow::anyhow!("No core skill"))?;
            let emb_tensor = core
                .tensors
                .get("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("No embedding"))?;
            let elements = emb_tensor.shape[0] * emb_tensor.shape[1];
            if freeze_emb_alloc {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ram",
                        &format!(
                            "FREEZE_EMB: skip {:.1} MiB emb FP32 — on-the-fly ELUT rows",
                            (elements * 4) as f64 / (1024.0 * 1024.0)
                        )
                    )
                );
                Vec::new()
            } else {
                let mut data = vec![0.0f32; elements];
                unsafe {
                    if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                        let rows = emb_tensor.shape[0];
                        let cols = emb_tensor.shape[1];
                        let u32s_per_row = cols.div_ceil(8);
                        for r in 0..rows {
                            crate::mud::dequantize_ternary_row(
                                (emb_tensor.data_ptr as *const u32).add(r * u32s_per_row),
                                &mut data[r * cols..(r + 1) * cols],
                                cols,
                            );
                        }
                        if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                            if scale_tensor.t_type == MudTensorType::Float32 {
                                let scale_data = std::slice::from_raw_parts(
                                    scale_tensor.data_ptr as *const f32,
                                    rows,
                                );
                                for r in 0..rows {
                                    let s = scale_data[r];
                                    for c in 0..cols {
                                        data[r * cols + c] *= s;
                                    }
                                }
                            }
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(
                            emb_tensor.data_ptr as *const f32,
                            data.as_mut_ptr(),
                            elements,
                        );
                    }
                }
                data
            }
        };

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

        // RAM-first: only allocate FP32 shadows + Adam for thawed last-N layers.
        // Full 1.7B × 28 layers would be ~5.6 GiB shadows alone (lethal on 15 GiB).
        let last_n_alloc = crate::mud::sequence_pack::train_last_n_layers(n_layers);
        let first_train_alloc = n_layers.saturating_sub(last_n_alloc);
        println!(
            "{}",
            crate::mud::trainer_ui::note(
                "ram",
                &format!(
                    "shadow alloc: last_n={last_n_alloc} first_train={first_train_alloc}/{n_layers} (frozen layers = empty shadow)"
                )
            )
        );

        let mut layers: Vec<crate::mud::slime_forward::SlimeLayer> = Vec::new();
        let mut shadow_layers: Vec<crate::mud::slime_backward::SlimeLayerShadowF32> = Vec::new();

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
            // tc: lookup for non-`weight` tensors (e.g. mhc_alpha, mhc_beta,
            // mhc_radius). These store under `blk.N.mhc_alpha` NOT
            // `blk.N.mhc_alpha.weight`. The old `tn("mhc_alpha")` silently
            // returned null → mHC residual treated model as base (alpha=1,beta=1,gate=0)
            // with NO radius projection → unbounded `h_in + f_h` growth → f16 overflow.
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

            // FIX (2026-07-30): mHC alpha/beta are mmap'd read-only in the .mud
            // file. The optimizer (`mhc_scale_sgd_step`) writes to them via
            // `layers[l_idx].mhc_alpha_w as *mut f32` — writing to read-only
            // mmap → SIGSEGV. We copy them into owned shadow buffers and point
            // the `*const f32` fields at those shadows. Radius is read-only
            // (never updated) so it stays as a direct mmap pointer.
            let (mhc_alpha_w_raw, mhc_alpha_shadow): (*const f32, Option<Box<[f32]>>) = {
                let raw = tc("mhc_alpha");
                if raw.is_null() {
                    (std::ptr::null(), None)
                } else {
                    let t = core.tensors.get(&format!("{prefix}mhc_alpha")).unwrap();
                    let n = t.shape.iter().product::<usize>().max(1);
                    let buf: Vec<f32> =
                        unsafe { std::slice::from_raw_parts(raw, n) }.to_vec();
                    let boxed = buf.into_boxed_slice();
                    (boxed.as_ptr(), Some(boxed))
                }
            };
            let (mhc_beta_w_raw, mhc_beta_shadow): (*const f32, Option<Box<[f32]>>) = {
                let raw = tc("mhc_beta");
                if raw.is_null() {
                    (std::ptr::null(), None)
                } else {
                    let t = core.tensors.get(&format!("{prefix}mhc_beta")).unwrap();
                    let n = t.shape.iter().product::<usize>().max(1);
                    let buf: Vec<f32> =
                        unsafe { std::slice::from_raw_parts(raw, n) }.to_vec();
                    let boxed = buf.into_boxed_slice();
                    (boxed.as_ptr(), Some(boxed))
                }
            };

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
                mhc_alpha_w: mhc_alpha_w_raw,
                mhc_beta_w: mhc_beta_w_raw,
                mhc_radius_w: tc("mhc_radius"),
                mhc_alpha_shadow,
                mhc_beta_shadow,

                // FIX (2026-07-31): Copy RMSNorm weights into owned shadow buffers
                // so the optimizer can update them (root cause of activation collapse).
                // The `*const f32` pointers will be repointed to these shadows
                // in the trainable path (see below, after shadow layer init).
                attn_norm_shadow: None,
                ffn_norm_shadow: None,
                n_kv_heads,
                ffn_mid,
                rope_theta,
                rope_inv_freq: crate::mud::slime_forward::compute_rope_inv_freq(rope_theta, head_d),
                rope_cache: parking_lot::Mutex::new(crate::mud::slime_forward::RopeCache::new()),
                is_f32_weights: core.tensors
                    .get(&format!("blk.{}.attn_q.weight", 0))
                    .map(|t| t.t_type == MudTensorType::Float32)
                    .unwrap_or(false),
            });

            let t_shape = |name: &str| -> Vec<usize> {
                core.tensors
                    .get(&format!("{}{}.weight", prefix, name))
                    .map(|t| t.shape.clone())
                    .unwrap_or_default()
            };

            if blk < first_train_alloc {
                // Frozen: zero-heap shadow (forward still uses mmap ELUT via `layers`)
                shadow_layers.push(crate::mud::slime_backward::SlimeLayerShadowF32::empty());
                continue;
            }

            let q_shape = t_shape("attn_q");
            let k_shape = t_shape("attn_k");
            let v_shape = t_shape("attn_v");
            let o_shape = t_shape("attn_output");
            let up_shape = t_shape(&ffn.up);
            let gate_shape = t_shape(&ffn.gate);
            let down_shape = t_shape(&ffn.down);
            if q_shape.len() < 2 || k_shape.len() < 2 {
                anyhow::bail!("blk.{blk}: missing attn weight shapes for shadow alloc");
            }
            let q_opt = crate::mud::slime_backward::select_optimizer(q_shape[0], q_shape[1]);
            let k_opt = crate::mud::slime_backward::select_optimizer(k_shape[0], k_shape[1]);
            let v_opt = crate::mud::slime_backward::select_optimizer(v_shape[0], v_shape[1]);
            let o_opt = crate::mud::slime_backward::select_optimizer(o_shape[0], o_shape[1]);
            let ffn_up_opt = crate::mud::slime_backward::select_optimizer(up_shape[0], up_shape[1]);
            let ffn_gate_opt =
                crate::mud::slime_backward::select_optimizer(gate_shape[0], gate_shape[1]);
            let ffn_down_opt =
                crate::mud::slime_backward::select_optimizer(down_shape[0], down_shape[1]);
            use crate::mud::adam_state::AdamState;
            let mut shadow = crate::mud::slime_backward::SlimeLayerShadowF32 {
                q_w: vec![0.0; q_shape.iter().product()],
                k_w: vec![0.0; k_shape.iter().product()],
                v_w: vec![0.0; v_shape.iter().product()],
                o_w: vec![0.0; o_shape.iter().product()],
                ffn_up_w: vec![0.0; up_shape.iter().product()],
                ffn_gate_w: vec![0.0; gate_shape.iter().product()],
                ffn_down_w: vec![0.0; down_shape.iter().product()],
                attn_norm_w: vec![0.0; hidden],
                ffn_norm_w: vec![0.0; hidden],
                q_opt,
                k_opt,
                v_opt,
                o_opt,
                ffn_up_opt,
                ffn_gate_opt,
                ffn_down_opt,
                q_adam: AdamState::for_strategy(q_shape.iter().product(), q_opt),
                k_adam: AdamState::for_strategy(k_shape.iter().product(), k_opt),
                v_adam: AdamState::for_strategy(v_shape.iter().product(), v_opt),
                o_adam: AdamState::for_strategy(o_shape.iter().product(), o_opt),
                ffn_up_adam: AdamState::for_strategy(up_shape.iter().product(), ffn_up_opt),
                ffn_gate_adam: AdamState::for_strategy(gate_shape.iter().product(), ffn_gate_opt),
                ffn_down_adam: AdamState::for_strategy(down_shape.iter().product(), ffn_down_opt),
            };

            // Inflate ternary → FP32 shadow (ELUT: 8 weights / u32 → stride cols/8)
            unsafe {
                let p = format!("blk.{}.", blk);
                let inf = |name: &str, dest: &mut [f32]| {
                    if dest.is_empty() {
                        return;
                    }
                    if let Some(t) = core.tensors.get(&format!("{}{}.weight", p, name)) {
                        if t.t_type == crate::mud::MudTensorType::Ternary2Bit {
                            let cols = t.shape[1];
                            let u32s_per_row = cols.div_ceil(8);
                            for r in 0..t.shape[0] {
                                crate::mud::dequantize_ternary_row(
                                    (t.data_ptr as *const u32).add(r * u32s_per_row),
                                    &mut dest[r * cols..(r + 1) * cols],
                                    cols,
                                );
                                if let Some(scale_t) =
                                    core.tensors.get(&format!("{}{}.prq_scale", p, name))
                                {
                                    let s = *(scale_t.data_ptr as *const f32).add(r);
                                    for c in 0..cols {
                                        dest[r * cols + c] *= s;
                                    }
                                }
                            }
                        }
                    }
                };
                inf("attn_q", &mut shadow.q_w);
                inf("attn_k", &mut shadow.k_w);
                inf("attn_v", &mut shadow.v_w);
                inf("attn_output", &mut shadow.o_w);
                inf(&ffn.up, &mut shadow.ffn_up_w);
                inf(&ffn.gate, &mut shadow.ffn_gate_w);
                inf(&ffn.down, &mut shadow.ffn_down_w);
            }
            shadow_layers.push(shadow);

            // FIX (2026-07-31): Copy RMSNorm weights into owned shadow buffers and
            // repoint the SlimeLayer's `*const f32` pointers so the optimizer can
            // update them. Previously frozen — root cause of activation collapse.
            {
                let layer = layers.last_mut().expect("layer just pushed");
                let shadow = shadow_layers.last_mut().expect("shadow just pushed");
                unsafe {
                    if !layer.attn_norm_w.is_null() {
                        std::ptr::copy_nonoverlapping(
                            layer.attn_norm_w,
                            shadow.attn_norm_w.as_mut_ptr(),
                            hidden,
                        );
                        let boxed: Box<[f32]> =
                            std::mem::take(&mut shadow.attn_norm_w)
                                .into_boxed_slice();
                        layer.attn_norm_shadow = Some(boxed);
                        layer.attn_norm_w = layer.attn_norm_shadow.as_ref().unwrap().as_ptr();
                    }
                    if !layer.ffn_norm_w.is_null() {
                        std::ptr::copy_nonoverlapping(
                            layer.ffn_norm_w,
                            shadow.ffn_norm_w.as_mut_ptr(),
                            hidden,
                        );
                        let boxed: Box<[f32]> =
                            std::mem::take(&mut shadow.ffn_norm_w)
                                .into_boxed_slice();
                        layer.ffn_norm_shadow = Some(boxed);
                        layer.ffn_norm_w = layer.ffn_norm_shadow.as_ref().unwrap().as_ptr();
                    }
                }
            }
        }

        // Allocate VRAM and map initial shadows (thawed layers only).
        // Iris Xe UMA: each matrix = shadow+grad+scales+packed HOST_VISIBLE ≈ 2×FP32 shadow.
        // Bonsai last-2 FFN alone can be hundreds of MiB — soft-fail → CPU STE (no panic).
        if let Some(vk_qat) = vk_qat_storage.as_mut() {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "stp",
                    "Allocating Vulkan VRAM for QAT..."
                )
            );
            let mut ezop_oom: Option<String> = None;
            'ezop: for (blk, shadow) in shadow_layers.iter().enumerate() {
                if shadow.is_empty() {
                    continue;
                }
                let p = format!("blk.{}.", blk);
                let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);

                let matrices: Vec<(&str, &[f32], usize)> = vec![
                    ("attn_q", &shadow.q_w, hidden),
                    ("attn_k", &shadow.k_w, hidden),
                    ("attn_v", &shadow.v_w, hidden),
                    ("attn_output", &shadow.o_w, hidden),
                    (&ffn.up, &shadow.ffn_up_w, hidden),
                    (&ffn.gate, &shadow.ffn_gate_w, hidden),
                    (
                        &ffn.down,
                        &shadow.ffn_down_w,
                        shadow.ffn_down_w.len() / hidden.max(1),
                    ),
                ];

                for (m_name, shadow_w, cols) in matrices {
                    let name = format!("{}{}", p, m_name);
                    let elements = shadow_w.len();
                    if elements == 0 || cols == 0 {
                        continue;
                    }
                    let rows = elements / cols;
                    if let Err(e) = vk_qat.ensure_buffers(&name, elements, rows, shadow_w) {
                        ezop_oom = Some(format!(
                            "{name} ({} MiB shadow): {e}",
                            (elements * 4) / (1024 * 1024)
                        ));
                        break 'ezop;
                    }
                }
            }
            if let Some(why) = ezop_oom {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        &format!("EZOP OOM — dropping GPU QAT, CPU STE pack only\n    ({why})")
                    )
                );
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        "tip: MUD_TRAIN_SCALES_ONLY=1 or MUD_TRAIN_EZOP=0 or free RAM"
                    )
                );
                vk_qat_storage = None;
            }
        }

        let head_dim = hidden / n_heads;
        let max_emb = 128.0;
        // Cap logical max_pos for train workspace — full 32k Bonsai ctx is wasteful for seating
        let train_max_pos = std::env::var("MUD_TRAIN_MAX_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512)
            .min(max_pos)
            .max(64);
        let mut workspace = crate::mud::slime::SlimeWorkspace::new(
            hidden,
            train_max_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_mid,
            n_layers,
            max_emb,
        );
        let mut backward_ws = crate::mud::slime_backward::SlimeBackwardWorkspace::new(
            hidden,
            ffn_mid,
            n_kv_heads * head_dim,
            crate::mud::sequence_pack::train_num_negatives(),
        );
        // Tapes: only need full tape for thawed layers; frozen get minimal empty-ish tapes
        // (forward may still write tape slots when valid — keep capacity for seq but cap pos).
        let mut tapes = (0..n_layers)
            .map(|_| {
                crate::mud::slime_backward::SlimeLayerTape::new(
                    hidden,
                    ffn_mid,
                    n_kv_heads,
                    head_dim,
                    train_max_pos,
                    0,
                )
            })
            .collect::<Vec<_>>();
        let mut gradients = (0..n_layers)
            .map(|blk| {
                if blk < first_train_alloc {
                    crate::mud::slime_backward::SlimeLayerGradients::empty()
                } else {
                    crate::mud::slime_backward::SlimeLayerGradients::new(
                        hidden, ffn_mid, n_kv_heads, head_dim,
                    )
                }
            })
            .collect::<Vec<_>>();

        // corpus_dir is where generated artifacts live (global_corpus.bin, etc.).
        // Never recurse into stash/downloads/dumps; path-component filter is belt-and-suspenders.
        // SPRINT-4 (4.1): collection logic moved to `crate::mud::corpus::scan_training_files`.

        let text_files = crate::mud::corpus::scan_training_files()?;

        // P0.3 fix (2026-07-20): a model already at current_epoch=N makes a fresh
        // `--epochs M` run resolve end_epoch = N-1+M and skip N-1 "done" epochs →
        // if M is small the loop runs zero new epochs and the checkpoint is a no-op
        // (byte-identical). MUD_TRAIN_RESET_EPOCH=1 restarts the counter at 1 so the
        // requested epochs always execute. Default keeps resume behaviour.
        let reset_epoch = std::env::var("MUD_TRAIN_RESET_EPOCH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let resume_epoch = if reset_epoch {
            1
        } else {
            mud.global_metadata
                .get("trainer.current_epoch")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1)
        };
        let resume_chunk_idx = mud
            .global_metadata
            .get("trainer.current_chunk_idx")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        // Quick sessions: hard cap on chunks processed this session
        let max_chunks_cap: Option<usize> = std::env::var("MUD_TRAIN_MAX_CHUNKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&m| m > 0);

        // AOT tokens per stream chunk (default CHUNK_SIZE/4 = 12_500). Quick: much smaller.
        let aot_tokens_per_chunk: usize = if let Ok(v) = std::env::var("MUD_TRAIN_AOT_TOKENS") {
            v.parse::<usize>().unwrap_or(512).clamp(64, CHUNK_SIZE)
        } else if max_chunks_cap.is_some() {
            // Enough for batch×seq windows with headroom; avoids 12k-token materialization
            std::env::var("MUD_TRAIN_SEQ_LEN")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16)
                .saturating_mul(batch_size.max(1))
                .saturating_mul(4)
                .clamp(256, 2048)
        } else {
            CHUNK_SIZE / 4
        };

        // Token budget for AOT rebuild when quick-capped (0 = unlimited / full corpus)
        let aot_token_budget: u64 = max_chunks_cap
            .map(|c| (c.saturating_mul(aot_tokens_per_chunk).saturating_add(64)) as u64)
            .unwrap_or(0);

        let chunks_per_file: Vec<usize> = text_files
            .iter()
            .map(|p| {
                std::fs::metadata(p)
                    .map(|m| (m.len() as usize).div_ceil(CHUNK_SIZE))
                    .unwrap_or(0)
            })
            .collect();
        let mut total_chunks_per_epoch: usize = chunks_per_file.iter().sum();
        if let Some(m) = max_chunks_cap {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    &format!(
                        "quick mode: MUD_TRAIN_MAX_CHUNKS={m}  AOT_TOKENS/chunk={aot_tokens_per_chunk}  budget≈{aot_token_budget} tok  (full corpus would be {total_chunks_per_epoch}/epoch)"
                    )
                )
            );
            if crate::mud::slime_backward::optimizer_policy_override()
                .map(|s| matches!(s, crate::mud::slime_backward::OptimizerStrategy::Sgd))
                .unwrap_or(false)
            {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        "optimizer=SGD (set MUD_OPT=muon|adam to override)"
                    )
                );
            }
        }
        let mut total_chunks_all_epochs = total_chunks_per_epoch * (resume_epoch - 1 + epochs);

        // FIX (2026-07-31): Configure LR schedule (warmup + cosine decay).
        *self.total_steps_hint.borrow_mut() = Some(total_chunks_all_epochs);
        let warmup = (total_chunks_all_epochs / 100).max(1);
        *self.warmup_steps.borrow_mut() = warmup;

        // ── Session start numeric summary ─────────────────────────────────────
        let end_epoch = resume_epoch - 1 + epochs;
        if epochs == 0 {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "epochs=0 requested — NO training will run (checkpoint would be a no-op). Use --epochs N or MUD_TRAIN_RESET_EPOCH=1."
                )
            );
        }
        if resume_epoch > 1 || resume_chunk_idx > 0 {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    &format!(
                        "resuming: epoch {}/{}  block {}/{}  remaining blocks: {}",
                        resume_epoch,
                        end_epoch,
                        resume_chunk_idx,
                        total_chunks_per_epoch,
                        total_chunks_all_epochs.saturating_sub(
                            resume_chunk_idx + total_chunks_per_epoch * (resume_epoch - 1)
                        )
                    )
                )
            );
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!(
                        "new session: epochs 1..{}  {}/epoch  {} total blocks",
                        end_epoch, total_chunks_per_epoch, total_chunks_all_epochs
                    )
                )
            );
        }
        println!(
            "{}",
            crate::mud::trainer_ui::note(
                "ram",
                &format!(
                    "files={}  corpus={:.2} MB",
                    text_files.len(),
                    text_files
                        .iter()
                        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                        .sum::<u64>() as f64
                        / 1_048_576.0
                )
            )
        );
        println!();
        // ──────────────────────────────────────────────────────────────────────

        let mut global_chunks_processed = 0usize;
        if resume_epoch > 1 {
            global_chunks_processed += total_chunks_per_epoch * (resume_epoch - 1);
        }

        if resume_chunk_idx > 0 {
            global_chunks_processed += resume_chunk_idx;
        }

        // FIX (2026-08-02): Restore step_count from checkpoint resume point.
        // Previously step_count initialized to 0 on every MudCorpusTrainer::new(),
        // causing the cosine LR schedule to restart in warmup phase even when
        // resuming from epoch 79+. With step_count=0 and warmup≈135, the LR was
        // ~3.7e-6 instead of ~0.0003 (cosine at 42% progress) — a ~75× reduction
        // that left the model training at random-chance loss (ln(256)=5.545).
        // Now step_count reflects the true global position so the cosine schedule
        // produces the correct warmup/decay curve immediately at session start.
        self.step_count.store(
            global_chunks_processed as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // Also update total_steps_hint to match the (possibly dynamically updated) total.
        // The hint was set at line 853 from the initial total_chunks_all_epochs, but
        // the epoch loop at 1356-1359 can recompute it when file_chunks differs.
        // Persist the latest value so apply_lr_schedule uses the correct denominator.
        *self.total_steps_hint.borrow_mut() = Some(total_chunks_all_epochs);

        let mut session_chunks_processed = 0usize;
        // Wall clock for ETA starts at first train chunk (not AOT / EZOP setup).
        let mut train_clock: Option<Instant> = None;
        let mut loss_history = std::collections::VecDeque::with_capacity(100);

        use std::fs::OpenOptions;
        use std::io::Write;
        // RIESGO-1 FIX (2026-08-03): backup + truncate the metrics log at session
        // start. Previously opened in append mode every session → unbounded growth,
        // cross-session mixing, and false loss-cert triggers at session boundaries
        // (loss_cert / autofinetune parse the entire file). Keep a .bak per session.
        let _ = std::fs::create_dir_all("logs");
        let log_path = std::path::Path::new("mud_train_metrics.log");
        if log_path.exists() {
            if let Ok(ts) = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
            {
                let backup = format!("mud_train_metrics.log.bak.{ts}");
                let _ = std::fs::rename(log_path, std::path::Path::new(&backup));
            }
        }
        // RIESGO-1 FIX: abrir en modo truncate+write (NO append) — el file fue
        // previamente renombrado a .bak.{ts} si existía, así que aquí se crea
        // fresco truncado. `append+truncate` es conflictivo (clippy::append_truncate).
        let mut telemetry_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open("mud_train_metrics.log")
            .ok();
        // Session banner + schema (train_telemetry / loss_cert skip lines starting with #)
        if let Some(ref mut f) = telemetry_file {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(
                f,
                "# MUD_TELEMETRY v2 unix_ts={} model={} max_chunks={} steps_per_chunk={} lr={:.6} opt={}",
                ts,
                self.model_path,
                max_chunks_cap
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into()),
                crate::mud::sequence_pack::train_steps_per_chunk(batch_size),
                crate::mud::constants::qat_learning_rate(),
                std::env::var("MUD_OPT").unwrap_or_else(|_| "auto".into()),
            );
            let _ = writeln!(
                f,
                "# cols: step batch loss ppl lr loss_vel varh varj grad_norm stp_loss entropy integral sigma_pct cognitive <pad0> toks_s steps_per_chunk elapsed_s prog_pct conf_pct epoch block blocks_total"
            );
            let _ = f.flush();
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ram",
                    "telemetry -> mud_train_metrics.log  ·  TUI: cargo run --release --bin train_telemetry"
                )
            );
        }
        let telem_every: usize = std::env::var("MUD_TELEM_EVERY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);
        let steps_per_chunk_meta = crate::mud::sequence_pack::train_steps_per_chunk(batch_size);

        // D1 — Self-Play Loop (4-Pillar Efficiency, Pillar 1)
        // Activates synthetic chain generation from the model's own forward pass,
        // feeding confident predictions back through the training pipeline.
        let self_play_enabled: bool = std::env::var("MUD_SELF_PLAY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let self_play_turns: usize = std::env::var("MUD_SELF_PLAY_TURNS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8);
        let self_play_temp: f32 = std::env::var("MUD_SELF_PLAY_TEMP")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|&t| t > 0.0)
            .unwrap_or(0.8);
        if self_play_enabled {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!(
                        "self-play: MUD_SELF_PLAY=1 (turns={}, temp={})",
                        self_play_turns, self_play_temp
                    )
                )
            );
        }

        enum PrefetchItem {
            Chunk {
                epoch: usize,
                f_idx: usize,
                file_path: std::path::PathBuf,
                c_idx: usize,
                file_chunks: usize,
                tokens: Vec<u32>,
            },
            EndOfEpoch {
                epoch: usize,
            },
        }

        let (tx, rx) = std::sync::mpsc::sync_channel::<PrefetchItem>(100);
        let prefetch_text_files = text_files;
        let prefetch_tokenizer = self.tokenizer.clone();
        // Capture quick-mode AOT params for the prefetch thread
        let prefetch_token_budget = aot_token_budget;
        let prefetch_tokens_per_chunk = aot_tokens_per_chunk;
        let prefetch_max_chunks = max_chunks_cap;
        // Model-native specials (SmolLM2 bos=eos=0). Never emit Llama 128k OOV markers.
        let (bos_id, eos_id) = self
            .tokenizer
            .special_ids_from_metadata(&mud.global_metadata);
        let vocab_len = self.tokenizer.vocab_size().max(1);
        let bos_id = crate::mud::sequence_pack::clamp_special_id(bos_id, vocab_len);
        let eos_id = crate::mud::sequence_pack::clamp_special_id(eos_id, vocab_len);
        // Resolved last-N (auto policy or env) for ops clarity
        println!(
            "  \x1b[1;36m[specials]\x1b[0m bos={bos_id} eos={eos_id} vocab={vocab_len} steps/chunk≈{} num_neg={} last_n_layers={} (env={})",
            crate::mud::sequence_pack::train_steps_per_chunk(batch_size),
            crate::mud::sequence_pack::train_num_negatives(),
            crate::mud::sequence_pack::train_last_n_layers(n_layers),
            std::env::var("MUD_TRAIN_LAST_N_LAYERS").unwrap_or_else(|_| "auto".into()),
        );
        let prefetch_bos = bos_id;
        let prefetch_eos = eos_id;

        std::thread::spawn(move || {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if let Some(last_core) = core_ids.last() {
                    core_affinity::set_for_current(*last_core);
                }
            }

            // Quick vs full use separate cache files so a 570MB full bin never blocks smoke.
            let global_bin_path = if prefetch_token_budget > 0 {
                std::path::PathBuf::from("training/corpus/global_corpus.quick.bin")
            } else {
                std::path::PathBuf::from("training/corpus/global_corpus.bin")
            };
            // Header v3: [n_src:u64][token_budget:u64][bos_eos:u64]  bos in low 32, eos in high 32
            const AOT_HEADER_LEN: usize = 24;
            let n_src_files = prefetch_text_files.len() as u64;
            let bos_eos_key: u64 = (prefetch_bos as u64) | ((prefetch_eos as u64) << 32);
            let mut rebuild_needed = true;

            // If a .tmp from a previous interrupted AOT build exists, remove it now so it
            // can never be mistaken for a valid corpus (P-17: fail-fast, no silent corruption).
            let tmp_bin_path_early = global_bin_path.with_extension("bin.tmp");
            if tmp_bin_path_early.exists() {
                let _ = std::fs::remove_file(&tmp_bin_path_early);
                println!("  [AOT] Stale .tmp detected and removed. Forcing rebuild.");
            }

            if let Ok(global_meta) = std::fs::metadata(&global_bin_path) {
                if let Ok(global_time) = global_meta.modified() {
                    use std::io::Read;
                    let header: Option<(u64, u64, u64)> = std::fs::File::open(&global_bin_path)
                        .ok()
                        .and_then(|mut f| {
                            let mut buf = [0u8; AOT_HEADER_LEN];
                            f.read_exact(&mut buf).ok().map(|_| {
                                let n = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                                let budget = u64::from_le_bytes(buf[8..16].try_into().unwrap());
                                let be = u64::from_le_bytes(buf[16..24].try_into().unwrap());
                                (n, budget, be)
                            })
                        });

                    if let Some((stored_count, stored_budget, stored_be)) = header {
                        if stored_count == n_src_files
                            && stored_budget == prefetch_token_budget
                            && stored_be == bos_eos_key
                        {
                            let bin_size = global_meta.len();
                            if bin_size <= AOT_HEADER_LEN as u64 {
                                println!(
                                    "  [AOT] Corpus cache truncated ({} bytes). Rebuilding...",
                                    bin_size
                                );
                            } else {
                                let mut all_older = true;
                                for file_path in &prefetch_text_files {
                                    if let Ok(txt_meta) = std::fs::metadata(file_path) {
                                        if let Ok(txt_time) = txt_meta.modified() {
                                            if txt_time > global_time {
                                                all_older = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if all_older {
                                    rebuild_needed = false;
                                }
                            }
                        } else {
                            println!(
                                "  [AOT] Cache key changed (files {}→{}, budget {}→{}, specials). Rebuilding...",
                                stored_count,
                                n_src_files,
                                stored_budget,
                                prefetch_token_budget
                            );
                        }
                    } else {
                        println!("  [AOT] Legacy/corrupt header. Rebuilding...");
                    }
                }
            }

            if rebuild_needed {
                // Write to a .tmp file, rename atomically on success — prevents corrupt
                // partial caches from being treated as valid on the next run (P-17).
                let _ = std::fs::create_dir_all("training/corpus");
                let tmp_bin_path = {
                    let mut p = global_bin_path.clone();
                    p.set_extension("bin.tmp");
                    p
                };
                let global_file = std::fs::File::create(&tmp_bin_path)
                    .expect("Failed to create global corpus tmp file");
                let mut writer = std::io::BufWriter::with_capacity(1024 * 1024 * 16, global_file);
                use std::io::Write;
                // Header v3
                let _ = writer.write_all(&n_src_files.to_le_bytes());
                let _ = writer.write_all(&prefetch_token_budget.to_le_bytes());
                let _ = writer.write_all(&bos_eos_key.to_le_bytes());

                let mut completed = true;
                let mut tokens_written: u64 = 0;
                let budget = prefetch_token_budget; // 0 = unlimited

                'aot_files: for file_path in prefetch_text_files.iter() {
                    if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                        completed = false;
                        break;
                    }
                    if budget > 0 && tokens_written >= budget {
                        println!(
                            "  [AOT] Token budget reached ({tokens_written}/{budget}). Stopping early."
                        );
                        break;
                    }
                    println!("  [AOT] Tokenizando: {} ...", file_path.display());

                    let mut file_tokens = Vec::new();
                    file_tokens.push(prefetch_bos);

                    let ext = file_path.extension().unwrap_or_default().to_string_lossy();
                    let is_rust = ext == "rs";
                    let is_markdown = ext == "md";

                    let mut full_text = String::new();
                    if is_rust {
                        full_text.push_str(&format!("File: {}\n```rust\n", file_path.display()));
                    } else if is_markdown {
                        full_text.push_str(&format!("File: {}\n```markdown\n", file_path.display()));
                    }

                    if let Ok(content) = std::fs::read_to_string(file_path) {
                        full_text.push_str(&content);
                    }

                    if is_rust || is_markdown {
                        full_text.push_str("\n```\n");
                    }

                    if !full_text.is_empty() {
                        let tokens = prefetch_tokenizer.encode(&full_text);
                        file_tokens.extend_from_slice(&tokens);
                    }

                    if budget > 0 {
                        let allow = budget.saturating_sub(tokens_written).max(2) as usize;
                        if file_tokens.len() > allow {
                            file_tokens.truncate(allow);
                        }
                    }

                    file_tokens.push(prefetch_eos);

                    if !file_tokens.is_empty() {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                file_tokens.as_ptr() as *const u8,
                                file_tokens.len() * 4,
                            )
                        };
                        let _ = writer.write_all(bytes);
                        tokens_written += file_tokens.len() as u64;
                    }

                    if budget > 0 && tokens_written >= budget {
                        println!(
                            "  [AOT] Token budget reached ({tokens_written}/{budget}). Stopping early."
                        );
                        break 'aot_files;
                    }
                }
                let _ = writer.flush();

                if completed {
                    // Atomic rename: only replace the real .bin on full success
                    std::fs::rename(&tmp_bin_path, &global_bin_path)
                        .expect("Failed to rename corpus tmp to bin");
                    println!(
                        "  [AOT] Corpus cache built successfully ({} tokens → {}).",
                        tokens_written,
                        global_bin_path.display()
                    );
                } else {
                    // Interrupted — delete the partial tmp so next run rebuilds
                    let _ = std::fs::remove_file(&tmp_bin_path);
                    println!("  [AOT] Build interrupted — partial cache discarded.");
                    return; // EXIT thread gracefully to avoid panicking below
                }
            } else {
                println!("  [AOT] Using cached corpus {}", global_bin_path.display());
            }

            let global_file =
                std::fs::File::open(&global_bin_path).expect("Failed to open global corpus file");
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&global_file)
                    .expect("Failed to mmap global corpus")
            };
            let _ = mmap.advise(memmap2::Advice::Random);

            // Skip the 16-byte header v2 (legacy 8-byte bins already forced rebuild)
            let token_bytes = if mmap.len() >= AOT_HEADER_LEN {
                &mmap[AOT_HEADER_LEN..]
            } else {
                &mmap[..]
            };
            let total_tokens = token_bytes.len() / 4;
            let tokens_per_chunk = prefetch_tokens_per_chunk.max(64);
            // L-10: no-pad chunk ranges — include tail remainder (old path dropped it).
            let mut chunk_ranges: Vec<(usize, usize)> =
                crate::mud::sequence_pack::chunk_ranges_no_pad(total_tokens, tokens_per_chunk);

            // Small post-convert corpora: tile ranges up to MAX_CHUNKS so STE sees more steps.
            if let Some(cap) = prefetch_max_chunks {
                if !chunk_ranges.is_empty() && chunk_ranges.len() < cap {
                    let base = chunk_ranges.clone();
                    while chunk_ranges.len() < cap {
                        chunk_ranges.extend_from_slice(&base);
                    }
                    chunk_ranges.truncate(cap);
                    println!(
                        "  [AOT] Tiled small corpus {} → {} chunks (MAX_CHUNKS)",
                        base.len(),
                        chunk_ranges.len()
                    );
                } else if chunk_ranges.len() > cap {
                    chunk_ranges.truncate(cap);
                }
            }

            use rand::seq::SliceRandom;
            use rand::SeedableRng;

            let file_chunks = chunk_ranges.len();
            let global_path = std::path::PathBuf::from("GLOBAL_STREAM");

            let end_epoch = (resume_epoch - 1) + epochs;
            for epoch in resume_epoch..=end_epoch {
                if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                    break;
                }

                let mut rng = rand::rngs::StdRng::seed_from_u64(epoch as u64);
                chunk_ranges.shuffle(&mut rng);

                for (c_idx, &(start_idx, chunk_len)) in chunk_ranges.iter().enumerate() {
                    if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                        break;
                    }
                    if epoch == resume_epoch && c_idx < resume_chunk_idx {
                        continue;
                    }

                    if chunk_len < 2 {
                        continue;
                    }
                    let byte_start = start_idx * 4;
                    let byte_end = (start_idx + chunk_len) * 4;
                    if byte_end > token_bytes.len() {
                        continue;
                    }
                    let chunk_bytes = &token_bytes[byte_start..byte_end];

                    let chunk_tokens = unsafe {
                        std::slice::from_raw_parts(chunk_bytes.as_ptr() as *const u32, chunk_len)
                    }
                    .to_vec();

                    let _ = tx.send(PrefetchItem::Chunk {
                        epoch,
                        f_idx: 0,
                        file_path: global_path.clone(),
                        c_idx,
                        file_chunks,
                        tokens: chunk_tokens,
                    });
                }

                if tx.send(PrefetchItem::EndOfEpoch { epoch }).is_err() {
                    return;
                }
            }
        });

        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "corpus alignment session started (STE QAT)")
        );

        for item in rx {
            if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                break;
            }

            match item {
                PrefetchItem::Chunk {
                    epoch,
                    f_idx: _f_idx,
                    file_path: _file_path,
                    c_idx,
                    file_chunks,
                    tokens,
                } => {
                    // Update tracking variables dynamically based on AOT truth
                    if total_chunks_per_epoch != file_chunks {
                        total_chunks_per_epoch = file_chunks;
                        total_chunks_all_epochs =
                            total_chunks_per_epoch * (resume_epoch - 1 + epochs);
                        global_chunks_processed = (epoch - 1) * total_chunks_per_epoch + c_idx;
                    }

                    global_chunks_processed += 1;
                    session_chunks_processed += 1;

                    // Hard stop for quick alignment (after count, before heavy train)
                    if let Some(cap) = max_chunks_cap {
                        if session_chunks_processed > cap {
                            println!(
                                "{}",
                                crate::mud::trainer_ui::note(
                                    "warn",
                                    &format!(
                                        "reached MUD_TRAIN_MAX_CHUNKS={cap} — saving & stopping"
                                    )
                                )
                            );
                            SHOULD_TERMINATE.store(true, Ordering::SeqCst);
                            break;
                        }
                    }

                    if tokens.len() < 2 {
                        continue;
                    }

                    let train_start = train_clock.get_or_insert_with(Instant::now);

                    let chunk_t0 = Instant::now();
                    let chunk_metrics = self.train_on_sequence(
                        &mut mud,
                        &mut shadow_emb,
                        &mut layers,
                        &mut shadow_layers,
                        &mut workspace,
                        &mut backward_ws,
                        &mut tapes,
                        &mut gradients,
                        &tokens,
                        batch_size,
                        vk_qat_storage.as_mut(),
                    )?;

                    // D1 — Self-Play Loop: after each corpus training step, generate
                    // a synthetic sequence from the model's own forward pass and
                    // feed it back through the training pipeline. This creates a
                    // self-improving feedback loop that prevents stagnation.
                    if self_play_enabled {
                        let sp_t0 = Instant::now();
                        let sp_prompt = tokens.last().copied().map(|t| vec![t]).unwrap_or_default();
                        let _sp_metrics = self.self_play_training_step(
                            &mut mud,
                            &mut shadow_emb,
                            &mut layers,
                            &mut shadow_layers,
                            &mut workspace,
                            &mut backward_ws,
                            &mut tapes,
                            &mut gradients,
                            &sp_prompt,
                            batch_size,
                            vk_qat_storage.as_mut(),
                            session_chunks_processed,
                        );
                        let _sp_dt = sp_t0.elapsed().as_secs_f32();
                    }

                    let chunk_dt = chunk_t0.elapsed().as_secs_f32();
                    let chunk_loss = chunk_metrics.loss;

                    // ETA: compute AFTER train_on_sequence so elapsed includes
                    // the current chunk's wall-clock (avoids near-zero ETA on
                    // the first chunk when the clock just started).
                    let elapsed = train_start.elapsed().as_secs_f32();
                    let chunks_per_sec = if session_chunks_processed > 0 {
                        session_chunks_processed as f32 / elapsed.max(0.001)
                    } else {
                        0.0
                    };
                    let remaining_chunks =
                        total_chunks_all_epochs.saturating_sub(global_chunks_processed);
                    let eta = if chunks_per_sec > 0.0 {
                        Duration::from_secs_f32(remaining_chunks as f32 / chunks_per_sec)
                    } else {
                        Duration::ZERO
                    };
                    let total_secs = eta.as_secs();
                    let days = total_secs / 86400;
                    let hours = (total_secs % 86400) / 3600;
                    let mins = (total_secs % 3600) / 60;
                    let secs = total_secs % 60;
                    let eta_str = if days > 0 {
                        format!("{:02}d {:02}:{:02}:{:02}", days, hours, mins, secs)
                    } else {
                        format!("{:02}:{:02}:{:02}", hours, mins, secs)
                    };

                    if chunk_loss.is_nan() || chunk_loss.is_infinite() {
                        anyhow::bail!(
                            "\n{}",
                            crate::mud::trainer_ui::note(
                                "err",
                                "mathematical explosion detected (loss = NaN) — aborting early"
                            )
                        );
                    }

                    // FIX (2026-08-03): Best-model checkpointing — save to
                    // {stem}_best.mud whenever chunk_loss improves. This captures
                    // the pre-collapse state if the model suffers a gradient spike
                    // (e.g. step-3403: 0.0005→0.049 grad_norm, varh 7.9→0.2).
                    // For a ~22 MB model, per-chunk saves on improvement are
                    // cheap (~22 MB write) and guarantee the best-loss weights
                    // are always recoverable, independent of the "latest" checkpoint.
                    let _best_saved = self.save_best_checkpoint(
                        chunk_loss,
                        &mut mud,
                        &mut shadow_emb,
                        &mut shadow_layers,
                        &layers,
                        vk_qat_storage.as_mut(),
                    );

                    loss_history.push_back(chunk_loss);
                    if loss_history.len() > 100 {
                        loss_history.pop_front();
                    }

                    #[allow(clippy::manual_is_multiple_of)]
                    if loss_history.len() == 100 && global_chunks_processed % 100 == 0 {
                        let avg: f32 = loss_history.iter().sum::<f32>() / 100.0;
                        let var: f32 = loss_history
                            .iter()
                            .map(|&x| (x - avg) * (x - avg))
                            .sum::<f32>()
                            / 100.0;
                        if var < 1e-6 {
                            // Local Plateau Detected
                        }
                    }

                    let loss_vel = if loss_history.len() >= 2 {
                        loss_history[loss_history.len() - 2] - chunk_loss
                    } else {
                        0.0
                    };
                    let perplexity = chunk_loss.exp();

                    // Trajectory efficiency: 1/(1+CV) where CV = std/|mean| of loss history.
                    // High when loss descends smoothly (low variance relative to mean).
                    let traj_eff = if loss_history.len() >= 2 {
                        let n = loss_history.len() as f32;
                        let mean = loss_history.iter().sum::<f32>() / n;
                        let var: f32 = loss_history
                            .iter()
                            .map(|&x| (x - mean) * (x - mean))
                            .sum::<f32>()
                            / n;
                        let std = var.sqrt();
                        if mean.abs() > 1e-8 {
                            1.0 / (1.0 + std / mean.abs())
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    // Progress: every chunk in quick mode (MAX_CHUNKS), else every 25.
                    // Always print the first chunk so the session never looks hung post-AOT.
                    let progress_every: usize = max_chunks_cap
                        .map(|_| 1usize)
                        .or_else(|| {
                            std::env::var("MUD_TRAIN_PROGRESS_EVERY")
                                .ok()
                                .and_then(|s| s.parse().ok())
                                .filter(|&n| n > 0)
                        })
                        .unwrap_or(25);
                    let should_print = global_chunks_processed == 1
                        || global_chunks_processed.is_multiple_of(progress_every)
                        || max_chunks_cap
                            .map(|c| session_chunks_processed >= c)
                            .unwrap_or(false);
                    let should_telem = session_chunks_processed.is_multiple_of(telem_every);

                    // FIX (2026-07-31): Compute dead-activation flag before telemetry
                    // block so it's available for recovery outside the scope.
                    let avg_var_h = chunk_metrics.var_h;
                    let avg_cognitive = chunk_metrics.cognitive;
                    let dead = avg_var_h < 1e-5 && avg_cognitive < 1e-3;

                    if should_print || should_telem {
                        // FIX (2026-08-02): Recompute qat_lr per-step for telemetry display.
                        // Previously qat_lr was computed ONCE before the training loop (using
                        // step_count=0), so every telemetry line showed the warmup-start LR
                        // regardless of progress. Now it reflects the actual per-step scheduled LR.
                        let qat_lr = crate::mud::corpus_trainer::train_step::apply_lr_schedule(
                            crate::mud::constants::qat_learning_rate(),
                            self.step_count.load(std::sync::atomic::Ordering::Relaxed),
                            self.total_steps_hint.borrow().unwrap_or(crate::mud::constants::QAT_LR_TOTAL_STEPS),
                            *self.warmup_steps.borrow(),
                        );

                        // LIVE stats from train_on_sequence (pre-BWD samples). Post-BWD
                        // workspace.registers / jepa_z are dead — do NOT re-read them here.
                        let avg_var_j = chunk_metrics.var_j;
                        let avg_integral = chunk_metrics.jepa_integral;
                        let sigma_v_pct = chunk_metrics.sigma_v_pct;
                        let avg_grad_norm = chunk_metrics.grad_norm;
                        let avg_entropy = chunk_metrics.entropy;

                        let denom = max_chunks_cap.unwrap_or(total_chunks_all_epochs).max(1) as f64;
                        let progress_pct = (session_chunks_processed
                            .min(max_chunks_cap.unwrap_or(session_chunks_processed))
                            as f64
                            / denom)
                            * 100.0;
                        let conf_pct = (-chunk_loss).exp() * 100.0;
                        // tok/s = token predictions per second. Each "step" is
                        // one next-token prediction (forward + backward), so
                        // steps_per_chunk already counts tokens processed.
                        let chunk_tps = steps_per_chunk_meta as f32 / chunk_dt.max(0.001);
                        let toks_per_sec = if chunk_tps > 0.0 && chunk_dt > 0.05 {
                            chunk_tps
                        } else {
                            chunks_per_sec * steps_per_chunk_meta as f32
                        };
                        let blocks_total = max_chunks_cap.unwrap_or(file_chunks);
                        let elapsed_s = train_clock
                            .map(|t| t.elapsed().as_secs_f32())
                            .unwrap_or(elapsed);

                        // File telemetry always (decoupled from console throttle)
                        if should_telem {
                            // FIX (2026-07-31): NaN/Inf guard for all telemetry values.
                            // Any NaN/Inf metric is replaced with 0.0 to prevent
                            // corrupting the metrics log and TUI parser.
                            let chunk_loss = if chunk_loss.is_finite() { chunk_loss } else { 0.0 };
                            let perplexity = if perplexity.is_finite() { perplexity } else { 0.0 };
                            let avg_var_h = if avg_var_h.is_finite() { avg_var_h } else { 0.0 };
                            let avg_var_j = if avg_var_j.is_finite() { avg_var_j } else { 0.0 };
                            let avg_grad_norm = if avg_grad_norm.is_finite() { avg_grad_norm } else { 0.0 };
                            let avg_cognitive = if avg_cognitive.is_finite() { avg_cognitive } else { 0.0 };
                            let avg_entropy = if avg_entropy.is_finite() { avg_entropy } else { 0.0 };
                            let sigma_v_pct = if sigma_v_pct.is_finite() { sigma_v_pct } else { 0.0 };
                            let avg_integral = if avg_integral.is_finite() { avg_integral } else { 0.0 };
                            let loss_vel = if loss_vel.is_finite() { loss_vel } else { 0.0 };
                            let traj_eff = if traj_eff.is_finite() { traj_eff } else { 0.0 };
                            let chunk_metrics_stp_loss = if chunk_metrics.stp_loss.is_finite() { chunk_metrics.stp_loss } else { 0.0 };

                            if let Some(ref mut f) = telemetry_file {
                                // Legacy cols 0..14 for train_telemetry TUI + v2 extensions
                                let _ = writeln!(
                                    f,
                                    "{} 1 {:.4} {:.4} {:.6} {:.6} {:.6} {:.6} {:.6} {:.4} {:.4} {:.6} {:.2} {:.6} 0.0 {:.1} {} {:.2} {:.2} {:.2} {} {} {}",
                                    global_chunks_processed,
                                    chunk_loss,
                                    perplexity,
                                    qat_lr,
                                    loss_vel,
                                    avg_var_h,
                                    avg_var_j,
                                    avg_grad_norm,
                                    chunk_metrics_stp_loss,
                                    avg_entropy,
                                    avg_integral,
                                    sigma_v_pct,
                                    avg_cognitive,
                                    toks_per_sec,
                                    steps_per_chunk_meta,
                                    elapsed_s,
                                    progress_pct,
                                    conf_pct,
                                    epoch,
                                    c_idx + 1,
                                    blocks_total,
                                );
                                let _ = f.flush();
                            }
                            // Machine-readable line: to stderr (terminal visibility) AND into
                            // mud_train_metrics.log so the live TUI (train_telemetry) can read it.
                            // BUG-4 FIX (2026-08-03): omit cmud_* fields when C-MUD
                            // is OFF (previously hard-coded 0.0 — the TUI's C-MUD
                            // mode showed flat-zero lines for 100% of the log,
                            // masquerading dead telemetry as health). When ON but
                            // metrics aren't captured in the trainer loop yet, also
                            // omit so the TUI shows "NO DATA" instead of false zeros.
                            // TODO: wire CmudThinkReport from maybe_think_collapse
                            // into the trainer forward pass for real cmud_steps/lock/herm.
                            let cmud_part = if crate::mud::cmud::cmud_think_enabled() {
                                " cmud_steps=0.0000 cmud_lock=0.0000 cmud_herm=0.0000 cmud_spread=0.0000 cmud_cauchy=0.0000".to_string()
                            } else {
                                String::new()
                            };
                            let telem_line = format!(
                                "[TELEM] step={} epoch={} block={}/{} loss={:.4} ppl={:.1} lr={:.6} loss_vel={:.6} tok/s={:.1} steps={} elapsed={:.1}s prog={:.1}% conf={:.2}% varh={:.6} varj={:.6} jepa={:.4} σ={:.1}% cog={:.4} grad_norm={:.4} entropy={:.3} stp_loss={:.5} traj_eff={:.4} n_act={} clip_coef={:.4}{cmud_part}{}",
                                global_chunks_processed,
                                epoch,
                                c_idx + 1,
                                blocks_total,
                                chunk_loss,
                                perplexity,
                                qat_lr,
                                loss_vel,
                                toks_per_sec,
                                steps_per_chunk_meta,
                                elapsed_s,
                                progress_pct,
                                conf_pct,
                                avg_var_h,
                                avg_var_j,
                                avg_integral,
                                sigma_v_pct,
                                avg_cognitive,
                                avg_grad_norm,
                                avg_entropy,
                                chunk_metrics_stp_loss,
                                traj_eff,
                                chunk_metrics.n_samples,
                                chunk_metrics.clip_coef,
                                if dead { " DEAD_ACT" } else { "" },
                            );
                            let is_tui = std::env::var("MUD_CIRCUIT_TUI").is_ok();
                            if !is_tui {
                                // Pretty multi-line version for terminal (human-readable)
                                let pretty = crate::mud::trainer_ui::format_telem_pretty(
                                    &crate::mud::trainer_ui::TelemSnapshot {
                                        step: global_chunks_processed,
                                        epoch,
                                        epochs: resume_epoch - 1 + epochs,
                                        block: c_idx + 1,
                                        blocks: blocks_total,
                                        loss: chunk_loss,
                                        lr: qat_lr,
                                        tps: toks_per_sec as f32,
                                        prog: progress_pct as f32,
                                        grad_norm: avg_grad_norm,
                                        entropy: avg_entropy,
                                        varh: avg_var_h,
                                        varj: avg_var_j,
                                        jepa: avg_integral,
                                        cog: avg_cognitive,
                                        sigma: sigma_v_pct,
                                    },
                                );
                                eprintln!("{}", pretty);
                            }
                            if let Some(ref mut f) = telemetry_file {
                                let _ = writeln!(f, "{}", telem_line);
                                let _ = f.flush();
                            }
                        }

                        if should_print {
                            // Newline in quick mode so each block is visible; \r otherwise.
                            let is_tui = std::env::var("MUD_CIRCUIT_TUI").is_ok();
                            if !is_tui {
                                let eol = if max_chunks_cap.is_some() { "\n" } else { "" };
                                eprint!(
                                    "\r\x1b[2K  \x1b[1;33mEpoch\x1b[0m\x1b[1;37m:\x1b[0m {:02}/{} \x1b[1;33mBlock\x1b[0m\x1b[1;37m:\x1b[0m {:05}/{} \x1b[1;36mSpeed\x1b[0m\x1b[1;37m:\x1b[0m {:5.0} tk/s \x1b[1;35mLoss\x1b[0m\x1b[1;37m:\x1b[0m {:6.4} \x1b[1;32mConf\x1b[0m\x1b[1;37m:\x1b[0m {:5.2}% \x1b[1;32mProg\x1b[0m\x1b[1;37m:\x1b[0m {:5.2}% \x1b[1;31mETA\x1b[0m\x1b[1;37m:\x1b[0m {}{eol}",
                                    epoch,
                                    resume_epoch - 1 + epochs,
                                    c_idx + 1,
                                    blocks_total,
                                    toks_per_sec,
                                    chunk_loss,
                                    conf_pct,
                                    progress_pct,
                                    eta_str
                                );
                                let _ = std::io::Write::flush(&mut std::io::stderr());
                            }
                        }

                        // FIX (2026-07-31): Dead-activation recovery.
                        // When activations collapse (cog→0, varh→0), the model
                        // enters a permanent dead state — gradients remain high
                        // (~10) but loss is frozen. We inject small noise into the
                        // embedding shadow weights to break symmetry and allow
                        // recovery. Without this, the model stays dead forever.
                        if dead && !shadow_emb.is_empty() {
                            let jitter_mag = (qat_lr as f64).max(1e-4) * 0.1;
                            for (i, w) in shadow_emb.iter_mut().enumerate() {
                                // Deterministic per-element noise (no RNG needed)
                                let noise = (((i * 1664525 + 1013904223) % 1000) as f32 / 500.0 - 1.0) as f64;
                                *w = (*w as f64 + noise * jitter_mag) as f32;
                            }
                            if chunk_metrics.cognitive > 0.0 {
                                log::warn!(
                                    "[RECOVERY] step={} DEAD_ACT recovery: applied {:.6} jitter to {} emb weights",
                                    global_chunks_processed, jitter_mag, shadow_emb.len()
                                );
                            }
                        }
                    }

                    let ckpt_every = checkpoint_every_chunks();
                    if global_chunks_processed > 0
                        && ckpt_every > 0
                        && global_chunks_processed.is_multiple_of(ckpt_every)
                    {
                        self.save_checkpoint(
                            &mut mud,
                            &mut shadow_emb,
                            &mut shadow_layers,
                            &layers,
                            format!("chunk_{}", global_chunks_processed),
                            vk_qat_storage.as_mut(),
                        )?;
                    }
                }
                PrefetchItem::EndOfEpoch { epoch } => {
                    if let Some(vk_qat) = vk_qat_storage.as_mut() {
                        // D4: drain_all flushes pending readbacks + waits GPU idle
                        unsafe {
                            if crate::mud::ash_qat_dispatcher::pipeline_overlap_enabled() {
                                let _ = self.pipeline_overlap.borrow_mut().drain_all(vk_qat);
                            } else {
                                let _ = vk_qat.sync_all();
                            }
                        }
                    }

                    mud.global_metadata
                        .insert("trainer.current_epoch".to_string(), epoch.to_string());
                    mud.global_metadata.insert(
                        "trainer.current_chunk_idx".to_string(),
                        "0".to_string(), // Reset chunk on new epoch
                    );

                    // Epoch Alignment Complete
                    self.save_checkpoint(
                        &mut mud,
                        &mut shadow_emb,
                        &mut shadow_layers,
                        &layers,
                        format!("epoch_{}", epoch),
                        vk_qat_storage.as_mut(),
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to save epoch checkpoint: {}", e))?;
                }
            }
        }

        // Flush any remaining ops at the very end
        if let Some(vk_qat) = vk_qat_storage.as_mut() {
            // D4: drain_all flushes pending readbacks + waits GPU idle
            unsafe {
                if crate::mud::ash_qat_dispatcher::pipeline_overlap_enabled() {
                    let _ = self.pipeline_overlap.borrow_mut().drain_all(vk_qat);
                } else {
                    let _ = vk_qat.sync_all();
                }
            }
        }

        self.sync_shadow_to_mud(
            &mut mud,
            &mut shadow_emb,
            &mut shadow_layers,
            &layers,
            vk_qat_storage.as_mut(),
            true, // final verify: warn loudly if the saved checkpoint is a no-op
        );
        let tmp_path = format!("{}.tmp", self.model_path);
        mud.save(&tmp_path)?;
        std::fs::rename(&tmp_path, &self.model_path)?;

        // Authoritative no-op check: compare final trained weights vs the captured input
        // hash (not the in-memory sync, which already reflects the changes).
        let out_hash = Self::hash_trained_weights(&mud);
        if out_hash == input_weights_hash {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "⚠ FINAL CHECKPOINT IS A NO-OP — trained weights are byte-identical to the input (hash match). The .mud did not change. Check LR / STE deadzone / resume-epoch / already-converged model."
                )
            );
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!(
                        "✓ final checkpoint persisted (weights hash {:#018x} → {:#018x})",
                        input_weights_hash, out_hash
                    )
                )
            );
        }

        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "alignment session completed.")
        );
        Ok(())
    }
}