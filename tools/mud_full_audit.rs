//! # MUD Full Audit Suite (L-15 era)
//!
//! Unified structural + policy + feature validation for a `.mud` model and the
//! live engine ledger. Exit code 0 only if all critical checks pass.
//!
//! Usage: `mud_full_audit [model.mud]`
//!        `./mud.sh audit-full [model]`

use forge_llm::mud::cmud::{self, ComplexF32, GaussTernary};
use forge_llm::mud::grad_checkpoint::CheckpointPolicy;
use forge_llm::mud::kv_context::KvContextPolicy;
use forge_llm::mud::p13::{health_constants_ok, parse_arch_dims};
use forge_llm::mud::slime::SlimeWorkspace;
use forge_llm::mud::slime_backward::select_optimizer;
use forge_llm::mud::{MudFile, MudTensorType};
use std::process::ExitCode;

fn main() -> ExitCode {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2.mud".to_string());

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  MUD FULL AUDIT  ·  ledger L-01…L-15 validation         ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Model: {model_path}\n");

    let mut critical = 0u32;
    let mut warnings = 0u32;

    // ── [0] Constants / P-13 SSOT ──────────────────────────────────────────
    print_section("0", "Constants & P-13 health");
    if health_constants_ok() {
        ok("EPSILON_FLOOR / PCorePool policy");
    } else {
        fail("constants health", &mut critical);
    }

    // ── [1] Load + architecture ────────────────────────────────────────────
    print_section("1", "Model load & architecture (P-13)");
    let mud = match MudFile::load(&model_path) {
        Ok(m) => {
            ok("MudFile::load");
            m
        }
        Err(e) => {
            fail(&format!("load: {e}"), &mut critical);
            return ExitCode::from(1);
        }
    };
    let dims = match parse_arch_dims(&mud) {
        Ok(d) => {
            ok(&format!(
                "arch hidden={} L={} heads={}/{} ffn={}",
                d.hidden_size, d.num_layers, d.num_heads, d.num_kv_heads, d.intermediate_size
            ));
            d
        }
        Err(e) => {
            fail(&format!("parse_arch_dims: {e}"), &mut critical);
            return ExitCode::from(1);
        }
    };

    // ── [2] Tensor inventory ───────────────────────────────────────────────
    print_section("2", "Tensor inventory");
    let Some(core) = mud.skills.get("core") else {
        fail("missing core skill", &mut critical);
        return ExitCode::from(1);
    };
    let mut ternary = 0usize;
    let mut scales = 0usize;
    let mut f32_tensors = 0usize;
    for (name, t) in &core.tensors {
        match t.t_type {
            MudTensorType::Ternary2Bit => ternary += 1,
            MudTensorType::Float32 => {
                f32_tensors += 1;
                if name.ends_with(".prq_scale") || name.ends_with("norm.weight") {
                    scales += 1;
                }
            }
            _ => {}
        }
    }
    ok(&format!(
        "ternary={ternary} f32={f32_tensors} scale/norm≈{scales}"
    ));
    if ternary == 0 {
        warn("no Ternary2Bit weights", &mut warnings);
    }

    // Sample Q shape
    let q_name = "blk.0.attn_q.weight";
    if let Some(t) = core.tensors.get(q_name) {
        let expect = vec![dims.hidden_size, dims.hidden_size];
        if t.shape == expect {
            ok(&format!("{q_name} shape {:?}", t.shape));
        } else {
            fail(
                &format!("{q_name} shape {:?} != {:?}", t.shape, expect),
                &mut critical,
            );
        }
    } else {
        warn(
            "blk.0.attn_q.weight missing (non-standard layout?)",
            &mut warnings,
        );
    }

    // ── [3] L-13 context policy ────────────────────────────────────────────
    print_section("3", "L-13 KV / HCA 32k policy");
    let max_pos = mud
        .global_metadata
        .get("max_position_embeddings")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048usize);
    let pol = KvContextPolicy::resolve(max_pos);
    let (naive, actual) = pol.savings_vs_naive(dims.num_layers, dims.num_kv_heads, dims.head_dim());
    ok(&format!(
        "logical={} dense_cap={} hca_slots={} ratio={}",
        pol.logical_max_pos, pol.dense_cap, pol.hca_slots, pol.hca_ratio
    ));
    ok(&format!(
        "KV bytes naive≈{:.1}MB actual≈{:.1}MB (savings {:.1}×)",
        naive as f64 / 1e6,
        actual as f64 / 1e6,
        naive as f64 / actual.max(1) as f64
    ));
    // Workspace smoke alloc
    let head_dim = dims.head_dim();
    let ws = SlimeWorkspace::new(
        dims.hidden_size,
        max_pos,
        dims.num_heads,
        dims.num_kv_heads,
        head_dim,
        dims.intermediate_size,
        dims.num_layers,
        1.0,
    );
    if ws.dense_kv_cap == pol.dense_cap && ws.max_pos == pol.logical_max_pos {
        ok("SlimeWorkspace matches policy");
    } else {
        fail("workspace/policy mismatch", &mut critical);
    }

    // ── [3b] CSA lightning indexer (stream E) ──────────────────────────────
    print_section("3b", "CSA top-k lightning indexer (gap E)");
    {
        use forge_llm::mud::csa_indexer::{
            approx_hca_flop_ratio, select_top_k_indices, CsaPolicy, DEFAULT_CSA_TOP_K,
        };
        let csa = CsaPolicy::resolve();
        ok(&csa.summary());
        flag("MUD_CSA", &std::env::var("MUD_CSA").unwrap_or_default());
        flag(
            "MUD_CSA_TOP_K",
            &std::env::var("MUD_CSA_TOP_K").unwrap_or_default(),
        );
        flag(
            "MUD_CSA_INDEX_DIM",
            &std::env::var("MUD_CSA_INDEX_DIM").unwrap_or_default(),
        );
        // Top-k smoke
        let scores = [0.2f32, 4.0, 1.0, 7.0, 3.0, 0.5];
        let mut idx = Vec::new();
        select_top_k_indices(&scores, 3, &mut idx);
        if idx == vec![1, 3, 4] {
            ok("select_top_k_indices picks largest in time order");
        } else {
            fail(&format!("top-k wrong: {idx:?}"), &mut critical);
        }
        let n_slots = pol.hca_slots;
        let ratio = approx_hca_flop_ratio(
            n_slots,
            csa.top_k,
            csa.effective_index_dim(head_dim),
            head_dim,
        );
        ok(&format!(
            "HCA slots={n_slots} top_k={} → approx HCA FLOP ratio vs full={:.2} (default_k={DEFAULT_CSA_TOP_K})",
            csa.top_k, ratio
        ));
        if csa.should_sparse(n_slots) {
            ok("policy would sparse-select at full HCA capacity (inference)");
        } else {
            ok("policy keeps full HCA scan at this capacity (below threshold)");
        }
        // Sparse only when history large; short ctx never regresses
        if !csa.should_sparse(8) {
            ok("short history (8 blocks) → full scan (no over-sparsify)");
        } else {
            fail("CSA should not sparse 8 blocks", &mut critical);
        }
    }

    // ── [4] L-15 checkpoint policy ─────────────────────────────────────────
    print_section("4", "L-15 gradient checkpointing");
    let ckpt = CheckpointPolicy::resolve();
    let (full_b, seg_b) = ckpt.peak_activation_bytes(
        dims.num_layers,
        dims.hidden_size,
        dims.intermediate_size,
        dims.num_kv_heads,
        head_dim,
        pol.scores_len(),
    );
    ok(&format!(
        "mode={:?} segment={} full_tape≈{:.1}MB peak_seg≈{:.1}MB",
        ckpt.mode,
        ckpt.segment_size,
        full_b as f64 / 1e6,
        seg_b as f64 / 1e6
    ));
    if ckpt.is_segmented() && seg_b >= full_b {
        warn("segmented mode not saving memory vs full", &mut warnings);
    }

    // ── [5] Optimizer LIVE shapes ──────────────────────────────────────────
    print_section("5", "L-01 optimizer strategy (LIVE shapes)");
    let strategies = [
        ("attn_q", dims.hidden_size, dims.hidden_size),
        ("attn_k", dims.num_kv_heads * head_dim, dims.hidden_size),
        ("ffn_up", dims.intermediate_size, dims.hidden_size),
        ("ffn_down", dims.hidden_size, dims.intermediate_size),
    ];
    for (name, rows, cols) in strategies {
        let s = select_optimizer(rows, cols);
        ok(&format!("{name} {rows}×{cols} → {s:?}"));
    }

    // ── [6] Feature flags ──────────────────────────────────────────────────
    print_section("6", "Feature flags (opt-in paths)");
    flag(
        "MUD_USE_VULKAN",
        &std::env::var("MUD_USE_VULKAN").unwrap_or_default(),
    );
    flag(
        "MUD_GPU_GEMV",
        &std::env::var("MUD_GPU_GEMV").unwrap_or_default(),
    );
    flag(
        "MUD_GPU_GEMV_MIN",
        &std::env::var("MUD_GPU_GEMV_MIN").unwrap_or_default(),
    );
    flag(
        "MUD_GPU_GEMV_LOG",
        &std::env::var("MUD_GPU_GEMV_LOG").unwrap_or_default(),
    );
    flag(
        "MUD_GRAD_CKPT",
        &std::env::var("MUD_GRAD_CKPT").unwrap_or_default(),
    );
    flag(
        "MUD_CMUD_THINK",
        &std::env::var("MUD_CMUD_THINK").unwrap_or_default(),
    );
    flag(
        "MUD_MAX_POS",
        &std::env::var("MUD_MAX_POS").unwrap_or_default(),
    );
    flag(
        "MUD_PCORE_THREADS",
        &std::env::var("MUD_PCORE_THREADS").unwrap_or_default(),
    );
    flag(
        "MUD_TRAIN_EXPERT",
        &std::env::var("MUD_TRAIN_EXPERT").unwrap_or_default(),
    );
    flag(
        "MUD_TRAIN_FULL_SEQ",
        &std::env::var("MUD_TRAIN_FULL_SEQ").unwrap_or_default(),
    );
    flag(
        "MUD_TRAIN_SEQ_LEN",
        &std::env::var("MUD_TRAIN_SEQ_LEN").unwrap_or_default(),
    );
    flag(
        "MUD_MOE_CLONE",
        &std::env::var("MUD_MOE_CLONE").unwrap_or_default(),
    );
    flag(
        "MUD_MOE_TOP_K",
        &std::env::var("MUD_MOE_TOP_K").unwrap_or_default(),
    );

    // ── [6a] GEMV auto policy (stream C) ───────────────────────────────────
    print_section("6a", "GPU GEMV auto policy (gap C)");
    {
        use forge_llm::vulkan::ash_backend::{AshContext, GEMV_GPU_MIN_WORK};
        use forge_llm::vulkan::gemv_policy::{
            self, parse_gemv_mode, should_try_gpu, GemvGpuMode, GEMV_NEVER,
        };
        let mode = parse_gemv_mode();
        ok(&format!("parse_gemv_mode → {mode:?} (unset=auto)"));
        ok(&gemv_policy::policy_summary());
        // Smoke: shape gates
        if !should_try_gpu(7, 256) {
            ok("reject n_in not multiple of 8");
        } else {
            fail("accepted misaligned n_in", &mut critical);
        }
        match mode {
            GemvGpuMode::Off => ok("Off → CPU only"),
            GemvGpuMode::On => ok(&format!(
                "On → GPU when work≥{} (or MUD_GPU_GEMV_MIN)",
                gemv_policy::env_min_work_override().unwrap_or(GEMV_GPU_MIN_WORK)
            )),
            GemvGpuMode::Auto => {
                if !gemv_policy::vulkan_not_disabled() {
                    warn("Auto but MUD_USE_VULKAN=0 → CPU", &mut warnings);
                } else {
                    match AshContext::new() {
                        Ok(mut ctx) if ctx.is_available() => {
                            unsafe { gemv_policy::ensure_calibrated(&mut ctx) };
                            let min = gemv_policy::effective_min_work_resolved();
                            if min >= GEMV_NEVER {
                                ok("Auto calibrated: GPU never wins → AVX2 only");
                            } else if min == 0 {
                                warn("Auto pending calib (unexpected)", &mut warnings);
                            } else {
                                ok(&format!(
                                    "Auto calibrated: min_work={min} (GPU for larger GEMVs)"
                                ));
                            }
                            if let Some(r) = gemv_policy::last_report() {
                                ok(&format!("calib note: {}", r.note));
                                ok(&format!("calib samples: {}", r.samples.len()));
                            }
                        }
                        _ => ok("Auto: no ash device → CPU (publish_no_device path)"),
                    }
                }
            }
        }
    }

    // ── [6b] MoE load path (L-11 product) ──────────────────────────────────
    print_section("6b", "MoE .mud load + train-expert (gap B)");
    {
        use forge_llm::mud::moe_load::{
            default_top_k, dense_ffn_names_for_train, discover_expert_ids, load_model_buses,
            model_expert_stats, model_has_multi_expert, resolve_expert_ffn_names,
        };
        let (max_e, multi_layers) = model_expert_stats(&core.tensors, dims.num_layers);
        ok(&format!(
            "expert inventory: max_per_layer={max_e}, multi-expert layers={multi_layers}/{}",
            dims.num_layers
        ));
        let ids0 = discover_expert_ids(&core.tensors, 0);
        ok(&format!("layer 0 expert ids: {ids0:?}"));
        match resolve_expert_ffn_names(&core.tensors, 0, 0) {
            Some(n) => {
                if n.up.contains("w3") && n.gate.contains("w1") {
                    ok(&format!(
                        "FFN names expert.0: up={} gate={} down={} (w3=up,w1=gate)",
                        n.up, n.gate, n.down
                    ));
                } else if n.up.contains(".up") {
                    ok(&format!(
                        "FFN names expert.0: up={} gate={} down={} (up/gate alt)",
                        n.up, n.gate, n.down
                    ));
                } else {
                    warn(
                        &format!("unexpected FFN names: up={} gate={}", n.up, n.gate),
                        &mut warnings,
                    );
                }
            }
            None => {
                // Dense models without expert.* may use other naming; warn not fail
                warn(
                    "layer 0 expert.0 FFN not resolved (dense non-expert layout?)",
                    &mut warnings,
                );
            }
        }
        let train_names = dense_ffn_names_for_train(&core.tensors, 0);
        ok(&format!(
            "dense_ffn_names_for_train L0: up={} gate={} down={}",
            train_names.up, train_names.gate, train_names.down
        ));
        // Inverted mapping regression guard (historical bug: w1 as up)
        if train_names.up.ends_with(".w1") && train_names.gate.ends_with(".w3") {
            fail(
                "inverted FFN map (w1=up w3=gate) — must be w3=up w1=gate",
                &mut critical,
            );
        }
        let top_k = default_top_k();
        let buses = load_model_buses(
            &mud,
            dims.num_layers,
            dims.hidden_size,
            dims.intermediate_size,
            top_k,
        );
        let mounted: usize = buses
            .iter()
            .filter_map(|b| b.as_ref())
            .map(|b| b.mounted_count())
            .sum();
        let multi = model_has_multi_expert(&buses);
        ok(&format!(
            "load_model_buses: layers_with_bus={}, total_mounted_slots={}, multi_expert={multi}, top_k={top_k}",
            buses.iter().filter(|b| b.is_some()).count(),
            mounted
        ));
        if max_e == 0 && mounted == 0 {
            warn(
                "no experts mounted — check expert.* tensor names",
                &mut warnings,
            );
        }
    }

    // ── [7] L-14 C-MUD reasoning audit ────────────────────────────────────
    print_section("7", "L-14 C-MUD complex reasoning audit");
    // 7a: algebra identities (kernel sanity)
    let x = ComplexF32::new(2.0, 3.0);
    let y = cmud::gauss_mul(x, GaussTernary::new(0, 1));
    if (y.re + 3.0).abs() < 1e-5 && (y.im - 2.0).abs() < 1e-5 {
        ok("gauss_mul (×i) identity");
    } else {
        fail("gauss_mul broken", &mut critical);
    }
    let p = cmud::project_hermitian(ComplexF32::new(3.0, 4.0), 1.0);
    if (p.hermite_norm() - 1.0).abs() < 1e-4 {
        ok("hermitian project");
    } else {
        fail("hermitian project", &mut critical);
    }
    // 7b: end-to-end reasoning pipeline over a real forward (research §3)
    let cka = forge_llm::mud::inference::cmud_audit(&mud);
    for line in cka.summary_lines() {
        println!("{line}");
    }
    if cka.healthy() {
        ok(&format!(
            "C-MUD reasoning healthy (τ={}, phase_lock={}, ball_ok)",
            cka.think.steps, cka.think.phase_locked
        ));
    } else {
        let mut reasons = Vec::new();
        if !cka.forward_ok {
            reasons.push("forward error");
        }
        if !cka.logits_finite {
            reasons.push("non-finite logits");
        }
        if cka.logit_range_min <= 0.0 {
            reasons.push("zero dynamic range");
        }
        if cka.token0_dominant {
            reasons.push("token-0 dominance");
        }
        if cka.think.max_herm_norm > cka.think.radius * 1.01 {
            reasons.push("Hermitian ball violated");
        }
        fail(
            &format!("C-MUD reasoning unhealthy: {}", reasons.join(", ")),
            &mut critical,
        );
    }

    // ── [8] Sequence pack + full-seq smoke ─────────────────────────────────
    print_section("8", "L-10 packing + Stream D full-seq");
    let toks = vec![
        forge_llm::mud::sequence_pack::DEFAULT_BOS,
        1,
        2,
        forge_llm::mud::sequence_pack::DEFAULT_EOS,
        forge_llm::mud::sequence_pack::DEFAULT_BOS,
        3,
        forge_llm::mud::sequence_pack::DEFAULT_EOS,
    ];
    let pairs = forge_llm::mud::sequence_pack::pairs_from_stream(
        &toks,
        32,
        200_000,
        forge_llm::mud::sequence_pack::DEFAULT_EOS,
    );
    if pairs
        .iter()
        .all(|&(a, _)| a != forge_llm::mud::sequence_pack::DEFAULT_EOS as usize)
    {
        ok(&format!(
            "pairs_from_stream n={} (no EOS as input)",
            pairs.len()
        ));
    } else {
        fail("packing cross-EOS leak", &mut critical);
    }
    {
        use forge_llm::mud::sequence_pack::{
            train_full_seq_enabled, train_seq_len, windows_for_target_preds, windows_from_stream,
            DEFAULT_EOS,
        };
        let sl = train_seq_len();
        ok(&format!(
            "full_seq={} seq_len={} (MUD_TRAIN_FULL_SEQ / MUD_TRAIN_SEQ_LEN)",
            train_full_seq_enabled(),
            sl
        ));
        // Longer synthetic stream for window extraction
        let mut long: Vec<u32> = vec![forge_llm::mud::sequence_pack::DEFAULT_BOS];
        for i in 1..64u32 {
            long.push(i);
        }
        long.push(DEFAULT_EOS);
        let n_win = windows_for_target_preds(32, sl.min(16));
        let wins = windows_from_stream(&long, n_win, sl.min(16), DEFAULT_EOS);
        if wins.is_empty() {
            fail(
                "full-seq windows_from_stream empty on 64-token doc",
                &mut critical,
            );
        } else {
            ok(&format!(
                "seq windows n={} first={{start={},len={}}} preds≈{}",
                wins.len(),
                wins[0].start,
                wins[0].len,
                wins.iter().map(|w| w.n_preds()).sum::<usize>()
            ));
            // Cross-EOS guard: no window should include both docs if we add a second
            let mut two = long.clone();
            two.push(forge_llm::mud::sequence_pack::DEFAULT_BOS);
            two.extend_from_slice(&[100, 101, 102, DEFAULT_EOS]);
            let wins2 = windows_from_stream(&two, 20, 32, DEFAULT_EOS);
            let mut bad = false;
            for w in &wins2 {
                let slice = &two[w.start..w.end()];
                let mut saw_eos = false;
                for &t in slice {
                    if saw_eos {
                        bad = true;
                        break;
                    }
                    if t == DEFAULT_EOS {
                        saw_eos = true;
                    }
                }
            }
            if bad {
                fail("full-seq window crossed EOS", &mut critical);
            } else {
                ok("full-seq windows respect EOS boundaries");
            }
        }
        flag(
            "MUD_TRAIN_FULL_SEQ",
            &std::env::var("MUD_TRAIN_FULL_SEQ").unwrap_or_default(),
        );
        flag(
            "MUD_TRAIN_SEQ_LEN",
            &std::env::var("MUD_TRAIN_SEQ_LEN").unwrap_or_default(),
        );
    }

    // ── Summary ────────────────────────────────────────────────────────────
    println!("\n══════════════════════════════════════════════════════════");
    if critical == 0 {
        println!("  🟢 CERTIFIED — {warnings} warning(s), 0 critical");
        println!("  Ledger: L-01…L-15 foundations present in tree");
        println!("══════════════════════════════════════════════════════════");
        ExitCode::SUCCESS
    } else {
        println!("  🔴 FAILED — {critical} critical, {warnings} warning(s)");
        println!("══════════════════════════════════════════════════════════");
        ExitCode::from(2)
    }
}

fn print_section(n: &str, title: &str) {
    println!("\n--- [{n}] {title} ---");
}
fn ok(msg: &str) {
    println!("  ✅ {msg}");
}
fn warn(msg: &str, w: &mut u32) {
    println!("  ⚠️  {msg}");
    *w += 1;
}
fn fail(msg: &str, c: &mut u32) {
    println!("  ❌ {msg}");
    *c += 1;
}
fn flag(name: &str, val: &str) {
    let display = if val.is_empty() { "(unset)" } else { val };
    println!("  · {name}={display}");
}
