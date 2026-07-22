//! Structural healthcheck for `.mud` models before QAT.
//! Reports LIVE compute stack facts + planned optimizer strategies (not claimed as active).
//! P-13/L-12: dims via `forge_llm::mud::p13` (no silent defaults).
//! See: docs/architecture/MUD_COMPUTE_STACK.md

use forge_llm::mud::inference::{cmud_audit, model_logits_collapsed};
use forge_llm::mud::p13::{health_constants_ok, parse_arch_dims};
use forge_llm::mud::slime_backward::{select_optimizer, OptimizerStrategy};
use forge_llm::mud::{MudFile, MudTensorType};

fn strategy_name(s: OptimizerStrategy) -> &'static str {
    match s {
        OptimizerStrategy::Sgd => "SGD",
        OptimizerStrategy::Muon { .. } => "Muon (Newton-Schulz)",
        OptimizerStrategy::GaLore { .. } => "GaLore",
        OptimizerStrategy::ChunkedAdam { .. } => "ChunkedAdam",
        OptimizerStrategy::SparseAdam { .. } => "SparseAdam",
        OptimizerStrategy::Adam => "Adam",
    }
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2.mud".to_string());
    println!("=== MUD Training Certification & Healthcheck ===");
    println!("Loading: {}", model_path);
    println!("Docs: docs/architecture/MUD_COMPUTE_STACK.md | ledger: GEMINI.md\n");

    if !health_constants_ok() {
        anyhow::bail!("L-12: constants health check failed (EPSILON / PCore)");
    }
    println!("--- [0] L-12 Constants SSOT ---");
    println!("  EPSILON_FLOOR / PCorePool policy: OK");

    let mud_file = MudFile::load(&model_path)?;
    let dims = parse_arch_dims(&mud_file)?;
    let hidden_size = dims.hidden_size;
    let num_layers = dims.num_layers;
    let num_heads = dims.num_heads;
    let num_kv_heads = dims.num_kv_heads;
    let intermediate_size = dims.intermediate_size;

    println!("\n--- [1] Architectural Analysis (P-13) ---");
    println!("  Hidden Size: {}", hidden_size);
    println!("  Layers     : {}", num_layers);
    println!("  Heads (Q)  : {}", num_heads);
    println!("  Heads (KV) : {}", num_kv_heads);
    println!("  FFN Size   : {}", intermediate_size);
    println!("  Head dim   : {}", dims.head_dim());

    println!("\n--- [1b] LIVE Compute Stack (engine truth) ---");
    println!("  Weights    : Ternary2Bit ELUT 4-bit (8 w / u32) + PRQ f32 scales");
    println!("  Accumul.   : FP32 (SlimeRegister.matmul_accum)");
    println!("  Forward GEMV: AVX2 ASM × PCorePool (QKV parallel); GPU tiled ash via gemv_policy");
    println!(
        "  MUD_GPU_GEMV : {} (0=off 1=on auto/unset=profiled break-even; MUD_GPU_GEMV_MIN override)",
        std::env::var("MUD_GPU_GEMV").unwrap_or_else(|_| "auto(default)".into())
    );
    println!(
        "  GEMV policy  : {}",
        forge_llm::vulkan::gemv_policy::policy_summary()
    );
    println!(
        "  QAT step   : LIVE = Muon/GaLore/Chunked→SGD; Adam/SparseAdam→moments (adam_step_avx2)"
    );
    println!("  Optimizer  : L-01+P0 Adam moments; L-02 Muon NS → ash when MUD_USE_VULKAN=1");
    println!(
        "  Packing    : L-10 + Stream D full-seq={} seq_len={} (pairs@pos0 if MUD_TRAIN_FULL_SEQ=0)",
        forge_llm::mud::sequence_pack::train_full_seq_enabled(),
        forge_llm::mud::sequence_pack::train_seq_len()
    );
    println!("  Mini MoE   : L-11 ExpertBus API (dense default)");
    println!(
        "  Context    : L-13 HCA ring + Stream E CSA ({})",
        forge_llm::mud::csa_indexer::CsaPolicy::resolve().summary()
    );
    println!("  C-MUD      : L-14 research kernel (MUD_CMUD_THINK=1 opt-in; f32 path default)");
    println!("  GradCkpt   : L-15 (MUD_GRAD_CKPT=1 recompute-on-reverse)");
    let vk = std::env::var("MUD_USE_VULKAN").unwrap_or_default() == "1";
    println!(
        "  MUD_USE_VULKAN: {} (NS GPU path {})",
        if vk { "1" } else { "0/unset" },
        if vk { "enabled" } else { "disabled" }
    );

    let core = mud_file.skills.get("core").expect("No core skill");
    let mut safe_to_train = true;

    println!("\n--- [2] Tensor Shape Validation ---");
    let expected_q_shape = vec![hidden_size, hidden_size];
    let expected_kv_shape = [hidden_size * num_kv_heads / num_heads.max(1), hidden_size];
    let expected_up_shape = [intermediate_size, hidden_size];
    let expected_down_shape = [hidden_size, intermediate_size];

    let mut tensors_checked = 0;
    let mut collapsed_tensors = 0;

    for blk in 0..num_layers {
        let p = format!("blk.{}.", blk);
        let q_name = format!("{}attn_q.weight", p);

        if let Some(t) = core.tensors.get(&q_name) {
            tensors_checked += 1;
            if t.shape != expected_q_shape {
                println!(
                    "  ❌ SHAPE MISMATCH: {} has shape {:?}, expected {:?}",
                    q_name, t.shape, expected_q_shape
                );
                safe_to_train = false;
            }
        } else {
            println!("  ❌ MISSING TENSOR: {}", q_name);
            safe_to_train = false;
        }

        // ELUT 4-bit collapse check only (2-bit legacy removed from live path)
        let check_collapse = |name: &str| {
            if let Some(t) = core.tensors.get(name) {
                if t.t_type == MudTensorType::Ternary2Bit {
                    let total = t.shape.iter().product::<usize>();
                    let samples = total.min(8192);
                    let stride = if total > samples { total / samples } else { 1 };
                    let mut nonzero = 0;
                    unsafe {
                        for i in 0..samples {
                            let offset = i * stride;
                            let u32_idx = offset / 8;
                            let shift = (offset % 8) * 4;
                            let val = (*(t.data_ptr as *const u32).add(u32_idx) >> shift) & 0xF;
                            if val == 0x1 || val == 0xF {
                                nonzero += 1;
                            }
                        }
                    }
                    if nonzero == 0 && samples > 0 {
                        return true;
                    }
                }
            }
            false
        };

        if check_collapse(&q_name) {
            println!("  ❌ COLLAPSE DETECTED: {} is entirely zeroes!", q_name);
            collapsed_tensors += 1;
            safe_to_train = false;
        }
    }

    println!(
        "  ✅ Validated shapes/integrity sample of {} Q matrices.",
        tensors_checked
    );
    if collapsed_tensors > 0 {
        println!(
            "  ❌ {} tensors collapsed to all-zero ELUT codes.",
            collapsed_tensors
        );
    }

    println!("\n--- [3] LIVE Optimizer Strategy (select_optimizer → step, L-01) ---");
    let recommend = |rows: usize, cols: usize, name: &str| {
        let s = select_optimizer(rows, cols);
        println!(
            "  [{}] {}×{} → {}  [LIVE at apply_optimizer_cpu_step_and_pack]",
            name,
            rows,
            cols,
            strategy_name(s)
        );
    };
    recommend(expected_q_shape[0], expected_q_shape[1], "attn_q.weight");
    recommend(expected_kv_shape[0], expected_kv_shape[1], "attn_k.weight");
    recommend(
        expected_up_shape[0],
        expected_up_shape[1],
        "ffn_up / expert.w1",
    );
    recommend(
        expected_down_shape[0],
        expected_down_shape[1],
        "ffn_down / expert.w2",
    );
    recommend(49152, hidden_size, "token_embd (example tall)");

    // ── [3b] Logit collapse gate (T0.3) ──────────────────────────────────
    println!("\n--- [3b] Logit Collapse Gate ---");
    let collapsed = model_logits_collapsed(&mud_file);
    if collapsed {
        println!("  ❌ model_logits_collapsed = TRUE (token-0 dominates across prompts)");
        safe_to_train = false;
    } else {
        println!("  ✅ logits not collapsed (no token-0 dominance across prompts)");
    }

    // ── [3c] C-MUD complex-reasoning audit (research §3, new work) ───────
    println!("\n--- [3c] C-MUD Complex Reasoning Audit (L-14) ---");
    let cka = cmud_audit(&mud_file);
    for line in cka.summary_lines() {
        println!("{line}");
    }
    if cka.healthy() {
        println!(
            "  ✅ C-MUD reasoning healthy (τ={}, phase_lock={}, ball_ok)",
            cka.think.steps, cka.think.phase_locked
        );
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
        println!("  ⚠️  C-MUD reasoning unhealthy: {}", reasons.join(", "));
        // Warning, not hard-fail: C-MUD is an opt-in research path.
        println!("      (non-fatal: MUD_CMUD_THINK=1 is off by default in production)");
    }

    println!("\n--- [4] Certification Result ---");
    if safe_to_train {
        println!("  🟢 CERTIFIED");
        println!("    structural shapes : OK");
        println!("    ELUT sample       : OK (no all-zero collapse)");
        println!("    logit collapse    : {}", if collapsed { "FAIL" } else { "OK" });
        println!(
            "    C-MUD reasoning    : {}",
            if cka.healthy() { "OK (validated)" } else { "WARN (opt-in)" }
        );
        println!("  Forward: ELUT×FP32 AVX2 / PCorePool(8). Train: L-01 strategies + L-02 Muon NS (GPU if MUD_USE_VULKAN=1).");
    } else {
        println!("  🔴 FAILED: structural or zero-collapse issues.");
        println!("  Do NOT train on this checkpoint. Restore from a clean base model.");
        std::process::exit(1);
    }

    Ok(())
}
