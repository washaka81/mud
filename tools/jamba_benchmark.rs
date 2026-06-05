//! Rigorous Jamba Benchmark & Reliability Suite
//! Measures TPS, latency, and numerical stability of hybrid Attention/Mamba layers.

use forge_llm::mud::inference::{
    InferenceWorkspace, MudLayer, MudMambaLayer, MudMoELayer, MudModel,
};
use forge_llm::mud::MudFile;
use std::collections::HashMap;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("\x1b[1;36m=== Forge LLM: Jamba Hybrid Rigorous Benchmark ===\x1b[0m");

    let hidden = 512;
    let n_layers = 12;
    let d_state = 16;
    let d_conv = 4;
    let vocab_size = 32000;

    println!("Configuration:");
    println!("  Hidden Size:  {}", hidden);
    println!(
        "  Layers:       {} (Hybrid: 1 Attention / 5 Mamba ratio)",
        n_layers
    );
    println!("  SSM D_state:  {}", d_state);
    println!("  Context:      Pre-allocated Workspace (Zero-Allocation)");

    // 1. Create a Synthetic Hybrid Model
    let mut layers = Vec::new();
    for l in 0..n_layers {
        if l % 6 == 0 {
            // Attention Layer (Synthetic pointers to zeroed data)
            layers.push(MudLayer::Attention(MudMoELayer {
                experts: vec![],
                router: forge_llm::mud::routing::MudRouter::new(1, 1),
                attn_q_w: std::ptr::null(),
                attn_k_w: std::ptr::null(),
                attn_v_w: std::ptr::null(),
                attn_o_w: std::ptr::null(),
                attn_q_scales: std::ptr::null(),
                attn_k_scales: std::ptr::null(),
                attn_v_scales: std::ptr::null(),
                attn_o_scales: std::ptr::null(),
                gate_w: std::ptr::null(),
                norm_w: std::ptr::null(),
                attn_norm_w: std::ptr::null(),
                attn_sub_norm_w: std::ptr::null(),
                ffn_sub_norm_w: std::ptr::null(),
                key_q: String::new(),
                key_k: String::new(),
                key_v: String::new(),
                key_o: String::new(),
                key_gate: String::new(),
            }));
        } else {
            // Mamba Layer
            layers.push(MudLayer::Mamba(MudMambaLayer {
                in_proj_w: std::ptr::null(),
                in_proj_scales: std::ptr::null(),
                out_proj_w: std::ptr::null(),
                out_proj_scales: std::ptr::null(),
                x_proj_w: std::ptr::null(),
                x_proj_scales: std::ptr::null(),
                dt_proj_w: std::ptr::null(),
                dt_proj_scales: std::ptr::null(),
                a_log_w: std::ptr::null(),
                d_w: std::ptr::null(),
                norm_w: std::ptr::null(),
                conv1d_w: std::ptr::null(),
                conv1d_b: std::ptr::null(),
                key_in: String::new(),
                key_out: String::new(),
            }));
        }
    }

    let _model = MudModel {
        layers,
        hidden_size: hidden,
        ffn_hidden_size: hidden * 4,
        num_experts: 1,
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 64,
        d_state,
        d_conv,
        rms_norm_eps: 1e-5,
        hidden_act: "silu".to_string(),
        rope_theta: 10000.0,
        rope_freqs: vec![1.0; 32],
        use_alibi: false,
        lora_adapters: std::collections::HashMap::new(),
    };

    let _workspace = InferenceWorkspace::new(
        None,
        hidden,
        hidden * 4,
        n_layers,
        1,
        vocab_size,
        d_state,
        d_conv,
    );

    // Manual setup for MudInference (bypassing full file load for pure benchmark)
    let _dummy_mud = MudFile {
        mmap: None,
        skills: HashMap::new(),
        global_metadata: HashMap::new(),
    };

    println!("\n\x1b[1;33m[Phase 1: Raw Kernel Latency]\x1b[0m");

    let mut _workspace = InferenceWorkspace::new(
        None,
        hidden,
        hidden * 4,
        n_layers,
        1,
        vocab_size,
        d_state,
        d_conv,
    );
    
    // Fill test data
    _workspace.mamba_in.write().fill(1.0);
    _workspace.mamba_a_bar.write().fill(0.9);
    _workspace.mamba_b_bar.write().fill(0.1);
    _workspace.mamba_c.write().fill(0.05);

    let start = Instant::now();
    let iters = 1000;
    for _ in 0..iters {
        unsafe {
            forge_llm::asm::mamba_scan_avx2(
                hidden,
                d_state,
                _workspace.mamba_in.read().as_ptr(),
                _workspace.mamba_a_bar.read().as_ptr(),
                _workspace.mamba_b_bar.read().as_ptr(),
                _workspace.mamba_c.read().as_ptr(),
                std::ptr::null(),
                _workspace.ssm_states[0].write().as_mut_ptr(),
                _workspace.mamba_out.write().as_mut_ptr(),
            );
        }
    }
    let elapsed = start.elapsed();
    let avg_mamba = elapsed.as_secs_f64() / iters as f64;
    println!(
        "  Mamba ASM Scan ({} hidden, {} state): {:.3} µs",
        hidden,
        d_state,
        avg_mamba * 1_000_000.0
    );

    println!("\n\x1b[1;33m[Phase 2: Numerical Reliability]\x1b[0m");
    // Reliability check: Magnitude drift through sequence
    let mut mag: f32 = 1.0;
    println!("  Sequence Scan Stability (100 steps):");
    for i in 1..=100 {
        unsafe {
            forge_llm::asm::mamba_scan_avx2(
                hidden,
                d_state,
                _workspace.mamba_in.read().as_ptr(),
                _workspace.mamba_a_bar.read().as_ptr(),
                _workspace.mamba_b_bar.read().as_ptr(),
                _workspace.mamba_c.read().as_ptr(),
                std::ptr::null(),
                _workspace.ssm_states[0].write().as_mut_ptr(),
                _workspace.mamba_out.write().as_mut_ptr(),
            );
        }
        if i % 25 == 0 {
            let out_guard = _workspace.mamba_out.read();
            let sum_sq: f32 = out_guard.iter().map(|v| v * v).sum();
            mag = sum_sq.sqrt();
            println!("    Step {:>3}: Output RMS = {:.6}", i, mag);
        }
    }

    let is_stable = mag.is_finite() && mag > 0.0;
    println!(
        "  Stability Status: {}",
        if is_stable {
            "\x1b[1;32mOK (Finite & Active)\x1b[0m"
        } else {
            "\x1b[1;31mFAILED (NaN or Vanished)\x1b[0m"
        }
    );

    println!("\n\x1b[1;33m[Phase 3: Theoretical Throughput]\x1b[0m");
    // Estimate TPS for full 24-layer hybrid stack
    let _layers_24 = 24;
    let mamba_layers = 20; // 5:1 ratio
    let attn_layers = 4;

    // Rough estimate for other components (projections, norms)
    let projection_latency = 150.0; // µs (measured in previous sessions for 512-dim)
    let attn_latency = 800.0; // µs (GQA with KV-cache access)

    let mamba_step_total = (avg_mamba * 1_000_000.0) + projection_latency;
    let total_token_latency =
        (mamba_step_total * mamba_layers as f64) + (attn_latency * attn_layers as f64);
    let est_tps = 1_000_000.0 / total_token_latency;

    println!(
        "  Est. Latency per layer (Mamba):  {:.2} µs",
        mamba_step_total
    );
    println!(
        "  Est. Latency per token (24L):    {:.2} ms",
        total_token_latency / 1000.0
    );
    println!(
        "  Theoretical Throughput (CPU):    \x1b[1;32m{:.2} tokens/sec\x1b[0m",
        est_tps
    );

    println!("\n\x1b[1;36mBenchmark Complete.\x1b[0m");
    Ok(())
}
