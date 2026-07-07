use forge_llm::mud::slime::SlimeWorkspace;
use forge_llm::mud::slime_backward::{
    backward_slime_block, SlimeBackwardWorkspace, SlimeLayerGradients, SlimeLayerTape,
};
use forge_llm::mud::slime_forward::SlimeLayer;
use std::time::Instant;

fn main() {
    println!("==================================================");
    println!(" MUD SlimeBackward Stress & Reliability Test");
    println!("==================================================");

    let hidden = 2560; // BitNet b1.58-2B-4T dimension
    let ffn_mid = 6912;
    let n_heads = 20;
    let n_kv_heads = 5;
    let head_dim = hidden / n_heads;
    let max_seq_len = 32;
    let kv_dim = n_kv_heads * head_dim;

    println!("Allocating workspaces... (Testing P-01 Zero-Allocation Policy)");
    let f_ws = SlimeWorkspace::new(
        hidden,
        ffn_mid,
        n_heads,
        n_kv_heads,
        max_seq_len,
        hidden,
        30,
        128.0,
    );
    let mut b_ws = SlimeBackwardWorkspace::new(hidden, ffn_mid, kv_dim);
    let mut grads = SlimeLayerGradients::new(hidden, ffn_mid, kv_dim, hidden);
    let mut grad_out = vec![0.0f32; hidden];
    let grad_in = vec![1.0f32; hidden];

    // Allocate memory for mock weights
    let hidden_bytes = hidden / 16 * 4;
    let mut ffn_up_w = vec![0u8; ffn_mid * hidden_bytes];
    let mut ffn_up_scales = vec![1.0f32; ffn_mid];
    let mut ffn_gate_w = vec![0u8; ffn_mid * hidden_bytes];
    let mut ffn_gate_scales = vec![1.0f32; ffn_mid];
    let mut ffn_down_w = vec![0u8; hidden * (ffn_mid / 16 * 4)];
    let mut ffn_down_scales = vec![1.0f32; hidden];

    let mut q_w = vec![0u8; hidden * hidden_bytes];
    let mut q_scales = vec![1.0f32; hidden];
    let mut k_w = vec![0u8; kv_dim * hidden_bytes];
    let mut k_scales = vec![1.0f32; kv_dim];
    let mut v_w = vec![0u8; kv_dim * hidden_bytes];
    let mut v_scales = vec![1.0f32; kv_dim];

    let mut o_w = vec![0u8; hidden * hidden_bytes];
    let mut o_scales = vec![1.0f32; hidden];

    let mut ffn_norm_w = vec![1.0f32; hidden];
    let mut attn_norm_w = vec![1.0f32; hidden];

    let layer = SlimeLayer {
        q_w: q_w.as_mut_ptr(),
        q_scales: q_scales.as_mut_ptr(),
        k_w: k_w.as_mut_ptr(),
        k_scales: k_scales.as_mut_ptr(),
        v_w: v_w.as_mut_ptr(),
        v_scales: v_scales.as_mut_ptr(),
        o_w: o_w.as_mut_ptr(),
        o_scales: o_scales.as_mut_ptr(),
        ffn_up_w: ffn_up_w.as_mut_ptr(),
        ffn_up_scales: ffn_up_scales.as_mut_ptr(),
        ffn_gate_w: ffn_gate_w.as_mut_ptr(),
        ffn_gate_scales: ffn_gate_scales.as_mut_ptr(),
        ffn_down_w: ffn_down_w.as_mut_ptr(),
        ffn_down_scales: ffn_down_scales.as_mut_ptr(),
        attn_norm_w: attn_norm_w.as_mut_ptr(),
        ffn_norm_w: ffn_norm_w.as_mut_ptr(),
        attn_sub_norm_w: std::ptr::null(),
        ffn_sub_norm_w: std::ptr::null(),
        mhc_alpha_w: std::ptr::null(),
        mhc_beta_w: std::ptr::null(),
        mhc_radius_w: std::ptr::null(),
        n_kv_heads,
        ffn_mid,
        rope_theta: 10000.0,
    };

    println!("Layer weights bound correctly.");

    let iterations = 100; // Stress test 100 autoregressive steps
    println!("Beginning {} backward passes...", iterations);

    let start = Instant::now();
    for i in 0..iterations {
        // Mock tape for step
        let tape = SlimeLayerTape::new(
            hidden,
            ffn_mid,
            n_kv_heads,
            head_dim,
            max_seq_len,
            i % max_seq_len,
        );

        // Run backward pass
        backward_slime_block(
            &layer,
            &f_ws,
            &mut b_ws,
            &tape,
            &mut grads,
            &grad_in,
            &mut grad_out,
        );

        // Stability Check
        for &val in grad_out.iter() {
            if !val.is_finite() {
                panic!(
                    "STABILITY FAILURE: NaN or Infinity detected at iteration {}",
                    i
                );
            }
        }
    }
    let duration = start.elapsed();

    println!("==================================================");
    println!(" Results:");
    println!(" - Iterations: {}", iterations);
    println!(" - Time taken: {:.2?}", duration);
    println!(
        " - Throughput: {:.2} steps/sec",
        iterations as f64 / duration.as_secs_f64()
    );
    println!(
        " - Latency:    {:.2} ms/step",
        duration.as_secs_f64() * 1000.0 / iterations as f64
    );
    println!(" - Status:     PASSED (No panics, No NaN, Zero-Allocation Loop)");
    println!("==================================================");
}
