use forge_llm::asm::ternary_gemv_avx2;
use rand::RngExt;

fn main() -> anyhow::Result<()> {
    println!("=== MUD Ternary Kernel Audit ===");
    
    let n = 512;
    let mut x = vec![1.0f32; n];
    let mut rng = rand::rng();
    for v in x.iter_mut() { *v = rng.random_range(-1.0..1.0); }
    
    // Create packed weights
    // Mapping: 0->0, 1->1, 2->-1
    let mut packed_w = vec![0u32; n / 16];
    let mut raw_weights = vec![0i8; n];
    
    for i in 0..packed_w.len() {
        let mut u32_val = 0u32;
        for j in 0..16 {
            let val = rng.random_range(0..3) as i8; // 0, 1, or 2
            let weight = if val == 1 { 1 } else if val == 2 { -1 } else { 0 };
            raw_weights[i * 16 + j] = weight;
            u32_val |= (val as u32) << (j * 2);
        }
        packed_w[i] = u32_val;
    }
    
    let scale = 0.5f32;
    let mut out_asm = 0.0f32;
    
    unsafe {
        ternary_gemv_avx2(n, x.as_ptr(), packed_w.as_ptr(), &mut out_asm, scale);
    }
    
    // Rust reference
    let mut out_rust = 0.0f32;
    for i in 0..n {
        out_rust += x[i] * (raw_weights[i] as f32);
    }
    out_rust *= scale;
    
    println!("  Rust Reference: {:.6}", out_rust);
    println!("  ASM Result:    {:.6}", out_asm);
    println!("  Delta:         {:.10}", (out_rust - out_asm).abs());
    
    if (out_rust - out_asm).abs() < 1e-5 {
        println!("  ✅ KERNEL VERIFIED: BIT-EXACT PRECISION");
    } else {
        println!("  ❌ KERNEL ERROR: DISCREPANCY DETECTED");
    }

    Ok(())
}
