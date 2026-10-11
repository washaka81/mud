//! Check PRQ/ELUT scale stats + mHC tensor presence for a .mud model.
//!
//! ```bash
//! ./mud.sh scale-audit models/ale.mud
//! ```
use forge_llm::mud::MudFile;

fn main() {
    let path = std::env::args().nth(1).expect("usage: check_scales <model.mud>");
    let mud = MudFile::load(&path).expect("load mud");
    let core = mud.skills.get("core").expect("core skill");

    // ── mHC tensor presence check ─────────────────────────────────
    // FIX (2026-07-30): mHC tensors are stored as `blk.N.mhc_alpha` (NOT
    // `blk.N.mhc_alpha.weight`). The `tn` closure that appends `.weight`
    // silently returned NULL → unbounded residual → training overflow.
    // This tool verifies the exact tensor names exist.
    println!("=== mHC tensor check (blk.0) ===");
    let mhc_alpha_key = "blk.0.mhc_alpha";
    let mhc_beta_key = "blk.0.mhc_beta";
    let mhc_radius_key = "blk.0.mhc_radius";
    let mhc_alpha_dot_key = "blk.0.mhc_alpha.weight";

    println!("  blk.0.mhc_alpha         exists: {}", core.tensors.contains_key(mhc_alpha_key));
    println!("  blk.0.mhc_beta          exists: {}", core.tensors.contains_key(mhc_beta_key));
    println!("  blk.0.mhc_radius        exists: {}", core.tensors.contains_key(mhc_radius_key));
    println!("  blk.0.mhc_alpha.weight  exists: {} (should be FALSE)", core.tensors.contains_key(mhc_alpha_dot_key));

    if let Some(t) = core.tensors.get(mhc_alpha_key) {
        unsafe {
            let p = t.data_ptr as *const f32;
            let n = t.shape.iter().product::<usize>().max(1);
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            let mut mean = 0.0;
            for i in 0..n.min(4096) {
                let v = *p.add(i);
                if v.is_finite() {
                    min = min.min(v);
                    max = max.max(v);
                    mean += v;
                }
            }
            mean /= n.min(4096) as f32;
            println!("  mhc_alpha values: min={min:.6} max={max:.6} mean={mean:.6} (n={n})");
        }
    }

    // ── PRQ scale stats ──────────────────────────────────────────
    println!("\n=== PRQ scale stats ===");
    let mut tensor_scales: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, t) in &core.tensors {
        if name.ends_with(".prq_scale") && t.t_type == forge_llm::mud::MudTensorType::Float32 {
            unsafe {
                let p = t.data_ptr as *const f32;
                let n = t.shape.iter().product::<usize>().max(1);
                let vals: Vec<f32> = (0..n).map(|i| *p.add(i)).collect();
                tensor_scales.push((name.clone(), vals));
            }
        }
    }

    let mut mean_of_means = 0.0f32;
    let mut count = 0;
    for (name, vals) in &tensor_scales {
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
        let count_outliers = vals.iter().filter(|v| v.abs() > 10.0).count();
        println!(
            "  {}: min={:.6} max={:.6} mean={:.6} count={} outliers(>10)={}",
            name, min, max, mean, vals.len(), count_outliers
        );
        mean_of_means += mean;
        count += 1;
    }
    if count > 0 {
        mean_of_means /= count as f32;
        println!("\n  Scale mean-of-means: {:.6} (healthy if < 2.0)", mean_of_means);
    }

    // ── F32 weight magnitudes ────────────────────────────────────
    println!("\n=== F32 weight tensor stats (max abs) ===");
    for (name, t) in &core.tensors {
        if t.t_type == forge_llm::mud::MudTensorType::Float32 && !name.ends_with(".prq_scale") {
            unsafe {
                let p = t.data_ptr as *const f32;
                let n = t.shape.iter().product::<usize>().max(1);
                let mut mx = 0.0f32;
                for i in 0..n.min(4096) {
                    let v = (*p.add(i)).abs();
                    if v > mx {
                        mx = v;
                    }
                }
                println!("  {}: max_abs={:.6} (n={})", name, mx, n);
            }
        }
    }
}
