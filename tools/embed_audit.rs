fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("models/core_skills.mud");
    println!("🔍 Embedding Audit: {}", path);

    let mf = forge_llm::mud::MudFile::load(path)?;
    let core = mf.skills.get("core").unwrap();
    let tensor = core.tensors.get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;

    let vocab = tensor.shape[0];
    let hidden = tensor.shape[1];
    let total = vocab * hidden;
    println!("  Vocab: {}, Hidden: {}, Total params: {}", vocab, hidden, total);

    let data = unsafe {
        std::slice::from_raw_parts(tensor.data_ptr as *const f32, total)
    };

    // Global stats
    let global_mean = data.iter().sum::<f32>() / total as f32;
    let global_var = data.iter().map(|v| (v - global_mean).powi(2)).sum::<f32>() / total as f32;
    let global_std = global_var.sqrt();
    let global_absmean = data.iter().map(|v| v.abs()).sum::<f32>() / total as f32;
    let global_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let global_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!();
    println!("=== GLOBAL STATS ===");
    println!("  Mean:    {:.6}", global_mean);
    println!("  StdDev:  {:.6}", global_std);
    println!("  AbsMean: {:.6}", global_absmean);
    println!("  Min:     {:.6}", global_min);
    println!("  Max:     {:.6}", global_max);

    // Per-row stats
    let mut row_absmeans = Vec::with_capacity(vocab);
    let mut row_means = Vec::with_capacity(vocab);
    let mut row_stds = Vec::with_capacity(vocab);
    let mut cos_sims = Vec::with_capacity(vocab);
    let mut cos_sims_global = Vec::with_capacity(vocab);
    let mut mses = Vec::with_capacity(vocab);
    let mut zero_fracs = Vec::with_capacity(vocab);
    let mut positives = 0u64;
    let mut negatives = 0u64;
    let mut zero_tern = 0u64;

    let max_rows = vocab.min(49152);

    for row_i in 0..max_rows {
        let start = row_i * hidden;
        let row = &data[start..start + hidden];

        let row_sum: f32 = row.iter().sum();
        let row_mean = row_sum / hidden as f32;
        let row_absmean = row.iter().map(|v| v.abs()).sum::<f32>() / hidden as f32;

        let row_var = row.iter().map(|v| (v - row_mean).powi(2)).sum::<f32>() / hidden as f32;
        row_means.push(row_mean);
        row_absmeans.push(row_absmean);
        row_stds.push(row_var.sqrt());

        // Row-wise ternary
        let scale = row_absmean.max(1e-10);
        let mut ternary = Vec::with_capacity(hidden);
        let mut recon = Vec::with_capacity(hidden);
        let mut dot = 0.0f32;
        let mut norm_t = 0.0f32;
        let mut se = 0.0f32;
        let mut pos = 0u64; let mut neg = 0u64; let mut zer = 0u64;

        for &v in row {
            let q = (v / scale).round().clamp(-1.0, 1.0) as i8;
            let r = q as f32 * scale;
            dot += v * r;
            norm_t += r * r;
            se += (v - r).powi(2);
            ternary.push(q);
            recon.push(r);
            match q { 1 => pos += 1, -1 => neg += 1, _ => zer += 1 }
        }

        positives += pos; negatives += neg; zero_tern += zer;
        let rn = norm_t.sqrt();
        let row_norm = row.iter().map(|v| v.powi(2)).sum::<f32>().sqrt();
        let cos = if rn > 0.0 && row_norm > 0.0 { dot / (rn * row_norm) } else { 1.0 };
        cos_sims.push(cos);
        mses.push(se / hidden as f32);
        zero_fracs.push(zer as f32 / hidden as f32);

        // Global-scale ternary
        let gscale = global_absmean.max(1e-10);
        let mut gdot = 0.0f32; let mut gn = 0.0f32;
        for &v in row {
            let q = (v / gscale).round().clamp(-1.0, 1.0);
            let r = q * gscale;
            gdot += v * r; gn += r * r;
        }
        let gcos = if gn.sqrt() > 0.0 && row_norm > 0.0 { gdot / (gn.sqrt() * row_norm) } else { 1.0 };
        cos_sims_global.push(gcos);
    }

    let mean_cos = cos_sims.iter().sum::<f32>() / max_rows as f32;
    let median_cos = {
        let mut sorted = cos_sims.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[max_rows / 2]
    };
    let min_cos = cos_sims.iter().cloned().fold(f32::INFINITY, f32::min);

    let mean_gcos = cos_sims_global.iter().sum::<f32>() / max_rows as f32;
    let mean_mse = mses.iter().sum::<f32>() / max_rows as f32;

    let below_09 = cos_sims.iter().filter(|&&c| c < 0.9).count();
    let below_095 = cos_sims.iter().filter(|&&c| c < 0.95).count();
    let below_099 = cos_sims.iter().filter(|&&c| c < 0.99).count();
    let above_097 = cos_sims.iter().filter(|&&c| c >= 0.97).count();

    let mean_absmean = row_absmeans.iter().sum::<f32>() / max_rows as f32;
    let min_absmean = row_absmeans.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_absmean = row_absmeans.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mean_stddev = row_stds.iter().sum::<f32>() / max_rows as f32;

    println!();
    println!("=== PER-ROW STATS ===");
    println!("  Row AbsMean: min={:.6}  max={:.6}  mean={:.6}", min_absmean, max_absmean, mean_absmean);
    println!("  Row StdDev:  mean={:.6}", mean_stddev);

    println!();
    println!("=== ROW-WISE TERNARY SIMULATION ===");
    println!("  Cosine sim:  mean={:.6}  median={:.6}  min={:.6}", mean_cos, median_cos, min_cos);
    println!("  MSE:         mean={:.8}", mean_mse);
    println!("  Cos < 0.90:  {}/{} rows", below_09, max_rows);
    println!("  Cos < 0.95:  {}/{} rows", below_095, max_rows);
    println!("  Cos < 0.99:  {}/{} rows", below_099, max_rows);
    println!("  Cos ≥ 0.97:  {}/{} rows", above_097, max_rows);

    println!();
    println!("=== GLOBAL-SCALE TERNARY ===");
    println!("  Cosine sim mean: {:.6}", mean_gcos);

    let total_par = total as f64;
    let pct_pos = positives as f64 / total_par * 100.0;
    let pct_neg = negatives as f64 / total_par * 100.0;
    let pct_zer = zero_tern as f64 / total_par * 100.0;
    println!();
    println!("=== TERNARY DISTRIBUTION ===");
    println!("  +1: {:.2}%   0: {:.2}%   -1: {:.2}%", pct_pos, pct_zer, pct_neg);
    println!("  Asymmetry (+/-): {:.2}%", (pct_pos - pct_neg).abs());

    println!();
    println!("=== COMPRESSION ===");
    let orig_fp32 = total as f64 * 4.0;
    let orig_fp16 = total as f64 * 2.0;
    let ternary_bits = total as f64 * 2.0 / 8.0; // 2-bit ternary
    let scale_bytes = if hidden < 256 { vocab as f64 } else { vocab as f64 * 2.0 }; // u8 if dim<256 else f16
    let total_mud = ternary_bits + scale_bytes;

    println!("  Original FP32:  {:.1} MB", orig_fp32 / 1_048_576.0);
    println!("  Original FP16:  {:.1} MB", orig_fp16 / 1_048_576.0);
    println!("  Row-wise Ternary: {:.2} MB", total_mud / 1_048_576.0);
    println!("    └─ data:  {:.2} MB", ternary_bits / 1_048_576.0);
    println!("    └─ scales: {:.2} KB", scale_bytes / 1024.0);
    println!("  Ratio vs FP16: {:.1}x", orig_fp16 / total_mud);
    println!("  Ratio vs FP32: {:.1}x", orig_fp32 / total_mud);

    // If MUD currently stores as FP32
    let current_mud = total as f64 * 4.0;
    println!();
    println!("=== CURRENT MUD (FP32) VS TERNARY ===");
    println!("  Current in .mud: {:.1} MB", current_mud / 1_048_576.0);
    println!("  Ternary (row-wise):  {:.2} MB", total_mud / 1_048_576.0);
    println!("  Savings: {:.1}x", current_mud / total_mud);

    Ok(())
}
