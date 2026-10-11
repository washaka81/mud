//! SPRINT-3: C-MUD perplexity benchmark.
//!
//! Measures cross-entropy (proxy for perplexity) on a small token corpus with
//! and without `MUD_CMUD_THINK=1`. Prints a CSV-like table comparing the two
//! runs. Pass criterion: `ppl_relative ≤ 1.02 && !spectral.collapsed`.
//!
//! Usage:
//!   cargo run --release --bin cmud_perplexity -- models/smollm2.mud corpus.txt
//!   cargo run --release --bin cmud_perplexity -- models/smollm2.mud corpus.txt --window 32 --max-seqs 64
//!
//! `corpus.txt` format: one token sequence per line, space-separated `u32` ids.
//! If the file isn't a `u32` corpus, the tool auto-tokenizes it via
//! `Tokenizer::from_mud_metadata` (see `cmud_train` for the same pattern).

use forge_llm::mud::inference::forward_last_logits;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

#[derive(Default, Clone)]
struct RunStats {
    #[allow(dead_code)]
    name: String,
    total_ce: f64,
    n_chunks: usize,
    logit_range_min: f32,
    spectral_collapsed: bool,
    spectral_cauchy_mag_at_2: f32,
    latency_ms_total: f64,
    any_nonfinite: bool,
}

impl RunStats {
    fn mean_ce(&self) -> f32 {
        if self.n_chunks == 0 {
            f32::NAN
        } else {
            (self.total_ce / self.n_chunks as f64) as f32
        }
    }
    fn ppl(&self) -> f32 {
        self.mean_ce().exp()
    }
    fn latency_ms_per_chunk(&self) -> f64 {
        if self.n_chunks == 0 {
            0.0
        } else {
            self.latency_ms_total / self.n_chunks as f64
        }
    }
    fn healthy(&self) -> bool {
        self.n_chunks > 0 && !self.any_nonfinite && !self.spectral_collapsed
    }
}

fn parse_args() -> (String, String, usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "models/smollm2.mud".to_string());
    let mut tokens = "corpus.txt".to_string();
    let mut window = 32usize;
    let mut max_seqs = 64usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" => {
                if let Some(v) = args.get(i + 1) {
                    tokens = v.clone();
                }
            }
            "--window" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    window = v;
                }
            }
            "--max-seqs" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_seqs = v;
                }
            }
            _ => {}
        }
        i += 1;
    }
    (model, tokens, window, max_seqs)
}

fn load_seqs(mud: &MudFile, tokens_path: &str, window: usize, max_seqs: usize, vocab_size: u32) -> Vec<Vec<u32>> {
    let raw = match std::fs::read_to_string(tokens_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[cmud-perplexity] cannot read {tokens_path}: {e}");
            return Vec::new();
        }
    };

    // Try u32 corpus first.
    let mut seqs: Vec<Vec<u32>> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let ids: Vec<u32> = l
                .split_whitespace()
                .filter_map(|w| w.parse::<u32>().ok())
                .filter(|&id| id < vocab_size)
                .collect();
            if ids.len() >= window { Some(ids) } else { None }
        })
        .collect();

    // Fallback: auto-tokenize the raw text.
    if seqs.is_empty() {
        let tokens_str = mud.global_metadata.get("tokenizer.tokens").map(|s| s.as_str()).unwrap_or("");
        let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
        let tk = forge_llm::model::tokenizer::Tokenizer::from_mud_metadata(tokens_str, merges_str);
        let ids: Vec<u32> = tk
            .encode(&raw)
            .into_iter()
            .filter(|&id| id < vocab_size)
            .collect();
        if ids.len() >= window {
            for window_ids in ids.windows(window) {
                seqs.push(window_ids.to_vec());
            }
        }
    }
    if max_seqs > 0 && seqs.len() > max_seqs {
        seqs.truncate(max_seqs);
    }
    seqs
}

fn run_one(
    mud: &MudFile,
    seqs: &[Vec<u32>],
    label: &str,
    think_enabled: bool,
) -> RunStats {
    // Toggle MUD_CMUD_THINK for this run.
    let prev = std::env::var("MUD_CMUD_THINK").ok();
    unsafe {
        if think_enabled {
            std::env::set_var("MUD_CMUD_THINK", "1");
        } else {
            std::env::remove_var("MUD_CMUD_THINK");
        }
    }

    let mut stats = RunStats {
        name: label.to_string(),
        logit_range_min: f32::INFINITY,
        ..Default::default()
    };

    for chunk in seqs {
        let target = *chunk.last().unwrap();
        let context = &chunk[..chunk.len() - 1];
        let t0 = std::time::Instant::now();
        let res = forward_last_logits(mud, context);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        stats.latency_ms_total += elapsed;

        match res {
            Ok(lg) => {
                stats.n_chunks += 1;
                if !lg.iter().all(|x| x.is_finite()) {
                    stats.any_nonfinite = true;
                }
                let max = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min = lg.iter().cloned().fold(f32::INFINITY, f32::min);
                stats.logit_range_min = stats.logit_range_min.min(max - min);

                // Cross-entropy: -log(softmax(target))
                if (target as usize) < lg.len() {
                    // Numerically-stable log-softmax.
                    let max_l = max;
                    let sum_exp: f64 = lg.iter().map(|x| ((*x - max_l) as f64).exp()).sum();
                    let log_z = max_l as f64 + sum_exp.ln();
                    let log_p_target = (lg[target as usize] as f64) - log_z;
                    stats.total_ce += -log_p_target;
                }
            }
            Err(e) => {
                eprintln!("[cmud-perplexity] forward error on chunk: {e}");
                stats.any_nonfinite = true;
            }
        }
    }

    // If CMUD think is on, also run a small audit to populate spectral health.
    if think_enabled {
        let audit = forge_llm::mud::inference::cmud_audit(mud);
        stats.spectral_collapsed = audit.spectral.collapsed;
        stats.spectral_cauchy_mag_at_2 = audit.spectral.cauchy_mag_at_2;
    }

    // Restore env.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("MUD_CMUD_THINK", v),
            None => std::env::remove_var("MUD_CMUD_THINK"),
        }
    }

    stats
}

fn main() -> ExitCode {
    let (model_path, tokens_path, window, max_seqs) = parse_args();

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║  C-MUD PERPLEXITY BENCHMARK  ·  baseline vs think=1                ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!("Model:   {model_path}");
    println!("Corpus:  {tokens_path}  │  window={window}  │  max_seqs={max_seqs}\n");

    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ❌ load failed: {e}");
            return ExitCode::from(1);
        }
    };
    let vocab_size = mud
        .global_metadata
        .get("vocab_size")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(49152);

    let seqs = load_seqs(&mud, &tokens_path, window, max_seqs, vocab_size);
    if seqs.is_empty() {
        eprintln!("  ❌ no usable sequences (need ≥{window} ids per chunk)");
        return ExitCode::from(1);
    }
    println!("  corpus chunks: {}\n", seqs.len());

    println!("  ┌─ Running baseline (MUD_CMUD_THINK unset) ─────────────────────");
    let baseline = run_one(&mud, &seqs, "baseline", false);
    println!("    n_chunks={}  mean_ce={:.4}  ppl={:.2}  latency={:.1} ms/chunk",
        baseline.n_chunks, baseline.mean_ce(), baseline.ppl(), baseline.latency_ms_per_chunk());

    println!("\n  ┌─ Running C-MUD think=1 ─────────────────────────────────────────");
    let cmud = run_one(&mud, &seqs, "cmud_think=1", true);
    println!("    n_chunks={}  mean_ce={:.4}  ppl={:.2}  latency={:.1} ms/chunk",
        cmud.n_chunks, cmud.mean_ce(), cmud.ppl(), cmud.latency_ms_per_chunk());
    println!("    spectral.collapsed={}  cauchy|G(2)|={:.4}",
        cmud.spectral_collapsed, cmud.spectral_cauchy_mag_at_2);

    // Verdict.
    println!("\n  ┌─ Summary ───────────────────────────────────────────────────────");
    let ppl_baseline = baseline.ppl();
    let ppl_cmud = cmud.ppl();
    let ppl_rel = if ppl_baseline > 0.0 { ppl_cmud / ppl_baseline } else { 0.0 };

    println!(
        "    baseline ppl = {:.3}   cmud ppl = {:.3}   ppl_relative = {:.3}",
        ppl_baseline, ppl_cmud, ppl_rel
    );
    println!(
        "    baseline healthy = {}  cmud healthy = {}",
        baseline.healthy(),
        cmud.healthy()
    );

    // Pass criterion: cmud is no worse than baseline +2% AND spectrum is healthy.
    let pass_ppl = ppl_rel <= 1.02;
    let pass_health = cmud.healthy() && baseline.healthy();
    let pass = pass_ppl && pass_health;

    println!();
    if pass {
        println!("  ✅ PASS  (ppl_relative = {ppl_rel:.3} ≤ 1.02  &&  spectral healthy)");
        ExitCode::from(0)
    } else {
        println!("  ❌ FAIL");
        if !pass_ppl {
            println!("     - ppl_relative {ppl_rel:.3} > 1.02");
        }
        if !pass_health {
            println!("     - spectral collapsed or non-finite logits");
        }
        ExitCode::from(1)
    }
}
