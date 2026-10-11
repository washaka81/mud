//! Herramienta continua de coherencia de inferencia.
//!
//! Mide tres métricas de calidad de inferencia en un modelo `.mud`:
//!
//! 1. **Éxito** (success): Top-1 accuracy — la fracción de prompts donde el token
//!    predicho (argmax de logits) coincide con el token esperado.
//!
//! 2. **Desviación** (deviation): Entropía de la distribución softmax de salida.
//!    Alta entropía ⇒ alta incertidumbre / desviación de la predicción.
//!    Baja entropía ⇒ predicción confiada. Se reporta la entropía media y el
//!    rango (máx − mín) como indicador de variabilidad.
//!
//! 3. **Camino por coherencia** (coherence path): Métricas del loop de razonamiento
//!    C-MUD (phase_lock, hermitian norm vs radio, spread de magnitud, cauchy|G(2)|)
//!    que indican si el modelo atraviesa un camino de razonamiento coherente
//!    (manifold no colapsado, fase bloqueada, normas dentro del radio hermitiano).
//!
//! Uso:
//!   cargo run --release --bin inference_coherence -- models/smollm2.mud prompts.txt
//!   cargo run --release --bin inference_coherence -- models/smollm2.mud prompts.txt --think
//!
//! `prompts.txt` formato: una línea por prompt, con el token esperado al final:
//!   <token_ids_espaciados> <token_esperado>
//! Ejemplo:
//!   123 456 789 101112
//!   (el último token ID es el esperado; los anteriores son el contexto)
//!
//! Si el archivo no es u32, se auto-tokeniza con `Tokenizer::from_mud_metadata`.

use forge_llm::mud::inference::{forward_last_logits, forward_last_logits_cmud};
use forge_llm::mud::MudFile;
use std::process::ExitCode;

/// Estadísticas agregadas de coherencia.
#[derive(Default)]
struct CoherenceStats {
    n: usize,
    successes: usize,
    entropy_sum: f64,
    entropy_min: f32,
    entropy_max: f32,
    logit_margin_sum: f64,
    herm_norm_max: f32,
    herm_radius: f32,
    phase_locked_count: usize,
    spectral_collapsed_count: usize,
    spectral_cauchy_sum: f64,
    spectral_spread_sum: f64,
    spectral_phase_r_sum: f64,
    any_nonfinite: bool,
}

impl CoherenceStats {
    fn success_rate(&self) -> f32 {
        if self.n == 0 {
            0.0
        } else {
            self.successes as f32 / self.n as f32
        }
    }
    fn mean_entropy(&self) -> f32 {
        if self.n == 0 {
            f32::NAN
        } else {
            (self.entropy_sum / self.n as f64) as f32
        }
    }
    fn mean_logit_margin(&self) -> f32 {
        if self.n == 0 {
            f32::NAN
        } else {
            (self.logit_margin_sum / self.n as f64) as f32
        }
    }
    fn mean_cauchy_mag(&self) -> f32 {
        if self.n == 0 {
            f32::NAN
        } else {
            (self.spectral_cauchy_sum / self.n as f64) as f32
        }
    }
    fn mean_spread(&self) -> f32 {
        if self.n == 0 {
            f32::NAN
        } else {
            (self.spectral_spread_sum / self.n as f64) as f32
        }
    }
    fn mean_phase_r(&self) -> f32 {
        if self.n == 0 {
            f32::NAN
        } else {
            (self.spectral_phase_r_sum / self.n as f64) as f32
        }
    }
    fn coherence_healthy(&self) -> bool {
        self.n > 0
            && !self.any_nonfinite
            && self.spectral_collapsed_count == 0
            && self.herm_norm_max <= self.herm_radius * 1.01
            && self.phase_locked_count > 0
    }
}

fn parse_args() -> (String, String, bool, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut model = "models/smollm2.mud".to_string();
    let mut prompts = "prompts.txt".to_string();
    let mut think = false;
    let mut max_prompts = 0usize; // 0 = unlimited

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--think" | "-t" => think = true,
            "--max" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_prompts = v;
                }
            }
            s if !s.starts_with('-') && model == "models/smollm2.mud" => {
                model = s.to_string();
            }
            s if !s.starts_with('-') => {
                prompts = s.to_string();
            }
            _ => {}
        }
        i += 1;
    }
    (model, prompts, think, max_prompts)
}

fn load_prompts(mud: &MudFile, prompts_path: &str, vocab_size: u32, max_prompts: usize) -> Vec<(Vec<u32>, u32)> {
    let raw = match std::fs::read_to_string(prompts_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[inference-coherence] cannot read {prompts_path}: {e}");
            return Vec::new();
        }
    };

    // Try u32 corpus first: each line has space-separated token IDs,
    // last one is the expected next token.
    let mut result: Vec<(Vec<u32>, u32)> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let ids: Vec<u32> = l
                .split_whitespace()
                .filter_map(|w| w.parse::<u32>().ok())
                .filter(|&id| id < vocab_size)
                .collect();
            if ids.len() >= 2 {
                let expected = ids[ids.len() - 1];
                let context = ids[..ids.len() - 1].to_vec();
                Some((context, expected))
            } else {
                None
            }
        })
        .collect();

    // Fallback: auto-tokenize raw text. Use the last token of each line as expected.
    if result.is_empty() {
        let tokens_str = mud.global_metadata.get("tokenizer.tokens").map(|s| s.as_str()).unwrap_or("");
        let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
        let tk = forge_llm::model::tokenizer::Tokenizer::from_mud_metadata(tokens_str, merges_str);
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ids: Vec<u32> = tk
                .encode(line)
                .into_iter()
                .filter(|&id| id < vocab_size)
                .collect();
            if ids.len() >= 2 {
                let expected = ids[ids.len() - 1];
                let context = ids[..ids.len() - 1].to_vec();
                result.push((context, expected));
            }
        }
    }

    if max_prompts > 0 && result.len() > max_prompts {
        result.truncate(max_prompts);
    }
    result
}

/// Numerically-stable softmax entropy of logits.
fn softmax_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits.iter().map(|x| ((*x - max_l) as f64).exp()).sum();
    let log_z = max_l as f64 + sum_exp.ln();
    let mut entropy = 0.0f64;
    for &l in logits {
        let p = (l as f64 - log_z).exp();
        if p > 1e-12 {
            entropy -= p * p.ln();
        }
    }
    entropy as f32
}

fn run_coherence(
    mud: &MudFile,
    prompts: &[(Vec<u32>, u32)],
    _label: &str,
    think_enabled: bool,
) -> CoherenceStats {
    // Toggle MUD_CMUD_THINK for this run.
    let prev = std::env::var("MUD_CMUD_THINK").ok();
    unsafe {
        if think_enabled {
            std::env::set_var("MUD_CMUD_THINK", "1");
        } else {
            std::env::remove_var("MUD_CMUD_THINK");
        }
    }

    let mut stats = CoherenceStats {
        entropy_min: f32::INFINITY,
        ..Default::default()
    };

    for (context, expected) in prompts.iter() {
        let result = if think_enabled {
            forward_last_logits_cmud(mud, context)
        } else {
            forward_last_logits(mud, context).map(|lg| (lg, Default::default()))
        };

        match result {
            Ok((lg, think_report)) => {
                stats.n += 1;
                if !lg.iter().all(|x| x.is_finite()) {
                    stats.any_nonfinite = true;
                }

                // Find top-1 prediction.
                let mut top1_idx = 0;
                let mut top1_logit = lg[0];
                for (i, &l) in lg.iter().enumerate().skip(1) {
                    if l > top1_logit {
                        top1_logit = l;
                        top1_idx = i;
                    }
                }

                let target_logit = if (*expected as usize) < lg.len() {
                    lg[*expected as usize]
                } else {
                    f32::NEG_INFINITY
                };

                let success = top1_idx == *expected as usize;
                let entropy = softmax_entropy(&lg);
                let logit_margin = top1_logit - target_logit;

                stats.successes += success as usize;
                stats.entropy_sum += entropy as f64;
                stats.entropy_min = stats.entropy_min.min(entropy);
                stats.entropy_max = stats.entropy_max.max(entropy);
                stats.logit_margin_sum += logit_margin as f64;

                // C-MUD coherence path metrics.
                stats.herm_norm_max = stats.herm_norm_max.max(think_report.max_herm_norm);
                stats.herm_radius = think_report.radius;
                if think_report.phase_locked {
                    stats.phase_locked_count += 1;
                }
                if think_report.spectral.collapsed {
                    stats.spectral_collapsed_count += 1;
                }
                stats.spectral_cauchy_sum += think_report.spectral.cauchy_mag_at_2 as f64;
                stats.spectral_spread_sum += think_report.spectral.spread_mag as f64;
                stats.spectral_phase_r_sum += think_report.spectral.circular_phase_r as f64;
            }
            Err(e) => {
                eprintln!("[inference-coherence] forward error on prompt: {e}");
                stats.any_nonfinite = true;
            }
        }
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

fn print_report(label: &str, stats: &CoherenceStats) {
    let sep = "═".repeat(60);
    println!("\n{sep}");
    println!("  COHERENCE REPORT — {} ({})", label, if stats.n > 0 { "OK" } else { "EMPTY" });
    println!("{sep}");
    println!();
    println!("  ┌─ Éxito (top-1 accuracy)");
    println!("  │  success_rate : {:.1}%  ({}/{})", stats.success_rate() * 100.0, stats.successes, stats.n);
    println!("  │  mean_logit_margin : {:+.4}  (top1 − target; >0 = correct)", stats.mean_logit_margin());
    println!();
    println!("  ┌─ Desviación (softmax entropy)");
    println!("  │  mean_entropy : {:.4} bits  (high = uncertain/divergent)", stats.mean_entropy());
    println!("  │  entropy_range : [{:.4}, {:.4}]  (variability indicator)", stats.entropy_min, stats.entropy_max);
    println!();
    println!("  ┌─ Camino por coherencia (C-MUD think path)");
    println!("  │  herm_norm_max : {:.4} / radius {:.4}  (ball: {})",
        stats.herm_norm_max, stats.herm_radius,
        if stats.herm_norm_max <= stats.herm_radius * 1.01 { "✓" } else { "✗" });
    println!("  │  phase_locked  : {}/{} prompts  ({:.0}%)",
        stats.phase_locked_count, stats.n,
        if stats.n > 0 { stats.phase_locked_count as f32 / stats.n as f32 * 100.0 } else { 0.0 });
    println!("  │  spectral_collapsed : {}/{}  ({})",
        stats.spectral_collapsed_count, stats.n,
        if stats.spectral_collapsed_count == 0 { "✓ healthy" } else { "✗ COLLAPSED" });
    println!("  │  mean_cauchy|G(2)| : {:.4}  (lower = more coherent manifold)", stats.mean_cauchy_mag());
    println!("  │  mean_spread_mag   : {:.4}  (magnitude diversity)", stats.mean_spread());
    println!("  │  mean_phase_R      : {:.4}  (circular phase coherence)", stats.mean_phase_r());
    println!();
    println!("  ┌─ Estado general");
    println!("  │  coherence_healthy : {}", if stats.coherence_healthy() { "✓ PASS" } else { "✗ FAIL" });
    println!("  │  any_nonfinite     : {}", if stats.any_nonfinite { "✗ YES" } else { "✓ no" });
    println!("{sep}");
}

fn main() -> ExitCode {
    let (model_path, prompts_path, think_enabled, max_prompts) = parse_args();

    println!("[inference-coherence] Loading model from {}...", model_path);
    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[inference-coherence] Failed to load model: {e}");
            return ExitCode::from(1);
        }
    };

    let vocab_size = mud
        .global_metadata
        .get("vocab_size")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(32000);

    println!("[inference-coherence] Loading prompts from {} (vocab={})...", prompts_path, vocab_size);
    let prompts = load_prompts(&mud, &prompts_path, vocab_size, max_prompts);
    if prompts.is_empty() {
        eprintln!("[inference-coherence] No valid prompts found. Exiting.");
        return ExitCode::from(1);
    }
    println!("[inference-coherence] {} prompts loaded.", prompts.len());

    // Run baseline (no think) and C-MUD think (if requested) for comparison.
    let baseline = run_coherence(&mud, &prompts, "baseline", false);
    print_report("baseline (no think)", &baseline);

    if think_enabled {
        let cmud = run_coherence(&mud, &prompts, "C-MUD think", true);
        print_report("C-MUD think", &cmud);

        // Comparison summary.
        let sep = "═".repeat(60);
        println!("\n{sep}");
        println!("  COMPARISON: baseline vs C-MUD think");
        println!("{sep}");
        println!("  success_rate   : {:.1}% → {:.1}%  (Δ {:+.1}%)",
            baseline.success_rate() * 100.0,
            cmud.success_rate() * 100.0,
            (cmud.success_rate() - baseline.success_rate()) * 100.0);
        println!("  mean_entropy   : {:.4} → {:.4}  (Δ {:+.4})",
            baseline.mean_entropy(),
            cmud.mean_entropy(),
            cmud.mean_entropy() - baseline.mean_entropy());
        println!("  mean_logit_margin : {:+.4} → {:+.4}",
            baseline.mean_logit_margin(),
            cmud.mean_logit_margin());
        println!("  coherence_healthy : {} → {}",
            if baseline.coherence_healthy() { "✓" } else { "✗" },
            if cmud.coherence_healthy() { "✓" } else { "✗" });
        println!("{sep}");
    }

    // Exit code: 0 if baseline is healthy, 1 otherwise.
    if baseline.coherence_healthy() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
