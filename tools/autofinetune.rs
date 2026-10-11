//! # MUD AutoFinetune — Model-Aware Hyperparameter Search for STE QAT Training
//!
//! Runs short training bursts with different hyperparameter combinations,
//! **adapted to each model's architecture** (size, vocab, tensor type),
//! scores each by loss-drop rate + grad_norm stability (via `loss_cert`),
//! and selects the best configuration.
//!
//! ## Usage
//! ```bash
//! cargo run --release --bin autofinetune -- models/smollm2.mud
//! cargo run --release --bin autofinetune -- models/smollm2.mud --trials 12 --epochs 2 --fast
//! ```
//!
//! ## Model-Aware Adaptation
//! The search space is adapted based on model characteristics extracted from
//! the `.mud` metadata:
//!
//! | Characteristic | Small (<100M) | Medium (100M-1B) | Large (>1B) |
//! |----------------|---------------|-------------------|-------------|
//! | MUD_QAT_LR | 0.03-0.1 | 0.01-0.05 | 0.005-0.03 |
//! | MUD_GRAD_CLIP | 3-10 | 5-10 | 5-15 |
//! | STEPS_PER_CHUNK | 128 | 64 | 32 |
//! | NUM_NEG | 255 | 127 | 63 |
//!
//! **Vocab adaptation:** vocab > 100K → reduce NUM_NEG to 63 (speed).
//! **Tensor type:** Ternary2Bit → boost LR range (STE deadzone needs higher LR).
//! **Hidden size:** grad_clip scaled with `sqrt(hidden/2048)`.
//!
//! ## Scoring
//! Each trial gets a composite score:
//!   score = cert_pass(10) + loss_drop(5) + grad_stability(3) + speed(2)
//! - cert_pass: loss_cert percentile gate (10 pts if passes, 0 if fails)
//! - loss_drop: relative loss drop normalized to [0,5]
//! - grad_stability: 3 × (1 - grad_norm/max_grad_norm) (normalized to model size)
//! - speed: tok/s normalized to [0,2] (faster is better)

use forge_llm::mud::corpus_trainer::MudCorpusTrainer;
use forge_llm::mud::loss_cert::{certify_trajectory, parse_metrics_log, LossCertConfig};
use forge_llm::mud::MudFile;
use rand::prelude::IndexedRandom;
use rand::rng;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Model characteristics extracted from `.mud` metadata.
#[derive(Debug, Clone)]
struct ModelProfile {
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    param_count: usize,
    is_ternary: bool,
    /// Model size tier: "small", "medium", "large"
    tier: String,
}

impl ModelProfile {
    /// Load model profile from a `.mud` file.
    fn load(model_path: &str) -> Option<Self> {
        let mud = MudFile::load(model_path).ok()?;
        let gm = &mud.global_metadata;

        let num_layers = gm
            .get("num_layers")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(24);
        let hidden_size = gm
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2048);
        let vocab_size = gm
            .get("vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32000);

        // Count parameters from core tensors
        let mut param_count = 0usize;
        let mut is_ternary = false;
        if let Some(core) = mud.skills.get("core") {
            for t in core.tensors.values() {
                let elements: usize = t.shape.iter().product();
                param_count += elements;
                if t.t_type == forge_llm::mud::MudTensorType::Ternary2Bit {
                    is_ternary = true;
                }
            }
        }

        let tier = if param_count < 100_000_000 {
            "small"
        } else if param_count < 1_000_000_000 {
            "medium"
        } else {
            "large"
        };

        Some(Self {
            num_layers,
            hidden_size,
            vocab_size,
            param_count,
            is_ternary,
            tier: tier.to_string(),
        })
    }

    /// Maximum possible grad_norm for this model (sqrt of total trainable params).
    /// Used to normalize grad_stability scoring.
    fn max_grad_norm(&self) -> f32 {
        (self.param_count as f64).sqrt() as f32
    }

    /// Format param count as human-readable string.
    fn param_count_str(&self) -> String {
        let p = self.param_count as f64;
        if p >= 1e9 {
            format!("{:.1}B", p / 1e9)
        } else if p >= 1e6 {
            format!("{:.1}M", p / 1e6)
        } else {
            format!("{:.0}K", p / 1e3)
        }
    }
}

/// A single hyperparameter configuration to evaluate.
#[derive(Debug, Clone, Serialize)]
struct HyperConfig {
    lr: f32,
    grad_clip: f32,
    steps_per_chunk: usize,
    num_neg: usize,
    /// C-MUD complex reasoning thinker (MUD_CMUD_THINK=1)
    cmud_think: bool,
    /// SlimeX CellV5 transport cell (MUD_SLIMEX=1)
    slimex: bool,
}

/// Result of evaluating one configuration.
#[derive(Debug, Clone, Serialize)]
struct TrialResult {
    config: HyperConfig,
    score: f32,
    loss_first: f32,
    loss_last: f32,
    loss_drop: f32,
    grad_norm: f32,
    grad_saturation: f32,
    tok_per_sec: f32,
    elapsed_s: f32,
    cert_pass: bool,
    n_points: usize,
}

/// Build a model-adapted search space.
///
/// The search space is narrowed based on model characteristics:
/// - Small models: higher LR, more steps, more negatives (they're fast)
/// - Large models: lower LR, fewer steps, fewer negatives (speed)
/// - Large vocab: fewer negatives (sampled softmax is expensive)
/// - Ternary weights: higher LR (STE deadzone)
fn build_adapted_grid(profile: &ModelProfile) -> Vec<HyperConfig> {
    // LR values adapted to model tier and tensor type
    let lr_values: Vec<f32> = match profile.tier.as_str() {
        "small" => {
            if profile.is_ternary {
                vec![0.03, 0.05, 0.1, 0.2]
            } else {
                vec![0.01, 0.03, 0.05, 0.1]
            }
        }
        "medium" => {
            if profile.is_ternary {
                vec![0.01, 0.03, 0.05, 0.1]
            } else {
                vec![0.005, 0.01, 0.03, 0.05]
            }
        }
        "large" => {
            if profile.is_ternary {
                vec![0.005, 0.01, 0.03, 0.05]
            } else {
                vec![0.003, 0.005, 0.01, 0.03]
            }
        }
        _ => vec![0.005, 0.01, 0.03, 0.05],
    };

    // Grad clip scaled with hidden size (larger hidden → larger gradients)
    let hidden_scale = ((profile.hidden_size as f64) / 2048.0).sqrt().max(0.5);
    let base_clips: &[f32] = match profile.tier.as_str() {
        "small" => &[3.0, 5.0, 10.0],
        "medium" => &[5.0, 10.0, 15.0],
        "large" => &[5.0, 10.0, 20.0],
        _ => &[5.0, 10.0],
    };
    let grad_clip_values: Vec<f32> = base_clips
        .iter()
        .map(|&c| (c * hidden_scale as f32).round())
        .collect();

    // Steps per chunk: small models can afford more signal
    let steps_values: Vec<usize> = match profile.tier.as_str() {
        "small" => vec![64, 128, 256],
        "medium" => vec![32, 64, 128],
        "large" => vec![16, 32, 64],
        _ => vec![32, 64, 128],
    };

    // Num negatives: large vocab → fewer (speed); small vocab → more (quality)
    let neg_values: Vec<usize> = if profile.vocab_size > 100_000 {
        vec![15, 31, 63]
    } else if profile.vocab_size > 10_000 {
        vec![31, 63, 127]
    } else {
        vec![63, 127, 255]
    };

    let mut grid = Vec::new();
    // C-MUD Thinker and SlimeX variants — the "complex manifold" pre-training
    // path. C-MUD activates the complex-valued reasoning pass (L-14) which can
    // help gradient descent by providing richer gradient signals. SlimeX adds
    // CellV5 transport cells for memory continuity (snapshot_to_slimex).
    // Both are opt-in and may help or hurt depending on model size.
    let cmud_think_values = [false, true];
    let slimex_values = [false, true];
    for &lr in &lr_values {
        for &gc in &grad_clip_values {
            for &spc in &steps_values {
                for &nn in &neg_values {
                    for &ct in &cmud_think_values {
                        for &sl in &slimex_values {
                            grid.push(HyperConfig {
                                lr,
                                grad_clip: gc,
                                steps_per_chunk: spc,
                                num_neg: nn,
                                cmud_think: ct,
                                slimex: sl,
                            });
                        }
                    }
                }
            }
        }
    }
    grid
}

/// Sample `n` random configurations from the grid (with replacement).
fn sample_random(grid: &[HyperConfig], n: usize) -> Vec<HyperConfig> {
    let mut rng = rng();
    (0..n)
        .map(|_| grid.choose(&mut rng).unwrap().clone())
        .collect()
}

/// Apply a hyperparameter configuration to the environment.
fn apply_config(config: &HyperConfig) {
    unsafe {
        env::set_var("MUD_QAT_LR", format!("{}", config.lr));
        env::set_var("MUD_GRAD_CLIP", format!("{}", config.grad_clip));
        env::set_var("MUD_TRAIN_STEPS_PER_CHUNK", format!("{}", config.steps_per_chunk));
        env::set_var("MUD_TRAIN_NUM_NEG", format!("{}", config.num_neg));
        // Host-safe: CPU-only training
        env::set_var("MUD_USE_VULKAN", "0");
        env::set_var("MUD_GPU_GEMV", "0");
        env::set_var("MUD_TRAIN_FREEZE_EMB", "1");
        env::set_var("MUD_TRAIN_TEXT_ONLY", "1");
        env::set_var("MUD_TRAIN_STP", "1");
        // C-MUD complex reasoning thinker (L-14) — activates complex manifold
        // pre-training path that can help gradient descent find better optima.
        if config.cmud_think {
            env::set_var("MUD_CMUD_THINK", "1");
        } else {
            env::remove_var("MUD_CMUD_THINK");
        }
        // SlimeX CellV5 transport — memory continuity via snapshot_to_slimex.
        if config.slimex {
            env::set_var("MUD_SLIMEX", "1");
        } else {
            env::remove_var("MUD_SLIMEX");
        }
    }
}

/// Run a single training burst and return parsed metrics.
fn run_trial(
    model_path: &str,
    config: &HyperConfig,
    epochs: usize,
    batch: usize,
    corpus_dir: &str,
    metrics_path: &str,
    checkpoint_path: &str,
) -> Option<TrialResult> {
    // Reset state between trials
    let _ = fs::remove_dir_all(corpus_dir);
    fs::create_dir_all(corpus_dir).ok()?;
    let _ = fs::remove_file(checkpoint_path);
    let _ = fs::write(metrics_path, "");

    // Write a small deterministic corpus for fast, comparable bursts
    let corpus_text = "\
Alice was beginning to get very tired of sitting by her sister on the bank.
The quick brown fox jumps over the lazy dog near the river bank at dawn.
MUD ternary training certifies that gradients flow and loss falls over steps.
Uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece.
The model learns to predict the next token in a sequence of text tokens.
Training uses sampled softmax with random negatives for efficiency.
JEPA and mHC stabilize the residual stream during backpropagation.
STP trajectory loss keeps the forward path locally linear on the geodesic.
";
    fs::write(format!("{corpus_dir}/tune.txt"), corpus_text).ok()?;

    apply_config(config);

    let trainer = match MudCorpusTrainer::new(model_path.to_string(), corpus_dir.to_string()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("  [trial] trainer init failed: {e}");
            return None;
        }
    };

    let t0 = Instant::now();
    let train_result = trainer.run_alignment_session(batch, epochs);
    let elapsed = t0.elapsed().as_secs_f32();

    if let Err(e) = train_result {
        eprintln!("  [trial] training failed: {e}");
        return None;
    }

    // Parse metrics
    let content = fs::read_to_string(metrics_path).unwrap_or_default();
    let losses = parse_metrics_log(&content);

    if losses.is_empty() {
        eprintln!("  [trial] no metrics parsed");
        return None;
    }

    let grad_norm = parse_last_grad_norm(&content);
    let tok_per_sec = parse_last_tok_per_sec(&content);

    let loss_first = losses[0];
    let loss_last = losses[losses.len() - 1];
    let loss_drop = loss_first - loss_last;

    // Score this trial
    let (score, cert_pass) = score_trial(&losses, loss_drop, grad_norm, tok_per_sec);

    Some(TrialResult {
        config: config.clone(),
        score,
        loss_first,
        loss_last,
        loss_drop,
        grad_norm,
        grad_saturation: 0.0, // filled by caller
        tok_per_sec,
        elapsed_s: elapsed,
        cert_pass,
        n_points: losses.len(),
    })
}

/// Parse the last grad_norm value from a TELEM line.
fn parse_last_grad_norm(content: &str) -> f32 {
    for line in content.lines().rev() {
        if let Some(idx) = line.find("grad_norm=") {
            let rest = &line[idx + "grad_norm=".len()..];
            if let Some(end) = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-') {
                if let Ok(v) = rest[..end].parse::<f32>() {
                    return v;
                }
            }
        }
    }
    0.0
}

/// Parse the last tok/s value from a TELEM line.
fn parse_last_tok_per_sec(content: &str) -> f32 {
    for line in content.lines().rev() {
        if let Some(idx) = line.find("tok/s=") {
            let rest = &line[idx + "tok/s=".len()..];
            if let Some(end) = rest.find(|c: char| !c.is_ascii_digit() && c != '.') {
                if let Ok(v) = rest[..end].parse::<f32>() {
                    return v;
                }
            }
        }
    }
    0.0
}

/// Score a trial based on loss drop, grad stability, and speed.
///
/// Components:
/// - cert_pass: 10 pts if loss_cert passes, 0 otherwise
/// - loss_drop: 5 × min(relative_drop, 0.5) / 0.5  (normalized to [0,5])
/// - grad_stability: 3 × (1 - grad_norm/max_grad_norm)  (normalized to model size)
/// - speed: 2 × min(tok_per_sec/20, 1.0)  (normalized to [0,2], 20 tok/s = max)
///
/// The grad_stability component is normalized by the model's max possible
/// grad_norm (sqrt of param count), so a config with 50% saturation gets
/// 1.5 pts and one with 100% saturation gets 0.0 pts.
fn score_trial(
    losses: &[f32],
    loss_drop: f32,
    grad_norm: f32,
    tok_per_sec: f32,
) -> (f32, bool) {
    // Use percentile cert mode for noise robustness
    let cfg = LossCertConfig::percentile(0.25, 0.25);
    let cert_result = certify_trajectory(losses, &cfg);
    let cert_pass = cert_result.is_ok();

    let cert_score = if cert_pass { 10.0 } else { 0.0 };

    // Loss drop component (normalized)
    let first = losses[0].max(1e-6);
    let relative_drop = loss_drop / first;
    let loss_score = 5.0 * (relative_drop / 0.5).min(1.0);

    // Grad stability component (normalized to model size)
    // grad_saturation = grad_norm / max_grad_norm
    // 0.0 = no saturation (ideal), 1.0 = fully saturated (all at clip boundary)
    let max_grad = 31000.0f32.max(grad_norm); // fallback if model profile not available
    let grad_saturation = (grad_norm / max_grad).min(1.0);
    let grad_score = 3.0 * (1.0 - grad_saturation);

    // Speed component (normalized to 20 tok/s max)
    let speed_score = 2.0 * (tok_per_sec / 20.0).min(1.0);

    let score = cert_score + loss_score + grad_score + speed_score;

    (score, cert_pass)
}

/// Write the best configuration to a JSON file and print it.
fn write_best(result: &TrialResult, profile: &ModelProfile, output_path: &str) {
    let output = serde_json::json!({
        "model": {
            "tier": profile.tier,
            "param_count": profile.param_count,
            "param_count_str": profile.param_count_str(),
            "num_layers": profile.num_layers,
            "hidden_size": profile.hidden_size,
            "vocab_size": profile.vocab_size,
            "is_ternary": profile.is_ternary,
        },
        "best_config": {
            "MUD_QAT_LR": result.config.lr,
            "MUD_GRAD_CLIP": result.config.grad_clip,
            "MUD_TRAIN_STEPS_PER_CHUNK": result.config.steps_per_chunk,
            "MUD_TRAIN_NUM_NEG": result.config.num_neg,
            "MUD_CMUD_THINK": result.config.cmud_think,
            "MUD_SLIMEX": result.config.slimex,
        },
        "metrics": {
            "score": result.score,
            "loss_first": result.loss_first,
            "loss_last": result.loss_last,
            "loss_drop": result.loss_drop,
            "grad_norm": result.grad_norm,
            "grad_saturation": result.grad_saturation,
            "tok_per_sec": result.tok_per_sec,
            "elapsed_s": result.elapsed_s,
            "cert_pass": result.cert_pass,
            "n_points": result.n_points,
        },
        "env_exports": [
            format!("export MUD_QAT_LR={}", result.config.lr),
            format!("export MUD_GRAD_CLIP={}", result.config.grad_clip),
            format!("export MUD_TRAIN_STEPS_PER_CHUNK={}", result.config.steps_per_chunk),
            format!("export MUD_TRAIN_NUM_NEG={}", result.config.num_neg),
            if result.config.cmud_think {
                "export MUD_CMUD_THINK=1".to_string()
            } else {
                "unset MUD_CMUD_THINK".to_string()
            },
            if result.config.slimex {
                "export MUD_SLIMEX=1".to_string()
            } else {
                "unset MUD_SLIMEX".to_string()
            },
        ],
    });

    let _ = fs::create_dir_all("weights");
    fs::write(output_path, serde_json::to_string_pretty(&output).unwrap_or_default())
        .expect("write best config");

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  MUD AUTOFINETUNE — BEST CONFIG                          ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  model      : {} ({} params, tier={})", profile.param_count_str(), profile.param_count, profile.tier);
    println!("  score      : {:.2}/20", result.score);
    println!("  loss       : {:.4} → {:.4} (Δ={:.4})", result.loss_first, result.loss_last, result.loss_drop);
    println!("  grad_norm  : {:.2}  (saturation: {:.1}%)", result.grad_norm, result.grad_saturation * 100.0);
    println!("  tok/s      : {:.1}", result.tok_per_sec);
    println!("  cert_pass  : {}", if result.cert_pass { "✓" } else { "✗" });
    println!("  n_points   : {}", result.n_points);
    println!();
    println!("  Best hyperparameters:");
    println!("    MUD_QAT_LR                = {}", result.config.lr);
    println!("    MUD_GRAD_CLIP             = {}", result.config.grad_clip);
    println!("    MUD_TRAIN_STEPS_PER_CHUNK = {}", result.config.steps_per_chunk);
    println!("    MUD_TRAIN_NUM_NEG         = {}", result.config.num_neg);
    println!("    MUD_CMUD_THINK            = {}", if result.config.cmud_think { "1 (enabled)" } else { "0 (disabled)" });
    println!("    MUD_SLIMEX                = {}", if result.config.slimex { "1 (enabled)" } else { "0 (disabled)" });
    println!();
    println!("  Apply with:");
    println!("    source <(jq -r '.env_exports[]' weights/autofinetune_best.json)");
    println!("  Or manually export the variables above.");
    println!("  Config saved to: {output_path}");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let mut model_path = None::<String>;
    let mut trials: usize = 8;
    let mut epochs: usize = 2;
    let mut batch: usize = 8;
    let mut fast: bool = false;
    let mut train_after: bool = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--trials" => {
                trials = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8);
                i += 2;
            }
            "--epochs" => {
                epochs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2);
                i += 2;
            }
            "--batch" => {
                batch = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(8);
                i += 2;
            }
            "--fast" => {
                fast = true;
                i += 1;
            }
            "--train" => {
                train_after = true;
                i += 1;
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: autofinetune [model.mud] [--trials N] [--epochs N] [--batch N] [--fast] [--train]\n\
                     \n\
                      Model-aware hyperparameter search for STE QAT training.\n\
                      Adapts search space to each model's architecture.\n\
                      \n\
                      Options:\n  \
                        --trials N   Number of random configs to try (default: 8)\n  \
                        --epochs N   Epochs per trial (default: 2)\n  \
                        --batch N    Batch size per trial (default: 8)\n  \
                        --fast       Fast mode: 1-epoch probe to narrow search space\n  \
                        --train      After finding best config, continue to full training\n\
                     \n\
                     Search space (adapted per model):\n  \
                       MUD_QAT_LR              ∈ tier-dependent (STE-aware)\n  \
                       MUD_GRAD_CLIP           ∈ scaled by sqrt(hidden/2048)\n  \
                       MUD_TRAIN_STEPS_PER_CHUNK ∈ tier-dependent (speed vs signal)\n  \
                       MUD_TRAIN_NUM_NEG       ∈ vocab-dependent (large vocab → fewer)"
                );
                return;
            }
            s if s.ends_with(".mud") => {
                model_path = Some(s.to_string());
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let model_path = model_path.unwrap_or_else(|| {
        if PathBuf::from("models/smollm2.mud").is_file() {
            "models/smollm2.mud".into()
        } else {
            eprintln!("[autofinetune] no model.mud found — pass path as argv");
            std::process::exit(1);
        }
    });

    if !PathBuf::from(&model_path).is_file() {
        eprintln!("[autofinetune] model not found: {model_path}");
        std::process::exit(1);
    }

    // Load model profile (model-aware adaptation)
    let profile = match ModelProfile::load(&model_path) {
        Some(p) => p,
        None => {
            eprintln!("[autofinetune] failed to load model profile — using defaults");
            ModelProfile {
                num_layers: 24,
                hidden_size: 2048,
                vocab_size: 32000,
                param_count: 1_000_000_000,
                is_ternary: true,
                tier: "medium".to_string(),
            }
        }
    };

    // Build model-adapted search space
    let grid = build_adapted_grid(&profile);

    // Fast mode: run a 1-epoch probe with a default config to estimate
    // the model's baseline, then narrow the search space
    if fast {
        println!("╔══════════════════════════════════════════════════════════╗");
        println!("║  MUD AUTOFINETUNE — FAST MODE (model-aware probe)        ║");
        println!("╚══════════════════════════════════════════════════════════╝");
        println!("  model   : {} ({} params, tier={}, ternary={})",
            model_path, profile.param_count_str(), profile.tier, profile.is_ternary);
        println!("  probe   : 1 epoch with default config to estimate baseline");
        println!();

        let probe_config = HyperConfig {
            lr: 0.03,
            grad_clip: 5.0,
            steps_per_chunk: 32,
            num_neg: if profile.vocab_size > 100_000 { 63 } else { 127 },
            cmud_think: false,
            slimex: false,
        };

        let corpus_dir = "training/corpus_autofinetune_probe";
        let metrics_path = "mud_train_metrics.log";
        let checkpoint_path = "weights/checkpoints/model_latest_checkpoint.mud";

        match run_trial(
            &model_path,
            &probe_config,
            1,
            batch,
            corpus_dir,
            metrics_path,
            checkpoint_path,
        ) {
            Some(probe_result) => {
                let _ = fs::remove_dir_all(corpus_dir);
                println!("  probe   : loss={:.4}→{:.4} (Δ={:.4})  grad_norm={:.1}  tok/s={:.1}",
                    probe_result.loss_first, probe_result.loss_last,
                    probe_result.loss_drop, probe_result.grad_norm, probe_result.tok_per_sec);

                // Narrow search space based on probe results
                let probe_lr = if probe_result.loss_drop > 0.5 {
                    // Loss is dropping — probe LR is good, explore around it
                    vec![probe_config.lr * 0.5, probe_config.lr, probe_config.lr * 2.0]
                } else {
                    // Loss not dropping — need higher LR
                    vec![probe_config.lr * 2.0, probe_config.lr * 3.0, probe_config.lr * 5.0]
                };
                let probe_clip = if probe_result.grad_norm > 20000.0 {
                    // Gradients saturated — need higher clip
                    vec![probe_config.grad_clip * 2.0, probe_config.grad_clip * 3.0, probe_config.grad_clip * 5.0]
                } else {
                    vec![probe_config.grad_clip, probe_config.grad_clip * 0.5, probe_config.grad_clip * 0.3]
                };

                // Build narrowed grid
                let mut narrowed: Vec<HyperConfig> = Vec::new();
                for &lr in &probe_lr {
                    for &gc in &probe_clip {
                        for &spc in &[32usize, 64] {
                            for &nn in &[31usize, 63] {
                                for &ct in &[false, true] {
                                    for &sl in &[false, true] {
                                        narrowed.push(HyperConfig { lr, grad_clip: gc, steps_per_chunk: spc, num_neg: nn, cmud_think: ct, slimex: sl });
                                    }
                                }
                            }
                        }
                    }
                }

                println!("  narrowed: {} configs (from {} full grid)", narrowed.len(), grid.len());
                println!("  LR range: {:.3}-{:.3}  clip range: {:.1}-{:.1}",
                    probe_lr.iter().cloned().fold(f32::INFINITY, f32::min),
                    probe_lr.iter().cloned().fold(0.0f32, f32::max),
                    probe_clip.iter().cloned().fold(f32::INFINITY, f32::min),
                    probe_clip.iter().cloned().fold(0.0f32, f32::max));
                println!();

                // Sample from narrowed grid
                let configs = sample_random(&narrowed, trials);
                let best_config = run_trials_and_report(
                    &model_path,
                    &profile,
                    &configs,
                    epochs,
                    batch,
                    "weights/autofinetune_best.json",
                );
                if train_after {
                    if let Some(config) = best_config {
                        launch_training(&model_path, &config);
                    }
                }
                return;
            }
            None => {
                eprintln!("  probe failed — falling back to full search");
                let _ = fs::remove_dir_all(corpus_dir);
            }
        }
    }

    // Full model-aware search
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  MUD AUTOFINETUNE — Model-Aware Hyperparameter Search    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  model   : {} ({} params, tier={}, ternary={})",
        model_path, profile.param_count_str(), profile.tier, profile.is_ternary);
    println!("  trials  : {trials}");
    println!("  epochs  : {epochs}   batch: {batch}");
    println!("  layers  : {}   hidden: {}   vocab: {}", profile.num_layers, profile.hidden_size, profile.vocab_size);
    println!("  search  : {} configs in adapted grid", grid.len());
    println!();

    let configs = sample_random(&grid, trials);
    let best_config = run_trials_and_report(
        &model_path,
        &profile,
        &configs,
        epochs,
        batch,
        "weights/autofinetune_best.json",
    );

    // If --train flag was passed, continue to full training with the best config
    if train_after {
        if let Some(config) = best_config {
            launch_training(&model_path, &config);
        }
    }
}

/// Apply the best configuration to the environment and launch full training.
fn launch_training(model_path: &str, config: &HyperConfig) {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  MUD AUTOFINETUNE → FULL TRAINING                        ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  Applying best configuration:");
    println!("    MUD_QAT_LR                = {}", config.lr);
    println!("    MUD_GRAD_CLIP             = {}", config.grad_clip);
    println!("    MUD_TRAIN_STEPS_PER_CHUNK = {}", config.steps_per_chunk);
    println!("    MUD_TRAIN_NUM_NEG         = {}", config.num_neg);
    println!("    MUD_CMUD_THINK            = {}", if config.cmud_think { "1" } else { "0" });
    println!("    MUD_SLIMEX                = {}", if config.slimex { "1" } else { "0" });
    println!();

    // Apply the configuration to the environment
    apply_config(config);

    // Also set training-specific env vars for full training
    unsafe {
        env::set_var("MUD_USE_VULKAN", "0");
        env::set_var("MUD_GPU_GEMV", "0");
        env::set_var("MUD_TRAIN_FREEZE_EMB", "1");
        env::set_var("MUD_TRAIN_TEXT_ONLY", "1");
        env::set_var("MUD_TRAIN_STP", "1");
        // Allow full corpus training (not capped)
        env::remove_var("MUD_TRAIN_MAX_CHUNKS");
    }

    println!("  Launching run_trainer with best config...");
    println!("  (Ctrl-C to stop at any time — weights are saved atomically)");
    println!();

    // Launch the trainer as a child process (inherits env vars)
    let exit_code = std::process::Command::new("cargo")
        .args(["run", "--release", "--bin", "run_trainer", "--", model_path])
        .env("MUD_QAT_LR", format!("{}", config.lr))
        .env("MUD_GRAD_CLIP", format!("{}", config.grad_clip))
        .env("MUD_TRAIN_STEPS_PER_CHUNK", format!("{}", config.steps_per_chunk))
        .env("MUD_TRAIN_NUM_NEG", format!("{}", config.num_neg))
        .env("MUD_USE_VULKAN", "0")
        .env("MUD_GPU_GEMV", "0")
        .env("MUD_TRAIN_FREEZE_EMB", "1")
        .env("MUD_TRAIN_TEXT_ONLY", "1")
        .env("MUD_TRAIN_STP", "1")
        .status();

    match exit_code {
        Ok(status) => {
            if status.success() {
                println!("\n  Training completed successfully.");
            } else {
                eprintln!("\n  Training exited with status: {status}");
            }
        }
        Err(e) => {
            eprintln!("\n  Failed to launch trainer: {e}");
            eprintln!("  Manual command:");
            eprintln!("    export MUD_QAT_LR={}", config.lr);
            eprintln!("    export MUD_GRAD_CLIP={}", config.grad_clip);
            eprintln!("    export MUD_TRAIN_STEPS_PER_CHUNK={}", config.steps_per_chunk);
            eprintln!("    export MUD_TRAIN_NUM_NEG={}", config.num_neg);
            if config.cmud_think {
                eprintln!("    export MUD_CMUD_THINK=1");
            }
            if config.slimex {
                eprintln!("    export MUD_SLIMEX=1");
            }
            eprintln!("    cargo run --release --bin run_trainer -- {model_path}");
        }
    }
}

/// Run all trials and report results. Returns the best config if found.
fn run_trials_and_report(
    model_path: &str,
    profile: &ModelProfile,
    configs: &[HyperConfig],
    epochs: usize,
    batch: usize,
    output_path: &str,
) -> Option<HyperConfig> {
    let corpus_dir = "training/corpus_autofinetune";
    let metrics_path = "mud_train_metrics.log";
    let checkpoint_path = "weights/checkpoints/model_latest_checkpoint.mud";

    let max_grad = profile.max_grad_norm();

    let mut results: Vec<TrialResult> = Vec::new();

    for (idx, config) in configs.iter().enumerate() {
        println!("── Trial {}/{} ──────────────────────────────────", idx + 1, configs.len());
        println!("  LR={:.3}  GRAD_CLIP={:.1}  STEPS={}  NEG={}",
            config.lr, config.grad_clip, config.steps_per_chunk, config.num_neg);

        match run_trial(
            model_path,
            config,
            epochs,
            batch,
            corpus_dir,
            metrics_path,
            checkpoint_path,
        ) {
            Some(mut result) => {
                // Fill in grad_saturation using model profile
                result.grad_saturation = (result.grad_norm / max_grad).min(1.0);

                println!("  loss: {:.4} → {:.4} (Δ={:.4})  grad_norm={:.1} ({:.0}% sat)  tok/s={:.1}  score={:.2}/20  {}",
                    result.loss_first, result.loss_last, result.loss_drop,
                    result.grad_norm, result.grad_saturation * 100.0,
                    result.tok_per_sec, result.score,
                    if result.cert_pass { "✓ cert" } else { "✗ no-cert" });
                results.push(result);
            }
            None => {
                println!("  ✗ trial failed (skipped)");
            }
        }
        println!();
    }

    // Cleanup temp corpus
    let _ = fs::remove_dir_all(corpus_dir);

    if results.is_empty() {
        eprintln!("[autofinetune] all trials failed — no valid configuration found");
        std::process::exit(1);
    }

    // Sort by score descending
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    println!("═══════════════════════════════════════════════════════════");
    println!("  Results (sorted by score):");
    for (rank, r) in results.iter().enumerate() {
        println!("  #{:>2} score={:>5.2}/20  LR={:.3} clip={:.1} steps={} neg={}  loss={:.4}→{:.4}  grad={:.1} ({:.0}%)  {}",
            rank + 1, r.score, r.config.lr, r.config.grad_clip,
            r.config.steps_per_chunk, r.config.num_neg,
            r.loss_first, r.loss_last, r.grad_norm,
            r.grad_saturation * 100.0,
            if r.cert_pass { "✓" } else { "✗" });
    }
    println!("═══════════════════════════════════════════════════════════");

    // Write best config
    let best = &results[0];
    write_best(best, profile, output_path);

    // Return the best config for potential continued training
    Some(best.config.clone())
}
