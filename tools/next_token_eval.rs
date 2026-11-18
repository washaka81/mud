//! Next-token prediction evaluation for `.mud` models.
//!
//! Each non-empty line of the prompts file is tokenized; the **last token** is
//! the expected next token and the preceding tokens are the context. For every
//! prompt the model runs one forward pass and we report:
//!
//! - rank of the correct token (1 = top-1)
//! - top-1 / top-5 / top-10 accuracy
//! - MRR (mean reciprocal rank)
//! - probability assigned to the correct token
//! - NLL (negative log-likelihood, nats) and perplexity = exp(mean NLL)
//! - softmax entropy (bits) and logit margin (correct − best other)
//!
//! Usage:
//!   cargo run --release --bin next_token_eval -- models/qwen_coder_0.5b.mud tests/data/next_token_eval.txt
//!   cargo run --release --bin next_token_eval -- models/ale.mud tests/data/next_token_eval.txt --max 8 --json
//!   cargo run --release --bin next_token_eval -- models/ale.mud tests/data/next_token_eval.txt --temperature-scan
//!
//! `--temperature-scan` reports T* (min NLL), PPL at T*, and ECE before/after
//! temperature scaling (10 bins).
//!
//! Lines may also be raw token IDs (`123 456 789`, last = expected).

use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::inference::forward_last_logits;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

#[derive(Default)]
struct Stats {
    n: usize,
    top1: usize,
    top5: usize,
    top10: usize,
    mrr_sum: f64,
    nll_sum: f64,
    p_sum: f64,
    rank_sum: f64,
    entropy_sum: f64,
    margin_sum: f64,
    ranks: Vec<usize>,
    /// Kept only for `--temperature-scan`; one logits vector per prompt.
    logits: Vec<Vec<f32>>,
    targets: Vec<usize>,
}

impl Stats {
    fn mean_nll(&self) -> f64 {
        if self.n == 0 { f64::NAN } else { self.nll_sum / self.n as f64 }
    }
    fn ppl(&self) -> f64 {
        self.mean_nll().exp()
    }
    fn pct(&self, hits: usize) -> f64 {
        if self.n == 0 { 0.0 } else { hits as f64 * 100.0 / self.n as f64 }
    }
    fn mean(&self, sum: f64) -> f64 {
        if self.n == 0 { f64::NAN } else { sum / self.n as f64 }
    }
    fn median_rank(&self) -> f64 {
        if self.ranks.is_empty() {
            return f64::NAN;
        }
        let mut r = self.ranks.clone();
        r.sort_unstable();
        let m = r.len() / 2;
        if r.len().is_multiple_of(2) {
            (r[m - 1] + r[m]) as f64 / 2.0
        } else {
            r[m] as f64
        }
    }
}

fn softmax_entropy_bits(logits: &[f32]) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f64 = logits
        .iter()
        .map(|&x| ((x - max) as f64).exp())
        .sum();
    let log_z = max as f64 + sum_exp.ln();
    let mut h = 0.0f64;
    for &x in logits {
        let p = (x as f64 - log_z).exp();
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h / std::f64::consts::LN_2
}

fn parse_prompt_line(line: &str, tk: &Tokenizer, vocab: usize) -> Option<(Vec<u32>, u32)> {
    let ids: Vec<u32> = if line
        .split_whitespace()
        .all(|w| w.parse::<u32>().is_ok())
    {
        line.split_whitespace()
            .filter_map(|w| w.parse::<u32>().ok())
            .filter(|&id| (id as usize) < vocab)
            .collect()
    } else {
        tk.encode(line)
            .into_iter()
            .filter(|&id| (id as usize) < vocab)
            .collect()
    };
    if ids.len() < 2 {
        return None;
    }
    let expected = *ids.last()?;
    let context = ids[..ids.len() - 1].to_vec();
    Some((context, expected))
}

fn run_eval(
    mud: &MudFile,
    tk: &Tokenizer,
    vocab: usize,
    lines: &[String],
    max_prompts: usize,
) -> Stats {
    let mut stats = Stats::default();
    for (idx, line) in lines.iter().enumerate() {
        if max_prompts > 0 && stats.n >= max_prompts {
            break;
        }
        let Some((context, expected)) = parse_prompt_line(line, tk, vocab) else {
            continue;
        };
        let logits = match forward_last_logits(mud, &context) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[{:02}] forward failed: {e}", idx + 1);
                continue;
            }
        };
        if logits.is_empty() || (expected as usize) >= logits.len() {
            eprintln!(
                "[{:02}] invalid target {} for logits len {}",
                idx + 1,
                expected,
                logits.len()
            );
            continue;
        }

        let target_logit = logits[expected as usize];
        let rank = 1 + logits.iter().filter(|&&l| l > target_logit).count();
        let best_other = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != expected as usize)
            .map(|(_, &l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let margin = target_logit - best_other;

        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logits
            .iter()
            .map(|&x| ((x - max) as f64).exp())
            .sum();
        let log_z = max as f64 + sum_exp.ln();
        let nll = log_z - target_logit as f64;
        let p = (-nll).exp();
        let entropy = softmax_entropy_bits(&logits);

        stats.n += 1;
        stats.top1 += usize::from(rank == 1);
        stats.top5 += usize::from(rank <= 5);
        stats.top10 += usize::from(rank <= 10);
        stats.mrr_sum += 1.0 / rank as f64;
        stats.nll_sum += nll;
        stats.p_sum += p;
        stats.rank_sum += rank as f64;
        stats.entropy_sum += entropy;
        stats.margin_sum += margin as f64;
        stats.ranks.push(rank);
        stats.logits.push(logits.clone());
        stats.targets.push(expected as usize);

        let target_str = tk.decode(&[expected]);
        println!(
            "[{:02}] ctx={:3} target={:6} {:>14?} rank={:5} p={:.3e} nll={:.4} top1={} margin={:+.4}",
            stats.n,
            context.len(),
            expected,
            target_str,
            rank,
            p,
            nll,
            if rank == 1 { "✓" } else { "✗" },
            margin
        );
    }
    stats
}

fn print_report(stats: &Stats) {
    let sep = "═".repeat(72);
    println!("\n{sep}");
    println!("  NEXT-TOKEN PREDICTION REPORT");
    println!("{sep}");
    println!("  prompts evaluated       : {}", stats.n);
    println!(
        "  top-1 accuracy          : {:.2}%  ({}/{})",
        stats.pct(stats.top1),
        stats.top1,
        stats.n
    );
    println!(
        "  top-5 accuracy          : {:.2}%  ({}/{})",
        stats.pct(stats.top5),
        stats.top5,
        stats.n
    );
    println!(
        "  top-10 accuracy         : {:.2}%  ({}/{})",
        stats.pct(stats.top10),
        stats.top10,
        stats.n
    );
    println!("  MRR                     : {:.6}", stats.mean(stats.mrr_sum));
    println!(
        "  mean NLL (nats)         : {:.6}",
        stats.mean_nll()
    );
    println!("  perplexity              : {:.6}", stats.ppl());
    println!("  mean p(correct)         : {:.6e}", stats.mean(stats.p_sum));
    println!("  mean rank               : {:.3}", stats.mean(stats.rank_sum));
    println!("  median rank             : {:.3}", stats.median_rank());
    println!("  mean entropy (bits)     : {:.6}", stats.mean(stats.entropy_sum));
    println!("  mean logit margin       : {:+.6}", stats.mean(stats.margin_sum));
    println!("{sep}");
}

fn print_json(stats: &Stats) {
    println!(
        "{{\"n\":{},\"top1\":{:.6},\"top5\":{:.6},\"top10\":{:.6},\"mrr\":{:.6},\"mean_nll\":{:.6},\"perplexity\":{:.6},\"mean_p\":{:.6e},\"mean_rank\":{:.6},\"median_rank\":{:.6},\"mean_entropy_bits\":{:.6},\"mean_margin\":{:.6}}}",
        stats.n,
        stats.pct(stats.top1),
        stats.pct(stats.top5),
        stats.pct(stats.top10),
        stats.mean(stats.mrr_sum),
        stats.mean_nll(),
        stats.ppl(),
        stats.mean(stats.p_sum),
        stats.mean(stats.rank_sum),
        stats.median_rank(),
        stats.mean(stats.entropy_sum),
        stats.mean(stats.margin_sum)
    );
}

fn logsumexp_t(logits: &[f32], t: f64) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits
        .iter()
        .map(|&x| ((x as f64 - max) / t).exp())
        .sum();
    // logsumexp(z/t) = max(z)/t + ln Σ exp((z_i - max(z))/t)
    max / t + sum.ln()
}

fn nll_at(logits: &[f32], target: usize, t: f64) -> f64 {
    logsumexp_t(logits, t) - logits[target] as f64 / t
}

fn confidence_at(logits: &[f32], t: f64) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits
        .iter()
        .map(|&x| ((x as f64 - max) / t).exp())
        .sum();
    1.0 / sum.max(1e-300)
}

fn ece_at(stats: &Stats, t: f64) -> f64 {
    if stats.n == 0 {
        return f64::NAN;
    }
    const BINS: usize = 10;
    let mut counts = [0usize; BINS];
    let mut conf_sum = [0.0f64; BINS];
    let mut correct_sum = [0usize; BINS];
    for (logits, &target) in stats.logits.iter().zip(stats.targets.iter()) {
        let conf = confidence_at(logits, t);
        let bin = ((conf * BINS as f64).floor() as usize).min(BINS - 1);
        counts[bin] += 1;
        conf_sum[bin] += conf;
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        correct_sum[bin] += usize::from(argmax == target);
    }
    let n = stats.n as f64;
    let mut ece = 0.0;
    for b in 0..BINS {
        if counts[b] == 0 {
            continue;
        }
        let acc = correct_sum[b] as f64 / counts[b] as f64;
        let conf = conf_sum[b] / counts[b] as f64;
        ece += (counts[b] as f64 / n) * (acc - conf).abs();
    }
    ece
}

fn print_temperature_report(stats: &Stats) {
    if stats.logits.is_empty() {
        println!("\n  temperature scan: no logits captured");
        return;
    }
    if std::env::var("MUD_DEBUG").is_ok() {
        for i in 0..stats.logits.len().min(3) {
            let l = &stats.logits[i];
            let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let max_idx = l
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            let t = stats.targets[i];
            eprintln!(
                "[debug] prompt {i}: target={t} z_y={:.4} max={:.4} max_idx={max_idx} nll@1={:.4} nll@0.5={:.4}",
                l[t],
                max,
                nll_at(l, t, 1.0),
                nll_at(l, t, 0.5)
            );
        }
    }
    // Grid 0.10..2.00 step 0.05 (logits can be very flat).
    let mut best_t = 1.0;
    let mut best_nll = f64::INFINITY;
    let mut t = 0.10;
    while t <= 2.0001 {
        let mean = stats
            .logits
            .iter()
            .zip(stats.targets.iter())
            .map(|(logits, &target)| nll_at(logits, target, t))
            .sum::<f64>()
            / stats.n.max(1) as f64;
        if mean < best_nll {
            best_nll = mean;
            best_t = t;
        }
        t += 0.05;
    }
    println!("\n{}", "─".repeat(72));
    println!("  TEMPERATURE / CALIBRATION");
    println!("{}", "─".repeat(72));
    println!("  T* (min NLL)           : {:.3}", best_t);
    println!("  NLL at T*              : {:.6}", best_nll);
    println!("  PPL at T*              : {:.6}", best_nll.exp());
    println!("  ECE at T=1.0           : {:.6}", ece_at(stats, 1.0));
    println!("  ECE at T*              : {:.6}", ece_at(stats, best_t));
    println!("{}", "─".repeat(72));
}

fn usage() -> ExitCode {
    eprintln!(
        "Usage: next_token_eval <model.mud> <prompts.txt> [--max N] [--json] [--temperature-scan]\n\
         Each non-empty line: context text (last token = expected), or raw token IDs."
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(model_path) = args.next() else {
        return usage();
    };
    let Some(prompts_path) = args.next() else {
        return usage();
    };
    let mut max_prompts = 0usize;
    let mut json = false;
    let mut temperature_scan = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max" => {
                if let Some(v) = args.next().and_then(|s| s.parse().ok()) {
                    max_prompts = v;
                }
            }
            "--json" => json = true,
            "--temperature-scan" => temperature_scan = true,
            other => {
                eprintln!("unknown argument: {other}");
                return usage();
            }
        }
    }

    println!("[next-token-eval] loading {model_path}");
    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[next-token-eval] load failed: {e}");
            return ExitCode::from(1);
        }
    };
    // P-13: metadata or fail. Never invent a vocab size — a wrong value
    // silently filters targets and invalidates every metric below.
    let vocab = mud
        .global_metadata
        .get("vocab_size")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| {
            mud.skills
                .get("core")
                .and_then(|c| c.tensors.get("token_embd.weight"))
                .and_then(|t| t.shape.first().copied())
        })
        .or_else(|| {
            mud.skills
                .get("core")
                .and_then(|c| c.tensors.get("output.weight"))
                .and_then(|t| t.shape.first().copied())
        });
    let Some(vocab) = vocab else {
        eprintln!(
            "[next-token-eval] P-13: no vocab_size in metadata and no rank-2 \
             token_embd.weight/output.weight to derive it from"
        );
        return ExitCode::from(1);
    };
    let tokens_str = mud
        .global_metadata
        .get("tokenizer.tokens")
        .map(String::as_str)
        .unwrap_or("");
    if tokens_str.is_empty() {
        eprintln!(
            "[next-token-eval] P-13: missing tokenizer.tokens; text prompts cannot be \
             tokenized (raw token-ID lines are still supported, but target decoding is disabled)"
        );
    }
    let merges_str = mud
        .global_metadata
        .get("tokenizer.merges")
        .map(String::as_str)
        .unwrap_or("");
    let tk = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let raw = match std::fs::read_to_string(&prompts_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[next-token-eval] cannot read {prompts_path}: {e}");
            return ExitCode::from(1);
        }
    };
    let lines: Vec<String> = raw
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    if lines.is_empty() {
        eprintln!("[next-token-eval] no prompts in {prompts_path}");
        return ExitCode::from(1);
    }

    println!(
        "[next-token-eval] {} lines, vocab={}, max_prompts={}",
        lines.len(),
        vocab,
        if max_prompts == 0 { "all".to_string() } else { max_prompts.to_string() }
    );
    let stats = run_eval(&mud, &tk, vocab, &lines, max_prompts);
    if stats.n == 0 {
        eprintln!("[next-token-eval] no valid prompts (need >= 2 tokens per line)");
        return ExitCode::from(1);
    }
    print_report(&stats);
    if temperature_scan {
        print_temperature_report(&stats);
    }
    if json {
        print_json(&stats);
    }
    ExitCode::SUCCESS
}
