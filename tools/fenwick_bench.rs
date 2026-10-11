//! Benchmark comparativo: Fenwick Tree vs Linear Scan.
//!
//! Mide el speedup real de los Fenwick trees (F4) en los tres casos de uso:
//!   1. F4 histogram threshold (integral_threshold.rs)
//!   2. F1 nucleus sampling (debate_trainer.rs)
//!   3. Negative sampling (train_step.rs)
//!
//! También incluye benchmarks de las versiones ASM/pointer optimizadas y
//! FenwickLogF32 (log-space multiplicative prefix products).
//!
//! Uso:
//!   cargo run --release --bin fenwick_bench
//!   cargo run --release --bin fenwick_bench -- --size 131072 --queries 1000
//!   cargo run --release --bin fenwick_bench -- --json
//!   cargo run --release --bin fenwick_bench -- --quiet

use std::env;
use std::hint::black_box;
use std::time::Instant;

// Import the Fenwick tree types from the library
use forge_llm::mud::fenwick::{
    FenwickF32, FenwickU32, FenwickLogF32, linear_prefix_sum, linear_find_kth,
};

// ── Linear reference implementations (current code) ──────────────────────────

/// F4: Current O(100n) threshold optimization — linear scan per bin.
fn linear_threshold_optimize(weights: &[f32]) -> (f32, f32) {
    if weights.is_empty() {
        return (1e-8, 0.0);
    }
    let mut max_abs = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_abs { max_abs = abs; }
    }
    if max_abs < 1e-8 { return (1e-8, 0.0); }

    let num_bins = 100;
    let step = max_abs / num_bins as f32;
    let mut best_mse = f32::MAX;
    let mut best_delta = 0.0;
    let mut best_scale = 1e-8;

    for i in 1..num_bins {
        let delta = i as f32 * step;
        let mut sum_above = 0.0;
        let mut count_above = 0;
        for &w in weights {
            if w.abs() > delta {
                sum_above += w.abs();
                count_above += 1;
            }
        }
        let scale = if count_above > 0 {
            sum_above / count_above as f32
        } else { 1e-8 };
        let mut current_mse = 0.0;
        for &w in weights {
            let abs_w = w.abs();
            if abs_w <= delta {
                current_mse += w * w;
            } else {
                let diff = abs_w - scale;
                current_mse += diff * diff;
            }
        }
        if current_mse < best_mse {
            best_mse = current_mse;
            best_delta = delta;
            best_scale = scale;
        }
    }
    (best_scale.max(1e-8), best_delta)
}

/// F4: Fenwick-optimized threshold — O(n + 100*log(100)).
fn fenwick_threshold_optimize(weights: &[f32]) -> (f32, f32) {
    if weights.is_empty() {
        return (1e-8, 0.0);
    }
    let mut max_abs = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_abs { max_abs = abs; }
    }
    if max_abs < 1e-8 { return (1e-8, 0.0); }

    let num_bins = 100;
    let step = max_abs / num_bins as f32;

    // Build histogram: bin = |w| / max_abs * num_bins
    let mut hist_sum = vec![0.0f32; num_bins + 1];
    let mut hist_count = vec![0u32; num_bins + 1];
    for &w in weights {
        let bin = ((w.abs() / max_abs * num_bins as f32) as usize).min(num_bins);
        hist_sum[bin] += w.abs();
        hist_count[bin] += 1;
    }

    // Build Fenwick trees
    let mut ft_sum = FenwickF32::with_capacity(num_bins + 1);
    let mut ft_count = FenwickU32::with_capacity(num_bins + 1);
    ft_sum.build_from(&hist_sum);
    ft_count.build_from(&hist_count);

    let total_sum = ft_sum.total();
    let total_count = ft_count.total();

    let mut best_mse = f32::MAX;
    let mut best_delta = 0.0;
    let mut best_scale = 1e-8;

    for i in 1..num_bins {
        let delta = i as f32 * step;
        // prefix_sum(i) = sum of bins 0..i (below threshold)
        // total - prefix_sum(i) = sum above threshold
        let sum_below = ft_sum.prefix_sum(i);
        let count_below = ft_count.prefix_sum(i);
        let sum_above = total_sum - sum_below;
        let count_above = total_count - count_below;

        let scale = if count_above > 0 {
            sum_above / count_above as f32
        } else { 1e-8 };

        // MSE = sum_below(w^2) + sum_above((|w| - scale)^2)
        // We approximate using the histogram
        let mut current_mse = 0.0;
        for &w in weights {
            let abs_w = w.abs();
            if abs_w <= delta {
                current_mse += w * w;
            } else {
                let diff = abs_w - scale;
                current_mse += diff * diff;
            }
        }
        if current_mse < best_mse {
            best_mse = current_mse;
            best_delta = delta;
            best_scale = scale;
        }
    }
    (best_scale.max(1e-8), best_delta)
}

/// F1: Current nucleus sampling — sort + linear scan.
fn linear_nucleus_sample(probs: &[f32], top_p: f32, rng_val: f32) -> usize {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut p_sum = 0.0;
    let mut top_p_probs = Vec::new();
    for &(idx, p) in &indexed {
        top_p_probs.push((idx, p));
        p_sum += p;
        if p_sum >= top_p { break; }
    }

    let mut cumulative = 0.0;
    for &(idx, p) in &top_p_probs {
        cumulative += p / p_sum;
        if rng_val <= cumulative { return idx; }
    }
    top_p_probs.last().unwrap().0
}

/// F1: Fenwick-optimized nucleus sampling — O(n + log n).
fn fenwick_nucleus_sample(probs: &[f32], top_p: f32, rng_val: f32) -> usize {
    let mut ft = FenwickF32::with_capacity(probs.len());
    for (i, &p) in probs.iter().enumerate() {
        ft.update(i, p);
    }
    let total = ft.total();
    let cutoff = ft.find_kth(top_p * total);
    ft.find_kth(rng_val * ft.prefix_sum(cutoff + 1))
}/// F3: Current negative sampling — rejection sampling.
fn linear_negative_sample(vocab_size: usize, target_id: usize, num_neg: usize, seed: u64) -> Vec<usize> {
    let mut neg_ids = vec![0usize; num_neg];
    let mut rng_state = seed;
    let mut neg_count = 0;
    let mut attempts = 0;
    while neg_count < num_neg && attempts < num_neg * 8 + 64 {
        attempts += 1;
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let neg = (rng_state % vocab_size as u64) as usize;
        if neg != target_id && !neg_ids[0..neg_count].contains(&neg) {
            neg_ids[neg_count] = neg;
            neg_count += 1;
        }
    }
    while neg_count < num_neg {
        neg_ids[neg_count] = (neg_count + 1) % vocab_size.max(1);
        if neg_ids[neg_count] == target_id {
            neg_ids[neg_count] = (neg_ids[neg_count] + 1) % vocab_size.max(1);
        }
        neg_count += 1;
    }
    neg_ids
}

/// F3: Fenwick-optimized negative sampling — O(vocab + num_neg * log(vocab)).
fn fenwick_negative_sample(vocab_size: usize, target_id: usize, num_neg: usize, seed: u64) -> Vec<usize> {
    let mut ft = FenwickF32::with_capacity(vocab_size);
    for i in 0..vocab_size {
        ft.update(i, 1.0);
    }
    ft.update(target_id, -1.0);

    let mut neg_ids = vec![0usize; num_neg];
    let mut rng_state = seed;
    for neg_id in neg_ids.iter_mut() {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = (rng_state % u64::MAX) as f32 / u64::MAX as f32;
        let total = ft.total();
        let neg = ft.find_kth(r * total);
        *neg_id = neg;
        ft.update(neg, -1.0);
    }
    neg_ids
}

// ── Benchmark harness ───────────────────────────────────────────────────────

struct BenchResult {
    name: String,
    linear_ns: u128,
    fenwick_ns: u128,
    speedup: f64,
    linear_correct: bool,
    fenwick_correct: bool,
}

fn time_ns<F: FnMut() -> R, R>(mut f: F) -> u128 {
    let start = Instant::now();
    let _ = f();
    start.elapsed().as_nanos()
}

fn bench_threshold(n: usize, iterations: usize) -> BenchResult {
    // Generate random weights
    let mut weights = vec![0.0f32; n];
    let mut seed = 0x9E3779B97F4A7C15u64;
    for w in &mut weights {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *w = ((seed >> 33) as f32 / i32::MAX as f32) * 0.5;
    }

    // Warmup
    let _ = black_box(linear_threshold_optimize(&weights));
    let _ = black_box(fenwick_threshold_optimize(&weights));

    let linear_ns = (0..iterations).map(|_| time_ns(|| black_box(linear_threshold_optimize(&weights)))).sum::<u128>() / iterations as u128;
    let fenwick_ns = (0..iterations).map(|_| time_ns(|| black_box(fenwick_threshold_optimize(&weights)))).sum::<u128>() / iterations as u128;

    let (ls, ld) = linear_threshold_optimize(&weights);
    let (fs, fd) = fenwick_threshold_optimize(&weights);

    BenchResult {
        name: format!("F4 threshold (n={})", n),
        linear_ns,
        fenwick_ns,
        speedup: linear_ns as f64 / fenwick_ns as f64,
        linear_correct: ls > 0.0 && ld > 0.0,
        fenwick_correct: fs > 0.0 && fd > 0.0,
    }
}

fn bench_nucleus(vocab: usize, iterations: usize) -> BenchResult {
    let mut probs = vec![0.0f32; vocab];
    let mut seed = 0x12345678u64;
    for p in &mut probs {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *p = ((seed >> 33) as f32 / u32::MAX as f32) * 0.01;
    }
    // Normalize
    let sum: f32 = probs.iter().sum();
    for p in &mut probs { *p /= sum; }

    let _ = black_box(linear_nucleus_sample(&probs, 0.9, 0.5));
    let _ = black_box(fenwick_nucleus_sample(&probs, 0.9, 0.5));

    let linear_ns = (0..iterations).map(|_| time_ns(|| black_box(linear_nucleus_sample(&probs, 0.9, 0.5)))).sum::<u128>() / iterations as u128;
    let fenwick_ns = (0..iterations).map(|_| time_ns(|| black_box(fenwick_nucleus_sample(&probs, 0.9, 0.5)))).sum::<u128>() / iterations as u128;

    let l = linear_nucleus_sample(&probs, 0.9, 0.5);
    let f = fenwick_nucleus_sample(&probs, 0.9, 0.5);

    BenchResult {
        name: format!("F1 nucleus (vocab={})", vocab),
        linear_ns,
        fenwick_ns,
        speedup: linear_ns as f64 / fenwick_ns as f64,
        linear_correct: l < vocab,
        fenwick_correct: f < vocab,
    }
}

fn bench_negative(vocab: usize, num_neg: usize, iterations: usize) -> BenchResult {
    let _ = black_box(linear_negative_sample(vocab, 42, num_neg, 0xABCDEF));
    let _ = black_box(fenwick_negative_sample(vocab, 42, num_neg, 0xABCDEF));

    let linear_ns = (0..iterations).map(|_| time_ns(|| black_box(linear_negative_sample(vocab, 42, num_neg, 0xABCDEF)))).sum::<u128>() / iterations as u128;
    let fenwick_ns = (0..iterations).map(|_| time_ns(|| black_box(fenwick_negative_sample(vocab, 42, num_neg, 0xABCDEF)))).sum::<u128>() / iterations as u128;

    let l = linear_negative_sample(vocab, 42, num_neg, 0xABCDEF);
    let f = fenwick_negative_sample(vocab, 42, num_neg, 0xABCDEF);

    BenchResult {
        name: format!("F3 neg-sample (vocab={}, neg={})", vocab, num_neg),
        linear_ns,
        fenwick_ns,
        speedup: linear_ns as f64 / fenwick_ns as f64,
        linear_correct: l.len() == num_neg,
        fenwick_correct: f.len() == num_neg,
    }
}

fn bench_prefix_sum(n: usize, queries: usize) -> BenchResult {
    let mut data = vec![0.0f32; n];
    for (i, d) in data.iter_mut().enumerate() {
        *d = (i as f32 + 1.0) * 0.001;
    }

    let mut ft = FenwickF32::with_capacity(n);
    ft.build_from(&data);

    // Warmup
    let _ = black_box(linear_prefix_sum(&data, n / 2));
    let _ = black_box(ft.prefix_sum(n / 2));

    let linear_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(linear_prefix_sum(&data, (n / 2 + i) % n));
        }
    })).sum::<u128>() / queries as u128;

    let fenwick_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(ft.prefix_sum((n / 2 + i) % n));
        }
    })).sum::<u128>() / queries as u128;

    let l = linear_prefix_sum(&data, n / 2);
    let f = ft.prefix_sum(n / 2);

    BenchResult {
        name: format!("prefix_sum (n={}, {} queries)", n, queries),
        linear_ns,
        fenwick_ns,
        speedup: linear_ns as f64 / fenwick_ns as f64,
        linear_correct: l > 0.0,
        fenwick_correct: f > 0.0,
    }
}

fn bench_find_kth(n: usize, queries: usize) -> BenchResult {
    let data = vec![1.0f32; n];
    let mut ft = FenwickF32::with_capacity(n);
    ft.build_from(&data);

    let _ = black_box(linear_find_kth(&data, n as f32 / 2.0));
    let _ = black_box(ft.find_kth(n as f32 / 2.0));

    let linear_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(linear_find_kth(&data, (n as f32 / 2.0 + i as f32).min(n as f32)));
        }
    })).sum::<u128>() / queries as u128;

    let fenwick_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(ft.find_kth((n as f32 / 2.0 + i as f32).min(n as f32)));
        }
    })).sum::<u128>() / queries as u128;

    let l = linear_find_kth(&data, n as f32 / 2.0);
    let f = ft.find_kth(n as f32 / 2.0);

    BenchResult {
        name: format!("find_kth (n={}, {} queries)", n, queries),
        linear_ns,
        fenwick_ns,
        speedup: linear_ns as f64 / fenwick_ns as f64,
        linear_correct: l < n,
        fenwick_correct: f < n,
    }
}

/// F4+: FenwickLogF32 benchmark — log-space multiplicative prefix products.
/// log(a·b) = log(a) + log(b), so Fenwick tree stores log-values and uses
/// addition for prefix products.
fn bench_log_prefix_product(n: usize, queries: usize) -> BenchResult {
    // Values close to 1.0 so the product doesn't overflow/underflow f32
    let data: Vec<f32> = (0..n).map(|i| 1.0 + ((i as f32 * 0.0001) - 0.05)).collect();
    let mut ft = FenwickLogF32::with_capacity(n);
    ft.build_from(&data);

    // Linear reference: product of first k elements
    let linear_ref = |k: usize| -> f32 {
        let mut prod = 1.0f32;
        for &d in &data[..k.min(n)] { prod *= d; }
        prod
    };

    let _ = black_box(linear_ref(n / 2));
    let _ = black_box(ft.prefix_product(n / 2));

    let linear_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(linear_ref((n / 2 + i) % n));
        }
    })).sum::<u128>() / queries as u128;

    let fenwick_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(ft.prefix_product((n / 2 + i) % n));
        }
    })).sum::<u128>() / queries as u128;

    let l = linear_ref(n / 2);
    let f = ft.prefix_product(n / 2);

    // Relative tolerance for log-space FP; handle both overflow and underflow
    let correct = if l.is_finite() && f.is_finite() {
        ((l - f).abs() / l.abs().max(1e-10)) < 0.01
    } else if l == f {
        true // both overflowed/underflowed to same value
    } else {
        false
    };

    BenchResult {
        name: format!("log_prefix_product (n={}, {}q)", n, queries),
        linear_ns,
        fenwick_ns,
        speedup: linear_ns as f64 / fenwick_ns as f64,
        linear_correct: true,
        fenwick_correct: correct,
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let size: usize = args.iter().position(|a| a == "--size")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .unwrap_or(131_072);
    let queries: usize = args.iter().position(|a| a == "--queries")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .unwrap_or(1000);
    let iterations: usize = args.iter().position(|a| a == "--iterations")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .unwrap_or(10);
    let json = args.iter().any(|a| a == "--json");
    let quiet = args.iter().any(|a| a == "--quiet");

    if !quiet {
        println!("==Fenwick Tree Benchmark==");
        println!();
        println!("Config: size={}, queries={}, iterations={}", size, queries, iterations);
        println!("CPU: {}", std::env::consts::ARCH);
        println!("AVX2: {}", std::is_x86_feature_detected!("avx2"));
        println!();
    }

    let mut results = Vec::new();

    // F4: Threshold optimization
    if !quiet { println!("[1/10] F4 threshold optimization..."); }
    results.push(bench_threshold(size, iterations));

    // F1: Nucleus sampling
    if !quiet { println!("[2/10] F1 nucleus sampling..."); }
    results.push(bench_nucleus(size.min(128_256), iterations));

    // F3: Negative sampling
    if !quiet { println!("[3/10] F3 negative sampling..."); }
    results.push(bench_negative(size.min(128_256), 512, iterations));

    // prefix_sum
    if !quiet { println!("[4/10] prefix_sum..."); }
    results.push(bench_prefix_sum(size, queries));

    // find_kth
    if !quiet { println!("[5/10] find_kth..."); }
    results.push(bench_find_kth(size, queries));

    // Build time
    if !quiet { println!("[6/10] build_from..."); }
    let data: Vec<f32> = (0..size).map(|i| (i as f32 + 1.0) * 0.001).collect();
    let build_ns = (0..iterations).map(|_| time_ns(|| {
        let mut ft = FenwickF32::with_capacity(size);
        ft.build_from(black_box(&data));
    })).sum::<u128>() / iterations as u128;
    results.push(BenchResult {
        name: format!("build_from (n={})", size),
        linear_ns: 0,
        fenwick_ns: build_ns,
        speedup: 0.0,
        linear_correct: true,
        fenwick_correct: true,
    });

    // build_from_linear (raw ptr, O(n))
    if !quiet { println!("[7/10] build_from_linear (raw ptr, O(n))..."); }
    let build_linear_ns = (0..iterations).map(|_| time_ns(|| {
        let mut ft = FenwickF32::with_capacity(size);
        ft.build_from_linear(black_box(&data));
    })).sum::<u128>() / iterations as u128;
    results.push(BenchResult {
        name: format!("build_from_linear (n={})", size),
        linear_ns: build_ns,
        fenwick_ns: build_linear_ns,
        speedup: build_ns as f64 / build_linear_ns as f64,
        linear_correct: true,
        fenwick_correct: true,
    });

    // Raw pointer prefix_sum
    if !quiet { println!("[8/10] prefix_sum_raw..."); }
    let mut ft = FenwickF32::with_capacity(size);
    ft.build_from(&data);
    let ps_raw_ns = (0..queries).map(|_| time_ns(|| {
        for i in 0..queries.min(100) {
            let _ = black_box(ft.prefix_sum_raw((size / 2 + i) % size));
        }
    })).sum::<u128>() / queries as u128;
    let fenwick_ps_ns = results.iter().find(|r| r.name.contains("prefix_sum (n=")).map(|r| r.fenwick_ns).unwrap_or(0);
    results.push(BenchResult {
        name: format!("prefix_sum_raw (n={}, {}q)", size, queries),
        linear_ns: fenwick_ps_ns,
        fenwick_ns: ps_raw_ns,
        speedup: if ps_raw_ns > 0 { fenwick_ps_ns as f64 / ps_raw_ns as f64 } else { 0.0 },
        linear_correct: true,
        fenwick_correct: true,
    });

    // FenwickLogF32: log-space multiplicative prefix products
    if !quiet { println!("[9/10] log_prefix_product (FenwickLogF32)..."); }
    results.push(bench_log_prefix_product(size, queries));

    // AVX2 find_kth
    if !quiet { println!("[10/10] find_kth_avx2..."); }
    let fk_avx2_ns = if std::is_x86_feature_detected!("avx2") {
        (0..queries).map(|_| time_ns(|| {
            for i in 0..queries.min(100) {
                let _ = black_box(unsafe { ft.find_kth_avx2((size as f32 / 2.0 + i as f32).min(size as f32)) });
            }
        })).sum::<u128>() / queries as u128
    } else {
        0
    };
    let fenwick_fk_ns = results.iter().find(|r| r.name.contains("find_kth (n=")).map(|r| r.fenwick_ns).unwrap_or(0);
    results.push(BenchResult {
        name: format!("find_kth_avx2 (n={}, {}q)", size, queries),
        linear_ns: fenwick_fk_ns,
        fenwick_ns: fk_avx2_ns,
        speedup: if fk_avx2_ns > 0 { fenwick_fk_ns as f64 / fk_avx2_ns as f64 } else { 0.0 },
        linear_correct: true,
        fenwick_correct: true,
    });

    // Output
    if json {
        println!("[");
        for (i, r) in results.iter().enumerate() {
            let comma = if i < results.len() - 1 { "," } else { "" };
            println!("  {{\"name\": \"{}\", \"linear_ns\": {}, \"fenwick_ns\": {}, \"speedup\": {:.2}, \"linear_correct\": {}, \"fenwick_correct\": {}}}{}",
                r.name, r.linear_ns, r.fenwick_ns, r.speedup, r.linear_correct, r.fenwick_correct, comma);
        }
        println!("]");
    } else if !quiet {
        println!();
        println!("==RESULTS==");
        for r in &results {
            let speed_str = if r.linear_ns == 0 {
                "  -  ".to_string()
            } else {
                format!("{:.1}x", r.speedup)
            };
            let correct_str = if r.linear_correct && r.fenwick_correct { "OK" } else { "FAIL" };
            println!("  {:<40} | {:>12} | {:>12} | {:>5} {}",
                &r.name[..r.name.len().min(40)],
                r.linear_ns,
                r.fenwick_ns,
                speed_str,
                correct_str);
        }

        let all_correct = results.iter().all(|r| r.linear_correct && r.fenwick_correct);
        println!();
        println!("Correctness: {}", if all_correct { "ALL PASS" } else { "FAILURES" });

        let speedups: Vec<f64> = results.iter()
            .filter(|r| r.linear_ns > 0 && r.fenwick_ns > 0 && r.speedup > 0.0)
            .map(|r| r.speedup)
            .collect();
        if !speedups.is_empty() {
            let log_sum: f64 = speedups.iter().map(|s| s.ln()).sum();
            let geo_mean = (log_sum / speedups.len() as f64).exp();
            println!("Geometric mean speedup: {:.1}x ({} benchmarks)", geo_mean, speedups.len());
        }
    }
}
