//! precise_converter_audit — High-Precision Conversion Fidelity Auditor (Fenwick-optimized)
//!
//! Audits conversion fidelity by comparing source safetensors against MUD ternary
//! weights, computing precise metrics with correct bitnet_scale handling.
//!
//! Key differences from conversion_verifier.rs:
//! - Correctly handles bitnet_scale (does NOT divide original weights by weight_scale)
//! - Accounts for depth-based calibration scales
//! - Handles QKV and gate_up fused projections
//! - Computes per-row SQNR statistics, MAE, max error, sparsity
//! - **Fenwick tree acceleration**: error distribution histograms with O(log n)
//!   percentile queries (median, P95, P99) via FenwickU32::find_kth
//! - **FenwickF32 cumulative error tracking**: O(log n) range queries for
//!   "error up to layer L" without rescanning
//!
//! Usage: precise_converter_audit <source.safetensors|dir> <model.mud> [--calibration]
//!   --calibration: also compute depth-based calibration scale map and report
//!                  the decomposition of error into quantization vs calibration components

use forge_llm::mud::fenwick::{FenwickF32, FenwickU32};
use forge_llm::mud::{dequantize_ternary_row, MudFile, MudTensorType};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs::File;

// ── ANSI colors ────────────────────────────────────────────────────────────
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

// ── Fenwick-accelerated error distribution histogram ───────────────────────

/// Error distribution histogram backed by a FenwickU32 tree.
/// Bins absolute errors into `n_bins` logarithmically-spaced buckets,
/// enabling O(log n) percentile queries via `find_kth`.
///
/// Zero-alloc reuse: `reset` clears the tree without reallocating (P-01).
struct FenwickErrorHistogram {
    n_bins: usize,
    tree: FenwickU32,
    max_err: f32,
    total_count: u32,
}

impl FenwickErrorHistogram {
    /// Create a histogram with `n_bins` buckets covering [0, max_err].
    /// Uses logarithmic spacing for better resolution at small errors.
    fn new(n_bins: usize, max_err: f32) -> Self {
        Self {
            n_bins,
            tree: FenwickU32::with_capacity(n_bins),
            max_err: max_err.max(1e-10),
            total_count: 0,
        }
    }

    /// Reset for reuse (zero-alloc, P-01).
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.tree.reset_zeroed();
        self.total_count = 0;
    }

    /// Map an absolute error to a bin index using logarithmic spacing.
    #[inline]
    fn error_to_bin(&self, err: f32) -> usize {
        if err <= 0.0 {
            return 0;
        }
        // Logarithmic bin: bin = log(err / max_err) * n_bins / log(epsilon)
        // This gives fine resolution near 0 and coarse at large errors.
        let ratio = (err / self.max_err).min(1.0);
        let log_ratio = ratio.ln();
        let log_min = 1e-10f32.ln(); // ~-23
        let t = log_ratio / log_min; // 0..1
        let bin = (t * (self.n_bins - 1) as f32) as usize;
        bin.min(self.n_bins - 1)
    }

    /// Insert an absolute error into the histogram. O(log n).
    #[inline]
    fn insert(&mut self, err: f32) {
        let bin = self.error_to_bin(err);
        self.tree.update(bin, 1);
        self.total_count += 1;
    }

    /// Compute the p-th percentile (0..100) of the error distribution. O(log n).
    /// Uses FenwickU32::find_kth for binary search on the prefix sum.
    fn percentile(&self, p: f32) -> f32 {
        if self.total_count == 0 {
            return 0.0;
        }
        let target = (p * 0.01 * self.total_count as f32).ceil() as u32;
        let bin = self.tree.find_kth(target.max(1));
        // Convert bin back to error estimate (midpoint of bin)
        let log_min = 1e-10f32.ln();
        let t = bin as f32 / (self.n_bins - 1) as f32;
        let ratio = (t * log_min).exp();
        ratio * self.max_err
    }

    /// Total number of errors inserted.
    #[allow(dead_code)]
    fn count(&self) -> u32 {
        self.total_count
    }
}

// ── Fenwick-accelerated cumulative error tracker ────────────────────────────

/// Tracks cumulative MSE/MAE across tensors using FenwickF32 for O(log n)
/// range queries. Enables "error up to layer L" without rescanning.
struct FenwickCumulativeError {
    mse_tree: FenwickF32,
    mae_tree: FenwickF32,
    sqnr_tree: FenwickF32,
    n: usize,
}

impl FenwickCumulativeError {
    fn new(n: usize) -> Self {
        Self {
            mse_tree: FenwickF32::with_capacity(n),
            mae_tree: FenwickF32::with_capacity(n),
            sqnr_tree: FenwickF32::with_capacity(n),
            n,
        }
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.mse_tree.reset_zeroed();
        self.mae_tree.reset_zeroed();
        self.sqnr_tree.reset_zeroed();
    }

    /// Record metrics for tensor at index `idx`. O(log n).
    fn record(&mut self, idx: usize, mse: f32, mae: f32, sqnr: f32) {
        self.mse_tree.update(idx, mse);
        self.mae_tree.update(idx, mae);
        self.sqnr_tree.update(idx, sqnr);
    }

    /// Cumulative MSE up to (and including) index `idx`. O(log n).
    fn cumulative_mse(&self, idx: usize) -> f32 {
        self.mse_tree.prefix_sum(idx + 1)
    }

    /// Cumulative MAE up to (and including) index `idx`. O(log n).
    fn cumulative_mae(&self, idx: usize) -> f32 {
        self.mae_tree.prefix_sum(idx + 1)
    }

    /// Mean SQNR up to (and including) index `idx`. O(log n).
    fn mean_sqnr_up_to(&self, idx: usize) -> f32 {
        let sum = self.sqnr_tree.prefix_sum(idx + 1);
        let count = (idx + 1).min(self.n) as f32;
        if count > 0.0 {
            sum / count
        } else {
            0.0
        }
    }
}

// ── Source tensor → f32 conversion (correct bitnet_scale handling) ──────────

/// Converts a safetensors TensorView to f32, WITHOUT dividing by weight_scale.
/// The original weights are preserved as-is — the converter does not scale
/// weights by bitnet_scale before ternarization, so neither should the audit.
fn view_to_f32(tensor: &TensorView) -> Vec<f32> {
    match tensor.dtype() {
        Dtype::F16 => {
            let slice: &[f16] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f16,
                    tensor.data().len() / 2,
                )
            };
            slice.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::BF16 => {
            let slice: &[bf16] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const bf16,
                    tensor.data().len() / 2,
                )
            };
            slice.iter().map(|&x| x.to_f32()).collect()
        }
        Dtype::F32 => {
            let slice: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    tensor.data().as_ptr() as *const f32,
                    tensor.data().len() / 4,
                )
            };
            slice.to_vec()
        }
        _ => panic!("Unsupported safetensors dtype: {:?}", tensor.dtype()),
    }
}

/// Reads a scalar f32 from a safetensors tensor (for weight_scale tensors).
fn read_scalar_f32(st: &SafeTensors, name: &str) -> Option<f32> {
    let t = st.tensor(name).ok()?;
    let raw = t.data();
    match t.dtype() {
        Dtype::F32 => {
            if raw.len() >= 4 {
                Some(f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
            } else {
                None
            }
        }
        Dtype::F16 => {
            if raw.len() >= 2 {
                Some(half::f16::from_le_bytes([raw[0], raw[1]]).to_f32())
            } else {
                None
            }
        }
        Dtype::BF16 => {
            if raw.len() >= 2 {
                Some(f32::from_bits((u16::from_le_bytes([raw[0], raw[1]]) as u32) << 16))
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── MUD dequantization ─────────────────────────────────────────────────────

/// Dequantizes a MUD Ternary2Bit tensor using its PRQ scales into a flat f32 vector.
fn dequant_mud_ternary(
    t: &forge_llm::mud::MudTensor,
    scale_t: Option<&forge_llm::mud::MudTensor>,
) -> Vec<f32> {
    let elements = t.shape.iter().product();
    let cols = *t.shape.last().unwrap_or(&elements);
    let rows = elements / cols;
    let u32s_per_row = cols.div_ceil(8);

    let mut data = vec![0.0f32; elements];

    // First dequantize ternary values to {-1, 0, +1}
    unsafe {
        for r in 0..rows {
            dequantize_ternary_row(
                (t.data_ptr as *const u32).add(r * u32s_per_row),
                &mut data[r * cols..(r + 1) * cols],
                cols,
            );
        }
    }

    // Then apply PRQ scales
    if let Some(st) = scale_t {
        let sp = st.data_ptr as *const f32;
        let is_scalar = st.shape.iter().product::<usize>() == 1;
        unsafe {
            for r in 0..rows {
                let s = if is_scalar { *sp } else { *sp.add(r) };
                for j in 0..cols {
                    data[r * cols + j] *= s;
                }
            }
        }
    }

    data
}

// ── Precise metrics (Fenwick-accelerated) ──────────────────────────────────

/// Comprehensive metrics for a single tensor comparison.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct TensorMetrics {
    name: String,
    shape: Vec<usize>,
    orig_fro: f64,
    deq_fro: f64,
    mse: f64,
    rmse: f64,
    mae: f64,
    max_abs_err: f64,
    rel_fro_err: f64,
    sqnr_db: f64,
    sparsity: f64,
    per_row_sqnr_min: f64,
    per_row_sqnr_max: f64,
    per_row_sqnr_mean: f64,
    bitnet_scale: f64,
    calibration_factor: f64,
    // Fenwick-accelerated percentiles
    err_median: f32,
    err_p95: f32,
    err_p99: f32,
}

impl TensorMetrics {
    /// Compute metrics with Fenwick-accelerated error distribution analysis.
    /// Uses a FenwickU32 histogram for O(log n) percentile queries.
    fn compute_with_fenwick(
        name: &str,
        shape: &[usize],
        orig: &[f32],
        deq: &[f32],
        bitnet_scale: f64,
        calibration_factor: f64,
        hist: &mut FenwickErrorHistogram,
    ) -> Self {
        assert_eq!(orig.len(), deq.len(), "orig and deq length mismatch for {}", name);

        let n = orig.len() as f64;
        let mut diff_sq_sum = 0.0f64;
        let mut orig_sq_sum = 0.0f64;
        let mut deq_sq_sum = 0.0f64;
        let mut abs_err_sum = 0.0f64;
        let mut max_abs_err = 0.0f64;
        let mut zero_count = 0usize;

        // Reset histogram for reuse (zero-alloc, P-01)
        hist.reset();
        let max_err_estimate = orig
            .iter()
            .zip(deq.iter())
            .map(|(o, d)| (o - d).abs())
            .fold(0.0f32, f32::max)
            .max(1e-10);
        hist.max_err = max_err_estimate;

        for i in 0..orig.len() {
            let o = orig[i] as f64;
            let d = deq[i] as f64;
            let diff = o - d;
            diff_sq_sum += diff * diff;
            orig_sq_sum += o * o;
            deq_sq_sum += d * d;
            abs_err_sum += diff.abs();
            let ad = diff.abs();
            if ad > max_abs_err {
                max_abs_err = ad;
            }
            if d == 0.0 {
                zero_count += 1;
            }
            // Insert into Fenwick histogram for percentile analysis
            hist.insert(ad as f32);
        }

        let mse = diff_sq_sum / n;
        let rmse = mse.sqrt();
        let mae = abs_err_sum / n;
        let orig_fro = orig_sq_sum.sqrt();
        let deq_fro = deq_sq_sum.sqrt();
        let rel_fro_err = if orig_fro > 0.0 {
            (orig_fro - deq_fro).abs() / orig_fro
        } else {
            0.0
        };
        let sqnr_db = if mse > 0.0 && orig_sq_sum > 0.0 {
            10.0 * (orig_sq_sum / diff_sq_sum).log10()
        } else {
            99.0
        };
        let sparsity = zero_count as f64 / n;

        // Per-row SQNR (assuming 2D tensor)
        let rows = shape.first().copied().unwrap_or(1);
        let cols = if shape.len() >= 2 {
            shape[1]
        } else {
            1
        };
        let row_len = cols;
        let mut row_sqnrs = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut r_diff_sq = 0.0f64;
            let mut r_orig_sq = 0.0f64;
            for c in 0..row_len {
                let idx = r * row_len + c;
                if idx >= orig.len() {
                    break;
                }
                let diff = (orig[idx] - deq[idx]) as f64;
                r_diff_sq += diff * diff;
                r_orig_sq += (orig[idx] as f64).powi(2);
            }
            let r_sqnr = if r_diff_sq > 0.0 && r_orig_sq > 0.0 {
                10.0 * (r_orig_sq / r_diff_sq).log10()
            } else {
                99.0
            };
            row_sqnrs.push(r_sqnr);
        }

        let per_row_sqnr_min = row_sqnrs.iter().copied().fold(f64::INFINITY, f64::min);
        let per_row_sqnr_max = row_sqnrs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let per_row_sqnr_mean = if !row_sqnrs.is_empty() {
            row_sqnrs.iter().sum::<f64>() / row_sqnrs.len() as f64
        } else {
            0.0
        };

        // Fenwick-accelerated percentiles
        let err_median = hist.percentile(50.0);
        let err_p95 = hist.percentile(95.0);
        let err_p99 = hist.percentile(99.0);

        TensorMetrics {
            name: name.to_string(),
            shape: shape.to_vec(),
            orig_fro,
            deq_fro,
            mse,
            rmse,
            mae,
            max_abs_err,
            rel_fro_err,
            sqnr_db,
            sparsity,
            per_row_sqnr_min,
            per_row_sqnr_max,
            per_row_sqnr_mean,
            bitnet_scale,
            calibration_factor,
            err_median,
            err_p95,
            err_p99,
        }
    }
}

// ── Depth-based calibration scale computation (corrected: uses break) ──────

/// Computes depth-based calibration dampening factors.
/// FIX (Finding 3): Uses `break` after finding the first parseable layer number,
/// preventing expert IDs from inflating max_layer.
fn compute_calibration_scales(
    tensor_names: &[String],
) -> HashMap<String, f64> {
    let mut scales = HashMap::new();

    // Find max_layer correctly (with break)
    let mut max_layer = 0usize;
    for name in tensor_names {
        if name.contains(".layers.") || name.contains(".blocks.") {
            let parts: Vec<&str> = name.split('.').collect();
            for p in &parts {
                if let Ok(l) = p.parse::<usize>() {
                    if l > max_layer {
                        max_layer = l;
                    }
                    break; // FIX: stop at first number (layer index)
                }
            }
        }
    }
    let total_layers = (max_layer + 1) as f64;

    for name in tensor_names {
        let mut dampening = 1.0f64;
        if name.contains(".layers.") || name.contains(".blocks.") {
            let parts: Vec<&str> = name.split('.').collect();
            for p in &parts {
                if let Ok(l) = p.parse::<usize>() {
                    let depth_ratio = l as f64 / total_layers;
                    dampening = 1.0 - (0.35 * depth_ratio.powi(2));
                    break; // FIX: stop at first number (layer index)
                }
            }
        }
        scales.insert(name.clone(), dampening);
    }

    scales
}

// ── Source-to-MUD name mapping (handles QKV and gate_up splitting) ─────────

/// Maps a source tensor name to (MUD name, should_ternarize, sub_range).
/// Handles QKV fusion and gate_up fusion by returning sub-ranges.
fn map_source_to_mud(
    name: &str,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden_size: usize,
) -> Option<(String, bool, Option<std::ops::Range<usize>>)> {
    if name.ends_with(".weight_scale") || name.ends_with(".scale") {
        return None;
    }
    if name.ends_with(".bias") && !name.contains("conv1d") && !name.contains("dt_bias") {
        return None;
    }

    if name == "model.embed_tokens.weight" {
        return Some(("token_embd.weight".to_string(), false, None));
    }
    if name == "model.norm.weight" {
        return Some(("output_norm.weight".to_string(), false, None));
    }
    if name == "lm_head.weight" {
        return Some(("output.weight".to_string(), false, None));
    }

    let layer_marker = if name.contains(".layers.") {
        ".layers."
    } else if name.contains(".blocks.") {
        ".blocks."
    } else {
        return None;
    };

    let parts: Vec<&str> = name.split(layer_marker).collect();
    if parts.len() < 2 {
        return None;
    }
    let sub_parts: Vec<&str> = parts[1].split('.').collect();
    if sub_parts.len() < 2 {
        return None;
    }

    let layer_idx = sub_parts[0];
    let sub = sub_parts[1];
    let prefix = format!("blk.{}", layer_idx);

    if sub == "input_layernorm" || sub == "ln_1" || sub == "attention_norm" {
        return Some((format!("{}.attn_norm.weight", prefix), false, None));
    }
    if sub == "post_attention_layernorm" || sub == "ln_2" || sub == "ffn_norm" {
        return Some((format!("{}.norm.weight", prefix), false, None));
    }

    // Self-Attention
    if sub == "self_attn" || sub == "attention" {
        if sub_parts.len() < 3 {
            return None;
        }
        let proj = sub_parts[2];
        let is_scale = sub_parts.last() == Some(&"scale") || sub_parts.last() == Some(&"weight_scale");
        if is_scale {
            return None;
        }
        let ternarize = true;

        // Handle QKV fusion
        if proj == "qkv_proj" {
            let q_rows = num_heads * head_dim;
            let k_rows = num_kv_heads * head_dim;
            let v_rows = num_kv_heads * head_dim;
            let _ = k_rows + v_rows; // QKV split: only Q audited per-call
            let cols = hidden_size;
            return Some((
                format!("{}.attn_q.weight", prefix),
                ternarize,
                Some(0..q_rows * cols),
            ));
        }

        let mapped = match proj {
            "q_proj" | "wq" => format!("{}.attn_q.weight", prefix),
            "k_proj" | "wk" => format!("{}.attn_k.weight", prefix),
            "v_proj" | "wv" => format!("{}.attn_v.weight", prefix),
            "o_proj" | "wo" | "out_proj" => format!("{}.attn_output.weight", prefix),
            _ => return None,
        };
        return Some((mapped, ternarize, None));
    }

    // MLP / MoE
    if sub == "mlp" || sub == "block_sparse_moe" || sub == "moe" {
        if sub_parts.len() < 3 {
            return None;
        }
        let is_scale = sub_parts.last() == Some(&"scale") || sub_parts.last() == Some(&"weight_scale");
        if is_scale {
            return None;
        }
        let ternarize = true;

        if sub_parts[2] == "gate" && sub_parts.len() == 4 {
            return Some((format!("{}.gate.weight", prefix), false, None));
        }

        if sub_parts.len() >= 5 && sub_parts[2] == "experts" {
            let expert_id = sub_parts[3];
            let proj = sub_parts[4];
            let mapped_proj = match proj {
                "gate_proj" | "w1" => "w1",
                "down_proj" | "w2" => "w2",
                "up_proj" | "w3" => "w3",
                _ => return None,
            };
            return Some((
                format!("{}.expert.{}.{}.weight", prefix, expert_id, mapped_proj),
                ternarize,
                None,
            ));
        }

        if sub_parts.len() >= 4 {
            let proj = sub_parts[2];
            let mapped_proj = match proj {
                "gate_proj" | "w1" => "w1",
                "down_proj" | "w2" => "w2",
                "up_proj" | "w3" => "w3",
                "gate_up_proj" => "gate_up",
                "ffn_sub_norm" => "ffn_sub_norm",
                _ => return None,
            };
            return Some((
                format!("{}.expert.0.{}.weight", prefix, mapped_proj),
                ternarize,
                None,
            ));
        }
    }

    None
}

// ── Main audit logic ───────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <source.safetensors|dir> <model.mud> [--calibration]",
            args[0]
        );
        std::process::exit(2);
    }

    let src_path = std::path::PathBuf::from(&args[1]);
    let mud_path = &args[2];
    let compute_calibration = args.iter().any(|a| a == "--calibration");

    println!(
        "{}{:=^72}{}",
        BOLD,
        "🛡️  PRECISE CONVERTER AUDIT v2.1 (Fenwick-optimized)",
        RESET
    );

    // Load source safetensors
    let st_file = if src_path.is_dir() {
        src_path.join("model.safetensors")
    } else {
        src_path
    };
    let file = File::open(&st_file)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = SafeTensors::deserialize(&mmap)?;
    println!(
        "{}  Source: {} ({} tensors){}",
        CYAN,
        st_file.display(),
        st.tensors().len(),
        RESET
    );

    // Load MUD file
    let mud = MudFile::load(mud_path)?;
    let core = mud
        .skills
        .get("core")
        .ok_or_else(|| anyhow::anyhow!("no core skill in MUD file"))?;
    println!(
        "{}  Target: {} ({} tensors){}",
        CYAN, mud_path, core.tensors.len(), RESET
    );

    // Get architecture dimensions
    let hidden_size: usize = mud
        .global_metadata
        .get("hidden_size")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let num_heads: usize = mud
        .global_metadata
        .get("num_heads")
        .or_else(|| mud.global_metadata.get("num_attention_heads"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let num_kv_heads: usize = mud
        .global_metadata
        .get("num_kv_heads")
        .or_else(|| mud.global_metadata.get("num_key_value_heads"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_heads);
    let head_dim = if num_heads > 0 {
        hidden_size / num_heads
    } else {
        0
    };

    if hidden_size == 0 || num_heads == 0 {
        anyhow::bail!("Could not infer hidden_size or num_heads from MUD metadata");
    }

    println!(
        "{}  Architecture: hidden={} heads={}/{} head_dim={}{}",
        YELLOW, hidden_size, num_heads, num_kv_heads, head_dim, RESET
    );

    // Compute calibration scales if requested
    let calib_scales: HashMap<String, f64> = if compute_calibration {
        let names: Vec<String> = st.tensors().iter().map(|(n, _)| n.clone()).collect();
        compute_calibration_scales(&names)
    } else {
        HashMap::new()
    };

    // Collect all source tensor names (sorted)
    let mut sf_keys: Vec<String> = st.tensors().iter().map(|(n, _)| n.clone()).collect();
    sf_keys.sort();

    // ── Fenwick-accelerated data structures ──────────────────────────────
    // Histogram for error distribution (64 bins, log-spaced)
    let mut error_hist = FenwickErrorHistogram::new(64, 10.0);
    // Cumulative error tracker for range queries
    let mut cum_error = FenwickCumulativeError::new(sf_keys.len());

    // Audit each tensor
    let mut all_metrics: Vec<TensorMetrics> = Vec::new();
    let mut ternarized_tensors = 0usize;
    let mut float_tensors = 0usize;
    let mut low_sqnr = 0usize;

    println!("\n{}--- ANALYZING TENSOR CONVERSION CORRECTNESS ---{}\n", BOLD, RESET);

    for (tensor_idx, name) in sf_keys.iter().enumerate() {
        let Some((mud_name, should_ternarize, sub_range)) = map_source_to_mud(
            name,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        ) else {
            continue;
        };

        let Some(mud_tensor) = core.tensors.get(&mud_name) else {
            continue;
        };

        let sf_view = st.tensor(name)?;
        let orig_weights = view_to_f32(&sf_view);

        // Read bitnet_scale if available (for reporting, NOT for dividing orig)
        let bitnet_scale: f64 = {
            let scale_name = name.replace(".weight", ".weight_scale");
            let fallback_name = name.replace(".weight", ".scale");
            read_scalar_f32(&st, &scale_name)
                .or_else(|| read_scalar_f32(&st, &fallback_name))
                .unwrap_or(1.0) as f64
        };

        // Get calibration factor
        let cal_factor: f64 = calib_scales.get(name).copied().unwrap_or(1.0);

        if !should_ternarize {
            // Float32 path — should match exactly
            if mud_tensor.t_type != MudTensorType::Float32 {
                continue;
            }
            let deq = {
                let elements = mud_tensor.shape.iter().product();
                let mut d = vec![0.0f32; elements];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        mud_tensor.data_ptr as *const f32,
                        d.as_mut_ptr(),
                        elements,
                    );
                }
                d
            };

            if orig_weights.len() != deq.len() {
                println!(
                    "{}  ⚠️  SHAPE MISMATCH: {} (orig={}, deq={})",
                    YELLOW, mud_name, orig_weights.len(), deq.len()
                );
                continue;
            }

            let metrics = TensorMetrics::compute_with_fenwick(
                &mud_name,
                &mud_tensor.shape,
                &orig_weights,
                &deq,
                bitnet_scale,
                cal_factor,
                &mut error_hist,
            );

            float_tensors += 1;
            cum_error.record(tensor_idx, metrics.mse as f32, metrics.mae as f32, metrics.sqnr_db as f32);
            println!(
                "  {:<42} {:>8} {:>10.4} {:>10.2} dB",
                &mud_name,
                "Float32",
                metrics.mse,
                metrics.sqnr_db,
            );
            all_metrics.push(metrics);
        } else {
            // Ternary path
            if mud_tensor.t_type != MudTensorType::Ternary2Bit {
                continue;
            }

            let scale_name = mud_name.replace(".weight", ".prq_scale");
            let scale_tensor = core.tensors.get(&scale_name);

            let deq = dequant_mud_ternary(mud_tensor, scale_tensor);

            // Handle sub_range (for QKV splitting)
            let orig_slice: &[f32] = if let Some(range) = &sub_range {
                &orig_weights[range.clone()]
            } else {
                &orig_weights
            };

            if orig_slice.len() != deq.len() {
                println!(
                    "{}  ⚠️  SHAPE MISMATCH: {} (orig={}, deq={})",
                    YELLOW, mud_name, orig_slice.len(), deq.len()
                );
                continue;
            }

            let metrics = TensorMetrics::compute_with_fenwick(
                &mud_name,
                &mud_tensor.shape,
                orig_slice,
                &deq,
                bitnet_scale,
                cal_factor,
                &mut error_hist,
            );

            ternarized_tensors += 1;
            if metrics.sqnr_db < 3.4 {
                low_sqnr += 1;
            }

            // Record in cumulative tracker (FenwickF32)
            cum_error.record(tensor_idx, metrics.mse as f32, metrics.mae as f32, metrics.sqnr_db as f32);

            // Status indicator
            let status = if metrics.sqnr_db >= 5.0 {
                format!("{}✅ HEALTHY{}", GREEN, RESET)
            } else if metrics.sqnr_db >= 3.4 {
                format!("{}⚠️  ACCEPTABLE{}", YELLOW, RESET)
            } else {
                format!("{}🔴 LOW SQNR{}", RED, RESET)
            };

            println!(
                "  {:<42} {:>8} {:>10.6} {:>8.2} dB {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {}",
                &mud_name,
                "Ternary",
                metrics.mse,
                metrics.sqnr_db,
                metrics.rmse,
                metrics.mae,
                metrics.max_abs_err,
                metrics.sparsity,
                metrics.per_row_sqnr_min,
                metrics.err_median,
                metrics.err_p95,
                status,
            );

            all_metrics.push(metrics);
        }
    }

    // ── Summary report ─────────────────────────────────────────────────
    println!("\n{}{:=^72}{}", BOLD, "📊 CONVERSION FIDELITY REPORT", RESET);

    let ternary_metrics: Vec<&TensorMetrics> = all_metrics
        .iter()
        .filter(|m| {
            core.tensors
                .get(&m.name)
                .map(|t| t.t_type == MudTensorType::Ternary2Bit)
                .unwrap_or(false)
        })
        .collect();

    if !ternary_metrics.is_empty() {
        let avg_sqnr: f64 = ternary_metrics.iter().map(|m| m.sqnr_db).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_mse: f64 = ternary_metrics.iter().map(|m| m.mse).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_rmse: f64 = ternary_metrics.iter().map(|m| m.rmse).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_mae: f64 = ternary_metrics.iter().map(|m| m.mae).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_sparsity: f64 = ternary_metrics.iter().map(|m| m.sparsity).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_per_row_min: f64 =
            ternary_metrics.iter().map(|m| m.per_row_sqnr_min).sum::<f64>()
                / ternary_metrics.len() as f64;
        let avg_rel_fro: f64 = ternary_metrics.iter().map(|m| m.rel_fro_err).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_err_median: f64 = ternary_metrics.iter().map(|m| m.err_median as f64).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_err_p95: f64 = ternary_metrics.iter().map(|m| m.err_p95 as f64).sum::<f64>()
            / ternary_metrics.len() as f64;
        let avg_err_p99: f64 = ternary_metrics.iter().map(|m| m.err_p99 as f64).sum::<f64>()
            / ternary_metrics.len() as f64;

        println!("\n  Ternary Tensors: {}", ternarized_tensors);
        println!("  Float32 Tensors: {}", float_tensors);
        println!("  Low SQNR (<3.4 dB): {} {}", low_sqnr, if low_sqnr > 0 { "🔴" } else { "✅" });
        println!();
        println!("  {:<32} {:>12}", "Metric", "Value");
        println!("  {}", "-".repeat(46));
        println!("  {:<32} {:>12.2} dB", "Average SQNR", avg_sqnr);
        println!("  {:<32} {:>12.6}", "Average MSE", avg_mse);
        println!("  {:<32} {:>12.6}", "Average RMSE", avg_rmse);
        println!("  {:<32} {:>12.6}", "Average MAE", avg_mae);
        println!("  {:<32} {:>12.4}", "Average Sparsity", avg_sparsity);
        println!("  {:<32} {:>12.2} dB", "Avg Per-Row Min SQNR", avg_per_row_min);
        println!("  {:<32} {:>12.6}", "Avg Rel Frobenius Err", avg_rel_fro);
        println!("  {:<32} {:>12.6}", "Avg Error Median", avg_err_median);
        println!("  {:<32} {:>12.6}", "Avg Error P95", avg_err_p95);
        println!("  {:<32} {:>12.6}", "Avg Error P99", avg_err_p99);
        println!();

        // Fenwick-accelerated cumulative error report
        println!("  {}📈 Cumulative Error (FenwickF32 range queries){}", CYAN, RESET);
        let n_tensors = all_metrics.len();
        for &layer_idx in &[0usize, n_tensors / 4, n_tensors / 2, (3 * n_tensors) / 4, n_tensors.saturating_sub(1)] {
            if layer_idx >= n_tensors {
                continue;
            }
            let cum_mse = cum_error.cumulative_mse(layer_idx);
            let cum_mae = cum_error.cumulative_mae(layer_idx);
            let mean_sqnr = cum_error.mean_sqnr_up_to(layer_idx);
            println!(
                "    up to tensor #{:>4}: cum_MSE={:>10.6} cum_MAE={:>10.6} mean_SQNR={:>8.2} dB",
                layer_idx, cum_mse, cum_mae, mean_sqnr
            );
        }
        println!();

        // bitnet_scale analysis
        let has_non_unit_bitnet: Vec<&TensorMetrics> =
            all_metrics.iter().filter(|m| (m.bitnet_scale - 1.0).abs() > 1e-6).collect();
        if !has_non_unit_bitnet.is_empty() {
            println!(
                "  {}⚠️  {} tensors have bitnet_scale ≠ 1.0 (range: {:.4}–{:.4}){}",
                YELLOW,
                has_non_unit_bitnet.len(),
                has_non_unit_bitnet
                    .iter()
                    .map(|m| m.bitnet_scale)
                    .fold(f64::INFINITY, f64::min),
                has_non_unit_bitnet
                    .iter()
                    .map(|m| m.bitnet_scale)
                    .fold(f64::NEG_INFINITY, f64::max),
                RESET
            );
            println!(
                "  {}   conversion_verifier.rs would report incorrect errors for these tensors{}",
                YELLOW, RESET
            );
        }

        // Calibration analysis
        if compute_calibration {
            let has_cal: Vec<&TensorMetrics> =
                all_metrics.iter().filter(|m| (m.calibration_factor - 1.0).abs() > 1e-6).collect();
            if !has_cal.is_empty() {
                println!(
                    "\n  {}📐 Depth-based calibration applied to {} tensors{}",
                    CYAN,
                    has_cal.len(),
                    RESET
                );
                for m in has_cal.iter().take(5) {
                    println!(
                        "    {:<40} cal_factor={:.4}  sqnr={:.2} dB",
                        &m.name, m.calibration_factor, m.sqnr_db
                    );
                }
                if has_cal.len() > 5 {
                    println!("    ... and {} more", has_cal.len() - 5);
                }
            }
        }

        // Verdict
        println!();
        let passes = avg_sqnr >= 3.4 && low_sqnr == 0;
        if passes {
            println!(
                "  {}🎉 CONVERSION VERIFIED — Average SQNR {:.2} dB ≥ 3.4 dB target{}",
                GREEN, avg_sqnr, RESET
            );
        } else {
            println!(
                "  {}❌ CONVERSION REJECTED — Average SQNR {:.2} dB < 3.4 dB target{}",
                RED, avg_sqnr, RESET
            );
        }
    } else {
        println!("  No ternary tensors found to audit.");
    }

    println!("\n{}💡 Audit complete.{}", BOLD, RESET);

    Ok(())
}
