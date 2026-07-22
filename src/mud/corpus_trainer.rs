#![allow(clippy::type_complexity)]
use crate::model::tokenizer::Tokenizer;
use crate::mud::{MudFile, MudTensorType};

use std::time::{Duration, Instant};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

macro_rules! println {
    () => {
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            std::println!();
        }
    };
    ($($arg:tt)*) => {
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            std::println!($($arg)*);
        }
    };
}

macro_rules! print {
    ($($arg:tt)*) => {
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            std::print!($($arg)*);
        }
    };
}

/// Live activation metrics sampled **after forward, before backward** (post-BWD kills VarH/VarJ).
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainChunkMetrics {
    pub loss: f32,
    pub var_h: f32,
    pub var_j: f32,
    pub jepa_integral: f32,
    pub sigma_v_pct: f32,
    pub cognitive: f32,
    pub n_samples: u32,
}

/// Running mean of per-step activation stats (Welford-lite / online mean).
#[derive(Debug, Default)]
struct ActStatsAccum {
    n: u32,
    var_h: f64,
    var_j: f64,
    jepa_i: f64,
    sigma: f64,
    cog: f64,
}

impl ActStatsAccum {
    fn push(&mut self, var_h: f32, var_j: f32, jepa_i: f32, sigma: f32, cog: f32) {
        self.n += 1;
        let n = self.n as f64;
        // online mean: m += (x - m) / n
        self.var_h += (var_h as f64 - self.var_h) / n;
        self.var_j += (var_j as f64 - self.var_j) / n;
        self.jepa_i += (jepa_i as f64 - self.jepa_i) / n;
        self.sigma += (sigma as f64 - self.sigma) / n;
        self.cog += (cog as f64 - self.cog) / n;
    }

    fn finish(self, loss: f32) -> TrainChunkMetrics {
        TrainChunkMetrics {
            loss,
            var_h: self.var_h as f32,
            var_j: self.var_j as f32,
            jepa_integral: self.jepa_i as f32,
            sigma_v_pct: self.sigma as f32,
            cognitive: self.cog as f32,
            n_samples: self.n,
        }
    }
}

/// Load one emb row: either from FP32 shadow or on-the-fly ELUT mmap (FREEZE_EMB / lazy).
fn load_emb_row_into(
    mud: &MudFile,
    shadow_emb: &[f32],
    emb_lazy: bool,
    hidden_size: usize,
    token_id: usize,
    out: &mut [f32],
) {
    if !emb_lazy {
        let off = token_id * hidden_size;
        if off + hidden_size <= shadow_emb.len() && out.len() >= hidden_size {
            out[..hidden_size].copy_from_slice(&shadow_emb[off..off + hidden_size]);
        }
        return;
    }
    let Some(core) = mud.skills.get("core") else {
        return;
    };
    let Some(emb_t) = core.tensors.get("token_embd.weight") else {
        return;
    };
    let cols = emb_t.shape.get(1).copied().unwrap_or(hidden_size);
    let rows = emb_t.shape.first().copied().unwrap_or(0);
    if token_id >= rows || out.len() < cols {
        return;
    }
    unsafe {
        if emb_t.t_type == MudTensorType::Ternary2Bit {
            let u32s = cols.div_ceil(8);
            crate::mud::dequantize_ternary_row(
                (emb_t.data_ptr as *const u32).add(token_id * u32s),
                &mut out[..cols],
                cols,
            );
            if let Some(sc) = core.tensors.get("token_embd.prq_scale") {
                if sc.t_type == MudTensorType::Float32 {
                    let s = *(sc.data_ptr as *const f32).add(token_id);
                    for v in out.iter_mut().take(cols) {
                        *v *= s;
                    }
                }
            }
        } else {
            std::ptr::copy_nonoverlapping(
                (emb_t.data_ptr as *const f32).add(token_id * cols),
                out.as_mut_ptr(),
                cols,
            );
        }
    }
}

/// Stddev of a f32 slice (population). Returns 0 if empty/flat.
#[inline]
fn slice_stddev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &x in xs {
        let v = x as f64;
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    var.sqrt() as f32
}

/// Tamaño de chunk en caracteres para procesamiento del corpus.
pub const CHUNK_SIZE: usize = 50_000;
/// Default hard-checkpoint interval (was 1 = save 100MB every chunk — catastrophic for align).
/// Override: `MUD_TRAIN_CKPT_EVERY=N` (0 = only epoch/end).
fn checkpoint_every_chunks() -> usize {
    std::env::var("MUD_TRAIN_CKPT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
}
/// Directorio donde se guardan los checkpoints.
const CHECKPOINT_DIR: &str = "weights/checkpoints";

pub static SHOULD_TERMINATE: AtomicBool = AtomicBool::new(false);

/// Implements a high-performance local corpus trainer for MUD.
pub struct MudCorpusTrainer {
    pub model_path: String,
    pub corpus_dir: String,
    pub tokenizer: Arc<Tokenizer>,
    pub vk: Option<std::sync::Arc<crate::vulkan::ash_backend::AshContext>>,
}

impl MudCorpusTrainer {
    pub fn new(model_path: String, corpus_dir: String) -> anyhow::Result<Self> {
        let mud = MudFile::load(&model_path)?;
        Self::validate_metadata(&mud)?;

        let tokens_str = mud
            .global_metadata
            .get("tokenizer.tokens")
            .ok_or_else(|| anyhow::anyhow!("No tokens"))?;
        let merges_str = mud
            .global_metadata
            .get("tokenizer.merges")
            .map(|s| s.as_str())
            .unwrap_or("");
        let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

        let vk = crate::vulkan::ash_backend::AshContext::new()
            .map(Arc::new)
            .ok();

        let trainer = Self {
            model_path,
            corpus_dir,
            tokenizer: Arc::new(tokenizer),
            vk,
        };
        trainer.audit_tokenization();
        Ok(trainer)
    }

    fn validate_metadata(mud: &MudFile) -> anyhow::Result<()> {
        println!(
            "{}",
            crate::mud::trainer_ui::phase("Phase 0", "Metadata Integrity Validation (P-13 / L-12)")
        );
        // L-12: shared fail-fast parser (alternate key names, consistency, ELUT multiples)
        let dims = crate::mud::p13::parse_arch_dims(mud)?;
        crate::mud::p13::validate_trainer_required_keys(&mud.global_metadata)?;
        println!(
            "{}",
            crate::mud::trainer_ui::note(
                "ok",
                &format!(
                    "arch hidden={} layers={} heads={}/{} ffn={}",
                    dims.hidden_size,
                    dims.num_layers,
                    dims.num_heads,
                    dims.num_kv_heads,
                    dims.intermediate_size
                )
            )
        );
        if let Some(core) = mud.skills.get("core") {
            let mut ternary_count = 0;
            let mut scale_count = 0;
            for (name, tensor) in &core.tensors {
                if tensor.t_type == MudTensorType::Ternary2Bit {
                    ternary_count += 1;
                }
                if name.ends_with(".prq_scale") {
                    scale_count += 1;
                }
            }
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!("found {ternary_count} ternary weights and {scale_count} scales")
                )
            );
        }
        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "metadata validated successfully (P-13)")
        );
        Ok(())
    }

    #[allow(unreachable_code, unused_variables, unused_mut, unused_assignments)]
    fn deep_local_alignment(&self, mud: &mut MudFile, print_stdout: bool) -> anyhow::Result<()> {
        if print_stdout {
            println!(
                "{}",
                crate::mud::trainer_ui::phase(
                    "AWAKE-01",
                    "Universal Agnostic Deep Local Alignment (L-QAT)"
                )
            );
        }

        let learning_rate = 0.001f32;
        // Post-convert: more L-QAT iters unless quick smoke (MAX_CHUNKS without explicit override)
        let ldt_iterations: usize = std::env::var("MUD_AWAKE_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                let quick = std::env::var("MUD_TRAIN_MAX_CHUNKS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|&m| m > 0 && m <= 16)
                    .is_some();
                if quick {
                    4
                } else {
                    12
                }
            });
        let weight_decay = 0.0001f32;

        let mut aligned_count = 0;
        let mut total_ternary = 0;

        for skill in mud.skills.values() {
            total_ternary += skill
                .tensors
                .values()
                .filter(|t| t.t_type == MudTensorType::Ternary2Bit)
                .count();
        }

        // We iterate over the keys to avoid mutable borrow conflicts
        for (skill_name, skill) in mud.skills.iter_mut() {
            let ternary_keys: Vec<String> = skill
                .tensors
                .iter()
                .filter(|(_, t)| t.t_type == MudTensorType::Ternary2Bit && t.shape.len() == 2)
                .map(|(k, _)| -> String { k.clone() })
                .collect();

            for t_name in ternary_keys {
                if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                    break;
                }

                let (rows, cols) = {
                    let t = skill.tensors.get(&t_name).unwrap();
                    (t.shape[0], t.shape[1])
                };
                let elements = rows * cols;

                let scale_name = t_name.replace(".weight", ".prq_scale");
                let mut scales = vec![1.0f32; rows];
                if let Some(scale_tensor) = skill.tensors.get(&scale_name) {
                    if scale_tensor.t_type == MudTensorType::Float32
                        && scale_tensor.shape[0] == rows
                    {
                        unsafe {
                            let ptr = scale_tensor.data_ptr as *const f32;
                            std::ptr::copy_nonoverlapping(ptr, scales.as_mut_ptr(), rows);
                        }
                    }
                }

                // 1. Dequantize to FP32 shadow weights
                let mut w_fp32 = vec![0.0f32; elements];
                let mut use_vulkan = false;
                if let Some(_vk) = &self.vk {
                    use_vulkan = true;
                    // TODO: Dispatch to Vulkan QAT here
                    // let shadow_w = vk.allocate_zero_copy_buffer(elements);
                    // vk.run_qat_optimizer_async(...);
                }

                if use_vulkan {
                    // Fast path fallback for now while hooking up buffers
                }
                unsafe {
                    let t = skill.tensors.get(&t_name).unwrap();
                    for r in 0..rows {
                        crate::mud::dequantize_ternary_row(
                            (t.data_ptr as *const u32).add(r * cols / 8),
                            &mut w_fp32[r * cols..(r + 1) * cols],
                            cols,
                        );
                        let s = scales[r];
                        for c in 0..cols {
                            w_fp32[r * cols + c] *= s;
                        }
                    }
                }

                // 2. Perform L-QAT SGD iterations (AVX dots, reused scratch)
                let mut x = vec![0.0f32; cols];
                let mut w_q_row = vec![0.0f32; cols];
                for _iter in 0..ldt_iterations {
                    let mut rng_state = 1337u32.wrapping_add(_iter as u32 * 0x9E37);
                    #[allow(clippy::needless_range_loop)]
                    for c in 0..cols {
                        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                        x[c] = (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    }

                    for r in 0..rows {
                        let row_start = r * cols;
                        let row = &w_fp32[row_start..row_start + cols];

                        let mut absmean = 0.0f32;
                        for &w in row {
                            absmean += w.abs();
                        }
                        absmean /= cols as f32;
                        let scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);

                        for c in 0..cols {
                            w_q_row[c] = (row[c] / scale).round().clamp(-1.0, 1.0) * scale;
                        }
                        let y_master =
                            unsafe { forge_autograd::avx_math::dot_product_avx2(row, &x[..cols]) };
                        let y_student = unsafe {
                            forge_autograd::avx_math::dot_product_avx2(&w_q_row, &x[..cols])
                        };
                        let err = y_student - y_master;

                        // Apply SGD gradients & Weight Decay
                        for c in 0..cols {
                            let mut grad = err * x[c] / (cols as f32); // Normalize by cols
                            grad = grad.clamp(-10.0, 10.0); // Clip gradient
                            w_fp32[row_start + c] -=
                                learning_rate * grad + weight_decay * w_fp32[row_start + c];
                            if !w_fp32[row_start + c].is_finite() {
                                w_fp32[row_start + c] = 0.0;
                            }
                        }
                    }
                }

                // 3. Re-quantize and Pack back to Ternary2Bit
                let mut new_scales = vec![0.0f32; rows];
                let u32_count = elements.div_ceil(8);
                let mut packed = vec![0u32; u32_count];

                #[allow(clippy::needless_range_loop)]
                for r in 0..rows {
                    let row_start = r * cols;
                    let mut absmean = 0.0f32;
                    for c in 0..cols {
                        absmean += w_fp32[row_start + c].abs();
                    }
                    absmean /= cols as f32;
                    let scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                    new_scales[r] = scale;

                    for c in 0..cols {
                        let idx = row_start + c;
                        let w_f = w_fp32[idx];
                        let w_q = (w_f / scale).round().clamp(-1.0, 1.0);
                        let bit = if w_q > 0.5 {
                            0x1u32
                        } else if w_q < -0.5 {
                            0xFu32
                        } else {
                            0x0u32
                        };
                        packed[idx / 8] |= bit << ((idx % 8) * 4);
                    }
                }

                // Update tensor data
                let packed_bytes = unsafe {
                    std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
                }
                .to_vec();
                if let Some(t) = skill.tensors.get_mut(&t_name) {
                    t.owned_data = Some(packed_bytes);
                    t.data_ptr = t.owned_data.as_ref().unwrap().as_ptr();
                }

                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(
                        new_scales.as_ptr() as *const u8,
                        new_scales.len() * 4,
                    )
                }
                .to_vec();
                if let Some(s_t) = skill.tensors.get_mut(&scale_name) {
                    s_t.owned_data = Some(scale_bytes);
                    s_t.data_ptr = s_t.owned_data.as_ref().unwrap().as_ptr();
                } else {
                    skill.tensors.insert(
                        scale_name.clone(),
                        crate::mud::MudTensor {
                            name: scale_name.clone(),
                            t_type: MudTensorType::Float32,
                            shape: vec![rows],
                            data_ptr: scale_bytes.as_ptr(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: Some(scale_bytes),
                        },
                    );
                    if let Some(s_t) = skill.tensors.get_mut(&scale_name) {
                        s_t.data_ptr = s_t.owned_data.as_ref().unwrap().as_ptr();
                    }
                }

                aligned_count += 1;
                if print_stdout {
                    print!(
                        "\r  \x1b[1;36m[L-QAT]\x1b[0m Aligned {}/{} tensors ({:.1}%)",
                        aligned_count,
                        total_ternary,
                        (aligned_count as f32 / total_ternary as f32) * 100.0
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            }
            // --- mHC Collapse Recovery (Priority 12 & 17) ---
            if print_stdout {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ok",
                        "auditing mHC residual collapse for skill..."
                    )
                );
            }
            // L-07 / P-13: skill metadata first, then global_metadata (converter often only sets global)
            let num_layers: usize = skill
                .metadata
                .get("num_hidden_layers")
                .or_else(|| skill.metadata.get("num_layers"))
                .or_else(|| mud.global_metadata.get("num_hidden_layers"))
                .or_else(|| mud.global_metadata.get("num_layers"))
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    panic!(
                        "P-13: missing num_hidden_layers/num_layers (skill={skill_name} and global)"
                    )
                });
            let hidden: usize = skill
                .metadata
                .get("hidden_size")
                .or_else(|| mud.global_metadata.get("hidden_size"))
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    panic!("P-13: missing hidden_size (skill={skill_name} and global)")
                });

            for l in 0..num_layers {
                let alpha_name = format!("blk.{}.mhc_alpha", l);
                let beta_name = format!("blk.{}.mhc_beta", l);
                let mut alpha_collapsed = false;

                if let Some(alpha_tensor) = skill.tensors.get(&alpha_name) {
                    if let Some(data) = &alpha_tensor.owned_data {
                        let alpha_slice: &[f32] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                        };
                        let mean_alpha = alpha_slice.iter().map(|v| v.abs()).sum::<f32>()
                            / (alpha_slice.len() as f32).max(1.0);
                        if mean_alpha < 0.1 {
                            alpha_collapsed = true;
                        }
                    } else {
                        alpha_collapsed = true;
                    }
                } else {
                    alpha_collapsed = true;
                }

                if alpha_collapsed {
                    let new_alpha = vec![0.85f32; hidden];
                    let new_beta = vec![0.15f32; hidden];

                    let alpha_bytes = unsafe {
                        std::slice::from_raw_parts(new_alpha.as_ptr() as *const u8, hidden * 4)
                    }
                    .to_vec();
                    let beta_bytes = unsafe {
                        std::slice::from_raw_parts(new_beta.as_ptr() as *const u8, hidden * 4)
                    }
                    .to_vec();

                    let t_alpha = skill.tensors.entry(alpha_name.clone()).or_insert_with(|| {
                        crate::mud::MudTensor {
                            name: alpha_name,
                            t_type: crate::mud::MudTensorType::Float32,
                            shape: vec![hidden],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: None,
                        }
                    });
                    t_alpha.owned_data = Some(alpha_bytes);
                    t_alpha.data_ptr = t_alpha.owned_data.as_ref().unwrap().as_ptr();

                    let t_beta = skill.tensors.entry(beta_name.clone()).or_insert_with(|| {
                        crate::mud::MudTensor {
                            name: beta_name,
                            t_type: crate::mud::MudTensorType::Float32,
                            shape: vec![hidden],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: None,
                        }
                    });
                    t_beta.owned_data = Some(beta_bytes);
                    t_beta.data_ptr = t_beta.owned_data.as_ref().unwrap().as_ptr();
                }

                let radius_name = format!("blk.{}.mhc_radius", l);
                let mut radius_collapsed = false;
                if let Some(radius_tensor) = skill.tensors.get(&radius_name) {
                    if let Some(data) = &radius_tensor.owned_data {
                        let rad_slice: &[f32] = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                        };
                        if rad_slice.is_empty() || rad_slice[0] < 5.0 {
                            radius_collapsed = true;
                        }
                    } else {
                        radius_collapsed = true;
                    }
                } else {
                    radius_collapsed = true;
                }

                if radius_collapsed {
                    let computed_max_emb = if let Some(emb) = skill.tensors.get("token_embd.weight")
                    {
                        if let Some(d) = &emb.owned_data {
                            let slice: &[f32] = unsafe {
                                std::slice::from_raw_parts(d.as_ptr() as *const f32, d.len() / 4)
                            };
                            slice
                                .iter()
                                .map(|v| v.abs())
                                .fold(0.0f32, |a: f32, b: f32| a.max(b))
                        } else {
                            128.0
                        }
                    } else {
                        128.0
                    };
                    let max_emb = computed_max_emb.max(5.0);

                    let safe_radius = max_emb * (hidden as f32).sqrt();
                    let new_radius = [safe_radius; 1];
                    let radius_bytes =
                        unsafe { std::slice::from_raw_parts(new_radius.as_ptr() as *const u8, 4) }
                            .to_vec();

                    let t_radius = skill.tensors.entry(radius_name.clone()).or_insert_with(|| {
                        crate::mud::MudTensor {
                            name: radius_name,
                            t_type: crate::mud::MudTensorType::Float32,
                            shape: vec![1],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            data_base: 0,
                            mmap: None,
                            owned_data: None,
                        }
                    });
                    t_radius.owned_data = Some(radius_bytes);
                    t_radius.data_ptr = t_radius.owned_data.as_ref().unwrap().as_ptr();
                }
            }
        } // end of mud.skills loop

        if print_stdout {
            println!(
                "{}",
                crate::mud::trainer_ui::note("ok", "AWAKE-01 alignment complete")
            );
        }
        Ok(())
    }

    fn audit_tokenization(&self) {
        println!(
            "{}",
            crate::mud::trainer_ui::phase("Phase 1", "Tokenization Sync Audit")
        );
        let test_phrases = [
            "MUD engine optimized.",
            "Inteligencia artificial.",
            "BPE Hello World!",
        ];
        for phrase in test_phrases {
            let ids = self.tokenizer.encode(phrase);
            let decoded = self.tokenizer.decode(&ids);
            println!(
                "   - original: \"{}\"  |  decoded: \"{}\"",
                phrase,
                decoded.trim()
            );
            println!("     numeric tokens: {:?}", ids);
        }
        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "tokenization audit complete")
        );
    }

    pub fn run_debate_session(
        &mut self,
        sender: Option<std::sync::mpsc::Sender<String>>,
        stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<()> {
        let print_stdout = sender.is_none();
        // Wrap the sender so EVERY arena/debate event is also persisted to
        // logs/circuit.log (timestamped) in addition to reaching any live TUI.
        let log_sender: Option<std::sync::mpsc::Sender<String>> = match &sender {
            Some(tx_out) => {
                let (tx_in, rx_in) = std::sync::mpsc::channel::<String>();
                let tx_out = tx_out.clone();
                std::thread::spawn(move || {
                    while let Ok(msg) = rx_in.recv() {
                        let _ = crate::mud::trainer_ui::circuit_event("arena", &msg);
                        if tx_out.send(msg).is_err() {
                            break;
                        }
                    }
                });
                Some(tx_in)
            }
            None => None,
        };
        if let Some(tx) = &log_sender {
            let _ = tx.send("⚔️ Starting MUD Debate Arena Session...".to_string());
        }
        if print_stdout {
            println!("⚔️ Starting MUD Debate Arena Session...");
        }
        let mut mud = MudFile::load(&self.model_path)?;
        self.deep_local_alignment(&mut mud, print_stdout)?;
        // Materialize ALL tensors into owned buffers so every `data_ptr` used
        // below (emb, norms, weights, scales) points at live backing memory.
        // Without this, a tensor left on a dropped mmap (or with `data_ptr`
        // reset by ecc_verify) yields a dangling/null pointer → reads as zeros
        // → "Dead RMSNorm" panic on the first forward (norm weights ~0).
        mud.materialize_writable();
        // Capture input weight hash (post deep_local_alignment) before the debate trains.
        let input_weights_hash = Self::hash_trained_weights(&mud);

        let hidden = mud
            .global_metadata
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing hidden_size");
        let n_layers = mud
            .global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_layers");
        let n_heads = mud
            .global_metadata
            .get("num_attention_heads")
            .or_else(|| mud.global_metadata.get("num_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_heads");
        let n_kv_heads = mud
            .global_metadata
            .get("num_key_value_heads")
            .or_else(|| mud.global_metadata.get("num_kv_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_kv_heads");
        let ffn_mid = mud
            .global_metadata
            .get("intermediate_size")
            .or_else(|| mud.global_metadata.get("ffn_hidden"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing ffn_mid");
        let max_pos = mud
            .global_metadata
            .get("max_position_embeddings")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing max_position_embeddings");
        let rope_theta: f32 = mud
            .global_metadata
            .get("rope_theta")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000.0);
        let core = mud
            .skills
            .get_mut("core")
            .ok_or_else(|| anyhow::anyhow!("No core skill"))?;
        let vocab_size = core
            .tensors
            .get("token_embd.weight")
            .map(|t| t.shape[0])
            .expect("Missing token_embd.weight");

        let mut emb = vec![0.0; vocab_size * hidden];
        let emb_tensor = core.tensors.get("token_embd.weight").unwrap();
        unsafe {
            let cols = hidden;
            if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                for r in 0..vocab_size {
                    crate::mud::dequantize_ternary_row(
                        (emb_tensor.data_ptr as *const u32).add(r * cols / 8),
                        &mut emb[r * cols..(r + 1) * cols],
                        cols,
                    );
                }
                if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                    if scale_tensor.t_type == MudTensorType::Float32 {
                        let scale_data = std::slice::from_raw_parts(
                            scale_tensor.data_ptr as *const f32,
                            vocab_size,
                        );
                        for r in 0..vocab_size {
                            let s = scale_data[r];
                            for c in 0..cols {
                                emb[r * cols + c] *= s;
                            }
                        }
                    }
                }
            } else {
                std::ptr::copy_nonoverlapping(
                    emb_tensor.data_ptr as *const f32,
                    emb.as_mut_ptr(),
                    vocab_size * hidden,
                );
            }
        }

        let computed_max_emb = emb.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
        let max_emb = mud
            .global_metadata
            .get("max_emb")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(computed_max_emb);

        let mut out_f32 = vec![0.0f32; vocab_size * hidden];
        let output_weight;
        if let Some(t) = core.tensors.get("output.weight") {
            unsafe {
                let cols = hidden;
                if t.t_type == MudTensorType::Ternary2Bit {
                    for r in 0..vocab_size {
                        crate::mud::dequantize_ternary_row(
                            (t.data_ptr as *const u32).add(r * cols / 8),
                            &mut out_f32[r * cols..(r + 1) * cols],
                            cols,
                        );
                    }
                    if let Some(scale_tensor) = core.tensors.get("output.prq_scale") {
                        if scale_tensor.t_type == MudTensorType::Float32 {
                            let scale_data = std::slice::from_raw_parts(
                                scale_tensor.data_ptr as *const f32,
                                vocab_size,
                            );
                            for r in 0..vocab_size {
                                let s = scale_data[r];
                                for c in 0..cols {
                                    out_f32[r * cols + c] *= s;
                                }
                            }
                        }
                    }
                    output_weight = out_f32.as_ptr();
                } else {
                    output_weight = t.data_ptr as *const f32;
                }
            }
        } else {
            // Tied weights fallback
            output_weight = emb.as_ptr();
        }

        let mut output_norm_w = std::ptr::null();
        if let Some(t) = core.tensors.get("output_norm.weight") {
            output_norm_w = t.data_ptr as *const f32;
        }

        let document = "La computación ternaria (1.58-bit) como MUD y BitNet, promete revolucionar la IA al eliminar las costosas multiplicaciones de punto flotante en la inferencia profunda. Sin embargo, su precisión en razonamiento matemático aún se considera un desafío abierto.";
        let debate_mode = std::env::var("MUD_DEBATE_MODE")
            .unwrap_or_else(|_| "debate".to_string())
            .to_lowercase();
        let game = crate::mud::arena_games::DocumentDebate::new(
            "El futuro de la Computación Ternaria en IA",
            document,
            10,
        );
        // Rotating exercise index for Professor-Student (each infinite cycle
        // poses a different local exercise — no API, fully deterministic).
        let ex_idx = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut layers: Vec<crate::mud::slime_forward::SlimeLayer> = Vec::new();
        for blk in 0..n_layers {
            let prefix = format!("blk.{}.", blk);
            let t = |name: &str| -> *const u8 {
                core.tensors
                    .get(&format!("{}{}.weight", prefix, name))
                    .map(|t| t.data_ptr)
                    .unwrap_or(std::ptr::null())
            };
            let ts = |name: &str| -> *const f32 {
                core.tensors
                    .get(&format!("{}{}.prq_scale", prefix, name))
                    .map(|t| t.data_ptr as *const f32)
                    .unwrap_or(std::ptr::null())
            };
            let tn = |name: &str| -> *const f32 {
                core.tensors
                    .get(&format!("{}{}.weight", prefix, name))
                    .map(|t| t.data_ptr as *const f32)
                    .unwrap_or(std::ptr::null())
            };
            // Llama/Smol: blk.N.norm; Qwen3/Bonsai: blk.N.ffn_norm
            let ffn_norm_w = {
                let a = tn("ffn_norm");
                if !a.is_null() {
                    a
                } else {
                    tn("norm")
                }
            };
            // w3=up, w1=gate, w2=down (or up/gate alt); honors MUD_TRAIN_EXPERT
            let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);

            layers.push(crate::mud::slime_forward::SlimeLayer {
                q_w: t("attn_q"),
                k_w: t("attn_k"),
                v_w: t("attn_v"),
                o_w: t("attn_output"),
                q_scales: ts("attn_q"),
                k_scales: ts("attn_k"),
                v_scales: ts("attn_v"),
                o_scales: ts("attn_output"),
                ffn_up_w: t(&ffn.up),
                ffn_gate_w: t(&ffn.gate),
                ffn_down_w: t(&ffn.down),
                ffn_up_scales: ts(&ffn.up),
                ffn_gate_scales: ts(&ffn.gate),
                ffn_down_scales: ts(&ffn.down),
                attn_norm_w: tn("attn_norm"),
                ffn_norm_w,
                attn_sub_norm_w: tn("attn_sub_norm"),
                ffn_sub_norm_w: tn("ffn_sub_norm"),
                q_norm_w: tn("attn_q_norm"),
                k_norm_w: tn("attn_k_norm"),
                mhc_alpha_w: tn("mhc_alpha"),
                mhc_beta_w: tn("mhc_beta"),
                mhc_radius_w: tn("mhc_radius"),
                n_kv_heads,
                ffn_mid,
                rope_theta,
            });
        }

        let mut arena = crate::mud::debate_trainer::DebateArena::new(
            crate::model::tokenizer::Tokenizer::from_mud_metadata(
                mud.global_metadata
                    .get("tokenizer.tokens")
                    .map(|s| s.as_str())
                    .unwrap_or(""),
                mud.global_metadata
                    .get("tokenizer.merges")
                    .map(|s| s.as_str())
                    .unwrap_or(""),
            ),
            max_pos,
            3600,
            hidden,
            n_layers,
            n_heads,
            n_kv_heads,
            ffn_mid,
            max_emb,
            vocab_size,
            output_weight,
            output_norm_w,
        );
        arena.sender = log_sender.clone();

        let mut shadow_layers = Vec::with_capacity(n_layers);
        for layer in layers.iter().take(n_layers) {
            let head_dim = hidden / n_heads;
            let mut shadow = crate::mud::slime_backward::SlimeLayerShadowF32::new(
                hidden, ffn_mid, n_kv_heads, head_dim,
            );

            // Dequantize from layers to shadow and apply scales
            unsafe {
                let dequant_with_scale =
                    |packed: *const u32, scales: *const f32, out: &mut [f32], cols: usize| {
                        if packed.is_null() || scales.is_null() {
                            return;
                        }
                        let rows = out.len() / cols;
                        for r in 0..rows {
                            let start = r * cols;
                            crate::mud::dequantize_ternary_row(
                                packed.add(r * cols / 8),
                                &mut out[start..start + cols],
                                cols,
                            );
                            let s = *scales.add(r);
                            for c in 0..cols {
                                out[start + c] *= s;
                            }
                        }
                    };

                dequant_with_scale(
                    layer.q_w as *const u32,
                    layer.q_scales,
                    &mut shadow.q_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.k_w as *const u32,
                    layer.k_scales,
                    &mut shadow.k_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.v_w as *const u32,
                    layer.v_scales,
                    &mut shadow.v_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.o_w as *const u32,
                    layer.o_scales,
                    &mut shadow.o_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.ffn_up_w as *const u32,
                    layer.ffn_up_scales,
                    &mut shadow.ffn_up_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.ffn_gate_w as *const u32,
                    layer.ffn_gate_scales,
                    &mut shadow.ffn_gate_w,
                    hidden,
                );
                dequant_with_scale(
                    layer.ffn_down_w as *const u32,
                    layer.ffn_down_scales,
                    &mut shadow.ffn_down_w,
                    ffn_mid,
                );
            }
            shadow_layers.push(shadow);
        }

        let _qat_opt: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher> = None;
        let mut emb = vec![0.0; vocab_size * hidden];
        let emb_tensor = core.tensors.get("token_embd.weight").unwrap();
        unsafe {
            let cols = hidden;
            if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                for r in 0..vocab_size {
                    crate::mud::dequantize_ternary_row(
                        (emb_tensor.data_ptr as *const u32).add(r * cols / 8),
                        &mut emb[r * cols..(r + 1) * cols],
                        cols,
                    );
                }
                if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                    if scale_tensor.t_type == MudTensorType::Float32 {
                        let scale_data = std::slice::from_raw_parts(
                            scale_tensor.data_ptr as *const f32,
                            vocab_size,
                        );
                        for r in 0..vocab_size {
                            let s = scale_data[r];
                            for c in 0..cols {
                                emb[r * cols + c] *= s;
                            }
                        }
                    }
                }
            } else {
                std::ptr::copy_nonoverlapping(
                    emb_tensor.data_ptr as *const f32,
                    emb.as_mut_ptr(),
                    vocab_size * hidden,
                );
            }
        }

        // Use global pool (sized pre-pin). Local new() after pin can collapse to 1 thread.
        let pool = crate::mud::pcore_pool::get_pool();
        let topic = "El futuro de la Computación Ternaria en IA".to_string();
        let doc = document.to_string();
        let max_t = game.max_turns();
        let ex_idx2 = ex_idx.clone();
        let factory = move || {
            match debate_mode.as_str() {
                "professor" => {
                    let i = ex_idx2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    Box::new(crate::mud::arena_games::ProfessorStudent::new(3, i))
                        as Box<dyn crate::mud::arena_games::ArenaGame>
                }
                "games" => {
                    // Verifiable seed-survival games (no API): Math / TicTacToe.
                    // Rotate so the model practices different checkable tasks.
                    let pick = ex_idx2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if pick.is_multiple_of(2) {
                        Box::new(crate::mud::arena_games::MathChallenge::random())
                            as Box<dyn crate::mud::arena_games::ArenaGame>
                    } else {
                        Box::new(crate::mud::arena_games::TicTacToe::new())
                            as Box<dyn crate::mud::arena_games::ArenaGame>
                    }
                }
                _ => Box::new(crate::mud::arena_games::DocumentDebate::new(
                    &topic, &doc, max_t,
                )) as Box<dyn crate::mud::arena_games::ArenaGame>,
            }
        };
        arena = arena.with_stop_flag(stop_flag.clone());
        arena.run_game(
            factory,
            &mut layers,
            &mut shadow_layers,
            &emb,
            vocab_size,
            pool,
        )?;

        // Fase 3: writeback survivors (here: the single arena pair) to the .mud.
        let learn_on = std::env::var("MUD_DEBATE_LEARN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);

        // Quantize back and save (Fase 3: only persist if MUD_DEBATE_LEARN=1).
        // The manual writeback below already mutates `core` in-place (no second
        // mutable borrow of `mud`), so it is gated instead of sync_shadow_to_mud.
        let writeback = learn_on;
        if writeback {
            if print_stdout {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ok",
                        "saving trained shadow layers back to MUD..."
                    )
                );
            } else if let Some(tx) = &log_sender {
                let _ = tx.send(crate::mud::trainer_ui::note(
                    "ok",
                    "saving trained shadow layers back to MUD...",
                ));
            }
        }
        if writeback {
            // C1/C2 fix (2026-07-20): reuse sync_shadow_to_mud instead of the manual
            // pack that (a) skipped the PRQ scale (s = absmean·√½) → lost all weight
            // magnitude (any |w|<0.5 became 0, large w became ±1), and (b) never wrote
            // the *.prq_scale tensors → stale/inflated scales persisted → vocab collapse.
            // Emb is frozen (default) so an empty shadow slices the skip branch.
            let mut empty_emb: Vec<f32> = Vec::new();
            self.sync_shadow_to_mud(&mut mud, &mut empty_emb, &mut shadow_layers, None, true);
            mud.save(&self.model_path)?;
            // Authoritative no-op check vs the captured pre-debate input hash.
            let out_hash = Self::hash_trained_weights(&mud);
            if out_hash == input_weights_hash {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        "⚠ DEBATE WRITE-BACK IS A NO-OP — trained weights byte-identical to input (hash match). The .mud did not change; debate learning produced no weight delta."
                    )
                );
            } else {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ok",
                        &format!(
                            "✓ debate write-back persisted (weights hash {:#018x} → {:#018x})",
                            input_weights_hash, out_hash
                        )
                    )
                );
            }
        }

        Ok(())
    }

    /// Returns `(alive, detail)` where `alive` is false when the model's
    /// RMSNorm weights are all-zero / non-finite in the first layers — the
    /// classic "collapsed .mud" that triggers `Dead RMSNorm` on the first
    /// forward. Norms are frozen on RO mmap during training, so a collapsed
    /// model cannot be repaired in-place and must be replaced with a healthy
    /// `.mud` before running the circuit.
    fn model_norms_alive(path: &str) -> (bool, String) {
        let mud = match MudFile::load(path) {
            Ok(m) => m,
            Err(e) => return (false, format!("load error: {}", e)),
        };
        let n_layers = mud
            .global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if n_layers == 0 {
            return (false, "no layers in metadata".to_string());
        }
        let Some(core) = mud.skills.get("core") else {
            return (false, "no core skill".to_string());
        };
        let mut dead = 0usize;
        let mut checked = 0usize;
        for blk in 0..n_layers.min(4) {
            for name in ["attn_norm.weight", "ffn_norm.weight", "norm.weight"] {
                let key = format!("blk.{}.{}", blk, name);
                let Some(t) = core.tensors.get(&key) else {
                    continue;
                };
                if t.t_type != MudTensorType::Float32 {
                    continue;
                }
                if t.data_ptr.is_null() {
                    dead += 1;
                    checked += 1;
                    continue;
                }
                let n = t.shape.iter().product::<usize>().clamp(1, 64);
                let slice = unsafe { std::slice::from_raw_parts(t.data_ptr as *const f32, n) };
                let nz = slice.iter().filter(|x| **x != 0.0 && x.is_finite()).count();
                checked += 1;
                if nz == 0 {
                    dead += 1;
                }
            }
        }
        if dead > 0 {
            (
                false,
                format!(
                    "{} RMSNorm tensors all-zero/non-finite in first layers (model collapsed; needs healthy .mud)",
                    dead
                ),
            )
        } else {
            (true, format!("norms alive ({} checked)", checked))
        }
    }

    /// Honors-mode evaluation of a `.mud` after a training phase.
    ///
    /// Returns `(integrity_ok, detail)`. Two considerations:
    ///   1. **Norms** — a collapsed model (all-zero RMSNorm) fails fast on the
    ///      forward and cannot be repaired; flagged here with a clear message.
    ///   2. **Integrity** — the file loads and its STE-writable weights are
    ///      present / non-null / non-zero size (no degenerate writeback).
    ///
    /// The circuit uses this to decide whether to KEEP the phase's writeback
    /// (honors) or roll it back (keep the previous `.mud`).
    fn circuit_eval_integrity(path: &str) -> (bool, String) {
        // A collapsed model (all-zero RMSNorm weights) cannot be repaired by the
        // circuit — norms are frozen on RO mmap and the forward fails fast with
        // "Dead RMSNorm". Fail the integrity gate early with a clear message.
        let (norms_ok, norms_detail) = Self::model_norms_alive(path);
        if !norms_ok {
            return (false, norms_detail);
        }
        let mut mud = match MudFile::load(path) {
            Ok(m) => m,
            Err(e) => return (false, format!("load error: {}", e)),
        };
        // Materialize so `data_ptr` is backed by a writable copy (owned_data).
        mud.materialize_writable();
        let n_layers = mud
            .global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if n_layers == 0 {
            return (false, "no layers in metadata".to_string());
        }
        // Structural integrity: every STE-writable weight tensor must be present,
        // have a valid (non-null) data pointer and non-zero size. A degenerate
        // writeback (e.g. a failed STE pack) drops/zeroes tensors — caught here.
        // Per-weight collapse (all-identical bytes) is intentionally NOT flagged,
        // because healthy ternary weights are naturally skewed toward 0x0; the
        // quality benchmark (verifiable-game win rate) is the regression gate.
        let core = match mud.skills.get("core") {
            Some(c) => c,
            None => return (false, "no core skill".to_string()),
        };
        let mut missing = 0usize;
        let mut inspected = 0usize;
        let mut checked: std::collections::HashSet<String> = std::collections::HashSet::new();
        let attn_names = ["attn_q", "attn_k", "attn_v", "attn_output"];
        let ffn_names = ["w1", "w2", "w3"];
        for blk in 0..n_layers {
            for name in attn_names {
                let key = format!("blk.{}.{}.weight", blk, name);
                checked.insert(key);
            }
            // MoE expert FFN (dense-path when MUD_TRAIN_EXPERT is set).
            for e in 0..1 {
                for name in ffn_names {
                    let key = format!("blk.{}.expert.{}.{}.weight", blk, e, name);
                    checked.insert(key);
                }
            }
        }
        for key in &checked {
            match core.tensors.get(key) {
                Some(t)
                    if t.shape.iter().product::<usize>() > 0
                        && (!t.data_ptr.is_null() || t.owned_data.is_some()) =>
                {
                    inspected += 1;
                }
                _ => missing += 1,
            }
        }
        if missing > 0 {
            (
                false,
                format!(
                    "{} weight tensors missing/null (inspected {})",
                    missing, inspected
                ),
            )
        } else {
            // T0.3: logits-collapse gate. A model whose forward already collapses
            // to token-0 across diverse prompts cannot be repaired by the circuit
            // (norms/weights frozen); refuse early with a clear message instead of
            // grinding on a degenerate base.
            match MudFile::load(path) {
                Ok(probe) => {
                    if crate::mud::inference::model_logits_collapsed(&probe) {
                        (
                            false,
                            "logits collapsed (token-0 dominance across probe prompts)".to_string(),
                        )
                    } else {
                        (
                            true,
                            format!("integrity ok ({} weight tensors present)", inspected),
                        )
                    }
                }
                Err(e) => (false, format!("probe load error: {}", e)),
            }
        }
    }

    /// Fast verifiable-game benchmark: the model (player A) plays N matches of
    /// `MathChallenge` / `TicTacToe`; returns `(win_rate, matches_played)`.
    /// Reuses `run_debate_session` in `games` mode over an internal channel so no
    /// RNG / API is needed — pure local inference + `VerifiableJudge`.
    fn circuit_benchmark_games(
        &mut self,
        stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<(f32, usize)> {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::env::set_var("MUD_DEBATE_MODE", "games");
        // Short, bounded benchmark so the circuit never stalls. A MathChallenge
        // match is <=4 attempts; keep tokens tiny so at least one match closes
        // within the time-box even on slow CPU inference (env-overridable).
        if std::env::var("MUD_DEBATE_MAX_TIME").is_err() {
            std::env::set_var("MUD_DEBATE_MAX_TIME", "90");
        }
        std::env::set_var("MUD_DEBATE_MAX_NEW_TOKENS", "3");
        let bench_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.run_debate_session(Some(tx), stop_flag);
        }));
        // Parse match results: "...match #N -> A:x.x B:y.y..." ; A>0 means model won.
        let mut wins = 0usize;
        let mut matches = 0usize;
        while let Ok(msg) = rx.try_recv() {
            if let Some(pos) = msg.find("match #") {
                let rest = &msg[pos + "match #".len()..];
                if let Some(ab) = rest.split("->").nth(1) {
                    // Format: "A:+0.00 B:-0.70" — strip the "A:" / "B:" labels.
                    let mut parts = ab.split('B').map(|s| s.trim());
                    if let (Some(a_s), Some(b_s)) = (parts.next(), parts.next()) {
                        let a_num = a_s.trim_start_matches('A').trim_start_matches(':').trim();
                        let b_num = b_s.trim_start_matches(':').trim();
                        if let (Ok(a), Ok(_b)) = (a_num.parse::<f32>(), b_num.parse::<f32>()) {
                            matches += 1;
                            if a > 0.0 {
                                wins += 1;
                            }
                        }
                    }
                }
            }
        }
        if bench_result.is_err() {
            // Benchmark panicked (e.g. model path issue) — treat as 0 quality.
            return Ok((0.0, matches));
        }
        let rate = if matches > 0 {
            wins as f32 / matches as f32
        } else {
            0.0
        };
        Ok((rate, matches))
    }

    /// Infinite training circuit: rotates batteries of phases until `quit` / Ctrl-C.
    ///
    /// Each **seed** generates a distinct **battery** (a shuffled ordering of the
    /// phase set) so training is never a fixed monotonic schedule — every seed
    /// explores the phases in a different order. Phases (local, no-API, P-07):
    ///   - align     — one corpus alignment epoch (STE QAT)
    ///   - debate    — RLVR document debate (TextJudge)
    ///   - games     — verifiable seed-survival games (Math / TicTacToe, VerifiableJudge)
    ///   - professor — professor→student→grade loop (ProfessorJudge rubrik)
    ///
    /// When a battery is exhausted a fresh seed produces a new battery, so the
    /// loop keeps varying. Each phase is time-boxed by `MUD_CIRCUIT_MAX_PER_MODE`
    /// (default 120s) so the loop never freezes on a single slow phase. Phases
    /// that persist write back to the `.mud` when `MUD_DEBATE_LEARN=1` (default
    /// ON inside the circuit). On `quit` the last phase stops and the model is
    /// left saved by the underlying session.
    ///
    /// Shuffling uses a tiny deterministic LCG (no external RNG, P-07 friendly);
    /// the seed itself derives from a monotonic counter + wall-clock so each run
    /// and each new battery differs.
    pub fn run_training_circuit(
        &mut self,
        sender: Option<std::sync::mpsc::Sender<String>>,
        stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<()> {
        // Unified announce: prints live AND appends to logs/circuit.log (timestamped).
        let announce = |msg: &str| {
            let line = crate::mud::trainer_ui::circuit_event("circuit", msg);
            if let Some(tx) = &sender {
                let _ = tx.send(line);
            } else {
                println!("{}", line);
            }
        };
        // Honors-mode persistence: default ON inside the circuit.
        let learn = std::env::var("MUD_DEBATE_LEARN")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        std::env::set_var("MUD_DEBATE_LEARN", if learn { "1" } else { "0" });
        // Per-phase time box so the loop never freezes (default 120s).
        let max_per_mode = std::env::var("MUD_CIRCUIT_MAX_PER_MODE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(120);
        std::env::set_var("MUD_DEBATE_MAX_TIME", max_per_mode.to_string());
        let epochs = std::env::var("MUD_CIRCUIT_EPOCHS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        let batch = std::env::var("MUD_CIRCUIT_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16);

        let mut cycle = 0usize;
        // Seed source: monotonic counter mixed with wall-clock nanos (no RNG crate).
        let mut seed: u64 = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E3779B97F4A7C15)
            ^ 0x00C0_FFEE_DEAD_BEEF_u64;

        // Usamos un orden fijo de currículum de aprendizaje en vez de barajar aleatoriamente.
        // Progresión lógica: align (base) -> professor (teoría) -> debate (razonamiento) -> games (evaluación final).
        let build_battery = |_s: &mut u64| -> Vec<&'static str> {
            vec!["align", "professor", "debate", "games"]
        };

        let mut battery = build_battery(&mut seed);
        let mut battery_seed = seed;
        let mut rpg = crate::mud::circuit_rpg::CircuitRpgStats::load(&self.model_path);

        announce(&format!(
            "⚙️  MUD Training Circuit iniciado (loop infinito por semilla). Modelo: {}. Fases base: align/debate/games/professor. Max/modo={}s, learn={}, epochs={}, batch={}. [q]/Ctrl-C para salir y guardar. Log: logs/circuit.log",
            self.model_path, max_per_mode, learn, epochs, batch
        ));
        announce(&format!(
            "🛡️  RPG Stats: HP {:.1}/{:.1} | Gen {} | WinRate {:.2} | Ciclos {} | A: {} | B: {}",
            rpg.hp, rpg.max_hp, rpg.generation, rpg.win_rate, rpg.cycles_survived, rpg.name, rpg.baseline_name
        ));
        announce(&format!(
            "🌱 Semilla {} · batería: {}",
            battery_seed,
            battery.join(" → ")
        ));

        // Fail-fast health check BEFORE any forward: a collapsed model (all-zero
        // RMSNorm) would otherwise panic with "Dead RMSNorm" inside a PCorePool
        // worker thread, which `catch_unwind` cannot capture (kills the process).
        // Report it clearly and stop instead.
        {
            let (alive, detail) = Self::model_norms_alive(&self.model_path);
            if !alive {
                announce(&format!(
                    "❌ Modelo colapsado: {}. El circuito NO puede repararlo (normas congeladas). Usa un .mud sano (p.ej. ternary_bonsai_1.7b.mud).",
                    detail
                ));
                anyhow::bail!(
                    "model collapsed (Dead RMSNorm risk): {} — replace with a healthy .mud",
                    detail
                );
            } else {
                announce(&format!("✓ Health-check: {}", detail));
            }
        }

        let wall0 = std::time::Instant::now();
        // Honors-mode evaluation toggle (default ON): only keep a phase's writeback
        // if integrity is clean AND quality did not regress vs the baseline.
        let eval_on = std::env::var("MUD_CIRCUIT_EVAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(true);
        let backup_path = format!("{}.bak_circuit", self.model_path);
        // Baseline quality (verifiable-game win rate) taken once at circuit start.
        let base_rate = if eval_on {
            // catch_unwind: a collapsed model can panic (Dead-RMSNorm) during the
            // forward inside the benchmark — never let that kill the circuit.
            let bench_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.circuit_benchmark_games(stop_flag.clone())
            }));
            match bench_res {
                Ok(Ok((r, m))) => {
                    announce(&format!(
                        "📊 Baseline calidad (juegos verificables): win_rate={:.2} sobre {} matches",
                        r, m
                    ));
                    r
                }
                Ok(Err(e)) => {
                    announce(&format!(
                        "⚠️  benchmark baseline falló (continuando): {}",
                        e
                    ));
                    0.0
                }
                Err(_) => {
                    announce("⚠️  benchmark baseline panic (modelo colapsado?) — baseline=0");
                    0.0
                }
            }
        } else {
            0.0
        };
        const HONORS_TOL: f32 = 0.15; // allow up to 15% regression before rollback

        loop {
            if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                announce("⏹  Circuito detenido por el usuario. El último estado ya fue guardado por la fase.");
                break;
            }
            // Draw next phase from the current battery; mint a new seed+battery
            // once the battery is exhausted (keeps training non-monotonic).
            if battery.is_empty() {
                seed = seed.wrapping_mul(2).wrapping_add(1).wrapping_add(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(1),
                );
                battery = build_battery(&mut seed);
                battery_seed = seed;
                announce(&format!(
                    "🌱 Nueva semilla {} · batería: {}",
                    battery_seed,
                    battery.join(" → ")
                ));
            }
            let phase = battery.remove(0);
            cycle += 1;
            let remaining = battery.join(", ");
            let phase_start = std::time::Instant::now();
            announce(&format!(
                "▶ Ciclo #{} · Semilla {} · FASE={} · batería restante: [{}]",
                cycle, battery_seed, phase, remaining
            ));

            // Snapshot the current (pre-phase) model so we can roll back if the
            // phase fails the honors evaluation.
            if eval_on && std::path::Path::new(&self.model_path).exists() {
                let _ = std::fs::copy(&self.model_path, &backup_path);
            }

            let mut ok = true;
            match phase {
                "align" => {
                    // Alignment already persists to the .mud at the end of its run.
                    // catch_unwind: a panic (e.g. Dead-RMSNorm fail-fast on a
                    // collapsed model) must not kill the whole circuit process.
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.run_alignment_session(batch, epochs)
                    }));
                    match res {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            ok = false;
                            announce(&format!("⚠️  align error (continuando loop): {}", e));
                        }
                        Err(_) => {
                            ok = false;
                            announce(
                                "⚠️  align panic (modelo colapsado?) — rollback y continuo loop",
                            );
                        }
                    }
                }
                "debate" | "games" | "professor" => {
                    std::env::set_var("MUD_DEBATE_MODE", phase);
                    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.run_debate_session(sender.clone(), stop_flag.clone())
                    }));
                    match res {
                        Ok(Ok(())) => {}
                        Ok(Err(e)) => {
                            ok = false;
                            announce(&format!("⚠️  {} error (continuando loop): {}", phase, e));
                        }
                        Err(_) => {
                            ok = false;
                            let msg = format!(
                                "⚠️  {} panic (modelo colapsado?) — rollback y continuo loop",
                                phase
                            );
                            announce(&msg);
                        }
                    }
                }
                _ => {}
            }
            let dur = phase_start.elapsed().as_secs_f32();
            let verdict = if ok { "OK" } else { "ERR" };
            announce(&format!(
                "✓ Ciclo #{} · {} completada [{}] en {:.1}s · total circuito: {:.1}s",
                cycle,
                phase,
                verdict,
                dur,
                wall0.elapsed().as_secs_f32()
            ));

            // A failed/panicked phase must never leave a corrupted/dead model on
            // disk: roll back to the pre-phase snapshot first.
            if !ok && eval_on && std::path::Path::new(&backup_path).exists() {
                let _ = std::fs::copy(&backup_path, &self.model_path);
                announce(&format!(
                    "↩️  rollback a snapshot previo (fase {} falló/panic)",
                    phase
                ));
            } else if eval_on && ok {
                // Honors evaluation: keep the phase only if integrity is clean and
                // the model did not regress vs baseline. Otherwise roll back.
                let (integrity, idetail) = Self::circuit_eval_integrity(&self.model_path);
                let (rate, m) = self
                    .circuit_benchmark_games(stop_flag.clone())
                    .unwrap_or((0.0, 0usize));
                let honors = integrity && (rate >= base_rate - HONORS_TOL);
                if honors {
                    rpg.heal(10.0);
                    rpg.cycles_survived += 1;
                    rpg.win_rate = rate;
                    announce(&format!(
                        "🏅 HONORES ✓ integridad: {} | calidad win_rate={:.2} (baseline {:.2}, {} matches) → se guarda",
                        idetail, rate, base_rate, m
                    ));
                    announce(&format!("💖 +10 HP (Total: {:.1}/{:.1})", rpg.hp, rpg.max_hp));
                } else {
                    let dead = rpg.take_damage(25.0);
                    announce(&format!(
                        "❌ SIN HONORES ✗ integridad={} ({}); calidad win_rate={:.2} vs baseline {:.2} → rollback",
                        integrity, idetail, rate, base_rate
                    ));
                    announce(&format!("💔 -25 HP (Total: {:.1}/{:.1})", rpg.hp, rpg.max_hp));
                    let _ = std::fs::copy(&backup_path, &self.model_path);

                    if dead {
                        announce("💀 El modelo ha MUERTO (HP=0). Iniciando Evolución (Reset HP & +1 Gen).");
                        rpg.generation += 1;
                        rpg.hp = rpg.max_hp;
                        rpg.cycles_survived = 0;
                        let nombres = ["Aspirante", "Gladiador", "Guerrero", "Sombra", "Paladín", "Espectro", "Berserker", "Ronin"];
                        let idx = (rpg.generation as usize) % nombres.len();
                        rpg.name = format!("{} (Gen {})", nombres[idx], rpg.generation);
                        announce(&format!("🌟 Nueva Generación: {} iniciada como {}.", rpg.generation, rpg.name));
                    }
                }
                
                // Defensa del título: si superamos estrictamente al baseline, tomamos su lugar.
                if honors && rate > base_rate {
                    announce(&format!("👑 ¡{} ha vencido al Baseline con {:.2} > {:.2}! Reclamando el título...", rpg.name, rate, base_rate));
                    rpg.baseline_name = rpg.name.clone();
                }
                
                rpg.save(&self.model_path);
            }

            // Force a small yield so Ctrl-C / quit is observed between phases.
            if stop_flag.load(std::sync::atomic::Ordering::SeqCst) {
                announce("⏹  Circuito detenido por el usuario.");
                break;
            }
        }

        // Best-effort cleanup of the snapshot.
        let _ = std::fs::remove_file(&backup_path);
        announce(&format!(
            "✅ Circuito finalizado tras {} ciclos en {:.1}s. Modelo: {} | Log: logs/circuit.log",
            cycle,
            wall0.elapsed().as_secs_f32(),
            self.model_path
        ));
        Ok(())
    }

    pub fn run_alignment_session(&self, batch_size: usize, epochs: usize) -> anyhow::Result<()> {
        // Pin main thread to the first P-core (Core 0) to maximize AVX2 throughput and L1/L2 cache locality
        if let Some(core_ids) = core_affinity::get_core_ids() {
            if let Some(first_core) = core_ids.first() {
                core_affinity::set_for_current(*first_core);
            }
        }
        let mut mud = MudFile::load(&self.model_path)?;
        // STE pack writes scales/weights in-place on thawed tensors only.
        // Full materialize doubles RAM (mmap + owned) — lethal for 1.7B+.
        // Keep frozen layers + norms on RO mmap; own only STE-writable tensors.
        {
            let n_layers_hint = mud
                .global_metadata
                .get("num_hidden_layers")
                .or_else(|| mud.global_metadata.get("num_layers"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let last_n = crate::mud::sequence_pack::train_last_n_layers(n_layers_hint.max(1));
            let first_train = n_layers_hint.saturating_sub(last_n);
            let freeze_emb = crate::mud::constants::train_freeze_emb();
            // Force full materialize only if explicitly requested (debug / non-LAST_N).
            let force_full = std::env::var("MUD_TRAIN_MATERIALIZE_FULL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if force_full || n_layers_hint == 0 {
                mud.materialize_writable();
            } else {
                mud.materialize_for_ste_train(first_train, freeze_emb);
            }
        }
        // Capture the input weight hash BEFORE training so the final save can prove the
        // checkpoint actually changed vs the original (guards against silent no-op runs).
        let input_weights_hash = Self::hash_trained_weights(&mud);
        // Persist model-native specials into metadata (converter often only has raw_config_json).
        {
            let (b, e) = self
                .tokenizer
                .special_ids_from_metadata(&mud.global_metadata);
            mud.global_metadata
                .entry("bos_token_id".into())
                .or_insert_with(|| b.to_string());
            mud.global_metadata
                .entry("eos_token_id".into())
                .or_insert_with(|| e.to_string());
        }

        // ── Numeric metadata panel (W=79, pure ASCII inside box) ──────────────
        let meta = &mud.global_metadata;
        let m_layers = meta
            .get("num_hidden_layers")
            .or_else(|| meta.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_hidden = meta
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_heads = meta
            .get("num_attention_heads")
            .or_else(|| meta.get("num_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_kv = meta
            .get("num_key_value_heads")
            .or_else(|| meta.get("num_kv_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_ffn = meta
            .get("intermediate_size")
            .or_else(|| meta.get("ffn_hidden"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_vocab = meta
            .get("vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let m_maxpos = meta
            .get("max_position_embeddings")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        // Approximate parameter count (ternary, not full f32)
        let param_attn = m_layers * 4 * m_hidden * m_hidden; // Q,K,V,O
        let param_ffn = m_layers * 3 * m_hidden * m_ffn; // up,gate,down
        let param_emb = m_vocab * m_hidden; // embedding
        let total_params = param_attn + param_ffn + param_emb;
        let model_size_bytes = std::fs::metadata(&self.model_path)
            .map(|m| m.len())
            .unwrap_or(0);

        use crate::mud::trainer_ui::{box_bottom, box_kv, box_section, box_title, box_top};
        // L-01 shape dispatch + L-02 Muon NS on ash when MUD_USE_VULKAN=1
        let vulkan_on = std::env::var("MUD_USE_VULKAN").unwrap_or_default() == "1";
        let opt_tag = if vulkan_on {
            "LIVE=Muon(NS GPU|CPU)/GaLore/SGD (L-01+L-02) ash=on"
        } else {
            "LIVE=Muon(NS CPU)/GaLore/SGD (L-01)  ash=off"
        };
        let lr_init = crate::mud::constants::qat_learning_rate();
        let lr_min = lr_init / 10.0;
        println!("{}", box_top());
        println!("{}", box_title("Training Configuration"));
        println!("{}", box_section("Dimensions"));
        println!(
            "{}",
            box_kv(
                "Arch",
                &format!(
                    "layers={} hidden={} heads={}/{} ffn={}",
                    m_layers, m_hidden, m_heads, m_kv, m_ffn
                )
            )
        );
        println!(
            "{}",
            box_kv(
                "Model",
                &format!(
                    "vocab={} maxpos={} ~{:.2}M params  file={:.2} MB",
                    m_vocab,
                    m_maxpos,
                    total_params as f64 / 1_000_000.0,
                    model_size_bytes as f64 / 1_048_576.0
                )
            )
        );
        println!("{}", box_section("Config"));
        println!(
            "{}",
            box_kv(
                "Schedule",
                &format!(
                    "epochs={} batch={} chunk={} chars  optimizer={}",
                    epochs,
                    batch_size,
                    crate::mud::corpus_trainer::CHUNK_SIZE,
                    opt_tag
                )
            )
        );
        let full_seq = crate::mud::sequence_pack::train_full_seq_enabled();
        let seq_len = crate::mud::sequence_pack::train_seq_len();
        let packing = if full_seq {
            format!("L-10 + Stream D full-seq (causal windows, seq_len={seq_len}, KV grows)")
        } else {
            "L-10 pairs@pos=0 (MUD_TRAIN_FULL_SEQ=0; set 1 for full-seq)".to_string()
        };
        println!("{}", box_kv("Packing", &packing));
        println!(
            "{}",
            box_kv(
                "QAT",
                &format!(
                    "STE Ternary2Bit (1.58-bit/weight, PRQ)  ·  LR {:.0e}→{:.0e} cosine",
                    lr_init, lr_min
                )
            )
        );
        let moe_line = if let Some(eid) = crate::mud::moe_load::train_expert_id() {
            format!("MUD_TRAIN_EXPERT={eid} (dense FFN → expert.{eid})")
        } else if crate::mud::moe_train::moe_train_enabled() {
            crate::mud::moe_train::summary_line(&[])
        } else {
            "dense expert.0 (MUD_TRAIN_EXPERT=N | MUD_MOE_TRAIN=1|hash)".to_string()
        };
        println!("{}", box_kv("MoE", &moe_line));
        println!("{}", box_bottom());
        println!();
        // ──────────────────────────────────────────────────────────────────────

        // EZOP / ash QAT VRAM: host-visible UMA on Iris Xe doubles shadow+grad in system RAM.
        // Skip when SCALES_ONLY (CPU pack only) or MUD_TRAIN_EZOP=0 — avoids OOM on 15 GiB hosts.
        let scales_only_boot = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let ezop_forced_off = std::env::var("MUD_TRAIN_EZOP")
            .map(|v| {
                let t = v.trim().to_ascii_lowercase();
                t == "0" || t == "off" || t == "false" || t == "no" || t == "cpu"
            })
            .unwrap_or(false);
        let mut vk_qat_storage = if scales_only_boot || ezop_forced_off {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ram",
                    &format!(
                        "EZOP QAT VRAM skipped ({})",
                        if scales_only_boot {
                            "SCALES_ONLY — ash pack unused"
                        } else {
                            "MUD_TRAIN_EZOP=0"
                        }
                    )
                )
            );
            None
        } else if self.vk.is_some() {
            match crate::mud::ash_qat_dispatcher::AshQatDispatcher::new() {
                Ok(d) if d.is_available() => Some(d),
                Ok(_) => {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note(
                            "ram",
                            "EZOP dispatcher not available — CPU STE"
                        )
                    );
                    None
                }
                Err(e) => {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note(
                            "ram",
                            &format!("EZOP init failed ({e}) — CPU STE")
                        )
                    );
                    None
                }
            }
        } else {
            None
        };

        // AWAKE-01: Pre-align structural ternary boundaries (SGD on synthetic noise).
        // OFF by default on low-resource hosts (i7-1260P / 15 GiB Iris Xe): it rewrites
        // every ternary row before real training and wastes CPU/RAM for no corpus signal.
        // Opt-in via MUD_TRAIN_AWAKE=1 (or unset skip). MUD_TRAIN_SKIP_AWAKE=1 also skips.
        let awake_opt_in = std::env::var("MUD_TRAIN_AWAKE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let skip_awake = std::env::var("MUD_TRAIN_SKIP_AWAKE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if awake_opt_in && !skip_awake {
            self.deep_local_alignment(&mut mud, true)?;
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "skipping AWAKE-01 (low-resource default; MUD_TRAIN_AWAKE=1 to enable)"
                )
            );
        }

        // FREEZE_EMB / large-vocab (Bonsai 152k×2048≈1.2GiB): skip full FP32 emb materialize.
        // Empty shadow_emb → on-the-fly row unpack from mmap ELUT in train_on_sequence.
        // Default: frozen (see constants::train_freeze_emb).
        let freeze_emb_alloc = crate::mud::constants::train_freeze_emb();
        let mut shadow_emb = {
            let core = mud
                .skills
                .get("core")
                .ok_or_else(|| anyhow::anyhow!("No core skill"))?;
            let emb_tensor = core
                .tensors
                .get("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("No embedding"))?;
            let elements = emb_tensor.shape[0] * emb_tensor.shape[1];
            if freeze_emb_alloc {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ram",
                        &format!(
                            "FREEZE_EMB: skip {:.1} MiB emb FP32 — on-the-fly ELUT rows",
                            (elements * 4) as f64 / (1024.0 * 1024.0)
                        )
                    )
                );
                Vec::new()
            } else {
                let mut data = vec![0.0f32; elements];
                unsafe {
                    if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                        let rows = emb_tensor.shape[0];
                        let cols = emb_tensor.shape[1];
                        let u32s_per_row = cols.div_ceil(8);
                        for r in 0..rows {
                            crate::mud::dequantize_ternary_row(
                                (emb_tensor.data_ptr as *const u32).add(r * u32s_per_row),
                                &mut data[r * cols..(r + 1) * cols],
                                cols,
                            );
                        }
                        if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                            if scale_tensor.t_type == MudTensorType::Float32 {
                                let scale_data = std::slice::from_raw_parts(
                                    scale_tensor.data_ptr as *const f32,
                                    rows,
                                );
                                for r in 0..rows {
                                    let s = scale_data[r];
                                    for c in 0..cols {
                                        data[r * cols + c] *= s;
                                    }
                                }
                            }
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(
                            emb_tensor.data_ptr as *const f32,
                            data.as_mut_ptr(),
                            elements,
                        );
                    }
                }
                data
            }
        };

        let hidden = mud
            .global_metadata
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing hidden_size");
        let n_layers = mud
            .global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_layers");
        let n_heads = mud
            .global_metadata
            .get("num_attention_heads")
            .or_else(|| mud.global_metadata.get("num_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_heads");
        let n_kv_heads = mud
            .global_metadata
            .get("num_key_value_heads")
            .or_else(|| mud.global_metadata.get("num_kv_heads"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing num_kv_heads");
        let ffn_mid = mud
            .global_metadata
            .get("intermediate_size")
            .or_else(|| mud.global_metadata.get("ffn_hidden"))
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing ffn_mid");
        let max_pos = mud
            .global_metadata
            .get("max_position_embeddings")
            .and_then(|s| s.parse::<usize>().ok())
            .expect("Missing max_position_embeddings");
        let rope_theta: f32 = mud
            .global_metadata
            .get("rope_theta")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000.0);
        let core = mud
            .skills
            .get_mut("core")
            .ok_or_else(|| anyhow::anyhow!("No core skill"))?;

        // RAM-first: only allocate FP32 shadows + Adam for thawed last-N layers.
        // Full 1.7B × 28 layers would be ~5.6 GiB shadows alone (lethal on 15 GiB).
        let last_n_alloc = crate::mud::sequence_pack::train_last_n_layers(n_layers);
        let first_train_alloc = n_layers.saturating_sub(last_n_alloc);
        println!(
            "{}",
            crate::mud::trainer_ui::note(
                "ram",
                &format!(
                    "shadow alloc: last_n={last_n_alloc} first_train={first_train_alloc}/{n_layers} (frozen layers = empty shadow)"
                )
            )
        );

        let mut layers: Vec<crate::mud::slime_forward::SlimeLayer> = Vec::new();
        let mut shadow_layers: Vec<crate::mud::slime_backward::SlimeLayerShadowF32> = Vec::new();

        for blk in 0..n_layers {
            let prefix = format!("blk.{}.", blk);
            let t = |name: &str| -> *const u8 {
                core.tensors
                    .get(&format!("{}{}.weight", prefix, name))
                    .map(|t| t.data_ptr)
                    .unwrap_or(std::ptr::null())
            };
            let ts = |name: &str| -> *const f32 {
                core.tensors
                    .get(&format!("{}{}.prq_scale", prefix, name))
                    .map(|t| t.data_ptr as *const f32)
                    .unwrap_or(std::ptr::null())
            };
            let tn = |name: &str| -> *const f32 {
                core.tensors
                    .get(&format!("{}{}.weight", prefix, name))
                    .map(|t| t.data_ptr as *const f32)
                    .unwrap_or(std::ptr::null())
            };
            // Llama/Smol: blk.N.norm; Qwen3/Bonsai: blk.N.ffn_norm
            let ffn_norm_w = {
                let a = tn("ffn_norm");
                if !a.is_null() {
                    a
                } else {
                    tn("norm")
                }
            };
            // w3=up, w1=gate, w2=down (or up/gate alt); honors MUD_TRAIN_EXPERT
            let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);

            layers.push(crate::mud::slime_forward::SlimeLayer {
                q_w: t("attn_q"),
                k_w: t("attn_k"),
                v_w: t("attn_v"),
                o_w: t("attn_output"),
                q_scales: ts("attn_q"),
                k_scales: ts("attn_k"),
                v_scales: ts("attn_v"),
                o_scales: ts("attn_output"),
                ffn_up_w: t(&ffn.up),
                ffn_gate_w: t(&ffn.gate),
                ffn_down_w: t(&ffn.down),
                ffn_up_scales: ts(&ffn.up),
                ffn_gate_scales: ts(&ffn.gate),
                ffn_down_scales: ts(&ffn.down),
                attn_norm_w: tn("attn_norm"),
                ffn_norm_w,
                attn_sub_norm_w: tn("attn_sub_norm"),
                ffn_sub_norm_w: tn("ffn_sub_norm"),
                q_norm_w: tn("attn_q_norm"),
                k_norm_w: tn("attn_k_norm"),
                mhc_alpha_w: tn("mhc_alpha"),
                mhc_beta_w: tn("mhc_beta"),
                mhc_radius_w: tn("mhc_radius"),
                n_kv_heads,
                ffn_mid,
                rope_theta,
            });

            let t_shape = |name: &str| -> Vec<usize> {
                core.tensors
                    .get(&format!("{}{}.weight", prefix, name))
                    .map(|t| t.shape.clone())
                    .unwrap_or_default()
            };

            if blk < first_train_alloc {
                // Frozen: zero-heap shadow (forward still uses mmap ELUT via `layers`)
                shadow_layers.push(crate::mud::slime_backward::SlimeLayerShadowF32::empty());
                continue;
            }

            let q_shape = t_shape("attn_q");
            let k_shape = t_shape("attn_k");
            let v_shape = t_shape("attn_v");
            let o_shape = t_shape("attn_output");
            let up_shape = t_shape(&ffn.up);
            let gate_shape = t_shape(&ffn.gate);
            let down_shape = t_shape(&ffn.down);
            if q_shape.len() < 2 || k_shape.len() < 2 {
                anyhow::bail!("blk.{blk}: missing attn weight shapes for shadow alloc");
            }
            let q_opt = crate::mud::slime_backward::select_optimizer(q_shape[0], q_shape[1]);
            let k_opt = crate::mud::slime_backward::select_optimizer(k_shape[0], k_shape[1]);
            let v_opt = crate::mud::slime_backward::select_optimizer(v_shape[0], v_shape[1]);
            let o_opt = crate::mud::slime_backward::select_optimizer(o_shape[0], o_shape[1]);
            let ffn_up_opt = crate::mud::slime_backward::select_optimizer(up_shape[0], up_shape[1]);
            let ffn_gate_opt =
                crate::mud::slime_backward::select_optimizer(gate_shape[0], gate_shape[1]);
            let ffn_down_opt =
                crate::mud::slime_backward::select_optimizer(down_shape[0], down_shape[1]);
            use crate::mud::adam_state::AdamState;
            let mut shadow = crate::mud::slime_backward::SlimeLayerShadowF32 {
                q_w: vec![0.0; q_shape.iter().product()],
                k_w: vec![0.0; k_shape.iter().product()],
                v_w: vec![0.0; v_shape.iter().product()],
                o_w: vec![0.0; o_shape.iter().product()],
                ffn_up_w: vec![0.0; up_shape.iter().product()],
                ffn_gate_w: vec![0.0; gate_shape.iter().product()],
                ffn_down_w: vec![0.0; down_shape.iter().product()],
                q_opt,
                k_opt,
                v_opt,
                o_opt,
                ffn_up_opt,
                ffn_gate_opt,
                ffn_down_opt,
                q_adam: AdamState::for_strategy(q_shape.iter().product(), q_opt),
                k_adam: AdamState::for_strategy(k_shape.iter().product(), k_opt),
                v_adam: AdamState::for_strategy(v_shape.iter().product(), v_opt),
                o_adam: AdamState::for_strategy(o_shape.iter().product(), o_opt),
                ffn_up_adam: AdamState::for_strategy(up_shape.iter().product(), ffn_up_opt),
                ffn_gate_adam: AdamState::for_strategy(gate_shape.iter().product(), ffn_gate_opt),
                ffn_down_adam: AdamState::for_strategy(down_shape.iter().product(), ffn_down_opt),
                slime_x: None,
            };

            // Inflate ternary → FP32 shadow (ELUT: 8 weights / u32 → stride cols/8)
            unsafe {
                let p = format!("blk.{}.", blk);
                let inf = |name: &str, dest: &mut [f32]| {
                    if dest.is_empty() {
                        return;
                    }
                    if let Some(t) = core.tensors.get(&format!("{}{}.weight", p, name)) {
                        if t.t_type == crate::mud::MudTensorType::Ternary2Bit {
                            let cols = t.shape[1];
                            let u32s_per_row = cols.div_ceil(8);
                            for r in 0..t.shape[0] {
                                crate::mud::dequantize_ternary_row(
                                    (t.data_ptr as *const u32).add(r * u32s_per_row),
                                    &mut dest[r * cols..(r + 1) * cols],
                                    cols,
                                );
                                if let Some(scale_t) =
                                    core.tensors.get(&format!("{}{}.prq_scale", p, name))
                                {
                                    let s = *(scale_t.data_ptr as *const f32).add(r);
                                    for c in 0..cols {
                                        dest[r * cols + c] *= s;
                                    }
                                }
                            }
                        }
                    }
                };
                inf("attn_q", &mut shadow.q_w);
                inf("attn_k", &mut shadow.k_w);
                inf("attn_v", &mut shadow.v_w);
                inf("attn_output", &mut shadow.o_w);
                inf(&ffn.up, &mut shadow.ffn_up_w);
                inf(&ffn.gate, &mut shadow.ffn_gate_w);
                inf(&ffn.down, &mut shadow.ffn_down_w);
            }
            shadow_layers.push(shadow);
        }

        // Allocate VRAM and map initial shadows (thawed layers only).
        // Iris Xe UMA: each matrix = shadow+grad+scales+packed HOST_VISIBLE ≈ 2×FP32 shadow.
        // Bonsai last-2 FFN alone can be hundreds of MiB — soft-fail → CPU STE (no panic).
        if let Some(vk_qat) = vk_qat_storage.as_mut() {
            println!("  \x1b[1;36m[EZOP]\x1b[0m Allocating Vulkan VRAM for QAT...");
            let mut ezop_oom: Option<String> = None;
            'ezop: for (blk, shadow) in shadow_layers.iter().enumerate() {
                if shadow.is_empty() {
                    continue;
                }
                let p = format!("blk.{}.", blk);
                let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);

                let matrices: Vec<(&str, &[f32], usize)> = vec![
                    ("attn_q", &shadow.q_w, hidden),
                    ("attn_k", &shadow.k_w, hidden),
                    ("attn_v", &shadow.v_w, hidden),
                    ("attn_output", &shadow.o_w, hidden),
                    (&ffn.up, &shadow.ffn_up_w, hidden),
                    (&ffn.gate, &shadow.ffn_gate_w, hidden),
                    (
                        &ffn.down,
                        &shadow.ffn_down_w,
                        shadow.ffn_down_w.len() / hidden.max(1),
                    ),
                ];

                for (m_name, shadow_w, cols) in matrices {
                    let name = format!("{}{}", p, m_name);
                    let elements = shadow_w.len();
                    if elements == 0 || cols == 0 {
                        continue;
                    }
                    let rows = elements / cols;
                    if let Err(e) = vk_qat.ensure_buffers(&name, elements, rows, shadow_w) {
                        ezop_oom = Some(format!(
                            "{name} ({} MiB shadow): {e}",
                            (elements * 4) / (1024 * 1024)
                        ));
                        break 'ezop;
                    }
                }
            }
            if let Some(why) = ezop_oom {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        &format!("EZOP OOM — dropping GPU QAT, CPU STE pack only\n    ({why})")
                    )
                );
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        "tip: MUD_TRAIN_SCALES_ONLY=1 or MUD_TRAIN_EZOP=0 or free RAM"
                    )
                );
                vk_qat_storage = None;
            }
        }

        let head_dim = hidden / n_heads;
        let max_emb = 128.0;
        // Cap logical max_pos for train workspace — full 32k Bonsai ctx is wasteful for seating
        let train_max_pos = std::env::var("MUD_TRAIN_MAX_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512)
            .min(max_pos)
            .max(64);
        let mut workspace = crate::mud::slime::SlimeWorkspace::new(
            hidden,
            train_max_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_mid,
            n_layers,
            max_emb,
        );
        let mut backward_ws = crate::mud::slime_backward::SlimeBackwardWorkspace::new(
            hidden,
            ffn_mid,
            n_kv_heads * head_dim,
        );
        // Tapes: only need full tape for thawed layers; frozen get minimal empty-ish tapes
        // (forward may still write tape slots when valid — keep capacity for seq but cap pos).
        let mut tapes = (0..n_layers)
            .map(|_| {
                crate::mud::slime_backward::SlimeLayerTape::new(
                    hidden,
                    ffn_mid,
                    n_kv_heads,
                    head_dim,
                    train_max_pos,
                    0,
                )
            })
            .collect::<Vec<_>>();
        let mut gradients = (0..n_layers)
            .map(|blk| {
                if blk < first_train_alloc {
                    crate::mud::slime_backward::SlimeLayerGradients::empty()
                } else {
                    crate::mud::slime_backward::SlimeLayerGradients::new(
                        hidden, ffn_mid, n_kv_heads, head_dim,
                    )
                }
            })
            .collect::<Vec<_>>();

        // corpus_dir is where generated artifacts live (global_corpus.bin, etc.).
        // Never recurse into stash/downloads/dumps; path-component filter is belt-and-suspenders.

        /// True if any path component is a known non-train dir (stash, downloads, dumps).
        fn path_is_excluded(path: &std::path::Path) -> bool {
            path.components().any(|c| {
                let s = c.as_os_str().to_string_lossy();
                s.starts_with("_stash") || s == "dumps" || s == "dumps_archive" || s == "downloads"
            })
        }

        fn collect_files(
            dir: &std::path::Path,
            files: &mut Vec<std::path::PathBuf>,
            is_root: bool,
        ) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().unwrap_or_default().to_string_lossy();

                    if name.starts_with('.') {
                        continue;
                    }
                    // Skip stash/backup/download dirs (quick-train isolation)
                    if name.starts_with("_stash")
                        || name == "dumps"
                        || name == "dumps_archive"
                        || name == "downloads"
                    {
                        continue;
                    }
                    if path_is_excluded(&path) {
                        continue;
                    }
                    if name == "project_corpus.txt" {
                        continue;
                    }

                    if path.is_dir() {
                        if !is_root
                            || name == "src"
                            || name == "forge_autograd"
                            || name == "training"
                        {
                            collect_files(&path, files, false);
                        }
                    } else if let Some(ext) = path.extension() {
                        let ext_str = ext.to_string_lossy();
                        if ext_str == "txt" || ext_str == "rs" || ext_str == "md" {
                            let path_str = path.to_string_lossy();
                            // Normalize: accept training/corpus even with ./ prefix
                            let in_corpus = path_str.contains("training/corpus/")
                                || path_str.contains("training\\corpus\\");
                            let text_only = std::env::var("MUD_TRAIN_TEXT_ONLY")
                                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                                .unwrap_or(false);
                            if in_corpus {
                                if ext_str == "txt" || ext_str == "md" {
                                    files.push(path);
                                }
                            } else if ext_str == "rs" && !text_only {
                                files.push(path);
                            }
                        }
                    }
                }
            }
        }

        let mut text_files = Vec::new();
        // Scan only allowed subdirectories from project root
        collect_files(std::path::Path::new("."), &mut text_files, true);
        // Prefer small files first so quick AOT fills budget from short aligns
        text_files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX));

        if text_files.is_empty() {
            anyhow::bail!("No training files found in the project root!");
        }

        // P0.3 fix (2026-07-20): a model already at current_epoch=N makes a fresh
        // `--epochs M` run resolve end_epoch = N-1+M and skip N-1 "done" epochs →
        // if M is small the loop runs zero new epochs and the checkpoint is a no-op
        // (byte-identical). MUD_TRAIN_RESET_EPOCH=1 restarts the counter at 1 so the
        // requested epochs always execute. Default keeps resume behaviour.
        let reset_epoch = std::env::var("MUD_TRAIN_RESET_EPOCH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let resume_epoch = if reset_epoch {
            1
        } else {
            mud.global_metadata
                .get("trainer.current_epoch")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1)
        };
        let resume_chunk_idx = mud
            .global_metadata
            .get("trainer.current_chunk_idx")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);

        // Quick sessions: hard cap on chunks processed this session
        let max_chunks_cap: Option<usize> = std::env::var("MUD_TRAIN_MAX_CHUNKS")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&m| m > 0);

        // AOT tokens per stream chunk (default CHUNK_SIZE/4 = 12_500). Quick: much smaller.
        let aot_tokens_per_chunk: usize = if let Ok(v) = std::env::var("MUD_TRAIN_AOT_TOKENS") {
            v.parse::<usize>().unwrap_or(512).clamp(64, CHUNK_SIZE)
        } else if max_chunks_cap.is_some() {
            // Enough for batch×seq windows with headroom; avoids 12k-token materialization
            std::env::var("MUD_TRAIN_SEQ_LEN")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(16)
                .saturating_mul(batch_size.max(1))
                .saturating_mul(4)
                .clamp(256, 2048)
        } else {
            CHUNK_SIZE / 4
        };

        // Token budget for AOT rebuild when quick-capped (0 = unlimited / full corpus)
        let aot_token_budget: u64 = max_chunks_cap
            .map(|c| (c.saturating_mul(aot_tokens_per_chunk).saturating_add(64)) as u64)
            .unwrap_or(0);

        let chunks_per_file: Vec<usize> = text_files
            .iter()
            .map(|p| {
                std::fs::metadata(p)
                    .map(|m| (m.len() as usize).div_ceil(CHUNK_SIZE))
                    .unwrap_or(0)
            })
            .collect();
        let mut total_chunks_per_epoch: usize = chunks_per_file.iter().sum();
        if let Some(m) = max_chunks_cap {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    &format!(
                        "quick mode: MUD_TRAIN_MAX_CHUNKS={m}  AOT_TOKENS/chunk={aot_tokens_per_chunk}  budget≈{aot_token_budget} tok  (full corpus would be {total_chunks_per_epoch}/epoch)"
                    )
                )
            );
            if crate::mud::slime_backward::optimizer_policy_override()
                .map(|s| matches!(s, crate::mud::slime_backward::OptimizerStrategy::Sgd))
                .unwrap_or(false)
            {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "warn",
                        "optimizer=SGD (set MUD_OPT=muon|adam to override)"
                    )
                );
            }
        }
        let mut total_chunks_all_epochs = total_chunks_per_epoch * (resume_epoch - 1 + epochs);

        // ── Session start numeric summary ─────────────────────────────────────
        let end_epoch = resume_epoch - 1 + epochs;
        if epochs == 0 {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "epochs=0 requested — NO training will run (checkpoint would be a no-op). Use --epochs N or MUD_TRAIN_RESET_EPOCH=1."
                )
            );
        }
        if resume_epoch > 1 || resume_chunk_idx > 0 {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    &format!(
                        "resuming: epoch {}/{}  block {}/{}  remaining blocks: {}",
                        resume_epoch,
                        end_epoch,
                        resume_chunk_idx,
                        total_chunks_per_epoch,
                        total_chunks_all_epochs.saturating_sub(
                            resume_chunk_idx + total_chunks_per_epoch * (resume_epoch - 1)
                        )
                    )
                )
            );
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!(
                        "new session: epochs 1..{}  {}/epoch  {} total blocks",
                        end_epoch, total_chunks_per_epoch, total_chunks_all_epochs
                    )
                )
            );
        }
        println!(
            "{}",
            crate::mud::trainer_ui::note(
                "ram",
                &format!(
                    "files={}  corpus={:.2} MB",
                    text_files.len(),
                    text_files
                        .iter()
                        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
                        .sum::<u64>() as f64
                        / 1_048_576.0
                )
            )
        );
        println!();
        // ──────────────────────────────────────────────────────────────────────

        let mut global_chunks_processed = 0usize;
        if resume_epoch > 1 {
            global_chunks_processed += total_chunks_per_epoch * (resume_epoch - 1);
        }

        if resume_chunk_idx > 0 {
            global_chunks_processed += resume_chunk_idx;
        }

        let mut session_chunks_processed = 0usize;
        // Wall clock for ETA starts at first train chunk (not AOT / EZOP setup).
        let mut train_clock: Option<Instant> = None;
        let mut loss_history = std::collections::VecDeque::with_capacity(100);

        use std::fs::OpenOptions;
        use std::io::Write;
        let mut telemetry_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("mud_train_metrics.log")
            .ok();
        // Session banner + schema (train_telemetry / loss_cert skip lines starting with #)
        if let Some(ref mut f) = telemetry_file {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(
                f,
                "# MUD_TELEMETRY v2 unix_ts={} model={} max_chunks={} steps_per_chunk={} lr={:.6} opt={}",
                ts,
                self.model_path,
                max_chunks_cap
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into()),
                crate::mud::sequence_pack::train_steps_per_chunk(batch_size),
                crate::mud::constants::qat_learning_rate(),
                std::env::var("MUD_OPT").unwrap_or_else(|_| "auto".into()),
            );
            let _ = writeln!(
                f,
                "# cols: step batch loss ppl lr loss_vel varh varj <pad0> <pad0> <pad0> integral sigma_pct cognitive <pad0> toks_s steps_per_chunk elapsed_s prog_pct conf_pct epoch block blocks_total"
            );
            let _ = f.flush();
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ram",
                    "telemetry -> mud_train_metrics.log  ·  TUI: cargo run --release --bin train_telemetry"
                )
            );
        }
        let telem_every: usize = std::env::var("MUD_TELEM_EVERY")
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or(1);
        let steps_per_chunk_meta = crate::mud::sequence_pack::train_steps_per_chunk(batch_size);
        let qat_lr = crate::mud::constants::qat_learning_rate();

        enum PrefetchItem {
            Chunk {
                epoch: usize,
                f_idx: usize,
                file_path: std::path::PathBuf,
                c_idx: usize,
                file_chunks: usize,
                tokens: Vec<u32>,
            },
            EndOfEpoch {
                epoch: usize,
            },
        }

        let (tx, rx) = std::sync::mpsc::sync_channel::<PrefetchItem>(100);
        let prefetch_text_files = text_files.clone();
        let prefetch_tokenizer = self.tokenizer.clone();
        // Capture quick-mode AOT params for the prefetch thread
        let prefetch_token_budget = aot_token_budget;
        let prefetch_tokens_per_chunk = aot_tokens_per_chunk;
        let prefetch_max_chunks = max_chunks_cap;
        // Model-native specials (SmolLM2 bos=eos=0). Never emit Llama 128k OOV markers.
        let (bos_id, eos_id) = self
            .tokenizer
            .special_ids_from_metadata(&mud.global_metadata);
        let vocab_len = self.tokenizer.id_to_token.len().max(1);
        let bos_id = crate::mud::sequence_pack::clamp_special_id(bos_id, vocab_len);
        let eos_id = crate::mud::sequence_pack::clamp_special_id(eos_id, vocab_len);
        // Resolved last-N (auto policy or env) for ops clarity
        println!(
            "  \x1b[1;36m[specials]\x1b[0m bos={bos_id} eos={eos_id} vocab={vocab_len} steps/chunk≈{} num_neg={} last_n_layers={} (env={})",
            crate::mud::sequence_pack::train_steps_per_chunk(batch_size),
            crate::mud::sequence_pack::train_num_negatives(),
            crate::mud::sequence_pack::train_last_n_layers(n_layers),
            std::env::var("MUD_TRAIN_LAST_N_LAYERS").unwrap_or_else(|_| "auto".into()),
        );
        let prefetch_bos = bos_id;
        let prefetch_eos = eos_id;

        std::thread::spawn(move || {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if let Some(last_core) = core_ids.last() {
                    core_affinity::set_for_current(*last_core);
                }
            }

            // Quick vs full use separate cache files so a 570MB full bin never blocks smoke.
            let global_bin_path = if prefetch_token_budget > 0 {
                std::path::PathBuf::from("training/corpus/global_corpus.quick.bin")
            } else {
                std::path::PathBuf::from("training/corpus/global_corpus.bin")
            };
            // Header v3: [n_src:u64][token_budget:u64][bos_eos:u64]  bos in low 32, eos in high 32
            const AOT_HEADER_LEN: usize = 24;
            let n_src_files = prefetch_text_files.len() as u64;
            let bos_eos_key: u64 = (prefetch_bos as u64) | ((prefetch_eos as u64) << 32);
            let mut rebuild_needed = true;

            // If a .tmp from a previous interrupted AOT build exists, remove it now so it
            // can never be mistaken for a valid corpus (P-17: fail-fast, no silent corruption).
            let tmp_bin_path_early = global_bin_path.with_extension("bin.tmp");
            if tmp_bin_path_early.exists() {
                let _ = std::fs::remove_file(&tmp_bin_path_early);
                println!("  [AOT] Stale .tmp detected and removed. Forcing rebuild.");
            }

            if let Ok(global_meta) = std::fs::metadata(&global_bin_path) {
                if let Ok(global_time) = global_meta.modified() {
                    use std::io::Read;
                    let header: Option<(u64, u64, u64)> = std::fs::File::open(&global_bin_path)
                        .ok()
                        .and_then(|mut f| {
                            let mut buf = [0u8; AOT_HEADER_LEN];
                            f.read_exact(&mut buf).ok().map(|_| {
                                let n = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                                let budget = u64::from_le_bytes(buf[8..16].try_into().unwrap());
                                let be = u64::from_le_bytes(buf[16..24].try_into().unwrap());
                                (n, budget, be)
                            })
                        });

                    if let Some((stored_count, stored_budget, stored_be)) = header {
                        if stored_count == n_src_files
                            && stored_budget == prefetch_token_budget
                            && stored_be == bos_eos_key
                        {
                            let bin_size = global_meta.len();
                            if bin_size <= AOT_HEADER_LEN as u64 {
                                println!(
                                    "  [AOT] Corpus cache truncated ({} bytes). Rebuilding...",
                                    bin_size
                                );
                            } else {
                                let mut all_older = true;
                                for file_path in &prefetch_text_files {
                                    if let Ok(txt_meta) = std::fs::metadata(file_path) {
                                        if let Ok(txt_time) = txt_meta.modified() {
                                            if txt_time > global_time {
                                                all_older = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if all_older {
                                    rebuild_needed = false;
                                }
                            }
                        } else {
                            println!(
                                "  [AOT] Cache key changed (files {}→{}, budget {}→{}, specials). Rebuilding...",
                                stored_count,
                                n_src_files,
                                stored_budget,
                                prefetch_token_budget
                            );
                        }
                    } else {
                        println!("  [AOT] Legacy/corrupt header. Rebuilding...");
                    }
                }
            }

            if rebuild_needed {
                // Write to a .tmp file, rename atomically on success — prevents corrupt
                // partial caches from being treated as valid on the next run (P-17).
                let _ = std::fs::create_dir_all("training/corpus");
                let tmp_bin_path = {
                    let mut p = global_bin_path.clone();
                    p.set_extension("bin.tmp");
                    p
                };
                let global_file = std::fs::File::create(&tmp_bin_path)
                    .expect("Failed to create global corpus tmp file");
                let mut writer = std::io::BufWriter::with_capacity(1024 * 1024 * 16, global_file);
                use std::io::Write;
                // Header v3
                let _ = writer.write_all(&n_src_files.to_le_bytes());
                let _ = writer.write_all(&prefetch_token_budget.to_le_bytes());
                let _ = writer.write_all(&bos_eos_key.to_le_bytes());

                let mut completed = true;
                let mut tokens_written: u64 = 0;
                let budget = prefetch_token_budget; // 0 = unlimited

                'aot_files: for file_path in prefetch_text_files.iter() {
                    if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                        completed = false;
                        break;
                    }
                    if budget > 0 && tokens_written >= budget {
                        println!(
                            "  [AOT] Token budget reached ({tokens_written}/{budget}). Stopping early."
                        );
                        break;
                    }
                    println!("  [AOT] Tokenizando: {} ...", file_path.display());

                    let mut file_tokens = Vec::new();
                    file_tokens.push(prefetch_bos);

                    let ext = file_path.extension().unwrap_or_default().to_string_lossy();
                    let is_rust = ext == "rs";
                    let is_markdown = ext == "md";

                    let mut full_text = String::new();
                    if is_rust {
                        full_text.push_str(&format!("File: {}\n```rust\n", file_path.display()));
                    } else if is_markdown {
                        full_text.push_str(&format!("File: {}\n```markdown\n", file_path.display()));
                    }

                    if let Ok(content) = std::fs::read_to_string(file_path) {
                        full_text.push_str(&content);
                    }

                    if is_rust || is_markdown {
                        full_text.push_str("\n```\n");
                    }

                    if !full_text.is_empty() {
                        let tokens = prefetch_tokenizer.encode(&full_text);
                        file_tokens.extend_from_slice(&tokens);
                    }

                    if budget > 0 {
                        let allow = budget.saturating_sub(tokens_written).max(2) as usize;
                        if file_tokens.len() > allow {
                            file_tokens.truncate(allow);
                        }
                    }

                    file_tokens.push(prefetch_eos);

                    if !file_tokens.is_empty() {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                file_tokens.as_ptr() as *const u8,
                                file_tokens.len() * 4,
                            )
                        };
                        let _ = writer.write_all(bytes);
                        tokens_written += file_tokens.len() as u64;
                    }

                    if budget > 0 && tokens_written >= budget {
                        println!(
                            "  [AOT] Token budget reached ({tokens_written}/{budget}). Stopping early."
                        );
                        break 'aot_files;
                    }
                }
                let _ = writer.flush();

                if completed {
                    // Atomic rename: only replace the real .bin on full success
                    std::fs::rename(&tmp_bin_path, &global_bin_path)
                        .expect("Failed to rename corpus tmp to bin");
                    println!(
                        "  [AOT] Corpus cache built successfully ({} tokens → {}).",
                        tokens_written,
                        global_bin_path.display()
                    );
                } else {
                    // Interrupted — delete the partial tmp so next run rebuilds
                    let _ = std::fs::remove_file(&tmp_bin_path);
                    println!("  [AOT] Build interrupted — partial cache discarded.");
                    return; // EXIT thread gracefully to avoid panicking below
                }
            } else {
                println!("  [AOT] Using cached corpus {}", global_bin_path.display());
            }

            let global_file =
                std::fs::File::open(&global_bin_path).expect("Failed to open global corpus file");
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .map(&global_file)
                    .expect("Failed to mmap global corpus")
            };
            let _ = mmap.advise(memmap2::Advice::Random);

            // Skip the 16-byte header v2 (legacy 8-byte bins already forced rebuild)
            let token_bytes = if mmap.len() >= AOT_HEADER_LEN {
                &mmap[AOT_HEADER_LEN..]
            } else {
                &mmap[..]
            };
            let total_tokens = token_bytes.len() / 4;
            let tokens_per_chunk = prefetch_tokens_per_chunk.max(64);
            // L-10: no-pad chunk ranges — include tail remainder (old path dropped it).
            let mut chunk_ranges: Vec<(usize, usize)> =
                crate::mud::sequence_pack::chunk_ranges_no_pad(total_tokens, tokens_per_chunk);

            // Small post-convert corpora: tile ranges up to MAX_CHUNKS so STE sees more steps.
            if let Some(cap) = prefetch_max_chunks {
                if !chunk_ranges.is_empty() && chunk_ranges.len() < cap {
                    let base = chunk_ranges.clone();
                    while chunk_ranges.len() < cap {
                        chunk_ranges.extend_from_slice(&base);
                    }
                    chunk_ranges.truncate(cap);
                    println!(
                        "  [AOT] Tiled small corpus {} → {} chunks (MAX_CHUNKS)",
                        base.len(),
                        chunk_ranges.len()
                    );
                } else if chunk_ranges.len() > cap {
                    chunk_ranges.truncate(cap);
                }
            }

            use rand::seq::SliceRandom;
            use rand::SeedableRng;

            let file_chunks = chunk_ranges.len();
            let global_path = std::path::PathBuf::from("GLOBAL_STREAM");

            let end_epoch = (resume_epoch - 1) + epochs;
            for epoch in resume_epoch..=end_epoch {
                if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                    break;
                }

                let mut rng = rand::rngs::StdRng::seed_from_u64(epoch as u64);
                chunk_ranges.shuffle(&mut rng);

                for (c_idx, &(start_idx, chunk_len)) in chunk_ranges.iter().enumerate() {
                    if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                        break;
                    }
                    if epoch == resume_epoch && c_idx < resume_chunk_idx {
                        continue;
                    }

                    if chunk_len < 2 {
                        continue;
                    }
                    let byte_start = start_idx * 4;
                    let byte_end = (start_idx + chunk_len) * 4;
                    if byte_end > token_bytes.len() {
                        continue;
                    }
                    let chunk_bytes = &token_bytes[byte_start..byte_end];

                    let chunk_tokens = unsafe {
                        std::slice::from_raw_parts(chunk_bytes.as_ptr() as *const u32, chunk_len)
                    }
                    .to_vec();

                    let _ = tx.send(PrefetchItem::Chunk {
                        epoch,
                        f_idx: 0,
                        file_path: global_path.clone(),
                        c_idx,
                        file_chunks,
                        tokens: chunk_tokens,
                    });
                }

                if tx.send(PrefetchItem::EndOfEpoch { epoch }).is_err() {
                    return;
                }
            }
        });

        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "corpus alignment session started (STE QAT)")
        );

        for item in rx {
            if SHOULD_TERMINATE.load(Ordering::SeqCst) {
                break;
            }

            match item {
                PrefetchItem::Chunk {
                    epoch,
                    f_idx: _f_idx,
                    file_path: _file_path,
                    c_idx,
                    file_chunks,
                    tokens,
                } => {
                    // Update tracking variables dynamically based on AOT truth
                    if total_chunks_per_epoch != file_chunks {
                        total_chunks_per_epoch = file_chunks;
                        total_chunks_all_epochs =
                            total_chunks_per_epoch * (resume_epoch - 1 + epochs);
                        global_chunks_processed = (epoch - 1) * total_chunks_per_epoch + c_idx;
                    }

                    global_chunks_processed += 1;
                    session_chunks_processed += 1;

                    // Hard stop for quick alignment (after count, before heavy train)
                    if let Some(cap) = max_chunks_cap {
                        if session_chunks_processed > cap {
                            println!(
                                "{}",
                                crate::mud::trainer_ui::note(
                                    "warn",
                                    &format!(
                                        "reached MUD_TRAIN_MAX_CHUNKS={cap} — saving & stopping"
                                    )
                                )
                            );
                            SHOULD_TERMINATE.store(true, Ordering::SeqCst);
                            break;
                        }
                    }

                    if tokens.len() < 2 {
                        continue;
                    }

                    let train_start = train_clock.get_or_insert_with(Instant::now);

                    let chunk_t0 = Instant::now();
                    let chunk_metrics = self.train_on_sequence(
                        &mut mud,
                        &mut shadow_emb,
                        &mut layers,
                        &mut shadow_layers,
                        &mut workspace,
                        &mut backward_ws,
                        &mut tapes,
                        &mut gradients,
                        &tokens,
                        batch_size,
                        vk_qat_storage.as_mut(),
                    )?;
                    let chunk_dt = chunk_t0.elapsed().as_secs_f32();
                    let chunk_loss = chunk_metrics.loss;

                    // ETA: compute AFTER train_on_sequence so elapsed includes
                    // the current chunk's wall-clock (avoids near-zero ETA on
                    // the first chunk when the clock just started).
                    let elapsed = train_start.elapsed().as_secs_f32();
                    let chunks_per_sec = if session_chunks_processed > 0 {
                        session_chunks_processed as f32 / elapsed.max(0.001)
                    } else {
                        0.0
                    };
                    let remaining_chunks =
                        total_chunks_all_epochs.saturating_sub(global_chunks_processed);
                    let eta = if chunks_per_sec > 0.0 {
                        Duration::from_secs_f32(remaining_chunks as f32 / chunks_per_sec)
                    } else {
                        Duration::ZERO
                    };
                    let total_secs = eta.as_secs();
                    let days = total_secs / 86400;
                    let hours = (total_secs % 86400) / 3600;
                    let mins = (total_secs % 3600) / 60;
                    let secs = total_secs % 60;
                    let eta_str = if days > 0 {
                        format!("{:02}d {:02}:{:02}:{:02}", days, hours, mins, secs)
                    } else {
                        format!("{:02}:{:02}:{:02}", hours, mins, secs)
                    };

                    if chunk_loss.is_nan() || chunk_loss.is_infinite() {
                        anyhow::bail!(
                            "\n{}",
                            crate::mud::trainer_ui::note(
                                "err",
                                "mathematical explosion detected (loss = NaN) — aborting early"
                            )
                        );
                    }

                    loss_history.push_back(chunk_loss);
                    if loss_history.len() > 100 {
                        loss_history.pop_front();
                    }

                    #[allow(clippy::manual_is_multiple_of)]
                    if loss_history.len() == 100 && global_chunks_processed % 100 == 0 {
                        let avg: f32 = loss_history.iter().sum::<f32>() / 100.0;
                        let var: f32 = loss_history
                            .iter()
                            .map(|&x| (x - avg) * (x - avg))
                            .sum::<f32>()
                            / 100.0;
                        if var < 1e-6 {
                            // Local Plateau Detected
                        }
                    }

                    let loss_vel = if loss_history.len() >= 2 {
                        loss_history[loss_history.len() - 2] - chunk_loss
                    } else {
                        0.0
                    };
                    let perplexity = chunk_loss.exp();

                    // Progress: every chunk in quick mode (MAX_CHUNKS), else every 25.
                    // Always print the first chunk so the session never looks hung post-AOT.
                    let progress_every: usize = max_chunks_cap
                        .map(|_| 1usize)
                        .or_else(|| {
                            std::env::var("MUD_TRAIN_PROGRESS_EVERY")
                                .ok()
                                .and_then(|s| s.parse().ok())
                                .filter(|&n| n > 0)
                        })
                        .unwrap_or(25);
                    let should_print = global_chunks_processed == 1
                        || global_chunks_processed.is_multiple_of(progress_every)
                        || max_chunks_cap
                            .map(|c| session_chunks_processed >= c)
                            .unwrap_or(false);
                    let should_telem = session_chunks_processed.is_multiple_of(telem_every);

                    if should_print || should_telem {
                        // LIVE stats from train_on_sequence (pre-BWD samples). Post-BWD
                        // workspace.registers / jepa_z are dead — do NOT re-read them here.
                        let avg_var_h = chunk_metrics.var_h;
                        let avg_var_j = chunk_metrics.var_j;
                        let avg_integral = chunk_metrics.jepa_integral;
                        let sigma_v_pct = chunk_metrics.sigma_v_pct;
                        let avg_cognitive = chunk_metrics.cognitive;

                        let denom = max_chunks_cap.unwrap_or(total_chunks_all_epochs).max(1) as f64;
                        let progress_pct = (session_chunks_processed
                            .min(max_chunks_cap.unwrap_or(session_chunks_processed))
                            as f64
                            / denom)
                            * 100.0;
                        let conf_pct = (-chunk_loss).exp() * 100.0;
                        // tok/s = token predictions per second. Each "step" is
                        // one next-token prediction (forward + backward), so
                        // steps_per_chunk already counts tokens processed.
                        let chunk_tps = steps_per_chunk_meta as f32 / chunk_dt.max(0.001);
                        let toks_per_sec = if chunk_tps > 0.0 && chunk_dt > 0.05 {
                            chunk_tps
                        } else {
                            chunks_per_sec * steps_per_chunk_meta as f32
                        };
                        let blocks_total = max_chunks_cap.unwrap_or(file_chunks);
                        let elapsed_s = train_clock
                            .map(|t| t.elapsed().as_secs_f32())
                            .unwrap_or(elapsed);

                        // File telemetry always (decoupled from console throttle)
                        if should_telem {
                            if let Some(ref mut f) = telemetry_file {
                                // Legacy cols 0..14 for train_telemetry TUI + v2 extensions
                                let _ = writeln!(
                                    f,
                                    "{} 1 {:.4} {:.4} {:.6} {:.6} {:.6} {:.6} 0.0 0.0 0.0 {:.6} {:.2} {:.6} 0.0 {:.1} {} {:.2} {:.2} {:.2} {} {} {}",
                                    global_chunks_processed,
                                    chunk_loss,
                                    perplexity,
                                    qat_lr,
                                    loss_vel,
                                    avg_var_h,
                                    avg_var_j,
                                    avg_integral,
                                    sigma_v_pct,
                                    avg_cognitive,
                                    toks_per_sec,
                                    steps_per_chunk_meta,
                                    elapsed_s,
                                    progress_pct,
                                    conf_pct,
                                    epoch,
                                    c_idx + 1,
                                    blocks_total,
                                );
                                let _ = f.flush();
                            }
                            // Machine-readable line: to stderr (terminal visibility) AND into
                            // mud_train_metrics.log so the live TUI (train_telemetry) can read it.
                            let dead = avg_var_h < 1e-5 && avg_cognitive < 1e-3;
                            let telem_line = format!(
                                "[TELEM] step={} epoch={} block={}/{} loss={:.4} ppl={:.1} lr={:.6} tok/s={:.1} steps={} elapsed={:.1}s prog={:.1}% conf={:.2}% varh={:.6} varj={:.6} jepa={:.4} σ={:.1}% cog={:.4} n_act={}{}",
                                global_chunks_processed,
                                epoch,
                                c_idx + 1,
                                blocks_total,
                                chunk_loss,
                                perplexity,
                                qat_lr,
                                toks_per_sec,
                                steps_per_chunk_meta,
                                elapsed_s,
                                progress_pct,
                                conf_pct,
                                avg_var_h,
                                avg_var_j,
                                avg_integral,
                                sigma_v_pct,
                                avg_cognitive,
                                chunk_metrics.n_samples,
                                if dead { " DEAD_ACT" } else { "" },
                            );
                            let is_tui = std::env::var("MUD_CIRCUIT_TUI").is_ok();
                            if !is_tui {
                                eprintln!("{}", telem_line);
                            }
                            if let Some(ref mut f) = telemetry_file {
                                let _ = writeln!(f, "{}", telem_line);
                                let _ = f.flush();
                            }
                        }

                        if should_print {
                            // Newline in quick mode so each block is visible; \r otherwise.
                            let is_tui = std::env::var("MUD_CIRCUIT_TUI").is_ok();
                            if !is_tui {
                                let eol = if max_chunks_cap.is_some() { "\n" } else { "" };
                                print!(
                                    "\r\x1b[2K  \x1b[1;33mEpoch\x1b[0m: {:02}/{} │ \x1b[1;33mBlock\x1b[0m: {:05}/{} │ \x1b[1;36mSpeed\x1b[0m: {:5.0} tk/s │ \x1b[1;35mLoss\x1b[0m: {:6.4} │ \x1b[1;32mConf\x1b[0m: {:5.2}% │ \x1b[1;32mProg\x1b[0m: {:5.2}% │ \x1b[1;31mETA\x1b[0m: {}{eol}",
                                    epoch,
                                    resume_epoch - 1 + epochs,
                                    c_idx + 1,
                                    blocks_total,
                                    toks_per_sec,
                                    chunk_loss,
                                    conf_pct,
                                    progress_pct,
                                    eta_str
                                );
                                let _ = std::io::Write::flush(&mut std::io::stdout());
                            }
                        }
                    }

                    let ckpt_every = checkpoint_every_chunks();
                    if global_chunks_processed > 0
                        && ckpt_every > 0
                        && global_chunks_processed.is_multiple_of(ckpt_every)
                    {
                        self.save_checkpoint(
                            &mut mud,
                            &mut shadow_emb,
                            &mut shadow_layers,
                            format!("chunk_{}", global_chunks_processed),
                            vk_qat_storage.as_mut(),
                        )?;
                    }
                }
                PrefetchItem::EndOfEpoch { epoch } => {
                    if let Some(vk_qat) = vk_qat_storage.as_mut() {
                        let _ = vk_qat.sync_all();
                    }

                    mud.global_metadata
                        .insert("trainer.current_epoch".to_string(), epoch.to_string());
                    mud.global_metadata.insert(
                        "trainer.current_chunk_idx".to_string(),
                        "0".to_string(), // Reset chunk on new epoch
                    );

                    // Epoch Alignment Complete
                    self.save_checkpoint(
                        &mut mud,
                        &mut shadow_emb,
                        &mut shadow_layers,
                        format!("epoch_{}", epoch),
                        vk_qat_storage.as_mut(),
                    )
                    .map_err(|e| anyhow::anyhow!("Failed to save epoch checkpoint: {}", e))?;
                }
            }
        }

        // Flush any remaining ops at the very end
        if let Some(vk_qat) = vk_qat_storage.as_mut() {
            let _ = vk_qat.sync_all();
        }

        self.sync_shadow_to_mud(
            &mut mud,
            &mut shadow_emb,
            &mut shadow_layers,
            vk_qat_storage.as_mut(),
            true, // final verify: warn loudly if the saved checkpoint is a no-op
        );
        let tmp_path = format!("{}.tmp", self.model_path);
        mud.save(&tmp_path)?;
        std::fs::rename(&tmp_path, &self.model_path)?;

        // Authoritative no-op check: compare final trained weights vs the captured input
        // hash (not the in-memory sync, which already reflects the changes).
        let out_hash = Self::hash_trained_weights(&mud);
        if out_hash == input_weights_hash {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "⚠ FINAL CHECKPOINT IS A NO-OP — trained weights are byte-identical to the input (hash match). The .mud did not change. Check LR / STE deadzone / resume-epoch / already-converged model."
                )
            );
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!(
                        "✓ final checkpoint persisted (weights hash {:#018x} → {:#018x})",
                        input_weights_hash, out_hash
                    )
                )
            );
        }

        println!(
            "{}",
            crate::mud::trainer_ui::note("ok", "alignment session completed.")
        );
        Ok(())
    }

    /// FNV-1a over the packed weight + PRQ-scale bytes of every trained `blk.N` layer.
    /// Captured at session start (before any sync mutates `mud`) and compared to the
    /// value at the final save, so a no-op checkpoint is detected against the ORIGINAL
    /// input — not against the already-synced in-memory tensors.
    fn hash_trained_weights(mud: &MudFile) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x100000001b3;
        let mut fnv = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        };
        let core = match mud.skills.get("core") {
            Some(c) => c,
            None => return h,
        };
        let n_layers = mud
            .global_metadata
            .get("num_hidden_layers")
            .or_else(|| mud.global_metadata.get("num_layers"))
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let last_n = crate::mud::sequence_pack::train_last_n_layers(n_layers.max(1));
        let first_train = n_layers.saturating_sub(last_n);
        for blk in first_train..n_layers {
            let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);
            let names = [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                &ffn.up,
                &ffn.gate,
                &ffn.down,
            ];
            for name in names.iter() {
                if let Some(t) = core.tensors.get(&format!("blk.{}.{}.weight", blk, name)) {
                    if let Some(ref od) = t.owned_data {
                        for &b in od.iter() {
                            fnv(b);
                        }
                    } else if !t.data_ptr.is_null() {
                        // Ternary2Bit: 1 byte per 2 elements; fall back safely if shape missing.
                        let nbytes = (t.shape.iter().product::<usize>() / 2).max(1);
                        let sl = unsafe { std::slice::from_raw_parts(t.data_ptr, nbytes) };
                        for &b in sl.iter() {
                            fnv(b);
                        }
                    }
                }
                if let Some(st) = core.tensors.get(&format!("blk.{}.{}.prq_scale", blk, name)) {
                    if let Some(ref od) = st.owned_data {
                        for &b in od.iter() {
                            fnv(b);
                        }
                    } else if !st.data_ptr.is_null() {
                        let sl = unsafe {
                            std::slice::from_raw_parts(
                                st.data_ptr,
                                st.shape.iter().product::<usize>() * 4,
                            )
                        };
                        for &b in sl.iter() {
                            fnv(b);
                        }
                    }
                }
            }
        }
        h
    }

    fn sync_shadow_to_mud(
        &self,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
        final_verify: bool,
    ) {
        let scales_only = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let freeze_emb = crate::mud::constants::train_freeze_emb();

        let debug_dw = std::env::var("MUD_TRAIN_DEBUG_DW")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // Always compute/verify the weight delta on the FINAL save (per user request:
        // guarantee the last epoch actually persisted and is not a no-op). Otherwise
        // only when MUD_TRAIN_DEBUG_DW=1 (diagnostic mode).
        let report = debug_dw || final_verify;
        let mut dbg_bits_changed: u64 = 0;
        let mut dbg_bytes_total: u64 = 0;
        let mut dbg_scale_absdelta: f64 = 0.0;
        let mut dbg_scale_absbase: f64 = 0.0;

        let core = mud.skills.get_mut("core").unwrap();
        let emb_tensor = core
            .tensors
            .get_mut("token_embd.weight")
            .expect("Missing token_embd.weight");

        // Never rewrite emb when frozen (or scales-only native import)
        if freeze_emb || scales_only {
            // scales-only emb: optionally refresh emb prq from shadow without re-pack — skip for freeze
        } else if emb_tensor.t_type == MudTensorType::Ternary2Bit {
            let rows = emb_tensor.shape[0];
            let cols = emb_tensor.shape[1];
            let mut scales = Vec::with_capacity(rows);
            let mut ternary_data = vec![0.0f32; shadow_emb.len()];

            for r in 0..rows {
                let start = r * cols;
                let absmean = shadow_emb[start..start + cols]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
                    / cols as f32;
                let s = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                scales.push(s);
                for c in 0..cols {
                    ternary_data[start + c] = (shadow_emb[start + c] / s).round().clamp(-1.0, 1.0);
                }
            }

            let packed = {
                let u32_count = ternary_data.len().div_ceil(8);
                let mut packed_vec = vec![0u32; u32_count];
                for i in 0..ternary_data.len() {
                    let bit = if ternary_data[i] > 0.5 {
                        0x1u32
                    } else if ternary_data[i] < -0.5 {
                        0xFu32
                    } else {
                        0x0u32
                    };
                    packed_vec[i / 8] |= bit << ((i % 8) * 4);
                }
                unsafe {
                    std::slice::from_raw_parts(
                        packed_vec.as_ptr() as *const u8,
                        packed_vec.len() * 4,
                    )
                }
                .to_vec()
            };
            if let Some(ref mut existing) = emb_tensor.owned_data {
                if existing.len() == packed.len() {
                    existing.copy_from_slice(&packed);
                } else {
                    emb_tensor.owned_data = Some(packed);
                }
            } else {
                emb_tensor.owned_data = Some(packed);
            }

            let scale_bytes = unsafe {
                std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
            }
            .to_vec();
            if let Some(scale_tensor) = core.tensors.get_mut("token_embd.prq_scale") {
                if let Some(ref mut existing) = scale_tensor.owned_data {
                    if existing.len() == scale_bytes.len() {
                        existing.copy_from_slice(&scale_bytes);
                    } else {
                        scale_tensor.owned_data = Some(scale_bytes);
                    }
                } else {
                    scale_tensor.owned_data = Some(scale_bytes);
                }
            } else {
                core.tensors.insert(
                    "token_embd.prq_scale".to_string(),
                    crate::mud::MudTensor {
                        name: "token_embd.prq_scale".to_string(),
                        t_type: MudTensorType::Float32,
                        shape: vec![rows],
                        data_ptr: std::ptr::null(),
                        offset: 0,
                        data_base: 0,
                        mmap: None,
                        owned_data: Some(scale_bytes),
                    },
                );
            }
        } else {
            let bytes = unsafe {
                std::slice::from_raw_parts(shadow_emb.as_ptr() as *const u8, shadow_emb.len() * 4)
            }
            .to_vec();
            if let Some(ref mut existing) = emb_tensor.owned_data {
                if existing.len() == bytes.len() {
                    existing.copy_from_slice(&bytes);
                } else {
                    emb_tensor.owned_data = Some(bytes);
                }
            } else {
                emb_tensor.owned_data = Some(bytes);
            }
        }

        for (blk, shadow) in shadow_layers.iter_mut().enumerate() {
            if shadow.is_empty() {
                continue; // frozen LAST_N: mmap ELUT untouched
            }
            let p = format!("blk.{}.", blk);
            let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);
            let mut update_tensor = |name: &str, weights: &[f32]| {
                if weights.is_empty() {
                    return;
                }
                // SCALES_ONLY: never rewrite ELUT codes — only refresh PRQ from shadow absmean
                if scales_only {
                    let (rows, cols) = match core.tensors.get(&format!("{}{}.weight", p, name)) {
                        Some(t) if t.t_type == MudTensorType::Ternary2Bit && t.shape.len() >= 2 => {
                            (t.shape[0], t.shape[1])
                        }
                        _ => return,
                    };
                    let mut new_scales = Vec::with_capacity(rows);
                    for r in 0..rows {
                        let start = r * cols;
                        let absmean = weights[start..start + cols]
                            .iter()
                            .map(|v| v.abs())
                            .sum::<f32>()
                            / cols as f32;
                        new_scales.push(absmean.max(1e-8));
                    }
                    let scale_bytes = unsafe {
                        std::slice::from_raw_parts(
                            new_scales.as_ptr() as *const u8,
                            new_scales.len() * 4,
                        )
                    }
                    .to_vec();
                    if let Some(scale_t) = core.tensors.get_mut(&format!("{}{}.prq_scale", p, name))
                    {
                        if let Some(ref mut existing) = scale_t.owned_data {
                            if existing.len() == scale_bytes.len() {
                                existing.copy_from_slice(&scale_bytes);
                            } else {
                                scale_t.owned_data = Some(scale_bytes);
                            }
                        } else {
                            scale_t.owned_data = Some(scale_bytes);
                        }
                        if let Some(ref owned) = scale_t.owned_data {
                            scale_t.data_ptr = owned.as_ptr();
                        }
                    }
                    return;
                }
                if let Some(t) = core.tensors.get_mut(&format!("{}{}.weight", p, name)) {
                    if t.t_type == MudTensorType::Ternary2Bit {
                        let rows = t.shape[0];
                        let cols = t.shape[1];
                        let mut new_scales = Vec::with_capacity(rows);
                        let mut ternary_data = vec![0.0f32; weights.len()];
                        for r in 0..rows {
                            let start = r * cols;
                            let absmean = weights[start..start + cols]
                                .iter()
                                .map(|v| v.abs())
                                .sum::<f32>()
                                / cols as f32;
                            let s = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                            new_scales.push(s);
                            for c in 0..cols {
                                ternary_data[start + c] =
                                    (weights[start + c] / s).round().clamp(-1.0, 1.0);
                            }
                        }
                        let packed = {
                            let u32_count = ternary_data.len().div_ceil(8);
                            let mut packed_vec = vec![0u32; u32_count];
                            for i in 0..ternary_data.len() {
                                let bit = if ternary_data[i] > 0.5 {
                                    0x1u32
                                } else if ternary_data[i] < -0.5 {
                                    0xFu32
                                } else {
                                    0x0u32
                                };
                                packed_vec[i / 8] |= bit << ((i % 8) * 4);
                            }
                            unsafe {
                                std::slice::from_raw_parts(
                                    packed_vec.as_ptr() as *const u8,
                                    packed_vec.len() * 4,
                                )
                            }
                            .to_vec()
                        };
                        // ΔW instrumentation runs every sync (not gated by `report`) so the
                        // live TUI ΔW panel is populated by default. The stdout print below
                        // stays gated by `report`.
                        {
                            let prev: Option<&[u8]> = if let Some(ref ex) = t.owned_data {
                                Some(ex.as_slice())
                            } else if !t.data_ptr.is_null() {
                                Some(unsafe {
                                    std::slice::from_raw_parts(t.data_ptr, packed.len())
                                })
                            } else {
                                None
                            };
                            if let Some(prev) = prev {
                                if prev.len() == packed.len() {
                                    for (a, b) in prev.iter().zip(packed.iter()) {
                                        dbg_bits_changed += (a != b) as u64;
                                    }
                                    dbg_bytes_total += packed.len() as u64;
                                }
                            }
                        }
                        if let Some(ref mut existing) = t.owned_data {
                            if existing.len() == packed.len() {
                                existing.copy_from_slice(&packed);
                            } else {
                                t.owned_data = Some(packed);
                            }
                        } else {
                            t.owned_data = Some(packed);
                        }

                        let scale_bytes = unsafe {
                            std::slice::from_raw_parts(
                                new_scales.as_ptr() as *const u8,
                                new_scales.len() * 4,
                            )
                        }
                        .to_vec();
                        if let Some(scale_t) =
                            core.tensors.get_mut(&format!("{}{}.prq_scale", p, name))
                        {
                            let prev_s: Option<&[f32]> = if let Some(ref ex) = scale_t.owned_data {
                                Some(unsafe {
                                    std::slice::from_raw_parts(
                                        ex.as_ptr() as *const f32,
                                        ex.len() / 4,
                                    )
                                })
                            } else if !scale_t.data_ptr.is_null() {
                                Some(unsafe {
                                    std::slice::from_raw_parts(
                                        scale_t.data_ptr as *const f32,
                                        new_scales.len(),
                                    )
                                })
                            } else {
                                None
                            };
                            if let Some(prev_s) = prev_s {
                                if prev_s.len() == new_scales.len() {
                                    for (a, b) in prev_s.iter().zip(new_scales.iter()) {
                                        dbg_scale_absdelta += (a - b).abs() as f64;
                                        dbg_scale_absbase += a.abs() as f64;
                                    }
                                }
                            }
                            if let Some(ref mut existing) = scale_t.owned_data {
                                if existing.len() == scale_bytes.len() {
                                    existing.copy_from_slice(&scale_bytes);
                                } else {
                                    scale_t.owned_data = Some(scale_bytes);
                                }
                            } else {
                                scale_t.owned_data = Some(scale_bytes);
                            }
                        } else {
                            core.tensors.insert(
                                format!("{}{}.prq_scale", p, name),
                                crate::mud::MudTensor {
                                    name: format!("{}{}.prq_scale", p, name),
                                    t_type: MudTensorType::Float32,
                                    shape: vec![rows],
                                    data_ptr: std::ptr::null(),
                                    offset: 0,
                                    data_base: 0,
                                    mmap: None,
                                    owned_data: Some(scale_bytes),
                                },
                            );
                        }
                    } else {
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                weights.as_ptr() as *const u8,
                                weights.len() * 4,
                            )
                        }
                        .to_vec();
                        if let Some(ref mut existing) = t.owned_data {
                            if existing.len() == bytes.len() {
                                existing.copy_from_slice(&bytes);
                            } else {
                                t.owned_data = Some(bytes);
                            }
                        } else {
                            t.owned_data = Some(bytes);
                        }
                    }
                }
            };

            macro_rules! read_shadow {
                ($name_suffix:expr, $cpu_weights:expr) => {{
                    if let Some(vk) = vk_qat.as_deref() {
                        let name = format!("blk.{}.{}", blk, $name_suffix);
                        unsafe {
                            vk.readback_shadow(&name, $cpu_weights);
                        }
                    }
                    $cpu_weights as &[f32]
                }};
            }

            update_tensor("attn_q", read_shadow!("attn_q", &mut shadow.q_w));
            update_tensor("attn_k", read_shadow!("attn_k", &mut shadow.k_w));
            update_tensor("attn_v", read_shadow!("attn_v", &mut shadow.v_w));
            update_tensor("attn_output", read_shadow!("attn_output", &mut shadow.o_w));
            // FFN: w3=up / w1=gate / w2=down (or up/gate alt); MUD_TRAIN_EXPERT
            update_tensor(&ffn.up, read_shadow!(ffn.up.as_str(), &mut shadow.ffn_up_w));
            update_tensor(
                &ffn.gate,
                read_shadow!(ffn.gate.as_str(), &mut shadow.ffn_gate_w),
            );
            update_tensor(
                &ffn.down,
                read_shadow!(ffn.down.as_str(), &mut shadow.ffn_down_w),
            );
        }

        // ΔW summary (computed every sync so the live TUI ΔW panel is populated
        // by default). The stdout print stays gated by `report`.
        let pct = if dbg_bytes_total > 0 {
            100.0 * dbg_bits_changed as f64 / dbg_bytes_total as f64
        } else {
            0.0
        };
        let scale_pct = if dbg_scale_absbase > 0.0 {
            100.0 * dbg_scale_absdelta / dbg_scale_absbase
        } else {
            0.0
        };
        let moved = dbg_bits_changed > 0 || dbg_scale_absdelta > 0.0;

        // Machine-parseable [DW] line for the live TUI (train_telemetry).
        {
            use std::io::Write;
            if let Ok(mut dwf) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("mud_train_metrics.log")
            {
                let _ = writeln!(
                    dwf,
                    "[DW] bytes={} total={} prq={:.4} moved={}",
                    dbg_bits_changed,
                    dbg_bytes_total,
                    scale_pct,
                    if moved { 1 } else { 0 }
                );
                let _ = dwf.flush();
            }
        }

        if report {
            // Per-sync ΔW report (vs the immediately-previous in-memory state). The
            // authoritative final no-op check is done by the caller via input-vs-output
            // hash, since this in-memory compare sees already-synced tensors.
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    if moved { "ok" } else { "warn" },
                    &format!(
                        "ΔW: ternary {dbg_bits_changed}/{dbg_bytes_total} bytes ({pct:.4}%) | PRQ scale Σ|Δ|/Σ|s|={scale_pct:.4}%{}",
                        if moved { "" } else { " — no change this sync" }
                    )
                )
            );
        }
    }

    fn save_checkpoint(
        &self,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        _suffix: String,
        mut vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
    ) -> anyhow::Result<()> {
        let checkpoint_name = format!("{}/model_latest_checkpoint.mud", CHECKPOINT_DIR);
        if let Some(vk) = vk_qat.as_deref_mut() {
            let _ = vk.sync_all();
        }
        self.sync_shadow_to_mud(mud, shadow_emb, shadow_layers, vk_qat, false);
        let tmp_path = format!("{}.tmp", checkpoint_name);
        mud.save(&tmp_path)?;
        std::fs::rename(&tmp_path, &checkpoint_name)?;

        // Print the log line matching what was historically printed
        // [Checkpoint Saved]
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_on_sequence(
        &self,
        _mud: &mut MudFile,
        shadow_emb: &mut [f32],
        layers: &mut [crate::mud::slime_forward::SlimeLayer],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        workspace: &mut crate::mud::slime::SlimeWorkspace,
        backward_ws: &mut crate::mud::slime_backward::SlimeBackwardWorkspace,
        tapes: &mut [crate::mud::slime_backward::SlimeLayerTape],
        gradients: &mut [crate::mud::slime_backward::SlimeLayerGradients],
        tokens: &[u32],
        batch_size: usize,
        vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
    ) -> anyhow::Result<TrainChunkMetrics> {
        // L-05: do NOT sync at entry. Previous step's GPU work overlaps this chunk's
        // Forward+Backward; fence+packed readback runs inside step_async_deferred
        // after backward (and at epoch/checkpoint via sync_all).

        let lr = crate::mud::constants::qat_learning_rate();
        let hidden_size = workspace.hidden_size;
        // Lazy emb (FREEZE_EMB): shadow_emb empty — vocab from mud metadata / tensor shape
        let vocab_size = if !shadow_emb.is_empty() {
            shadow_emb.len() / hidden_size.max(1)
        } else {
            _mud.skills
                .get("core")
                .and_then(|c| c.tensors.get("token_embd.weight"))
                .map(|t| t.shape[0])
                .or_else(|| {
                    _mud.global_metadata
                        .get("vocab_size")
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(1)
        };
        let emb_lazy = shadow_emb.is_empty();
        // Model-native EOS (SmolLM2 = 0). Never use Llama 128001 when OOV.
        let (_bos, eos_raw) = self
            .tokenizer
            .special_ids_from_metadata(&_mud.global_metadata);
        let eos = crate::mud::sequence_pack::clamp_special_id(eos_raw, vocab_size);

        // Stream D: full-seq (causal windows, pos>0, KV grows) is default.
        // MUD_TRAIN_FULL_SEQ=0 → classic L-10 independent pairs at pos=0.
        let full_seq = crate::mud::sequence_pack::train_full_seq_enabled();
        let seq_len = crate::mud::sequence_pack::train_seq_len()
            .min(workspace.dense_kv_cap.max(2))
            .min(workspace.max_pos.max(2));
        // Stream H: long windows auto-enable L-15 segmented ckpt when unset
        crate::mud::sequence_pack::maybe_enable_grad_ckpt_for_long_seq(seq_len);

        // Align mode densifies gradients per AOT chunk (batch × 4, clamped).
        let target_preds = crate::mud::sequence_pack::train_steps_per_chunk(batch_size);

        // Stream G: multi-expert pool for round-robin dense STE
        let n_layers_meta = layers.len();
        let moe_pool = {
            let core = _mud.skills.get("core");
            core.map(|c| {
                crate::mud::moe_train::discover_train_expert_pool(&c.tensors, n_layers_meta)
            })
            .unwrap_or_else(|| vec![0])
        };
        if crate::mud::moe_train::moe_train_enabled() {
            crate::mud::moe_train::reset_utilization();
        }

        // Build step list: (input_id, target_id, pos, reset_kv)
        let mut steps: Vec<(usize, usize, usize, bool)> = Vec::new();
        if full_seq {
            let n_win = crate::mud::sequence_pack::windows_for_target_preds(target_preds, seq_len);
            let windows =
                crate::mud::sequence_pack::windows_from_stream(tokens, n_win, seq_len, eos);
            let mut n_pred = 0usize;
            'windows: for w in windows {
                let mut first = true;
                for pos in 0..w.n_preds() {
                    if n_pred >= target_preds {
                        break 'windows;
                    }
                    let abs_i = w.start + pos;
                    if abs_i + 1 >= tokens.len() {
                        break;
                    }
                    let inp_tok = tokens[abs_i];
                    if inp_tok == eos {
                        break; // do not train past EOS inside a window
                    }
                    let input_id = inp_tok as usize;
                    let target_id = tokens[abs_i + 1] as usize;
                    if input_id >= vocab_size || target_id >= vocab_size {
                        continue;
                    }
                    // Clamp pos to dense ring / logical max
                    let pos_clamped = pos.min(workspace.max_pos.saturating_sub(1));
                    steps.push((input_id, target_id, pos_clamped, first));
                    first = false;
                    n_pred += 1;
                }
            }
        }
        if steps.is_empty() {
            // Fallback: L-10 independent pairs at pos=0 (also when full-seq finds nothing)
            let pairs =
                crate::mud::sequence_pack::pairs_from_stream(tokens, target_preds, vocab_size, eos);
            steps = pairs
                .into_iter()
                .map(|(a, b)| (a, b, 0usize, true))
                .collect();
        }

        let mut total_loss = 0.0f32;
        let mut pair_count = 0;
        let eps = 1e-6; // rms norm eps
        let mut act_accum = ActStatsAccum::default();

        // Sampled softmax size (speed: fewer negs in align/quick)
        let num_neg = crate::mud::sequence_pack::train_num_negatives();
        let num_classes = 1 + num_neg;
        // Speed: only BWD+opt last N layers (forward still full stack)
        let last_n = crate::mud::sequence_pack::train_last_n_layers(layers.len());
        let first_train_layer = layers.len().saturating_sub(last_n);

        let mut x_data = vec![0.0f32; hidden_size];
        let mut pre_norm_x = vec![0.0f32; hidden_size];
        let mut final_x = vec![0.0f32; hidden_size];
        let mut class_embs_flat = vec![0.0f32; num_classes * hidden_size];
        let mut class_logits = vec![0.0f32; num_classes];
        let mut prob_q = vec![0.0f32; num_classes];
        let mut d_logits = vec![0.0f32; num_classes];
        let mut grad_in = vec![0.0f32; hidden_size];
        let mut grad_out = vec![0.0f32; hidden_size];
        let mut neg_ids = vec![0usize; num_neg];

        // Phase 2 (STP): per-window ring of recent top-of-stack residual states
        // (pre-output-norm `matmul_accum`). STP is train-only, zero inference cost.
        let stp_on = crate::mud::stp_loss::stp_enabled();
        let stp_lambda = crate::mud::stp_loss::stp_lambda();
        const STP_HIST: usize = 4; // small ring; triples sampled from it
        let mut stp_hist: Vec<Vec<f32>> = if stp_on {
            (0..STP_HIST).map(|_| vec![0.0f32; hidden_size]).collect()
        } else {
            Vec::new()
        };
        let mut stp_hist_len = 0usize; // valid entries in current window
        let mut stp_hist_head = 0usize; // next write slot (ring)
        let mut stp_grad_t = if stp_on {
            vec![0.0f32; hidden_size]
        } else {
            Vec::new()
        };
        let mut stp_grad_r = if stp_on {
            vec![0.0f32; hidden_size]
        } else {
            Vec::new()
        };
        let mut stp_grad_s = if stp_on {
            vec![0.0f32; hidden_size]
        } else {
            Vec::new()
        };
        let mut stp_loss_accum = 0.0f32;
        let mut stp_triples = 0usize;
        let mut stp_rng: u64 = 0x9E3779B97F4A7C15;

        for g in gradients.iter_mut() {
            g.reset();
        }

        for (input_id, target_id, pos, reset_kv) in steps.iter().copied() {
            if crate::mud::corpus_trainer::SHOULD_TERMINATE
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                break;
            }

            if reset_kv {
                // New window / independent pair: wipe KV + HCA + JEPA context
                workspace.clear_kv_all();
                workspace.jepa_mu.fill(0.0);
                workspace.jepa_inv_sigma.fill(0.0);
                workspace.jepa_var_ema.fill(0.0);
                // STP trajectory is per-window; reset the history ring.
                stp_hist_len = 0;
                stp_hist_head = 0;
            }

            workspace.clear_registers();
            for t in tapes.iter_mut() {
                t.reset();
            }

            // 1. Load embedding. FREEZE_EMB/lazy: already ternary from mmap (no re-STE).
            // Shadow FP32 path: re-quantize to simulate Ternary Shock for trainable emb.
            load_emb_row_into(
                _mud,
                shadow_emb,
                emb_lazy,
                hidden_size,
                input_id,
                &mut x_data[..hidden_size],
            );
            if !emb_lazy {
                let absmean_x = x_data.iter().map(|v| v.abs()).sum::<f32>() / hidden_size as f32;
                let scale_x = (absmean_x * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                for v in &mut x_data {
                    *v = (*v / scale_x).round().clamp(-1.0, 1.0) * scale_x;
                }
            }

            // Stream G / G+: pick expert after STE emb is ready (hash needs x)
            if crate::mud::moe_train::moe_train_enabled() {
                let eid = match crate::mud::moe_train::moe_train_mode() {
                    crate::mud::moe_train::MoeTrainMode::Hash => {
                        let (e, _route) = crate::mud::moe_train::begin_step_hash(
                            &moe_pool,
                            &x_data[..hidden_size],
                            crate::mud::moe_load::default_top_k(),
                        );
                        e
                    }
                    _ => crate::mud::moe_train::begin_step(&moe_pool),
                };
                if let Some(core) = _mud.skills.get("core") {
                    for (blk, layer) in layers.iter_mut().enumerate() {
                        let ffn =
                            crate::mud::moe_load::resolve_expert_ffn_names(&core.tensors, blk, eid)
                                .unwrap_or(crate::mud::moe_load::ExpertFfnNames {
                                    up: format!("expert.{eid}.w3"),
                                    gate: format!("expert.{eid}.w1"),
                                    down: format!("expert.{eid}.w2"),
                                });
                        let p = format!("blk.{blk}.");
                        let t = |name: &str| -> *const u8 {
                            core.tensors
                                .get(&format!("{p}{name}.weight"))
                                .map(|t| t.data_ptr)
                                .unwrap_or(std::ptr::null())
                        };
                        let ts = |name: &str| -> *const f32 {
                            core.tensors
                                .get(&format!("{p}{name}.prq_scale"))
                                .map(|t| t.data_ptr as *const f32)
                                .unwrap_or(std::ptr::null())
                        };
                        layer.ffn_up_w = t(&ffn.up);
                        layer.ffn_gate_w = t(&ffn.gate);
                        layer.ffn_down_w = t(&ffn.down);
                        layer.ffn_up_scales = ts(&ffn.up);
                        layer.ffn_gate_scales = ts(&ffn.gate);
                        layer.ffn_down_scales = ts(&ffn.down);
                    }
                }
            }

            for (i, &x_val) in x_data.iter().enumerate().take(hidden_size) {
                crate::mud::slime::SlimeRegister::init_from_embed(
                    &mut workspace.registers[i],
                    &mut workspace.jepa_z,
                    i,
                    hidden_size,
                    layers.len(),
                    x_val,
                    true,
                );
            }

            // 2. Forward pass through layers (L-15: optional activation checkpointing)
            // Stream D: use causal `pos` so RoPE + KV history match inference.
            // Stream H: residual bank when MUD_GRAD_CKPT_RESIDUAL=1 + segmented.
            // FWD_LAST_N: skip lower layers for 1260P seating speed (approximate residual).
            let ckpt = crate::mud::grad_checkpoint::CheckpointPolicy::resolve();
            let n_layers = layers.len();
            let fwd_n = crate::mud::sequence_pack::train_fwd_last_n_layers(n_layers);
            let first_fwd = n_layers.saturating_sub(fwd_n);
            // one-shot log per process would spam — only first step of first chunk if env debug
            if first_fwd > 0 && pair_count == 0 {
                println!(
                    "{}",
                    crate::mud::trainer_ui::note(
                        "ram",
                        &format!(
                            "FWD_LAST_N={fwd_n} first_fwd={first_fwd}/{n_layers} (approx residual; 1260P seating speed)"
                        )
                    )
                );
            }
            let use_residual_bank = ckpt.is_segmented()
                && crate::mud::grad_checkpoint::residual_bank_recompute_enabled()
                && first_fwd == 0; // residual bank assumes full stack
            let mut residual_bank = if use_residual_bank {
                Some(crate::mud::grad_checkpoint::ResidualBank::with_workspace(
                    ckpt.num_residual_slots(n_layers),
                    workspace,
                ))
            } else {
                None
            };
            // Keep quantized emb for exact recompute when MUD_GRAD_CKPT=1 (fallback)
            let emb_ckpt: Option<Vec<f32>> = if ckpt.is_segmented() && first_fwd == 0 {
                Some(x_data[..hidden_size].to_vec())
            } else {
                None
            };
            for l_idx in first_fwd..n_layers {
                let layer = &layers[l_idx];
                if let Some(ref mut bank) = residual_bank {
                    // Save residual at each segment boundary (before layer runs)
                    if l_idx == ckpt.segment_start(l_idx) {
                        bank.save_from_workspace(ckpt.segment_of(l_idx), workspace);
                    }
                }
                // Tape only needed for thawed BWD layers
                let tape = if l_idx >= first_train_layer {
                    Some(&mut tapes[l_idx])
                } else {
                    None
                };
                crate::mud::slime_forward::evaluate_slime_block(
                    layer, l_idx, workspace, pos, eps, tape,
                );
            }
            if let Some(ref mut bank) = residual_bank {
                // Final residual after last layer (last slot)
                let last_slot = ckpt.num_residual_slots(n_layers).saturating_sub(1);
                bank.save_from_workspace(last_slot, workspace);
            }
            // Segmented: free all tapes after forward — recompute on backward
            if ckpt.is_segmented() {
                for t in tapes.iter_mut() {
                    crate::mud::grad_checkpoint::discard_tape(t);
                }
            }

            for (i, val) in pre_norm_x.iter_mut().enumerate().take(hidden_size) {
                *val = workspace.registers[i].matmul_accum;
            }

            let output_norm_w = _mud
                .skills
                .get("core")
                .and_then(|c| c.tensors.get("output_norm.weight"))
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null());
            crate::mud::slime_forward::apply_output_norm(workspace, output_norm_w, eps);

            for (i, val) in final_x.iter_mut().enumerate().take(hidden_size) {
                *val = workspace.registers[i].matmul_accum;
            }

            // ── LIVE activation telemetry (must be pre-BWD) ──────────────────
            // Post-BWD registers are garbage; sample here only.
            // VarH: max(stddev pre-output-norm, stddev post-norm, RMS as floor)
            let var_h_pre = slice_stddev(&pre_norm_x[..hidden_size]);
            let var_h_post = slice_stddev(&final_x[..hidden_size]);
            let mut sum_sq = 0.0f32;
            let mut abs_sum = 0.0f32;
            for &v in pre_norm_x.iter().take(hidden_size) {
                sum_sq += v * v;
                abs_sum += v.abs();
            }
            let rms_h = (sum_sq / hidden_size as f32).sqrt();
            let mean_abs_h = abs_sum / hidden_size as f32;
            // Prefer spatial stddev; if flat but non-zero RMS, report RMS so metric is not "dead zero"
            let var_h =
                var_h_pre
                    .max(var_h_post)
                    .max(if mean_abs_h > 1e-8 { rms_h * 0.1 } else { 0.0 });
            // VarJ: stddev of JEPA z-trackers + jepa_var_ema blend
            let var_j_z = slice_stddev(&workspace.jepa_z);
            let var_j_ema = if workspace.jepa_var_ema.is_empty() {
                0.0
            } else {
                let s: f32 = workspace.jepa_var_ema.iter().map(|v| v.abs()).sum();
                s / workspace.jepa_var_ema.len() as f32
            };
            let var_j = var_j_z.max(var_j_ema);
            // JEPA gate energy
            let mut e_sum = 0.0f32;
            for r in workspace.registers.iter().take(hidden_size) {
                e_sum += r.jepa_energy;
            }
            let mut jepa_i = e_sum / hidden_size as f32;
            if workspace.jepa_integral.is_finite() {
                jepa_i = 0.5 * jepa_i + 0.5 * workspace.jepa_integral;
            }
            let sigma_v = 100.0 / (1.0 + (-jepa_i).exp());
            // Cognitive: mean |pre_norm| × 100 (mai_bytes was never written → always 0)
            let cognitive = mean_abs_h * 100.0;
            act_accum.push(var_h, var_j, jepa_i, sigma_v, cognitive);

            // 3. Sampled Softmax (1 target + num_neg random negatives; env MUD_TRAIN_NUM_NEG)
            let mut rng_state = input_id.wrapping_mul(1664525).wrapping_add(target_id);
            let mut neg_count = 0;
            // Cap rejection sampling to avoid infinite loops on tiny vocabs
            let mut attempts = 0usize;
            while neg_count < num_neg && attempts < num_neg * 8 + 64 {
                attempts += 1;
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let neg = rng_state % vocab_size;
                if neg != target_id && !neg_ids[0..neg_count].contains(&neg) {
                    neg_ids[neg_count] = neg;
                    neg_count += 1;
                }
            }
            // If short (tiny vocab), pad with wraps
            while neg_count < num_neg {
                neg_ids[neg_count] = (neg_count + 1) % vocab_size.max(1);
                if neg_ids[neg_count] == target_id {
                    neg_ids[neg_count] = (neg_ids[neg_count] + 1) % vocab_size.max(1);
                }
                neg_count += 1;
            }

            // 1) Target emb → index 0
            load_emb_row_into(
                _mud,
                shadow_emb,
                emb_lazy,
                hidden_size,
                target_id,
                &mut class_embs_flat[0..hidden_size],
            );
            let scale = if emb_lazy {
                let absmean = class_embs_flat[0..hidden_size]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
                    / (hidden_size as f32);
                absmean.max(1e-8)
            } else {
                let absmean = class_embs_flat[0..hidden_size]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
                    / (hidden_size as f32);
                let scale = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
                for v in &mut class_embs_flat[0..hidden_size] {
                    *v = (*v / scale).round().clamp(-1.0, 1.0) * scale;
                }
                scale
            };

            // 2) Neg embs (lazy mmap or shadow)
            for (idx, &neg) in neg_ids.iter().take(num_neg).enumerate() {
                let c_start = (idx + 1) * hidden_size;
                load_emb_row_into(
                    _mud,
                    shadow_emb,
                    emb_lazy,
                    hidden_size,
                    neg,
                    &mut class_embs_flat[c_start..c_start + hidden_size],
                );
            }

            // Logit scale_up: reuse target STE scale path; avoid full-vocab emb RMS each step
            let scale_up = (1.0 / scale.max(1e-8)).clamp(1.0, 128.0);

            let temp_scale = 1.0 / (hidden_size as f32).sqrt();
            for (i, class_logit) in class_logits.iter_mut().enumerate().take(num_classes) {
                let c_start = i * hidden_size;
                let dot = unsafe {
                    forge_autograd::avx_math::dot_product_avx2(
                        &final_x[..hidden_size],
                        &class_embs_flat[c_start..c_start + hidden_size],
                    )
                };
                *class_logit = dot * scale_up * temp_scale;
            }

            // Softmax
            let max_logit = class_logits[0..num_classes]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0;
            for i in 0..num_classes {
                prob_q[i] = (class_logits[i] - max_logit).exp();
                exp_sum += prob_q[i];
            }
            for prob in prob_q.iter_mut().take(num_classes) {
                *prob /= exp_sum;
            }

            // Cross Entropy Loss (target is always index 0)
            let loss = -prob_q[0].ln();
            total_loss += loss;
            pair_count += 1;

            // Calculate gradients
            for (i, d_l) in d_logits.iter_mut().enumerate().take(num_classes) {
                *d_l = (prob_q[i] - if i == 0 { 1.0 } else { 0.0 }) * temp_scale;
            }

            grad_in.fill(0.0); // gradient of x
            for (i, d_l) in d_logits.iter().enumerate().take(num_classes) {
                let c_start = i * hidden_size;
                unsafe {
                    forge_autograd::avx_math::axpy_avx2(
                        &mut grad_in[..hidden_size],
                        *d_l,
                        &class_embs_flat[c_start..c_start + hidden_size],
                    );
                }
            }

            if !output_norm_w.is_null() {
                let mut sq_sum = 0.0;
                for &v in &pre_norm_x {
                    sq_sum += v * v;
                }
                let rms = (sq_sum / hidden_size as f32 + eps).sqrt();
                let rms_inv = 1.0 / rms;

                let mut sum_dy_xnorm = 0.0;
                for i in 0..hidden_size {
                    let w = unsafe { *output_norm_w.add(i) };
                    let x_norm = pre_norm_x[i] * rms_inv;
                    sum_dy_xnorm += grad_in[i] * w * x_norm;
                }
                let mean_dy_xnorm = sum_dy_xnorm / hidden_size as f32;

                for i in 0..hidden_size {
                    let w = unsafe { *output_norm_w.add(i) };
                    let x_norm = pre_norm_x[i] * rms_inv;
                    let dx = (w * rms_inv) * (grad_in[i] - x_norm * mean_dy_xnorm);
                    grad_in[i] = dx;
                }
            }

            grad_out.fill(0.0);

            // ── Phase 2: STP trajectory aux loss (train-only, zero inference cost) ──
            // Current position is `t`; sample `s < r < t` from the window history ring.
            // Inject λ·∂L_STP/∂h_t into the top-of-stack residual grad (grad wrt
            // pre_norm_x). h_s,h_r treated as constants this step (stochastic estimator);
            // over the window every position gets STP signal as it becomes `t`.
            if stp_on && stp_hist_len >= 2 {
                // Pick two distinct history indices (r more recent than s when possible).
                stp_rng = stp_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let a = (stp_rng >> 33) as usize % stp_hist_len;
                stp_rng = stp_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let mut b = (stp_rng >> 33) as usize % stp_hist_len;
                if b == a {
                    b = (b + 1) % stp_hist_len;
                }
                // Map ring slots to chronological order: older -> s, newer -> r.
                let oldest = if stp_hist_len < STP_HIST {
                    0
                } else {
                    stp_hist_head
                };
                let age = |k: usize| (k + STP_HIST - oldest) % STP_HIST;
                let (idx_s, idx_r) = {
                    let (ia, ib) = (
                        (oldest + a.min(stp_hist_len - 1)) % STP_HIST,
                        (oldest + b.min(stp_hist_len - 1)) % STP_HIST,
                    );
                    if age(ia) <= age(ib) {
                        (ia, ib)
                    } else {
                        (ib, ia)
                    }
                };
                if idx_s != idx_r {
                    stp_grad_t.fill(0.0);
                    stp_grad_r.fill(0.0);
                    stp_grad_s.fill(0.0);
                    // Only dL/dh_t is applied (s,r are frozen history this step).
                    let l_stp = crate::mud::stp_loss::stp_loss_and_grad(
                        &stp_hist[idx_s],
                        &stp_hist[idx_r],
                        &pre_norm_x[..hidden_size],
                        stp_lambda,
                        &mut stp_grad_s, // dL/dh_s (discarded: s is history)
                        &mut stp_grad_r, // dL/dh_r (discarded: r is history)
                        &mut stp_grad_t, // dL/dh_t -> current position
                    );
                    if stp_grad_t.iter().all(|v| v.is_finite()) {
                        for i in 0..hidden_size {
                            grad_in[i] += stp_grad_t[i];
                        }
                        stp_loss_accum += l_stp;
                        stp_triples += 1;
                    }
                }
            }
            // Push current top-of-stack residual into the window ring for future triples.
            if stp_on {
                stp_hist[stp_hist_head].copy_from_slice(&pre_norm_x[..hidden_size]);
                stp_hist_head = (stp_hist_head + 1) % STP_HIST;
                if stp_hist_len < STP_HIST {
                    stp_hist_len += 1;
                }
            }

            if grad_in.iter().all(|v| v.is_finite()) {
                let norm_sq: f32 = grad_in.iter().map(|&g| g * g).sum();
                let clip = if norm_sq.sqrt() > 1.0 {
                    1.0 / norm_sq.sqrt()
                } else {
                    1.0
                };
                for v in &mut grad_in {
                    *v *= clip;
                }

                // L-15 / H: reverse layers; residual-bank recompute or emb fallback
                // Speed path: stop BWD below first_train_layer (frozen lower blocks)
                let mut l_idx = n_layers;
                let mut last_recomputed_seg: Option<usize> = None;
                while l_idx > first_train_layer {
                    l_idx -= 1;
                    if ckpt.is_segmented() && !tapes[l_idx].valid {
                        let seg = ckpt.segment_of(l_idx);
                        // Only recompute each segment once when walking reverse
                        if last_recomputed_seg != Some(seg) {
                            if use_residual_bank {
                                if let Some(ref bank) = residual_bank {
                                    crate::mud::grad_checkpoint::recompute_from_residual_bank(
                                        bank, layers, l_idx, workspace, tapes, ckpt, eps, pos,
                                    );
                                }
                            } else if let Some(ref emb) = emb_ckpt {
                                let end = ckpt.segment_end(l_idx, n_layers);
                                crate::mud::grad_checkpoint::recompute_from_embedding(
                                    emb, layers, end, workspace, tapes, eps, pos,
                                );
                            }
                            last_recomputed_seg = Some(seg);
                        }
                    }
                    crate::mud::slime_backward::backward_slime_block(
                        &layers[l_idx],
                        workspace,
                        backward_ws,
                        &tapes[l_idx],
                        &mut gradients[l_idx],
                        &grad_in,
                        &mut grad_out,
                    );
                    grad_in.copy_from_slice(&grad_out);
                    if ckpt.is_segmented() {
                        crate::mud::grad_checkpoint::discard_tape(&mut tapes[l_idx]);
                    }
                }

                // grad_in → input emb (skip if freeze / lazy)
                if !emb_lazy
                    && !shadow_emb.is_empty()
                    && !crate::mud::constants::train_freeze_emb()
                    && grad_in.iter().all(|v| v.is_finite())
                {
                    let norm_sq: f32 = grad_in.iter().map(|&g| g * g).sum();
                    let clip = if norm_sq.sqrt() > 1.0 {
                        1.0 / norm_sq.sqrt()
                    } else {
                        1.0
                    };
                    for v in &mut grad_in {
                        *v *= clip;
                    }
                    let target_slice =
                        &mut shadow_emb[input_id * hidden_size..(input_id + 1) * hidden_size];
                    unsafe {
                        forge_autograd::avx_math::axpy_avx2(target_slice, -lr, &grad_in);
                    }
                }
            }

            // 5. Update target and negative embeddings (skip if freeze / lazy emb)
            let freeze_emb = emb_lazy || crate::mud::constants::train_freeze_emb();
            if !freeze_emb && !shadow_emb.is_empty() && final_x.iter().all(|v| v.is_finite()) {
                let norm_sq_x: f32 = final_x.iter().map(|&x| x * x).sum();
                let x_norm = norm_sq_x.sqrt();

                let mut target_clip = 1.0;
                let target_dl = d_logits[0];
                if target_dl.abs() * x_norm > 1.0 {
                    target_clip = 1.0 / (target_dl.abs() * x_norm).max(1e-8);
                }
                let target_row =
                    &mut shadow_emb[target_id * hidden_size..(target_id + 1) * hidden_size];
                unsafe {
                    forge_autograd::avx_math::axpy_avx2(
                        target_row,
                        -lr * target_clip * target_dl,
                        &final_x,
                    );
                }

                for (ni, &neg_id) in neg_ids.iter().enumerate() {
                    let dl = d_logits[1 + ni];
                    let mut neg_clip = 1.0;
                    if dl.abs() * x_norm > 1.0 {
                        neg_clip = 1.0 / (dl.abs() * x_norm).max(1e-8);
                    }
                    let neg_row = &mut shadow_emb[neg_id * hidden_size..(neg_id + 1) * hidden_size];
                    unsafe {
                        forge_autograd::avx_math::axpy_avx2(neg_row, -lr * neg_clip * dl, &final_x);
                    }
                }
            }

            // EDGE-09: Dispatch Heartbeat to Vulkan to prevent RC6 deep sleep penalty
            if let Some(ref vk) = vk_qat {
                let _ = unsafe { vk.dispatch_heartbeat_sync() };
            }
            crate::mud::moe_train::end_step();
        }

        if crate::mud::moe_train::moe_train_enabled() {
            let util = crate::mud::moe_train::utilization_snapshot();
            if !util.is_empty() {
                println!(
                    "  [MoE-train] util={}",
                    util.iter()
                        .map(|(e, c)| format!("e{e}:{c}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }

        // 6. Apply gradients to deep layers
        let num_tokens = pair_count.max(1) as f32;
        let _weight_decay = 0.01;
        let _vk_updates: Vec<i32> = Vec::new();
        let mut vk_readbacks = Vec::new();

        // 1) First pass: collect all steps if we have Vulkan available
        let _use_vk = vk_qat.is_some();
        let lr = crate::mud::constants::qat_learning_rate();
        let decay = 0.01;
        // SCALES_ONLY freezes ELUT codes — ash GPU pack would re-threshold trits.
        // Keep ash for heartbeat + GEMV; force AVX2×PCorePool scales-only optimizer.
        let scales_only = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let use_ash_opt = vk_qat.is_some() && !scales_only;
        if use_ash_opt {
            let vk_qat = vk_qat.unwrap();
            let mut all_steps = Vec::new();
            let last_n_vk = crate::mud::sequence_pack::train_last_n_layers(shadow_layers.len());
            let first_train_vk = shadow_layers.len().saturating_sub(last_n_vk);
            for (l_idx, shadow_layer) in shadow_layers.iter().enumerate() {
                if l_idx < first_train_vk {
                    continue;
                }
                let grad = &gradients[l_idx];

                let matrices: Vec<(&str, &[f32], &[f32], *mut u8, *mut f32)> = vec![
                    (
                        "q",
                        &shadow_layer.q_w,
                        &grad.q_w_grad,
                        layers[l_idx].q_w as *mut u8,
                        layers[l_idx].q_scales as *mut f32,
                    ),
                    (
                        "k",
                        &shadow_layer.k_w,
                        &grad.k_w_grad,
                        layers[l_idx].k_w as *mut u8,
                        layers[l_idx].k_scales as *mut f32,
                    ),
                    (
                        "v",
                        &shadow_layer.v_w,
                        &grad.v_w_grad,
                        layers[l_idx].v_w as *mut u8,
                        layers[l_idx].v_scales as *mut f32,
                    ),
                    (
                        "o",
                        &shadow_layer.o_w,
                        &grad.o_w_grad,
                        layers[l_idx].o_w as *mut u8,
                        layers[l_idx].o_scales as *mut f32,
                    ),
                    (
                        "up",
                        &shadow_layer.ffn_up_w,
                        &grad.ffn_up_w_grad,
                        layers[l_idx].ffn_up_w as *mut u8,
                        layers[l_idx].ffn_up_scales as *mut f32,
                    ),
                    (
                        "gate",
                        &shadow_layer.ffn_gate_w,
                        &grad.ffn_gate_w_grad,
                        layers[l_idx].ffn_gate_w as *mut u8,
                        layers[l_idx].ffn_gate_scales as *mut f32,
                    ),
                    (
                        "down",
                        &shadow_layer.ffn_down_w,
                        &grad.ffn_down_w_grad,
                        layers[l_idx].ffn_down_w as *mut u8,
                        layers[l_idx].ffn_down_scales as *mut f32,
                    ),
                ];

                for (m_name, shadow_cpu, grad_cpu, packed_ptr, scales_ptr) in matrices {
                    if packed_ptr.is_null() || scales_ptr.is_null() {
                        continue;
                    }
                    let elements = shadow_cpu.len();
                    let cols = match m_name {
                        "q" | "k" | "v" | "o" | "up" | "gate" => hidden_size,
                        "down" => elements / hidden_size,
                        _ => hidden_size,
                    };
                    let name = format!("blk.{}.{}", l_idx, m_name);
                    let rows = elements / cols;

                    vk_readbacks.push(crate::mud::ash_qat_dispatcher::PendingReadback {
                        name: name.clone(),
                        packed_ptr,
                        scales_ptr,
                        packed_len: elements.div_ceil(8) * 4,
                        rows,
                    });

                    all_steps.push(crate::mud::ash_qat_dispatcher::AshTensorStep {
                        name,
                        shadow: shadow_cpu,
                        grad: grad_cpu,
                        elements,
                        cols,
                        rows,
                    });
                }
            }

            // L-05: submit async only — packed readback deferred to next chunk start
            // so Forward N+1 overlaps GPU Optimizer N (CPU packed ≠ VRAM packed).
            unsafe {
                vk_qat
                    .step_async_deferred(&all_steps, vk_readbacks, lr, decay, num_tokens)
                    .unwrap();
            }

            // Phase 1 (mHC trainable): alpha/beta are dense f32 (CPU-resident); the ash
            // path only handles ternary GEMV weights. Apply the CPU scale step here too.
            let last_n_vk = crate::mud::sequence_pack::train_last_n_layers(shadow_layers.len());
            let first_train_vk = shadow_layers.len().saturating_sub(last_n_vk);
            for l_idx in first_train_vk..shadow_layers.len() {
                let grad = &gradients[l_idx];
                if grad.mhc_alpha_grad.is_empty() {
                    continue;
                }
                let inv_tok = 1.0 / num_tokens.max(1.0);
                unsafe {
                    mhc_scale_sgd_step(
                        layers[l_idx].mhc_alpha_w as *mut f32,
                        &grad.mhc_alpha_grad,
                        lr,
                        inv_tok,
                    );
                    mhc_scale_sgd_step(
                        layers[l_idx].mhc_beta_w as *mut f32,
                        &grad.mhc_beta_grad,
                        lr,
                        inv_tok,
                    );
                }
            }
        } else {
            // AVX2 CPU + STE pack (or SCALES_ONLY) via P-Core Pool
            // When SCALES_ONLY + ash: heartbeat already dispatched; GEMV may still use ash.
            let pool = crate::mud::pcore_pool::get_pool();
            let last_n = crate::mud::sequence_pack::train_last_n_layers(shadow_layers.len());
            let first_train_layer = shadow_layers.len().saturating_sub(last_n);
            for (l_idx, shadow_layer) in shadow_layers.iter_mut().enumerate() {
                if l_idx < first_train_layer {
                    continue; // frozen lower layers: skip opt+pack
                }
                let grad = &gradients[l_idx];

                // Split mut access: strategy + optional Adam moments per matrix
                let opts = [
                    (
                        "q",
                        shadow_layer.q_opt,
                        &mut shadow_layer.q_w as *mut Vec<f32>,
                        &grad.q_w_grad as *const Vec<f32>,
                        layers[l_idx].q_w as *mut u8,
                        layers[l_idx].q_scales as *mut f32,
                        &mut shadow_layer.q_adam as *mut Option<crate::mud::adam_state::AdamState>,
                    ),
                    (
                        "k",
                        shadow_layer.k_opt,
                        &mut shadow_layer.k_w as *mut _,
                        &grad.k_w_grad as *const _,
                        layers[l_idx].k_w as *mut u8,
                        layers[l_idx].k_scales as *mut f32,
                        &mut shadow_layer.k_adam as *mut _,
                    ),
                    (
                        "v",
                        shadow_layer.v_opt,
                        &mut shadow_layer.v_w as *mut _,
                        &grad.v_w_grad as *const _,
                        layers[l_idx].v_w as *mut u8,
                        layers[l_idx].v_scales as *mut f32,
                        &mut shadow_layer.v_adam as *mut _,
                    ),
                    (
                        "o",
                        shadow_layer.o_opt,
                        &mut shadow_layer.o_w as *mut _,
                        &grad.o_w_grad as *const _,
                        layers[l_idx].o_w as *mut u8,
                        layers[l_idx].o_scales as *mut f32,
                        &mut shadow_layer.o_adam as *mut _,
                    ),
                    (
                        "up",
                        shadow_layer.ffn_up_opt,
                        &mut shadow_layer.ffn_up_w as *mut _,
                        &grad.ffn_up_w_grad as *const _,
                        layers[l_idx].ffn_up_w as *mut u8,
                        layers[l_idx].ffn_up_scales as *mut f32,
                        &mut shadow_layer.ffn_up_adam as *mut _,
                    ),
                    (
                        "gate",
                        shadow_layer.ffn_gate_opt,
                        &mut shadow_layer.ffn_gate_w as *mut _,
                        &grad.ffn_gate_w_grad as *const _,
                        layers[l_idx].ffn_gate_w as *mut u8,
                        layers[l_idx].ffn_gate_scales as *mut f32,
                        &mut shadow_layer.ffn_gate_adam as *mut _,
                    ),
                    (
                        "down",
                        shadow_layer.ffn_down_opt,
                        &mut shadow_layer.ffn_down_w as *mut _,
                        &grad.ffn_down_w_grad as *const _,
                        layers[l_idx].ffn_down_w as *mut u8,
                        layers[l_idx].ffn_down_scales as *mut f32,
                        &mut shadow_layer.ffn_down_adam as *mut _,
                    ),
                ];

                for (m_name, strategy, sw_p, gw_p, packed_ptr, scales_ptr, adam_p) in opts {
                    if packed_ptr.is_null() || scales_ptr.is_null() {
                        continue;
                    }
                    // SAFETY: exclusive access to this matrix within the layer loop
                    let shadow_w = unsafe { &mut *sw_p };
                    let grad_w = unsafe { &*gw_p };
                    let elements = shadow_w.len();
                    let cols = match m_name {
                        "q" | "k" | "v" | "o" | "up" | "gate" => hidden_size,
                        "down" => elements / hidden_size.max(1),
                        _ => hidden_size,
                    };
                    let adam = unsafe { (*adam_p).as_mut() };
                    unsafe {
                        apply_optimizer_cpu_step_and_pack(
                            shadow_w, grad_w, packed_ptr, scales_ptr, lr, decay, num_tokens, cols,
                            pool, strategy, adam,
                        );
                    }
                }

                // Phase 1 (mHC trainable): dense f32 SGD step on alpha/beta scales.
                // These are Float32 tensors (init 0.85/0.15), not ternary — no ELUT pack.
                // Grads are empty for base models / frozen layers, so this is a no-op then.
                if !grad.mhc_alpha_grad.is_empty() {
                    let inv_tok = 1.0 / num_tokens.max(1.0);
                    unsafe {
                        mhc_scale_sgd_step(
                            layers[l_idx].mhc_alpha_w as *mut f32,
                            &grad.mhc_alpha_grad,
                            lr,
                            inv_tok,
                        );
                        mhc_scale_sgd_step(
                            layers[l_idx].mhc_beta_w as *mut f32,
                            &grad.mhc_beta_grad,
                            lr,
                            inv_tok,
                        );
                    }
                }
            }
        }
        let avg_loss = if pair_count > 0 {
            total_loss / pair_count as f32
        } else {
            0.0
        };
        if stp_on {
            let l_stp = if stp_triples > 0 {
                stp_loss_accum / stp_triples as f32
            } else {
                0.0
            };
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "stp",
                    &format!("λ={stp_lambda:.3} triples={stp_triples} L_STP={l_stp:.5} (aux, train-only)")
                )
            );
        }
        Ok(act_accum.finish(avg_loss))
    }
}

/// Phase 1 (mHC trainable): in-place dense SGD step on a Float32 hyper-connection scale
/// vector (`alpha` or `beta`), with token-mean grad scaling and a hard clamp to keep the
/// residual geometry bounded (HC paper uses tanh; a clamp is simpler + STE-friendly).
///
/// # Safety
/// `w` must point to `grad.len()` valid, writable `f32` (the model's owned mHC tensor data).
pub unsafe fn mhc_scale_sgd_step(w: *mut f32, grad: &[f32], lr: f32, inv_tok: f32) {
    if w.is_null() {
        return;
    }
    for (i, &g) in grad.iter().enumerate() {
        let g = if g.is_finite() { g * inv_tok } else { 0.0 };
        let p = unsafe { w.add(i) };
        let updated = (*p - lr * g).clamp(0.0, 4.0);
        *p = updated;
    }
}

/// Apply optimizer according to [`OptimizerStrategy`], then STE-pack to ELUT 4-bit.
///
/// **L-01 + P0:** Muon/GaLore/Chunked preprocess → SGD; **Adam / SparseAdam** use real
/// moments via [`crate::mud::adam_state`] (`adam_step_avx2` when available).
/// **L-09:** EZOP TLS grad scratch.
///
/// # Safety
///
/// `packed_ptr` must point to a writable buffer of at least
/// `shadow_w.len()` ELUT 4-bit nibbles (`(shadow_w.len() + 1) / 2` bytes);
/// the caller guarantees it is valid for the duration of the call and the
/// `grad_w`/`shadow_w` slices are non-overlapping and properly aligned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_optimizer_cpu_step_and_pack(
    shadow_w: &mut [f32],
    grad_w: &[f32],
    packed_ptr: *mut u8,
    scales_ptr: *mut f32,
    lr: f32,
    weight_decay: f32,
    num_tokens: f32,
    cols: usize,
    pool: &crate::mud::pcore_pool::PCorePool,
    strategy: crate::mud::slime_backward::OptimizerStrategy,
    adam: Option<&mut crate::mud::adam_state::AdamState>,
) {
    use crate::mud::slime_backward::OptimizerStrategy;

    let cols = cols.max(1);
    let n = shadow_w.len();
    let rows = n / cols;
    if rows == 0 || n != rows * cols || grad_w.len() < n {
        return;
    }

    // L-09: reuse TLS scratch instead of grad_w.to_vec() every call (P-01).
    crate::mud::ezop::with_grad_scratch(n, |grad| {
        // SAFETY: grad len n; grad_w at least n.
        unsafe {
            crate::mud::ezop::copy_f32(grad.as_mut_ptr(), grad_w.as_ptr(), n);
            crate::mud::ezop::sanitize_f32(grad.as_mut_ptr(), n);
        }
        crate::mud::adam_state::scale_grad_by_tokens(grad, num_tokens);

        match strategy {
            OptimizerStrategy::Sgd => {
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::Muon { ns_iters } => {
                crate::mud::muon::newton_schulz_orthogonalize(grad, rows, cols, ns_iters);
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::GaLore {
                rank,
                update_freq: _,
            } => {
                crate::mud::galore::galore_step(grad, rows, cols, rank.max(1));
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::ChunkedAdam { chunk_cols } => {
                let chunk = chunk_cols.max(1);
                let ns = crate::mud::muon::muon_ns_iters();
                if cols > chunk {
                    crate::mud::muon::chunked_muon_step(grad, rows, cols, chunk, ns);
                } else {
                    crate::mud::muon::newton_schulz_orthogonalize(grad, rows, cols, ns);
                }
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::Adam => {
                if let Some(st) = adam {
                    crate::mud::adam_state::adam_step(shadow_w, grad, st, lr, weight_decay, 1.0);
                } else {
                    // No state allocated — safe SGD fallback
                    apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
                }
            }
            OptimizerStrategy::SparseAdam { only_active_rows } => {
                if let Some(st) = adam {
                    crate::mud::adam_state::sparse_adam_step(
                        shadow_w,
                        grad,
                        st,
                        rows,
                        cols,
                        lr,
                        weight_decay,
                        1.0,
                        only_active_rows,
                    );
                } else {
                    apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
                }
            }
        }

        // ── Shadow magnitude guard (P1 fix, 2026-07-20) ───────────────────────
        // Prevent PRQ-scale inflation: an unbounded shadow (high LR / weak decay)
        // makes per-row absmean → large `s`, so W = s·T explodes (observed 27× on
        // last layers → logit domination → vocabulary collapse). Clamp each element
        // to ±K·row_absmax before pack. K default 8 (env MUD_TRAIN_WCLAMP_K, 0=off).
        {
            let k = std::env::var("MUD_TRAIN_WCLAMP_K")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(8.0);
            if k > 0.0 {
                // Raw-pointer pass (P-00): no bounds checks, no iterator overhead.
                // `bound` depends on the full row absmean, so sum then clamp in two
                // pointer strides over the same row.
                let p = shadow_w.as_mut_ptr();
                let floor = crate::mud::constants::EPSILON_FLOOR;
                for r in 0..rows {
                    let base = (r * cols) as isize;
                    let mut sum = 0.0f32;
                    for c in 0..cols {
                        sum += unsafe { (*p.offset(base + c as isize)).abs() };
                    }
                    let bound = (sum / cols as f32 * k).max(floor);
                    for c in 0..cols {
                        let off = base + c as isize;
                        let v = unsafe { *p.offset(off) };
                        unsafe { *p.offset(off) = v.clamp(-bound, bound) };
                    }
                }
            }
        }

        // ── Scales-only path (Bonsai / native ternary seating) ─────────────────
        // Freeze ELUT trit codes; update PRQ scales so W≈s·T matches post-SGD shadow.
        // Avoids re-thresholding {-1,0,+1} that destroys native low-bit structure.
        let scales_only = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if scales_only {
            unsafe {
                update_prq_scales_only(shadow_w, packed_ptr, scales_ptr, rows, cols);
            }
            return;
        }

        // STE pack: serial `pack_elut_prq` is faster than PCorePool for post-step pack
        // (avoids 30L×7 task-queue roundtrips; pool stays free for next GEMV).
        // Optional parallel pack: MUD_TRAIN_PACK_POOL=1
        let use_pack_pool = std::env::var("MUD_TRAIN_PACK_POOL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if use_pack_pool && rows >= 64 && pool.num_threads() > 1 {
            let n_tasks = pool.num_threads().max(1);
            let rows_per_task = (rows / n_tasks).max(1);
            let shadow_ptr = shadow_w.as_ptr() as usize;
            let packed_p = packed_ptr as usize;
            let scales_p = scales_ptr as usize;
            for i in 0..n_tasks {
                let start_row = i * rows_per_task;
                let end_row = if i + 1 == n_tasks {
                    rows
                } else {
                    start_row + rows_per_task
                };
                if start_row >= end_row {
                    break;
                }
                let nrows = end_row - start_row;
                pool.execute(move || {
                    let sw = (shadow_ptr as *const f32).wrapping_add(start_row * cols);
                    let sc = (scales_p as *mut f32).wrapping_add(start_row);
                    let pk = packed_p as *mut u8;
                    unsafe {
                        // Pack row range into global buffer (absolute nibble indices)
                        for r in 0..nrows {
                            let abs_row = start_row + r;
                            let row_start = abs_row * cols;
                            let mut abs_sum = 0.0f32;
                            for c in 0..cols {
                                abs_sum += (*sw.add(r * cols + c)).abs();
                            }
                            let s = ((abs_sum / cols as f32) * std::f32::consts::FRAC_1_SQRT_2)
                                .max(crate::mud::constants::EPSILON_FLOOR);
                            *sc.add(r) = s;
                            let threshold = s * 0.7;
                            for c in 0..cols {
                                let idx = row_start + c;
                                let v = *sw.add(r * cols + c);
                                let bit = if v > threshold {
                                    0x1u8
                                } else if v < -threshold {
                                    0xFu8
                                } else {
                                    0x0u8
                                };
                                let byte_idx = idx / 2;
                                let nibble_pos = (idx % 2) * 4;
                                let current = *pk.add(byte_idx);
                                let mask = !(0xFu8 << nibble_pos);
                                *pk.add(byte_idx) = (current & mask) | (bit << nibble_pos);
                            }
                        }
                    }
                });
            }
            pool.wait_all();
        } else {
            // SAFETY: shadow/scales/packed valid for full matrix after optimizer step.
            unsafe {
                crate::mud::ezop::pack_elut_prq(
                    shadow_w.as_ptr(),
                    rows,
                    cols,
                    scales_ptr,
                    packed_ptr,
                );
            }
        }
    });
}

/// Update PRQ scales only: keep existing ELUT codes, fit s per row to shadow.
///
/// For each row: `s = Σ(W·T) / Σ(T²)` with T ∈ {-1,0,+1} from packed ELUT.
/// Then project shadow `W ← s·T` so next STE step stays on the ternary manifold.
///
/// # Safety
/// `packed_ptr` / `scales_ptr` must cover `rows` of ELUT (cols/8 u32s per row) and `rows` f32.
unsafe fn update_prq_scales_only(
    shadow_w: &mut [f32],
    packed_ptr: *mut u8,
    scales_ptr: *mut f32,
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 || shadow_w.len() < rows * cols {
        return;
    }
    let u32s_per_row = cols.div_ceil(8);
    let packed_u32 = packed_ptr as *const u32;
    for r in 0..rows {
        let row_off = r * cols;
        let mut dot = 0.0f32;
        let mut t2 = 0.0f32;
        for c in 0..cols {
            let word = *packed_u32.add(r * u32s_per_row + c / 8);
            let nibble = (word >> ((c % 8) * 4)) & 0xF;
            let t = if nibble == 0x1 {
                1.0f32
            } else if nibble == 0xF {
                -1.0f32
            } else {
                0.0f32
            };
            let w = shadow_w[row_off + c];
            dot += w * t;
            t2 += t * t;
        }
        let s = if t2 > crate::mud::constants::EPSILON_FLOOR {
            (dot / t2).abs().max(crate::mud::constants::EPSILON_FLOOR)
        } else {
            // Fallback absmean of shadow if row is all zeros in codes
            let mut abs_sum = 0.0f32;
            for c in 0..cols {
                abs_sum += shadow_w[row_off + c].abs();
            }
            (abs_sum / cols as f32).max(crate::mud::constants::EPSILON_FLOOR)
        };
        // Preserve sign of least-squares scale if anti-correlated (rare):
        // W≈s·T with T∈{-1,0,+1}, so a negative `dot` means the shadow is
        // anti-correlated with the frozen codes and the signed scale must stay
        // negative (the forward applies `out = w * scale`, scale carries sign).
        let s_signed = if t2 > crate::mud::constants::EPSILON_FLOOR && dot < 0.0 {
            -s
        } else {
            s
        };
        // Clamp magnitude into a sane PRQ range; keep sign.
        let mag = s_signed
            .abs()
            .clamp(crate::mud::constants::EPSILON_FLOOR, 1.0f32);
        *scales_ptr.add(r) = if s_signed < 0.0 { -mag } else { mag };
        let s_use = *scales_ptr.add(r);
        // Project shadow onto frozen codes so magnitude tracks scales for next step
        for c in 0..cols {
            let word = *packed_u32.add(r * u32s_per_row + c / 8);
            let nibble = (word >> ((c % 8) * 4)) & 0xF;
            let t = if nibble == 0x1 {
                1.0f32
            } else if nibble == 0xF {
                -1.0f32
            } else {
                0.0f32
            };
            shadow_w[row_off + c] = t * s_use;
        }
    }
}

/// SGD-style shadow update (shared by all strategies after optional grad transform).
/// L-09: AVX2 when available, else EZOP scalar raw pointers.
fn apply_sgd_shadow_update(
    shadow_w: &mut [f32],
    grad_w: &[f32],
    lr: f32,
    weight_decay: f32,
    num_tokens: f32,
) {
    let n = shadow_w.len().min(grad_w.len());
    if n == 0 {
        return;
    }
    if core_arch_x86_64_has_avx2() {
        unsafe {
            forge_autograd::avx_math::sgd_step_avx2(
                &mut shadow_w[..n],
                &grad_w[..n],
                lr,
                weight_decay,
                num_tokens,
            );
        }
    } else {
        // SAFETY: n elements of each slice.
        unsafe {
            crate::mud::ezop::sgd_step(
                shadow_w.as_mut_ptr(),
                grad_w.as_ptr(),
                n,
                lr,
                weight_decay,
                num_tokens,
            );
        }
    }
}

#[inline(always)]
fn core_arch_x86_64_has_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(test)]
mod optimizer_strategy_tests {
    use super::*;
    use crate::mud::slime_backward::{select_optimizer, OptimizerStrategy};
    use std::sync::Mutex;

    // Serialize tests that mutate global optimizer-selection env vars
    // (MUD_TRAIN_MAX_CHUNKS / MUD_OPT / ...). They race under the parallel
    // test runner and produce flaky results otherwise.
    static OPT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn run_step(
        rows: usize,
        cols: usize,
        strategy: OptimizerStrategy,
        shadow0: f32,
        g0: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
        let mut shadow = vec![shadow0; rows * cols];
        let grad = vec![g0; rows * cols];
        // seed a non-uniform gradient so Muon/GaLore actually transform
        let mut grad = grad;
        for (i, g) in grad.iter_mut().enumerate() {
            *g = g0 * (1.0 + (i % 7) as f32 * 0.1);
        }
        let mut packed = vec![0u8; (rows * cols).div_ceil(2)];
        let mut scales = vec![0.0f32; rows];
        let pool = crate::mud::pcore_pool::PCorePool::new(2);
        let mut adam = crate::mud::adam_state::AdamState::for_strategy(rows * cols, strategy);
        unsafe {
            apply_optimizer_cpu_step_and_pack(
                &mut shadow,
                &grad,
                packed.as_mut_ptr(),
                scales.as_mut_ptr(),
                1e-2,
                0.0,
                1.0,
                cols,
                &pool,
                strategy,
                adam.as_mut(),
            );
        }
        (shadow, scales, packed)
    }

    fn clear_opt_env() {
        std::env::remove_var("MUD_TRAIN_MAX_CHUNKS");
        std::env::remove_var("MUD_OPT");
        std::env::remove_var("MUD_TRAIN_OPT");
        std::env::remove_var("MUD_MUON_NS_ITERS");
    }

    #[test]
    fn test_select_optimizer_square() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        // Run sequentially to avoid a global env-var race with the quick-mode
        // assertion below (both touch MUD_TRAIN_MAX_CHUNKS via clear_opt_env).
        // Contract: square -> Muon by default; square with MUD_TRAIN_MAX_CHUNKS
        // (quick mode) -> Sgd.
        clear_opt_env();
        assert!(
            matches!(select_optimizer(576, 576), OptimizerStrategy::Muon { .. }),
            "expected Muon for square"
        );
        assert!(
            matches!(select_optimizer(2560, 2560), OptimizerStrategy::Muon { .. }),
            "expected Muon for large square"
        );
        std::env::set_var("MUD_TRAIN_MAX_CHUNKS", "8");
        assert!(
            matches!(select_optimizer(576, 576), OptimizerStrategy::Sgd),
            "expected Sgd in quick mode (MUD_TRAIN_MAX_CHUNKS set)"
        );
        clear_opt_env();
    }

    #[test]
    fn test_select_optimizer_tall_is_galore() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        clear_opt_env();
        match select_optimizer(1536, 576) {
            OptimizerStrategy::GaLore { rank, .. } => assert!(rank >= 8),
            other => panic!("expected GaLore for tall, got {other:?}"),
        }
    }

    #[test]
    fn test_select_optimizer_wide_is_chunked() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        clear_opt_env();
        match select_optimizer(576, 1536) {
            OptimizerStrategy::ChunkedAdam { chunk_cols } => assert!(chunk_cols >= 1),
            other => panic!("expected ChunkedAdam for wide, got {other:?}"),
        }
    }

    #[test]
    fn test_apply_optimizer_muon_runs_and_packs() {
        let (shadow, scales, packed) =
            run_step(8, 8, OptimizerStrategy::Muon { ns_iters: 2 }, 0.5, 0.01);
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
        assert!(shadow.iter().all(|w| w.is_finite()));
        assert!(packed.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_apply_optimizer_galore_tall_finite() {
        let (shadow, scales, packed) = run_step(
            32,
            8,
            OptimizerStrategy::GaLore {
                rank: 4,
                update_freq: 100,
            },
            0.3,
            0.02,
        );
        assert!(shadow.iter().all(|w| w.is_finite()));
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
        assert!(packed.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_apply_optimizer_chunked_wide_finite() {
        let (shadow, scales, _) = run_step(
            8,
            32,
            OptimizerStrategy::ChunkedAdam { chunk_cols: 8 },
            0.4,
            0.015,
        );
        assert!(shadow.iter().all(|w| w.is_finite()));
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn test_muon_differs_from_plain_adam_sgd() {
        // Same initial state; Muon preprocess should not match plain Adam path exactly
        let rows = 8usize;
        let cols = 8usize;
        let mut shadow_m = vec![0.5f32; rows * cols];
        let mut shadow_a = vec![0.5f32; rows * cols];
        let mut grad = vec![0.0f32; rows * cols];
        for (i, g) in grad.iter_mut().enumerate() {
            *g = 0.05 * ((i % 5) as f32 - 2.0);
        }
        let mut packed = vec![0u8; (rows * cols).div_ceil(2)];
        let mut scales = vec![0.0f32; rows];
        let pool = crate::mud::pcore_pool::PCorePool::new(2);

        unsafe {
            apply_optimizer_cpu_step_and_pack(
                &mut shadow_m,
                &grad,
                packed.as_mut_ptr(),
                scales.as_mut_ptr(),
                1e-2,
                0.0,
                1.0,
                cols,
                &pool,
                OptimizerStrategy::Muon { ns_iters: 3 },
                None,
            );
        }
        let mut adam = crate::mud::adam_state::AdamState::zeros(rows * cols);
        unsafe {
            apply_optimizer_cpu_step_and_pack(
                &mut shadow_a,
                &grad,
                packed.as_mut_ptr(),
                scales.as_mut_ptr(),
                1e-2,
                0.0,
                1.0,
                cols,
                &pool,
                OptimizerStrategy::Adam,
                Some(&mut adam),
            );
        }

        let max_delta = shadow_m
            .iter()
            .zip(shadow_a.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-6,
            "Muon path should diverge from plain Adam/SGD path, max_delta={max_delta}"
        );
    }

    #[test]
    fn test_slime_layer_shadow_opts_match_select() {
        // Isolate from quick-train env overrides (MUD_TRAIN_MAX_CHUNKS → SGD).
        std::env::remove_var("MUD_TRAIN_MAX_CHUNKS");
        std::env::remove_var("MUD_OPT");
        std::env::remove_var("MUD_TRAIN_OPT");
        std::env::remove_var("MUD_MUON_NS_ITERS");
        // hidden=64, ffn_mid=192 (ratio 3 > 2.5 → GaLore up; down ratio < 0.4 → Chunked)
        let s = crate::mud::slime_backward::SlimeLayerShadowF32::new(64, 192, 2, 32);
        assert!(matches!(s.q_opt, OptimizerStrategy::Muon { .. }));
        assert!(matches!(s.ffn_up_opt, OptimizerStrategy::GaLore { .. }));
        assert!(matches!(
            s.ffn_down_opt,
            OptimizerStrategy::ChunkedAdam { .. }
        ));
        // GQA k: kv_dim=64, cols=64 → square-ish Muon
        assert!(matches!(s.k_opt, OptimizerStrategy::Muon { .. }));
    }
}
