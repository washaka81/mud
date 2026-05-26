use std::thread;
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::mud::store::MudStore;
use crate::mud::{MudFile, MudTensor, MudTensorType, dequantize_ternary_row};

pub static SHOULD_TERMINATE: AtomicBool = AtomicBool::new(false);
pub static IS_TRAINING: AtomicBool = AtomicBool::new(false);
pub static TRAINING_CURRENT: AtomicUsize = AtomicUsize::new(0);
pub static TRAINING_TOTAL: AtomicUsize = AtomicUsize::new(0);
pub static LAST_ACTIVITY: AtomicUsize = AtomicUsize::new(0);

use std::collections::HashMap;
use forge_autograd::Tape;

const LR: f32 = 0.002;
const WEIGHT_DECAY: f32 = 0.01;
const MAX_GRAD_NORM: f32 = 1.0;
const NUM_NEGATIVES: usize = 5;
const LAYERS_TO_TRAIN: usize = 3;
const VOCAB_FULL_CE_THRESHOLD: usize = 50000;

struct TrainingGuard;
impl Drop for TrainingGuard {
    fn drop(&mut self) {
        IS_TRAINING.store(false, Ordering::SeqCst);
        TRAINING_CURRENT.store(0, Ordering::SeqCst);
        TRAINING_TOTAL.store(0, Ordering::SeqCst);
    }
}

struct ExpertShadow {
    shadow_w1: Vec<f32>,
    shadow_w2: Vec<f32>,
    shadow_w3: Vec<f32>,
    w1_shape: Vec<usize>,
    w2_shape: Vec<usize>,
    w3_shape: Vec<usize>,
    #[allow(dead_code)]
    w1_scale: f32,
    #[allow(dead_code)]
    w2_scale: f32,
    #[allow(dead_code)]
    w3_scale: f32,
    modified: bool,
}

struct GateShadow {
    gate_weights: Vec<f32>,
    #[allow(dead_code)]
    shape: Vec<usize>,
    #[allow(dead_code)]
    modified: bool,
}

fn current_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn dequantize_tensor_f32(tensor: &MudTensor, scale_val: f32) -> Vec<f32> {
    let elements: usize = tensor.shape.iter().product();
    let mut out = vec![0.0f32; elements];
    match tensor.t_type {
        MudTensorType::Float32 => {
            unsafe {
                std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, out.as_mut_ptr(), elements);
            }
        }
        MudTensorType::Ternary2Bit => {
            unsafe {
                dequantize_ternary_row(tensor.data_ptr as *const u32, &mut out, elements);
            }
            if scale_val != 1.0 {
                for v in out.iter_mut() { *v *= scale_val; }
            }
        }
        _ => {}
    }
    out
}

fn pack_ternary_from_f32(data: &[f32]) -> Vec<u8> {
    let u32_count = data.len().div_ceil(16);
    let mut packed = vec![0u32; u32_count];
    for i in 0..data.len() {
        let bit = if data[i] > 0.5 { 1u32 } else if data[i] < -0.5 { 2u32 } else { 0u32 };
        packed[i / 16] |= bit << ((i % 16) * 2);
    }
    unsafe {
        std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
    }.to_vec()
}

fn compute_rowwise_scale(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut scales = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        let absmean = data[start..start + cols].iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
        scales.push(absmean.max(1e-10));
    }
    scales
}

fn compute_optimal_scale_for_tensor(data: &[f32]) -> f32 {
    let max_abs = data.iter().map(|v| v.abs()).reduce(f32::max).unwrap_or(0.0);
    if max_abs < 1e-6 { return 1.0; }
    let mut best_scale = 1.0;
    let mut best_mse = f32::MAX;
    for i in 1..=100 {
        let tau = (i as f32) / 100.0 * max_abs;
        let s = 1.0 / tau;
        let mut mse = 0.0;
        for &w in data {
            let q = (w * s).round().clamp(-1.0, 1.0);
            let diff = w - q * tau;
            mse += diff * diff;
        }
        if mse < best_mse { best_mse = mse; best_scale = s; }
    }
    best_scale
}

fn save_shadows_to_mud(
    mud_path: &str,
    loaded_experts: &HashMap<(usize, usize), ExpertShadow>,
    loaded_gates: &HashMap<usize, GateShadow>,
    emb_weights: Option<&[f32]>,
    emb_vocab: usize,
    emb_hidden: usize,
) -> anyhow::Result<()> {
    let mf = MudFile::load(mud_path)?;
    let mut new_tensors = HashMap::new();
    let core = mf.skills.get("core").unwrap();

    for (name, t) in &core.tensors {
        let mut replaced = false;

        if let Some(emb) = emb_weights {
            if name == "token_embd.weight" {
                let total = emb_vocab * emb_hidden;
                let mut ternary_data = vec![0.0f32; total];
                let scales = compute_rowwise_scale(emb, emb_vocab, emb_hidden);
                for r in 0..emb_vocab {
                    let s = scales[r];
                    let start = r * emb_hidden;
                    for j in 0..emb_hidden {
                        ternary_data[start + j] = (emb[start + j] / s).round().clamp(-1.0, 1.0);
                    }
                }
                let packed = pack_ternary_from_f32(&ternary_data);
                // Store scales as f32 tensor
                let scales_f32_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();

                new_tensors.insert("token_embd.weight".to_string(), MudTensor {
                    name: "token_embd.weight".to_string(),
                    t_type: MudTensorType::Ternary2Bit,
                    shape: vec![emb_vocab, emb_hidden],
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    mmap: None,
                    owned_data: Some(packed),
                });
                new_tensors.insert("embed_scales".to_string(), MudTensor {
                    name: "embed_scales".to_string(),
                    t_type: MudTensorType::Float32,
                    shape: vec![emb_vocab],
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    mmap: None,
                    owned_data: Some(scales_f32_bytes),
                });
                replaced = true;
            }
        }

            if !replaced {
                // Check if this is a modified expert weight
                let mut found_expert = false;
                for ((layer, expert), shadow) in &*loaded_experts {
                    if !shadow.modified { continue; }
                    let target_w1 = format!("blk.{}.expert.{}.w1.weight", layer, expert);
                    let target_w2 = format!("blk.{}.expert.{}.w2.weight", layer, expert);
                    let target_w3 = format!("blk.{}.expert.{}.w3.weight", layer, expert);
                    let scale_w1 = format!("blk.{}.expert.{}.w1.scale", layer, expert);
                    let scale_w2 = format!("blk.{}.expert.{}.w2.scale", layer, expert);
                    let scale_w3 = format!("blk.{}.expert.{}.w3.scale", layer, expert);

                    if *name == target_w1 || *name == target_w2 || *name == target_w3 {
                        let (data, target_scale_name) = if *name == target_w1 {
                            (shadow.shadow_w1.as_slice(), scale_w1)
                        } else if *name == target_w2 {
                            (shadow.shadow_w2.as_slice(), scale_w2)
                        } else {
                            (shadow.shadow_w3.as_slice(), scale_w3)
                        };

                        let new_scale = compute_optimal_scale_for_tensor(data);
                        let packed = pack_ternary_from_f32(data);
                        new_tensors.insert(name.clone(), MudTensor {
                            name: name.clone(),
                            t_type: MudTensorType::Ternary2Bit,
                            shape: t.shape.clone(),
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            mmap: None,
                            owned_data: Some(packed),
                        });
                        let scale_bytes = new_scale.to_le_bytes().to_vec();
                        new_tensors.insert(target_scale_name.clone(), MudTensor {
                            name: target_scale_name.clone(),
                            t_type: MudTensorType::Float32,
                            shape: vec![1],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            mmap: None,
                            owned_data: Some(scale_bytes),
                        });
                        found_expert = true;
                        break;
                    }
                }
                if found_expert { continue; }

                // Check if this is a modified gate weight
                let mut found_gate = false;
                for (layer, gate) in &*loaded_gates {
                    if !gate.modified { continue; }
                    let gname = format!("blk.{}.gate.weight", layer);
                    if *name == gname {
                        let packed = pack_ternary_from_f32(&gate.gate_weights);
                        new_tensors.insert(name.clone(), MudTensor {
                            name: name.clone(),
                            t_type: MudTensorType::Ternary2Bit,
                            shape: t.shape.clone(),
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            mmap: None,
                            owned_data: Some(packed),
                        });
                        found_gate = true;
                        break;
                    }
                }
                if found_gate { continue; }

                new_tensors.insert(name.clone(), t.clone());
            }
    }

    // Update global metadata for embed scales
    let mut global_meta = mf.global_metadata.clone();
    if emb_weights.is_some() {
        global_meta.insert("embed_ternarized".to_string(), "row_absmean".to_string());
    }

    let new_mf = MudFile {
        mmap: None,
        skills: HashMap::from([("core".to_string(), crate::mud::MudSkill {
            name: "core".to_string(),
            tensors: new_tensors,
            metadata: HashMap::new(),
        })]),
        global_metadata: global_meta,
    };

    let tmp_path = format!("{}.tmp", mud_path);
    new_mf.save(&tmp_path)?;
    std::fs::rename(&tmp_path, mud_path)?;
    println!("  ✅ Saved trained weights to {}", mud_path);
    Ok(())
}

fn uniform_seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u32
}

/// Background daemon for MUD autonomous learning.
pub struct MudAutoTrainer {
    db_path: String,
    threshold: usize,
    model_path: String,
}

impl MudAutoTrainer {
    pub fn new(db_path: String, threshold: usize, model_path: String) -> Self {
        Self { db_path, threshold, model_path }
    }

    pub fn start_background_monitor(&self) {
        let threshold = self.threshold;
        let model_path = self.model_path.clone();
        let db_path = self.db_path.clone();

        thread::spawn(move || {
            let store = match MudStore::open(&db_path) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    eprintln!("  [Auto-Trainer] Failed to open dedicated connection: {}", e);
                    return;
                }
            };
            loop {
                match store.get_unassimilated() {
                    Ok(unassimilated) => {
                        if unassimilated.len() >= threshold {
                            self::run_local_training_cycle(&store, &unassimilated, &model_path);
                        }
                    }
                    Err(e) => eprintln!("  [Auto-Trainer] Monitor error: {}", e),
                }
                thread::sleep(Duration::from_secs(5));
            }
        });
    }

    pub fn run_training_cycle(&self) -> anyhow::Result<usize> {
        let store = Arc::new(MudStore::open(&self.db_path)?);
        let unassimilated = store.get_unassimilated()?;
        if unassimilated.is_empty() {
            return Ok(0);
        }
        let count = unassimilated.len();
        self::run_local_training_cycle(&store, &unassimilated, &self.model_path);
        Ok(count)
    }
}

fn run_local_training_cycle(store: &Arc<MudStore>, facts: &[(i32, String)], model_path: &str) {
    let batch_size = facts.len().min(10);
    IS_TRAINING.store(true, Ordering::SeqCst);
    TRAINING_TOTAL.store(batch_size, Ordering::SeqCst);
    TRAINING_CURRENT.store(0, Ordering::SeqCst);

    let mut ids_to_mark = Vec::new();
    let _guard = TrainingGuard;

    let mud_file = match MudFile::load(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  [Auto-Trainer-Native] Failed to load MUD: {}", e);
            IS_TRAINING.store(false, Ordering::SeqCst);
            return;
        }
    };

    let core = match mud_file.skills.get("core") {
        Some(s) => s,
        None => { eprintln!("  [Auto-Trainer] Missing core skill"); IS_TRAINING.store(false, Ordering::SeqCst); return; }
    };

    let num_layers = match mud_file.global_metadata.get("num_layers").and_then(|s| s.parse::<usize>().ok()) {
        Some(n) => n,
        None => { eprintln!("  [Auto-Trainer] Missing num_layers"); IS_TRAINING.store(false, Ordering::SeqCst); return; }
    };
    let num_experts = match mud_file.global_metadata.get("num_experts").and_then(|s| s.parse::<usize>().ok()) {
        Some(n) => n,
        None => { eprintln!("  [Auto-Trainer] Missing num_experts"); IS_TRAINING.store(false, Ordering::SeqCst); return; }
    };
    let top_k: usize = mud_file.global_metadata.get("top_k").and_then(|s| s.parse().ok()).unwrap_or(1);

    let tokens_str = match mud_file.global_metadata.get("tokenizer.tokens") {
        Some(s) => s,
        None => { eprintln!("  [Auto-Trainer] Missing tokenizer.tokens"); IS_TRAINING.store(false, Ordering::SeqCst); return; }
    };
    let merges_str = mud_file.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    let tokenizer = crate::model::tokenizer::Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let emb_tensor = match core.tensors.get("token_embd.weight") {
        Some(t) => t,
        None => { eprintln!("  [Auto-Trainer] Missing token_embd.weight"); IS_TRAINING.store(false, Ordering::SeqCst); return; }
    };

    let vocab_size = emb_tensor.shape[0];
    let hidden_size = emb_tensor.shape[1];
    let emb_elements = vocab_size * hidden_size;

    // Load embedding: type-aware (Float32 or Ternary2Bit + embed_scales)
    let mut emb_weights = vec![0.0f32; emb_elements];
    let embed_scales_tensor = core.tensors.get("embed_scales");
    match emb_tensor.t_type {
        MudTensorType::Float32 => {
            unsafe {
                std::ptr::copy_nonoverlapping(emb_tensor.data_ptr as *const f32, emb_weights.as_mut_ptr(), emb_elements);
            }
        }
        MudTensorType::Ternary2Bit => {
            for row in 0..vocab_size {
                let start = row * hidden_size;
                let row_u32_offset = start / 16;
                unsafe {
                    let ptr = emb_tensor.data_ptr as *const u32;
                    dequantize_ternary_row(ptr.add(row_u32_offset), &mut emb_weights[start..start + hidden_size], hidden_size);
                }
            }
            // Apply per-row scales if available
            if let Some(scales_t) = embed_scales_tensor {
                if scales_t.t_type == MudTensorType::Float32 {
                    unsafe {
                        let scales_ptr = scales_t.data_ptr as *const f32;
                        for row in 0..vocab_size {
                            let scale = *scales_ptr.add(row);
                            if scale != 1.0 {
                                let start = row * hidden_size;
                                for j in 0..hidden_size {
                                    emb_weights[start + j] *= scale;
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            eprintln!("  [Auto-Trainer] Unsupported embedding type");
            IS_TRAINING.store(false, Ordering::SeqCst);
            return;
        }
    }
    // Always keep a f32 shadow for training updates

    let use_full_ce = vocab_size <= VOCAB_FULL_CE_THRESHOLD;
    if use_full_ce {
        println!("  [Auto-Trainer] Using full-vocabulary cross-entropy (vocab={})", vocab_size);
    } else {
        println!("  [Auto-Trainer] Using contrastive loss with {} negatives (vocab={})", NUM_NEGATIVES, vocab_size);
    }

    let lr = LR;
    let mut rng_seed = uniform_seed();

    let mut loaded_experts: HashMap<(usize, usize), ExpertShadow> = HashMap::new();
    let mut loaded_gates: HashMap<usize, GateShadow> = HashMap::new();
    let mut nan_skips = 0usize;
    let mut total_tokens = 0usize;

    for (batch_i, (id, content)) in facts.iter().take(10).enumerate() {
        TRAINING_CURRENT.store(batch_i + 1, Ordering::SeqCst);
        if SHOULD_TERMINATE.load(Ordering::SeqCst) {
            println!("  [Auto-Trainer] Interrupted! Skipping remaining facts...");
            break;
        }

        let tokens = tokenizer.encode(content);
        if tokens.len() < 2 {
            ids_to_mark.push(*id);
            continue;
        }

        for i in 0..tokens.len() - 1 {
            let t_in = tokens[i] as usize;
            let t_target = tokens[i + 1] as usize;

            if t_in >= vocab_size || t_target >= vocab_size { continue; }

            total_tokens += 1;

            // Content-aware layer selection: hash the input embedding for distribution
            rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let mut layer_hash = rng_seed;
            for j in (0..hidden_size).step_by(4) {
                let v = emb_weights[t_in * hidden_size + j].to_bits();
                layer_hash = layer_hash.wrapping_mul(1664525).wrapping_add(v);
            }
            let base_layer = (layer_hash as usize) % num_layers;
            let layers_to_process = LAYERS_TO_TRAIN.min(num_layers).max(1);
            let end_layer = (base_layer + layers_to_process).min(num_layers);

            // Load input embedding
            let x_current = emb_weights[t_in * hidden_size .. (t_in + 1) * hidden_size].to_vec();

            // Build autograd tape for the FFN stack
            let mut tape = Tape::new();
            let x_node = tape.push_leaf(x_current, vec![1, hidden_size]);
            let mut current_node = x_node;

            // Track expert selections for load-balancing loss
            let mut expert_selections: Vec<(usize, usize, f32)> = Vec::new(); // (layer, expert, gate_logit)

            for layer in base_layer..end_layer {
                // --- Gate / routing ---
                let num_active = if num_experts > 1 { num_experts } else { 1 };

                if num_active > 1 {
                    let gate_name = format!("blk.{}.gate.weight", layer);
                    let gate_shadow = if let Some(g) = loaded_gates.get(&layer) {
                        &g.gate_weights
                    } else {
                        let gate_tensor = match core.tensors.get(&gate_name) {
                            Some(t) => t,
                            None => { break; } // no gate = single expert
                        };
                        let gate_data = dequantize_tensor_f32(gate_tensor, 1.0);
                        loaded_gates.insert(layer, GateShadow {
                            gate_weights: gate_data,
                            shape: gate_tensor.shape.clone(),
                            modified: false,
                        });
                        &loaded_gates.get(&layer).unwrap().gate_weights
                    };

                    let mut gate_logits = Vec::with_capacity(num_experts);
                    for e in 0..num_experts {
                        let row_start = e * hidden_size;
                        let mut logit = 0.0;
                        // Dot product of current hidden state with gate row
                        for h in 0..hidden_size {
                            logit += tape.nodes[current_node.0].data[h] * gate_shadow[row_start + h];
                        }
                        gate_logits.push(logit);
                    }

                    let router = crate::mud::routing::MudRouter::new(num_experts, top_k);
                    let routing = router.route(&gate_logits);
                    if routing.is_empty() { break; }

                    let expert_idx = routing[0].0;
                    let gate_logit = routing[0].1;
                    expert_selections.push((layer, expert_idx, gate_logit));

                    // --- Expert FFN ---
                    let shadow = self::get_or_load_expert(&core, &mut loaded_experts, layer, expert_idx);
                    let shadow = match shadow {
                        Some(s) => s,
                        None => { break; }
                    };

                    let ffn_hidden = shadow.w1_shape[0];
                    let w1 = tape.push_leaf(shadow.shadow_w1.clone(), vec![ffn_hidden, hidden_size]);
                    let w3 = tape.push_leaf(shadow.shadow_w3.clone(), vec![ffn_hidden, hidden_size]);
                    let w2 = tape.push_leaf(shadow.shadow_w2.clone(), vec![hidden_size, ffn_hidden]);

                    let z1 = tape.linear(current_node, w1);
                    let z3 = tape.linear(current_node, w3);
                    let a = tape.silu(z1);
                    let mul_out = tape.mul(a, z3);
                    let z2 = tape.linear(mul_out, w2);
                    current_node = z2;

                } else {
                    // Single expert mode (e.g., non-MoE models mapped to expert 0)
                    let shadow = self::get_or_load_expert(&core, &mut loaded_experts, layer, 0);
                    let shadow = match shadow {
                        Some(s) => s,
                        None => { break; }
                    };

                    let ffn_hidden = shadow.w1_shape[0];
                    let w1 = tape.push_leaf(shadow.shadow_w1.clone(), vec![ffn_hidden, hidden_size]);
                    let w3 = tape.push_leaf(shadow.shadow_w3.clone(), vec![ffn_hidden, hidden_size]);
                    let w2 = tape.push_leaf(shadow.shadow_w2.clone(), vec![hidden_size, ffn_hidden]);

                    let z1 = tape.linear(current_node, w1);
                    let z3 = tape.linear(current_node, w3);
                    let a = tape.silu(z1);
                    let mul_out = tape.mul(a, z3);
                    let z2 = tape.linear(mul_out, w2);
                    current_node = z2;
                }
            }

            // --- Loss computation ---
            let mut neg_ids: Vec<usize> = Vec::new();
            if use_full_ce {
                let full_emb_node = tape.push_leaf(emb_weights.clone(), vec![vocab_size, hidden_size]);
                let logits = tape.linear(current_node, full_emb_node);
                let loss = tape.cross_entropy(logits, t_target);
                tape.backward(loss);
            } else {
                for _ in 0..NUM_NEGATIVES {
                    rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let neg = (rng_seed as usize) % vocab_size;
                    if neg != t_target && !neg_ids.contains(&neg) {
                        neg_ids.push(neg);
                    }
                }
                if neg_ids.is_empty() { continue; }

                let num_classes = 1 + neg_ids.len();
                let mut class_embs = Vec::with_capacity(num_classes * hidden_size);
                class_embs.extend_from_slice(&emb_weights[t_target * hidden_size .. (t_target + 1) * hidden_size]);
                for &neg in &neg_ids {
                    class_embs.extend_from_slice(&emb_weights[neg * hidden_size .. (neg + 1) * hidden_size]);
                }

                let emb_node = tape.push_leaf(class_embs, vec![num_classes, hidden_size]);
                let logits = tape.linear(current_node, emb_node);
                let loss = tape.cross_entropy(logits, 0);
                tape.backward(loss);
            }

            // Update input embedding (t_in) gradient
            let dx = &tape.nodes[x_node.0].grad;
            let has_nan = dx.iter().any(|&v| !v.is_finite());
            if has_nan { nan_skips += 1; continue; }

            let mut dx_norm_sq = 0.0;
            for &g in dx.iter() { dx_norm_sq += g * g; }
            let dx_norm = dx_norm_sq.sqrt();
            let dx_coef = if dx_norm > MAX_GRAD_NORM && dx_norm > 0.0 { MAX_GRAD_NORM / dx_norm } else { 1.0 };
            for j in 0..hidden_size {
                emb_weights[t_in * hidden_size + j] -= lr * dx[j] * dx_coef;
            }

            // Update expert + embedding weights from tape gradients
            self::apply_expert_updates_from_tape(&tape, &mut loaded_experts,
                &mut emb_weights, t_target, hidden_size, lr, &neg_ids, use_full_ce);
        }

        // Throttle
        let last_act_ms = LAST_ACTIVITY.load(Ordering::Relaxed) as u64;
        let now_ms = current_time_millis();
        let throttle_ms = if last_act_ms > 0 && now_ms.saturating_sub(last_act_ms) > 60_000 { 25 } else { 50 };
        thread::sleep(Duration::from_millis(throttle_ms));
        ids_to_mark.push(*id);
    }

    // Save trained weights back to .mud
    let modified_count = loaded_experts.values().filter(|s| s.modified).count();
    let modified_gates = loaded_gates.values().filter(|g| g.modified).count();
    if modified_count > 0 || modified_gates > 0 {
        println!("  [Auto-Trainer] Saving {} modified experts and {} modified gates...", modified_count, modified_gates);
        if let Err(e) = save_shadows_to_mud(model_path, &loaded_experts, &loaded_gates,
            Some(&emb_weights), vocab_size, hidden_size)
        {
            eprintln!("  [Auto-Trainer] Save failed: {}", e);
        }
    }

    if let Err(e) = store.mark_as_packed(&ids_to_mark) {
        eprintln!("  [Auto-Trainer] Failed to mark as assimilated: {}", e);
    } else {
        let _ = store.enforce_ttl();
        let _ = store.checkpoint();

        if SHOULD_TERMINATE.load(Ordering::SeqCst) {
            let total_remaining = store.get_unassimilated().map(|v| v.len()).unwrap_or(0);
            println!("\n  [SHUTDOWN TELEMETRY]");
            println!("    Batch: {}/{} chunks assimilated", ids_to_mark.len(), facts.len());
            println!("    Modified experts: {}/{}", modified_count, loaded_experts.len());
            println!("    Modified gates: {}/{}", modified_gates, loaded_gates.len());
            println!("    NaN skips: {}", nan_skips);
            println!("    Tokens processed: {}", total_tokens);
            println!("    Remaining in DB: {}", total_remaining);
        }
    }
}

/// Helper: load or retrieve an expert shadow from cache
fn get_or_load_expert<'a>(
    core: &crate::mud::MudSkill,
    loaded_experts: &'a mut HashMap<(usize, usize), ExpertShadow>,
    layer: usize,
    expert_idx: usize,
) -> Option<&'a mut ExpertShadow> {
    if loaded_experts.contains_key(&(layer, expert_idx)) {
        return loaded_experts.get_mut(&(layer, expert_idx));
    }

    let w1_name = format!("blk.{}.expert.{}.w1.weight", layer, expert_idx);
    let w2_name = format!("blk.{}.expert.{}.w2.weight", layer, expert_idx);
    let w3_name = format!("blk.{}.expert.{}.w3.weight", layer, expert_idx);
    let s1_name = format!("blk.{}.expert.{}.w1.scale", layer, expert_idx);
    let s2_name = format!("blk.{}.expert.{}.w2.scale", layer, expert_idx);
    let s3_name = format!("blk.{}.expert.{}.w3.scale", layer, expert_idx);

    let t_w1 = core.tensors.get(&w1_name)?;
    let t_w2 = core.tensors.get(&w2_name)?;
    let t_w3 = core.tensors.get(&w3_name)?;

    let get_scale = |name: &str| -> f32 {
        core.tensors.get(name)
            .filter(|t| t.t_type == MudTensorType::Float32)
            .map(|t| unsafe { *(t.data_ptr as *const f32) })
            .unwrap_or(1.0)
    };

    let s1 = get_scale(&s1_name);
    let s2 = get_scale(&s2_name);
    let s3 = get_scale(&s3_name);

    let sd_w1 = dequantize_tensor_f32(t_w1, s1);
    let sd_w2 = dequantize_tensor_f32(t_w2, s2);
    let sd_w3 = dequantize_tensor_f32(t_w3, s3);

    Some(loaded_experts.entry((layer, expert_idx)).or_insert(ExpertShadow {
        shadow_w1: sd_w1,
        shadow_w2: sd_w2,
        shadow_w3: sd_w3,
        w1_shape: t_w1.shape.clone(),
        w2_shape: t_w2.shape.clone(),
        w3_shape: t_w3.shape.clone(),
        w1_scale: s1,
        w2_scale: s2,
        w3_scale: s3,
        modified: false,
    }))
}

/// Apply expert and gate updates from tape gradients
fn apply_expert_updates_from_tape(
    tape: &Tape,
    loaded_experts: &mut HashMap<(usize, usize), ExpertShadow>,
    emb_weights: &mut [f32],
    t_target: usize,
    hidden_size: usize,
    lr: f32,
    neg_ids: &[usize],
    use_full_ce: bool,
) {
    // Collect all (shape, grad, data_len) from leaves with non-zero gradients
    struct LeafMatch {
        shape: Vec<usize>,
        grad: Vec<f32>,
        data_len: usize,
    }
    let leaves: Vec<LeafMatch> = tape.nodes.iter()
        .filter(|n| matches!(n.op, forge_autograd::Op::Leaf))
        .filter(|n| n.grad.iter().any(|&g| g != 0.0))
        .filter(|n| n.shape.len() == 2)
        .map(|n| LeafMatch { shape: n.shape.clone(), grad: n.grad.clone(), data_len: n.data.len() })
        .collect();

    for leaf in &leaves {
        for (_key, shadow) in loaded_experts.iter_mut() {
            let mut found = false;

            if leaf.shape == shadow.w1_shape && leaf.data_len == shadow.shadow_w1.len() {
                let grad = &leaf.grad;
                let mut sum_sq = 0.0;
                for &g in grad.iter() { sum_sq += g * g; }
                let norm = sum_sq.sqrt();
                let clip_coef = if norm > 1.0 && norm > 0.0 { 1.0 / norm } else { 1.0 };
                for j in 0..shadow.shadow_w1.len() {
                    // Neural Kick: add 1e-5 jitter to prevent stagnation
                    let jitter = (rand::random::<f32>() - 0.5) * 1e-5;
                    shadow.shadow_w1[j] -= lr * (grad[j] * clip_coef + WEIGHT_DECAY * shadow.shadow_w1[j]) + jitter;
                }
                shadow.modified = true;
                found = true;
            }
            if !found && leaf.shape == shadow.w3_shape && leaf.data_len == shadow.shadow_w3.len() {
                let grad = &leaf.grad;
                let mut sum_sq = 0.0;
                for &g in grad.iter() { sum_sq += g * g; }
                let norm = sum_sq.sqrt();
                let clip_coef = if norm > 1.0 && norm > 0.0 { 1.0 / norm } else { 1.0 };
                for j in 0..shadow.shadow_w3.len() {
                    let jitter = (rand::random::<f32>() - 0.5) * 1e-5;
                    shadow.shadow_w3[j] -= lr * (grad[j] * clip_coef + WEIGHT_DECAY * shadow.shadow_w3[j]) + jitter;
                }
                shadow.modified = true;
                found = true;
            }
            if !found && leaf.shape == shadow.w2_shape && leaf.data_len == shadow.shadow_w2.len() {
                let grad = &leaf.grad;
                let mut sum_sq = 0.0;
                for &g in grad.iter() { sum_sq += g * g; }
                let norm = sum_sq.sqrt();
                let clip_coef = if norm > 1.0 && norm > 0.0 { 1.0 / norm } else { 1.0 };
                for j in 0..shadow.shadow_w2.len() {
                    let jitter = (rand::random::<f32>() - 0.5) * 1e-5;
                    shadow.shadow_w2[j] -= lr * (grad[j] * clip_coef + WEIGHT_DECAY * shadow.shadow_w2[j]) + jitter;
                }
                shadow.modified = true;
            }
        }
    }

    // Update target and negative embeddings
    if !use_full_ce && !neg_ids.is_empty() {
        for leaf in &leaves {
            if leaf.shape.len() == 2 && leaf.shape[0] == 1 + neg_ids.len() && leaf.shape[1] == hidden_size {
                let demb = &leaf.grad;
                for j in 0..hidden_size {
                    emb_weights[t_target * hidden_size + j] -= lr * demb[j];
                }
                for (ni, &neg_id) in neg_ids.iter().enumerate() {
                    let cls_idx = 1 + ni;
                    for j in 0..hidden_size {
                        emb_weights[neg_id * hidden_size + j] -= lr * demb[cls_idx * hidden_size + j];
                    }
                }
                break;
            }
        }
    }
}
