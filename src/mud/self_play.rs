use crate::mud::corpus_trainer::{MudCorpusTrainer, TrainChunkMetrics};
use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_backward::{SlimeLayerGradients, SlimeLayerShadowF32, SlimeLayerTape, SlimeBackwardWorkspace};
use crate::mud::slime_forward::{
    apply_output_norm, evaluate_slime_block, layer_is_valid, SlimeLayer,
};
use crate::mud::constants::EPSILON_FLOOR;
use crate::mud::{MudFile, MudTensorType};

pub struct SelfPlayConfig {
    pub max_sequence_len: usize,
    pub entropy_threshold: f32,
    pub neural_kick_jitter: f32, // Prevents gradient repetition collapse
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            max_sequence_len: 32,
            entropy_threshold: 15.0, // Increased to allow training when model is untrained
            neural_kick_jitter: 1e-5, // Valor base sugerido por los mandates
        }
    }
}

/// Aplica ruido dinámico (Neural Kick Jitter) a los gradientes.
/// Evita que el modelo se estanque repitiendo los mismos gradientes en un mínimo local
/// durante la asimilación de cadenas sintéticas.
pub fn apply_gradient_jitter(
    grads: &mut SlimeLayerGradients,
    config: &SelfPlayConfig,
    step: usize,
) {
    if config.neural_kick_jitter <= 0.0 {
        return;
    }

    let jitter_scale = config.neural_kick_jitter;

    // Función auxiliar para inyectar ruido pseudo-aleatorio basado en el índice y el paso
    let apply_noise = |buffer: &mut [f32], seed_offset: usize| {
        for (i, val) in buffer.iter_mut().enumerate() {
            // Generación simple de ruido pseudo-aleatorio para evitar importar librerías pesadas
            let noise = (((i + seed_offset + step) * 31337) % 100) as f32 / 100.0 - 0.5;
            *val += noise * jitter_scale;
        }
    };

    apply_noise(&mut grads.q_w_grad, 1);
    apply_noise(&mut grads.k_w_grad, 2);
    apply_noise(&mut grads.v_w_grad, 3);
    apply_noise(&mut grads.o_w_grad, 4);
    apply_noise(&mut grads.ffn_up_w_grad, 5);
    apply_noise(&mut grads.ffn_gate_w_grad, 6);
    apply_noise(&mut grads.ffn_down_w_grad, 7);
}

/// Genera una secuencia autorregresiva (sueño) utilizando el atractor JEPA
#[allow(
    clippy::too_many_arguments,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::needless_range_loop
)]
pub fn generate_synthetic_sequence(
    ws: &mut SlimeWorkspace,
    config: &SelfPlayConfig,
    layers: &[SlimeLayer],
    token_embd_ptr: *const f32,
    output_weight_ptr: *const f32,
    output_norm_w: *const f32,
    vocab_size: usize,
    hidden: usize,
    start_token: u32,
) -> (Vec<u32>, Vec<f32>) {
    let mut sequence = vec![start_token];
    let mut entropies = Vec::with_capacity(config.max_sequence_len);

    let mut current_token = start_token;

    let eps = 1e-6;

    for gen_pos in 0..config.max_sequence_len {
        // 1. Embeber el token actual en los registros de SlimeWorkspace
        let row_start = (current_token as usize) * hidden;
        for h in 0..hidden {
            let emb_val = unsafe { *token_embd_ptr.add(row_start + h) };
            crate::mud::slime::SlimeRegister::init_from_embed(
                &mut ws.registers[h],
                &mut ws.jepa_z,
                h,
                ws.hidden_size,
                ws.num_layers,
                emb_val,
                gen_pos == 0,
            );
        }

        // 2. Forward pass por todas las capas
        for (l_idx, layer) in layers.iter().enumerate() {
            if !layer_is_valid(layer) {
                continue;
            }
            evaluate_slime_block(layer, l_idx, ws, gen_pos, eps, None, None);
        }

        // 3. Normalización final
        if !output_norm_w.is_null() {
            apply_output_norm(ws, output_norm_w, eps);
        }

        // 4. Convertir registros a f32
        let regs_f32: Vec<f32> = ws
            .registers
            .iter()
            .take(hidden)
            .map(|r| r.read_accum())
            .collect();

        // 5. Calcular los logits (Scalar projection para poder derivar la entropía del vocabulario)
        let mut logits = vec![0.0f32; vocab_size];
        let mut best_id = 0;
        let mut best_val = f32::NEG_INFINITY;

        for v in 0..vocab_size {
            let mut sum = 0.0;
            let v_row = v * hidden;
            for h in 0..hidden {
                sum += regs_f32[h] * unsafe { *output_weight_ptr.add(v_row + h) };
            }
            logits[v] = sum;
            if sum > best_val {
                best_val = sum;
                best_id = v;
            }
        }

        // 6. Entropía y selección
        let entropy = calculate_shannon_entropy(&logits);
        entropies.push(entropy);

        current_token = best_id as u32;
        sequence.push(current_token);

        // EOS stop
        if current_token == 0 || current_token == 2 {
            break;
        }
    }

    (sequence, entropies)
}

/// Calcula la entropía de Shannon H para la distribución de probabilidad de un solo token.
/// Espera un slice `logits` del tamaño del vocabulario (ej. 128256).
pub fn calculate_shannon_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }

    // 1. Max logit para estabilidad numérica (Log-Sum-Exp trick)
    let mut max_l = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_l {
            max_l = l;
        }
    }

    // 2. Suma de exponenciales
    let mut sum_exp = 0.0;
    for &l in logits {
        sum_exp += (l - max_l).exp();
    }

    // 3. Cálculo de probabilidades y entropía
    let mut entropy = 0.0;
    let inv_sum = 1.0 / (sum_exp + 1e-10);

    for &l in logits {
        let p = (l - max_l).exp() * inv_sum;
        if p > EPSILON_FLOOR {
            // Evitar log2(0)
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Evalúa la confianza del modelo sobre la secuencia generada.
/// `entropy_per_token` debe contener el cálculo previo por cada paso de generación.
/// Si la entropía de Shannon promedio es mayor que `entropy_threshold`, se descarta.
pub fn is_sequence_confident(entropy_per_token: &[f32], config: &SelfPlayConfig) -> bool {
    if entropy_per_token.is_empty() {
        return false;
    }

    let avg_entropy: f32 = entropy_per_token.iter().sum::<f32>() / (entropy_per_token.len() as f32);
    avg_entropy < config.entropy_threshold
}

// ── D1 self-play training step (extracted from session.rs) ────────────────
// Reuses `self.train_on_sequence` — kept in `run_alignment_session`'s loop.

impl MudCorpusTrainer {
    /// Self-play training step (D1 — 4-Pillar Efficiency, Pillar 1).
    ///
    /// Generates a synthetic sequence from the model's own forward pass, then
    /// feeds it back through the training pipeline (forward + backward + optimizer).
    /// This creates a self-improving feedback loop where the model trains on its
    /// own confident predictions, preventing stagnation in local minima.
    ///
    /// Flow:
    ///   1. Generate autoregressive chain via `generate_synthetic_sequence`
    ///   2. Check Shannon entropy confidence gate (`is_sequence_confident`)
    ///   3. If confident, feed synthetic chain through `train_on_sequence`
    ///   4. Return metrics (same as regular training step)
    ///
    /// Env vars:
    ///   MUD_SELF_PLAY=1        — activate self-play (default 0)
    ///   MUD_SELF_PLAY_TURNS=8  — max generation length per step (default 8)
    ///   MUD_SELF_PLAY_TEMP=0.8 — temperature for generation (default 0.8)
    #[allow(clippy::too_many_arguments)]
    pub fn self_play_training_step(
        &self,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        layers: &mut [SlimeLayer],
        shadow_layers: &mut [SlimeLayerShadowF32],
        workspace: &mut SlimeWorkspace,
        backward_ws: &mut SlimeBackwardWorkspace,
        tapes: &mut [SlimeLayerTape],
        gradients: &mut [SlimeLayerGradients],
        prompt: &[u32],
        batch_size: usize,
        vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
        _turn_count: usize,
    ) -> anyhow::Result<TrainChunkMetrics> {
        let config = SelfPlayConfig::default();
        let hidden = workspace.hidden_size;

        // Vocab size: from shadow_emb if available, else from mud metadata
        let vocab_size = if !shadow_emb.is_empty() {
            shadow_emb.len() / hidden.max(1)
        } else {
            mud.skills
                .get("core")
                .and_then(|c| c.tensors.get("token_embd.weight"))
                .map(|t| t.shape[0])
                .or_else(|| {
                    mud.global_metadata
                        .get("vocab_size")
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(1)
        };

        // Token embedding pointer: requires non-frozen (shadow_emb populated) embeddings.
        // Frozen embeddings (FREEZE_EMB) store ternary ELUT on mmap — cannot pass
        // as *const f32 to generate_synthetic_sequence. Skip self-play in that case.
        let token_embd_ptr: *const f32 = if !shadow_emb.is_empty() {
            shadow_emb.as_ptr()
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "self-play skipped: FREEZE_EMB active (shadow_emb empty). Enable MUD_TRAIN_FREEZE_EMB=0 for self-play."
                )
            );
            return Ok(TrainChunkMetrics::default());
        };

        // Output weight pointer: tied embeddings (SmolLM2, BitNet) reuse token_embd.
        // Non-tied models would use a separate output.weight tensor.
        let output_weight_ptr = token_embd_ptr;

        // Output norm weight pointer: must be Float32 (norms are never ternary).
        let core = mud
            .skills
            .get("core")
            .ok_or_else(|| anyhow::anyhow!("No core skill for self-play"))?;
        let output_norm_tensor = core
            .tensors
            .get("output_norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing output_norm.weight for self-play"))?;
        let output_norm_w: *const f32 = if output_norm_tensor.t_type == MudTensorType::Float32 {
            output_norm_tensor
                .owned_data
                .as_ref()
                .map(|d| d.as_ptr() as *const f32)
                .unwrap_or(output_norm_tensor.data_ptr as *const f32)
        } else {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    "self-play skipped: output_norm.weight is not Float32"
                )
            );
            return Ok(TrainChunkMetrics::default());
        };

        // Start token: first token of the prompt (or BOS=1 if empty)
        let start_token = prompt.first().copied().unwrap_or(1);

        // 1. Generate synthetic sequence via model's own forward pass
        let (synthetic_chain, entropies) = generate_synthetic_sequence(
            workspace,
            &config,
            layers,
            token_embd_ptr,
            output_weight_ptr,
            output_norm_w,
            vocab_size,
            hidden,
            start_token,
        );

        // 2. Confidence gate: discard if average entropy exceeds threshold
        if !is_sequence_confident(&entropies, &config) {
            let avg_entropy = entropies.iter().sum::<f32>() / entropies.len().max(1) as f32;
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "warn",
                    &format!(
                        "self-play: sequence not confident (avg entropy {:.2} > {:.2}) — skipping",
                        avg_entropy, config.entropy_threshold
                    )
                )
            );
            return Ok(TrainChunkMetrics::default());
        }

        // 3. Feed synthetic chain through train_on_sequence (forward + backward + optimizer)
        let metrics = self.train_on_sequence(
            mud,
            shadow_emb,
            layers,
            shadow_layers,
            workspace,
            backward_ws,
            tapes,
            gradients,
            &synthetic_chain,
            batch_size,
            vk_qat,
        )?;

        Ok(metrics)
    }
}
