use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_backward::SlimeLayerGradients;

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

use crate::mud::slime_forward::{
    apply_output_norm, evaluate_slime_block, layer_is_valid, SlimeLayer,
};

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
            evaluate_slime_block(layer, l_idx, ws, gen_pos, eps, None);
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
            .map(|r| r.matmul_accum)
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
        if p > 1e-8 {
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
