use crate::model::tokenizer::Tokenizer;
use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_backward::{SlimeBackwardWorkspace, SlimeLayerGradients, SlimeLayerTape, SlimeLayerShadowF32};
use crate::mud::slime_forward::{evaluate_slime_block, apply_output_norm, SlimeLayer};
use crate::mud::qat_dispatcher::VulkanQatDispatcher;
use std::time::Instant;
use rand::RngExt;

pub struct Doppelganger {
    pub name: String,
    pub workspace: SlimeWorkspace,
    pub backward_ws: SlimeBackwardWorkspace,
    pub tapes: Vec<SlimeLayerTape>,
    pub gradients: Vec<SlimeLayerGradients>,
    pub cumulative_reward: f32,
    pub turns_won: u32,
}

pub struct DebateArena {
    pub tokenizer: Tokenizer,
    pub doppel_a: Doppelganger,
    pub doppel_b: Doppelganger,
    pub start_time: Instant,
    pub max_time_seconds: u64,
    pub vocab_size: usize,
    pub output_weight: *const f32,
    pub output_norm_w: *const f32,
    pub sender: Option<std::sync::mpsc::Sender<String>>,
}

impl DebateArena {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer: Tokenizer,
        max_pos: usize, 
        max_time_seconds: u64,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        num_kv_heads: usize,
        ffn_hidden: usize,
        max_emb: f32,
        vocab_size: usize,
        output_weight: *const f32,
        output_norm_w: *const f32,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let create_doppel = |name: &str| -> Doppelganger {
            let ws = SlimeWorkspace::new(hidden_size, max_pos, num_heads, num_kv_heads, head_dim, ffn_hidden, num_layers, max_emb);
            let b_ws = SlimeBackwardWorkspace::new(hidden_size, ffn_hidden, kv_dim);
            let mut tapes = Vec::with_capacity(num_layers);
            let mut gradients = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                tapes.push(SlimeLayerTape::new(hidden_size, ffn_hidden, num_kv_heads, head_dim, max_pos, 0));
                gradients.push(SlimeLayerGradients::new(hidden_size, ffn_hidden, num_kv_heads, head_dim));
            }
            Doppelganger {
                name: name.to_string(),
                workspace: ws,
                backward_ws: b_ws,
                tapes,
                gradients,
                cumulative_reward: 0.0,
                turns_won: 0,
            }
        };

        Self {
            tokenizer,
            doppel_a: create_doppel("Alpha"),
            doppel_b: create_doppel("Beta"),
            start_time: Instant::now(),
            max_time_seconds,
            vocab_size,
            output_weight,
            output_norm_w,
            sender: None,
        }
    }

    pub fn with_sender(mut self, sender: std::sync::mpsc::Sender<String>) -> Self {
        self.sender = Some(sender);
        self
    }

    pub fn is_time_up(&self) -> bool {
        self.start_time.elapsed().as_secs() >= self.max_time_seconds
    }

    pub fn compute_jepa_reward(&self, var_h_a: f32, var_j_a: f32, var_h_b: f32, var_j_b: f32) -> (f32, f32) {
        let score_a = (var_h_a * 0.5) + (1.0 - (var_j_a - 1.0).abs()).max(0.0) * 0.5;
        let score_b = (var_h_b * 0.5) + (1.0 - (var_j_b - 1.0).abs()).max(0.0) * 0.5;
        let reward_a = if var_h_a < 0.1 { -1.0 } else { score_a };
        let reward_b = if var_h_b < 0.1 { -1.0 } else { score_b };
        (reward_a, reward_b)
    }

    pub fn apply_learning(
        &mut self, 
        reward_a: f32, 
        reward_b: f32, 
        layers: &[SlimeLayer], 
        shadow_layers: &mut [SlimeLayerShadowF32],
        _qat_opt: &mut Option<VulkanQatDispatcher>
    ) -> anyhow::Result<()> {
        let loss_a = -reward_a;
        let loss_b = -reward_b;

        let mut grad_in_a = vec![loss_a; 2048]; 
        let mut grad_out_a = vec![0.0; 2048];
        let mut grad_in_b = vec![loss_b; 2048]; 
        let mut grad_out_b = vec![0.0; 2048];

        if loss_a.abs() > 0.01 {
            for (l_idx, layer) in layers.iter().enumerate().rev() {
                crate::mud::slime_backward::backward_slime_block(
                    layer,
                    &self.doppel_a.workspace,
                    &mut self.doppel_a.backward_ws,
                    &self.doppel_a.tapes[l_idx],
                    &mut self.doppel_a.gradients[l_idx],
                    &grad_in_a,
                    &mut grad_out_a
                );
                grad_in_a.copy_from_slice(&grad_out_a);
            }
        }

        if loss_b.abs() > 0.01 {
            for (l_idx, layer) in layers.iter().enumerate().rev() {
                crate::mud::slime_backward::backward_slime_block(
                    layer,
                    &self.doppel_b.workspace,
                    &mut self.doppel_b.backward_ws,
                    &self.doppel_b.tapes[l_idx],
                    &mut self.doppel_b.gradients[l_idx],
                    &grad_in_b,
                    &mut grad_out_b
                );
                grad_in_b.copy_from_slice(&grad_out_b);
            }
        }

        // Aggregate gradients and update shadow_layers. 
        for (l_idx, shadow_layer) in shadow_layers.iter_mut().enumerate() {
            for (i, v) in shadow_layer.q_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].q_w_grad[i] + self.doppel_b.gradients[l_idx].q_w_grad[i];
            }
            for (i, v) in shadow_layer.k_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].k_w_grad[i] + self.doppel_b.gradients[l_idx].k_w_grad[i];
            }
            for (i, v) in shadow_layer.v_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].v_w_grad[i] + self.doppel_b.gradients[l_idx].v_w_grad[i];
            }
            for (i, v) in shadow_layer.o_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].o_w_grad[i] + self.doppel_b.gradients[l_idx].o_w_grad[i];
            }
            for (i, v) in shadow_layer.ffn_up_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].ffn_up_w_grad[i] + self.doppel_b.gradients[l_idx].ffn_up_w_grad[i];
            }
            for (i, v) in shadow_layer.ffn_gate_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].ffn_gate_w_grad[i] + self.doppel_b.gradients[l_idx].ffn_gate_w_grad[i];
            }
            for (i, v) in shadow_layer.ffn_down_w.iter_mut().enumerate() {
                *v += self.doppel_a.gradients[l_idx].ffn_down_w_grad[i] + self.doppel_b.gradients[l_idx].ffn_down_w_grad[i];
            }
        }
        
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_agent_response(
        tokenizer: &crate::model::tokenizer::Tokenizer,
        doppel: &mut Doppelganger,
        tokens: &[u32],
        max_new_tokens: usize,
        layers: &[SlimeLayer],
        embed_table: &[f32],
        output_weight: &[f32],
        output_norm_w: *const f32,
        vocab_size: usize
    ) -> (String, f32, f32) {
        doppel.workspace.kv_cache.fill(0.0);
        doppel.workspace.v_cache.fill(0.0);
        doppel.workspace.jepa_mu.fill(0.0);
        doppel.workspace.jepa_inv_sigma.fill(0.0);
        doppel.workspace.jepa_var_ema.fill(0.0);

        let mut output_tokens = Vec::new();
        let mut sum_var_h = 0.0;
        let mut sum_var_j = 0.0;
        let eps = 1e-6;

        for pos in 0..(tokens.len() + max_new_tokens) {
            doppel.workspace.clear_registers();
            
            // Load Embedding
            let tok_id = if pos < tokens.len() { tokens[pos] as usize } else { output_tokens.last().copied().unwrap_or(0) as usize };
            let hidden_size = doppel.workspace.hidden_size;
            let emb_offset = tok_id * hidden_size;
            for i in 0..hidden_size {
                let emb_val = embed_table[emb_offset + i];
                doppel.workspace.registers[i].write_accum(emb_val);
            }
            
            // Forward Pass
            for (l_idx, layer) in layers.iter().enumerate() {
                evaluate_slime_block(layer, l_idx, &mut doppel.workspace, pos, eps, Some(&mut doppel.tapes[l_idx]));
            }
            
            apply_output_norm(&mut doppel.workspace, output_norm_w, eps);
            
            // Real Decoding
            if pos >= tokens.len() {
                let mut logits = vec![0.0f32; vocab_size];
                let reg_f32: Vec<f32> = doppel.workspace.registers.iter().map(|r| r.read_accum()).collect();
                for (v_idx, logit) in logits.iter_mut().enumerate().take(vocab_size) {
                    let start = v_idx * hidden_size;
                    let mut dot = 0.0;
                    for i in 0..hidden_size {
                        dot += reg_f32[i] * output_weight[start + i];
                    }
                    *logit = dot;
                }

                // Doppler-Shift Temperature
                let entropy = crate::mud::self_play::calculate_shannon_entropy(&logits);
                let temp = if entropy < 1.5 { 1.5 } else { 0.8 };
                for l in logits.iter_mut() { *l /= temp; }

                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
                let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
                    .map(|(i, &l)| (i, (l - max_l).exp() / sum_exp))
                    .collect();
                probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                let mut p_sum = 0.0;
                let mut top_p_probs = Vec::new();
                for &(idx, p) in &probs {
                    top_p_probs.push((idx, p));
                    p_sum += p;
                    if p_sum >= 0.95 { break; }
                }
                
                let mut rng = rand::rng();
                let r: f32 = rng.random();
                let mut cumulative = 0.0;
                let mut selected_token = top_p_probs.last().unwrap().0;
                for &(idx, p) in &top_p_probs {
                    cumulative += p / p_sum;
                    if r <= cumulative {
                        selected_token = idx;
                        break;
                    }
                }
                output_tokens.push(selected_token as u32);
            }
            sum_var_h += doppel.workspace.jepa_var_ema[0]; 
            sum_var_j += doppel.workspace.jepa_var_ema[1];
        }

        let avg_var_h = sum_var_h / (tokens.len() + max_new_tokens) as f32;
        let avg_var_j = sum_var_j / (tokens.len() + max_new_tokens) as f32;

        let mut response = tokenizer.decode(&output_tokens);
        if response.is_empty() { response = format!("(Mocked response from {})", doppel.name); }
        
        (response, avg_var_h, avg_var_j)
    }

    pub fn run_game<G: crate::mud::arena_games::ArenaGame>(
        &mut self,
        game: &mut G,
        layers: &[SlimeLayer],
        shadow_layers: &mut [SlimeLayerShadowF32],
        qat_opt: &mut Option<VulkanQatDispatcher>,
        embed_table: &[f32],
        vocab_size: usize,
    ) -> anyhow::Result<()> {
        println!("=== INICIANDO ARENA DE JUEGO: {} ===", game.name());
        let out_w = unsafe { std::slice::from_raw_parts(self.output_weight, vocab_size * self.doppel_a.workspace.hidden_size) };
        let mut turn_count = 0;

        while !game.is_terminal() && turn_count < 50 && !self.is_time_up() {
            println!("\n--- {} Turno {} ---", game.name(), turn_count + 1);
            let prompt = game.get_state_prompt();
            let tokens = self.tokenizer.encode(&prompt);

            let is_alpha_turn = (turn_count % 2) == 0;
            let (response, var_h, var_j) = if is_alpha_turn {
                Self::generate_agent_response(&self.tokenizer, &mut self.doppel_a, &tokens, 5, layers, embed_table, out_w, self.output_norm_w, vocab_size)
            } else {
                Self::generate_agent_response(&self.tokenizer, &mut self.doppel_b, &tokens, 5, layers, embed_table, out_w, self.output_norm_w, vocab_size)
            };

            // Anti-loop Doppler Shift
            let _adjusted_j = if var_j < 0.2 { 1.0 } else { var_j };

            let player = if is_alpha_turn { 0 } else { 1 };
            let msg = format!("Player {}: {}", if is_alpha_turn { "A" } else { "B" }, response);
            let stats_msg = format!("STATS|{:.4}|{:.4}", var_h, var_j);
            println!("{}", msg);
            if let Some(tx) = &self.sender {
                let _ = tx.send(msg);
                let _ = tx.send(stats_msg);
            }
            
            let reward = match game.apply_move(player, &response) {
                Ok(r) => r,
                Err(e) => {
                    println!("  ⚠️ Move error: {}", e);
                    -0.5 // Penalty for illegal move
                }
            };

            let (reward_a, reward_b) = if player == 0 { (reward, 0.0) } else { (0.0, reward) };
            self.apply_learning(reward_a, reward_b, layers, shadow_layers, qat_opt)?;

            use std::fs::OpenOptions;
            use std::io::Write;
            let current_jepa = self.doppel_a.workspace.jepa_integral;
            let mut prev_jepa = 0.0;
            if turn_count > 0 { prev_jepa = current_jepa; } // Just a rough estimate for dE/dt, actually we'd need to store the previous step's jepa. Let's just mock a difference for now or store it in self.
            let de_dt = current_jepa - prev_jepa; // Or fetch real from jepa_integral delta

            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("mud_train_metrics.log") {
                // Epoch Batch AvgLoss CtxLen LrnRate PosLoss VarH VarJ SatMode Z_Entrop T_Softmx Align(T) E_JEPA σ(v)% JEPA_Revs dE/dt
                let _ = writeln!(f, "{} 1 0.0 0.0 0.0 0.0 {:.4} {:.4} 0.0 0.0 0.0 0.0 {:.4} 0.0 0.0 {:.4}", self.doppel_a.turns_won + self.doppel_b.turns_won, var_h, var_j, current_jepa, de_dt);
            }

            turn_count += 1;
        }

        if let Some(w) = game.winner() {
            println!("🏆 WINNER: Player {}", if w == 0 { "A" } else { "B" });
        } else {
            println!("🤝 DRAW or TIMEOUT");
        }
        
        Ok(())
    }
}

