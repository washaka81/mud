use crate::model::tokenizer::Tokenizer;
use crate::mud::arena_judge::{Player, ProfessorJudge, TextJudge, VerifiableJudge};
use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_backward::{
    SlimeBackwardWorkspace, SlimeLayerGradients, SlimeLayerShadowF32, SlimeLayerTape,
};
use crate::mud::slime_forward::{apply_output_norm, evaluate_slime_block, SlimeLayer};

use rand::RngExt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct Doppelganger {
    pub name: String,
    pub workspace: SlimeWorkspace,
    pub backward_ws: SlimeBackwardWorkspace,
    pub tapes: Vec<SlimeLayerTape>,
    pub gradients: Vec<SlimeLayerGradients>,
    pub cumulative_reward: f32,
    pub turns_won: u32,
    pub seed: u64,
}

pub struct DebateArena {
    pub tokenizer: Tokenizer,
    pub doppel_a: Doppelganger,
    pub doppel_b: Doppelganger,
    pub bos_id: u32,
    pub max_new_tokens: usize,
    pub start_time: Instant,
    pub max_time_seconds: u64,
    pub vocab_size: usize,
    pub output_weight: *const f32,
    pub output_norm_w: *const f32,
    pub sender: Option<std::sync::mpsc::Sender<String>>,
    pub stop_flag: Arc<AtomicBool>,
    last_a_resp: String,
    last_b_resp: String,
}

impl DebateArena {
    /// Auto-tune tokens-per-turn from free RAM so we neither swap on 15 GiB
    /// laptops nor waste headroom when more is available. CPU generation cost
    /// grows linearly with tokens; memory headroom is the safety bound.
    fn auto_max_new_tokens() -> usize {
        let free_mb = {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            sys.available_memory() / 1024 / 1024
        };
        // Free RAM -> tokens (each generated token is one 147M forward; cheap
        // in RAM, but we cap to keep turns responsive on low-memory boxes).
        if free_mb < 2_048 {
            16
        } else if free_mb < 4_096 {
            24
        } else if free_mb < 8_192 {
            32
        } else if free_mb < 16_384 {
            48
        } else if free_mb < 32_768 {
            64
        } else {
            96
        }
    }

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

        let create_doppel = |name: &str, seed: u64| -> Doppelganger {
            let ws = SlimeWorkspace::new(
                hidden_size,
                max_pos,
                num_heads,
                num_kv_heads,
                head_dim,
                ffn_hidden,
                num_layers,
                max_emb,
            );
            let b_ws = SlimeBackwardWorkspace::new(hidden_size, ffn_hidden, kv_dim);
            let mut tapes = Vec::with_capacity(num_layers);
            let mut gradients = Vec::with_capacity(num_layers);
            for _ in 0..num_layers {
                tapes.push(SlimeLayerTape::new(
                    hidden_size,
                    ffn_hidden,
                    num_kv_heads,
                    head_dim,
                    max_pos,
                    0,
                ));
                gradients.push(SlimeLayerGradients::new(
                    hidden_size,
                    ffn_hidden,
                    num_kv_heads,
                    head_dim,
                ));
            }
            Doppelganger {
                name: name.to_string(),
                workspace: ws,
                backward_ws: b_ws,
                tapes,
                gradients,
                cumulative_reward: 0.0,
                turns_won: 0,
                seed,
            }
        };

        Self {
            tokenizer,
            doppel_a: create_doppel("Alpha", 0xABC0),
            doppel_b: create_doppel("Beta", 0xBEEF),
            bos_id: std::env::var("MUD_DEBATE_BOS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0),
            max_new_tokens: std::env::var("MUD_DEBATE_MAX_NEW_TOKENS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(Self::auto_max_new_tokens),
            start_time: Instant::now(),
            max_time_seconds: std::env::var("MUD_DEBATE_MAX_TIME")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(max_time_seconds),
            vocab_size,
            output_weight,
            output_norm_w,
            sender: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            last_a_resp: String::new(),
            last_b_resp: String::new(),
        }
    }

    pub fn with_sender(mut self, sender: std::sync::mpsc::Sender<String>) -> Self {
        self.sender = Some(sender);
        self
    }

    pub fn with_stop_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.stop_flag = flag;
        self
    }

    pub fn is_time_up(&self) -> bool {
        if self.stop_flag.load(Ordering::SeqCst) {
            return true;
        }
        self.start_time.elapsed().as_secs() >= self.max_time_seconds
    }

    pub fn compute_jepa_reward(
        &self,
        var_h_a: f32,
        var_j_a: f32,
        var_h_b: f32,
        var_j_b: f32,
    ) -> (f32, f32) {
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
        layers: &mut [SlimeLayer],
        shadow_layers: &mut [SlimeLayerShadowF32],
        pool: &crate::mud::pcore_pool::PCorePool,
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
                    &mut grad_out_a,
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
                    &mut grad_out_b,
                );
                grad_in_b.copy_from_slice(&grad_out_b);
            }
        }

        let lr = crate::mud::constants::QAT_LEARNING_RATE;
        let decay = 0.01;
        let num_tokens = 1.0; // Gradients are already computed per token

        for (l_idx, shadow_layer) in shadow_layers.iter_mut().enumerate() {
            let grad_a = &self.doppel_a.gradients[l_idx];
            let grad_b = &self.doppel_b.gradients[l_idx];

            // Function to sum gradients and apply optimizer
            let apply_opt =
                |shadow_w: &mut [f32],
                 g_a: &[f32],
                 g_b: &[f32],
                 packed_ptr: *mut u8,
                 scales_ptr: *mut f32,
                 cols: usize,
                 strategy: crate::mud::slime_backward::OptimizerStrategy| {
                    if packed_ptr.is_null() || scales_ptr.is_null() {
                        return;
                    }
                    let mut combined_grad = vec![0.0f32; shadow_w.len()];
                    // Seed-survival (elitism): only apply the gradient of the winner.
                    // The loser's mutation is discarded, simulating replacement by the mutated clone.
                    let g_elite = if reward_a >= reward_b { g_a } else { g_b };
                    combined_grad.copy_from_slice(g_elite);
                    unsafe {
                        crate::mud::corpus_trainer::apply_optimizer_cpu_step_and_pack(
                            shadow_w,
                            &combined_grad,
                            packed_ptr,
                            scales_ptr,
                            lr,
                            decay,
                            num_tokens,
                            cols,
                            pool,
                            strategy,
                            None, // debate path: SGD after strategy preprocess (no persistent Adam bank yet)
                        );
                    }
                };

            let hidden = self.doppel_a.workspace.hidden_size;
            apply_opt(
                &mut shadow_layer.q_w,
                &grad_a.q_w_grad,
                &grad_b.q_w_grad,
                layers[l_idx].q_w as *mut u8,
                layers[l_idx].q_scales as *mut f32,
                hidden,
                shadow_layer.q_opt,
            );
            apply_opt(
                &mut shadow_layer.k_w,
                &grad_a.k_w_grad,
                &grad_b.k_w_grad,
                layers[l_idx].k_w as *mut u8,
                layers[l_idx].k_scales as *mut f32,
                hidden,
                shadow_layer.k_opt,
            );
            apply_opt(
                &mut shadow_layer.v_w,
                &grad_a.v_w_grad,
                &grad_b.v_w_grad,
                layers[l_idx].v_w as *mut u8,
                layers[l_idx].v_scales as *mut f32,
                hidden,
                shadow_layer.v_opt,
            );
            apply_opt(
                &mut shadow_layer.o_w,
                &grad_a.o_w_grad,
                &grad_b.o_w_grad,
                layers[l_idx].o_w as *mut u8,
                layers[l_idx].o_scales as *mut f32,
                hidden,
                shadow_layer.o_opt,
            );

            apply_opt(
                &mut shadow_layer.ffn_up_w,
                &grad_a.ffn_up_w_grad,
                &grad_b.ffn_up_w_grad,
                layers[l_idx].ffn_up_w as *mut u8,
                layers[l_idx].ffn_up_scales as *mut f32,
                hidden,
                shadow_layer.ffn_up_opt,
            );
            apply_opt(
                &mut shadow_layer.ffn_gate_w,
                &grad_a.ffn_gate_w_grad,
                &grad_b.ffn_gate_w_grad,
                layers[l_idx].ffn_gate_w as *mut u8,
                layers[l_idx].ffn_gate_scales as *mut f32,
                hidden,
                shadow_layer.ffn_gate_opt,
            );

            let ffn_mid = shadow_layer.ffn_down_w.len() / hidden.max(1);
            apply_opt(
                &mut shadow_layer.ffn_down_w,
                &grad_a.ffn_down_w_grad,
                &grad_b.ffn_down_w_grad,
                layers[l_idx].ffn_down_w as *mut u8,
                layers[l_idx].ffn_down_scales as *mut f32,
                ffn_mid,
                shadow_layer.ffn_down_opt,
            );
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_agent_response(
        tokenizer: &crate::model::tokenizer::Tokenizer,
        doppel: &mut Doppelganger,
        tokens: &[u32],
        max_new_tokens: usize,
        bos_id: u32,
        layers: &[SlimeLayer],
        embed_table: &[f32],
        output_weight: &[f32],
        output_norm_w: *const f32,
        vocab_size: usize,
    ) -> (String, f32, f32) {
        // Prepend BOS so the model sees a well-formed context (SmolLM2 bos=eos=0).
        let prompt_tokens = if bos_id > 0 {
            let mut v = vec![bos_id];
            v.extend_from_slice(tokens);
            v
        } else {
            tokens.to_vec()
        };
        let tokens = &prompt_tokens;
        doppel.workspace.kv_cache.fill(0.0);
        doppel.workspace.v_cache.fill(0.0);
        doppel.workspace.jepa_mu.fill(0.0);
        doppel.workspace.jepa_inv_sigma.fill(0.0);
        doppel.workspace.jepa_var_ema.fill(0.0);

        let mut output_tokens = Vec::new();
        let mut final_var_h = 0.0;
        let mut final_var_j = 0.0;
        let eps = 1e-6;

        for pos in 0..(tokens.len() + max_new_tokens) {
            doppel.workspace.clear_registers();

            // Load Embedding
            let tok_id = if pos < tokens.len() {
                tokens[pos] as usize
            } else {
                output_tokens.last().copied().unwrap_or(0) as usize
            };
            let hidden_size = doppel.workspace.hidden_size;
            let num_layers = doppel.workspace.num_layers;
            let emb_offset = tok_id * hidden_size;
            for i in 0..hidden_size {
                let emb_val = embed_table[emb_offset + i];
                crate::mud::slime::SlimeRegister::init_from_embed(
                    &mut doppel.workspace.registers[i],
                    &mut doppel.workspace.jepa_z,
                    i,
                    hidden_size,
                    num_layers,
                    emb_val,
                    pos == 0,
                );
            }

            // Forward Pass
            for (l_idx, layer) in layers.iter().enumerate() {
                evaluate_slime_block(
                    layer,
                    l_idx,
                    &mut doppel.workspace,
                    pos,
                    eps,
                    Some(&mut doppel.tapes[l_idx]),
                );
            }

            apply_output_norm(&mut doppel.workspace, output_norm_w, eps);

            // Real Decoding
            if pos >= tokens.len() {
                let mut logits = vec![0.0f32; vocab_size];
                let reg_f32: Vec<f32> = doppel
                    .workspace
                    .registers
                    .iter()
                    .map(|r| r.matmul_accum)
                    .collect();
                for (v_idx, logit) in logits.iter_mut().enumerate().take(vocab_size) {
                    let start = v_idx * hidden_size;
                    let mut dot = 0.0;
                    for i in 0..hidden_size {
                        dot += reg_f32[i] * output_weight[start + i];
                    }
                    *logit = dot;
                }

                // DC Bias Removal (matches main.rs inference path)
                let logit_mean = logits.iter().sum::<f32>() / vocab_size as f32;
                for l in logits.iter_mut() {
                    *l -= logit_mean;
                }

                // Thermodynamic rescaling — boost peak to ~8.0 only if real signal
                let shifted_max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let min_peak = 0.05f32;
                let target_peak = 8.0f32;
                if shifted_max > min_peak {
                    let boost = (target_peak / shifted_max).clamp(0.5, 16.0);
                    for l in logits.iter_mut() {
                        *l *= boost;
                    }
                }

                // Repetition penalty (break ternary loops / digit spam)
                let rep = 1.25f32;
                let window = output_tokens.len().saturating_sub(64);
                for &prev_token in &output_tokens[window..] {
                    let i = prev_token as usize;
                    if i >= logits.len() {
                        continue;
                    }
                    if logits[i] > 0.0 {
                        logits[i] /= rep;
                    } else {
                        logits[i] *= rep;
                    }
                }

                // Doppler-Shift Temperature
                let entropy = crate::mud::self_play::calculate_shannon_entropy(&logits);
                let temp = if entropy < 1.5 { 1.5 } else { 0.8 };
                for l in logits.iter_mut() {
                    *l /= temp;
                }

                let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
                let mut probs: Vec<(usize, f32)> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &l)| (i, (l - max_l).exp() / sum_exp))
                    .collect();
                probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let mut p_sum = 0.0;
                let mut top_p_probs = Vec::new();
                for &(idx, p) in &probs {
                    top_p_probs.push((idx, p));
                    p_sum += p;
                    if p_sum >= 0.95 {
                        break;
                    }
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
            let len_half = (doppel.workspace.jepa_var_ema.len() / 2).max(1) as f32;
            final_var_h = doppel.workspace.jepa_var_ema.iter().step_by(2).sum::<f32>() / len_half;
            final_var_j = doppel
                .workspace
                .jepa_var_ema
                .iter()
                .skip(1)
                .step_by(2)
                .sum::<f32>()
                / len_half;
        }

        let mut response = tokenizer.decode(&output_tokens);
        if response.is_empty() {
            response = format!("(Mocked response from {})", doppel.name);
        }

        (response, final_var_h, final_var_j)
    }

    pub fn run_game<F>(
        &mut self,
        game_factory: F,
        layers: &mut [SlimeLayer],
        shadow_layers: &mut [SlimeLayerShadowF32],
        embed_table: &[f32],
        vocab_size: usize,
        pool: &crate::mud::pcore_pool::PCorePool,
    ) -> anyhow::Result<()>
    where
        F: Fn() -> Box<dyn crate::mud::arena_games::ArenaGame>,
    {
        let print_stdout = self.sender.is_none();
        let mut game = game_factory();
        let game_name = game.name().to_string();

        if print_stdout {
            println!(
                "{}",
                crate::mud::trainer_ui::note("ok", &format!("arena iniciada: {}", game_name))
            );
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ram",
                    &format!(
                        "max_new_tokens auto = {} (RAM-disponible)",
                        self.max_new_tokens
                    )
                )
            );
        } else if let Some(tx) = &self.sender {
            let _ = tx.send(format!("=== INICIANDO ARENA DE JUEGO: {} ===", game_name));
            let _ = tx.send(format!(
                "[JUEZ] max_new_tokens auto = {} (RAM-disponible)",
                self.max_new_tokens
            ));
        }

        let out_w = unsafe {
            std::slice::from_raw_parts(
                self.output_weight,
                vocab_size * self.doppel_a.workspace.hidden_size,
            )
        };

        // Local, no-API text judge for DocumentDebate (uses the already-loaded emb table).
        let text_judge =
            TextJudge::new(embed_table, vocab_size, self.doppel_a.workspace.hidden_size);
        let verifiable = VerifiableJudge;

        let r_lose = std::env::var("MUD_DEBATE_RLOSE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.7);
        let jepa_lambda: f32 = std::env::var("MUD_DEBATE_JEPB_LAMBDA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.05);

        let mut match_count = 0usize;
        let mut turn_count = 0usize;

        // INFINITE MODE: keep playing matches until the user says stop (stop_flag /
        // Ctrl-C). Each finished game is scored by the verifiable judge; for
        // DocumentDebate we use the local TextJudge on the last pair of responses.
        loop {
            if self.is_time_up() {
                if print_stdout {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note("warn", "deteniendo arena (senal de parada)")
                    );
                }
                break;
            }

            if game.is_terminal() {
                // Score the finished match and start a fresh one (infinite survival).
                let (ra, rb) = self.score_match(
                    &*game,
                    game.name(),
                    &self.last_a_resp,
                    &self.last_b_resp,
                    &verifiable,
                    &text_judge,
                    game.winner(),
                    &self.tokenizer,
                );
                self.apply_learning(ra, rb, layers, shadow_layers, pool)?;
                if print_stdout {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note(
                            "stp",
                            &format!("match #{} -> A:{:+.2} B:{:+.2}", match_count + 1, ra, rb)
                        )
                    );
                }
                match_count += 1;
                game = game_factory();
                continue;
            }

            let prompt = game.get_state_prompt();
            let tokens = self.tokenizer.encode(&prompt);

            let is_alpha_turn = turn_count.is_multiple_of(2);
            // Progress hint so the TUI isn't silent during the (slow) CPU generation.
            if let Some(tx) = &self.sender {
                let who = if is_alpha_turn {
                    "Alpha (A)"
                } else {
                    "Beta (B)"
                };
                let _ = tx.send(format!("[thinking] {} generando respuesta...", who));
            }
            let (response, var_h, var_j) = if is_alpha_turn {
                Self::generate_agent_response(
                    &self.tokenizer,
                    &mut self.doppel_a,
                    &tokens,
                    self.max_new_tokens,
                    self.bos_id,
                    layers,
                    embed_table,
                    out_w,
                    self.output_norm_w,
                    vocab_size,
                )
            } else {
                Self::generate_agent_response(
                    &self.tokenizer,
                    &mut self.doppel_b,
                    &tokens,
                    self.max_new_tokens,
                    self.bos_id,
                    layers,
                    embed_table,
                    out_w,
                    self.output_norm_w,
                    vocab_size,
                )
            };

            let player = if is_alpha_turn { Player::A } else { Player::B };
            if player == Player::A {
                self.last_a_resp = response.clone();
            } else {
                self.last_b_resp = response.clone();
            }
            let msg = format!(
                "Player {}: {}",
                if is_alpha_turn { "A" } else { "B" },
                response
            );
            let stats_msg = format!("STATS|{:.4}|{:.4}", var_h, var_j);
            if print_stdout {
                println!("{}", msg);
            } else if let Some(tx) = &self.sender {
                let _ = tx.send(msg);
                let _ = tx.send(stats_msg);
            }

            // Penalty for aphasia / illegal moves; small positive baseline otherwise.
            let step_reward = match game.apply_move(player as usize, &response) {
                Ok(r) => r,
                Err(e) => {
                    if print_stdout {
                        println!("  [warn] move error: {}", e);
                    }
                    -0.5
                }
            };

            // DocumentDebate: local TextJudge override on the last pair of responses.
            let (mut reward_a, mut reward_b) = if player == Player::A {
                (step_reward, 0.0)
            } else {
                (0.0, step_reward)
            };
            if game.name() == "Document Debate" {
                let s_a =
                    text_judge.score_pair(&self.last_a_resp, &self.last_b_resp, &self.tokenizer);
                let s_b =
                    text_judge.score_pair(&self.last_b_resp, &self.last_a_resp, &self.tokenizer);
                if player == Player::A {
                    reward_a = s_a;
                } else {
                    reward_b = s_b;
                }
            }

            // JEPA intrinsic aux (anti-collapse), kept small like STP.
            let (jepa_a, jepa_b) = self.compute_jepa_reward(var_h, var_j, var_h, var_j);
            reward_a += jepa_lambda * jepa_a;
            reward_b += jepa_lambda * jepa_b;

            // Degenerate-play penalty: entropy collapse / dead activation.
            if var_h < 0.1 {
                if player == Player::A {
                    reward_a -= r_lose;
                } else {
                    reward_b -= r_lose;
                }
            }

            self.apply_learning(reward_a, reward_b, layers, shadow_layers, pool)?;

            // Anti-collapse neural kick if representation died.
            if var_h < 1e-3 {
                let cfg = crate::mud::self_play::SelfPlayConfig::default();
                for g in self.doppel_a.gradients.iter_mut() {
                    crate::mud::self_play::apply_gradient_jitter(g, &cfg, turn_count);
                }
                for g in self.doppel_b.gradients.iter_mut() {
                    crate::mud::self_play::apply_gradient_jitter(g, &cfg, turn_count);
                }
            }

            if !print_stdout {
                if let Some(tx) = &self.sender {
                    let _ = tx.send(format!("REWARD|A:{:.3}|B:{:.3}", reward_a, reward_b));
                }
            }

            turn_count += 1;
        }

        if print_stdout {
            println!(
                "{}",
                crate::mud::trainer_ui::note(
                    "ok",
                    &format!(
                        "arena detenida tras {} matches, {} turnos",
                        match_count, turn_count
                    )
                )
            );
        }
        Ok(())
    }

    /// Score a finished match from each player's POV using the verifiable judge,
    /// the local TextJudge for DocumentDebate, or ProfessorJudge for Professor-Student.
    #[allow(clippy::too_many_arguments)]
    fn score_match(
        &self,
        game: &dyn crate::mud::arena_games::ArenaGame,
        game_name: &str,
        last_a: &str,
        last_b: &str,
        verifiable: &VerifiableJudge,
        text_judge: &TextJudge,
        winner: Option<usize>,
        tokenizer: &crate::model::tokenizer::Tokenizer,
    ) -> (f32, f32) {
        if game_name == "Document Debate" {
            let s_a = text_judge.score_pair(last_a, last_b, tokenizer);
            let s_b = text_judge.score_pair(last_b, last_a, tokenizer);
            (s_a, s_b)
        } else if game_name == "Professor-Student" {
            // Student (B) gets the graded reward; professor (A) gets a small
            // mirror so it learns to pose+grade (RLVR both roles).
            if let Some((ex, ans, corr, _rub)) = game.professor_data() {
                let pj = ProfessorJudge::new(text_judge);
                let (rubrik, reward) = pj.grade(&ex, &ans, &corr, tokenizer);
                // Persist rubrik for telemetry (best-effort: log to stdout).
                if self.sender.is_none() {
                    println!(
                        "{}",
                        crate::mud::trainer_ui::note(
                            "stp",
                            &format!(
                                "rubrik gram={:.2} syn={:.2} coh={:.2} prag={:.2} -> R={:+.2}",
                                rubrik[0], rubrik[1], rubrik[2], rubrik[3], reward
                            )
                        )
                    );
                }
                (reward * 0.3, reward) // A=profesor (minor), B=alumno (full)
            } else {
                (0.0, 0.0)
            }
        } else {
            let ra = verifiable.score_dyn(winner, Player::A);
            let rb = verifiable.score_dyn(winner, Player::B);
            (ra, rb)
        }
    }
}
