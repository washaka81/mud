/// Micro-Intelligence LDT (Lattice-based Deduction Tree)
/// Phase 16: Zero-Latency Sub-2M Parameter Architecture.
/// Designed to fit entirely inside L3 Cache (4-8MB).
pub struct LdtMicroModel {
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Discrete constraint matrices guiding the continuous latent wave
    pub lattice_constraints: Vec<f32>,
    /// Weights mapped for Group Relative Policy Optimization (GRPO)
    /// Wrapped in Mutex for interior mutability — updated during inference without &mut self.
    pub policy_weights: parking_lot::Mutex<Vec<f32>>,
    /// PRIORITY 7: RLVR Critic
    pub rlvr_critic: crate::mud::rlvr::RlvrCritic,
    /// PRIORITY 17: MCTS Arena
    pub mcts_arena: parking_lot::Mutex<MctsArena>,
}

pub const MAX_MCTS_NODES: usize = 128;

#[derive(Clone)]
pub struct MctsNode {
    pub parent_idx: Option<usize>,
    pub children_indices: [usize; 8],
    pub num_children: usize,
    pub visits: usize,
    pub total_value: f32,
}

impl Default for MctsNode {
    fn default() -> Self {
        Self {
            parent_idx: None,
            children_indices: [0; 8],
            num_children: 0,
            visits: 0,
            total_value: 0.0,
        }
    }
}

pub struct MctsArena {
    pub nodes: [MctsNode; MAX_MCTS_NODES],
    pub states: Vec<crate::mud::workspace::UnifiedBuffer>,
    pub next_free: usize,
}

impl MctsArena {
    pub fn new(hidden_size: usize) -> Self {
        let mut states = Vec::with_capacity(MAX_MCTS_NODES);
        for _ in 0..MAX_MCTS_NODES {
            states.push(crate::mud::workspace::UnifiedBuffer::new_cpu(hidden_size));
        }
        Self {
            nodes: std::array::from_fn(|_| MctsNode::default()),
            states,
            next_free: 0,
        }
    }

    pub fn reset(&mut self) {
        for i in 0..self.next_free {
            self.nodes[i] = MctsNode::default();
        }
        self.next_free = 0;
    }

    pub fn add_node(&mut self, parent_idx: Option<usize>) -> Option<usize> {
        if self.next_free >= MAX_MCTS_NODES {
            return None; // Arena full
        }
        let idx = self.next_free;
        self.next_free += 1;
        self.nodes[idx].parent_idx = parent_idx;
        
        if let Some(p) = parent_idx {
            let parent = &mut self.nodes[p];
            if parent.num_children < 8 {
                parent.children_indices[parent.num_children] = idx;
                parent.num_children += 1;
            }
        }
        Some(idx)
    }
}

impl LdtMicroModel {
    /// Initialize a new sub-2M parameter LDT model.
    pub fn new(hidden_size: usize, num_layers: usize) -> Self {
        Self {
            hidden_size,
            num_layers,
            lattice_constraints: Vec::new(),
            policy_weights: parking_lot::Mutex::new(Vec::new()),
            rlvr_critic: crate::mud::rlvr::RlvrCritic::new(),
            mcts_arena: parking_lot::Mutex::new(MctsArena::new(hidden_size)),
        }
    }

    /// Run the latent wave through internal reflections (Slow Thinking)
    /// before collapsing into a vocabulary distribution.
    pub fn evaluate_latent_wave(&self, input_wave: &[f32], max_reflections: usize) -> Vec<f32> {
        let mut wave = input_wave.to_vec();
        for _ in 0..max_reflections {
            // Apply lattice constraints (LDT-01)
            self.apply_lattice_constraints(&mut wave);

            // Modulate the wave structurally with the learned GRPO policy baseline
            let pol_guard = self.policy_weights.lock();
            let p_len = pol_guard.len();
            if p_len > 0 {
                let mut policy_factor = 0.0;
                for p in pol_guard.iter() {
                    policy_factor += p;
                }
                policy_factor /= p_len as f32;
                
                // Add minor directional pull towards the optimal state (Slow Thinking drift)
                for w in wave.iter_mut() {
                    *w += policy_factor * 0.01;
                }
            }
        }
        wave
    }

    /// Project the continuous wave onto the discrete lattice matrix
    fn apply_lattice_constraints(&self, wave: &mut [f32]) {
        for w in wave.iter_mut() {
            // Simulated projection: snap to nearest discrete state (-1.0, 0.0, 1.0)
            // mimicking the 1.58b ternary boundary.
            if *w > 0.5 {
                *w = 1.0;
            } else if *w < -0.5 {
                *w = -1.0;
            } else {
                *w = 0.0;
            }
        }
    }

    /// Applies Group Relative Policy Optimization (GRPO) to a group of parallel latent trajectories.
    /// By calculating relative advantages internally within the group, we completely eliminate
    /// the need for a massive Critic model, making this viable for sub-2M parameter caches.
    /// STRICT ZERO-ALLOCATION POLICY: Uses pre-allocated UnifiedBuffers.
    pub fn grpo_latent_selection(
        &self,
        parallel_waves: &[crate::mud::workspace::UnifiedBuffer],
        reference_lattice: &crate::mud::workspace::UnifiedBuffer,
        out_wave: &crate::mud::workspace::UnifiedBuffer,
        active_len: usize,
    ) {
        let g = parallel_waves.len();
        if g == 0 {
            return;
        }

        let mut scores = [0.0f32; 16]; // Fixed static capacity to avoid Vec allocations (Max G=16)
        let ref_guard = reference_lattice.read();

        // 1. Evaluate each trajectory against the declarative constraints (Reward function)
        for (i, wave_buf) in parallel_waves.iter().enumerate() {
            if i >= 16 {
                break;
            }
            let wave_guard = wave_buf.read();
            scores[i] = self.evaluate_lattice_reward(&wave_guard, &ref_guard, active_len);
        }

        // 2. Compute mean and standard deviation of the group's scores
        let sum: f32 = scores.iter().take(g).sum();
        let mean = sum / (g as f32);

        let var_sum: f32 = scores.iter().take(g).map(|&s| (s - mean).powi(2)).sum();
        let variance = var_sum / (g as f32);
        let std_dev = (variance + 1e-8).sqrt(); // EPSILON_FLOOR for stability (Mandate)

        // 3. Calculate Relative Advantages & update policy weights (EMA of per-wave rewards)
        let mut best_idx = 0;
        let mut best_adv = f32::NEG_INFINITY;
        let ema_alpha = 0.1; // Smoothing factor for reward baseline

        let mut pol_guard = self.policy_weights.lock();
        if pol_guard.len() < g {
            pol_guard.resize(g, 0.0);
        }

        for (i, &score) in scores.iter().enumerate().take(g) {
            let baseline = pol_guard[i];
            let adv = (score - mean) / std_dev;
            // Update EMA baseline: new = (1-α)*old + α*score
            pol_guard[i] = (1.0 - ema_alpha) * baseline + ema_alpha * score;

            if adv > best_adv {
                best_adv = adv;
                best_idx = i;
            }
        }

        // 4. Return the winning latent wave directly into the output UnifiedBuffer (Copia-Cero/Zero-Allocation)
        // We only copy the active_len to avoid overwriting padding if not needed,
        // but it's safer and faster to copy the entire buffer structure, or just the active part.
        let winning_guard = parallel_waves[best_idx].read();
        let mut out_guard = out_wave.write();
        let copy_len = active_len.min(winning_guard.len()).min(out_guard.len());
        out_guard[..copy_len].copy_from_slice(&winning_guard[..copy_len]);
    }

    /// Calculates how well a latent wave adheres to the desired algebraic lattice.
    /// Higher reward = better structural alignment.
    fn evaluate_lattice_reward(&self, wave: &[f32], reference: &[f32], active_len: usize) -> f32 {
        let mut reward = 0.0;
        let n = wave.len().min(reference.len()).min(active_len);
        for i in 0..n {
            // Negative MSE serves as the reward signal.
            // We penalize deviation from the strict deterministic lattice matrix.
            let diff = wave[i] - reference[i];
            reward -= diff * diff;
        }
        reward
    }

    /// PRIORITY 17: Interactive MCTS (Monte Carlo Tree Search)
    /// Escalar dinámicamente el Test-Time Compute integrado con el LDT.
    /// Realiza bifurcaciones ("Slow Thinking") explorando trayectorias sobre la arena preasignada.
    pub fn run_mcts_search(
        &self,
        root_wave: &crate::mud::workspace::UnifiedBuffer,
        reference_lattice: &crate::mud::workspace::UnifiedBuffer,
        active_len: usize,
        search_budget: usize,
    ) {
        let mut arena = self.mcts_arena.lock();
        arena.reset();

        // Inicializar la raíz
        let root_idx = arena.add_node(None).unwrap();
        {
            let src = root_wave.read();
            let mut dst = arena.states[root_idx].write();
            let n = active_len.min(src.len()).min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
        }

        for _ in 0..search_budget {
            // 1. Selection (UCT - Upper Confidence Bound applied to Trees)
            let mut current = root_idx;
            while arena.nodes[current].num_children > 0 && arena.nodes[current].num_children == 8 {
                // Select child with highest UCB1 score
                let mut best_score = f32::NEG_INFINITY;
                let mut best_child = current;
                let parent_visits = (arena.nodes[current].visits as f32).max(1.0);
                
                for i in 0..arena.nodes[current].num_children {
                    let child_idx = arena.nodes[current].children_indices[i];
                    let child = &arena.nodes[child_idx];
                    let exploitation = if child.visits > 0 { child.total_value / child.visits as f32 } else { 0.0 };
                    let exploration = 1.414 * ((parent_visits.ln()) / (child.visits as f32).max(1.0)).sqrt();
                    let ucb1 = exploitation + exploration;
                    
                    if ucb1 > best_score {
                        best_score = ucb1;
                        best_child = child_idx;
                    }
                }
                current = best_child;
            }

            // 2. Expansion
            if arena.nodes[current].num_children < 8 {
                if let Some(new_child) = arena.add_node(Some(current)) {
                    // Copiar estado del padre
                    {
                        let src = arena.states[current].read();
                        let mut dst = arena.states[new_child].write();
                        let n = active_len.min(src.len()).min(dst.len());
                        dst[..n].copy_from_slice(&src[..n]);
                    }
                    
                    // Simular un paso LDT (Reflection step)
                    {
                        let mut wave_guard = arena.states[new_child].write();
                        self.apply_lattice_constraints(&mut wave_guard);
                        
                        // Jitter para bifurcación (Neural Kick)
                        for w in wave_guard.iter_mut().take(active_len) {
                            *w += (rand::random::<f32>() - 0.5) * 0.05; // pequeña perturbación de rama
                        }
                    }
                    current = new_child;
                } else {
                    break; // Arena full
                }
            }

            // 3. Rollout / Evaluation (Reward)
            let reward = {
                let wave_guard = arena.states[current].read();
                let ref_guard = reference_lattice.read();
                self.evaluate_lattice_reward(&wave_guard, &ref_guard, active_len)
            };

            // 4. Backpropagation
            let mut backprop_node = Some(current);
            while let Some(idx) = backprop_node {
                arena.nodes[idx].visits += 1;
                arena.nodes[idx].total_value += reward;
                backprop_node = arena.nodes[idx].parent_idx;
            }
        }

        // Selección final (mejor valor medio o más visitado)
        let mut best_child = root_idx;
        let mut most_visits = 0;
        for i in 0..arena.nodes[root_idx].num_children {
            let child_idx = arena.nodes[root_idx].children_indices[i];
            if arena.nodes[child_idx].visits > most_visits {
                most_visits = arena.nodes[child_idx].visits;
                best_child = child_idx;
            }
        }

        // Sobrescribir el buffer original con el estado ganador
        {
            let src = arena.states[best_child].read();
            let mut dst = root_wave.write();
            let n = active_len.min(src.len()).min(dst.len());
            dst[..n].copy_from_slice(&src[..n]);
        }
    }
}
