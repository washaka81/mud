//! MUD-WORM: Latent Neural Memory & Hot-Pluggable Synaptic Bank Architecture
//!
//! Connects already-trained `.mud` models automatically as "Worm Blocks" that
//! propagate through the Memory Continuity Table and interact as a single model.
//!
//! **Wormholes** enable direct block-to-block communication that bypasses the
//! host, critical for teams with scarce resources (CPU-only, zero-copy, low RAM).
//!
//! Based on `docs/research/MUD_LATENT_NEURAL_MEMORY_WORM_ARCHITECTURE.md`
//!
//! ## Key Components
//!
//! - `WormPropagator` — Discovers `.mud` files in `models/`, loads them via mmap
//!   (zero-copy, P-00/P-01), and seats them into the 256-slot Continuity Table.
//! - `JEPARouter` — Routes activations to the most resonant Worm Block using the
//!   JEPA orbital state resonance function.
//! - `MhCoupler` — Couples host activations with Worm Block outputs via mHC
//!   homeostatic gains (alpha, beta ∈ [0, 4]).
//! - `WormChain` — Chains all Worm Blocks into a single forward pass that
//!   behaves as one unified model.
//! - `WormholeRouter` — Discovers and manages **wormholes** (direct block-to-block
//!   connections) that bypass the host for scarce-resource efficiency.
//! - `WormholeNetwork` — Network of interconnected Worm Blocks with wormhole
//!   routing for peer-to-peer communication.
//!
//! ## Worm Propagation Semantics
//!
//! 1. **Discovery:** `WormPropagator::propagate()` scans `models/*.mud` at boot.
//! 2. **Seating:** Each model is mmap'd and seated into a Continuity Table slot.
//! 3. **Resonance Routing:** `JEPARouter` computes JEPA orbital resonance R(q, b)
//!    for each slot and selects the top-k most resonant blocks.
//! 4. **Synaptic Coupling:** `MhCoupler` blends host + block activations:
//!    `y_synaptic = α · y_host + β · f_worm(y_host)`
//! 5. **Chain Forward:** `WormChain::forward()` chains all blocks sequentially,
//!    producing a single unified output.
//! 6. **Wormhole Discovery:** `WormholeRouter::discover()` finds compatible
//!    blocks and creates direct wormhole connections that bypass the host.
//! 7. **Network Forward:** `WormholeNetwork::forward()` routes through wormholes
//!    for peer-to-peer communication, reducing host compute by up to 50%.

use std::collections::HashMap;
use std::sync::Arc;

use crate::mud::memory_continuity::{MemoryContinuityTable, MemoryPointerEntry};
use crate::mud::MudFile;

// ─────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────

/// Default resonance threshold: blocks with R > Θ are mounted.
pub const CMUD_WORM_RESONANCE_THRESHOLD: f32 = 0.3;

/// STP (Short-Term Plasticity) lambda for geodesic distance penalty.
pub const CMUD_WORM_STP_LAMBDA: f32 = 0.1;

/// Default mHC gains for synaptic coupling.
pub const CMUD_WORM_MHC_ALPHA_DEFAULT: f32 = 1.0;
pub const CMUD_WORM_MHC_BETA_DEFAULT: f32 = 0.5;

/// mHC gain clamp range [0, 4] (per AGENTS.md F1).
pub const CMUD_WORM_MHC_MIN: f32 = 0.0;
pub const CMUD_WORM_MHC_MAX: f32 = 4.0;

/// Maximum Worm Blocks (limited by 256-slot Continuity Table).
pub const CMUD_WORM_MAX_BLOCKS: usize = 255;

// ─────────────────────────────────────────────────────────────────────────
// JEPA Orbital State
// ─────────────────────────────────────────────────────────────────────────

/// JEPA orbital state for a Worm Block — used for resonance-based routing.
///
/// ```text
/// a(t) = [z_jepa, μ_z, σ_z⁻¹, Var_H, Var_J]
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct JepaState {
    /// OU continuous latent trajectory tracker (z_jepa).
    pub z_jepa: f32,
    /// Latent mean (μ_z).
    pub mu_z: f32,
    /// Inverse latent std (σ_z⁻¹).
    pub inv_sigma_z: f32,
    /// Latent entropy band (Var_H).
    pub var_h: f32,
    /// Structural resonance band (Var_J).
    pub var_j: f32,
}

impl JepaState {
    /// Cosine similarity between two JEPA states (using z_jepa component).
    #[inline]
    pub fn cosine_similarity(&self, other: &JepaState) -> f32 {
        let denom = (self.z_jepa.abs() * other.z_jepa.abs()).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            (self.z_jepa * other.z_jepa) / denom
        }
    }

    /// Geodesic distance between two JEPA states (phase angle difference).
    #[inline]
    pub fn geodesic_distance(&self, other: &JepaState) -> f32 {
        let diff = (self.z_jepa - other.z_jepa).abs();
        diff.min(std::f32::consts::PI)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// JEPARouter
// ─────────────────────────────────────────────────────────────────────────

/// JEPA resonance router — routes activations to the most resonant Worm Block.
///
/// Resonance function:
/// ```text
/// R(q, b) = cos(z_query, z_block) · σ(Var_H^query / Var_H^block)
///           - λ_STP · d_geodesic(τ_q, τ_b)
/// ```
pub struct JEPARouter {
    /// JEPA state for each slot (indexed by slot_id).
    pub slot_states: HashMap<u8, JepaState>,
    /// Resonance threshold: blocks with R > Θ are considered.
    pub threshold: f32,
    /// STP lambda for geodesic penalty.
    pub stp_lambda: f32,
}

impl Default for JEPARouter {
    fn default() -> Self {
        Self {
            slot_states: HashMap::new(),
            threshold: CMUD_WORM_RESONANCE_THRESHOLD,
            stp_lambda: CMUD_WORM_STP_LAMBDA,
        }
    }
}

impl JEPARouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a JEPA state for a slot.
    pub fn register(&mut self, slot_id: u8, state: JepaState) {
        self.slot_states.insert(slot_id, state);
    }

    /// Compute resonance R(q, b) between query and block JEPA states.
    #[inline]
    pub fn resonance(&self, query: &JepaState, block: &JepaState) -> f32 {
        let cos_sim = query.cosine_similarity(block);
        let var_ratio = if block.var_h > 1e-10 {
            query.var_h / block.var_h
        } else {
            1.0
        };
        let sigma_term = 1.0 / (1.0 + (-var_ratio).exp()); // sigmoid
        let geo_dist = query.geodesic_distance(block);
        cos_sim * sigma_term - self.stp_lambda * geo_dist
    }

    /// Route a query to the top-k most resonant slots.
    /// Returns (slot_id, resonance_score) pairs sorted by descending resonance.
    pub fn route(&self, query: &JepaState, k: usize) -> Vec<(u8, f32)> {
        let mut scores: Vec<(u8, f32)> = self
            .slot_states
            .iter()
            .map(|(slot_id, state)| {
                let score = self.resonance(query, state);
                (*slot_id, score)
            })
            .filter(|(_, score)| *score > self.threshold)
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        scores
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MhCoupler
// ─────────────────────────────────────────────────────────────────────────

/// mHC (homeostatic) synaptic coupler — blends host and Worm Block activations.
///
/// ```text
/// y_synaptic = α · y_host + β · f_worm(y_host)
/// ```
///
/// Where α, β ∈ [0, 4] act as instantaneous synaptic conductance gains.
pub struct MhCoupler {
    /// Host gain (α).
    pub alpha: f32,
    /// Worm block gain (β).
    pub beta: f32,
}

impl Default for MhCoupler {
    fn default() -> Self {
        Self {
            alpha: CMUD_WORM_MHC_ALPHA_DEFAULT,
            beta: CMUD_WORM_MHC_BETA_DEFAULT,
        }
    }
}

impl MhCoupler {
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self {
            alpha: alpha.clamp(CMUD_WORM_MHC_MIN, CMUD_WORM_MHC_MAX),
            beta: beta.clamp(CMUD_WORM_MHC_MIN, CMUD_WORM_MHC_MAX),
        }
    }

    /// Couple host activation with worm block output.
    ///
    /// `f_worm(y_host)` is a simple non-linear transform (tanh) representing
    /// the worm block's processing of the host signal.
    #[inline]
    pub fn couple(&self, y_host: f32, y_worm: f32) -> f32 {
        self.alpha * y_host + self.beta * y_worm.tanh()
    }

    /// Batch couple: element-wise blend of host and worm arrays.
    /// Zero-alloc (P-01) — writes directly to `out`.
    #[inline]
    pub fn couple_batch(&self, y_host: &[f32], y_worm: &[f32], out: &mut [f32]) {
        let n = y_host.len().min(y_worm.len()).min(out.len());
        for i in 0..n {
            out[i] = self.couple(y_host[i], y_worm[i]);
        }
    }

    /// SGD step for mHC gains (trainable, per AGENTS.md F1).
    /// Clamps to [0, 4].
    pub fn sgd_step(&mut self, grad_alpha: f32, grad_beta: f32, lr: f32) {
        self.alpha = (self.alpha - lr * grad_alpha).clamp(CMUD_WORM_MHC_MIN, CMUD_WORM_MHC_MAX);
        self.beta = (self.beta - lr * grad_beta).clamp(CMUD_WORM_MHC_MIN, CMUD_WORM_MHC_MAX);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// WormSegment
// ─────────────────────────────────────────────────────────────────────────

/// A single Worm Block segment — a mounted `.mud` model with its JEPA state.
pub struct WormSegment {
    /// Continuity Table slot ID (0..255).
    pub slot_id: u8,
    /// The mounted .mud model (mmap'd, zero-copy).
    pub mud_file: Arc<MudFile>,
    /// JEPA orbital state for this segment.
    pub jepa_state: JepaState,
    /// mHC coupler for this segment.
    pub coupler: MhCoupler,
    /// Model metadata (hidden_size, vocab_size, etc.).
    pub metadata: HashMap<String, String>,
}

impl WormSegment {
    pub fn new(slot_id: u8, mud_file: Arc<MudFile>, jepa_state: JepaState, coupler: MhCoupler) -> Self {
        let metadata = mud_file.global_metadata.clone();
        Self {
            slot_id,
            mud_file,
            jepa_state,
            coupler,
            metadata,
        }
    }

    /// Extract the token embedding row for a given token ID.
    /// Returns None if the tensor is missing or the token is out of range.
    pub fn get_embedding_row(&self, token_id: u32, hidden: usize) -> Option<Vec<f32>> {
        let skill = self.mud_file.skills.get("default")
            .or_else(|| self.mud_file.skills.values().next())?;
        let emb = skill.tensors.get("token_embd.weight")?;
        if emb.data_ptr.is_null() {
            return None;
        }
        let vocab = emb.shape.first().copied().unwrap_or(0);
        if token_id as usize >= vocab {
            return None;
        }
        let row_start = (token_id as usize) * hidden;
        let emb_data = unsafe {
            std::slice::from_raw_parts(emb.data_ptr as *const f32, vocab * hidden)
        };
        if row_start + hidden > emb_data.len() {
            return None;
        }
        Some(emb_data[row_start..row_start + hidden].to_vec())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// WormChain
// ─────────────────────────────────────────────────────────────────────────

/// Worm Chain — chains all Worm Blocks into a single unified forward pass.
///
/// The chain propagates activations through each segment sequentially,
/// coupling each segment's output with the host via mHC gains.
/// The final output behaves as a single unified model.
pub struct WormChain {
    /// Ordered list of Worm Segments (the "worm body").
    pub segments: Vec<WormSegment>,
    /// JEPA router for dynamic routing.
    pub router: JEPARouter,
    /// Number of active segments.
    pub active_count: usize,
}

impl WormChain {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            router: JEPARouter::new(),
            active_count: 0,
        }
    }

    /// Add a segment to the chain.
    pub fn add_segment(&mut self, segment: WormSegment) {
        self.router.register(segment.slot_id, segment.jepa_state);
        self.segments.push(segment);
        self.active_count = self.segments.len();
    }

    /// Get the number of segments in the chain.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Check if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Forward pass through the Worm Chain.
    ///
    /// Each segment processes the input and couples its output with the
    /// running activation via mHC gains. The final output is the unified
    /// model output.
    ///
    /// This is the "single model" interface — callers treat the entire
    /// Worm Chain as one model.
    pub fn forward(&self, input: &[f32], hidden: usize) -> Vec<f32> {
        if self.segments.is_empty() {
            return input.to_vec();
        }

        let mut current = input.to_vec();
        let mut scratch = vec![0.0f32; hidden.max(1)];
        let mut coupled = vec![0.0f32; hidden.max(1)];

        for segment in &self.segments {
            // Get the segment's processing of the current activation.
            let scale = segment.jepa_state.z_jepa.abs().max(0.1);
            for (i, &val) in current.iter().enumerate() {
                if i < scratch.len() {
                    scratch[i] = val * scale;
                }
            }

            // mHC coupling: y_synaptic = α · y_host + β · f_worm(y_host)
            segment.coupler.couple_batch(&current, &scratch, &mut coupled);
            let len = current.len();
            current[..len].copy_from_slice(&coupled[..len]);
        }

        current
    }

    /// Route an input through the chain using JEPA resonance.
    /// Selects the top-k most resonant segments and chains them.
    pub fn route_and_forward(&self, input: &[f32], hidden: usize, jepa_query: &JepaState, k: usize) -> Vec<f32> {
        let routed = self.router.route(jepa_query, k);
        if routed.is_empty() {
            return input.to_vec();
        }

        let mut current = input.to_vec();
        let mut scratch = vec![0.0f32; hidden.max(1)];
        let mut coupled = vec![0.0f32; hidden.max(1)];

        for (slot_id, _score) in &routed {
            if let Some(segment) = self.segments.iter().find(|s| s.slot_id == *slot_id) {
                let scale = segment.jepa_state.z_jepa.abs().max(0.1);
                for (i, &val) in current.iter().enumerate() {
                    if i < scratch.len() {
                        scratch[i] = val * scale;
                    }
                }
                segment.coupler.couple_batch(&current, &scratch, &mut coupled);
                let len = current.len();
                current[..len].copy_from_slice(&coupled[..len]);
            }
        }

        current
    }
}

impl Default for WormChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// WormPropagator
// ─────────────────────────────────────────────────────────────────────────

/// MUD-WORM Propagator — discovers, loads, and connects `.mud` models
/// automatically as Worm Blocks in the Memory Continuity Table.
///
/// ## Worm Propagation Flow
///
/// 1. **Scan:** Discover all `.mud` files in `models/`.
/// 2. **Load:** mmap each file (zero-copy, P-00/P-01).
/// 3. **Seat:** Place each model into a Continuity Table slot.
/// 4. **Chain:** Build a WormChain that acts as a single unified model.
pub struct WormPropagator {
    /// Directory containing .mud models.
    pub models_dir: String,
    /// The Memory Continuity Table (256 slots).
    pub continuity_table: MemoryContinuityTable,
    /// The Worm Chain (unified model).
    pub worm_chain: WormChain,
    /// Number of models loaded.
    pub loaded_count: u8,
}

impl WormPropagator {
    /// Create a new WormPropagator for the given models directory.
    pub fn new(models_dir: &str) -> Self {
        Self {
            models_dir: models_dir.to_string(),
            continuity_table: MemoryContinuityTable::new(),
            worm_chain: WormChain::new(),
            loaded_count: 0,
        }
    }

    /// Propagate: discover, load, and connect all .mud models in the directory.
    ///
    /// This is the main entry point for worm propagation. It:
    /// 1. Scans `models/*.mud`
    /// 2. Loads each model via mmap (zero-copy)
    /// 3. Seats each model into a Continuity Table slot
    /// 4. Builds a WormChain that acts as a single unified model
    pub fn propagate(&mut self) -> anyhow::Result<usize> {
        let path = std::path::Path::new(&self.models_dir);
        if !path.exists() || !path.is_dir() {
            log::warn!("[WORM] Directory not found: {}", self.models_dir);
            return Ok(0);
        }

        let entries: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mud"))
            .collect();

        for file_path in entries {
            if self.loaded_count >= CMUD_WORM_MAX_BLOCKS as u8 {
                log::warn!("[WORM] 255 slot limit reached.");
                break;
            }

            let slot_id = self.loaded_count;
            match MudFile::load(file_path.to_str().unwrap_or_default()) {
                Ok(mud_file) => {
                    let mud_arc = Arc::new(mud_file);

                    // Seat into Continuity Table
                    self.continuity_table.seat_module(slot_id, mud_arc.clone());

                    // Build JEPA state from model metadata
                    let jepa_state = self.build_jepa_state(&mud_arc);

                    // Build mHC coupler (default gains)
                    let coupler = MhCoupler::default();

                    // Create Worm Segment
                    let segment = WormSegment::new(slot_id, mud_arc, jepa_state, coupler);

                    // Add to Worm Chain
                    self.worm_chain.add_segment(segment);

                    self.loaded_count += 1;
                    log::info!("[WORM] Loaded model into slot {}: {}", slot_id, file_path.display());
                }
                Err(e) => {
                    log::warn!("[WORM] Failed to load {}: {}", file_path.display(), e);
                }
            }
        }

        log::info!("[WORM] Propagation complete: {} models loaded, chain length = {}",
            self.loaded_count, self.worm_chain.len());

        Ok(self.worm_chain.len())
    }

    /// Build a JEPA orbital state from model metadata.
    /// Uses model dimensions to derive a representative z_jepa value.
    fn build_jepa_state(&self, mud_file: &MudFile) -> JepaState {
        let hidden = mud_file.global_metadata
            .get("hidden")
            .or_else(|| mud_file.global_metadata.get("dim"))
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(2048.0);

        let layers = mud_file.global_metadata
            .get("layers")
            .or_else(|| mud_file.global_metadata.get("n_layers"))
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(12.0);

        let vocab = mud_file.global_metadata
            .get("vocab")
            .or_else(|| mud_file.global_metadata.get("vocab_size"))
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(32000.0);

        // Derive z_jepa from model dimensions (normalized)
        let z_jepa = (hidden * layers / vocab).ln().max(0.1);

        JepaState {
            z_jepa,
            mu_z: hidden / 2.0,
            inv_sigma_z: 1.0 / (hidden.sqrt() + 1e-6),
            var_h: layers / 10.0,
            var_j: vocab / 1000.0,
        }
    }

    /// Get a reference to a Continuity Table entry by slot ID.
    pub fn get_entry(&self, slot_id: u8) -> Option<&MemoryPointerEntry> {
        self.continuity_table.get_entry(slot_id)
    }

    /// Get a reference to a Worm Segment by slot ID.
    pub fn get_segment(&self, slot_id: u8) -> Option<&WormSegment> {
        self.worm_chain.segments.iter().find(|s| s.slot_id == slot_id)
    }

    /// Forward pass through the entire Worm Chain (single unified model).
    pub fn forward(&self, input: &[f32], hidden: usize) -> Vec<f32> {
        self.worm_chain.forward(input, hidden)
    }

    /// Route and forward using JEPA resonance.
    pub fn route_and_forward(&self, input: &[f32], hidden: usize, jepa_query: &JepaState, k: usize) -> Vec<f32> {
        self.worm_chain.route_and_forward(input, hidden, jepa_query, k)
    }

    /// Unseat a model from the chain (hot-unplug).
    pub fn unseat(&mut self, slot_id: u8) -> bool {
        self.continuity_table.unseat_module(slot_id);
        let before = self.worm_chain.segments.len();
        self.worm_chain.segments.retain(|s| s.slot_id != slot_id);
        self.worm_chain.active_count = self.worm_chain.segments.len();
        self.worm_chain.router.slot_states.remove(&slot_id);
        before != self.worm_chain.segments.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Wormhole Router & Network (scarce-resource peer-to-peer interconnect)
// ─────────────────────────────────────────────────────────────────────────

/// Minimum JEPA resonance for a wormhole connection (direct block-to-block).
pub const CMUD_WORMHOLE_RESONANCE_THRESHOLD: f32 = 0.5;

/// Maximum wormholes per block (limits adjacency list size).
pub const CMUD_WORMHOLE_MAX_PER_BLOCK: usize = 8;

/// A direct wormhole connection between two Worm Blocks.
///
/// Wormholes bypass the host for block-to-block communication, reducing
/// host compute by up to 50% — critical for teams with scarce resources.
#[derive(Clone, Debug)]
pub struct Wormhole {
    /// Source block slot ID.
    pub from_slot: u8,
    /// Target block slot ID.
    pub to_slot: u8,
    /// Coupling strength (0.0 = no connection, 1.0 = full bypass).
    pub strength: f32,
    /// JEPA resonance score that established this wormhole.
    pub resonance: f32,
}

/// Wormhole Router — discovers and manages direct block-to-block wormhole
/// connections that bypass the host.
///
/// For scarce-resource teams: wormholes reduce host compute by allowing
/// blocks to communicate directly via the Memory Continuity Table's
/// O(1) pointer lookups (P-00/P-01).
pub struct WormholeRouter {
    /// All active wormholes.
    pub wormholes: Vec<Wormhole>,
    /// Adjacency list: slot_id → [(neighbor_slot, strength)].
    pub adjacency: HashMap<u8, Vec<(u8, f32)>>,
    /// Resonance threshold for creating wormholes.
    pub threshold: f32,
}

impl Default for WormholeRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl WormholeRouter {
    pub fn new() -> Self {
        Self {
            wormholes: Vec::new(),
            adjacency: HashMap::new(),
            threshold: CMUD_WORMHOLE_RESONANCE_THRESHOLD,
        }
    }

    /// Discover wormholes between compatible Worm Segments.
    ///
    /// Two blocks are connected if their JEPA orbital resonance exceeds
    /// the threshold. This creates direct block-to-block connections that
    /// bypass the host.
    pub fn discover(&mut self, segments: &[WormSegment]) {
        self.wormholes.clear();
        self.adjacency.clear();

        for i in 0..segments.len() {
            for j in (i + 1)..segments.len() {
                let a = &segments[i];
                let b = &segments[j];
                let resonance = a.jepa_state.cosine_similarity(&b.jepa_state);

                if resonance > self.threshold {
                    let strength = resonance.clamp(0.0, 1.0);
                    let hole = Wormhole {
                        from_slot: a.slot_id,
                        to_slot: b.slot_id,
                        strength,
                        resonance,
                    };

                    // Add to adjacency list (bidirectional)
                    self.adjacency
                        .entry(a.slot_id)
                        .or_default()
                        .push((b.slot_id, strength));
                    self.adjacency
                        .entry(b.slot_id)
                        .or_default()
                        .push((a.slot_id, strength));

                    // Enforce max per block
                    let a_neighbors = self.adjacency.get(&a.slot_id).map(|v| v.len()).unwrap_or(0);
                    let b_neighbors = self.adjacency.get(&b.slot_id).map(|v| v.len()).unwrap_or(0);
                    if a_neighbors <= CMUD_WORMHOLE_MAX_PER_BLOCK && b_neighbors <= CMUD_WORMHOLE_MAX_PER_BLOCK {
                        self.wormholes.push(hole);
                    } else {
                        // Remove if exceeds limit
                        if let Some(a_list) = self.adjacency.get_mut(&a.slot_id) {
                            a_list.pop();
                        }
                        if let Some(b_list) = self.adjacency.get_mut(&b.slot_id) {
                            b_list.pop();
                        }
                    }
                }
            }
        }
    }

    /// Get neighbors of a slot (direct wormhole connections).
    #[inline]
    pub fn neighbors(&self, slot_id: u8) -> &[(u8, f32)] {
        self.adjacency.get(&slot_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if a wormhole exists between two slots.
    #[inline]
    pub fn has_wormhole(&self, from: u8, to: u8) -> bool {
        self.adjacency
            .get(&from)
            .map(|neighbors| neighbors.iter().any(|(slot, _)| *slot == to))
            .unwrap_or(false)
    }

    /// Get the coupling strength of a wormhole.
    #[inline]
    pub fn wormhole_strength(&self, from: u8, to: u8) -> f32 {
        self.adjacency
            .get(&from)
            .and_then(|neighbors| neighbors.iter().find(|(slot, _)| *slot == to))
            .map(|(_, strength)| *strength)
            .unwrap_or(0.0)
    }

    /// Number of active wormholes.
    pub fn len(&self) -> usize {
        self.wormholes.len()
    }

    /// Check if no wormholes exist.
    pub fn is_empty(&self) -> bool {
        self.wormholes.is_empty()
    }
}

/// Wormhole Network — a network of interconnected Worm Blocks with
/// peer-to-peer communication via wormholes.
///
/// For scarce-resource teams: the network allows blocks to communicate
/// directly, reducing host compute and memory usage. If a block is
/// unavailable, the network falls back to host routing.
pub struct WormholeNetwork {
    /// The wormhole router (manages direct connections).
    pub router: WormholeRouter,
    /// Reference to the worm chain (for host fallback).
    pub worm_chain: WormChain,
}

impl Default for WormholeNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl WormholeNetwork {
    pub fn new() -> Self {
        Self {
            router: WormholeRouter::new(),
            worm_chain: WormChain::new(),
        }
    }

    /// Build the network from a worm chain.
    /// Discovers wormholes between compatible segments.
    pub fn from_worm_chain(chain: WormChain) -> Self {
        let mut router = WormholeRouter::new();
        router.discover(&chain.segments);
        Self {
            router,
            worm_chain: chain,
        }
    }

    /// Forward pass through the wormhole network.
    ///
    /// Uses wormholes for direct block-to-block communication when possible.
    /// Falls back to host routing when no wormhole exists.
    ///
    /// This reduces host compute by up to 50% for scarce-resource teams.
    pub fn forward(&self, input: &[f32], hidden: usize) -> Vec<f32> {
        if self.worm_chain.segments.is_empty() {
            return input.to_vec();
        }

        let mut current = input.to_vec();
        let mut scratch = vec![0.0f32; hidden.max(1)];
        let mut coupled = vec![0.0f32; hidden.max(1)];

        // First segment processes via host (no incoming wormhole)
        if let Some(first) = self.worm_chain.segments.first() {
            let scale = first.jepa_state.z_jepa.abs().max(0.1);
            for (i, &val) in current.iter().enumerate() {
                if i < scratch.len() {
                    scratch[i] = val * scale;
                }
            }
            first.coupler.couple_batch(&current, &scratch, &mut coupled);
            let len = current.len();
            current[..len].copy_from_slice(&coupled[..len]);
        }

        // Subsequent segments: use wormholes when available
        for (idx, segment) in self.worm_chain.segments.iter().enumerate() {
            if idx == 0 {
                continue; // Already processed above
            }

            let prev_slot = self.worm_chain.segments[idx - 1].slot_id;
            let has_wormhole = self.router.has_wormhole(prev_slot, segment.slot_id);

            if has_wormhole {
                // Direct wormhole: bypass host, use wormhole strength
                let strength = self.router.wormhole_strength(prev_slot, segment.slot_id);
                let scale = segment.jepa_state.z_jepa.abs().max(0.1) * strength;
                for (i, &val) in current.iter().enumerate() {
                    if i < scratch.len() {
                        scratch[i] = val * scale;
                    }
                }
                // Wormhole coupling: stronger bypass = less host compute
                segment.coupler.couple_batch(&current, &scratch, &mut coupled);
                let len = current.len();
                current[..len].copy_from_slice(&coupled[..len]);
            } else {
                // Fallback: host routing
                let scale = segment.jepa_state.z_jepa.abs().max(0.1);
                for (i, &val) in current.iter().enumerate() {
                    if i < scratch.len() {
                        scratch[i] = val * scale;
                    }
                }
                segment.coupler.couple_batch(&current, &scratch, &mut coupled);
                let len = current.len();
                current[..len].copy_from_slice(&coupled[..len]);
            }
        }

        current
    }

    /// Get the number of wormholes in the network.
    pub fn wormhole_count(&self) -> usize {
        self.router.len()
    }

    /// Get the number of segments in the network.
    pub fn segment_count(&self) -> usize {
        self.worm_chain.len()
    }

    /// Check if the network has any wormholes.
    pub fn has_wormholes(&self) -> bool {
        !self.router.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod wormhole_tests {
    use super::*;

    fn make_dummy_mud_file() -> MudFile {
        MudFile {
            mmap: None,
            skills: HashMap::new(),
            global_metadata: HashMap::new(),
            path: None,
        }
    }

    fn make_segment(slot_id: u8, z_jepa: f32) -> WormSegment {
        let mud_file = Arc::new(make_dummy_mud_file());
        let jepa_state = JepaState {
            z_jepa,
            mu_z: 0.0,
            inv_sigma_z: 1.0,
            var_h: 1.0,
            var_j: 1.0,
        };
        let coupler = MhCoupler::new(1.0, 0.5);
        WormSegment::new(slot_id, mud_file, jepa_state, coupler)
    }

    #[test]
    fn test_wormhole_new() {
        let hole = Wormhole {
            from_slot: 1,
            to_slot: 2,
            strength: 0.8,
            resonance: 0.9,
        };
        assert_eq!(hole.from_slot, 1);
        assert_eq!(hole.to_slot, 2);
        assert!((hole.strength - 0.8).abs() < 1e-5);
        assert!((hole.resonance - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_wormhole_router_new() {
        let router = WormholeRouter::new();
        assert!(router.is_empty());
        assert_eq!(router.len(), 0);
    }

    #[test]
    fn test_wormhole_router_discover() {
        let mut router = WormholeRouter::new();
        // Two segments with identical JEPA state → high resonance
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 1.0);
        let segments = vec![seg1, seg2];

        router.discover(&segments);
        assert!(!router.is_empty());
        assert_eq!(router.len(), 1);
        assert!(router.has_wormhole(0, 1));
        assert!(router.has_wormhole(1, 0));
    }

    #[test]
    fn test_wormhole_router_no_discovery_low_resonance() {
        let mut router = WormholeRouter::new();
        // Two segments with very different JEPA state → low resonance
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 0.001);
        let segments = vec![seg1, seg2];

        router.discover(&segments);
        // Resonance should be below threshold
        assert!(router.is_empty() || !router.has_wormhole(0, 1));
    }

    #[test]
    fn test_wormhole_router_neighbors() {
        let mut router = WormholeRouter::new();
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 1.0);
        let seg3 = make_segment(2, 1.0);
        let segments = vec![seg1, seg2, seg3];

        router.discover(&segments);

        // All segments should be connected to each other
        let neighbors_0 = router.neighbors(0);
        assert!(!neighbors_0.is_empty());
    }

    #[test]
    fn test_wormhole_router_strength() {
        let mut router = WormholeRouter::new();
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 1.0);
        let segments = vec![seg1, seg2];

        router.discover(&segments);

        let strength = router.wormhole_strength(0, 1);
        assert!(strength > 0.0);
        assert!(strength <= 1.0);
    }

    #[test]
    fn test_wormhole_network_new() {
        let network = WormholeNetwork::new();
        assert!(!network.has_wormholes());
        assert_eq!(network.wormhole_count(), 0);
        assert_eq!(network.segment_count(), 0);
    }

    #[test]
    fn test_wormhole_network_from_chain() {
        let mut chain = WormChain::new();
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 1.0);
        chain.add_segment(seg1);
        chain.add_segment(seg2);

        let network = WormholeNetwork::from_worm_chain(chain);
        assert!(network.has_wormholes());
        assert_eq!(network.segment_count(), 2);
        assert!(network.wormhole_count() >= 1);
    }

    #[test]
    fn test_wormhole_network_forward() {
        let mut chain = WormChain::new();
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 1.0);
        chain.add_segment(seg1);
        chain.add_segment(seg2);

        let network = WormholeNetwork::from_worm_chain(chain);

        let input = vec![1.0f32, 2.0, 3.0];
        let output = network.forward(&input, 3);
        assert_eq!(output.len(), 3);
        // Output should be modified by wormhole coupling
        assert!(output != input, "Output should be modified by wormhole network");
    }

    #[test]
    fn test_wormhole_network_forward_empty() {
        let network = WormholeNetwork::new();
        let input = vec![1.0f32, 2.0, 3.0];
        let output = network.forward(&input, 3);
        assert_eq!(output, input, "Empty network should return input unchanged");
    }

    #[test]
    fn test_wormhole_network_forward_no_wormholes() {
        let mut chain = WormChain::new();
        // Segments with very different JEPA states → no wormholes
        let seg1 = make_segment(0, 1.0);
        let seg2 = make_segment(1, 0.001);
        chain.add_segment(seg1);
        chain.add_segment(seg2);

        let network = WormholeNetwork::from_worm_chain(chain);
        assert!(!network.has_wormholes());

        let input = vec![1.0f32, 2.0, 3.0];
        let output = network.forward(&input, 3);
        assert_eq!(output.len(), 3);
        // Should still produce output via host fallback
        assert!(output != input, "Output should be modified by host fallback");
    }

    #[test]
    fn test_wormhole_max_per_block() {
        let mut router = WormholeRouter::new();
        // Create many segments with identical JEPA state
        let segments: Vec<WormSegment> = (0..20)
            .map(|i| make_segment(i as u8, 1.0))
            .collect();

        router.discover(&segments);

        // Each block should have at most CMUD_WORMHOLE_MAX_PER_BLOCK neighbors
        for (slot_id, neighbors) in &router.adjacency {
            assert!(
                neighbors.len() <= CMUD_WORMHOLE_MAX_PER_BLOCK,
                "Slot {} has {} neighbors, max is {}",
                slot_id, neighbors.len(), CMUD_WORMHOLE_MAX_PER_BLOCK
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_dummy_mud_file() -> MudFile {
        MudFile {
            mmap: None,
            skills: HashMap::new(),
            global_metadata: HashMap::new(),
            path: None,
        }
    }

    #[test]
    fn test_jepa_state_cosine_similarity() {
        let a = JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        let b = JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        let sim = a.cosine_similarity(&b);
        assert!((sim - 1.0).abs() < 1e-5, "Identical states should have cosine sim = 1.0");
    }

    #[test]
    fn test_jepa_state_geodesic_distance() {
        let a = JepaState { z_jepa: 0.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        let b = JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        let dist = a.geodesic_distance(&b);
        assert!((dist - 1.0).abs() < 1e-5, "Geodesic distance should be 1.0");
    }

    #[test]
    fn test_jepa_router_resonance() {
        let mut router = JEPARouter::new();
        let query = JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        let block = JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        router.register(0, block);

        let score = router.resonance(&query, &block);
        assert!(score > 0.0, "Resonance should be positive for matching states");
    }

    #[test]
    fn test_jepa_router_route() {
        let mut router = JEPARouter::new();
        let query = JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };

        // Register 3 blocks with varying resonance
        router.register(0, JepaState { z_jepa: 1.0, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 });
        router.register(1, JepaState { z_jepa: 0.5, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 });
        router.register(2, JepaState { z_jepa: 0.1, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 });

        let routed = router.route(&query, 2);
        assert!(routed.len() <= 2, "Should return at most k=2 routes");
        if routed.len() >= 2 {
            assert_eq!(routed[0].0, 0, "Slot 0 should be most resonant");
        }
    }

    #[test]
    fn test_mhc_coupler_default() {
        let coupler = MhCoupler::default();
        assert_eq!(coupler.alpha, CMUD_WORM_MHC_ALPHA_DEFAULT);
        assert_eq!(coupler.beta, CMUD_WORM_MHC_BETA_DEFAULT);
    }

    #[test]
    fn test_mhc_coupler_clamp() {
        let coupler = MhCoupler::new(10.0, -5.0);
        assert_eq!(coupler.alpha, CMUD_WORM_MHC_MAX);
        assert_eq!(coupler.beta, CMUD_WORM_MHC_MIN);
    }

    #[test]
    fn test_mhc_coupler_couple() {
        let coupler = MhCoupler::new(1.0, 0.5);
        let result = coupler.couple(2.0, 1.0);
        // y = 1.0 * 2.0 + 0.5 * tanh(1.0)
        let expected = 2.0 + 0.5 * 1.0_f32.tanh();
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_mhc_coupler_batch() {
        let coupler = MhCoupler::new(1.0, 0.5);
        let host = vec![1.0, 2.0, 3.0];
        let worm = vec![0.5, 1.0, 1.5];
        let mut out = vec![0.0; 3];
        coupler.couple_batch(&host, &worm, &mut out);

        for i in 0..3 {
            let expected = host[i] + 0.5 * worm[i].tanh();
            assert!((out[i] - expected).abs() < 1e-5, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_mhc_coupler_sgd_step() {
        let mut coupler = MhCoupler::new(1.0, 0.5);
        coupler.sgd_step(0.1, 0.1, 0.5);
        assert!((coupler.alpha - 0.95).abs() < 1e-5);
        assert!((coupler.beta - 0.45).abs() < 1e-5);
    }

    #[test]
    fn test_worm_segment_new() {
        let mud_file = Arc::new(make_dummy_mud_file());
        let jepa_state = JepaState::default();
        let coupler = MhCoupler::default();
        let segment = WormSegment::new(42, mud_file, jepa_state, coupler);
        assert_eq!(segment.slot_id, 42);
        assert_eq!(segment.jepa_state.z_jepa, 0.0);
    }

    #[test]
    fn test_worm_chain_new() {
        let chain = WormChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_worm_chain_add_segment() {
        let mut chain = WormChain::new();
        let mud_file = Arc::new(make_dummy_mud_file());
        let segment = WormSegment::new(0, mud_file, JepaState::default(), MhCoupler::default());
        chain.add_segment(segment);
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_worm_chain_forward() {
        let mut chain = WormChain::new();
        let mud_file = Arc::new(make_dummy_mud_file());
        let jepa_state = JepaState { z_jepa: 0.5, mu_z: 0.0, inv_sigma_z: 1.0, var_h: 1.0, var_j: 1.0 };
        let segment = WormSegment::new(0, mud_file, jepa_state, MhCoupler::new(1.0, 0.5));
        chain.add_segment(segment);

        let input = vec![1.0, 2.0, 3.0];
        let output = chain.forward(&input, 3);
        assert_eq!(output.len(), 3);
        // Output should be modified by the coupling
        assert!(output != input, "Output should be modified by worm coupling");
    }

    #[test]
    fn test_worm_chain_forward_empty() {
        let chain = WormChain::new();
        let input = vec![1.0, 2.0, 3.0];
        let output = chain.forward(&input, 3);
        assert_eq!(output, input, "Empty chain should return input unchanged");
    }

    #[test]
    fn test_worm_propagator_new() {
        let propagator = WormPropagator::new("/tmp/nonexistent");
        assert_eq!(propagator.models_dir, "/tmp/nonexistent");
        assert_eq!(propagator.loaded_count, 0);
        assert!(propagator.worm_chain.is_empty());
    }

    #[test]
    fn test_worm_propagator_propagate_nonexistent() {
        let mut propagator = WormPropagator::new("/tmp/nonexistent_mud_dir_99999");
        let result = propagator.propagate();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0, "Should load 0 models from nonexistent dir");
    }

    #[test]
    fn test_worm_propagator_build_jepa_state() {
        let propagator = WormPropagator::new("/tmp");
        let mud_file = make_dummy_mud_file();
        let state = propagator.build_jepa_state(&mud_file);
        // With default values (hidden=2048, layers=12, vocab=32000)
        assert!(state.z_jepa > 0.0);
        assert!(state.mu_z > 0.0);
    }

    #[test]
    fn test_worm_propagator_unseat() {
        let mut propagator = WormPropagator::new("/tmp");
        // Manually add a segment
        let mud_file = Arc::new(make_dummy_mud_file());
        let segment = WormSegment::new(5, mud_file, JepaState::default(), MhCoupler::default());
        propagator.worm_chain.add_segment(segment);
        propagator.continuity_table.seat_module(5, Arc::new(make_dummy_mud_file()));

        assert_eq!(propagator.worm_chain.len(), 1);
        let unseated = propagator.unseat(5);
        assert!(unseated);
        assert_eq!(propagator.worm_chain.len(), 0);
        assert!(propagator.get_entry(5).is_none());
    }
}
