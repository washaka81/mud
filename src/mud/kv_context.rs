//! # L-13: CSA/HCA context policy — 32k-ready KV without O(L·T) dense RAM
//!
//! **Heavily Compressed Attention (HCA)** already mean-pools history in the forward
//! path. L-13 makes memory scale for long contexts:
//!
//! - **Logical** context up to [`MAX_LOGICAL_CONTEXT`] (32 768)
//! - **Dense ring** only keeps the recent window (+ one compression block)
//! - **HCA slots** store `logical_max / ratio` compressed KV vectors
//!
//! Effective attention still sees: all HCA blocks + recent dense tokens (same as before),
//! but physical dense KV is `O(window)` not `O(max_pos)`.

/// Hard ceiling for logical sequence length (32k roadmap target).
pub const MAX_LOGICAL_CONTEXT: usize = 32_768;
/// Default recent high-fidelity window (tokens).
pub const DEFAULT_HCA_WINDOW: usize = 256;
/// Default mean-pool block size for historical compression.
pub const DEFAULT_HCA_RATIO: usize = 10;
/// Cap on compressed history slots (keeps 32k footprint laptop-friendly ≈ tens of MB).
pub const MAX_HCA_SLOTS: usize = 512;

/// Resolved KV / HCA geometry for a workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvContextPolicy {
    /// Max absolute position (generation must keep `pos < logical_max_pos`).
    pub logical_max_pos: usize,
    /// Physical dense ring length (`>= hca_window + hca_ratio`).
    pub dense_cap: usize,
    /// Sliding-window size for full-fidelity tokens.
    pub hca_window: usize,
    /// Tokens per compressed HCA block.
    pub hca_ratio: usize,
    /// Number of compressed history slots (`logical_max / ratio`).
    pub hca_slots: usize,
}

impl KvContextPolicy {
    /// Build policy from model `max_position_embeddings` + env overrides.
    ///
    /// Env:
    /// - `MUD_MAX_POS` — override logical context (capped at 32k)
    /// - `MUD_HCA_WINDOW` — recent dense window
    /// - `MUD_HCA_RATIO` — compression block size
    pub fn resolve(requested_max_pos: usize) -> Self {
        let logical_max_pos = std::env::var("MUD_MAX_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(requested_max_pos)
            .clamp(1, MAX_LOGICAL_CONTEXT);

        let hca_window = std::env::var("MUD_HCA_WINDOW")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_HCA_WINDOW)
            .clamp(32, 4096);

        let hca_ratio = std::env::var("MUD_HCA_RATIO")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_HCA_RATIO)
            .clamp(2, 64);

        Self::from_parts(logical_max_pos, hca_window, hca_ratio)
    }

    /// Pure constructor (tests / deterministic tools).
    pub fn from_parts(logical_max_pos: usize, hca_window: usize, hca_ratio: usize) -> Self {
        let logical_max_pos = logical_max_pos.clamp(1, MAX_LOGICAL_CONTEXT);
        let hca_window = hca_window.clamp(32, 4096);
        let mut hca_ratio = hca_ratio.clamp(2, 64);
        // Auto-raise ratio so compressed slots stay ≤ MAX_HCA_SLOTS (32k → ~16:1).
        let min_ratio_for_cap = logical_max_pos.div_ceil(MAX_HCA_SLOTS).max(2);
        if hca_ratio < min_ratio_for_cap {
            hca_ratio = min_ratio_for_cap.min(64);
        }
        // Live dense span is at most window + ratio (see HCA recent_start math).
        let min_dense = hca_window.saturating_add(hca_ratio);
        let dense_cap = if logical_max_pos <= min_dense {
            logical_max_pos
        } else {
            min_dense
        };
        let hca_slots = (logical_max_pos / hca_ratio).clamp(1, MAX_HCA_SLOTS);
        Self {
            logical_max_pos,
            dense_cap,
            hca_window,
            hca_ratio,
            hca_slots,
        }
    }

    /// Bytes for K+V dense ring (one layer stack).
    pub fn dense_kv_bytes(self, num_layers: usize, n_kv_heads: usize, head_dim: usize) -> usize {
        let elems = num_layers
            .saturating_mul(n_kv_heads)
            .saturating_mul(self.dense_cap)
            .saturating_mul(head_dim);
        elems.saturating_mul(4).saturating_mul(2) // K + V
    }

    /// Bytes for HCA K+V compressed history.
    pub fn hca_kv_bytes(self, num_layers: usize, n_kv_heads: usize, head_dim: usize) -> usize {
        let elems = num_layers
            .saturating_mul(n_kv_heads)
            .saturating_mul(self.hca_slots)
            .saturating_mul(head_dim);
        elems.saturating_mul(4).saturating_mul(2)
    }

    /// Total KV-related footprint (dense ring + HCA).
    pub fn total_kv_bytes(self, num_layers: usize, n_kv_heads: usize, head_dim: usize) -> usize {
        self.dense_kv_bytes(num_layers, n_kv_heads, head_dim)
            + self.hca_kv_bytes(num_layers, n_kv_heads, head_dim)
    }

    /// Score buffer length: compressed slots + dense recent (+ margin).
    pub fn scores_len(self) -> usize {
        self.hca_slots
            .saturating_add(self.dense_cap)
            .saturating_add(8)
    }

    /// Map absolute position → dense ring slot.
    #[inline]
    pub fn dense_slot(self, pos: usize) -> usize {
        pos % self.dense_cap.max(1)
    }

    /// Memory savings vs naive dense `logical_max` allocation (K+V only).
    pub fn savings_vs_naive(
        self,
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (usize, usize) {
        let naive = num_layers
            .saturating_mul(n_kv_heads)
            .saturating_mul(self.logical_max_pos)
            .saturating_mul(head_dim)
            .saturating_mul(4)
            .saturating_mul(2);
        let actual = self.total_kv_bytes(num_layers, n_kv_heads, head_dim);
        (naive, actual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_32k_policy_caps_dense() {
        let p = KvContextPolicy::from_parts(32_768, 256, 10);
        assert_eq!(p.logical_max_pos, 32_768);
        // ratio auto-bumped so hca_slots ≤ 512 → ratio ≥ 64 for 32k
        assert!(p.hca_ratio >= 64);
        assert!(p.hca_slots <= MAX_HCA_SLOTS);
        assert_eq!(p.dense_cap, 256 + p.hca_ratio);
        assert!(p.dense_cap < p.logical_max_pos);
    }

    #[test]
    fn test_short_context_full_dense() {
        let p = KvContextPolicy::from_parts(128, 256, 10);
        // logical <= window+ratio → dense_cap == logical
        assert_eq!(p.dense_cap, 128);
        assert_eq!(p.logical_max_pos, 128);
    }

    #[test]
    fn test_dense_slot_unique_in_window() {
        let p = KvContextPolicy::from_parts(4096, 256, 10);
        let d = p.dense_cap;
        // Any run of `d` consecutive positions → unique slots
        let start = 1000usize;
        let mut seen = vec![false; d];
        for t in start..start + d {
            let s = p.dense_slot(t);
            assert!(!seen[s], "collision at t={t} slot={s}");
            seen[s] = true;
        }
    }

    #[test]
    fn test_memory_savings_32k() {
        // BitNet-ish: 30 layers, 3 kv heads, head 64
        let p = KvContextPolicy::from_parts(32_768, 256, 10);
        let (naive, actual) = p.savings_vs_naive(30, 3, 64);
        assert!(
            naive > actual * 5,
            "expected >>5× savings: naive={naive} actual={actual}"
        );
        // Sanity: actual under ~200 MB for this toy geo
        assert!(actual < 200 * 1024 * 1024, "actual footprint {actual}");
    }

    #[test]
    fn test_scores_len_covers_attn() {
        let p = KvContextPolicy::from_parts(32_768, 256, 10);
        // max attn elements ≈ hca_slots + dense span
        let max_attn = p.hca_slots + p.dense_cap;
        assert!(p.scores_len() >= max_attn);
    }
}
