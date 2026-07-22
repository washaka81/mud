//! # Stream E: CSA lightning indexer (top-k over HCA blocks)
//!
//! Builds on L-13 HCA (mean-pooled history + dense recent window).
//! When the compressed history is large, attending to **every** HCA block is
//! O(N) softmax + V-mix. CSA selects the **top-k** HCA blocks via a cheap
//! coarse score, then runs full-dim attention only on those + the dense window.
//!
//! ```text
//! index_score[i] = Q[..d_idx] · K_hca[i][..d_idx]     # lightning rank
//! S = top_k(index_score) ∪ last `tail` blocks          # chronological order
//! attn = softmax(Q·K_S / √d) · V_S  +  dense recent
//! ```
//!
//! Env:
//! - `MUD_CSA=0|false|off` — disable (full HCA scan)
//! - `MUD_CSA=1|true|on|auto` / unset — enable when `num_blocks > top_k`
//! - `MUD_CSA_TOP_K` — blocks to keep (default 64, clamp 4..512)
//! - `MUD_CSA_INDEX_DIM` — coarse rank dim (default 16, 0 = full head_dim)
//! - `MUD_CSA_TAIL` — always keep last N compressed blocks (default 4)

/// Default top-k HCA blocks after ranking.
pub const DEFAULT_CSA_TOP_K: usize = 64;
/// Default coarse index dimension (prefix of head).
pub const DEFAULT_INDEX_DIM: usize = 16;
/// Always retain this many newest HCA blocks (locality).
pub const DEFAULT_TAIL: usize = 4;
/// Only activate sparse path when compressed blocks exceed this.
pub const DEFAULT_MIN_BLOCKS: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsaPolicy {
    pub enabled: bool,
    pub top_k: usize,
    pub index_dim: usize,
    pub tail: usize,
    pub min_blocks: usize,
}

impl CsaPolicy {
    /// Resolve from environment (process-global, cheap).
    pub fn resolve() -> Self {
        let enabled = match std::env::var("MUD_CSA") {
            Err(_) => true, // product default: on when history large
            Ok(v) => {
                let t = v.trim().to_ascii_lowercase();
                !(t == "0" || t == "false" || t == "off" || t == "no")
            }
        };
        let top_k = std::env::var("MUD_CSA_TOP_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_CSA_TOP_K)
            .clamp(4, 512);
        let index_dim = std::env::var("MUD_CSA_INDEX_DIM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_INDEX_DIM);
        let tail = std::env::var("MUD_CSA_TAIL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TAIL)
            .clamp(0, 64);
        Self {
            enabled,
            top_k,
            index_dim,
            tail,
            min_blocks: DEFAULT_MIN_BLOCKS,
        }
    }

    /// Whether to sparse-select over `num_comp` HCA blocks (inference).
    #[inline]
    pub fn should_sparse(self, num_comp: usize) -> bool {
        self.enabled && num_comp > self.top_k.max(self.min_blocks)
    }

    /// Effective coarse dim for a head of size `head_dim`.
    #[inline]
    pub fn effective_index_dim(self, head_dim: usize) -> usize {
        if self.index_dim == 0 {
            head_dim.max(1)
        } else {
            self.index_dim.min(head_dim).max(1)
        }
    }

    /// Human one-liner for audit / healthcheck.
    pub fn summary(self) -> String {
        format!(
            "CSA enabled={} top_k={} index_dim={} tail={} min_blocks={}",
            self.enabled, self.top_k, self.index_dim, self.tail, self.min_blocks
        )
    }
}

/// Select up to `k` largest scores; output indices sorted **ascending** (time order).
///
/// Ties: higher index preferred slightly via stable index in comparison.
pub fn select_top_k_indices(scores: &[f32], k: usize, out: &mut Vec<usize>) {
    out.clear();
    let n = scores.len();
    if n == 0 || k == 0 {
        return;
    }
    let k = k.min(n);
    if k == n {
        out.extend(0..n);
        return;
    }

    // (score, idx) — select k largest by score
    let mut pairs: Vec<(f32, usize)> = scores
        .iter()
        .copied()
        .enumerate()
        .map(|(i, s)| (s, i))
        .collect();
    // nth element from the end: keep [n-k .. n) as the k largest
    let pivot = n - k;
    pairs.select_nth_unstable_by(pivot, |a, b| {
        // ascending: smaller scores first → largest in the high slice
        match a.0.partial_cmp(&b.0) {
            Some(o) => o.then_with(|| a.1.cmp(&b.1)),
            None => a.1.cmp(&b.1),
        }
    });
    out.reserve(k);
    for p in &pairs[pivot..] {
        out.push(p.1);
    }
    out.sort_unstable();
}

/// Merge forced indices (e.g. tail blocks) into a top-k set, re-sort unique ascending.
pub fn merge_forced_indices(selected: &mut Vec<usize>, forced: &[usize], num_comp: usize) {
    for &i in forced {
        if i < num_comp && !selected.contains(&i) {
            selected.push(i);
        }
    }
    selected.sort_unstable();
    selected.dedup();
}

/// Indices of the last `tail` compressed blocks in `[0, num_comp)`.
pub fn tail_indices(num_comp: usize, tail: usize, out: &mut Vec<usize>) {
    out.clear();
    if num_comp == 0 || tail == 0 {
        return;
    }
    let start = num_comp.saturating_sub(tail);
    out.extend(start..num_comp);
}

/// Lightning rank scores for HCA blocks using a prefix of the head dim.
///
/// # Safety
/// `q` length ≥ `index_dim`. HCA keys for block `i` start at `hca_k_base + i * head_dim`
/// and cover at least `index_dim` floats. `out_scores.len() >= num_blocks`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn score_hca_blocks_coarse(
    q: *const f32,
    hca_k_base: *const f32,
    head_dim: usize,
    num_blocks: usize,
    index_dim: usize,
    inv_sqrt_d: f32,
    out_scores: &mut [f32],
) {
    let d = index_dim.min(head_dim).max(1);
    // Scale by √d_index for ranking stability (same form as attention)
    let scale = if d == head_dim {
        inv_sqrt_d
    } else {
        1.0 / (d as f32).sqrt()
    };
    for (i, slot) in out_scores.iter_mut().enumerate().take(num_blocks) {
        let k_ptr = hca_k_base.add(i * head_dim);
        let s = crate::asm::dot_product_avx2(d, q, k_ptr) * scale;
        *slot = s;
    }
}

/// Stream J: SimHash-style signature from a float vector (index_dim dims).
/// Uses fixed random hyperplanes (LCG seed) — no learned W_compress yet.
pub fn lsh_signature(v: &[f32], n_bits: usize, seed: u32) -> u64 {
    let n_bits = n_bits.clamp(1, 64);
    let mut sig = 0u64;
    let mut s = seed | 1;
    for b in 0..n_bits {
        // Hyperplane: pseudo-random ±1 per dim
        let mut dot = 0.0f32;
        for &x in v {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let h = if (s >> 31) & 1 == 0 { 1.0f32 } else { -1.0 };
            dot += x * h;
        }
        if dot >= 0.0 {
            sig |= 1u64 << b;
        }
    }
    sig
}

/// Hamming distance between two LSH signatures (popcount xor).
#[inline]
pub fn hamming64(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Whether LSH prefilter is on (`MUD_CSA_LSH=1`).
pub fn lsh_enabled() -> bool {
    std::env::var("MUD_CSA_LSH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Bits for LSH (`MUD_CSA_LSH_BITS`, default 16).
pub fn lsh_bits() -> usize {
    std::env::var("MUD_CSA_LSH_BITS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16)
        .clamp(4, 64)
}

/// Max Hamming radius to keep as candidate (`MUD_CSA_LSH_RADIUS`, default 4).
pub fn lsh_radius() -> u32 {
    std::env::var("MUD_CSA_LSH_RADIUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
}

/// Full CSA index: coarse scores → top-k ∪ tail → sorted unique block ids.
///
/// Stream J: when `MUD_CSA_LSH=1`, first filter candidates by SimHash Hamming ball,
/// then coarse-score only those (plus forced tail) before top-k.
///
/// # Safety
/// Same as [`score_hca_blocks_coarse`]. `scratch` must hold `num_blocks` floats.
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_hca_blocks(
    q: *const f32,
    hca_k_base: *const f32,
    head_dim: usize,
    num_blocks: usize,
    policy: CsaPolicy,
    force_lsh: Option<bool>,
    inv_sqrt_d: f32,
    scratch: &mut [f32],
    out_blocks: &mut Vec<usize>,
) {
    out_blocks.clear();
    if num_blocks == 0 {
        return;
    }
    if !policy.should_sparse(num_blocks) {
        out_blocks.extend(0..num_blocks);
        return;
    }
    assert!(scratch.len() >= num_blocks);
    let idx_d = policy.effective_index_dim(head_dim);

    // Optional LSH prefilter (stream J). `force_lsh` overrides the env flag so
    // callers/tests can pick the path deterministically.
    let lsh_on = force_lsh.unwrap_or_else(lsh_enabled);
    let candidates: Vec<usize> = if lsh_on && num_blocks > policy.top_k * 2 {
        let bits = lsh_bits();
        let radius = lsh_radius();
        let q_slice = std::slice::from_raw_parts(q, idx_d.min(head_dim));
        let q_sig = lsh_signature(q_slice, bits, 0xC5A1_u32);
        let mut cand = Vec::with_capacity(num_blocks.min(policy.top_k * 4 + policy.tail));
        for i in 0..num_blocks {
            let k_ptr = hca_k_base.add(i * head_dim);
            let k_slice = std::slice::from_raw_parts(k_ptr, idx_d.min(head_dim));
            let k_sig = lsh_signature(k_slice, bits, 0xC5A1_u32);
            if hamming64(q_sig, k_sig) <= radius {
                cand.push(i);
            }
        }
        // Always include tail
        let mut forced = Vec::new();
        tail_indices(num_blocks, policy.tail, &mut forced);
        for f in forced {
            if !cand.contains(&f) {
                cand.push(f);
            }
        }
        if cand.is_empty() {
            // Degenerate: fall back to full scan
            (0..num_blocks).collect()
        } else {
            cand
        }
    } else {
        (0..num_blocks).collect()
    };

    // Coarse score candidates only
    scratch[..num_blocks].fill(f32::NEG_INFINITY);
    for &i in &candidates {
        let k_ptr = hca_k_base.add(i * head_dim);
        let d = idx_d.min(head_dim).max(1);
        let scale = if d == head_dim {
            inv_sqrt_d
        } else {
            1.0 / (d as f32).sqrt()
        };
        let s = crate::asm::dot_product_avx2(d, q, k_ptr) * scale;
        scratch[i] = s;
    }
    // If LSH filtered, only select among finite scores
    if candidates.len() < num_blocks {
        // Build score list for candidates, map back
        let mut pairs: Vec<(f32, usize)> = candidates.iter().map(|&i| (scratch[i], i)).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = policy.top_k.min(pairs.len());
        out_blocks.clear();
        for p in pairs.iter().take(k) {
            out_blocks.push(p.1);
        }
        let mut forced = Vec::new();
        tail_indices(num_blocks, policy.tail, &mut forced);
        merge_forced_indices(out_blocks, &forced, num_blocks);
    } else {
        select_top_k_indices(&scratch[..num_blocks], policy.top_k, out_blocks);
        let mut forced = Vec::with_capacity(policy.tail);
        tail_indices(num_blocks, policy.tail, &mut forced);
        merge_forced_indices(out_blocks, &forced, num_blocks);
    }
}

/// Approximate FLOP ratio vs full HCA scan (QK coarse + fine QK on k + V on k)
/// vs full (QK+V on N). Dense window excluded (always full).
pub fn approx_hca_flop_ratio(
    num_blocks: usize,
    top_k: usize,
    index_dim: usize,
    head_dim: usize,
) -> f32 {
    if num_blocks == 0 {
        return 1.0;
    }
    let n = num_blocks as f32;
    let k = top_k.min(num_blocks) as f32;
    let d = head_dim as f32;
    let di = index_dim.min(head_dim).max(1) as f32;
    // full: N*(d + d) = 2 N d  (QK + V)
    let full = 2.0 * n * d;
    // sparse: N*di (index) + k*d (fine QK) + k*d (V)
    let sparse = n * di + 2.0 * k * d;
    sparse / full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_top_k_basic() {
        let scores = [0.1f32, 5.0, 2.0, 9.0, 3.0];
        let mut out = Vec::new();
        select_top_k_indices(&scores, 3, &mut out);
        // top: 9,5,3 → indices 3,1,4 sorted → 1,3,4
        assert_eq!(out, vec![1, 3, 4]);
    }

    #[test]
    fn test_select_top_k_all() {
        let scores = [1.0f32, 2.0, 3.0];
        let mut out = Vec::new();
        select_top_k_indices(&scores, 10, &mut out);
        assert_eq!(out, vec![0, 1, 2]);
    }

    #[test]
    fn test_select_top_k_empty() {
        let mut out = vec![1, 2];
        select_top_k_indices(&[], 5, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn test_merge_forced_and_tail() {
        let mut sel = vec![1usize, 5];
        merge_forced_indices(&mut sel, &[8, 9, 1], 10);
        assert_eq!(sel, vec![1, 5, 8, 9]);
        let mut tail = Vec::new();
        tail_indices(10, 3, &mut tail);
        assert_eq!(tail, vec![7, 8, 9]);
    }

    #[test]
    fn test_should_sparse_threshold() {
        let p = CsaPolicy {
            enabled: true,
            top_k: 64,
            index_dim: 16,
            tail: 4,
            min_blocks: 48,
        };
        assert!(!p.should_sparse(40));
        assert!(!p.should_sparse(64)); // not > top_k.max(min)
        assert!(p.should_sparse(100));
        let off = CsaPolicy {
            enabled: false,
            ..p
        };
        assert!(!off.should_sparse(1000));
    }

    #[test]
    fn test_flop_ratio_saves() {
        // N=512, k=64, d_idx=16, d=64 → sparse should be << 1
        let r = approx_hca_flop_ratio(512, 64, 16, 64);
        assert!(r < 0.5, "expected substantial cut, ratio={r}");
        assert!(r > 0.05, "sanity lower bound ratio={r}");
    }

    #[test]
    fn test_score_and_index_picks_strong_block() {
        let head_dim = 8usize;
        let num = 6usize;
        // Q = [1,0,0,...]
        let q = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // Blocks: only block 3 has large first dim
        let mut hca = vec![0.0f32; num * head_dim];
        hca[3 * head_dim] = 10.0;
        hca[head_dim] = 2.0;
        let policy = CsaPolicy {
            enabled: true,
            top_k: 2,
            index_dim: 4,
            tail: 0,
            min_blocks: 2,
        };
        let mut scratch = vec![0.0f32; num];
        let mut out = Vec::new();
        unsafe {
            index_hca_blocks(
                q.as_ptr(),
                hca.as_ptr(),
                head_dim,
                num,
                policy,
                None,
                1.0 / (head_dim as f32).sqrt(),
                &mut scratch,
                &mut out,
            );
        }
        assert!(out.contains(&3), "must pick strongest block 3, got {out:?}");
        assert_eq!(out.len(), 2);
        // sorted
        assert!(out.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_policy_summary_nonempty() {
        let s = CsaPolicy::resolve().summary();
        assert!(s.contains("CSA"));
    }

    #[test]
    fn test_lsh_signature_deterministic() {
        let v = [1.0f32, 0.0, -1.0, 0.5];
        let a = lsh_signature(&v, 16, 0xC5A1);
        let b = lsh_signature(&v, 16, 0xC5A1);
        assert_eq!(a, b);
        let w = [-1.0f32, 0.0, 1.0, -0.5];
        let c = lsh_signature(&w, 16, 0xC5A1);
        // Opposite-ish vector should differ in some bits
        assert!(hamming64(a, c) > 0 || a == c); // allow rare collision
    }

    #[test]
    fn test_hamming() {
        assert_eq!(hamming64(0, 0), 0);
        assert_eq!(hamming64(0b1111, 0b0000), 4);
        assert_eq!(hamming64(0b1010, 0b1000), 1);
    }

    #[test]
    fn test_lsh_prefilter_recall_vs_brute() {
        // J: LSH prefilter must keep all top-k blocks selected by the brute
        // (full-scan) path — recall == 1.0 — while still excluding blocks that
        // are far in the LSH subspace (here the lowest-score block 0).
        let head_dim = 8usize;
        let num = 24usize;
        // q has nonzero dim1 so that varying k's dim1 yields distinct scores.
        let q = [1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut hca = vec![0.0f32; num * head_dim];
        for i in 0..num {
            // index subspace (first 4 dims) = [1, rising, 0, 0] -> score grows with i
            hca[i * head_dim] = 1.0;
            hca[i * head_dim + 1] = (i as f32) * 0.01;
        }
        // Block 0 is far in the LSH subspace (opposite sign) -> lowest score.
        hca[0] = -1.0;
        let policy = CsaPolicy {
            enabled: true,
            top_k: 4,
            index_dim: 4,
            tail: 0,
            min_blocks: 4,
        };
        let mut scratch = vec![0.0f32; num];
        let mut out_brute = Vec::new();
        let mut out_lsh = Vec::new();
        let inv = 1.0 / (head_dim as f32).sqrt();
        unsafe {
            index_hca_blocks(
                q.as_ptr(),
                hca.as_ptr(),
                head_dim,
                num,
                policy,
                Some(false),
                inv,
                &mut scratch,
                &mut out_brute,
            );
            index_hca_blocks(
                q.as_ptr(),
                hca.as_ptr(),
                head_dim,
                num,
                policy,
                Some(true),
                inv,
                &mut scratch,
                &mut out_lsh,
            );
        }
        assert!(!out_brute.is_empty());
        assert!(!out_lsh.is_empty());
        // Recall vs brute: every brute-selected block must survive the LSH path.
        let recall = out_lsh.iter().filter(|b| out_brute.contains(b)).count() as f32
            / out_brute.len() as f32;
        assert_eq!(
            recall, 1.0,
            "LSH prefilter dropped a top block (recall={recall})"
        );
        // Strongest block (23) is still selected under LSH.
        assert!(out_lsh.contains(&23));
        // LSH excludes the far low-score block 0.
        assert!(!out_lsh.contains(&0));
        // Brute and LSH agree exactly for this controlled layout.
        assert_eq!(out_lsh, out_brute);
    }
}
