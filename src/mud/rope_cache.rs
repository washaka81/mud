//! RoPE cache and inverse-frequency precomputation (extracted from `slime_forward.rs`).
//!
//! PERF-1 (V25 audit): precompute `inv_freq[i] = rope_theta^(-2i/head_d)` once.
//! Avoids `powf` (libm, ~50 ns) per (pos, head, i) call site. The recurrence
//! `theta[i+1] = theta[i] * inv_freq[i]` uses 1 multiply (~1 ns).

/// Precompute `rope_theta^(-2*i/head_d)` for i in 0..half_d.
/// Returns empty vec when RoPE is disabled (theta ≤ 0 or head_d == 0).
pub fn compute_rope_inv_freq(rope_theta: f32, head_d: usize) -> Vec<f32> {
    if rope_theta <= 0.0 || head_d == 0 {
        return Vec::new();
    }
    let half_d = head_d / 2;
    let mut inv = Vec::with_capacity(half_d);
    for i in 0..half_d {
        inv.push(rope_theta.powf(-2.0 * i as f32 / head_d as f32));
    }
    inv
}

/// Per-position RoPE cos/sin cache (16-entry ring).
/// Key insight: pos increments by 1 in both training and AR inference, so a small
/// LRU is near-perfect after the first token. Locks are per-layer and only
/// acquired on cache miss; the inner RoPE loop is lock-free.
pub struct RopeCache {
    entries: [(usize, Vec<f32>, Vec<f32>); 16],
}

impl RopeCache {
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        const EMPTY: (usize, Vec<f32>, Vec<f32>) = (usize::MAX, Vec::new(), Vec::new());
        Self { entries: [EMPTY; 16] }
    }
    /// Returns borrowed cos/sin slices (zero-clone on cache hit).
    /// Caller must hold the `MutexGuard` while using the returned slices.
    #[inline]
    pub fn get_or_compute(&mut self, pos: usize, inv_freq: &[f32]) -> (&[f32], &[f32]) {
        // Phase 1: Check cache (immutable borrow ends here).
        if let Some(idx) = self.find_cache(pos, inv_freq.len()) {
            return (&self.entries[idx].1, &self.entries[idx].2);
        }
        // Phase 2: Cache miss — compute and insert (mutable borrow).
        let half_d = inv_freq.len();
        let mut cos = vec![0.0f32; half_d];
        let mut sin = vec![0.0f32; half_d];
        let mut theta = pos as f32;
        for i in 0..half_d {
            let (s, c) = theta.sin_cos();
            cos[i] = c;
            sin[i] = s;
            theta *= inv_freq[i];
        }
        // Evict the oldest entry (index 0) by shifting — small N, fast.
        // Uses raw pointers to avoid borrow-checker conflicts with the return refs.
        let n = self.entries.len();
        for i in 0..n - 1 {
            unsafe {
                let tmp = core::ptr::read(self.entries.as_ptr().add(i + 1));
                core::ptr::write(self.entries.as_mut_ptr().add(i), tmp);
                // Invalidate source to prevent double-free on drop.
                core::ptr::write(
                    self.entries.as_mut_ptr().add(i + 1),
                    (usize::MAX, Vec::new(), Vec::new()),
                );
            }
        }
        let last = n - 1;
        self.entries[last] = (pos, cos, sin);
        // Phase 3: Return references (immutable borrow).
        // SAFETY: self.entries[last] was just written and no further mutation occurs
        // before the references are used. The caller holds the MutexGuard, preventing
        // concurrent access.
        unsafe {
            let cos_ptr = self.entries[last].1.as_ptr();
            let sin_ptr = self.entries[last].2.as_ptr();
            let cos_len = self.entries[last].1.len();
            let sin_len = self.entries[last].2.len();
            (
                std::slice::from_raw_parts(cos_ptr, cos_len),
                std::slice::from_raw_parts(sin_ptr, sin_len),
            )
        }
    }

    #[inline]
    fn find_cache(&self, pos: usize, len: usize) -> Option<usize> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.0 == pos && entry.1.len() == len {
                return Some(i);
            }
        }
        None
    }
}
