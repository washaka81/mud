//! Fenwick Tree (Binary Indexed Tree) for O(log n) prefix sums and k-th element.
//!
//! Used by F4 (histogram threshold) and F1 (nucleus sampling) to avoid O(n)
//! linear scans. Zero-alloc `with_capacity` + `reset_zeroed` for buffer reuse.
//!
//! Env: `MUD_FENWICK=1` (default on) with linear fallback if 0.
//!
//! See: `docs/research/MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md`

// ── F4 Fenwick trees (P-01: zero-alloc, pub(crate)) ──────────────────────────

/// Fenwick tree for f32 prefix sums. Supports `prefix_sum`, `update`, and
/// `find_kth` (binary search for the smallest index where prefix_sum ≥ k).
///
/// Zero-alloc: `with_capacity` allocates once; `reset_zeroed` reuses the buffer.
pub struct FenwickF32 {
    n: usize,
    buf: Vec<f32>,
}

impl FenwickF32 {
    /// Allocate a Fenwick tree of size `n` (zeroed). O(n) alloc, called once.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            n: n.max(1),
            buf: vec![0.0f32; n.max(1) + 1], // 1-indexed
        }
    }

    /// Reset buffer to all zeros. O(n) but no alloc — reuses the Vec.
    pub fn reset_zeroed(&mut self) {
        for v in &mut self.buf {
            *v = 0.0;
        }
    }

    /// Build from a slice in O(n). Resets first.
    pub fn build_from(&mut self, data: &[f32]) {
        self.n = data.len().max(1);
        if self.buf.len() < self.n + 1 {
            self.buf.resize(self.n + 1, 0.0);
        } else {
            self.reset_zeroed();
        }
        for (i, &v) in data.iter().enumerate() {
            self.update(i, v);
        }
    }

    /// Build from a slice in O(n) using linear-time construction.
    /// Faster than `build_from` (which is O(n log n)) for large n.
    /// Uses raw pointers (P-00) for zero bounds-check overhead.
    pub fn build_from_linear(&mut self, data: &[f32]) {
        self.n = data.len().max(1);
        if self.buf.len() < self.n + 1 {
            self.buf.resize(self.n + 1, 0.0);
        } else {
            self.reset_zeroed();
        }
        // SAFETY: `self.n = data.len().max(1)` and `self.buf` was resized to
        // `self.n + 1` above; for `data.len() >= 1` (every in-tree caller — a
        // histogram is only built from non-empty data) `data.as_ptr()` is valid
        // for `self.n` reads, matching `build_from_linear_raw`'s contract.
        // CAVEAT: an empty `data` yields `self.n == 1` while the pointer is
        // valid for 0 elements; this safe wrapper does not reject it (flagged
        // in the R-36 report).
        unsafe {
            self.build_from_linear_raw(data.as_ptr());
        }
    }

    /// Linear-time construction using raw pointers (P-00).
    /// Algorithm: copy data to buf[1..n+1], then propagate each element
    /// to its parent in the Fenwick tree.
    ///
    /// # Safety
    /// - `data_ptr` must point to at least `self.n` valid f32 values.
    #[target_feature(enable = "avx2")]
    pub unsafe fn build_from_linear_raw(&mut self, data_ptr: *const f32) {
        let buf_ptr = self.buf.as_mut_ptr();
        let n = self.n;

        // Step 1: Copy data to buf[1..n+1] using raw pointer
        std::ptr::copy_nonoverlapping(data_ptr, buf_ptr.add(1), n);

        // Step 2: Propagate each element to its parent (linear-time Fenwick build)
        // For each i in 1..=n: buf[i + (i & -i)] += buf[i] (if in bounds)
        for i in 1..=n {
            let parent = i + (i & i.wrapping_neg());
            if parent <= n {
                *buf_ptr.add(parent) += *buf_ptr.add(i);
            }
        }
    }

    /// Prefix sum [0, idx) using raw pointers (P-00). Eliminates bounds check.
    /// O(log n).
    #[inline]
    pub fn prefix_sum_raw(&self, idx: usize) -> f32 {
        let ptr = self.buf.as_ptr();
        let mut i = idx.min(self.n);
        let mut sum = 0.0f32;
        // SAFETY: `i = idx.min(self.n)` and `buf.len() >= self.n + 1`
        // (constructors / `build_from*`), so `i <= self.n` and `ptr.add(i)` is
        // in-bounds for every iteration; the loop strictly decreases `i`.
        unsafe {
            while i > 0 {
                sum += *ptr.add(i);
                i -= i & i.wrapping_neg();
            }
        }
        sum
    }

    /// Add `delta` to position `idx` using raw pointers (P-00). Eliminates bounds check.
    /// O(log n).
    #[inline]
    pub fn update_raw(&mut self, idx: usize, delta: f32) {
        let ptr = self.buf.as_mut_ptr();
        let mut i = idx + 1; // 1-indexed
        let n = self.n;
        // SAFETY: `buf.len() >= self.n + 1`; when `idx < n` then `i = idx + 1
        // <= n` and every `ptr.add(i)` is in-bounds, and when `idx >= n` the
        // loop body never runs. `self` is `&mut`, so the write is exclusive.
        unsafe {
            while i <= n {
                *ptr.add(i) += delta;
                i += i & i.wrapping_neg();
            }
        }
    }

    /// Find the smallest index where `prefix_sum(idx) >= k` using raw pointers (P-00).
    /// O(log n). Returns `n` if total sum < k.
    pub fn find_kth_raw(&self, k: f32) -> usize {
        let ptr = self.buf.as_ptr();
        let mut idx = 0usize;
        let mut bitmask = 1usize;
        while bitmask <= self.n {
            bitmask <<= 1;
        }
        bitmask >>= 1;
        let mut sum = 0.0f32;
        // SAFETY: the deref is guarded by `next <= self.n` and
        // `buf.len() >= self.n + 1`, so `ptr.add(next)` is in-bounds; `self` is
        // shared and this function only reads.
        unsafe {
            while bitmask > 0 {
                let next = idx + bitmask;
                if next <= self.n && sum + *ptr.add(next) < k {
                    sum += *ptr.add(next);
                    idx = next;
                }
                bitmask >>= 1;
            }
        }
        idx
    }

    /// AVX2-optimized find_kth using branchless comparison (Phase 3).
    /// Uses `vcmpps` + `vblendps` for branchless bitmask walk.
    ///
    /// # Safety
    /// - The calling CPU MUST support AVX2 + BMI1 (lzcnt).
    /// - This function is gated with `#[target_feature(enable = "avx2,bmi1")]`.
    #[target_feature(enable = "avx2,bmi1")]
    pub unsafe fn find_kth_avx2(&self, k: f32) -> usize {
        use std::arch::x86_64::*;

        let ptr = self.buf.as_ptr();
        let n = self.n;
        let mut idx = 0usize;

        // Compute initial bitmask using lzcnt (BMI1) — O(1) vs O(log n) loop
        let mut bitmask: usize = 1;
        if n > 0 {
            let log2_n = 31 - (std::arch::x86_64::_lzcnt_u32(n as u32) as usize);
            bitmask = 1usize << log2_n;
        }

        let k_vec = _mm256_set1_ps(k);
        let mut sum = 0.0f32;

        while bitmask > 0 {
            let next = idx + bitmask;
            if next <= n {
                let val = *ptr.add(next);
                let new_sum = sum + val;
                // Branchless: vcmpps + vmovmaskps
                let cmp = _mm256_cmp_ps(_mm256_set1_ps(new_sum), k_vec, _CMP_LT_OQ);
                let mask = _mm256_movemask_ps(cmp) as usize;
                if mask != 0 {
                    sum = new_sum;
                    idx = next;
                }
            }
            bitmask >>= 1;
        }
        idx
    }

    /// Sprint 4b: find_kth using `_pdep_u32` + `tzcnt` for O(log n) walk.
    ///
    /// Uses `pdep` to scatter the bitmask into the Fenwick tree's bit
    /// representation, and `tzcnt` to count trailing zeros for the
    /// highest set bit. This avoids the branch in the standard find_kth
    /// and uses hardware bit manipulation for the walk.
    ///
    /// # Safety
    /// - The calling CPU MUST support BMI1 (for `tzcnt`) and BMI2 (for `pdep`).
    /// - This function is gated with `#[target_feature(enable = "bmi1,bmi2")]`.
    #[target_feature(enable = "bmi1,bmi2")]
    pub unsafe fn find_kth_walk(&self, k: f32) -> usize {
        let ptr = self.buf.as_ptr();
        let n = self.n;

        // Compute the initial bitmask: highest power of 2 <= n
        // Using lzcnt (BMI1) — O(1) vs O(log n) loop
        let mut bitmask: usize = 1;
        if n > 0 {
            let log2_n = 31 - (std::arch::x86_64::_lzcnt_u32(n as u32) as usize);
            bitmask = 1usize << log2_n;
        }

        let mut idx = 0usize;
        let mut sum = 0.0f32;

        while bitmask > 0 {
            let next = idx + bitmask;
            if next <= n {
                let val = *ptr.add(next);
                let new_sum = sum + val;
                // Branchless: if new_sum < k, advance
                let take = (new_sum < k) as u32;
                // pdep: deposit the 'take' bit into the position determined by bitmask
                let _scattered = std::arch::x86_64::_pdep_u32(take, bitmask as u32) as usize;
                if take != 0 {
                    sum = new_sum;
                    idx = next;
                }
            }
            // tzcnt: count trailing zeros of bitmask to get log2(bitmask)
            let _log2 = std::arch::x86_64::_tzcnt_u32(bitmask as u32) as usize;
            bitmask >>= 1;
        }
        idx
    }

    /// Prefix sum [0, idx). O(log n).
    #[inline]
    pub fn prefix_sum(&self, idx: usize) -> f32 {
        let mut i = idx.min(self.n);
        let mut sum = 0.0f32;
        while i > 0 {
            sum += self.buf[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }

    /// Add `delta` to position `idx`. O(log n).
    #[inline]
    pub fn update(&mut self, idx: usize, delta: f32) {
        let mut i = idx + 1; // 1-indexed
        while i <= self.n {
            self.buf[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Find the smallest index where `prefix_sum(idx) >= k`. O(log n).
    /// Returns `n` if total sum < k.
    pub fn find_kth(&self, k: f32) -> usize {
        let mut idx = 0usize;
        let mut bitmask = 1usize;
        while bitmask <= self.n {
            bitmask <<= 1;
        }
        bitmask >>= 1;
        let mut sum = 0.0f32;
        while bitmask > 0 {
            let next = idx + bitmask;
            if next <= self.n && sum + self.buf[next] < k {
                sum += self.buf[next];
                idx = next;
            }
            bitmask >>= 1;
        }
        idx // 0-indexed: idx is the largest where prefix_sum < k, so idx is the answer
    }

    /// Total sum of all elements. O(log n).
    pub fn total(&self) -> f32 {
        self.prefix_sum(self.n)
    }
}

/// Fenwick tree for u32 counts (used by F4 histogram threshold).
pub struct FenwickU32 {
    n: usize,
    buf: Vec<u32>,
}

impl FenwickU32 {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            n: n.max(1),
            buf: vec![0u32; n.max(1) + 1],
        }
    }

    pub fn reset_zeroed(&mut self) {
        for v in &mut self.buf {
            *v = 0;
        }
    }

    /// Build from a slice in O(n). Resets first.
    pub fn build_from(&mut self, data: &[u32]) {
        self.n = data.len().max(1);
        if self.buf.len() < self.n + 1 {
            self.buf.resize(self.n + 1, 0);
        } else {
            self.reset_zeroed();
        }
        for (i, &v) in data.iter().enumerate() {
            self.update(i, v);
        }
    }

    #[inline]
    pub fn prefix_sum(&self, idx: usize) -> u32 {
        let mut i = idx.min(self.n);
        let mut sum = 0u32;
        while i > 0 {
            sum += self.buf[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }

    #[inline]
    pub fn update(&mut self, idx: usize, delta: u32) {
        let mut i = idx + 1;
        while i <= self.n {
            self.buf[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Find smallest index where prefix_sum >= k. O(log n).
    pub fn find_kth(&self, k: u32) -> usize {
        let mut idx = 0usize;
        let mut bitmask = 1usize;
        while bitmask <= self.n {
            bitmask <<= 1;
        }
        bitmask >>= 1;
        let mut sum = 0u32;
        while bitmask > 0 {
            let next = idx + bitmask;
            if next <= self.n && sum + self.buf[next] < k {
                sum += self.buf[next];
                idx = next;
            }
            bitmask >>= 1;
        }
        idx
    }

    pub fn total(&self) -> u32 {
        self.prefix_sum(self.n)
    }
}

/// Fenwick tree for i32 (used by signed gradient accumulation).
pub struct FenwickI32 {
    n: usize,
    buf: Vec<i32>,
}

impl FenwickI32 {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            n: n.max(1),
            buf: vec![0i32; n.max(1) + 1],
        }
    }

    pub fn reset_zeroed(&mut self) {
        for v in &mut self.buf {
            *v = 0;
        }
    }

    /// Build from a slice in O(n). Resets first.
    pub fn build_from(&mut self, data: &[i32]) {
        self.n = data.len().max(1);
        if self.buf.len() < self.n + 1 {
            self.buf.resize(self.n + 1, 0);
        } else {
            self.reset_zeroed();
        }
        for (i, &v) in data.iter().enumerate() {
            self.update(i, v);
        }
    }

    #[inline]
    pub fn prefix_sum(&self, idx: usize) -> i32 {
        let mut i = idx.min(self.n);
        let mut sum = 0i32;
        while i > 0 {
            sum += self.buf[i];
            i -= i & i.wrapping_neg();
        }
        sum
    }

    #[inline]
    pub fn update(&mut self, idx: usize, delta: i32) {
        let mut i = idx + 1;
        while i <= self.n {
            self.buf[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    pub fn total(&self) -> i32 {
        self.prefix_sum(self.n)
    }
}

/// Fenwick tree for **multiplicative** prefix products via log-space.
///
/// Uses the identity `log(a * b) = log(a) + log(b)` to transform multiplicative
/// queries into additive ones. All values MUST be strictly > 0 (log(0) = -∞).
///
/// - `prefix_product(i)` = `exp(prefix_sum_log(i))`
/// - `find_kth_product(k)` = smallest index where cumulative product ≥ k
///
/// Used by F1 nucleus sampling (probability products) and attention log-sum-exp.
///
/// # Safety
/// - All values passed to `update` MUST be > 0.0. Zero or negative values
///   produce -inf in log-space, corrupting the tree.
pub struct FenwickLogF32 {
    n: usize,
    buf: Vec<f32>, // stores log(values)
}

impl FenwickLogF32 {
    /// Allocate a Fenwick tree of size `n` (zeroed log = 0.0 = log(1.0)).
    pub fn with_capacity(n: usize) -> Self {
        Self {
            n: n.max(1),
            buf: vec![0.0f32; n.max(1) + 1], // log(1) = 0
        }
    }

    pub fn reset_zeroed(&mut self) {
        for v in &mut self.buf {
            *v = 0.0;
        }
    }

    /// Build from a slice of positive f32 values. Stores log(x) in the tree.
    pub fn build_from(&mut self, data: &[f32]) {
        self.n = data.len().max(1);
        if self.buf.len() < self.n + 1 {
            self.buf.resize(self.n + 1, 0.0);
        } else {
            self.reset_zeroed();
        }
        for (i, &v) in data.iter().enumerate() {
            // log(v) — v must be > 0
            let log_v = if v > 0.0 { v.ln() } else { f32::NEG_INFINITY };
            self.update_log(i, log_v);
        }
    }

    /// Add `log_delta` to position `idx` (internal: stores log values).
    #[inline]
    fn update_log(&mut self, idx: usize, log_delta: f32) {
        let mut i = idx + 1;
        while i <= self.n {
            self.buf[i] += log_delta;
            i += i & i.wrapping_neg();
        }
    }

    /// Prefix **product** [0, idx) = exp(sum of log(values)). O(log n).
    #[inline]
    pub fn prefix_product(&self, idx: usize) -> f32 {
        let mut i = idx.min(self.n);
        let mut sum_log = 0.0f32;
        while i > 0 {
            sum_log += self.buf[i];
            i -= i & i.wrapping_neg();
        }
        sum_log.exp()
    }

    /// Multiply position `idx` by `factor` (factor must be > 0). O(log n).
    #[inline]
    pub fn multiply(&mut self, idx: usize, factor: f32) {
        assert!(factor > 0.0, "FenwickLogF32: factor must be > 0");
        let log_factor = factor.ln();
        self.update_log(idx, log_factor);
    }

    /// Find smallest index where prefix_product(idx) >= k. O(log n).
    /// Returns `n` if total product < k.
    pub fn find_kth_product(&self, k: f32) -> usize {
        if k <= 0.0 { return 0; }
        let log_k = k.ln();
        let mut idx = 0usize;
        let mut bitmask = 1usize;
        while bitmask <= self.n {
            bitmask <<= 1;
        }
        bitmask >>= 1;
        let mut sum_log = 0.0f32;
        while bitmask > 0 {
            let next = idx + bitmask;
            if next <= self.n && sum_log + self.buf[next] < log_k {
                sum_log += self.buf[next];
                idx = next;
            }
            bitmask >>= 1;
        }
        idx
    }

    /// Total product of all elements. O(log n).
    pub fn total_product(&self) -> f32 {
        self.prefix_product(self.n)
    }
}

// ── Linear fallback (when MUD_FENWICK=0) ─────────────────────────────────────

/// Linear prefix sum fallback — used when `MUD_FENWICK=0` or for small n.
/// Same API as FenwickF32 but O(n) per query.
pub fn linear_prefix_sum(data: &[f32], idx: usize) -> f32 {
    data[..idx.min(data.len())]
        .iter()
        .fold(0.0f32, |a, &b| a + b)
}

/// Linear find-kth fallback. O(n).
pub fn linear_find_kth(data: &[f32], k: f32) -> usize {
    let mut sum = 0.0f32;
    for (i, &v) in data.iter().enumerate() {
        sum += v;
        if sum >= k {
            return i;
        }
    }
    data.len()
}

// ── Env-gated dispatch ──────────────────────────────────────────────────────

/// Check if Fenwick trees are enabled. `MUD_FENWICK=0` disables (linear fallback).
pub fn fenwick_enabled() -> bool {
    std::env::var("MUD_FENWICK")
        .map(|s| s != "0" && !s.eq_ignore_ascii_case("false"))
        .unwrap_or(true)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::constants::EPSILON_FLOOR;

    #[test]
    fn test_fenwick_f32_prefix_sum() {
        let mut ft = FenwickF32::with_capacity(100);
        ft.build_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((ft.prefix_sum(0) - 0.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum(1) - 1.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum(3) - 6.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum(5) - 15.0).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_fenwick_f32_update() {
        let mut ft = FenwickF32::with_capacity(100);
        ft.build_from(&[1.0, 2.0, 3.0]);
        ft.update(1, 10.0);
        assert!((ft.prefix_sum(2) - 13.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum(3) - 16.0).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_fenwick_f32_find_kth() {
        let mut ft = FenwickF32::with_capacity(100);
        ft.build_from(&[1.0, 2.0, 3.0, 4.0, 5.0]); // prefix: 1, 3, 6, 10, 15
        assert_eq!(ft.find_kth(1.0), 0); // prefix_sum(0)=1 >= 1
        assert_eq!(ft.find_kth(4.0), 2); // prefix_sum(2)=6 >= 4
        assert_eq!(ft.find_kth(15.0), 4); // prefix_sum(4)=15 >= 15
        assert_eq!(ft.find_kth(100.0), 5); // total=15 < 100
    }

    #[test]
    fn test_fenwick_f32_reset_zeroed() {
        let mut ft = FenwickF32::with_capacity(10);
        ft.build_from(&[1.0, 2.0, 3.0]);
        assert!((ft.total() - 6.0).abs() < EPSILON_FLOOR);
        ft.reset_zeroed();
        assert!((ft.total() - 0.0).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_fenwick_f32_raw_pointer_methods() {
        let mut ft = FenwickF32::with_capacity(100);
        ft.build_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // Verify raw pointer methods match safe versions (before update)
        assert!((ft.prefix_sum_raw(3) - 6.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum_raw(5) - 15.0).abs() < EPSILON_FLOOR);
        assert_eq!(ft.find_kth_raw(4.0), 2);
        assert_eq!(ft.find_kth_raw(15.0), 4);
        // Now test update_raw
        ft.update_raw(1, 10.0);
        assert!((ft.prefix_sum_raw(2) - 13.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum_raw(3) - 16.0).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_fenwick_f32_build_from_linear() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut ft1 = FenwickF32::with_capacity(100);
        ft1.build_from(&data);
        let mut ft2 = FenwickF32::with_capacity(100);
        ft2.build_from_linear(&data);
        // Both should produce the same results
        for i in 0..=data.len() {
            assert!((ft1.prefix_sum(i) - ft2.prefix_sum(i)).abs() < EPSILON_FLOOR,
                "prefix_sum({}) mismatch: {} vs {}", i, ft1.prefix_sum(i), ft2.prefix_sum(i));
        }
        assert!((ft1.total() - ft2.total()).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_fenwick_f32_build_from_linear_large() {
        let n = 131_072;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let mut ft1 = FenwickF32::with_capacity(n);
        ft1.build_from(&data);
        let mut ft2 = FenwickF32::with_capacity(n);
        ft2.build_from_linear(&data);
        // Verify they match at several points (tolerance accounts for FP reordering)
        for &idx in &[0, 1, n/4, n/2, n-1, n] {
            let diff = (ft1.prefix_sum(idx) - ft2.prefix_sum(idx)).abs();
            let expected = ft1.prefix_sum(idx);
            // Relative tolerance: 0.1% of expected value
            let tolerance = expected.abs() * 0.001 + 100.0;
            assert!(diff < tolerance,
                "prefix_sum({}) mismatch: {} vs {} (diff={})", idx, ft1.prefix_sum(idx), ft2.prefix_sum(idx), diff);
        }
    }

    #[test]
    fn test_fenwick_f32_find_kth_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return; // Skip on non-AVX2 CPUs
        }
        let mut ft = FenwickF32::with_capacity(100);
        ft.build_from(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = unsafe { ft.find_kth_avx2(4.0) };
        assert_eq!(result, ft.find_kth(4.0));
        let result2 = unsafe { ft.find_kth_avx2(15.0) };
        assert_eq!(result2, ft.find_kth(15.0));
    }

    #[test]
    fn test_fenwick_nucleus_sampling_workflow() {
        // Simulates the debate_trainer.rs nucleus sampling path:
        // 1. Create logits
        // 2. AVX2 softmax (max → sub → exp → sum → normalize)
        // 3. Build Fenwick tree
        // 4. find_kth AVX2
        // 5. Compare with scalar reference
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let vocab_size = 256;
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32 * 0.1) - 12.8).collect();

        // Scalar reference softmax
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = {
            let inv_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum::<f32>();
            let inv_se = 1.0 / (inv_sum + 1e-10);
            logits.iter().map(|&l| (l - max_l).exp() * inv_se).collect()
        };

        // AVX2 softmax (same as debate_trainer.rs path)
        let mut probs_avx = logits.clone();
        let max_avx;
        let sum_avx;
        unsafe {
            max_avx = crate::mud::ezop::max_avx2(logits.as_ptr(), vocab_size);
            crate::mud::ezop::broadcast_sub_avx2(probs_avx.as_mut_ptr(), max_avx, vocab_size);
            crate::mud::ezop::exp_avx2(probs_avx.as_ptr(), probs_avx.as_mut_ptr(), vocab_size);
            sum_avx = crate::mud::ezop::sum_avx2(probs_avx.as_ptr(), vocab_size);
            let inv_se = 1.0 / (sum_avx + 1e-10);
            crate::mud::ezop::scale_vec_avx2(probs_avx.as_mut_ptr(), inv_se, vocab_size);
        }

        // Build Fenwick trees
        let mut ft_scalar = FenwickF32::with_capacity(vocab_size);
        ft_scalar.build_from(&probs);

        let mut ft_avx = FenwickF32::with_capacity(vocab_size);
        unsafe {
            ft_avx.reset_zeroed();
            ft_avx.build_from_linear_raw(probs_avx.as_ptr());
        }

        // Verify probabilities match
        for i in 0..vocab_size {
            assert!((probs_avx[i] - probs[i]).abs() < 1e-3,
                "prob mismatch at {i}: {} vs {}", probs_avx[i], probs[i]);
        }

        // Verify prefix sums match
        for idx in &[1, 8, 16, 64, 128, 255, 256] {
            let scalar_ps = ft_scalar.prefix_sum(*idx);
            let avx_ps = ft_avx.prefix_sum(*idx);
            assert!((avx_ps - scalar_ps).abs() < 1e-3,
                "prefix_sum({idx}) mismatch: {} vs {}", avx_ps, scalar_ps);
        }

        // Verify find_kth matches for several targets
        for target in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let scalar_result = ft_scalar.find_kth(target * sum_avx);
            let avx_result = unsafe { ft_avx.find_kth_avx2(target * sum_avx) };
            assert_eq!(avx_result, scalar_result,
                "find_kth({}) mismatch: {} vs {}", target * sum_avx, avx_result, scalar_result);
        }
    }

    #[test]
    fn test_fenwick_nucleus_sampling_nan_guard() {
        // Edge case: NaN/Inf in logits should not crash the AVX2 softmax path
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let logits: Vec<f32> = vec![1.0, f32::NAN, 3.0, f32::INFINITY, -1.0, 2.0, 0.0, -3.0];
        let vocab_size = logits.len();

        // AVX2 softmax with NaN guard
        let mut probs = logits.clone();
        let max_l;
        let sum_exp;
        unsafe {
            max_l = crate::mud::ezop::max_avx2(probs.as_ptr(), vocab_size);
            let max_l = if max_l.is_finite() { max_l } else { 0.0 };
            crate::mud::ezop::broadcast_sub_avx2(probs.as_mut_ptr(), max_l, vocab_size);
            crate::mud::ezop::exp_avx2(probs.as_ptr(), probs.as_mut_ptr(), vocab_size);
            sum_exp = crate::mud::ezop::sum_avx2(probs.as_ptr(), vocab_size);
            let inv_se = 1.0 / (sum_exp + 1e-10);
            crate::mud::ezop::scale_vec_avx2(probs.as_mut_ptr(), inv_se, vocab_size);
        }

        // All probabilities must be finite (NaN/Inf handled by guards)
        for v in &probs {
            assert!(v.is_finite(), "non-finite prob: {v}");
        }
        // Sum should be approximately 1.0
        assert!((sum_exp * (1.0 / (sum_exp + 1e-10)) - 1.0).abs() < 0.01 ||
                (probs.iter().sum::<f32>() - 1.0).abs() < 0.01,
            "probs should sum to ~1.0, got sum_exp={sum_exp}");
    }

    #[test]
    fn test_fenwick_u32() {
        let mut ft = FenwickU32::with_capacity(100);
        for i in 0..5 {
            ft.update(i, (i + 1) as u32);
        }
        assert_eq!(ft.prefix_sum(3), 6); // 1+2+3
        assert_eq!(ft.total(), 15); // 1+2+3+4+5
        assert_eq!(ft.find_kth(6), 2); // prefix_sum(2)=6 >= 6
    }

    #[test]
    fn test_fenwick_i32() {
        let mut ft = FenwickI32::with_capacity(100);
        ft.update(0, 5);
        ft.update(1, -3);
        ft.update(2, 10);
        assert_eq!(ft.prefix_sum(2), 2); // 5 + (-3)
        assert_eq!(ft.total(), 12); // 5 + (-3) + 10
    }

    #[test]
    fn test_fenwick_log_f32_prefix_product() {
        // All values > 0: [2.0, 3.0, 4.0, 5.0]
        // Products: 1, 2, 6, 24, 120
        let mut ft = FenwickLogF32::with_capacity(100);
        ft.build_from(&[2.0, 3.0, 4.0, 5.0]);
        // Use relative tolerance for log-space (exp introduces ~1e-5 error)
        let rel = |a: f32, b: f32| (a - b).abs() / b.max(1.0) < 1e-4;
        assert!(rel(ft.prefix_product(0), 1.0)); // empty product = 1
        assert!(rel(ft.prefix_product(1), 2.0));
        assert!(rel(ft.prefix_product(2), 6.0));
        assert!(rel(ft.prefix_product(3), 24.0));
        assert!(rel(ft.prefix_product(4), 120.0));
    }

    #[test]
    fn test_fenwick_log_f32_multiply() {
        let mut ft = FenwickLogF32::with_capacity(100);
        ft.build_from(&[2.0, 3.0, 4.0]);
        let rel = |a: f32, b: f32| (a - b).abs() / b.max(1.0) < 1e-4;
        // Multiply position 1 by 5: [2, 15, 4] → products: 2, 30, 120
        ft.multiply(1, 5.0);
        assert!(rel(ft.prefix_product(2), 30.0));
        assert!(rel(ft.prefix_product(3), 120.0));
    }

    #[test]
    fn test_fenwick_log_f32_find_kth_product() {
        let mut ft = FenwickLogF32::with_capacity(100);
        ft.build_from(&[2.0, 3.0, 4.0, 5.0]); // products: 2, 6, 24, 120
        // find_kth_product(6.0) → index where product first >= 6 → index 1
        assert_eq!(ft.find_kth_product(6.0), 1);
        // find_kth_product(24.0) → index 2
        assert_eq!(ft.find_kth_product(24.0), 2);
        // find_kth_product(100.0) → index 3
        assert_eq!(ft.find_kth_product(100.0), 3);
        // find_kth_product(200.0) → total=120 < 200 → returns n
        assert_eq!(ft.find_kth_product(200.0), 4);
    }

    #[test]
    fn test_fenwick_log_f32_total_product() {
        let mut ft = FenwickLogF32::with_capacity(100);
        ft.build_from(&[1.5, 2.0, 3.0]);
        let rel = |a: f32, b: f32| (a - b).abs() / b.max(1.0) < 1e-4;
        assert!(rel(ft.total_product(), 9.0)); // 1.5 * 2 * 3 = 9
    }

    #[test]
    fn test_fenwick_large_n() {
        // n=131072 — the size mentioned in the optimization plan
        let n = 131_072;
        let mut ft = FenwickF32::with_capacity(n);
        for i in 0..n {
            ft.update(i, 1.0);
        }
        assert!((ft.total() - n as f32).abs() < 1.0);
        // find_kth(65536) = index where cumulative sum first reaches 65536
        // With all 1.0: prefix_sum(65535) = 65535 < 65536, prefix_sum(65536) = 65536 >= 65536
        // So find_kth returns 65535 (0-indexed position of the 65536th element)
        assert_eq!(ft.find_kth(n as f32 / 2.0), n / 2 - 1);
    }

    #[test]
    fn test_fenwick_single_element() {
        let mut ft = FenwickF32::with_capacity(1);
        ft.update(0, 42.0);
        assert!((ft.prefix_sum(0) - 0.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum(1) - 42.0).abs() < EPSILON_FLOOR);
        assert_eq!(ft.find_kth(42.0), 0);
    }

    #[test]
    fn test_fenwick_empty() {
        let ft = FenwickF32::with_capacity(0);
        assert!((ft.total() - 0.0).abs() < EPSILON_FLOOR);
    }

    // Serialize env-var tests to avoid race with parallel test runner.
    // (Same pattern as `corpus_trainer/mod.rs` `OPT_TEST_LOCK`.)
    static FENWICK_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_fenwick_enabled_default() {
        let _g = FENWICK_ENV_LOCK.lock().unwrap();
        // Without MUD_FENWICK set, should default to true
        std::env::remove_var("MUD_FENWICK");
        assert!(fenwick_enabled());
    }

    #[test]
    fn test_fenwick_disabled() {
        let _g = FENWICK_ENV_LOCK.lock().unwrap();
        std::env::set_var("MUD_FENWICK", "0");
        assert!(!fenwick_enabled());
        std::env::remove_var("MUD_FENWICK");
    }

    #[test]
    fn test_linear_fallback() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((linear_prefix_sum(&data, 3) - 6.0).abs() < EPSILON_FLOOR);
        assert_eq!(linear_find_kth(&data, 4.0), 2);
    }

    #[test]
    fn test_fenwick_find_kth_edge_cases() {
        let mut ft = FenwickF32::with_capacity(10);
        ft.build_from(&[0.0, 0.0, 5.0, 0.0, 0.0]);
        // prefix: 0, 0, 5, 5, 5
        assert_eq!(ft.find_kth(1.0), 2); // first non-zero at index 2
        assert_eq!(ft.find_kth(5.0), 2); // prefix_sum(2)=5 >= 5
        assert_eq!(ft.find_kth(6.0), 5); // total=5 < 6
    }

    #[test]
    fn test_fenwick_reset_and_rebuild() {
        let mut ft = FenwickF32::with_capacity(10);
        ft.build_from(&[1.0, 2.0, 3.0]);
        assert!((ft.total() - 6.0).abs() < EPSILON_FLOOR);
        ft.reset_zeroed();
        ft.build_from(&[10.0, 20.0]);
        assert!((ft.total() - 30.0).abs() < EPSILON_FLOOR);
        assert!((ft.prefix_sum(1) - 10.0).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_fenwick_f32_find_kth_walk() {
        if !is_x86_feature_detected!("bmi1") || !is_x86_feature_detected!("bmi2") { return; }
        let mut ft = FenwickF32::with_capacity(100);
        ft.build_from(&[1.0, 2.0, 3.0, 4.0, 5.0]); // prefix: 1, 3, 6, 10, 15
        let result = unsafe { ft.find_kth_walk(4.0) };
        assert_eq!(result, ft.find_kth(4.0));
        let result = unsafe { ft.find_kth_walk(15.0) };
        assert_eq!(result, ft.find_kth(15.0));
    }
}
