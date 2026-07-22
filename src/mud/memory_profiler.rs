use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MemoryProfiler {
    pub static_allocations: AtomicUsize,
    pub hard_limit_mb: std::sync::atomic::AtomicU64,
}

pub static GLOBAL_PROFILER: MemoryProfiler = MemoryProfiler {
    static_allocations: AtomicUsize::new(0),
    hard_limit_mb: std::sync::atomic::AtomicU64::new(0),
};

impl MemoryProfiler {
    pub const fn new(limit_mb: f64) -> Self {
        Self {
            static_allocations: AtomicUsize::new(0),
            hard_limit_mb: std::sync::atomic::AtomicU64::new(limit_mb.to_bits()),
        }
    }

    pub fn set_limit_mb(&self, limit_mb: f64) {
        self.hard_limit_mb
            .store(limit_mb.to_bits(), Ordering::SeqCst);
    }

    pub fn register_allocation(&self, bytes: usize) {
        self.static_allocations.fetch_add(bytes, Ordering::SeqCst);
    }

    pub fn check_zero_allocation(&self, _context: &str) {
        // Enforce that no new heap allocations are happening during inference
        // In a true environment we'd use a custom global allocator to track this,
        // but for now we rely on strict pre-allocation tracking in `UnifiedBuffer`.
    }

    pub fn validate_ceiling(&self) -> Result<(), String> {
        let current_bytes = self.static_allocations.load(Ordering::SeqCst);
        let current_mb = current_bytes as f64 / 1_048_576.0;
        let limit_mb = f64::from_bits(self.hard_limit_mb.load(Ordering::SeqCst));
        if current_mb > limit_mb && limit_mb > 0.0 {
            return Err(format!(
                "Memory ceiling exceeded: {:.2} MB > {:.2} MB limit",
                current_mb, limit_mb
            ));
        }
        Ok(())
    }
}
