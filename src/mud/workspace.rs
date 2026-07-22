//! Pre-allocated buffer primitives for zero-alloc hot paths.
//!
//! **L-03 (2026-07-17):** Removed dead `InferenceWorkspace` / `ExpertWorkspace` /
//! `SamplingConfig` / `TokenizerBuffer` (~240 LOC, never instantiated). Live
//! inference/training uses `SlimeWorkspace` (`slime.rs`). This module keeps
//! `UnifiedBuffer` / `AlignedBuffer` for LDT and related callers.

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

// P-13 / L-07: single definition in constants.rs
pub use crate::mud::constants::EPSILON_FLOOR;

pub struct AlignedBuffer {
    pub ptr: *mut f32,
    layout: std::alloc::Layout,
    pub len: usize,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        let byte_size = size.checked_mul(4).expect("AlignedBuffer size overflow");
        let layout = std::alloc::Layout::from_size_align(byte_size, 64)
            .expect("AlignedBuffer: invalid layout");
        // SAFETY: layout was computed from valid size/align; alloc_zeroed returns null on OOM (checked below)
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        assert!(
            !ptr.is_null(),
            "AlignedBuffer: alloc_zeroed returned null (OOM, {} bytes)",
            size * 4
        );
        Self {
            ptr,
            layout,
            len: size,
        }
    }
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: self.ptr is valid, non-null, aligned for f32, and points to `self.len` elements
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: self.ptr is valid and non-null; no other reference aliases this region because &mut self is exclusive
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl std::ops::Deref for AlignedBuffer {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was allocated by alloc_zeroed with the same layout in new(); this is the unique Drop call
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

// SAFETY: AlignedBuffer owns its heap allocation; Send/Sync are safe because mutation goes through &mut
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

/// # Soundness
/// `Cpu(RwLock<AlignedBuffer>)` provides safe interior mutability:
/// `read()` and `write()` both take `&self`, but the RwLock enforces
/// mutual exclusion at runtime, preventing aliased mutable references.
pub enum UnifiedBuffer {
    Cpu(RwLock<AlignedBuffer>),
}

pub enum UnifiedReadGuard<'a> {
    Cpu(RwLockReadGuard<'a, AlignedBuffer>),
}

impl std::ops::Deref for UnifiedReadGuard<'_> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            UnifiedReadGuard::Cpu(g) => g.deref(),
        }
    }
}

pub enum UnifiedWriteGuard<'a> {
    Cpu(RwLockWriteGuard<'a, AlignedBuffer>),
}

impl std::ops::Deref for UnifiedWriteGuard<'_> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            UnifiedWriteGuard::Cpu(g) => g.deref(),
        }
    }
}

impl std::ops::DerefMut for UnifiedWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            UnifiedWriteGuard::Cpu(g) => g.deref_mut(),
        }
    }
}

impl UnifiedBuffer {
    pub fn new_cpu(size: usize) -> Self {
        crate::mud::memory_profiler::GLOBAL_PROFILER.register_allocation(size * 4); // f32
        UnifiedBuffer::Cpu(RwLock::new(AlignedBuffer::new(size)))
    }

    pub fn new_cpu_from_slice(slice: &[f32]) -> Self {
        crate::mud::memory_profiler::GLOBAL_PROFILER.register_allocation(slice.len() * 4); // f32
        let mut buf = AlignedBuffer::new(slice.len());
        buf.as_mut_slice().copy_from_slice(slice);
        UnifiedBuffer::Cpu(RwLock::new(buf))
    }

    pub fn read(&self) -> UnifiedReadGuard<'_> {
        match self {
            UnifiedBuffer::Cpu(b) => UnifiedReadGuard::Cpu(b.read()),
        }
    }

    pub fn write(&self) -> UnifiedWriteGuard<'_> {
        match self {
            UnifiedBuffer::Cpu(b) => UnifiedWriteGuard::Cpu(b.write()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            UnifiedBuffer::Cpu(b) => b.read().len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn fill(&self, val: f32) {
        self.write().fill(val);
    }
}

// SAFETY: RwLock<AlignedBuffer> is Send+Sync because AlignedBuffer is Send+Sync.
unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_buffer_roundtrip() {
        let buf = UnifiedBuffer::new_cpu(16);
        assert_eq!(buf.len(), 16);
        {
            let mut w = buf.write();
            for (i, v) in w.iter_mut().enumerate() {
                *v = i as f32;
            }
        }
        {
            let r = buf.read();
            assert!((r[3] - 3.0).abs() < 1e-6);
        }
        // Drop read guard before write (RwLock would otherwise deadlock)
        buf.fill(0.0);
        {
            let r = buf.read();
            assert!((r[3]).abs() < 1e-6);
        }
    }
}
