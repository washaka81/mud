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

/// Generic 64-byte aligned raw-pointer array replacing `Vec<T>` in `SlimeWorkspace`.
/// Provides zero-cost `Deref`/`DerefMut` to `[T]`, unchecked indexing helpers, zero allocations in hot path.
pub struct AlignedArray<T> {
    pub ptr: *mut T,
    layout: std::alloc::Layout,
    pub len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> AlignedArray<T> {
    pub fn new(size: usize, align: usize) -> Self {
        let align_effective = align.max(std::mem::align_of::<T>()).max(1);
        if size == 0 {
            let layout = std::alloc::Layout::from_size_align(0, align_effective).unwrap();
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                layout,
                len: 0,
                _phantom: std::marker::PhantomData,
            };
        }
        let byte_size = size
            .checked_mul(std::mem::size_of::<T>())
            .expect("AlignedArray size overflow");
        let layout = std::alloc::Layout::from_size_align(byte_size, align_effective)
            .expect("AlignedArray: invalid layout");
        // SAFETY: layout is non-zero byte_size; alloc_zeroed returns valid pointer or null
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut T };
        assert!(
            !ptr.is_null(),
            "AlignedArray: alloc_zeroed returned null (OOM, {} bytes)",
            byte_size
        );
        Self {
            ptr,
            layout,
            len: size,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn from_elem(elem: T, size: usize) -> Self
    where
        T: Clone,
    {
        let mut arr = Self::new(size, 64);
        for item in arr.as_mut_slice() {
            *item = elem.clone();
        }
        arr
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let mut arr = Self::new(slice.len(), 64);
        arr.as_mut_slice().copy_from_slice(slice);
        arr
    }

    /// # Safety
    /// Caller must ensure `i < self.len` and `self.ptr` is valid and non-null.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, i: usize) -> T
    where
        T: Copy,
    {
        *self.ptr.add(i)
    }

    /// # Safety
    /// Caller must ensure `i < self.len` and `self.ptr` is valid, non-null, and writeable.
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, i: usize, val: T) {
        self.ptr.add(i).write(val);
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: self.ptr is valid and points to self.len initialized elements
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: self.ptr is valid and points to self.len initialized elements
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> std::ops::Deref for AlignedArray<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for AlignedArray<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T> Drop for AlignedArray<T> {
    fn drop(&mut self) {
        if self.len > 0 && !self.ptr.is_null() {
            unsafe {
                std::alloc::dealloc(self.ptr as *mut u8, self.layout);
            }
        }
    }
}

impl<T: Clone> Clone for AlignedArray<T> {
    fn clone(&self) -> Self {
        let mut new_arr = Self::new(self.len, self.layout.align());
        new_arr.as_mut_slice().clone_from_slice(self.as_slice());
        new_arr
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for AlignedArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

// SAFETY (SEC-H3 audit 2026-07-27): AlignedArray owns its heap allocation
// (allocated in `new`, freed in `Drop` with the same `Layout`). The bound
// `T: Send` (resp. `T: Sync`) delegates to T's thread-safety: if T is Send,
// the owned buffer of T can be moved across threads; if T is Sync, the buffer
// can be shared via `&AlignedArray<T>` because all mutation goes through
// `&mut self` (DerefMut requires `&mut`). There is no `UnsafeCell` — the raw
// `ptr` field is only accessed through `as_slice`/`as_mut_slice` which take
// `&self`/`&mut self` respectively, enforcing the aliasing rules at the type
// level. Therefore the conditional Send+Sync impls are sound.
unsafe impl<T: Send> Send for AlignedArray<T> {}
unsafe impl<T: Sync> Sync for AlignedArray<T> {}

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

    #[test]
    fn test_aligned_array_ops() {
        let mut arr = AlignedArray::<f32>::new(16, 64);
        assert_eq!(arr.len(), 16);
        assert_eq!(arr.ptr as usize % 64, 0);

        for i in 0..16 {
            arr[i] = i as f32 * 2.0;
        }

        for i in 0..16 {
            unsafe {
                assert_eq!(arr.get_unchecked(i), i as f32 * 2.0);
            }
        }

        let arr_cloned = arr.clone();
        assert_eq!(arr_cloned.len(), 16);
        assert_eq!(arr_cloned[5], 10.0);

        let empty = AlignedArray::<u8>::new(0, 64);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.as_slice(), &[] as &[u8]);
    }
}
