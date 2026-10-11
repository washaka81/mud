//! Memory Continuity Table (RAM Flat Pointer Table for 1-Cycle O(1) Dereferencing)
//! Respects P-00 (Pointer Mastery) and P-01 (Zero Allocation on Hot Path).

use std::sync::Arc;
use crate::mud::MudFile;

/// Cache-line aligned (64 bytes) flat memory pointer entry.
#[repr(C, align(64))]
pub struct MemoryPointerEntry {
    pub module_id: u8,         // Module ID (0..255)
    pub layer_idx: u16,        // Layer index
    pub role_id: u8,           // Tensor role ID (Q, K, V, O, Up, Gate, Down)
    pub active: u8,            // 1 if active, 0 if empty
    pub _padding: [u8; 3],     // Alignment padding
    pub weight_ptr: *const u8, // Raw pointer to weight data
    pub scale_ptr: *const f32, // Raw pointer to PRQ scales
    pub rows: u32,
    pub cols: u32,
    pub data_len: u64,
}

// SAFETY: Pointers reference mmap-backed read-only memory shared across threads.
unsafe impl Send for MemoryPointerEntry {}
unsafe impl Sync for MemoryPointerEntry {}

impl Default for MemoryPointerEntry {
    fn default() -> Self {
        Self {
            module_id: 0,
            layer_idx: 0,
            role_id: 0,
            active: 0,
            _padding: [0; 3],
            weight_ptr: std::ptr::null(),
            scale_ptr: std::ptr::null(),
            rows: 0,
            cols: 0,
            data_len: 0,
        }
    }
}

/// 256-Slot Flat RAM Memory Continuity Table.
pub struct MemoryContinuityTable {
    pub entries: Box<[MemoryPointerEntry; 256]>,
    pub active_modules: Vec<Arc<MudFile>>,
}

impl MemoryContinuityTable {
    pub fn new() -> Self {
        let entries = Box::new(std::array::from_fn(|_| MemoryPointerEntry::default()));
        Self {
            entries,
            active_modules: Vec::new(),
        }
    }
}

impl Default for MemoryContinuityTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryContinuityTable {
    /// Seats a mounted MudFile module into a specified slot tag (0..255).
    pub fn seat_module(&mut self, slot_id: u8, mud_file: Arc<MudFile>) -> bool {
        let idx = slot_id as usize;
        let entry = &mut self.entries[idx];

        if let Some(skill) = mud_file.skills.get("default").or_else(|| mud_file.skills.values().next()) {
            if let Some(t_embd) = skill.tensors.get("token_embd.weight") {
                entry.module_id = slot_id;
                entry.layer_idx = 0;
                entry.role_id = 0;
                entry.active = 1;
                entry.weight_ptr = t_embd.data_ptr;
                entry.rows = t_embd.shape.first().copied().unwrap_or(0) as u32;
                entry.cols = t_embd.shape.get(1).copied().unwrap_or(0) as u32;
                entry.data_len = t_embd.data_size() as u64;
            } else {
                entry.module_id = slot_id;
                entry.active = 1;
            }
        } else {
            entry.module_id = slot_id;
            entry.active = 1;
        }

        self.active_modules.push(mud_file);
        true
    }

    /// Unseats a module from a slot tag.
    pub fn unseat_module(&mut self, slot_id: u8) {
        let idx = slot_id as usize;
        self.entries[idx] = MemoryPointerEntry::default();
    }

    /// Direct 1-cycle O(1) pointer dereferencing for a given slot_id.
    #[inline(always)]
    pub fn get_entry(&self, slot_id: u8) -> Option<&MemoryPointerEntry> {
        let entry = &self.entries[slot_id as usize];
        if entry.active == 1 {
            Some(entry)
        } else {
            None
        }
    }
}

// ── SPRINT-2: helper that bridges SlimeX cells ↔ Continuity Table. ──────────
//
// The SlimeXCellV5 packs an 8-bit `address_tag` (bits 8..15) that indexes directly
// into the 256-slot table. This helper is the production read path: given a packed
// `u32` cell, extract the tag and resolve the corresponding `MemoryPointerEntry`.
//
// Used in two flows:
//   1. Expert dispatch in MoE FFN (per-dim cell → which expert's weights?)
//   2. Memory-augmented forward when submodels are mounted (per-token cell).
//
// Lookup latency: 1 atomic load + 1 bounds check + 1 branch (all L1d-resident).

/// Extract the 8-bit address tag from a packed `SlimeXCellV5` cell.
#[inline(always)]
pub const fn slimex_address_tag(cell: u32) -> u8 {
    ((cell >> 8) & 0xFF) as u8
}

/// Extract the 8-bit FP8 accumulator from a packed `SlimeXCellV5` cell.
#[inline(always)]
pub const fn slimex_accum_fp8(cell: u32) -> u8 {
    (cell & 0xFF) as u8
}

/// Extract the 8-bit STE parity from a packed `SlimeXCellV5` cell.
#[inline(always)]
pub const fn slimex_ste_parity(cell: u32) -> u8 {
    ((cell >> 24) & 0xFF) as u8
}

/// Resolve a packed cell to a Continuity Table entry (1-cycle O(1) lookup).
#[inline]
pub fn resolve_cell(
    cell: u32,
    table: &MemoryContinuityTable,
) -> Option<&MemoryPointerEntry> {
    table.get_entry(slimex_address_tag(cell))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_memory_pointer_entry_alignment() {
        assert_eq!(std::mem::align_of::<MemoryPointerEntry>(), 64);
    }

    #[test]
    fn test_continuity_table_seating() {
        let mut table = MemoryContinuityTable::new();
        assert!(table.get_entry(42).is_none());

        let mud_file = Arc::new(MudFile {
            mmap: None,
            skills: HashMap::new(),
            global_metadata: HashMap::new(),
            path: None,
        });

        table.seat_module(42, mud_file);
        let entry = table.get_entry(42);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().module_id, 42);
        assert_eq!(entry.unwrap().active, 1);

        table.unseat_module(42);
        assert!(table.get_entry(42).is_none());
    }

    // ── SPRINT-2: cell-extraction + resolve_cell tests ─────────────────────

    #[test]
    fn test_slimex_address_tag_extraction() {
        // Byte 1 (bits 8..15) = address_tag.
        // Cell layout: [accum_fp8=0xAB | tag=0x42<<8 | jepa=(-12)<<16 | ste=0x01<<24]
        let cell: u32 = 0xAB | (0x42u32 << 8) | ((((-12i8) as u8) as u32) << 16) | (0x01u32 << 24);
        assert_eq!(slimex_address_tag(cell), 0x42);
        assert_eq!(slimex_accum_fp8(cell), 0xAB);
        assert_eq!(slimex_ste_parity(cell), 0x01);
    }

    #[test]
    fn test_resolve_cell_inactive_returns_none() {
        let table = MemoryContinuityTable::new();
        // Any cell → tag → inactive slot → None.
        let cell: u32 = (5u8 as u32) | ((7u8 as u32) << 8);
        assert!(resolve_cell(cell, &table).is_none());
    }

    #[test]
    fn test_resolve_cell_active_returns_entry() {
        let mut table = MemoryContinuityTable::new();
        let mud_file = Arc::new(MudFile {
            mmap: None,
            skills: HashMap::new(),
            global_metadata: HashMap::new(),
            path: None,
        });
        table.seat_module(7, mud_file);
        // Cell with tag=7.
        let cell: u32 = (0u8 as u32) | ((7u8 as u32) << 8);
        let entry = resolve_cell(cell, &table);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().module_id, 7);
    }
}
