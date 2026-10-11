//! SlimeX CellV5 — ultra-compressed 8:8:8:8 u32 transport cell.
//!
//! P-08: `ShadowExpertBus`, `SlimeXSlot`, and `unpack_mud_tensor` removed as
//! dead code (allocated but never wired into backward; mount/unmount/evict
//! never called from any training path). `SlimeXCellV5` is retained — it is
//! the cell format used by `slime.rs` SX-03 `snapshot_to_slimex()`.

/// SlimeX Master Cell v5 (32-bit u32 transport cell: 8:8:8:8 ratio).
///
/// Layout (per AGENTS.md SX-01):
/// - bits  0..8:  `accum_fp8`  — FP8 accumulator (L-08 NaN guard applied upstream)
/// - bits  8..16: `address_tag` — 8-bit slot index for Memory Continuity Table
/// - bits 16..24: `jepa_velocity` — signed i8 OU velocity (JEPA integral slope)
/// - bits 24..32: `ste_parity`    — u8 STE parity byte (P-02 f16 dual packing tag)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SlimeXCellV5 {
    pub raw: u32,
}

impl SlimeXCellV5 {
    #[inline(always)]
    pub fn pack(accum_fp8: u8, address_tag: u8, jepa_velocity: i8, ste_parity: u8) -> Self {
        Self {
            raw: (accum_fp8 as u32)
                | ((address_tag as u32) << 8)
                | (((jepa_velocity as u8) as u32) << 16)
                | ((ste_parity as u32) << 24),
        }
    }

    #[inline(always)]
    pub fn accum_fp8(&self) -> u8 {
        (self.raw & 0xFF) as u8
    }

    #[inline(always)]
    pub fn address_tag(&self) -> u8 {
        ((self.raw >> 8) & 0xFF) as u8
    }

    #[inline(always)]
    pub fn jepa_velocity(&self) -> i8 {
        ((self.raw >> 16) & 0xFF) as i8
    }

    #[inline(always)]
    pub fn ste_parity(&self) -> u8 {
        ((self.raw >> 24) & 0xFF) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slimex_cell_v5_roundtrip() {
        let accum_fp8 = 0xABu8;
        let address_tag = 0x42u8; // Slot 66
        let jepa_velocity = -12i8;
        let ste_parity = 0x01u8;

        let cell = SlimeXCellV5::pack(accum_fp8, address_tag, jepa_velocity, ste_parity);
        assert_eq!(cell.accum_fp8(), accum_fp8);
        assert_eq!(cell.address_tag(), address_tag);
        assert_eq!(cell.jepa_velocity(), jepa_velocity);
        assert_eq!(cell.ste_parity(), ste_parity);
    }

    #[test]
    fn test_slimex_cell_v5_equi_byte_partition() {
        let cell = SlimeXCellV5::pack(255, 128, 127, 200);
        assert_eq!(cell.accum_fp8(), 255);
        assert_eq!(cell.address_tag(), 128);
        assert_eq!(cell.jepa_velocity(), 127);
        assert_eq!(cell.ste_parity(), 200);

        let cell_min = SlimeXCellV5::pack(0, 0, -128, 0);
        assert_eq!(cell_min.accum_fp8(), 0);
        assert_eq!(cell_min.address_tag(), 0);
        assert_eq!(cell_min.jepa_velocity(), -128);
        assert_eq!(cell_min.ste_parity(), 0);
    }
}
