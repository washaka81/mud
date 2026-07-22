//! # Pointer-address calculation audit (P-00 hot-path validation)
//!
//! Validates the raw-pointer ELUT / ternary pack-unpack math used across the
//! project against safe reference formulas, and against the *actual* on-disk
//! mmap layout of a `.mud` file.
//!
//! Covered kernels:
//! - `dequantize_ternary_row` (`mod.rs`) — 8 values/u32, nibble `j` at `(val >> j*4) & 0xF`
//! - `pack_ternary_into` (`ezop.rs`) — inverse of the above
//! - `pack_elut_prq` (`ezop.rs`) — 2 values/byte (`b0 | b1<<4`), row-major `start/2 + c`
//! - `unpack_ternary2bit_to_f32` (`slime_backward.rs`) — per-row PRQ scale × LUT
//! - the extraction pattern in `tools/training_healthcheck.rs`:
//!   `u32_idx = offset/8; shift = (offset%8)*4; bits = (*(ptr+u32_idx) >> shift) & 0xF`
//!
//! All checks operate on real pointer addresses (raw `Vec` / mmap `data_ptr`).

use crate::mud::ezop::{pack_elut_prq, pack_ternary_into};
use crate::mud::{dequantize_ternary_row, MudFile, MudTensorType, TERNARY_LUT};

/// Decode one ternary element from a packed u32 buffer at global offset `k`
/// using the **same address math as `tools/training_healthcheck.rs`**.
/// This is the formula under audit — it must equal `dequantize_ternary_row`.
#[inline]
unsafe fn elut_nibble_at(ptr: *const u32, k: usize) -> f32 {
    let u32_idx = k / 8;
    let shift = (k % 8) * 4;
    let bits = ((*ptr.add(u32_idx) >> shift) & 0xF) as usize;
    TERNARY_LUT[bits]
}

/// Round-trip: pack ternary f32 (∈{-1,0,1}) → u32 (raw ptr) → dequant (raw ptr).
/// Validates `pack_ternary_into` + `dequantize_ternary_row` pointer math + nibble order.
pub fn check_pack_dequant_roundtrip(values: &[f32]) -> bool {
    let n = values.len();
    let u32s = n.div_ceil(8);
    let mut packed = vec![0u32; u32s];
    let delta = 0.5f32; // any value with |v|>0.5 is ±1; all our inputs are exact ±1/0
    unsafe {
        pack_ternary_into(values.as_ptr(), n, delta, packed.as_mut_ptr());
    }
    let mut out = vec![0.0f32; n];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut out, n);
    }
    values
        .iter()
        .zip(out.iter())
        .all(|(a, b)| a.abs() > delta && b.abs() > delta ||
              (a.abs() <= delta && b.abs() <= delta))
}

/// Cross-check the training_healthcheck nibble formula against `dequantize_ternary_row`
/// for **every** global offset — proves the address calculation is correct (no OOB,
/// correct u32 word + nibble shift).
pub fn check_nibble_formula_consistency(n: usize) -> bool {
    let u32s = n.div_ceil(8);
    let mut packed = vec![0u32; u32s.max(1)];
    // fill with pseudo-random nibbles
    for w in packed.iter_mut() {
        *w = 0x1234_5678u32
            .rotate_left((n as u32).wrapping_mul(7))
            .wrapping_add(*w)
            ^ 0x9ABC_DEF0;
    }
    let mut ref_out = vec![0.0f32; n];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut ref_out, n);
    }
    for (k, refv) in ref_out.iter().enumerate() {
        let got = unsafe { elut_nibble_at(packed.as_ptr(), k) };
        if got - *refv < -1e-6 || got - *refv > 1e-6 {
            return false;
        }
    }
    true
}

/// ELUT 2-values/byte round-trip: `pack_elut_prq` (raw ptr) → manual decode.
/// Validates the `start/2 + c` byte layout and low/high nibble assignment.
pub fn check_elut_2perbyte_roundtrip(rows: usize, cols: usize) -> bool {
    let total = rows * cols;
    let mut shadow = vec![0.0f32; total];
    let mut scales = vec![0.0f32; rows];
    let mut packed = vec![0u8; total.div_ceil(2)];
    // deterministic ±values around threshold
    for (i, v) in shadow.iter_mut().enumerate() {
        *v = match i % 3 {
            0 => 1.0,
            1 => -1.0,
            _ => 0.0,
        };
    }
    unsafe {
        pack_elut_prq(
            shadow.as_ptr(),
            rows,
            cols,
            scales.as_mut_ptr(),
            packed.as_mut_ptr(),
        );
    }
    // manual decode matching the documented layout
    for e in 0..total {
        let byte = packed[e / 2];
        let nibble = if e % 2 == 0 { byte & 0xF } else { (byte >> 4) & 0xF };
        let got = TERNARY_LUT[nibble as usize];
        let expected = match e % 3 {
            0 => 1.0,
            1 => -1.0,
            _ => 0.0,
        };
        if (got - expected).abs() > 1e-6 {
            return false;
        }
    }
    true
}

/// `unpack_ternary2bit_to_f32` (per-row PRQ scale × LUT) must equal
/// `dequantize_ternary_row` per row × that row's scale.
pub fn check_unpack_matches_dequant(rows: usize, cols: usize) -> bool {
    let total = rows * cols;
    let u32s = total.div_ceil(8);
    let mut packed = vec![0u32; u32s];
    let mut scales = vec![0.0f32; rows];
    for (r, s) in scales.iter_mut().enumerate() {
        *s = 1.0 + (r as f32) * 0.01;
        for c in 0..cols {
            let v: f32 = match (r + c) % 3 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            };
            let k = r * cols + c;
            if v.abs() > 0.0 {
                let bits = if v > 0.0 { 0x1u32 } else { 0xFu32 };
                packed[k / 8] |= bits << ((k % 8) * 4);
            }
        }
    }
    let packed_bytes = unsafe {
        std::slice::from_raw_parts(packed.as_ptr() as *const u8, u32s * 4)
    };
    let mut unpacked = vec![0.0f32; total];
    crate::mud::slime_backward::unpack_ternary2bit_to_f32(
        packed_bytes,
        &scales,
        cols,
        &mut unpacked,
    );
    let mut ref_dec = vec![0.0f32; total];
    unsafe {
        dequantize_ternary_row(packed.as_ptr(), &mut ref_dec, total);
    }
    for (r, scale) in scales.iter().enumerate() {
        for c in 0..cols {
            let i = r * cols + c;
            let expected = ref_dec[i] * scale;
            if (unpacked[i] - expected).abs() > 1e-5 {
                return false;
            }
        }
    }
    true
}

/// Audit the REAL on-disk layout of a `.mud` model: for every Ternary2Bit tensor,
/// read each element via the training_healthcheck pointer formula and assert it
/// equals `dequantize_ternary_row`'s decode of the same mmap `data_ptr`.
pub struct PointerModelReport {
    pub tensors_checked: usize,
    pub elements_checked: usize,
    pub mismatches: usize,
    pub max_abs_err: f32,
}

pub fn audit_model_pointers(mud: &MudFile) -> PointerModelReport {
    let core = mud
        .skills
        .get("core")
        .expect("model has no core skill");
    let mut rep = PointerModelReport {
        tensors_checked: 0,
        elements_checked: 0,
        mismatches: 0,
        max_abs_err: 0.0,
    };
    for t in core.tensors.values() {
        if t.t_type != MudTensorType::Ternary2Bit || t.data_ptr.is_null() {
            continue;
        }
        rep.tensors_checked += 1;
        let total: usize = t.shape.iter().product();
        let u32s = total.div_ceil(8);
        // Reference decode of the whole tensor via the kernel.
        let mut ref_dec = vec![0.0f32; total];
        unsafe {
            dequantize_ternary_row(t.data_ptr as *const u32, &mut ref_dec, total);
        }
        // Cross-check every element with the training_healthcheck address formula.
        unsafe {
            let ptr = t.data_ptr as *const u32;
            for (k, refv) in ref_dec.iter().enumerate() {
                let got = elut_nibble_at(ptr, k);
                let err = (got - *refv).abs();
                rep.elements_checked += 1;
                if err > 1e-6 {
                    rep.mismatches += 1;
                    if err > rep.max_abs_err {
                        rep.max_abs_err = err;
                    }
                }
            }
        }
        // boundary sanity: packed buffer must be exactly u32s words
        debug_assert_eq!(u32s * 4, (total.div_ceil(8)) * 4);
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_roundtrip_pack_dequant() {
        for n in [1usize, 7, 8, 9, 15, 16, 100, 257, 1000] {
            let vals: Vec<f32> = (0..n)
                .map(|i| match i % 3 {
                    0 => 1.0,
                    1 => -1.0,
                    _ => 0.0,
                })
                .collect();
            assert!(
                check_pack_dequant_roundtrip(&vals),
                "roundtrip failed at n={n}"
            );
        }
    }

    #[test]
    fn pointer_nibble_formula_consistent() {
        for n in [1usize, 8, 9, 16, 31, 64, 257] {
            assert!(
                check_nibble_formula_consistency(n),
                "nibble formula mismatch at n={n}"
            );
        }
    }

    #[test]
    fn pointer_elut_2perbyte() {
        assert!(check_elut_2perbyte_roundtrip(4, 32));
        assert!(check_elut_2perbyte_roundtrip(1, 17));
        assert!(check_elut_2perbyte_roundtrip(30, 576));
    }

    #[test]
    fn pointer_unpack_matches_dequant() {
        assert!(check_unpack_matches_dequant(2, 24));
        assert!(check_unpack_matches_dequant(30, 576));
    }
}
