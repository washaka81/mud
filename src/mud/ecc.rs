/// SEC(38,32): 32 data bits + 6 parity bits (single error correction, no DED).
///
/// This is a systematic Hamming code where 6 parity bits p1,p2,p4,p8,p16,p32
/// are stored separately (not interleaved). The syndrome directly encodes the
/// position of the flipped bit (1-indexed), allowing single-bit correction.
///
/// Syndrome = 0 → no error.
/// Syndrome ∈ [1, 32] → data bit (syndrome-1) is flipped, corrected.
/// Syndrome ∈ [33, 38] → parity bit error (data intact).
/// Syndrome > 38 → unreachable (6 bits max = 63).
const PARITY_MASKS: [u32; 6] = [
    0x55555555, // p1:  1-indexed position LSB=1 (even idx: 0,2,4,...,30)
    0x66666666, // p2:  1-indexed position bit1=1 (idx 1,2,5,6,9,10,...)
    0x78787878, // p4:  1-indexed position bit2=1 (idx 3..6,11..14,19..22,27..30)
    0x7F807F80, // p8:  1-indexed position bit3=1 (idx 7..14,23..30)
    0x7FFF8000, // p16: 1-indexed position bit4=1 (idx 15..30)
    0x80000000, // p32: 1-indexed position bit5=1 (idx 31 only)
];

fn parity_of(x: u32) -> u32 {
    x.count_ones() & 1
}

/// Compute 6-bit ECC parity for a single u32 of packed ternary weights.
/// Returns a u8 with lower 6 bits as parity (p1=bit0, p2=bit1, p4=bit2, p8=bit3, p16=bit4, p32=bit5).
pub fn ecc_parity(data: u32) -> u8 {
    let p1 = parity_of(data & PARITY_MASKS[0]);
    let p2 = parity_of(data & PARITY_MASKS[1]);
    let p4 = parity_of(data & PARITY_MASKS[2]);
    let p8 = parity_of(data & PARITY_MASKS[3]);
    let p16 = parity_of(data & PARITY_MASKS[4]);
    let p32 = parity_of(data & PARITY_MASKS[5]);
    (p1 as u8)
        | ((p2 as u8) << 1)
        | ((p4 as u8) << 2)
        | ((p8 as u8) << 3)
        | ((p16 as u8) << 4)
        | ((p32 as u8) << 5)
}

/// Verify a single u32 using 6-bit parity and correct single-bit errors.
///
/// Returns (corrected_data, error_kind):
///   0 = no error
///   1 = corrected single bit flip
///   2 = parity bit error (data intact)
pub fn ecc_verify(data: u32, parity: u8) -> (u32, u32) {
    let stored = parity as u32;
    let p1 = parity_of(data & PARITY_MASKS[0]);
    let p2 = parity_of(data & PARITY_MASKS[1]);
    let p4 = parity_of(data & PARITY_MASKS[2]);
    let p8 = parity_of(data & PARITY_MASKS[3]);
    let p16 = parity_of(data & PARITY_MASKS[4]);
    let p32 = parity_of(data & PARITY_MASKS[5]);

    // Syndrome directly encodes the 1-indexed position of the flipped bit.
    // For a data flip at 0-indexed bit i: syndrome = i+1
    let syndrome = ((stored & 1) ^ p1)
        | ((((stored >> 1) & 1) ^ p2) << 1)
        | ((((stored >> 2) & 1) ^ p4) << 2)
        | ((((stored >> 3) & 1) ^ p8) << 3)
        | ((((stored >> 4) & 1) ^ p16) << 4)
        | ((((stored >> 5) & 1) ^ p32) << 5);

    if syndrome == 0 {
        return (data, 0);
    }

    // Syndrome = 1-indexed position. Positions 1..32 are data bits, 33+ are parity-only.
    if syndrome <= 32 {
        let bit_pos = (syndrome - 1) as usize;
        (data ^ (1 << bit_pos), 1)
    } else {
        (data, 2) // parity bit error
    }
}

/// Compute ECC parity for every u32 in a slice.
pub fn ecc_compute_buf(data: &[u32]) -> Vec<u8> {
    data.iter().map(|&w| ecc_parity(w)).collect()
}

/// Verify and correct a buffer of u32s in-place.
/// Returns (errors_corrected, parity_errors).
pub fn ecc_verify_buf(data: &mut [u32], parity: &[u8]) -> (u32, u32) {
    let mut corrected = 0;
    let mut parity_err = 0;
    for (i, w) in data.iter_mut().enumerate() {
        if let Some(&p) = parity.get(i) {
            let (new_w, kind) = ecc_verify(*w, p);
            *w = new_w;
            match kind {
                1 => corrected += 1,
                2 => parity_err += 1,
                _ => {}
            }
        }
    }
    (corrected, parity_err)
}

/// Reinterpret a byte buffer as little-endian u32 slice (read-only).
pub fn as_u32_slice_le(buf: &[u8]) -> &[u32] {
    assert!(
        buf.len().is_multiple_of(4),
        "ECC buffer must be multiple of 4 bytes"
    );
    bytemuck::cast_slice(buf)
}

/// Reinterpret a byte buffer as little-endian u32 slice (mutable).
pub fn as_u32_slice_le_mut(buf: &mut [u8]) -> &mut [u32] {
    assert!(
        buf.len().is_multiple_of(4),
        "ECC buffer must be multiple of 4 bytes"
    );
    bytemuck::cast_slice_mut(buf)
}

/// Create a zero-filled `Vec<u8>` with guaranteed 4-byte alignment.
/// The returned vector has length `u32_count * 4`.
pub fn aligned_u8_vec(u32_count: usize) -> Vec<u8> {
    let mut u32_vec = vec![0u32; u32_count];
    let ptr = u32_vec.as_mut_ptr() as *mut u8;
    let len = u32_vec.len() * 4;
    let cap = u32_vec.capacity() * 4;
    std::mem::forget(u32_vec);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

/// Copy bytes into a 4-byte-aligned `Vec<u8>`.
/// Panics if `src.len()` is not a multiple of 4.
pub fn aligned_copy(src: &[u8]) -> Vec<u8> {
    assert!(
        src.len().is_multiple_of(4),
        "aligned_copy: source length must be multiple of 4"
    );
    let mut dst = aligned_u8_vec(src.len() / 4);
    dst.copy_from_slice(src);
    dst
}

/// ECC tensor naming convention.
pub fn ecc_tensor_name(tensor_name: &str) -> String {
    format!("{}.ecc", tensor_name)
}

/// Check if a tensor name is an ECC tensor.
pub fn is_ecc_tensor(name: &str) -> bool {
    name.ends_with(".ecc")
}

/// Extract the base tensor name from an ECC tensor name.
pub fn base_tensor_name(ecc_name: &str) -> Option<&str> {
    ecc_name.strip_suffix(".ecc")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_error() {
        let data = 0x12345678u32;
        let p = ecc_parity(data);
        let (d, e) = ecc_verify(data, p);
        assert_eq!(e, 0, "no error expected");
        assert_eq!(d, data);
    }

    #[test]
    fn test_single_bit_flip() {
        let data = 0x12345678u32;
        let p = ecc_parity(data);
        for bit in 0..32 {
            let flipped = data ^ (1 << bit);
            let (d, e) = ecc_verify(flipped, p);
            assert_eq!(e, 1, "bit {}: should correct", bit);
            assert_eq!(d, data, "bit {}: wrong correction", bit);
        }
    }

    #[test]
    fn test_all_zeros() {
        let p = ecc_parity(0);
        let (d, e) = ecc_verify(0, p);
        assert_eq!(e, 0);
        assert_eq!(d, 0);
    }

    #[test]
    fn test_all_ones() {
        let p = ecc_parity(0xFFFFFFFF);
        let (d, e) = ecc_verify(0xFFFFFFFF, p);
        assert_eq!(e, 0);
        assert_eq!(d, 0xFFFFFFFF);
    }

    #[test]
    fn test_specific_patterns() {
        for &data in &[0xDEADBEEFu32, 0x0, 0x1, 0xFFFF, 0xAAAAAAAA, 0x55555555] {
            let p = ecc_parity(data);
            for bit in [0, 7, 13, 31] {
                let flipped = data ^ (1 << bit);
                let (d, e) = ecc_verify(flipped, p);
                assert_eq!(e, 1, "0x{:08x} bit {}: should correct", data, bit);
                assert_eq!(d, data, "0x{:08x} bit {}: wrong value", data, bit);
            }
        }
    }

    #[test]
    fn test_buf_roundtrip() {
        let orig = [0x12345678u32, 0x9ABCDEF0u32, 0x0F1E2D3Cu32];
        let mut buf = orig;
        let parity = ecc_compute_buf(&buf);
        buf[1] ^= 1 << 14; // flip bit 14 of second element
        let (c, p_err) = ecc_verify_buf(&mut buf, &parity);
        assert_eq!(c, 1, "should correct 1 error");
        assert_eq!(p_err, 0, "no parity errors expected");
        assert_eq!(buf, orig, "buffer should be fully restored");
    }

    #[test]
    fn test_naming() {
        assert_eq!(
            ecc_tensor_name("blk.0.attn_q.weight"),
            "blk.0.attn_q.weight.ecc"
        );
        assert!(is_ecc_tensor("blk.0.attn_q.weight.ecc"));
        assert!(!is_ecc_tensor("blk.0.attn_q.weight"));
        assert_eq!(
            base_tensor_name("blk.0.attn_q.weight.ecc"),
            Some("blk.0.attn_q.weight")
        );
        assert_eq!(base_tensor_name("blk.0.attn_q.weight"), None);
    }
}
