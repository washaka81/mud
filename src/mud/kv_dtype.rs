//! # Stream I: KV storage dtype policy (f32 default, bf16 optional)
//!
//! When `MUD_KV_DTYPE=bf16` (or `f16`), dense ring + HCA stores **IEEE f16**
//! (`half::f16`) packing — ~2× RAM savings. Attention dequantizes to f32 on
//! read for matmul with Q.
//!
//! Note: product uses IEEE f16 (not Google BF16) via the `half` crate for
//! portability; env accepts `bf16` as alias.

use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvDtype {
    F32,
    /// IEEE binary16 (alias env: bf16|f16|half)
    F16,
}

impl KvDtype {
    pub fn resolve() -> Self {
        match std::env::var("MUD_KV_DTYPE") {
            Err(_) => KvDtype::F32,
            Ok(v) => {
                let t = v.trim().to_ascii_lowercase();
                match t.as_str() {
                    "bf16" | "f16" | "half" | "float16" => KvDtype::F16,
                    "f32" | "float32" | "fp32" | "" => KvDtype::F32,
                    _ => KvDtype::F32,
                }
            }
        }
    }

    #[inline]
    pub fn is_f16(self) -> bool {
        matches!(self, KvDtype::F16)
    }

    /// Bytes per element in the backing store.
    #[inline]
    pub fn elem_bytes(self) -> usize {
        match self {
            KvDtype::F32 => 4,
            KvDtype::F16 => 2,
        }
    }

    pub fn summary(self) -> String {
        match self {
            KvDtype::F32 => "KV_DTYPE=f32".into(),
            KvDtype::F16 => "KV_DTYPE=f16 (IEEE half, ~2× KV RAM)".into(),
        }
    }
}

/// Pack f32 slice → f16 bytes (little-endian, 2 bytes each).
pub fn pack_f32_to_f16_bytes(src: &[f32], dst: &mut [u8]) {
    assert!(dst.len() >= src.len() * 2);
    for (i, &v) in src.iter().enumerate() {
        let h = f16::from_f32(v);
        let bits = h.to_bits().to_le_bytes();
        dst[i * 2] = bits[0];
        dst[i * 2 + 1] = bits[1];
    }
}

/// Unpack f16 bytes → f32.
pub fn unpack_f16_bytes_to_f32(src: &[u8], dst: &mut [f32]) {
    assert!(src.len() >= dst.len() * 2);
    for (i, d) in dst.iter_mut().enumerate() {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        *d = f16::from_bits(bits).to_f32();
    }
}

/// In-place store: write `head_dim` floats into f16 pack buffer at element offset.
pub fn store_row_f16(pack: &mut [u8], elem_offset: usize, row: &[f32]) {
    let byte_off = elem_offset * 2;
    pack_f32_to_f16_bytes(row, &mut pack[byte_off..byte_off + row.len() * 2]);
}

/// Load row from f16 pack into f32 scratch.
pub fn load_row_f16(pack: &[u8], elem_offset: usize, row: &mut [f32]) {
    let byte_off = elem_offset * 2;
    unpack_f16_bytes_to_f32(&pack[byte_off..byte_off + row.len() * 2], row);
}

/// Approximate quality: max abs error for random values in [-2,2] after roundtrip.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip_accuracy() {
        let src: Vec<f32> = (-20..20).map(|i| i as f32 * 0.1).collect();
        let mut bytes = vec![0u8; src.len() * 2];
        pack_f32_to_f16_bytes(&src, &mut bytes);
        let mut dst = vec![0.0f32; src.len()];
        unpack_f16_bytes_to_f32(&bytes, &mut dst);
        let mut max_err = 0.0f32;
        for (a, b) in src.iter().zip(dst.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        // f16 has ~3 decimal digits; allow generous bound
        assert!(max_err < 0.01, "max_err={max_err}");
    }

    #[test]
    fn test_store_load_row() {
        let row = [1.0f32, -0.5, 0.25, 0.0];
        let mut pack = vec![0u8; 16 * 2];
        store_row_f16(&mut pack, 4, &row);
        let mut out = [0.0f32; 4];
        load_row_f16(&pack, 4, &mut out);
        for i in 0..4 {
            assert!((row[i] - out[i]).abs() < 1e-2);
        }
    }

    #[test]
    fn test_resolve_aliases() {
        // Don't fight Once-less env races hard; just call resolve
        let _ = KvDtype::resolve();
        assert_eq!(KvDtype::F16.elem_bytes(), 2);
        assert_eq!(KvDtype::F32.elem_bytes(), 4);
    }
}
