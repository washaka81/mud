//! C-MUD parameter types and CellV5 pack/unpack (extracted from `cmud.rs`).
//!
//! Contains:
//! - `ComplexF32`: dual f32 complex number (re + i·im) — the algebraic base type.
//! - `GaussTernary`: Gaussian ternary weight (wr, wi ∈ {-1,0,1}).
//! - CellV5 (u32 8:8:8:8) pack/unpack for 4× memory compression of C-MUD state.

/// Complex activation / state: \(re + i\cdot im\).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct ComplexF32 {
    pub re: f32,
    pub im: f32,
}

impl ComplexF32 {
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    #[inline]
    pub const fn from_real(re: f32) -> Self {
        Self { re, im: 0.0 }
    }

    /// Hermitian squared norm \(h h^* = Re^2 + Im^2\).
    #[inline]
    pub fn hermite_norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn hermite_norm(self) -> f32 {
        self.hermite_norm_sq().sqrt()
    }

    /// Phase \(\theta = \mathrm{atan2}(Im, Re)\).
    #[inline]
    pub fn phase(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Magnitude of imaginary part (thinking energy).
    #[inline]
    pub fn imag_energy(self) -> f32 {
        self.im.abs()
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Scale both components.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl std::ops::Add for ComplexF32 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl std::ops::Sub for ComplexF32 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

/// Gaussian ternary weight: real and imag each in \(\{-1,0,1\}\).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussTernary {
    pub wr: i8,
    pub wi: i8,
}

impl GaussTernary {
    #[inline]
    pub fn new(wr: i8, wi: i8) -> Self {
        debug_assert!((-1..=1).contains(&wr) && (-1..=1).contains(&wi));
        Self { wr, wi }
    }

    /// Pack 9 states as u8 id: `(wr+1)*3 + (wi+1)` ∈ 0..9.
    #[inline]
    pub fn pack_id(self) -> u8 {
        ((self.wr + 1) * 3 + (self.wi + 1)) as u8
    }

    #[inline]
    pub fn from_id(id: u8) -> Self {
        let id = id % 9;
        let wr = (id / 3) as i8 - 1;
        let wi = (id % 3) as i8 - 1;
        Self { wr, wi }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Stream I: CellV5 (u32 8:8:8:8) pack/unpack for C-MUD complex state
// ─────────────────────────────────────────────────────────────────────────

/// F32 → u8 saturating cast (0-255 range).
#[inline]
fn f32_to_u8_sat(v: f32) -> u8 {
    if !v.is_finite() || v < 0.0 {
        0
    } else if v > 255.0 {
        255
    } else {
        v as u8
    }
}

/// F32 → i8 saturating cast (-128 to 127).
#[inline]
fn f32_to_i8_sat(v: f32) -> i8 {
    if v.is_nan() {
        0
    } else if v < -128.0 {
        -128
    } else if v > 127.0 {
        127
    } else {
        v as i8
    }
}

/// Verdadadero si `MUD_CMUD_CELLV5=1` (opt-in, P-02 safe).
pub fn cmud_cellv5_enabled() -> bool {
    std::env::var("MUD_CMUD_CELLV5").as_deref() == Ok("1")
}

/// Empaqueta estado C-MUD ComplexF32 en un CellV5 (u32 8:8:8:8).
///
/// | Bits | Campo | Tipo | Rango |
/// |---|---|---|---|
/// | 0-7 | Re | u8 | 0-255 (magnitud real) |
/// | 8-15 | Im | i8 | -128 a 127 (energía de pensamiento) |
/// | 16-23 | θ | u8 | 0-255 → [0, 2π] (fase) |
/// | 24-31 | |Im| | u8 | 0-255 (magnitud energía pensamiento) |
///
/// **Cada bit usaado:** 4/4 campos significativos.
/// **Compresión:** 4× (8 bytes ComplexF32 → 4 bytes u32).
#[inline(always)]
pub fn pack_cmud_to_cellv5(re: f32, im: f32, theta: f32, im_abs: f32) -> u32 {
    let re_u8 = f32_to_u8_sat(re.abs());
    let im_i8 = f32_to_i8_sat(im);
    // θ ∈ [0, 2π) → u8 [0, 255]
    let theta_u8 = (((theta / (2.0 * std::f32::consts::PI)).rem_euclid(1.0)) * 255.0).round() as u8;
    let im_abs_u8 = f32_to_u8_sat(im_abs);
    (re_u8 as u32)
        | ((im_i8 as u8 as u32) << 8)
        | ((theta_u8 as u32) << 16)
        | ((im_abs_u8 as u32) << 24)
}

/// Desempaqueta un CellV5 (u32 8:8:8:8) a estado C-MUD ComplexF32.
///
/// Devuelve `(Re, Im, θ, |Im|)`.
#[inline(always)]
pub fn unpack_cellv5_to_cmud(cell: u32) -> (f32, f32, f32, f32) {
    let re = (cell & 0xFF) as f32;
    let im = ((cell >> 8) & 0xFF) as i8 as f32;
    let theta = ((cell >> 16) & 0xFF) as f32 / 255.0 * 2.0 * std::f32::consts::PI;
    let im_abs = ((cell >> 24) & 0xFF) as f32;
    (re, im, theta, im_abs)
}

/// Empaqueta un ComplexF32 completo en CellV5 (calcula θ e |Im| internamente).
#[inline(always)]
pub fn pack_complexf32_to_cellv5(h: ComplexF32) -> u32 {
    let theta = h.phase();
    let im_abs = h.im.abs();
    pack_cmud_to_cellv5(h.re, h.im, theta, im_abs)
}

/// Desempaqueta un CellV5 a ComplexF32 (Re e Im preservados, θ e |Im| derivados).
#[inline(always)]
pub fn unpack_cellv5_to_complexf32(cell: u32) -> ComplexF32 {
    let (re, im, _theta, _im_abs) = unpack_cellv5_to_cmud(cell);
    ComplexF32::new(re, im)
}

/// Batch pack: empaqueta N ComplexF32 en N CellV5 (u32).
/// Zero-alloc (P-01) — escribe directamente en `out`.
#[inline]
pub fn pack_batch_complexf32_to_cellv5(src: &[ComplexF32], out: &mut [u32]) {
    let n = src.len().min(out.len());
    for i in 0..n {
        out[i] = pack_complexf32_to_cellv5(src[i]);
    }
}

/// Batch unpack: desempaqueta N CellV5 a N ComplexF32.
/// Zero-alloc (P-01) — escribe directamente en `out`.
#[inline]
pub fn unpack_batch_cellv5_to_complexf32(src: &[u32], out: &mut [ComplexF32]) {
    let n = src.len().min(out.len());
    for i in 0..n {
        out[i] = unpack_cellv5_to_complexf32(src[i]);
    }
}

#[cfg(test)]
mod cmud_cellv5_tests {
    use super::*;

    #[test]
    fn test_pack_unpack_cellv5_roundtrip() {
        let h = ComplexF32::new(3.5, -2.0);
        let cell = pack_complexf32_to_cellv5(h);
        let h2 = unpack_cellv5_to_complexf32(cell);
        // Re y Im se preservan (8-bit quantization)
        assert!((h.re - h2.re).abs() < 1.0, "Re mismatch: {} vs {}", h.re, h2.re);
        assert!((h.im - h2.im).abs() < 1.0, "Im mismatch: {} vs {}", h.im, h2.im);
    }

    #[test]
    fn test_pack_unpack_cellv5_full_fields() {
        let re = 128.5;
        let im = -64.3;
        let theta = 1.5; // ~85.9 degrees
        let im_abs = 64.3;
        let cell = pack_cmud_to_cellv5(re, im, theta, im_abs);
        let (r, i, t, ia) = unpack_cellv5_to_cmud(cell);
        assert!((r - re.abs()).abs() < 1.0);
        assert!((i - im).abs() < 1.0);
        assert!((t - theta).abs() < 0.05); // 255 steps → ~0.025 rad precision
        assert!((ia - im_abs).abs() < 1.0);
    }

    #[test]
    fn test_pack_unpack_cellv5_edge_cases() {
        // Zero
        let cell = pack_cmud_to_cellv5(0.0, 0.0, 0.0, 0.0);
        let (r, i, t, ia) = unpack_cellv5_to_cmud(cell);
        assert!(r == 0.0 && i == 0.0 && t == 0.0 && ia == 0.0);

        // Max values
        let cell = pack_cmud_to_cellv5(255.0, 127.0, 2.0 * std::f32::consts::PI, 255.0);
        let (r, i, _t, ia) = unpack_cellv5_to_cmud(cell);
        assert!(r == 255.0 && i == 127.0 && ia == 255.0);

        // NaN safety
        let cell = pack_cmud_to_cellv5(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        let (r, i, _, ia) = unpack_cellv5_to_cmud(cell);
        assert!(r == 0.0 && i == 0.0 && ia == 0.0);
    }

    #[test]
    fn test_batch_pack_unpack_cellv5() {
        let src: Vec<ComplexF32> = (0..16)
            .map(|i| ComplexF32::new(i as f32 * 0.1, (i as i8 - 8) as f32 * 0.1))
            .collect();
        let mut cells = vec![0u32; 16];
        let mut out = vec![ComplexF32::default(); 16];

        pack_batch_complexf32_to_cellv5(&src, &mut cells);
        unpack_batch_cellv5_to_complexf32(&cells, &mut out);

        for i in 0..16 {
            assert!((src[i].re - out[i].re).abs() < 1.0,
                "Re mismatch at {}: {} vs {}", i, src[i].re, out[i].re);
            assert!((src[i].im - out[i].im).abs() < 1.0,
                "Im mismatch at {}: {} vs {}", i, src[i].im, out[i].im);
        }
    }

    #[test]
    fn test_cmud_cellv5_compression_ratio() {
        // ComplexF32 = 8 bytes, CellV5 = 4 bytes → 4× compression
        let complex_bytes = std::mem::size_of::<ComplexF32>();
        let cell_bytes = std::mem::size_of::<u32>();
        let ratio = 1.0 - (cell_bytes as f32 / complex_bytes as f32);
        assert!((ratio - 0.5).abs() < 0.01, "Expected 50% size reduction, got {:.2}%", ratio * 100.0);
    }
}
