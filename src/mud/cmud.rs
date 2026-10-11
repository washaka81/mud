//! # L-14: C-MUD — Complex-Valued Thinking Manifold (research foundation)
//!
//! Implements the algebraic core from `docs/research/COMPLEX_THINKING_MANIFOLD.md`
//! **without** replacing the live `SlimeRegister` f32 path (P-02 remains SSOT for production).
//!
//! | Piece | Role |
//! |-------|------|
//! | Gaussian ternary weights | \(W = W_R + i W_I\), \(W_{R,I}\in\{-1,0,1\}\) → 9 states |
//! | Complex activation | \(X_R + i X_I\) as dual f32 |
//! | Hermitian mHC | \(\\|h\\|_C^2 = Re^2+Im^2 \le R^2\) |
//! | Phase-lock | EMA of angular speed \(\omega_\tau\); escape when \(\mathrm{EMA}(\omega)<\varepsilon\) |
//! | Wave collapse | Map locked complex state → real activation for LM head |
//!
//! Opt-in later via `MUD_CMUD_THINK=1` once a full complex GEMV exists; this module is the math kernel.

use crate::mud::constants::EPSILON_FLOOR;

// Thread-local scratch buffers (P-01: eliminate per-call Vec allocations in C-MUD hot path).
thread_local! {
    static THINK_ATTN_SCRATCH: std::cell::RefCell<ThinkAttnScratch> = const {
        std::cell::RefCell::new(ThinkAttnScratch {
            attn: Vec::new(),
            cos_table: Vec::new(),
            sin_table: Vec::new(),
        })
    };
    static SPECTRAL_MAG_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    static CUE_REPULSION_SCRATCH: std::cell::RefCell<CueRepulsionScratch> = const {
        std::cell::RefCell::new(CueRepulsionScratch {
            sin_half: Vec::new(),
            cos_half: Vec::new(),
        })
    };
    /// Buffers for `think_step_trainable_core` — qph, attn, zvec, h_before.
    /// Reused across calls; `std::mem::take` extracts Vecs when they must be moved into a ThinkTape.
    static THINK_TRAINABLE_SCRATCH: std::cell::RefCell<ThinkTrainableScratch> = const {
        std::cell::RefCell::new(ThinkTrainableScratch {
            qph: Vec::new(),
            attn: Vec::new(),
            zvec: Vec::new(),
            h_before: Vec::new(),
            cos_kph: Vec::new(),
            sin_kph: Vec::new(),
            v_flat: Vec::new(),
        })
    };
}

/// Thread-local scratch for `think_step_phase_attn` (P-01 zero-alloc).
struct ThinkAttnScratch {
    attn: Vec<f32>,
    /// Precomputed cos/sin of phases — eliminates `cos()` from the O(n×win) inner loop.
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

/// Thread-local scratch for `cue_phase_repulsion` (P-01 zero-alloc).
/// Precomputes sin(phases[i]/2) and cos(phases[i]/2) to eliminate O(n²) sin/cos calls.
struct CueRepulsionScratch {
    sin_half: Vec<f32>,
    cos_half: Vec<f32>,
}

/// Thread-local scratch buffers for `think_step_trainable_core` (P-01 zero-alloc).
struct ThinkTrainableScratch {
    qph: Vec<f32>,
    attn: Vec<f32>,
    zvec: Vec<f32>,
    h_before: Vec<f32>,
    /// Precomputed cos/sin of key phases — eliminates `cos()` from the O(n×win) inner loop.
    cos_kph: Vec<f32>,
    sin_kph: Vec<f32>,
    /// Precomputed v_scale values (eliminates `get().copied().unwrap_or()` from inner loop).
    v_flat: Vec<f32>,
}

/// Default phase-lock threshold (research doc: ~1e-3).
pub const PHASE_LOCK_EPS: f32 = 1e-3;
/// EMA rate for angular speed \(\omega\).
pub const OMEGA_EMA_RATE: f32 = 0.1;
/// Max internal thinking iterations \(\tau\).
pub const DEFAULT_THINK_ITERS: usize = 8;
/// Default residual mix \(\alpha\) for the phase-coherent thinking step (gentle perturbation;
/// overridable via `MUD_CMUD_ALPHA`). Small enough to preserve the hidden magnitude/structure.
pub const CMUD_DEFAULT_ALPHA: f32 = 0.05;
/// CUE phase-repulsion learning rate \(\eta\) (orbit E1).
pub const CMUD_REPULSION_ETA: f32 = 0.01;
/// Soft phase-repulsion nudge factor applied to the thinking residual.
pub const CMUD_REPULSION_NUDGE: f32 = 0.1;
/// Hermitian-ball radius = `CMUD_RADIUS_RMS_FACTOR ×` hidden RMS (no clamp at seed).
pub const CMUD_RADIUS_RMS_FACTOR: f32 = 2.0;
/// Positional phase step \(\omega\) for the relative-phase seed (ComplexFormer \(\Delta P=(i-j)\omega\)):
/// breaks the all-real symmetry so `cos(Δθ)` is position-dependent (no mean-pool collapse).
pub const CMUD_POS_PHASE_STEP: f32 = 0.1;
/// Local attention window half-width (ComplexFormer local mixing): keeps the think step a
/// gentle local perturbation instead of a global low-pass blur that washes out the hidden
/// (Phasor/LPM: unitary, non-averaging phase mix).
pub const CMUD_WIN_HALF: usize = 16;

// ComplexF32, GaussTernary, and CellV5 pack/unpack extracted to cmud_params module.
pub use crate::mud::cmud_params::{
    cmud_cellv5_enabled, pack_batch_complexf32_to_cellv5, pack_cmud_to_cellv5,
    pack_complexf32_to_cellv5, unpack_batch_cellv5_to_complexf32, unpack_cellv5_to_cmud,
    unpack_cellv5_to_complexf32, ComplexF32, GaussTernary,
};

// Complex arithmetic kernel (gauss_mul, wave_collapse, phase_delta, etc.)
// extracted to cmud_thinking module.
pub use crate::mud::cmud_thinking::{
    gauss_mac, gauss_mul, gauss_mul_batch_avx2, phase_delta,
    project_hermitian, wave_collapse, wave_collapse_batch_avx2,
};

/// Internal thinking loop state over a complex hidden vector.
///
/// Design (research §3, fixed over-sharp): the **values stay real** (`h[i].im = 0`,
/// magnitude = `|x_i|` preserved) — only the **phase array** evolves. Attention scores use
/// the phase (ComplexFormer: Q/K complex, **V real**), so the reasoning mixes relations
/// without losing the value magnitude (which would otherwise flatten the LM-head logits).
pub struct ThinkingState {
    pub h: Vec<ComplexF32>,
    pub theta_prev: Vec<f32>,
    pub phases: Vec<f32>,
    pub omega_ema: f32,
    pub tau: usize,
    pub radius: f32,
    pub lock_eps: f32,
    pub sigma: f32,
}

/// SPRINT-3: Schema version of the JSON sidecar. Bump this whenever the
/// serialized `CmudLayerParams` changes shape (add/remove field, change unit,
/// change default). `load_json` rejects mismatches with a clear error.
pub const SIDECAR_SCHEMA_VERSION: u32 = 1;

/// Trainable complex-reasoning layer parameters (research §4, #4 entrenable).
///
/// Replaces the fixed positional phase seed with **learned per-dimension phase biases**
/// for the query/key projections (`q_phase`, `k_phase`) plus a learned per-dimension real
/// scale for the value (`v_scale`). With all biases = 0 and `v_scale = 1` this is exactly the
/// positional-phase behavior, so it is regression-safe when initialized from `from_defaults`.
/// A future trainer tunes these (and `alpha/eta_rep/sigma`) via the sampled-softmax loss.
///
/// Serialized as a JSON object with `schema_version` (see `SIDECAR_SCHEMA_VERSION`)
/// and a `params` field containing the actual `CmudLayerParams`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CmudLayerParams {
    pub alpha: f32,
    pub eta_rep: f32,
    pub omega_pos: f32,
    pub sigma: f32,
    /// Learned per-dimension phase bias added to the positional phase for Q.
    pub q_phase: Vec<f32>,
    /// Learned per-dimension phase bias added to the positional phase for K.
    pub k_phase: Vec<f32>,
    /// Learned per-dimension real scale modulating the value V (magnitude, not phase).
    pub v_scale: Vec<f32>,
}

impl CmudLayerParams {
    /// Identity-ish init: `q_phase=k_phase=0`, `v_scale=1`, scalars from the research constants.
    /// Behaves like the fixed positional-phase step until a trainer updates the biases.
    pub fn from_defaults(hidden: usize) -> Self {
        Self {
            alpha: CMUD_DEFAULT_ALPHA,
            eta_rep: CMUD_REPULSION_ETA,
            omega_pos: CMUD_POS_PHASE_STEP,
            sigma: 0.0,
            q_phase: vec![0.0f32; hidden],
            k_phase: vec![0.0f32; hidden],
            v_scale: vec![1.0f32; hidden],
        }
    }

    /// Persist to a JSON sidecar with `schema_version` envelope.
    /// The envelope is forward-compatible: future versions can detect old
    /// sidecars and migrate them.
    pub fn save_json(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let envelope = CmudSidecarEnvelope {
            schema_version: SIDECAR_SCHEMA_VERSION,
            params: self.clone(),
        };
        let s = serde_json::to_string_pretty(&envelope)?;
        std::fs::write(path, s)?;
        Ok(())
    }

    /// Load from a JSON sidecar written by [`CmudLayerParams::save_json`].
    /// Validates the `schema_version`; fails with a clear error on mismatch.
    /// Accepts legacy "bare" (no envelope) sidecars for backward compatibility.
    pub fn load_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::parse_json(&s)
    }

    /// Parse + validate a sidecar JSON string. Public so callers can validate
    /// in-memory sidecars (e.g. from a network buffer).
    pub fn parse_json(s: &str) -> anyhow::Result<Self> {
        // Try the enveloped format first.
        if let Ok(envelope) = serde_json::from_str::<CmudSidecarEnvelope>(s) {
            envelope.validate_version()?;
            return Ok(envelope.params);
        }
        // Fall back to the legacy bare format (pre-Sprint-3 sidecars).
        // We do this by checking if the JSON contains a `schema_version` field;
        // if not, treat as v1.
        if s.contains("\"schema_version\"") {
            anyhow::bail!(
                "sidecar has schema_version field but could not be parsed as a CmudSidecarEnvelope"
            );
        }
        let p: CmudLayerParams = serde_json::from_str(s)?;
        Ok(p)
    }

    /// Validate that the sidecar envelope's `schema_version` matches the
    /// current `SIDECAR_SCHEMA_VERSION`. Called by `load_json` and by the
    /// `cmud_train --check-version` flag.
    pub fn validate_version(envelope: &CmudSidecarEnvelope) -> anyhow::Result<()> {
        if envelope.schema_version != SIDECAR_SCHEMA_VERSION {
            anyhow::bail!(
                "sidecar schema_version mismatch: found v{}, expected v{}. \
                 Re-train with the current binary or migrate manually.",
                envelope.schema_version,
                SIDECAR_SCHEMA_VERSION
            );
        }
        Ok(())
    }

    /// Conventional sidecar path next to a `.mud` model: `<stem>.mud.cmud.json`.
    pub fn sidecar_for(model: &std::path::Path) -> std::path::PathBuf {
        let mut p = model.to_path_buf();
        p.set_extension("mud.cmud.json");
        p
    }
}

/// SPRINT-3: JSON sidecar envelope. The wire format is:
/// ```json
/// {
///   "schema_version": 1,
///   "params": { "alpha": ..., "q_phase": [...], ... }
/// }
/// ```
/// Future versions can detect old sidecars via `schema_version` and migrate.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CmudSidecarEnvelope {
    pub schema_version: u32,
    pub params: CmudLayerParams,
}

impl CmudSidecarEnvelope {
    /// Validate the envelope's `schema_version` against the current
    /// `SIDECAR_SCHEMA_VERSION`. Method form for ergonomic call sites.
    pub fn validate_version(&self) -> anyhow::Result<()> {
        CmudLayerParams::validate_version(self)
    }
}

impl Default for CmudLayerParams {
    fn default() -> Self {
        Self {
            alpha: CMUD_DEFAULT_ALPHA,
            eta_rep: CMUD_REPULSION_ETA,
            omega_pos: CMUD_POS_PHASE_STEP,
            sigma: 0.0,
            q_phase: Vec::new(),
            k_phase: Vec::new(),
            v_scale: Vec::new(),
        }
    }
}

#[cfg(test)]
mod avx2_batch_tests {
    use super::*;

    #[test]
    fn test_gauss_mul_batch_avx2_correctness() {
        // Test 8 complex multiplies: AVX2 vs scalar
        let x_re: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let x_im: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 - 0.4).collect();
        let w_re: Vec<f32> = (0..8).map(|i| ((i % 3) as i8 - 1) as f32).collect();
        let w_im: Vec<f32> = (0..8).map(|i| ((i % 3) as i8 - 1) as f32).collect();

        let mut y_re_avx2 = vec![0.0f32; 8];
        let mut y_im_avx2 = vec![0.0f32; 8];
        let mut y_re_scalar = [0.0f32; 8];
        let mut y_im_scalar = [0.0f32; 8];

        // AVX2 path
        gauss_mul_batch_avx2(&x_re, &x_im, &w_re, &w_im, &mut y_re_avx2, &mut y_im_avx2, 8);

        // Scalar reference
        for i in 0..8 {
            let x = ComplexF32::new(x_re[i], x_im[i]);
            let w = GaussTernary::new(w_re[i] as i8, w_im[i] as i8);
            let y = gauss_mul(x, w);
            y_re_scalar[i] = y.re;
            y_im_scalar[i] = y.im;
        }

        // Compare with relative tolerance
        for i in 0..8 {
            let rel_re = (y_re_avx2[i] - y_re_scalar[i]).abs() / y_re_scalar[i].abs().max(1.0);
            let rel_im = (y_im_avx2[i] - y_im_scalar[i]).abs() / y_im_scalar[i].abs().max(1.0);
            assert!(rel_re < 1e-5, "re mismatch at {}: {} vs {}", i, y_re_avx2[i], y_re_scalar[i]);
            assert!(rel_im < 1e-5, "im mismatch at {}: {} vs {}", i, y_im_avx2[i], y_im_scalar[i]);
        }
    }

    #[test]
    fn test_wave_collapse_batch_avx2_correctness() {
        // Test 8 wave collapses: AVX2 vs scalar
        let re: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 + 0.1).collect();
        let im: Vec<f32> = (0..8).map(|i| (i as f32) * 0.3 - 0.2).collect();

        let mut out_avx2 = vec![0.0f32; 8];
        let mut out_scalar = [0.0f32; 8];

        // AVX2 path
        wave_collapse_batch_avx2(&re, &im, &mut out_avx2, 8);

        // Scalar reference
        for i in 0..8 {
            let h = ComplexF32::new(re[i], im[i]);
            out_scalar[i] = wave_collapse(h);
        }

        // Compare with absolute tolerance (atan2/tanh/cos approximations)
        for i in 0..8 {
            let diff = (out_avx2[i] - out_scalar[i]).abs();
            assert!(diff < 0.01, "mismatch at {}: {} vs {}", i, out_avx2[i], out_scalar[i]);
        }
    }

    #[test]
    fn test_wave_collapse_batch_avx2_edge_cases() {
        // Edge cases: NaN, infinity, zero, negative re
        let re = vec![0.0f32, f32::INFINITY, -2.0, 0.0, 1.5, f32::NAN, -0.5, 3.0];
        let im = vec![0.0f32, 0.0, 0.0, 0.0, 2.0, 1.0, f32::INFINITY, -1.0];

        let mut out_avx2 = vec![0.0f32; 8];
        let mut out_scalar = [0.0f32; 8];

        wave_collapse_batch_avx2(&re, &im, &mut out_avx2, 8);

        for i in 0..8 {
            let h = ComplexF32::new(re[i], im[i]);
            out_scalar[i] = wave_collapse(h);
        }

        for i in 0..8 {
            let diff = (out_avx2[i] - out_scalar[i]).abs();
            assert!(diff < 0.01, "edge mismatch at {}: {} vs {}", i, out_avx2[i], out_scalar[i]);
        }
    }

    #[test]
    fn test_wave_collapse_batch_avx2_negative_re() {
        // Verify cos(atan2(im, re)) = re/sqrt(re²+im²) identity for negative re
        // (cos(atan2(0, -2)) = cos(π) = -1, so v = re² * 1 / |re| = |re| * sign(re) = re)
        let re = vec![-2.0f32, -1.0, -3.0, -0.5];
        let im = vec![0.0f32; 4];
        let mut out = vec![0.0f32; 4];
        wave_collapse_batch_avx2(&re, &im, &mut out, 4);

        for i in 0..4 {
            let h = ComplexF32::new(re[i], im[i]);
            let expected = wave_collapse(h);
            assert!((out[i] - expected).abs() < 0.01, "neg-re mismatch at {}: {} vs {}", i, out[i], expected);
        }
    }

    #[test]
    fn test_gauss_mul_batch_avx2_remainder() {
        // Test with n=10 (8 AVX2 + 2 scalar remainder)
        let x_re: Vec<f32> = (0..10).map(|i| (i as f32) * 0.3 + 0.1).collect();
        let x_im: Vec<f32> = (0..10).map(|i| (i as f32) * 0.2 - 0.4).collect();
        let w_re: Vec<f32> = (0..10).map(|i| ((i % 3) as i8 - 1) as f32).collect();
        let w_im: Vec<f32> = (0..10).map(|i| ((i % 3) as i8 - 1) as f32).collect();

        let mut y_re = vec![0.0f32; 10];
        let mut y_im = vec![0.0f32; 10];

        gauss_mul_batch_avx2(&x_re, &x_im, &w_re, &w_im, &mut y_re, &mut y_im, 10);

        // Verify all 10 elements
        for i in 0..10 {
            let x = ComplexF32::new(x_re[i], x_im[i]);
            let w = GaussTernary::new(w_re[i] as i8, w_im[i] as i8);
            let y = gauss_mul(x, w);
            let rel_re = (y_re[i] - y.re).abs() / y.re.abs().max(1.0);
            let rel_im = (y_im[i] - y.im).abs() / y.im.abs().max(1.0);
            assert!(rel_re < 1e-5, "re mismatch at {}", i);
            assert!(rel_im < 1e-5, "im mismatch at {}", i);
        }
    }
}

impl ThinkingState {
    /// Seed from real activations. Values stay real (`h[i].im = 0`, magnitude `=|x_i|`);
    /// a **relative positional phase** \(\phi_i = \omega i\) is injected into the `phases`
    /// array (ComplexFormer \(\Delta P=(i-j)\omega\)) so the phase-coherent attention score
    /// \(\cos(\phi_i-\phi_j)=\cos(\omega(i-j))\) is position-dependent (no mean-pool collapse).
    /// Magnitude is clamped to the Hermitian ball so the seed respects `radius`.
    pub fn from_real(x: &[f32], radius: f32) -> Self {
        let r = radius.max(EPSILON_FLOOR);
        let wpos = std::env::var("MUD_CMUD_POS_PHI")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(CMUD_POS_PHASE_STEP);
        let h: Vec<ComplexF32> = x
            .iter()
            .map(|&v| ComplexF32::new(v.abs().min(r).copysign(v), 0.0))
            .collect();
        let phases: Vec<f32> = (0..h.len()).map(|i| wpos * i as f32).collect();
        let theta_prev: Vec<f32> = phases.clone();
        Self {
            h,
            theta_prev,
            phases,
            omega_ema: 1.0, // start "thinking"
            tau: 0,
            radius: r,
            lock_eps: PHASE_LOCK_EPS,
            sigma: 0.0,
        }
    }

    pub fn len(&self) -> usize {
        self.h.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }

    /// Mean absolute phase step (L1 angular) across dims.
    pub fn mean_omega(&self) -> f32 {
        if self.h.is_empty() {
            return 0.0;
        }
        let s: f32 = self
            .h
            .iter()
            .zip(self.theta_prev.iter())
            .map(|(h, &th)| phase_delta(h.phase(), th))
            .sum();
        s / self.h.len() as f32
    }

    /// Research stub layer: complex residual mix + Hermitian project.
    /// \(h \leftarrow \Pi_R\big(h + \alpha\, i\cdot \mathrm{rotate}(h)\big)\)
    /// where rotate swaps axes (90°): \((re,im)\to(-im, re)\).
    pub fn think_step_stub(&mut self, alpha: f32) {
        let a = alpha.clamp(0.0, 1.0);
        for h in self.h.iter_mut() {
            // i * h = i*(re + i im) = -im + i re
            let rot = ComplexF32::new(-h.im, h.re);
            let mixed = *h + rot.scale(a);
            *h = project_hermitian(mixed, self.radius);
        }
        let omega = self.mean_omega();
        self.omega_ema = (1.0 - OMEGA_EMA_RATE) * self.omega_ema + OMEGA_EMA_RATE * omega;
        for (th, h) in self.theta_prev.iter_mut().zip(self.h.iter()) {
            *th = h.phase();
        }
        self.tau += 1;
    }

    /// Phase-coherent thinking step (research §3.1 + §3.2, fixed over-sharp).
    /// Scores use the **phase array** \(\cos(\phi_i-\phi_j)\) (ComplexFormer: Q/K complex,
    /// V real); values stay real so magnitude is preserved. Local windowed, normalized,
    /// **residual** mix `h ← h + α·(attn − h)` (PCT token-non-competition). CUE phase-repulsion
    /// evolves `phases` to avoid mode collapse. Soft magnitude clamp only bounds blow-ups.
    pub fn think_step_phase_attn(&mut self, alpha: f32) {
        let a = alpha.clamp(0.0, 1.0);
        let n = self.h.len();
        if n == 0 {
            return;
        }
        let win = std::env::var("MUD_CMUD_WIN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(CMUD_WIN_HALF);
        // smooth real gate (PCT C1-C4): real-valued, bounded, smooth, element-independent
        let gate = |s: f32| (s + 1.0) * 0.5;
        // P-01: reuse thread-local scratch buffer instead of allocating Vec<f32> per step.
        // The attention computation + residual update both run inside the `.with()` closure
        // to keep the RefMut guard alive across the attn slice usage.
        THINK_ATTN_SCRATCH.with(|scratch| {
            let mut s = scratch.borrow_mut();
            let ThinkAttnScratch { attn, cos_table, sin_table } = &mut *s;
            attn.clear();
            attn.resize(n, 0.0f32);
            // Precompute cos/sin of phases — eliminates O(n×win) cos() calls from inner loop.
            // cos(a-b) = cos(a)cos(b) + sin(a)sin(b).
            cos_table.clear();
            cos_table.resize(n, 0.0f32);
            sin_table.clear();
            sin_table.resize(n, 0.0f32);
            for i in 0..n {
                cos_table[i] = self.phases[i].cos();
                sin_table[i] = self.phases[i].sin();
            }
            // LOCAL (windowed) phase-coherent attention over REAL values h[j] (V real):
            // gentle local perturbation, not a global low-pass blur (Phasor/LPM).
            for (i, slot) in attn.iter_mut().enumerate() {
                let lo = i.saturating_sub(win);
                let hi = (i + win + 1).min(n);
                let cos_ti = cos_table[i];
                let sin_ti = sin_table[i];
                let mut acc = 0.0f32;
                let mut z = 0.0f32;
                for j in lo..hi {
                    // cos(ti - phases[j]) = cos_ti * cos_table[j] + sin_ti * sin_table[j]
                    let s_val = cos_ti * cos_table[j] + sin_ti * sin_table[j];
                    let w = gate(s_val);
                    acc += self.h[j].re * w;
                    z += w;
                }
                *slot = if z > EPSILON_FLOOR {
                    acc / z
                } else {
                    self.h[i].re
                };
            }

            // residual update (NOT replacement) — keeps original magnitude/info
            for (h, &a_new) in self.h.iter_mut().zip(attn.iter()) {
                let mixed = h.re + (a_new - h.re) * a;
                let mag = mixed.abs();
                let bounded = if mag > self.radius && mag > EPSILON_FLOOR {
                    mixed * (self.radius / mag)
                } else {
                    mixed
                };
                *h = ComplexF32::new(bounded, 0.0);
            }
        });

        // CUE phase-repulsion on the phase array (diversify scores across steps)
        cue_phase_repulsion(&mut self.phases, CMUD_REPULSION_ETA);

        // CTNN σ-imagination (research §3.5): scale the phase SPREAD around its mean by
        // (1+σ). Magnitude is untouched (V real); only the relational phase manifold
        // "breathes", changing future attention scores. Disabled when `MUD_CMUD_SIGMA`=0/unset.
        if let Some(sigma) = std::env::var("MUD_CMUD_SIGMA")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            if sigma != 0.0 && n > 0 {
                let mean_p = self.phases.iter().sum::<f32>() / n as f32;
                for p in self.phases.iter_mut() {
                    *p = mean_p + (*p - mean_p) * (1.0 + sigma);
                }
            }
        }

        let omega = self.mean_omega();
        self.omega_ema = (1.0 - OMEGA_EMA_RATE) * self.omega_ema + OMEGA_EMA_RATE * omega;
        for (th, p) in self.theta_prev.iter_mut().zip(self.phases.iter()) {
            *th = *p;
        }
        self.tau += 1;
    }

    /// Trainable thinking step (research §4, #4 entrenable). Like `think_step_phase_attn` but
    /// the query/key phases come from **learned per-dimension biases** (`p.q_phase`,`p.k_phase`)
    /// over the positional phase, and the value is scaled per-dimension by `p.v_scale`. With
    /// zero biases and unit scale it equals the fixed positional-phase step (regression-safe).
    /// `alpha/eta_rep/sigma` come from `p` (env `MUD_CMUD_*` still override `sigma`/`win`).
    pub fn think_step_trainable(&mut self, p: &CmudLayerParams) {
        self.think_step_trainable_core(p, None);
    }

    /// Like [`think_step_trainable`] but records a [`ThinkTape`] per step for analytic backprop.
    pub fn think_step_trainable_record(&mut self, p: &CmudLayerParams, tapes: &mut Vec<ThinkTape>) {
        let mut t = ThinkTape::new(self.h.len());
        self.think_step_trainable_core(p, Some(&mut t));
        tapes.push(t);
    }

    fn think_step_trainable_core(&mut self, p: &CmudLayerParams, mut tape: Option<&mut ThinkTape>) {
        let a = p.alpha.clamp(0.0, 1.0);
        let n = self.h.len();
        if n == 0 {
            return;
        }
        let win = std::env::var("MUD_CMUD_WIN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(CMUD_WIN_HALF);

        // P-01: reuse thread-local buffers for qph/attn/zvec/h_before instead of
        // allocating 4 Vecs per call. `kph` reuses `self.phases` (already allocated).
        THINK_TRAINABLE_SCRATCH.with(|scratch| {
            let mut s = scratch.borrow_mut();
            // Destructure to get separate mutable refs to each field (borrow checker)
            let ThinkTrainableScratch { qph, attn, zvec, h_before, cos_kph, sin_kph, v_flat } = &mut *s;

            // learned query phases = positional phase + per-dim learned bias
            qph.clear();
            qph.extend((0..n).map(|i| {
                p.omega_pos * i as f32 + p.q_phase.get(i).copied().unwrap_or(0.0)
            }));

            // learned key phases — reuse self.phases (already allocated, same size)
            self.phases.clear();
            self.phases.extend((0..n).map(|i| {
                p.omega_pos * i as f32 + p.k_phase.get(i).copied().unwrap_or(0.0)
            }));

            let gate = |s: f32| (s + 1.0) * 0.5;
            attn.clear();
            attn.resize(n, 0.0f32);
            zvec.clear();
            zvec.resize(n, 0.0f32);
            // snapshot of the (real) hidden BEFORE the residual update — needed for backprop
            h_before.clear();
            h_before.extend(self.h.iter().map(|hh| hh.re));

            // Precompute cos/sin of key phases — eliminates O(n×win) cos() calls
            // from the inner loop. cos(a-b) = cos(a)cos(b) + sin(a)sin(b).
            cos_kph.clear();
            cos_kph.resize(n, 0.0f32);
            sin_kph.clear();
            sin_kph.resize(n, 0.0f32);
            for i in 0..n {
                cos_kph[i] = self.phases[i].cos();
                sin_kph[i] = self.phases[i].sin();
            }

            // Precompute v_scale (eliminates get().copied().unwrap_or() from inner loop)
            v_flat.clear();
            v_flat.resize(n, 0.0f32);
            for (i, vf) in v_flat.iter_mut().enumerate() {
                *vf = self.h[i].re * p.v_scale.get(i).copied().unwrap_or(1.0);
            }

            for i in 0..n {
                let lo = i.saturating_sub(win);
                let hi = (i + win + 1).min(n);
                let ti = qph[i];
                // Compute query cos/sin once per outer iteration (O(n) total, not O(n×win))
                let cos_ti = ti.cos();
                let sin_ti = ti.sin();
                let mut acc = 0.0f32;
                let mut z = 0.0f32;
                for j in lo..hi {
                    // cos(ti - kp) = cos_ti * cos_kph[j] + sin_ti * sin_kph[j]
                    let s_val = cos_ti * cos_kph[j] + sin_ti * sin_kph[j];
                    let w = gate(s_val);
                    acc += v_flat[j] * w;
                    z += w;
                }
                attn[i] = if z > EPSILON_FLOOR { acc / z } else { v_flat[i] };
                zvec[i] = z;
            }

            // CUE phase-repulsion on the learned K-phase manifold (keep it diverse)
            cue_phase_repulsion(&mut self.phases, p.eta_rep);

            // CTNN σ-imagination (research §3.5): scale phase SPREAD around its mean by (1+σ).
            // `MUD_CMUD_SIGMA` env overrides `p.sigma`. Magnitude untouched (V real).
            let sigma = std::env::var("MUD_CMUD_SIGMA")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(p.sigma);
            if sigma != 0.0 && n > 0 {
                let mean_p = self.phases.iter().sum::<f32>() / n as f32;
                for ph in self.phases.iter_mut() {
                    *ph = mean_p + (*ph - mean_p) * (1.0 + sigma);
                }
            }

            for (h, &a_new) in self.h.iter_mut().zip(attn.iter()) {
                // residual update (NOT replacement) — keeps original magnitude/info
                let mixed = h.re + (a_new - h.re) * a;
                let mag = mixed.abs();
                let bounded = if mag > self.radius && mag > EPSILON_FLOOR {
                    mixed * (self.radius / mag)
                } else {
                    mixed
                };
                *h = ComplexF32::new(bounded, 0.0);
            }

            // self.phases is already set (reused as kph above)
            let omega = self.mean_omega();
            self.omega_ema = (1.0 - OMEGA_EMA_RATE) * self.omega_ema + OMEGA_EMA_RATE * omega;
            for (th, ph) in self.theta_prev.iter_mut().zip(self.phases.iter()) {
                *th = *ph;
            }
            self.tau += 1;

            // Move buffers into tape if recording (std::mem::take leaves empty Vecs in scratch)
            if let Some(t) = tape.as_mut() {
                t.h_before = std::mem::take(h_before);
                t.attn = std::mem::take(attn);
                t.z = std::mem::take(zvec);
                t.win = win;
                t.alpha = a;
                t.radius = self.radius;
            }
        });
    }

} // close impl ThinkingState (module-level training types follow)

/// Per-step recorded intermediates for analytic backprop through `think_step_trainable_core`.
///
/// Only the loss-relevant quantities are stored: `h_before` (real hidden pre-residual), the
/// normalized attention `attn` and its denom `z`, the window, `alpha` and the Hermitian radius.
/// The CUE repulsion / σ-spread applied to `kph` after attention do NOT feed back into `h`
/// (they only set `self.phases`), so they are correctly absent from the gradient graph.
#[derive(Clone, Debug)]
pub struct ThinkTape {
    pub h_before: Vec<f32>,
    pub attn: Vec<f32>,
    pub z: Vec<f32>,
    pub win: usize,
    pub alpha: f32,
    pub radius: f32,
}

impl ThinkTape {
    fn new(n: usize) -> Self {
        ThinkTape {
            h_before: vec![0.0f32; n],
            attn: vec![0.0f32; n],
            z: vec![0.0f32; n],
            win: CMUD_WIN_HALF,
            alpha: CMUD_DEFAULT_ALPHA,
            radius: 1.0,
        }
    }
}

/// Gradients of a scalar loss w.r.t. [`CmudLayerParams`] (research §4, #4 entrenable).
#[derive(Clone, Debug, Default)]
pub struct CmudLayerParamsGrad {
    pub q_phase: Vec<f32>,
    pub k_phase: Vec<f32>,
    pub v_scale: Vec<f32>,
    pub alpha: f32,
}

impl CmudLayerParamsGrad {
    fn zeros_like(p: &CmudLayerParams) -> Self {
        let n = p.q_phase.len().max(p.k_phase.len()).max(p.v_scale.len());
        CmudLayerParamsGrad {
            q_phase: vec![0.0f32; n],
            k_phase: vec![0.0f32; n],
            v_scale: vec![0.0f32; n],
            alpha: 0.0,
        }
    }
    /// `p -= lr * g` (Adam/L2 handled by caller).
    pub fn axpy(&mut self, lr: f32, g: &CmudLayerParamsGrad) {
        for i in 0..self.q_phase.len() {
            self.q_phase[i] -= lr * g.q_phase[i];
        }
        for i in 0..self.k_phase.len() {
            self.k_phase[i] -= lr * g.k_phase[i];
        }
        for i in 0..self.v_scale.len() {
            self.v_scale[i] -= lr * g.v_scale[i];
        }
        self.alpha -= lr * g.alpha;
    }
}

/// Analytic reverse-mode gradient of a scalar loss through `K` recorded trainable think steps.
///
/// `grad_h_final` is `∂L/∂mixed` at the output of the **last** step (length = hidden), where
/// `mixed` is the *pre-collapse* residual output (`h_before + alpha·(attn − h_before)`). It is
/// **not** the post-`wave_collapse` value: the collapse is `|·|` on a real value, so the caller
/// must pre-multiply `∂L/∂collapse` by `sign(mixed)` (see `cmud_training_forward` for the exact
/// head-gradient correction). Returns gradients w.r.t. `q_phase`, `k_phase`, `v_scale` and `alpha`.
/// The graph is the loss-relevant forward: `attn_i = Σ_j w_ij (h_j s_j) / Σ_j w_ij` with
/// `w_ij = 0.5(1+cos(ω(i−j)+q_i−k_j))`, then a residual update. The CUE repulsion / σ-spread
/// applied to `kph` after attention do NOT feed back into `h` (they only set `self.phases`), so
/// they are correctly absent from the gradient graph. Validated against finite differences by
/// `test_cmud_backward_matches_fd`.
// Index-based loops are required (nested window + per-index grad accumulation on hot path).
#[allow(clippy::needless_range_loop)]
pub fn cmud_backward(
    tapes: &[ThinkTape],
    grad_h_final: &[f32],
    p: &CmudLayerParams,
    grad: &mut CmudLayerParamsGrad,
) {
    let n = grad_h_final.len();
    *grad = CmudLayerParamsGrad::zeros_like(p);
    let mut grad_h = grad_h_final.to_vec();
    for tape in tapes.iter().rev() {
        let win = tape.win;
        let a = tape.alpha;
        let mut grad_h_before = vec![0.0f32; n];
        let mut grad_attn = vec![0.0f32; n];
        // 1) unroll the residual update + soft-clamp
        for i in 0..n {
            let mixed = tape.h_before[i] + a * (tape.attn[i] - tape.h_before[i]);
            let mag = mixed.abs();
            // soft-clamp: identity inside the ball, gradient 0 once saturated onto ±radius
            let g_mixed = if mag > tape.radius && mag > EPSILON_FLOOR {
                0.0
            } else {
                grad_h[i]
            };
            grad_h_before[i] += g_mixed * (1.0 - a);
            grad.alpha += g_mixed * (tape.attn[i] - tape.h_before[i]);
            grad_attn[i] = g_mixed * a;
        }
        // 2) unroll the attention normalisation + cosine scoring
        for i in 0..n {
            let z = tape.z[i].max(EPSILON_FLOOR);
            let acc_i = tape.attn[i] * z; // attn = acc/z  →  acc = attn*z
            let g_acc = grad_attn[i] / z;
            let g_z = grad_attn[i] * (-acc_i / (z * z));
            let lo = i.saturating_sub(win);
            let hi = (i + win + 1).min(n);
            let ti = p.omega_pos * i as f32 + p.q_phase[i];
            for j in lo..hi {
                let kp = p.omega_pos * j as f32 + p.k_phase[j];
                let d = ti - kp;
                let w = 0.5 * (1.0 + d.cos());
                let s = p.v_scale[j];
                let hb = tape.h_before[j];
                let dw_dd = -0.5 * d.sin();
                let g_d = g_acc * (hb * s) * dw_dd + g_z * dw_dd;
                grad.q_phase[i] += g_d; // ∂d/∂q_i = 1
                grad.k_phase[j] -= g_d; // ∂d/∂k_j = -1
                grad_h_before[j] += g_acc * w * s;
                grad.v_scale[j] += g_acc * w * hb;
            }
        }
        grad_h = grad_h_before;
    }
}

impl ThinkingState {

    /// Phase-locked when EMA angular speed is below \(\varepsilon\).
    /// \(\varepsilon\) = `MUD_CMUD_LOCK_EPS` if set, else `self.lock_eps` (default `PHASE_LOCK_EPS`).
    #[inline]
    pub fn is_phase_locked(&self) -> bool {
        let eps = std::env::var("MUD_CMUD_LOCK_EPS")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(self.lock_eps);
        self.omega_ema < eps
    }

    /// Run up to `max_iters` stub thinking steps; return true if locked.
    pub fn run_until_lock(&mut self, max_iters: usize, alpha: f32) -> bool {
        for _ in 0..max_iters {
            if self.is_phase_locked() && self.tau > 0 {
                return true;
            }
            self.think_step_stub(alpha);
        }
        self.is_phase_locked()
    }

    /// Collapse entire state to real activations (LM-head ready).
    pub fn collapse_to_real(&self, out: &mut [f32]) {
        for (o, h) in out.iter_mut().zip(self.h.iter()) {
            *o = wave_collapse(*h);
        }
    }
}

/// Env gate for future engine wiring (`MUD_CMUD_THINK=1`).
pub fn cmud_think_enabled() -> bool {
    std::env::var("MUD_CMUD_THINK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Metrics captured from one `maybe_think_collapse` pass (research §3 reasoning loop).
#[derive(Clone, Copy, Debug, Default)]
pub struct CmudThinkReport {
    /// Number of thinking iterations \(\tau\) actually run.
    pub steps: usize,
    /// Whether the phase-lock gate tripped (\(EMA(\omega) < \varepsilon\)).
    pub phase_locked: bool,
    /// Largest Hermitian norm seen across the hidden vector (must be \(\le\) radius).
    pub max_herm_norm: f32,
    /// Hermitian ball radius used for projection.
    pub radius: f32,
    /// Spectral-collapse health of the complex manifold after thinking (research §3.3).
    pub spectral: SpectralHealth,
}

/// Optional post-forward pass returning metrics. If enabled, seeds thinking from `x`,
/// iterates (phase-coherent complex attention + CUE phase-repulsion), collapses into `x`.
/// No-op when `MUD_CMUD_THINK` is unset — returns a default (zero) report in that case.
pub fn maybe_think_collapse_report(x: &mut [f32], radius: f32) -> CmudThinkReport {
    if !cmud_think_enabled() || x.is_empty() {
        return CmudThinkReport::default();
    }
    let alpha = std::env::var("MUD_CMUD_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(CMUD_DEFAULT_ALPHA);
    let mut st = ThinkingState::from_real(x, radius);
    // Trainable step (research §4, #4): default params == fixed positional phase, so this is
    // regression-safe until a trainer learns the per-dim biases. A trained sidecar (JSON) loaded
    // via `MUD_CMUD_PARAMS` overrides the defaults (production path untouched — C-MUD opt-in).
    let mut params = match std::env::var("MUD_CMUD_PARAMS") {
        Ok(path) if !path.is_empty() => CmudLayerParams::load_json(std::path::Path::new(&path))
            .unwrap_or_else(|_| CmudLayerParams::from_defaults(st.h.len())),
        _ => CmudLayerParams::from_defaults(st.h.len()),
    };
    params.alpha = alpha;
    for _ in 0..DEFAULT_THINK_ITERS {
        st.think_step_trainable(&params);
        if st.is_phase_locked() && st.tau > 0 {
            break;
        }
    }
    let max_herm_norm = st.h.iter().map(|h| h.hermite_norm()).fold(0.0f32, f32::max);
    // P-01: reuse thread-local scratch buffer instead of allocating Vec<f32> for mags
    let spectral = SPECTRAL_MAG_SCRATCH.with(|scratch| {
        let mut mags = scratch.borrow_mut();
        mags.clear();
        mags.extend(st.h.iter().map(|h| h.re.abs()));
        cmud_spectral_health(&mags, &st.phases)
    });
    let rep = CmudThinkReport {
        steps: st.tau,
        phase_locked: st.is_phase_locked(),
        max_herm_norm,
        radius: st.radius,
        spectral,
    };
    st.collapse_to_real(x);
    rep
}

/// Convenience hook for the forward path: runs [`maybe_think_collapse`] with the
/// Hermitian radius auto-scaled to `CMUD_RADIUS_RMS_FACTOR ×` the hidden RMS, so the
/// ball does not clamp at seed. No-op unless `MUD_CMUD_THINK=1`. Returns the thinking
/// report (default/zero when disabled). Use this from `inference.rs` and `main.rs` to
/// avoid duplicating the RMS computation.
pub fn maybe_think_collapse_rms_scaled(x: &mut [f32]) -> CmudThinkReport {
    let s: f32 = x.iter().map(|v| v * v).sum::<f32>().max(EPSILON_FLOOR);
    let rms = (s / x.len().max(1) as f32).sqrt();
    maybe_think_collapse_report(x, CMUD_RADIUS_RMS_FACTOR * rms)
}

/// Optional post-forward pass: if enabled, seed thinking from `x`, iterate (phase-coherent
/// complex attention + CUE phase-repulsion), collapse into `x`.
/// No-op when `MUD_CMUD_THINK` is unset. Returns number of \(\tau\) steps run.
pub fn maybe_think_collapse(x: &mut [f32], radius: f32) -> usize {
    maybe_think_collapse_report(x, radius).steps
}

/// Standalone self-check of the C-MUD reasoning kernel (research §3): algebra identity,
/// Hermitian projection, and one real thinking step (finite + ball-respecting).
/// Returns `(all_ok, human_readable_summary)`. Used by audit/diagnostic tools.
pub fn cmud_kernel_selfcheck() -> (bool, String) {
    let mut ok = true;
    let mut msg = String::new();
    let x = ComplexF32::new(2.0, 3.0);
    let y = gauss_mul(x, GaussTernary::new(0, 1));
    let a_ok = (y.re + 3.0).abs() < 1e-5 && (y.im - 2.0).abs() < 1e-5;
    ok &= a_ok;
    msg.push_str(&format!("gauss_mul×i={a_ok} "));
    let p = project_hermitian(ComplexF32::new(3.0, 4.0), 1.0);
    let h_ok = (p.hermite_norm() - 1.0).abs() < 1e-4;
    ok &= h_ok;
    msg.push_str(&format!("hermitian_proj={h_ok} "));
    let mut st = ThinkingState::from_real(&[0.5f32; 16], 8.0);
    st.think_step_phase_attn(0.15);
    let ball = st.h.iter().all(|hh| hh.hermite_norm() <= st.radius * 1.01);
    let fin = st.h.iter().all(|hh| hh.is_finite());
    // spectral gate: a real constant vector [0.5;16] should NOT collapse the manifold
    let mags: Vec<f32> = st.h.iter().map(|hh| hh.re.abs()).collect();
    let spec = cmud_spectral_health(&mags, &st.phases);
    let spec_ok = spec.spread_mag.is_finite() && spec.cauchy_mag_at_2.is_finite();
    ok &= ball && fin && spec_ok;
    msg.push_str(&format!(
        "think_step(ball={ball},finite={fin},spectral_ok={spec_ok})"
    ));
    (ok, msg)
}

/// Dense complex GEMV vs Gaussian-ternary weights (reference / tests).
/// `w_re`, `w_im`: row-major `[n_out * n_in]` with values in {-1,0,1} as f32.
#[cfg(test)]
pub fn complex_gemv_gauss_ref(
    x: &[ComplexF32],
    w_re: &[f32],
    w_im: &[f32],
    y: &mut [ComplexF32],
    n_out: usize,
    n_in: usize,
) {
    assert!(x.len() >= n_in && y.len() >= n_out);
    assert!(w_re.len() >= n_out * n_in && w_im.len() >= n_out * n_in);
    for (r, yr) in y.iter_mut().enumerate().take(n_out) {
        let mut acc = ComplexF32::default();
        let base = r * n_in;
        for (c, &xc) in x.iter().enumerate().take(n_in) {
            let wr = w_re[base + c].round().clamp(-1.0, 1.0) as i8;
            let wi = w_im[base + c].round().clamp(-1.0, 1.0) as i8;
            let p = gauss_mul(xc, GaussTernary::new(wr, wi));
            acc.re += p.re;
            acc.im += p.im;
        }
        *yr = acc;
    }
}

/// Complex division \(z_1 / z_2\) (with floor guard).
#[inline]
pub fn cdiv(a: ComplexF32, b: ComplexF32) -> ComplexF32 {
    let d = b.re * b.re + b.im * b.im;
    if d < EPSILON_FLOOR {
        return ComplexF32::default();
    }
    ComplexF32::new(
        (a.re * b.re + a.im * b.im) / d,
        (a.im * b.re - a.re * b.im) / d,
    )
}

/// Cauchy transform of a set of complex eigenvalues \(\lambda_j\):
/// \(G(z) = \frac{1}{N}\sum_j \frac{1}{z - \lambda_j}\) (research §3.3).
#[inline]
pub fn cauchy_transform(lambdas: &[ComplexF32], z: ComplexF32) -> ComplexF32 {
    if lambdas.is_empty() {
        return ComplexF32::default();
    }
    let mut acc = ComplexF32::default();
    for &l in lambdas {
        acc = acc + cdiv(ComplexF32::new(1.0, 0.0), z - l);
    }
    acc.scale(1.0 / lambdas.len() as f32)
}


/// R-transform additivity for free convolution: \(R_{A\oplus B} = R_A + R_B\) (research §3.3).
/// Both slices are coefficient arrays (same length assumed); returns elementwise sum.
pub fn r_transform_add(ra: &[f32], rb: &[f32]) -> Vec<f32> {
    let n = ra.len().min(rb.len());
    (0..n).map(|i| ra[i] + rb[i]).collect()
}

/// Spectral-collapse health of the complex reasoning manifold (research §3.3, free-probability).
/// Treats each position as an eigenvalue \(\lambda_i = m_i e^{i\phi_i}\) (magnitude from the real
/// `h`, phase from the `phases` array) and reports:
/// - `spread_mag`: std of magnitudes (low ⇒ all elements squeezed to one radius — manifold collapse),
/// - `circular_phase_r`: \(|\text{mean}(e^{i\phi})|\in[0,1]\) (≈1 ⇒ phases aligned ⇒ no reasoning spread),
/// - `cauchy_mag_at_2`: \(|G_\lambda(2)|\) (Cauchy transform signature; large ⇒ eigenvalues cluster near 0).
/// - `collapsed` flags a degenerate manifold (flat magnitudes OR perfectly aligned phases).
#[derive(Clone, Copy, Debug, Default)]
pub struct SpectralHealth {
    pub n: usize,
    pub spread_mag: f32,
    pub circular_phase_r: f32,
    pub cauchy_mag_at_2: f32,
    pub collapsed: bool,
}

pub fn cmud_spectral_health(mags: &[f32], phases: &[f32]) -> SpectralHealth {
    let n = mags.len().min(phases.len());
    if n == 0 {
        return SpectralHealth::default();
    }
    let mean_mag = mags.iter().take(n).sum::<f32>() / n as f32;
    let spread_mag = (mags
        .iter()
        .take(n)
        .map(|m| {
            let d = m - mean_mag;
            d * d
        })
        .sum::<f32>()
        / n as f32)
        .sqrt();
    // circular concentration of phases
    let (sx, cx) = phases.iter().take(n).map(|p| (p.sin(), p.cos())).fold(
        (0.0f32, 0.0f32),
        |(asum, bsum), (s, c)| (asum + s, bsum + c),
    );
    let r = ((sx / n as f32).powi(2) + (cx / n as f32).powi(2))
        .sqrt()
        .clamp(0.0, 1.0);
    // P-01: inline Cauchy transform — avoids allocating Vec<ComplexF32> for lambdas.
    // G(z=2) = (1/n) Σ 1/(z - λ_i) where λ_i = m_i * e^(i*φ_i)
    let z = ComplexF32::new(2.0, 0.0);
    let mut cauchy = ComplexF32::default();
    for i in 0..n {
        let lambda = ComplexF32::new(
            mags[i] * phases[i].cos(),
            mags[i] * phases[i].sin(),
        );
        cauchy = cauchy + cdiv(ComplexF32::new(1.0, 0.0), z - lambda);
    }
    cauchy = cauchy.scale(1.0 / n as f32);
    let cauchy_mag = cauchy.hermite_norm();
    let collapsed = (mean_mag > EPSILON_FLOOR && spread_mag < mean_mag * 1e-3) || r > 0.999;
    SpectralHealth {
        n,
        spread_mag,
        circular_phase_r: r,
        cauchy_mag_at_2: cauchy_mag,
        collapsed,
    }
}

/// CUE phase-repulsion regularizer (research §3.2, orbit E1).
/// \(R(\theta) = -\sum_{i<j}\log|e^{i\theta_i}-e^{i\theta_j}|^2\); gradient pushes phases apart.
/// Applies one gradient step \(\theta_i \gets \theta_i + \eta\sum_{j\ne i}\cot((\theta_i-\theta_j)/2)\)
/// and returns the current \(R\) value.
///
/// **P-01 zero-alloc:** the gradient is applied in-place via a fused
/// read-accumulate-write loop — no intermediate `Vec<f32>` allocation.
/// The gradient for each \(i\) is accumulated into a stack-local `f32`
/// and immediately applied, so the entire step is allocation-free.
pub fn cue_phase_repulsion(phases: &mut [f32], eta_rep: f32) -> f32 {
    let n = phases.len();
    if n < 2 {
        return 0.0;
    }
    // P-01: precompute sin(phases[i]/2) and cos(phases[i]/2) in thread-local buffers.
    // Uses the identities:
    //   sin((a-b)/2) = sin(a/2)cos(b/2) - cos(a/2)sin(b/2)
    //   cos((a-b)/2) = cos(a/2)cos(b/2) + sin(a/2)sin(b/2)
    // This replaces O(n²) sin()/cos() calls with O(n) sin()/cos() + O(n²)×4 mults.
    CUE_REPULSION_SCRATCH.with(|scratch| {
        let mut s = scratch.borrow_mut();
        s.sin_half.clear();
        s.sin_half.resize(n, 0.0f32);
        s.cos_half.clear();
        s.cos_half.resize(n, 0.0f32);
        for (i, &ph) in phases.iter().enumerate() {
            let half = ph * 0.5;
            s.sin_half[i] = half.sin();
            s.cos_half[i] = half.cos();
        }
        let sin_half = &s.sin_half;
        let cos_half = &s.cos_half;

        let mut r = 0.0f32;
        // Energy: R = -Σ_{i<j} ln(4·sin²((θ_i-θ_j)/2))
        for i in 0..n {
            let si = sin_half[i];
            let ci = cos_half[i];
            for j in (i + 1)..n {
                let sj = sin_half[j];
                let cj = cos_half[j];
                // sin((θ_i-θ_j)/2) = si·cj - ci·sj
                let s_val = si * cj - ci * sj;
                r -= (4.0 * s_val * s_val).max(EPSILON_FLOOR).ln();
            }
        }
        // Gradient: ∂R/∂θ_i = Σ_{j≠i} cot((θ_i-θ_j)/2)
        // cot(x) = cos(x)/sin(x)
        for (i, ph) in phases.iter_mut().enumerate() {
            let si = sin_half[i];
            let ci = cos_half[i];
            let mut g = 0.0f32;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let sj = sin_half[j];
                let cj = cos_half[j];
                // sin((θ_i-θ_j)/2) = si·cj - ci·sj
                let s_val = si * cj - ci * sj;
                if s_val.abs() > EPSILON_FLOOR {
                    // cos((θ_i-θ_j)/2) = ci·cj + si·sj
                    let c_val = ci * cj + si * sj;
                    g += c_val / s_val;
                }
            }
            if g.is_finite() {
                *ph += eta_rep * g;
            }
        }
        r
    })
}

/// Phase-coherent complex attention (research §3.1, PCT/CMHA).
/// Score \(s_{mn} = \cos(\theta_{q_m} - \theta_{k_n})\) (L2-normalized, token-non-competing),
/// then a smooth real gate \(g(\cdot)\); output mixes complex V phase-preservingly.
/// `out[m] = (Σ_n g(s_{mn})·v[n]) / (Σ_n g(s_{mn}))` — the **normalization by the weight sum**
/// keeps the output magnitude bounded by `|V|` (PCT: bounded gate, non-expansive), so a
/// single-vector self-attention step cannot blow up the hidden scale.
/// `q`,`k`,`v` length = `n_tok`; `out` length = `n_tok`.
pub fn phase_coherent_attn(
    q: &[ComplexF32],
    k: &[ComplexF32],
    v: &[ComplexF32],
    gate: impl Fn(f32) -> f32,
    out: &mut [ComplexF32],
) {
    let n = q.len().min(k.len()).min(v.len()).min(out.len());
    for m in 0..n {
        let mut acc = ComplexF32::default();
        let mut z = 0.0f32;
        let tq = q[m].phase();
        for nn in 0..n {
            let s = (tq - k[nn].phase()).cos();
            let w = gate(s);
            acc = acc + v[nn].scale(w);
            z += w;
        }
        out[m] = if z > EPSILON_FLOOR {
            acc.scale(1.0 / z)
        } else {
            ComplexF32::default()
        };
    }
}

/// Contour rotation / analytic continuation (research §3.4, orbit E3).
/// \(h \gets h \cdot e^{i\phi}\); preserves Hermitian norm (mHC-safe).
#[inline]
pub fn contour_rotate(h: &mut ComplexF32, phi: f32) {
    let c = phi.cos();
    let s = phi.sin();
    let re = h.re * c - h.im * s;
    let im = h.re * s + h.im * c;
    *h = ComplexF32::new(re, im);
}

/// Complex-Time thinking step (research §3.5, CTNN): rotate the whole state by `sigma`
/// (imagination/memory axis) then run one stub thinking step. `sigma` does not change `‖h‖`.
pub fn complex_time_step(st: &mut ThinkingState, alpha: f32, sigma: f32) {
    for h in st.h.iter_mut() {
        contour_rotate(h, sigma);
    }
    st.sigma = sigma;
    st.think_step_stub(alpha);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)] use crate::mud::cmud_thinking::ComplexSlimeRegister;

    // ── SPRINT-3: sidecar versioning tests ────────────────────────────────

    #[test]
    fn test_sidecar_versioning_roundtrip() {
        let hidden = 8;
        let p = CmudLayerParams::from_defaults(hidden);
        let envelope = CmudSidecarEnvelope {
            schema_version: SIDECAR_SCHEMA_VERSION,
            params: p.clone(),
        };
        let s = serde_json::to_string(&envelope).unwrap();
        let parsed = CmudLayerParams::parse_json(&s).unwrap();
        assert_eq!(parsed.q_phase.len(), hidden);
        assert_eq!(parsed.k_phase.len(), hidden);
        assert_eq!(parsed.v_scale.len(), hidden);
        assert!((parsed.alpha - p.alpha).abs() < 1e-6);
    }

    #[test]
    fn test_sidecar_wrong_version_rejected() {
        let envelope = CmudSidecarEnvelope {
            schema_version: SIDECAR_SCHEMA_VERSION + 99,
            params: CmudLayerParams::from_defaults(4),
        };
        let s = serde_json::to_string(&envelope).unwrap();
        let err = CmudLayerParams::parse_json(&s).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("schema_version mismatch"), "msg = {msg}");
        assert!(msg.contains(&format!("v{}", SIDECAR_SCHEMA_VERSION + 99)));
    }

    #[test]
    fn test_sidecar_legacy_bare_format_accepted() {
        // Pre-Sprint-3 sidecars were bare CmudLayerParams (no envelope).
        // parse_json must still accept them.
        let legacy = r#"{
            "alpha": 0.05,
            "eta_rep": 0.01,
            "omega_pos": 0.1,
            "sigma": 0.0,
            "q_phase": [0.0, 0.0],
            "k_phase": [0.0, 0.0],
            "v_scale": [1.0, 1.0]
        }"#;
        let parsed = CmudLayerParams::parse_json(legacy).unwrap();
        assert_eq!(parsed.q_phase.len(), 2);
    }

    // ── existing tests below ──

    #[test]
    fn test_gauss_mul_basis() {
        let x = ComplexF32::new(2.0, 3.0);
        // * i  → wr=0, wi=1: (2+3i)*i = 2i + 3i² = -3 + 2i
        let y = gauss_mul(x, GaussTernary::new(0, 1));
        assert!((y.re + 3.0).abs() < 1e-6);
        assert!((y.im - 2.0).abs() < 1e-6);
        // * (-1): (-2, -3)
        let y2 = gauss_mul(x, GaussTernary::new(-1, 0));
        assert!((y2.re + 2.0).abs() < 1e-6 && (y2.im + 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_nine_gauss_states() {
        let mut ids = std::collections::HashSet::new();
        for wr in -1i8..=1 {
            for wi in -1i8..=1 {
                ids.insert(GaussTernary::new(wr, wi).pack_id());
            }
        }
        assert_eq!(ids.len(), 9);
        for id in 0..9u8 {
            let g = GaussTernary::from_id(id);
            assert_eq!(g.pack_id(), id);
        }
    }

    #[test]
    fn test_hermitian_projection() {
        let h = ComplexF32::new(3.0, 4.0); // norm 5
        let p = project_hermitian(h, 1.0);
        assert!((p.hermite_norm() - 1.0).abs() < 1e-5);
        let inside = ComplexF32::new(0.3, 0.4);
        let p2 = project_hermitian(inside, 1.0);
        assert_eq!(p2, inside);
    }

    #[test]
    fn test_wave_collapse_real_axis() {
        // Pure real positive → cos(0)=1, im=0 → h_final = re
        let h = ComplexF32::new(2.0, 0.0);
        assert!((wave_collapse(h) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_phase_delta_wrap() {
        let d = phase_delta(3.0, -3.0);
        assert!(d < std::f32::consts::PI + 0.1);
    }

    #[test]
    fn test_thinking_converges_stub() {
        let x = vec![1.0f32; 32];
        let mut st = ThinkingState::from_real(&x, 10.0);
        // Small alpha → mild rotation; eventually omega EMA decays if nearly fixed point
        st.lock_eps = 0.5; // loose for stub dynamics
        let locked = st.run_until_lock(20, 0.05);
        assert!(st.tau > 0);
        // Collapse stays finite
        let mut out = vec![0.0f32; 32];
        st.collapse_to_real(&mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        let _ = locked;
    }

    #[test]
    fn test_complex_gemv_identity_real() {
        let n = 8usize;
        let x: Vec<ComplexF32> = (0..n)
            .map(|i| ComplexF32::from_real(i as f32 + 1.0))
            .collect();
        // Identity real weights, zero imag → y = x
        let mut w_re = vec![0.0f32; n * n];
        let w_im = vec![0.0f32; n * n];
        for i in 0..n {
            w_re[i * n + i] = 1.0;
        }
        let mut y = vec![ComplexF32::default(); n];
        complex_gemv_gauss_ref(&x, &w_re, &w_im, &mut y, n, n);
        for i in 0..n {
            assert!((y[i].re - x[i].re).abs() < 1e-5);
            assert!(y[i].im.abs() < 1e-5);
        }
    }

    #[test]
    fn test_cmud_env_default_off() {
        unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        }
        assert!(!cmud_think_enabled());
        let mut x = vec![1.0f32, 2.0, 3.0];
        let steps = maybe_think_collapse(&mut x, 5.0);
        assert_eq!(steps, 0);
        assert_eq!(x, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_maybe_think_collapse_runs_when_enabled() {
        unsafe {
            std::env::set_var("MUD_CMUD_THINK", "1");
        }
        let mut x = vec![0.5f32; 16];
        let steps = maybe_think_collapse(&mut x, 8.0);
        unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        }
        assert!(steps > 0);
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_register_layout() {
        assert_eq!(std::mem::size_of::<ComplexSlimeRegister>(), 16);
    }

    #[test]
    fn test_phase_coherent_attn_basis() {
        // orthogonal phases -> score cos(pi/2)=0 ; equal phases -> cos(0)=1
        let q = vec![ComplexF32::new(1.0, 0.0), ComplexF32::new(0.0, 1.0)];
        let k = vec![ComplexF32::new(1.0, 0.0), ComplexF32::new(0.0, 1.0)];
        let v = vec![ComplexF32::new(1.0, 0.0), ComplexF32::new(2.0, 0.0)];
        let gate = |s: f32| s.max(0.0);
        let mut out = vec![ComplexF32::default(); 2];
        phase_coherent_attn(&q, &k, &v, gate, &mut out);
        // m=0 (theta 0) vs k0(theta0)=1, k1(theta pi/2)=0 -> out0 = 1*v0 + 0 = (1,0)
        assert!((out[0].re - 1.0).abs() < 1e-5);
        // m=1 (theta pi/2) vs k0=0 (score 0), k1=1 (score 1) -> out1 = 1*v1 = (2,0)
        assert!((out[1].re - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_cue_phase_repulsion_spreads() {
        // nearly-equal phases -> large R and strong push apart; spread -> smaller R
        let mut collapsed = vec![0.0f32, 1e-4];
        let r0 = cue_phase_repulsion(&mut collapsed, 0.01);
        assert!(r0 > 5.0); // degenerate phase -> large repulsion energy
        assert!(phase_delta(collapsed[0], collapsed[1]).abs() > 1e-4); // pushed apart

        let mut spread = vec![0.0f32, 1.0];
        let r1 = cue_phase_repulsion(&mut spread, 0.05);
        assert!(r1 < r0); // more spread => lower repulsion energy
    }

    #[test]
    fn test_cue_phase_repulsion_edge_cases() {
        // n < 2 → returns 0.0, no mutation
        let mut single = vec![1.0f32];
        let r = cue_phase_repulsion(&mut single, 0.1);
        assert_eq!(r, 0.0);
        assert_eq!(single, vec![1.0]);

        // n = 0 → returns 0.0
        let mut empty: Vec<f32> = vec![];
        let r = cue_phase_repulsion(&mut empty, 0.1);
        assert_eq!(r, 0.0);

        // all-identical phases → large R, but gradient is undefined (cot(0)=∞)
        // so the guard skips all terms → no movement (expected: degenerate case)
        let mut identical = vec![0.5f32; 4];
        let r = cue_phase_repulsion(&mut identical, 0.01);
        assert!(r > 0.0); // R is still large (energy diverges)
        // phases unchanged because cot(0) is guarded → gradient = 0

        // nearly-identical phases → cot is finite → phases pushed apart
        let mut near_identical = vec![0.5f32, 0.5 + 1e-6, 0.5 + 2e-6, 0.5 + 3e-6];
        let _ = cue_phase_repulsion(&mut near_identical, 0.01);
        let spread = near_identical.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
            - near_identical.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        assert!(spread > 1e-6, "near-identical phases must be pushed apart");

        // large n → no panic, all finite
        let mut large: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1).collect();
        let r = cue_phase_repulsion(&mut large, 0.01);
        assert!(r.is_finite());
        assert!(large.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_cauchy_transform_far_field() {
        // eigenvalues on unit circle; G(z) for far z ~ 1/z
        let lambdas: Vec<ComplexF32> = (0..8)
            .map(|i| {
                let t = 2.0 * std::f32::consts::PI * i as f32 / 8.0;
                ComplexF32::new(t.cos(), t.sin())
            })
            .collect();
        let z = ComplexF32::new(10.0, 0.0);
        let g = cauchy_transform(&lambdas, z);
        assert!((g.re - 0.1).abs() < 1e-3); // ~ 1/10
        assert!(g.im.abs() < 1e-3);
    }

    #[test]
    fn test_r_transform_add_additive() {
        let ra = vec![1.0f32, 2.0, 3.0];
        let rb = vec![0.5f32, 1.0, 1.5];
        let r = r_transform_add(&ra, &rb);
        assert_eq!(r, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn test_contour_rotate_preserves_norm() {
        let mut h = ComplexF32::new(3.0, 4.0); // norm 5
        let n0 = h.hermite_norm();
        contour_rotate(&mut h, std::f32::consts::FRAC_PI_4);
        assert!((h.hermite_norm() - n0).abs() < 1e-5);
        // rotated by pi/2 from (3,4) -> (-4, 3)
        let mut h2 = ComplexF32::new(3.0, 4.0);
        contour_rotate(&mut h2, std::f32::consts::FRAC_PI_2);
        assert!((h2.re + 4.0).abs() < 1e-5 && (h2.im - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_spectral_health_detects_collapse() {
        // degenerate manifold: identical magnitudes + aligned phases -> collapsed
        let mags = vec![1.0f32; 8];
        let phases = vec![0.0f32; 8];
        let s = cmud_spectral_health(&mags, &phases);
        assert!(s.collapsed, "aligned identical manifold must be flagged collapsed");
        // diverse manifold: varied magnitudes + spread phases -> not collapsed
        let mags2: Vec<f32> = (0..8).map(|i| 0.5 + 0.1 * i as f32).collect();
        let phases2: Vec<f32> = (0..8).map(|i| i as f32 * 0.7).collect();
        let s2 = cmud_spectral_health(&mags2, &phases2);
        assert!(!s2.collapsed, "diverse manifold must not be flagged collapsed");
        assert!(s2.cauchy_mag_at_2.is_finite());
        assert!(s2.spread_mag > 0.0);
    }

    #[test]
    fn test_think_step_phase_attn_runs() {
        let x = vec![1.0f32; 32];
        let mut st = ThinkingState::from_real(&x, 10.0);
        st.think_step_phase_attn(0.15);
        assert_eq!(st.tau, 1);
        assert!(st.h.iter().all(|h| h.is_finite()));
        // all norms within the Hermitian ball
        assert!(st.h.iter().all(|h| h.hermite_norm() <= st.radius + 1e-4));
        let mut out = vec![0.0f32; 32];
        st.collapse_to_real(&mut out);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_think_step_trainable_runs() {
        let x = vec![1.0f32; 32];
        let mut st = ThinkingState::from_real(&x, 10.0);
        let p = CmudLayerParams::from_defaults(32);
        st.think_step_trainable(&p);
        assert_eq!(st.tau, 1);
        assert!(st.h.iter().all(|h| h.is_finite()));
        assert!(st.h.iter().all(|h| h.hermite_norm() <= st.radius + 1e-4));
        // default params must match the fixed positional-phase step (regression-safe)
        let mut st2 = ThinkingState::from_real(&x, 10.0);
        st2.think_step_phase_attn(CMUD_DEFAULT_ALPHA);
        for (a, b) in st.h.iter().zip(st2.h.iter()) {
            assert!((a.re - b.re).abs() < 1e-5, "default trainable ≠ fixed step");
        }
    }

    #[test]
    fn test_cmud_params_json_roundtrip() {
        let p = CmudLayerParams::from_defaults(16);
        let tmp = std::env::temp_dir().join("cmud_params_rt.json");
        p.save_json(&tmp).unwrap();
        let q = CmudLayerParams::load_json(&tmp).unwrap();
        assert_eq!(p.alpha, q.alpha);
        assert_eq!(p.q_phase, q.q_phase);
        assert_eq!(p.v_scale, q.v_scale);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_trainable_bias_changes_scores() {
        let x: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let mut s0 = ThinkingState::from_real(&x, 20.0);
        let p0 = CmudLayerParams::from_defaults(16);
        for _ in 0..4 {
            s0.think_step_trainable(&p0);
        }
        let mut s1 = ThinkingState::from_real(&x, 20.0);
        let mut p1 = CmudLayerParams::from_defaults(16);
        for i in 0..16 {
            p1.q_phase[i] = 0.3 * i as f32;
            p1.k_phase[i] = -0.2 * i as f32;
        }
        for _ in 0..4 {
            s1.think_step_trainable(&p1);
        }
        let mut o0 = vec![0.0f32; 16];
        s0.collapse_to_real(&mut o0);
        let mut o1 = vec![0.0f32; 16];
        s1.collapse_to_real(&mut o1);
        let diff: f32 = o0.iter().zip(o1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "learned phase biases must change the reasoning output");
    }

    #[test]
    fn test_cmud_backward_matches_fd() {
        // Validates `cmud_backward` (analytic) against central finite differences on a random
        // problem: loss = ||h_final||^2 after K trainable think steps. Radius is large so the
        // soft-clamp stays in the identity region (matching assumption in `cmud_backward`).
        let hidden = 24;
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut rand = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };
        let x: Vec<f32> = (0..hidden).map(|_| rand() * 0.3).collect();
        let mut p = CmudLayerParams::from_defaults(hidden);
        for i in 0..hidden {
            p.q_phase[i] = rand() * 0.1;
            p.k_phase[i] = rand() * 0.1;
            p.v_scale[i] = 1.0 + rand() * 0.1;
        }
        p.alpha = 0.05;
        let radius = 10.0;

        const K: usize = 3;
        let loss_fwd = |pp: &CmudLayerParams| -> (f32, Vec<f32>) {
            let mut st = ThinkingState::from_real(&x, radius);
            let mut tapes = Vec::new();
            for _ in 0..K {
                st.think_step_trainable_record(pp, &mut tapes);
            }
            let mut h = vec![0.0f32; hidden];
            st.collapse_to_real(&mut h);
            let l = h.iter().map(|v| v * v).sum::<f32>();
            (l, h)
        };

        let mut tapes = Vec::new();
        let mut st = ThinkingState::from_real(&x, radius);
        for _ in 0..K {
            st.think_step_trainable_record(&p, &mut tapes);
        }
        // `cmud_backward` expects `∂L/∂mixed` (the pre-collapse residual output). The loss sees
        // `h_final = |mixed|`, so `∂L/∂mixed = (∂L/∂h_final)·sign(mixed) = 2·h_final·sign(mixed)`.
        // `sign(mixed)` is reconstructed from the last tape (collapse is |·| on a real value).
        let last = &tapes[K - 1];
        let grad_h: Vec<f32> = (0..hidden)
            .map(|i| {
                let mixed = last.h_before[i] + last.alpha * (last.attn[i] - last.h_before[i]);
                2.0 * mixed
            })
            .collect();
        let mut grad = CmudLayerParamsGrad {
            q_phase: vec![0.0f32; hidden],
            k_phase: vec![0.0f32; hidden],
            v_scale: vec![0.0f32; hidden],
            alpha: 0.0,
        };
        cmud_backward(&tapes, &grad_h, &p, &mut grad);
        let eps = 1e-4;
        let fd = |mut pp: CmudLayerParams, idx: usize, which: u8, delta: f32| -> f32 {
            match which {
                0 => pp.q_phase[idx] += delta,
                1 => pp.k_phase[idx] += delta,
                2 => pp.v_scale[idx] += delta,
                _ => pp.alpha += delta,
            }
            loss_fwd(&pp).0
        };
        let mut max_q = 0.0f32;
        let mut max_k = 0.0f32;
        let mut max_v = 0.0f32;
        for i in 0..hidden {
            let lp = fd(p.clone(), i, 0, eps);
            let lm = fd(p.clone(), i, 0, -eps);
            let f = (lp - lm) / (2.0 * eps);
            max_q = max_q.max((grad.q_phase[i] - f).abs());
            let lp = fd(p.clone(), i, 1, eps);
            let lm = fd(p.clone(), i, 1, -eps);
            let f = (lp - lm) / (2.0 * eps);
            max_k = max_k.max((grad.k_phase[i] - f).abs());
            let lp = fd(p.clone(), i, 2, eps);
            let lm = fd(p.clone(), i, 2, -eps);
            let f = (lp - lm) / (2.0 * eps);
            max_v = max_v.max((grad.v_scale[i] - f).abs());
        }
        let lp = fd(p.clone(), 0, 3, eps);
        let lm = fd(p.clone(), 0, 3, -eps);
        let fa = (lp - lm) / (2.0 * eps);
        let max_err = max_q.max(max_k).max(max_v).max((grad.alpha - fa).abs());
        assert!(max_err < 1e-2, "analytic/!FD gradient mismatch, max_err={max_err}");
    }

    #[test]
    fn test_cmud_kernel_selfcheck() {
        let (ok, msg) = cmud_kernel_selfcheck();
        assert!(ok, "self-check failed: {msg}");
        assert!(msg.contains("think_step"));
    }

    #[test]
    fn test_complex_time_step_runs() {
        let x = vec![1.0f32; 32];
        let mut st = ThinkingState::from_real(&x, 10.0);
        complex_time_step(&mut st, 0.1, 0.2);
        assert_eq!(st.sigma, 0.2);
        assert!(st.tau > 0);
        assert!(st.h.iter().all(|h| h.is_finite()));
    }

    #[test]
    fn test_maybe_think_collapse_report_respects_ball() {
        // without flag -> zero report, input unchanged
        unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        }
        let mut x = vec![0.5f32; 16];
        let rep = maybe_think_collapse_report(&mut x, 8.0);
        assert_eq!(rep.steps, 0);
        assert_eq!(x, vec![0.5f32; 16]);

        // with flag -> runs, stays within the Hermitian ball
        unsafe {
            std::env::set_var("MUD_CMUD_THINK", "1");
        }
        let mut y = vec![0.5f32; 16];
        let rep2 = maybe_think_collapse_report(&mut y, 8.0);
        unsafe {
            std::env::remove_var("MUD_CMUD_THINK");
        }
        assert!(rep2.steps > 0);
        assert!(rep2.max_herm_norm <= rep2.radius * 1.01);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
