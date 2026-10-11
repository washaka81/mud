//! Embedding row loader + stddev helper.
//!
//! `load_emb_row_into` is the dual-path loader for `token_embd` rows:
//!  - `shadow_emb` (free FP32 buffer) when `emb_lazy = false`;
//!  - direct mmap-decode of the ELUT ternary + PRQ scale when `emb_lazy = true`
//!    (MUD_TRAIN_FREEZE_EMB / quick mode).
//!
//! `slice_stddev` is a tiny inline stddev used by `train_on_sequence` for the
//! `VarH` activation-telemetry panel.

use crate::mud::{MudFile, MudTensorType};

/// Load one emb row: either from FP32 shadow or on-the-fly ELUT mmap (FREEZE_EMB / lazy).
///
/// # Safety
/// When `emb_lazy` is `true`, the `mud` must hold a valid `core` skill with a
/// `token_embd.weight` tensor; the function dereferences raw pointers
/// (`data_ptr` and the PRQ scale tensor) without further checks.
pub fn load_emb_row_into(
    mud: &MudFile,
    shadow_emb: &[f32],
    emb_lazy: bool,
    hidden_size: usize,
    token_id: usize,
    out: &mut [f32],
) {
    if !emb_lazy {
        let off = token_id * hidden_size;
        if off + hidden_size <= shadow_emb.len() && out.len() >= hidden_size {
            out[..hidden_size].copy_from_slice(&shadow_emb[off..off + hidden_size]);
        }
        return;
    }
    let Some(core) = mud.skills.get("core") else {
        return;
    };
    let Some(emb_t) = core.tensors.get("token_embd.weight") else {
        return;
    };
    let cols = emb_t.shape.get(1).copied().unwrap_or(hidden_size);
    let rows = emb_t.shape.first().copied().unwrap_or(0);
    if token_id >= rows || out.len() < cols {
        return;
    }
    unsafe {
        if emb_t.t_type == MudTensorType::Ternary2Bit {
            let u32s = cols.div_ceil(8);
            crate::mud::dequantize_ternary_row(
                (emb_t.data_ptr as *const u32).add(token_id * u32s),
                &mut out[..cols],
                cols,
            );
            if let Some(sc) = core.tensors.get("token_embd.prq_scale") {
                if sc.t_type == MudTensorType::Float32 {
                    let s = *(sc.data_ptr as *const f32).add(token_id);
                    for v in out.iter_mut().take(cols) {
                        *v *= s;
                    }
                }
            }
        } else {
            std::ptr::copy_nonoverlapping(
                (emb_t.data_ptr as *const f32).add(token_id * cols),
                out.as_mut_ptr(),
                cols,
            );
        }
    }
}

/// Stddev of a f32 slice (population). Returns 0 if empty/flat.
#[inline]
pub fn slice_stddev(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &x in xs {
        let v = x as f64;
        sum += v;
        sum_sq += v * v;
    }
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    var.sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_stddev_empty_returns_zero() {
        assert_eq!(slice_stddev(&[]), 0.0);
    }

    #[test]
    fn test_slice_stddev_constant_is_zero() {
        let x = [3.0f32; 8];
        assert!(slice_stddev(&x) < 1e-5);
    }

    #[test]
    fn test_slice_stddev_known_value() {
        // values [2, 4, 4, 4, 5, 5, 7, 9] → population stddev = 2.0
        let x = [2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = slice_stddev(&x);
        assert!((s - 2.0).abs() < 0.01, "got {s}");
    }

    #[test]
    fn test_load_emb_row_into_oob_shadow_safe() {
        // The non-lazy path early-returns when offset+len doesn't fit. The
        // `mud` argument is not touched in this branch, so we use a
        // `Box::leak` of a default-constructed MudFile for the &MudFile
        // ref. (MudFile has no `Default` impl; we construct one inline.)
        let mud = Box::leak(Box::new(MudFile {
            mmap: None,
            skills: std::collections::HashMap::new(),
            global_metadata: std::collections::HashMap::new(),
            path: None,
        }));
        let shadow = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        // token_id 5 is OOB → no copy, out stays zero.
        load_emb_row_into(mud, &shadow, false, 4, 5, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }
}
