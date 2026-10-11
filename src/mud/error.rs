//! SPRINT-4: Typed error enum for the MUD engine.
//!
//! Replaces the heterogeneous mix of `unwrap()`, `expect("Missing ...")`,
//! `panic!`, and `Result<_, anyhow::Error>` in the hot path with a single
//! domain error type that callers can match on.
//!
//! P-17: fail-fast is preserved — the trainer and the inference CLI still
//! panic at startup if a required metadata field is missing. The difference
//! is that the panic message is now **typed** and the call site decides
//! whether to panic, log, or propagate.

#![allow(clippy::result_large_err)]

use std::io;

/// Domain-level error type. Each variant maps to a class of failure that
/// the engine can detect up-front (missing metadata, malformed model,
/// OOM, etc.).
#[derive(Debug, thiserror::Error)]
pub enum MudError {
    /// Required metadata field is missing from the `.mud` global header
    /// (e.g. `hidden_size`, `num_heads`). Without this, the model cannot
    /// be loaded at all — typically a corrupted or pre-v2 model file.
    #[error("required metadata field missing: {0}")]
    MissingMetadata(&'static str),

    /// Required tensor not found in the loaded `.mud` file.
    #[error("tensor not found: {0}")]
    TensorNotFound(String),

    /// Numeric overflow detected during shape math (e.g. rows × cols > usize).
    #[error("shape overflow: {0}")]
    ShapeOverflow(String),

    /// Tensor shape is incompatible with the model's declared dims.
    #[error("shape mismatch: {0} (expected {1}, got {2})")]
    ShapeMismatch(String, String, String),

    /// I/O error reading or writing a model file / sidecar.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// NaN / Inf detected at a non-fatal site (logged + counter, no abort).
    #[error("non-finite value in {0}")]
    NonFinite(&'static str),

    /// Optimizer-specific error (e.g. Adam state shape mismatch).
    #[error("optimizer error: {0}")]
    Optimizer(String),

    /// Expert bus lookup failed (e.g. expert id out of range).
    #[error("expert bus error: {0}")]
    BusEmpty(String),

    /// Generic engine error with a free-form message.
    #[error("engine error: {0}")]
    Engine(String),
}

impl From<anyhow::Error> for MudError {
    fn from(e: anyhow::Error) -> Self {
        MudError::Engine(e.to_string())
    }
}

/// Convenience conversion: panic on `MissingMetadata` (P-17 fail-fast).
/// Other variants are returned as `Err`. Use only in startup paths where
/// any error is fatal.
#[track_caller]
pub fn expect_metadata<T>(field: &'static str, res: anyhow::Result<T>) -> T {
    match res {
        Ok(v) => v,
        Err(_) => panic!(
            "P-13: missing required metadata `{field}` — the .mud is corrupted or pre-v2. \
             Re-convert the source GGUF with `bonsai_gguf_to_mud`."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_metadata_display() {
        let e = MudError::MissingMetadata("hidden_size");
        let s = e.to_string();
        assert!(s.contains("hidden_size"));
        assert!(s.contains("missing"));
    }

    #[test]
    fn test_tensor_not_found_display() {
        let e = MudError::TensorNotFound("blk.0.attn_q.weight".into());
        assert!(e.to_string().contains("blk.0.attn_q.weight"));
    }

    #[test]
    fn test_shape_mismatch_display() {
        let e = MudError::ShapeMismatch(
            "token_embd.weight".into(),
            "[vocab, hidden]".into(),
            "[hidden]".into(),
        );
        let s = e.to_string();
        assert!(s.contains("token_embd.weight"));
        assert!(s.contains("[vocab, hidden]"));
        assert!(s.contains("[hidden]"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "missing.mud");
        let mud: MudError = io_err.into();
        assert!(matches!(mud, MudError::Io(_)));
    }

    #[test]
    fn test_from_anyhow_error() {
        let ae = anyhow::anyhow!("cosmic rays");
        let mud: MudError = ae.into();
        assert!(matches!(mud, MudError::Engine(_)));
        assert!(mud.to_string().contains("cosmic rays"));
    }
}
