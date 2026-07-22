//! # L-12 / P-13: Anti-hardcoding metadata policy
//!
//! Network dimensions **must** come from `.mud` global metadata (or hard error).
//! No silent magic defaults for hidden / layers / heads / FFN mid.
//!
//! Used by: corpus trainer, `training_healthcheck`, property tests, CI battery.

use crate::mud::MudFile;
use std::collections::HashMap;

/// Canonical architecture dims resolved from metadata (P-13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchDims {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub intermediate_size: usize,
}

impl ArchDims {
    /// Head dimension `hidden / num_heads`.
    pub fn head_dim(self) -> usize {
        self.hidden_size / self.num_heads.max(1)
    }

    /// KV dim for GQA: `num_kv_heads * head_dim`.
    pub fn kv_dim(self) -> usize {
        self.num_kv_heads * self.head_dim()
    }
}

/// Read first present key as `usize`.
pub fn meta_usize(meta: &HashMap<String, String>, keys: &[&str]) -> Option<usize> {
    for k in keys {
        if let Some(v) = meta.get(*k).and_then(|s| s.parse::<usize>().ok()) {
            return Some(v);
        }
    }
    None
}

/// Read first present key as `f32`.
pub fn meta_f32(meta: &HashMap<String, String>, keys: &[&str]) -> Option<f32> {
    for k in keys {
        if let Some(v) = meta.get(*k).and_then(|s| s.parse::<f32>().ok()) {
            return Some(v);
        }
    }
    None
}

/// Require usize metadata or fail with P-13 message (no silent default).
pub fn require_meta_usize(
    meta: &HashMap<String, String>,
    keys: &[&str],
    what: &str,
) -> anyhow::Result<usize> {
    meta_usize(meta, keys).ok_or_else(|| {
        anyhow::anyhow!(
            "P-13: missing required metadata for {what} (tried keys: {})",
            keys.join(", ")
        )
    })
}

/// Parse architecture from MudFile global metadata. Fail-fast on missing required dims.
pub fn parse_arch_dims(mud: &MudFile) -> anyhow::Result<ArchDims> {
    parse_arch_dims_map(&mud.global_metadata)
}

/// Parse architecture from a metadata map (unit-test friendly).
pub fn parse_arch_dims_map(meta: &HashMap<String, String>) -> anyhow::Result<ArchDims> {
    let hidden_size = require_meta_usize(meta, &["hidden_size"], "hidden_size")?;
    let num_layers = require_meta_usize(
        meta,
        &["num_layers", "num_hidden_layers", "n_layers"],
        "num_layers",
    )?;
    let num_heads = require_meta_usize(
        meta,
        &["num_attention_heads", "num_heads", "n_heads"],
        "num_heads",
    )?;
    // KV heads may default to num_heads only when Q heads known (GQA optional) —
    // this is the single documented fallback (not a magic hidden size).
    let num_kv_heads = meta_usize(meta, &["num_key_value_heads", "num_kv_heads", "n_kv_heads"])
        .unwrap_or(num_heads);
    let intermediate_size = require_meta_usize(
        meta,
        &["intermediate_size", "ffn_hidden", "ffn_mid"],
        "intermediate_size",
    )?;

    let dims = ArchDims {
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
    };
    validate_arch_consistency(dims)?;
    Ok(dims)
}

/// Geometric / ELUT consistency checks (property domain for L-12).
pub fn validate_arch_consistency(d: ArchDims) -> anyhow::Result<()> {
    if d.hidden_size == 0 || d.num_layers == 0 || d.num_heads == 0 || d.intermediate_size == 0 {
        anyhow::bail!("P-13: architecture dims must be > 0, got {d:?}");
    }
    if d.num_kv_heads == 0 {
        anyhow::bail!("P-13: num_kv_heads must be > 0");
    }
    if !d.hidden_size.is_multiple_of(d.num_heads) {
        anyhow::bail!(
            "P-13: hidden_size ({}) not divisible by num_heads ({})",
            d.hidden_size,
            d.num_heads
        );
    }
    if !d.num_heads.is_multiple_of(d.num_kv_heads) {
        anyhow::bail!(
            "P-13: num_heads ({}) not divisible by num_kv_heads ({}) (GQA)",
            d.num_heads,
            d.num_kv_heads
        );
    }
    // ELUT 4-bit: n_in must be multiple of 8 for GEMV packing
    if !d.hidden_size.is_multiple_of(8) {
        anyhow::bail!(
            "P-13: hidden_size ({}) must be multiple of 8 (ELUT)",
            d.hidden_size
        );
    }
    if !d.intermediate_size.is_multiple_of(8) {
        anyhow::bail!(
            "P-13: intermediate_size ({}) must be multiple of 8 (ELUT)",
            d.intermediate_size
        );
    }
    Ok(())
}

/// Essential keys for trainer load (tokenizer + at least one layer count key).
pub fn validate_trainer_required_keys(meta: &HashMap<String, String>) -> anyhow::Result<()> {
    // Dims via parse (accepts alternate key names)
    let _ = parse_arch_dims_map(meta)?;
    if !meta.contains_key("tokenizer.tokens") {
        anyhow::bail!("P-13: missing essential metadata key: 'tokenizer.tokens'");
    }
    Ok(())
}

/// Stream L: ensure canonical key aliases exist so every consumer finds dims.
/// Writes missing alternate names from resolved [`ArchDims`].
pub fn ensure_canonical_metadata_aliases(
    meta: &mut HashMap<String, String>,
) -> anyhow::Result<ArchDims> {
    let dims = parse_arch_dims_map(meta)?;
    // Primary names preferred by healthcheck / auditor
    meta.entry("hidden_size".into())
        .or_insert_with(|| dims.hidden_size.to_string());
    meta.entry("num_layers".into())
        .or_insert_with(|| dims.num_layers.to_string());
    meta.entry("num_hidden_layers".into())
        .or_insert_with(|| dims.num_layers.to_string());
    meta.entry("num_heads".into())
        .or_insert_with(|| dims.num_heads.to_string());
    meta.entry("num_attention_heads".into())
        .or_insert_with(|| dims.num_heads.to_string());
    meta.entry("num_kv_heads".into())
        .or_insert_with(|| dims.num_kv_heads.to_string());
    meta.entry("num_key_value_heads".into())
        .or_insert_with(|| dims.num_kv_heads.to_string());
    meta.entry("intermediate_size".into())
        .or_insert_with(|| dims.intermediate_size.to_string());
    meta.entry("ffn_hidden".into())
        .or_insert_with(|| dims.intermediate_size.to_string());
    // Optional but frequently required by tools
    if !meta.contains_key("vocab_size") {
        if let Some(v) = meta_usize(meta, &["n_vocab"]) {
            meta.insert("vocab_size".into(), v.to_string());
        }
    }
    if !meta.contains_key("max_position_embeddings") {
        if let Some(v) = meta_usize(meta, &["max_seq_len", "n_ctx", "context_length"]) {
            meta.insert("max_position_embeddings".into(), v.to_string());
        }
    }
    if !meta.contains_key("rms_norm_eps") {
        meta.insert("rms_norm_eps".into(), "1e-5".into());
    }
    Ok(dims)
}

/// Full converter emit check: arch + trainer keys + vocab/max_pos when possible.
pub fn validate_converter_emit(meta: &HashMap<String, String>) -> anyhow::Result<()> {
    validate_trainer_required_keys(meta)?;
    if meta_usize(meta, &["vocab_size", "n_vocab"]).is_none() {
        anyhow::bail!("P-13: converter emit missing vocab_size");
    }
    if meta_usize(
        meta,
        &[
            "max_position_embeddings",
            "max_seq_len",
            "n_ctx",
            "context_length",
        ],
    )
    .is_none()
    {
        anyhow::bail!("P-13: converter emit missing max_position_embeddings");
    }
    Ok(())
}

/// CI / health battery: constants SSOT invariants.
pub fn health_constants_ok() -> bool {
    use crate::mud::constants::{
        default_pcore_threads, DEPTH_DAMPENING_FACTOR, EPSILON_FLOOR, PCORE_THREADS_CAP,
        SPARSITY_THRESHOLD_RATIO,
    };
    EPSILON_FLOOR > 0.0
        && EPSILON_FLOOR <= 1e-6
        && (DEPTH_DAMPENING_FACTOR - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6
        && (SPARSITY_THRESHOLD_RATIO - 0.7).abs() < 1e-6
        && (1..=PCORE_THREADS_CAP.max(64)).contains(&default_pcore_threads())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::constants::{default_pcore_threads, EPSILON_FLOOR};

    fn meta(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn valid_base() -> HashMap<String, String> {
        meta(&[
            ("hidden_size", "256"),
            ("num_layers", "4"),
            ("num_attention_heads", "8"),
            ("num_key_value_heads", "2"),
            ("intermediate_size", "512"),
            ("tokenizer.tokens", "a\nb"),
        ])
    }

    #[test]
    fn test_parse_happy_path() {
        let d = parse_arch_dims_map(&valid_base()).unwrap();
        assert_eq!(d.hidden_size, 256);
        assert_eq!(d.num_heads, 8);
        assert_eq!(d.num_kv_heads, 2);
        assert_eq!(d.head_dim(), 32);
        assert_eq!(d.kv_dim(), 64);
    }

    #[test]
    fn test_alternate_layer_keys() {
        let mut m = valid_base();
        m.remove("num_layers");
        m.insert("num_hidden_layers".into(), "6".into());
        let d = parse_arch_dims_map(&m).unwrap();
        assert_eq!(d.num_layers, 6);
    }

    #[test]
    fn test_kv_defaults_to_heads() {
        let mut m = valid_base();
        m.remove("num_key_value_heads");
        let d = parse_arch_dims_map(&m).unwrap();
        assert_eq!(d.num_kv_heads, d.num_heads);
    }

    #[test]
    fn test_missing_hidden_fails() {
        let mut m = valid_base();
        m.remove("hidden_size");
        let err = parse_arch_dims_map(&m).unwrap_err().to_string();
        assert!(err.contains("P-13") && err.contains("hidden_size"));
    }

    #[test]
    fn test_missing_tokenizer_fails_trainer_keys() {
        let mut m = valid_base();
        m.remove("tokenizer.tokens");
        assert!(validate_trainer_required_keys(&m).is_err());
    }

    /// Property: random valid GQA geometries pass consistency.
    #[test]
    fn property_valid_gqa_geometries() {
        // heads in {4,8,16}, kv_ratio in {1,2,4}, head_dim in {16,32,64}
        for heads in [4usize, 8, 16] {
            for ratio in [1usize, 2, 4] {
                if heads % ratio != 0 {
                    continue;
                }
                let kv = heads / ratio;
                for hd in [16usize, 32, 64] {
                    let hidden = heads * hd;
                    let inter = hidden * 2;
                    let d = ArchDims {
                        hidden_size: hidden,
                        num_layers: 2,
                        num_heads: heads,
                        num_kv_heads: kv,
                        intermediate_size: inter,
                    };
                    assert!(
                        validate_arch_consistency(d).is_ok(),
                        "expected ok for {d:?}"
                    );
                }
            }
        }
    }

    /// Property: inconsistent geometries always fail.
    #[test]
    fn property_invalid_geometries_fail() {
        // hidden not divisible by heads
        assert!(validate_arch_consistency(ArchDims {
            hidden_size: 100,
            num_layers: 1,
            num_heads: 8,
            num_kv_heads: 8,
            intermediate_size: 256,
        })
        .is_err());
        // GQA mismatch
        assert!(validate_arch_consistency(ArchDims {
            hidden_size: 256,
            num_layers: 1,
            num_heads: 8,
            num_kv_heads: 3,
            intermediate_size: 512,
        })
        .is_err());
        // ELUT: hidden not multiple of 8
        assert!(validate_arch_consistency(ArchDims {
            hidden_size: 12,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_size: 16,
        })
        .is_err());
        // zero
        assert!(validate_arch_consistency(ArchDims {
            hidden_size: 0,
            num_layers: 1,
            num_heads: 1,
            num_kv_heads: 1,
            intermediate_size: 8,
        })
        .is_err());
    }

    #[test]
    fn test_ensure_canonical_aliases() {
        let mut m = meta(&[
            ("hidden_size", "576"),
            ("num_hidden_layers", "30"),
            ("num_attention_heads", "9"),
            ("num_key_value_heads", "3"),
            ("intermediate_size", "1536"),
            ("vocab_size", "49152"),
            ("max_position_embeddings", "8192"),
            ("tokenizer.tokens", "x"),
        ]);
        let d = ensure_canonical_metadata_aliases(&mut m).unwrap();
        assert_eq!(d.num_layers, 30);
        assert!(m.contains_key("num_layers"));
        assert!(m.contains_key("num_heads"));
        assert!(m.contains_key("ffn_hidden"));
        validate_converter_emit(&m).unwrap();
    }

    /// Property: no silent default for required intermediate_size.
    #[test]
    fn property_no_silent_ffn_default() {
        let mut m = valid_base();
        m.remove("intermediate_size");
        assert!(parse_arch_dims_map(&m).is_err());
    }

    #[test]
    fn test_health_constants_ok() {
        assert!(health_constants_ok());
        const {
            assert!(EPSILON_FLOOR > 0.0);
        }
        let n = default_pcore_threads();
        assert!((1..=64).contains(&n));
    }

    #[test]
    fn property_pcore_env_clamp() {
        // SAFETY: test-only env mutation; serial test threads for this module recommended
        unsafe {
            std::env::set_var("MUD_PCORE_THREADS", "0");
        }
        let n0 = default_pcore_threads();
        assert_eq!(n0, 1, "0 clamps to 1");
        unsafe {
            std::env::set_var("MUD_PCORE_THREADS", "999");
        }
        let n1 = default_pcore_threads();
        assert_eq!(n1, 64, "999 clamps to 64");
        unsafe {
            std::env::remove_var("MUD_PCORE_THREADS");
            std::env::remove_var("RAYON_NUM_THREADS");
        }
        let n2 = default_pcore_threads();
        assert!((1..=8).contains(&n2), "default capped at PCORE_THREADS_CAP");
    }
}
