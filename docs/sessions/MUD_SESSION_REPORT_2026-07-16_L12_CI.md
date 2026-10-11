# Session Report — L-12 P-13 properties + CI health battery (2026-07-16)

## Delivered

### `src/mud/p13.rs`
- `ArchDims`, `parse_arch_dims` / `parse_arch_dims_map`
- Alternate keys: `num_hidden_layers`, `n_heads`, `ffn_mid`, …
- `validate_arch_consistency`: GQA, ELUT ×8, dims > 0
- Fail-fast: no silent defaults for hidden / layers / heads / FFN
- `health_constants_ok` for EPSILON / PCore SSOT
- Property tests: valid GQA grid, invalid geometries, env clamp for `MUD_PCORE_THREADS`

### Integration
- `corpus_trainer::validate_metadata` → `p13::parse_arch_dims`
- `training_healthcheck` → `parse_arch_dims` + constants check + LIVE stack banner updated

### CI battery
| Surface | Command |
|---------|---------|
| GitHub Actions | `.github/workflows/ci.yml` |
| Local | `./mud.sh ci` |

Battery steps: `cargo test --lib` → `cargo test --lib p13` → `clippy -D warnings` → optional model healthcheck.

## Validation
- **137** lib tests passed
- `./mud.sh ci` complete on local model
- clippy clean

## Ledger
| ID | Status |
|----|--------|
| L-12 | **DONE** |
| Next | Phase B+ or deferred L-13 |
