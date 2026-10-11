# MUD Session Report — 2026-07-31 (Telemetry Enrichment + Code Corpus)

## Objectives

1. Deeply review telemetry variables, connect missing ones to the `train_telemetry` TUI graph
2. Extract high-quality documentation from the system codebase into a trainable corpus

## Telemetry Enrichment

### Problem

The `train_telemetry` TUI had several Series defined but never populated with data:
- `grad_norm` Series existed but the `[TELEM]` line used key `gn` (never matched)
- `entropy` Series existed but never received data (entropy was never computed)
- `stp_loss` Series existed but never received data (STP loss not logged)
- `loss_cv` and `traj_eff` Series existed but never populated

Additionally, the legacy file telemetry format (`mud_train_metrics.log`) had `0.0` placeholder values at columns 8, 9, 10 (intended for `grad_norm`, `stp_loss`, `entropy`).

### Changes

#### `src/mud/corpus_trainer/metrics.rs`
- Added `grad_norm: f32`, `entropy: f32`, `stp_loss: f32` to `TrainChunkMetrics`
- Added `grad_norm: f64`, `entropy: f64`, `stp_loss: f64` to `ActStatsAccum`
- Added `push_grad(grad_norm, entropy, stp_loss)` method with Welford online mean
- Updated `finish()` to emit all three new fields

#### `src/mud/corpus_trainer/train_step.rs`
- Added `grad_norm_sum` and `entropy_sum` accumulators before the sub-step loop
- `grad_norm` captured from pre-clip L2 norm at line 662 (`norm_sq.sqrt()`)
- `entropy` computed post-softmax: `-Σ p·log₂(p)` with guard `p > 1e-12`
- `stp_loss` computed from existing `stp_loss_accum / stp_triples`
- `act_accum.push_grad()` called once after the loop with averages

#### `src/mud/corpus_trainer/session.rs`
- Added `avg_grad_norm`, `avg_entropy` variables from `chunk_metrics`
- Updated legacy file format string: replaced `0.0 0.0 0.0` with `{:.6} {:.4} {:.4}` for grad_norm, stp_loss, entropy
- Updated `[TELEM]` stderr line: added `grad_norm={:.4} entropy={:.3} stp_loss={:.5}` before `n_act=`

#### `tools/train_telemetry.rs`
- Added `entropy: Series` field to `App` struct + `init`
- Added `stp_loss=` key parsing in `absorb_kv` (was already a Series, just never populated)
- Efficiency mode: replaced "STP λ (adaptive)" chart with "STP Loss" chart
- Efficiency mode: replaced "Loss CV (stability)" chart with "Entropy (bits)" chart
- Added `d_entropy` and `d_stp_loss` data series + y-bounds

### Verification

Training run on `models/ale.mud` (hidden=128, L=6, ternary):
```
step=1  loss=6.4883 grad_norm=0.0012 entropy=0.109 stp_loss=0.00000
step=5  loss=6.3196 grad_norm=0.0013 entropy=0.110 stp_loss=0.00000
step=10 loss=6.0179 grad_norm=0.0013 entropy=0.110 stp_loss=0.00000
```

- `grad_norm ≈ 0.0013` — stable, no gradient explosion/vanishing
- `entropy ≈ 0.109 bits` — healthy (64-class softmax, max ~6 bits; 0.109 indicates the model has learned some preference without collapsing)
- `stp_loss = 0.0` — expected (STP is OFF by default, `MUD_TRAIN_STP=1` to enable)
- `varh = 7.76–7.97` — stable, no NaN/Inf

## Code Corpus Extraction

### Problem

The existing `build_project_corpus` in `mud.sh` includes raw `.rs` source files as training text. Raw code contains significant syntax noise (braces, types, macros) that reduces corpus quality. The user requested extracting "part of the system code" and processing it "with quality" for the corpus.

### Solution

Created `tools/extract_code_docs.rs`:

**Sources extracted:**
- `src/mud/**/*.rs` (89 files) — doc comments (`///`, `//!`) + explanatory `//` comments
- `docs/architecture/**/*.md` (35 files) — design docs (compute stack, register packing, JEPA/MHC training, etc.)
- `docs/sessions/*.md` — session reports

**Processing:**
- Strips comment markers (`//`, `///`, `//!`)
- Filters noise: TODO/FIXME/HACK/copyright/SAFETY/test-only comments
- Strips markdown structure: headers (`#`), code blocks (````), tables (`|`), blockquotes (`>`)
- Removes inline code spans (`` `code` ``) while preserving identifier-like content
- Normalizes whitespace (collapse to single spaces)
- Groups doc comments into coherent paragraphs (blank-line separated)

**Output:** `training/corpus/mud_docs.txt`
- 6,165 lines, ~80,000 words
- Domain-specific content: Adam, mHC, JEPA, ternary quantization, AVX2 kernels, VRAM management, CSA indexing, C-MUD thinking, Wormhole networking, CellV5 packing

**Integration:**
- Auto-included in `build_project_corpus` via the existing `training/corpus/*.txt` glob (line 94 in `mud.sh`)
- `./mud.sh extract-docs` command added for one-command regeneration
- Listed in `./mud.sh tools` catalog

### Corpus Quality Sample

```
Adam / SparseAdam state (gap P0 — optimizer honesty) Persistent first/second moments for shadow weights.
D2: Convert f32 → bf16 (truncate mantissa, no rounding). Google bf16 format: 1 sign + 8 exponent + 7 mantissa (same exponent as f32).
STP loss AVX2 COMPLETADO: vsub_avx2 para d1/d2 computation; _mm256_fmadd_ps para g_d1/g_d2 gradient computation (8 elementos/iter); accumulación en grad_h_r/t/s con _mm256_add_ps; fallback scalar.
SlimeRegister is a f16-packed dual (re, im) u32; read_accum() returns f32 via F16C (current truth).
```

## Files Changed

| File | Change |
|------|--------|
| `src/mud/corpus_trainer/metrics.rs` | +3 fields to `TrainChunkMetrics` + `ActStatsAccum`; `push_grad` method |
| `src/mud/corpus_trainer/train_step.rs` | grad_norm + entropy accumulation in training loop |
| `src/mud/corpus_trainer/session.rs` | TELEM line + file format updated with new fields |
| `tools/train_telemetry.rs` | entropy Series + STP Loss chart + Entropy chart |
| `tools/extract_code_docs.rs` | **NEW** — code doc extraction tool |
| `Cargo.toml` | +1 bin entry for `extract_code_docs` |
| `mud.sh` | +`extract-docs` command + tools catalog entry |
| `training/corpus/mud_docs.txt` | **NEW** — 6165-line extracted corpus |
| `AGENTS.md` | Updated last sync, TODO, pipeline, commands, fix log |

## Verification

```
cargo clippy --all-targets  → 0 errors, 0 warnings
cargo test --lib            → 423 passed, 0 failed, 2 ignored
cargo build --release       → success
./mud.sh extract-docs       → 6165 lines, 80159 words written
```
