# MUD Audit Report V29: Ternary2Bit Format Mismatch & JEPA Collapse

**Date:** 2026-06-20
**Severity:** CRITICAL
**Components:** `slime_forward.rs`, `slime_backward.rs`, `slime.rs`, `slime_jepa.rs`
**Status:** RESOLVED
**Supersedes:** V27 (ALL_ZEROS_RESOLVED), V28 (JEPA_SPIRAL)

---

## 1. Executive Summary

The QAT STE trainer exhibited catastrophic signal collapse: `LAYER PRODUCED ALL ZEROS FOR POS X`
flooded the logs from position 0 of every segment. The root cause was a **format mismatch** between
the weight storage format (Ternary2Bit: 2 bits/weight, 16 per u32) and the GEMV kernel expectations
(ELUT 4-bit: 4 bits/weight, 2 per byte). This mismatch affected both the forward pass and the
backward pass, producing garbage outputs and NaN gradients that corrupted the embedding table after
the first training segment.

Three additional JEPA stabilizer bugs (dead initialization branch, aggressive initial gating, and
cross-sublayer state contamination) compounded the problem but were secondary to the format mismatch.

---

## 2. Symptom Timeline

### Phase 1 — Pre-fix behavior (original report)
```
LAYER PRODUCED ALL ZEROS FOR POS 0  (×30 layers)
APPLY_OUTPUT_NORM PRODUCED ALL ZEROS!
Loss: 16.49 | ETA: 4641.7m
```
All 30 layers produce zero output from the very first token.

### Phase 2 — After JEPA fixes only
```
Segment 1: Sigma=8-327, VarH>0, E_JEPA=0.74-0.94  (HEALTHY)
Segment 2: Sigma=0, VarH=0, E_JEPA=0.85-0.89       (COLLAPSE)
Segment 3: Sigma=0, VarH=NaN, E_JEPA=1.00           (DEATH)
```
The first segment works, but subsequent segments collapse to zero and then NaN.

### Phase 3 — After all 5 fixes
Signal propagates correctly through all segments. No "ALL ZEROS" warnings.

---

## 3. Root Cause Analysis

### Bug #1 (CRITICAL): Forward GEMV uses wrong weight format

**File:** `src/mud/slime_forward.rs:37-58`
**Function:** `ternary_gemv_rowwise()`

The forward pass called `elut_gemv_avx2` (raw ASM) which expects **ELUT 4-bit nibble** format:

```rust
// WRONG — ELUT 4-bit
let row_bytes = n_in / 2;  // 2560/2 = 1280 bytes/row
elut_gemv_avx2(acts_i8.as_ptr(), w_u8.add(row * row_bytes), &mut accum, n_in);
out_f32[row] = accum as f32 * (*scales.add(row)) * act_scale;
```

The model stores weights in **Ternary2Bit** format (confirmed via `mud_dump`):

```
blk.0.attn_q.weight | shape=[2560, 2560] | type=Ternary2Bit
```

| Property | ELUT 4-bit (expected by kernel) | Ternary2Bit (actual in model) |
|----------|-------------------------------|-------------------------------|
| Bits per weight | 4 | 2 |
| Weights per byte | 2 | 4 |
| Row stride (2560 cols) | 1280 bytes | 640 bytes |
| Encoding | 0x0=0, 0x1=+1, 0xF=-1 | 00=0, 01=+1, 10=-1 |
| Row pointer math | `w_u8 + row * 1280` | `w_u32 + row * 160` |

**Consequences:**
1. Row stride 2x too large → from row 1 onward, reads data from adjacent rows
2. 2-bit fields interpreted as 4-bit nibbles → completely wrong weight values
3. Every GEMV output is garbage → all downstream computations meaningless

**Fix:** Replace `elut_gemv_avx2` with ISA-dispatched `ternary_gemv_i8act`:

```rust
// CORRECT — Ternary2Bit
let row_u32s = n_in / 16;
let w_u32 = w_u8 as *const u32;
for row in 0..n_out {
    ternary_gemv_i8act(n_in, acts_i8.as_ptr(), w_u32.add(row * row_u32s),
        &mut out_f32[row], *scales.add(row), act_scale);
}
```

---

### Bug #2 (CRITICAL): Backward GEMV uses wrong weight format

**File:** `src/mud/slime_backward.rs:107-131`
**Function:** `ternary_gemv_backward()` — grad_x computation

The backward pass decoded weights as ELUT 4-bit for the `grad_x = grad_y * W_q` computation:

```rust
// WRONG — ELUT 4-bit
let u8_count = n_in / 2;
for b in 0..u8_count {
    let val = *row_ptr.add(b);
    for j in 0..2 {
        let bits = (val >> (j * 4)) & 0xF;
        let w_val: f32 = match bits {
            0x1 => 1.0,
            0xF => -1.0,
            _ => 0.0,
        };
        grad_x[b * 2 + j] += gy_scaled * w_val;
    }
}
```

**Consequences:**
1. `grad_x` computed with wrong weight values → gradient flows in wrong direction
2. `shadow_emb` updated with corrupt gradients → embeddings become NaN after first segment
3. NaN propagates to all subsequent segments → permanent death spiral

**Fix:** Ternary2Bit decoding (16 weights per u32, 2 bits each):

```rust
// CORRECT — Ternary2Bit
let u32_count = n_in / 16;
let row_ptr = (w_u8 as *const u32).add(row * u32_count);
for b in 0..u32_count {
    let val = *row_ptr.add(b);
    for j in 0..16 {
        let bits = (val >> (j * 2)) & 3;
        let w_val: f32 = match bits { 1 => 1.0, 2 => -1.0, _ => 0.0 };
        grad_x[b * 16 + j] += gy_scaled * w_val;
    }
}
```

---

### Bug #3: JEPA `var_ema` initialization makes init branch dead code

**File:** `src/mud/slime.rs:77`, `src/mud/corpus_trainer.rs:938,1140`

`jepa_stabilizer()` has a first-token initialization branch:
```rust
if *var_ema == 0.0 {
    *mu_ctx = batch_mu;
    *var_ema = batch_var.max(1.0);
}
```

But `jepa_var_ema` was initialized to `vec![1.0f32; 128]` and reset to `1.0` between segments.
Since `1.0 != 0.0`, the init branch **never fires**.

**Fix:** Initialize `jepa_var_ema` to `vec![0.0f32; 256]` and reset to `0.0`.

---

### Bug #4: JEPA `inv_sigma = 1.0` causes gate collapse on first token

**File:** `src/mud/slime.rs:76`

With `mu_ctx = 0.0` and `inv_sigma = 1.0`, the first token's gate:
```
delta = |z - 0| = 14.8   (embedding magnitude)
gate = (1 - 14.8 * 1.0).clamp(0.01, 1.0) = 0.01
```

Signal killed to 1% on first JEPA call. After Layer 0: 32743 → 327. After Layer 1: 3 → 0.

**Fix:** Initialize `jepa_inv_sigma` to `vec![0.0f32; 256]`. With `inv_sigma = 0.0`:
```
gate = (1 - delta * 0.0).clamp(0.01, 1.0) = 1.0   (no killing)
```

---

### Bug #5: JEPA state shared between attention and FFN sublayers

**File:** `src/mud/slime_forward.rs:264,364`

`evaluate_slime_block` calls `jepa_stabilizer` twice per layer:
- Step 7 (attention residual): `ws.jepa_mu[layer_idx]`
- Step 13 (FFN residual): `ws.jepa_mu[layer_idx]` — SAME index

Post-attention statistics contaminate FFN stabilization and vice versa.

**Fix:** Separate indices — attention uses `2*layer_idx`, FFN uses `2*layer_idx+1`.
Workspace size increased from 128 to 256 slots.

---

## 4. Telemetry Column Reference

| Column | Field | Source | Healthy Range |
|--------|-------|--------|---------------|
| Sigma | `|registers[0].matmul_accum|` | `SlimeRegister` | > 0 |
| E_JEPA | `(1/n) * Sigma(z^2 - gamma)^2` | `TensorDiagnostics.jepa_energy` | < 1.0 |
| Rho(p) | Pearson correlation hard↔JEPA | `TensorDiagnostics.rho_cross_corr` | close to 1.0 |
| Cov | Covariance hard↔JEPA | `TensorDiagnostics.cov_hard_jepa` | > 0 |
| VarH | Variance of `matmul_accum * iscale` | `TensorDiagnostics.var_hard` | > 0 |
| VarJ | Variance of `z` (f16 JEPA tracking) | `TensorDiagnostics.var_jepa` | > 0 |
| Sat% | Fraction at +/-32767/32768 | `TensorDiagnostics.saturation_ratio` | < 5% |
| Mode | Most frequent `matmul_accum` value | `TensorDiagnostics.mode_hard` | != 0 |
| Delta(u) | `jepa_mu[0]` — EMA mean | `SlimeWorkspace.jepa_mu` | finite |
| Eps(inv) | `jepa_inv_sigma[0]` — 1/sqrt(var) | `SlimeWorkspace.jepa_inv_sigma` | [0, 1] |
| Omega(v) | `jepa_var_ema[0]` — EMA variance | `SlimeWorkspace.jepa_var_ema` | > 0 |

---

## 5. Files Modified

| File | Change |
|------|--------|
| `src/mud/slime_forward.rs` | Replace `elut_gemv_avx2` with `ternary_gemv_i8act`; separate JEPA indices |
| `src/mud/slime_backward.rs` | Fix `grad_x` to decode Ternary2Bit; update test |
| `src/mud/slime.rs` | Init `jepa_inv_sigma=0`, `jepa_var_ema=0`, size 256 |
| `src/mud/corpus_trainer.rs` | Reset `jepa_inv_sigma.fill(0.0)`, `jepa_var_ema.fill(0.0)` |

---

## 6. Verification

- `cargo check --release` — compiles clean
- `cargo clippy --all-targets` — 0 warnings (P-06)
- `cargo test` — 85/85 pass

---

## 7. Lessons Learned

1. **Format contracts must be enforced at the type level.** The `SlimeLayer` stores weight pointers
   as `*const u8`, which erases the format information. A `*const Ternary2Bit` newtype would have
   prevented the ELUT/Ternary2Bit confusion at compile time.

2. **The backward pass must mirror the forward pass exactly.** Both must decode weights identically.
   Any mismatch produces gradient corruption that manifests as delayed failures (NaN after first
   segment), making the root cause hard to trace.

3. **JEPA state initialization must match the stabilizer's expectations.** The `var_ema == 0.0`
   guard was correct logic defeated by wrong initialization values. Initialization contracts
   should be documented next to the variable definition, not only in the consumer function.

4. **Shared mutable state between sublayers is a latent bug.** The attention and FFN sublayers
   have fundamentally different activation distributions. Sharing JEPA EMA state between them
   causes cross-contamination that manifests as subtle signal degradation over many layers.
