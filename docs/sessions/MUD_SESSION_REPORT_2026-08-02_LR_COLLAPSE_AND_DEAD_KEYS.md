# MUD Session Report — 2026-08-02

## Title: LR Schedule Collapse → Dead Attention Keys → ALLZERO Warnings in `ale.mud`

**Date:** 2026-08-02  
**Session:** Telemetry forensic + trainer LR fixes  
**Status:** ✅ Fixes applied (4/4) | 🚨 Model requires retraining

---

## 1. Trigger: Chat with `ale.mud` Produces ALLZERO Warnings

```
$ ./mud.sh chat models/ale.mud
Loading model: models/ale.mud
[Inference] emb_rms=6.2012e-1 scale_up=1.000 logit_scale=0.0884 tied=false ternary=true

LAYER L4 POS 0 ALLZERO (a0=4.0000 b0=4.0000 R=90.51 regs[0]=0.0008 ...)
LAYER L5 POS 0 ALLZERO (a0=4.0000 b0=4.0000 R=90.51 regs[0]=-0.0002 ...)
```

The ALLZERO recovery noise fires on layers L4–L5 — only 8 of 128 registers non-zero after mHC residual.

---

## 2. Telemetry Snapshot (Training Run Before Fixes)

```
step=13035 epoch=79 block=9/167 loss=5.5553 ppl=258.6 lr=0.000004
loss_vel=-0.012613 varh=0.059027 varj=0.319233 jepa=0.0020
σ=50.0% cog=0.0025 grad_norm=6.8176 entropy=0.125 stp_loss=0.00000
traj_eff=0.9990 conf=0.39%
```

### Analysis of Key Values

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `loss=5.5553` | ≈ ln(256) = 5.545 | **Random-chance loss** (uniform token distribution) |
| `lr=0.000004` | 125× too low | **LR schedule collapsed** — should be ~0.0003 |
| `entropy=0.125` | max=8.0 | **Overconfident but wrong** — predicts with high confidence at random accuracy |
| `conf=0.39%` | = 1/256 | **Random confidence** — no better than uniform |
| `grad_norm=6.8176` | High | Gradients are large but **unused** because LR ≈ 0 |
| `traj_eff=0.9990` | High | Loss trajectory is "stable" — but stable at random-chance plateau |
| `cog=0.0025` | Near zero | Cognitive activation collapsed → model not processing context |
| `varh=0.059` | Low | Hidden variance collapsed → dead activations |

**Conclusion:** The model was stuck at random-chance performance with an effectively-zero learning rate. The telemetry was showing the same stale LR (`0.000004`) for the entire session because `qat_lr` was computed once at session start and never updated.

---

## 3. Root Cause Analysis

### 3.1 Fix #1: `step_count` Not Restored on Resume

**Location:** `src/mud/corpus_trainer/mod.rs:111` + `src/mud/corpus_trainer/session.rs`

**Bug:** `MudCorpusTrainer::new()` initializes `step_count: AtomicU64::new(0)`. When training resumes from a checkpoint at epoch 79, `global_chunks_processed` is correctly computed from `resume_epoch` and `resume_chunk_idx`, but **`step_count` is never updated**. The cosine schedule reads `step_count` to determine the warmup/cosine position.

**Impact:** `step_count` starts at 0 instead of ~13,026 (79 epochs × 167 chunks/epoch). The cosine schedule computes warmup LR:
```
lr = base_lr × (step + 1) / warmup_steps = 0.0005 × 1/135 ≈ 3.7e-6
```
Telemetry showed `lr=0.000004` — matches the calculation exactly.

**Fix:** Added after `global_chunks_processed` is computed in `session.rs`:
```rust
self.step_count.store(global_chunks_processed as u64, Relaxed);
*self.total_steps_hint.borrow_mut() = Some(total_chunks_all_epochs);
```

### 3.2 Fix #2: `qat_lr` Computed Once, Reused for All Telemetry

**Location:** `src/mud/corpus_trainer/session.rs`

**Bug:** `qat_lr` was computed in a pre-loop block (~line 973-979) using `qat_learning_rate()` (the raw constant from `constants.rs:13`). This single value was used for:
- The `[TELM]` telemetry line (displayed LR)
- The recovery jitter magnitude (`jitter_mag = qat_lr.max(1e-4) * 0.1`)
- The entropy display

The telemetry line showed `lr=0.000004` for the entire session, masking that the schedule was broken.

**Fix:** Removed the pre-loop computation. Added per-step recomputation inside the `if should_print || should_telem { ... }` block:
```rust
let qat_lr = apply_lr_schedule(
    base_lr,
    self.step_count.load(Relaxed),
    self.total_steps_hint.borrow().unwrap_or(QAT_LR_TOTAL_STEPS),
    *self.warmup_steps.borrow(),
);
```

**Cascade effect:** This also fixes Fix #5 (recovery jitter) — the jitter at line 1703 now uses the correct per-step `qat_lr` instead of the stale session-start value.

### 3.3 Fix #4: Cosine Schedule Decays to Zero

**Location:** `src/mud/corpus_trainer/train_step.rs:49`

**Bug:** The original cosine formula:
```rust
0.5 * base_lr * (1.0 + cos(π * progress))
```
At `progress=1.0` (end of schedule): `lr = 0.5 * base_lr * (1 + cos(π)) = 0.5 * base_lr * 0 = 0`. The LR reaches absolute zero at schedule end.

**Impact:** Even if `step_count` were correctly restored, if the resume point is near the end of the cosine schedule (e.g., epoch 79 of 81 total), the LR would still be near zero. For the `ale.mud` training run with `epochs=3` resuming from epoch 79:
```
progress = (13035 - 135) / (13527 - 135) = 0.989
lr = 0.5 * 0.0005 * (1 + cos(π * 0.989)) ≈ 1.6e-6
```
Still effectively zero.

**Fix:** Changed to cosine annealing with min_lr floor:
```rust
let min_lr = base_lr * 0.05;
min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(π * progress))
```
At `progress=0` → `lr=base_lr`; at `progress=1` → `lr=min_lr` (5% of base, not zero).

### 3.4 Fix #3 (Cancelled): `loss_vel` Was Already 1-Step

**Initial hypothesis:** `loss_history[loss_history.len() - 2] - chunk_loss` was computing a 2-step velocity.

**Verification:** Traced the code execution order:
1. Line 1479: `loss_history.push_back(chunk_loss)` — pushes current chunk loss
2. Line 1497: `loss_history[loss_history.len() - 2]` — reads the **second-to-last** entry, which is l_{t-1}

Since `chunk_loss` was already pushed, `loss_history[len-1] = l_t` (current) and `loss_history[len-2] = l_{t-1}` (previous). The formula `l_{t-1} - l_t` is the **1-step velocity** — positive means loss decreased (improving), negative means loss increased (worsening).

**Verified with telemetry:**
- Step 13031: loss=5.5675, loss_vel=-0.0311 → l_13030 = 5.5364 (loss increased) ✓
- Step 13032: loss=5.5465, loss_vel=+0.0211 → l_13031 = 5.5675 ✓ (matches step 13031 loss)

The code was correct. No change needed.

---

## 4. Model Diagnosis: `ale.mud` Structural Issues

### 4.1 Diagnostic Verdict

```
$ diagnose_model models/ale.mud
verdict: 🔴 STRUCTURAL ISSUES
```

### 4.2 Dead Attention Key Projections (ALL Layers)

```
Layer  | attn_k.weight (ternary) | attn_k.prq_scale | attn_q.weight | attn_v.weight | attn_output.weight
-------|------------------------|-----------------|---------------|---------------|--------------------
blk.0  | 0/32 nonzero           | [1e-8, ...]     | 23/32 nonzero | 25/32 nonzero | 22/32 nonzero
blk.1  | 0/32 nonzero           | [1e-8, ...]     | 24/32 nonzero | 24/32 nonzero | 24/32 nonzero
blk.2  | 0/32 nonzero           | [1e-8, ...]     | ? nonzero     | ? nonzero     | ? nonzero
blk.3  | 0/32 nonzero           | [1e-8, ...]     | ? nonzero     | ? nonzero     | ? nonzero
blk.4  | 0/32 nonzero           | [1e-8, ...]     | 24/32 nonzero | 27/32 nonzero | 28/32 nonzero
blk.5  | 0/32 nonzero           | [1e-8, ...]     | 28/32 nonzero | 29/32 nonzero | 32/32 nonzero
```

**Every layer's attention key projection is completely dead.** All ternary weights are zero AND the PRQ scale is `1e-8` (effectively zero). This is the "dead key collapse" — the attention mechanism has no keys to compute similarity against.

### 4.3 mHC Weight Saturation

```
Layer  | mhc_alpha sample                  | mhc_beta sample
-------|-----------------------------------|-----------------------------------
blk.0  | [4.0, 4.0, 0.0, 0.0, 4.0, 0.0, ...] | [4.0, 0.0, 0.0, 0.0, 4.0, 4.0, ...]
blk.1  | [4.0, 0.0, 4.0, 0.0, 4.0, 0.0, ...] | [4.0, 4.0, 4.0, 0.0, 4.0, 4.0, ...]
blk.2  | [4.0, 4.0, 4.0, 4.0, 0.0, 0.0, ...] | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]
```

mHC alpha/beta are saturated at binary values (0.0 or 4.0). Healthy values should be near 1.0 (blending residual with FFN output). Values of 0.0 zero out dimensions; values of 4.0 over-amplify. When both alpha=0 and beta=0 for the same dimension → output = 0 → dead dimension.

### 4.4 Cascade Pathology

```
attn_k = 0  → key vectors = 0
           → Q·Kᵀ = 0 for all positions
           → softmax(0,0,...,0) = [1/N, 1/N, ..., 1/N]  (uniform)
           → attention output = mean(V) — context-free
           → signal degrades through layers 0→4
           → by layer 4-5, residual + FFN output too weak to survive mHC
           → all registers collapse to 0 → ALLZERO warning
```

### 4.5 Embedding Health (OK)

```
token_embd.weight: 85/128 nonzero per token — healthy
output_norm: mean=0.6116, not all-zero, not all-one — healthy
attn_norm weights: mean≈1.0 across all layers — healthy
```

The embedding and norm layers are healthy — the collapse is isolated to attention key projections and mHC values.

### 4.6 Missing Expert Tensors

```
18 missing tensors: blk.{0..5}.expert.0.w{1,2,3}.weight
```

These are expected — the model uses `ffn_gate`/`ffn_up`/`ffn_down` naming instead of `expert.0.w*` naming. Not a real issue, just a naming convention mismatch in the diagnostic.

---

## 5. Fixes Applied

### Files Modified

| File | Change | Fix # |
|------|--------|-------|
| `src/mud/corpus_trainer/session.rs` | Restore `step_count` from `global_chunks_processed` after resume computation | #1 |
| `src/mud/corpus_trainer/session.rs` | Move `qat_lr` computation from pre-loop to per-step inside telemetry block | #2 |
| `src/mud/corpus_trainer/train_step.rs` | Add `min_lr = base_lr * 0.05` floor to cosine annealing formula | #4 |

### Verification

```
cargo check --release --lib     → ✅ Finished (0 warnings)
cargo clippy --lib              → ✅ (0 errors, 0 warnings)
cargo test --lib --test-threads=2 → ✅ 435 passed, 2 ignored, 0 failed
```

LR schedule tests:
- `test_lr_schedule_warmup_linear_ramp` ... ✅
- `test_lr_schedule_cosine_decay` ... ✅ (updated to assert `lr_end ≈ min_lr = base * 0.05`)

---

## 6. Recovery Plan for `ale.mud`

### Option A: Retrain with Fixed LR (Recommended)

```bash
# The fixes prevent future LR collapse. Use aggressive LR to escape the dead basin.
MUD_QAT_LR=0.03 ./mud.sh train models/ale.mud --epochs 20
```

**Why this works:** The STE gradient still flows through `attn_q` → `attn_v` → `attn_output` even when `attn_k` is zero (softmax(zeros) has nonzero Jacobian). With LR=0.03 (vs the broken 3.7e-6), the gradient can push the dead `attn_k` ternary weights to non-zero values within a few hundred steps.

**Risk:** Without `MUD_QAT_LR` override, the cosine schedule will decay the LR. With Fix #4 (min_lr floor), the LR never drops below 0.000025 — but 0.03 is needed for escape. Use `--epochs 20` so the schedule doesn't reach the floor too quickly.

### Option B: Start from Healthy Base

```bash
./mud.sh convert models/smollm2  # AGENTS.md: 🟢 CERTIFIED healthy
```

Convert from the BF16 source in `models/smollm2/` (mentioned in AGENTS.md as a healthy base). Then train the fresh ternary model:
```bash
cargo run --release --bin run_trainer -- models/smollm2.mud --epochs 10
```

### Option C: Forge Fresh

```bash
./mud.sh forge nano models/fresh.mud --vocab-size 151643
cargo run --release --bin run_trainer -- models/fresh.mud --epochs 10
```

Create a new 6-layer model from scratch (matching `ale.mud` architecture) and train with the fixed LR schedule.

---

## 7. Preventing Recurrence

### Configuration Recommendations

| Setting | Value | Rationale |
|---------|-------|-----------|
| `MUD_QAT_LR` | 0.0005 (default) or 0.03 (recovery) | Default works with the min_lr floor; 0.03 for collapsing models |
| `--epochs` | ≥ 20 when resuming | Ensures the cosine schedule doesn't reach the floor before meaningful training |
| `MUD_TRAIN_WCLAMP_K` | 8 (default) | Prevents PRQ-scale inflation (observed 27× on last layers in audit) |
| `MUD_SELF_PLAY` | 1 (optional) | D1 self-play pillar prevents stagnation in local minima |

### Code-Level Guards (Already in Place)

1. **Fix #4 (min_lr floor):** `apply_lr_schedule` now floors at 5% of base_lr
2. **Recovery noise injection:** `slime_forward.rs:1226-1239` injects deterministic noise into dead registers
3. **`circuit_eval_integrity`:** Rejects models with all-zero `attn_norm.weight` (but not attn_k)
4. **NaN guards:** All forward/backward paths guard against NaN propagation (P-17)

### Missing Guard (Recommendation)

Add a "dead key detector" to `circuit_eval_integrity` that scans for all-zero `attn_k.weight` tensors before training starts, and either:
- Refuses to train (fail-fast)
- Automatically resets `attn_k` weights from embedding initialization

---

## 8. Telemetry Enrichment (Applied)

The `[TELEM]` line was enriched with three new fields per the session's telemetry goals:

```
[TELM] ... grad_norm={:.4} entropy={:.3} stp_loss={:.5} traj_eff={:.4} ...
```

| Field | Source | Purpose |
|-------|--------|---------|
| `grad_norm` | `norm_sq.sqrt()` from backward pass | Detects gradient explosion/vanishing |
| `entropy` | `-Σ p·log₂(p)` from softmax | Detects overconfident-but-wrong predictions (entropy=0.125 = pathologically low) |
| `stp_loss` | `stp_loss_accum / stp_triples` | STP trajectory loss (off when 0.0) |
| `traj_eff` | `1/(1+CV)` of recent loss history | Detects stagnation vs. productive training |

### Telemetry File Format

```
# Legacy format (columns 0-14):
#   step loss ppl lr loss_vel tok/s elapsed prog conf varh varj jepa sigma cog grad_norm entropy stp_loss traj_eff n_act
#   1  13035 5.5553 258.6 0.000004 -0.012613 9.3 12.5 79.5% 0.39% 0.059 0.319 0.002 50.0% 0.0025 6.8176 0.125 0.00000 0.9990 4096
```

Previously columns 8/9/10 (grad_norm, entropy, stp_loss) were hardcoded `0.0` placeholders. They are now populated with real values.

---

## 9. Environment

```
Model:        models/ale.mud
Architecture: 6-layer transformer, hidden=128, ffn=512, heads=2, kv_heads=1
Vocab:        151,643 tokens
Quant:        Ternary2Bit (ELUT 4-bit + PRQ)
Tokenizer:    BPE (from metadata)
Train frozen: emb=1, stp=1, 255 negatives, 64 steps/chunk
Optimizer:    adam (MUD_OPT)
Total tests:  435 passed (lib)
Clippy:       0 errors, 0 warnings (lib)
```

---

## 10. Cross-References

- **AGENTS.md** §"Telemetry enrichment" — prior session's telemetry fixes (loss_vel, traj_eff, cmud_* keys)
- **AGENTS.md** §"Trainer perf fixes" — `emb_ckpt` pre-allocation, `mem::take` zero-copy, ΔW tracking
- **AGENTS.md** §"FIX APLICADO (2026-07-20)" — PRQ-scale inflation, checkpoint hash no-op detection
- **docs/audits/MUD_AUDIT_REPORT_V21_QAT_RECOVERY.md** — QAT recovery strategies
- **docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md** — mHC α/β training (F1)
- **docs/sessions/MUD_SESSION_REPORT_2026-07-31_MHC_SYNC_FIX.md** — prior mHC sync fix (#47)
