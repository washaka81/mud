# Recovery Plan: `ale.mud` Attention Key Collapse

**Model:** `models/ale.mud` (6-layer transformer, hidden=128, Ternary2Bit)  
**Status:** 🔴 Dead key collapse — all `attn_k` weights are zero across all layers  
**Date:** 2026-08-02

---

## Problem Summary

`ale.mud` has **completely dead attention key projections** in all 6 layers:

| Tensor | Ternary Nonzero | PRQ Scale |
|--------|----------------|------------|
| `blk.N.attn_k.weight` | 0/32 (all layers, all rows) | 1e-8 (effectively zero) |

When all attention keys are zero:
- `Q·Kᵀ = 0` for all position pairs
- `softmax(0, 0, ..., 0) = [1/N, ..., 1/N]` — uniform attention
- The model **cannot attend to any positions** — it's context-blind
- Signal degrades through layers → L4/L5 produce near-zero activations → ALLZERO warnings

The mHC alpha/beta weights are also saturated at binary values (0.0 or 4.0), zeroing out dimensions in the residual path.

## Root Cause

The training session that produced `ale.mud` had a **broken LR schedule**:

1. `step_count` was not restored from checkpoint → LR started at warmup position (3.7e-6)
2. `qat_lr` was computed once per session → telemetry showed stale LR, couldn't diagnose
3. No `min_lr` floor → cosine decay drove LR to 0 at schedule end
4. Effective LR of ~1e-6 for the entire session → STE gradients couldn't escape the all-zero key basin

## Recovery Options

### Option A: Retrain `ale.mud` In-Place (Recommended)

Use the fixed LR schedule + aggressive LR to push dead attn_k weights back to life.

```bash
# The 4 fixes (step_count restore, per-step qat_lr, min_lr floor, recovery jitter)
# are already applied to cargo build --release. Use aggressive LR for escape.
MUD_QAT_LR=0.03 ./mud.sh train models/ale.mud --epochs 20
```

**Why LR=0.03:**
- At LR=3.7e-6, each step moves weights by ~3.7e-6 × gradient. With ternary STE, the gradient magnitude is O(|x|) where x is the input. For hidden=128, typical gradient ≈ 0.01. So step Δ ≈ 3.7e-8 — negligible.
- At LR=0.03, step Δ ≈ 0.03 × 0.01 = 3e-4. This is large enough to push ternary weights through the 0.5 threshold (STE deadzone) and revive dead keys.
- The `MUD_TRAIN_WCLAMP_K=8` default prevents the revived weights from blowing up.

**Expected behavior after ~500-1000 steps:**
- `loss` should drop from 5.555 (ln 256) toward 3-4
- `entropy` should rise from 0.125 toward 2-4 bits
- `grad_norm` should decrease from 6.8 toward 0.1-1.0
- `conf` should drop from 1/256 toward 0.5-1% (model becomes less overconfident)
- ALLZERO warnings should disappear

**Monitor:** `./mud.sh telemetry` or tail `logs/mud_train_metrics.log`:
```bash
tail -f logs/mud_train_metrics.log | grep -E "loss=|entropy=|grad_norm="
```

### Option B: Convert from Healthy Source

The `smollm2` source in `models/smollm2/` is noted as "🟢 CERTIFIED healthy" in AGENTS.md (2026-07-20 fixes).

```bash
# Convert BF16 source → ternary .mud
cargo run --release --bin universal_converter -- \
    models/smollm2/ models/smollm2_fresh.mud \
    --check
```

Then train the fresh ternary model with the fixed schedule:
```bash
MUD_QAT_LR=0.0005 ./mud.sh train models/smollm2_fresh.mud --epochs 10
```

### Option C: Forge and Train from Scratch

```bash
# Create new 6-layer model matching ale.mud architecture
./mud.sh forge nano models/fresh_6l.mud --vocab-size 151643

# Train with fixed LR schedule
MUD_QAT_LR=0.0005 ./mud.sh train models/fresh_6l.mud --epochs 20
```

## What NOT To Do

| ❌ | Action | Reason |
|----|--------|--------|
| 1 | Train with default LR (0.0005) from epoch 79 resume | Step count restore helps, but cosine will still decay to near-zero. Need `--epochs 20+` or `MUD_QAT_LR=0.03` |
| 2 | Train without the 4 fixes | Step_count=0 → LR=3.7e-6 → same collapse |
| 3 | Try to patch weights manually | Ternary weights are bit-packed; can't partially recover without full retrain |
| 4 | Use `MUD_CMUD_THINK=1` | C-MUD adds reasoning complexity — don't layer on top of a collapsing model |

## Monitoring Checklist

Watch these metrics in telemetry to confirm recovery:

```
Phase 1 (0-500 steps):  Loss starts dropping from 5.555 → ~5.0
  - entropy rises from 0.125 → ~0.5
  - grad_norm stays > 6.0 (steep gradient, low LR still)
  - conf drops from 0.39% → ~0.2% (model becomes less overconfident)

Phase 2 (500-2000 steps): Loss 5.0 → ~3.5
  - entropy 0.5 → ~2.0
  - grad_norm 6.0 → ~1.0
  - traj_eff drops below 0.99 (now learning, not just stable)

Phase 3 (2000+ steps): Loss 3.5 → ~2.0+
  - entropy 2.0 → ~3-5 (healthy diversity)
  - grad_norm 1.0 → ~0.1-0.5 (converging)
  - NO ALLZERO warnings
  - conf rises to 5-20% (model has real understanding)
```

## Post-Recovery Validation

Once training stabilizes (loss < 3.0, entropy > 2.0, no ALLZERO):

```bash
# Verify structural health
cargo run --release --bin diagnose_model -- models/ale.mud | grep -E "verdict|attn_k.*non"

# Run inference health check
./mud.sh health models/ale.mud

# Quick chat test
./mud.sh chat models/ale.mud <<< "hola"
```

Expected: `verdict: 🟢 ALL CHECKS PASSED` (or at least no ALLZERO warnings, attn_k weights nonzero).
