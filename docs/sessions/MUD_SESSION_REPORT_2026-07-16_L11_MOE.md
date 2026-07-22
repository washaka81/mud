# Session Report — L-11 Mini MoE (2026-07-16)

## Delivered

### `src/mud/slime_expert.rs` (C1)
- `SlimeExpert`: ternary SwiGLU FFN (up / gate / down + PRQ scales)
- `from_ptrs` / `from_dense_layer` (wrap dense `SlimeLayer` FFN)
- `forward_swiglu` via shared `ternary_gemv_rowwise` + ASM SiLU

### `src/mud/expert_bus.rs` (C2 + C3 + C7)
- Up to 64 slots, **hot `mount` / `unmount`**
- `MudRouter` Softmax / Gumbel / Hash modes
- Optional ternary router GEMV (`set_router`)
- **Dense fallback** when ≤1 expert or missing router (C7)
- `ExpertScratch` prealloc (P-01)
- `forward` → route + weighted sum of expert outputs

### Forward hook (C7-compatible)
- `evaluate_slime_block` unchanged for callers (delegates)
- `evaluate_slime_block_moe(..., moe, scratch)` replaces dense FFN when bus mounted
- On bus error → dense FFN fallback + stderr notice

### Reused
- `routing.rs` top-k + Gumbel + hash (already tested)

## Not in this slice (follow-ups)
| Item | Notes |
|------|--------|
| C5 MUD v2 tensor names | `blk.N.expert.K.*` load path |
| C6 expert-only training | `--train-expert` |
| Full STE backward per expert | needs expert grad buffers |
| Parallel multi-expert on PCorePool | sequential weighted loop today |

## Validation
- `cargo test --lib` → **126 passed**
- `cargo clippy --lib -- -D warnings` → clean

## Ledger
| ID | Status |
|----|--------|
| L-11 | **DONE** (core bus + hook) |
| Next | Phase B perf / L-12 |
