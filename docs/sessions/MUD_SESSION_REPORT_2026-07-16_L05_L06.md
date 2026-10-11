# Session Report — L-05 + L-06 (2026-07-16)

## Scope

Close Phase A tail items **L-05** (true CPU/GPU double-buffer) and **L-06** (`mha.comp` / `rms_norm.comp`).

## L-05 — True double-buffer

### Backend (`ash_backend.rs`)
- Replaced ceremonial single-CB double-fence with **`DoubleFrame`**: 2 command buffers + 2 fences.
- `dispatch_optimizer_batch_async` → `frame.acquire` → record → `frame.submit` (slot rotates).
- `sync` / `sync_frames` wait both slots.

### Trainer path (`ash_qat_dispatcher.rs` + `corpus_trainer.rs`)
- **`step_async_deferred`**: submit optimizer, queue packed/scales readbacks, **do not wait**.
- **`flush_pending`**: fence + readback; called at start of next deferred step and via `sync_all` (checkpoint/epoch).
- Removed immediate `sync_and_readback_all` after `step_async` (was killing overlap).
- Forward N+1 uses CPU packed while GPU writes VRAM packed (distinct buffers → safe overlap).

## L-06 — mha / rms_norm

### Shaders
- Compiled SPIR-V (`--target-env=vulkan1.1`).
- Replaced fragile `subgroupAdd`-only reductions with **shared-memory tree reduce** (portable across subgroup sizes on Iris Xe).

### Dispatch
- `AshContext::dispatch_rms_norm_sync` / `dispatch_mha_sync`.
- Thresholds: `RMS_GPU_MIN_HIDDEN = 512`, `MHA_GPU_MIN_WORK = 64`.
- `apply_output_norm` tries GPU RMSNorm when hidden ≥ 512.
- `try_gpu_dense_mha` for short dense prefills (seq ≤ 64); **decode path stays CPU+HCA** (layout mismatch).

## Validation

- `cargo test --lib` → **104 passed**
- `cargo clippy --lib -- -D warnings` → clean
- GPU tests (skip if no Vulkan): DoubleFrame slot rotate, RMSNorm vs CPU, MHA seq=1 identity

## Ledger

| ID | Status |
|----|--------|
| L-05 | **DONE** |
| L-06 | **DONE** |
| Phase A (L-01…L-08) | **CLOSED** |

## Next

L-09 EZOP / Phase B perf; L-10 packing; L-11 Mini MoE.
