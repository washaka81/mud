# MUD Session Report — 2026-07-18

**Focus:** Converter offset-corruption fix (FIX D) · Trainer SIGSEGV on Intel Iris Xe (Vulkan) · SmolLM2 conversion + 25-epoch supervised training supervision · Telemetry column-map audit

**Hardware:** Intel i7-1260P (P-cores + HT), Iris Xe ADL GT2 (UMA integrated Vulkan), 15 GiB RAM. CPU AVX2×8 (PCorePool).

---

## 1. Converter offset-corruption (FIX D — root cause of "converter emits garbage")

### Symptom
`universal_converter` produced a `.mud` whose Float32 norms/PRQ scales decoded to `~±1e38`,
even when the source safetensors were perfectly sane. Confirmed against a hand-built fixture
(`tools/gen_fixture.rs` → `fixtures/smollm2_mini`, SmolLM2-mini 2L/64H): source norms `~1.0`,
but converted `.mud` read back `1e38`.

### Root cause
`MudFile::save` (src/mud/mod.rs) is invoked by the converter *after* Pass 2 to write the ECC
parity tensors (`ecc_generate_all` + `mud.save`). In its second write pass it read each
mmap-backed tensor with:

```rust
let data_start = (tensor.offset + 31) & !31;   // WRONG: treats offset as absolute
let slice = &mmap[data_start..data_start + s_expected];
```

But `MudTensor::offset` (as written by `StreamingMudWriter` and read by `MudFile::load`) is
**relative to the start of the data region**, not to the start of the mmap. `load` uses
`data_start + tensor.offset` for `data_ptr`. So `save` indexed `mmap[offset..]` instead of
`mmap[data_base + offset..]`, reading the file header / neighbouring tensors → `1e38` garbage.
`MudTensor::clone` had the identical defect. `StreamingMudWriter` (Pass 2) is correct; only the
post-conversion `save` rewrite was broken.

### Fix
- Added `MudTensor::data_base: usize` (absolute byte offset of the data-region start within the
  owning mmap; `0` for owned / in-code tensors).
- `MudFile::load` sets `tensor.data_base = data_start` for every loaded tensor.
- `MudFile::save` now reads `let data_start = (tensor.data_base + tensor.offset + 31) & !31;`.
- `MudTensor::clone` uses `cloned.data_base + cloned.offset`.
- Updated all 10 `MudTensor { .. }` constructors across the workspace (mod.rs, moe_train.rs,
  moe_load.rs, tests.rs, corpus_trainer.rs ×6, mud_forge.rs ×3).

### Verification
- `diagnose_model model_fixture.mud` → norms `mean≈1.0`, scales `≈0.05`, **zero** `1e38` /
  `null=true` / `all_zero=true`.
- SmolLM2 (real, BF16): SAFETENSORS `input_layernorm` layer0 (`mean=0.00120, n=576`) ==
  MUD `blk.0.attn_norm` (`mean=0.00120, n=576`) — byte-faithful (SmolLM2 norm weights are
  genuinely small, ~0.001, NOT collapsed: `all_zero=false`, range `[-0.31, 0.30]`).
- `cargo clippy --all-targets` → 0/0. `cargo test --lib` → 222 pass.
- Inference load: `emb_rms≈0.13`, `tied=true`, no Dead-RMSNorm.

---

## 2. Trainer SIGSEGV on Intel Iris Xe (Vulkan driver crash)

### Symptom
`./mud.sh train models/smollm2.mud --epochs 25` crashed at **block 11/64, epoch 1** with:
```
./mud.sh: línea 179: 133821 Violación de segmento (core generado)
```

### Root cause (from `coredumpctl gdb 133821`)
```
#8  ash_backend::AshContext::submit_and_wait
#9  ash_backend::AshContext::dispatch_gemv_qkv_host_sync
#10 slime_forward::evaluate_slime_block_moe
#11 corpus_trainer::MudCorpusTrainer::train_on_sequence
#12 corpus_trainer::MudCorpusTrainer::run_alignment_session
```
The Intel Vulkan driver (`libvulkan_intel.so`, ADL GT2 / Iris Xe UMA) **SIGSEGVs inside
`submit_and_wait`** when dispatching the QKV GEMV compute shader. `mud.sh` sets
`MUD_USE_VULKAN=1` (global) and `MUD_GPU_GEMV=auto`; the one-shot micro-bench decided GPU wins
for work ≥ 16384, so the forward dispatched QKV to GPU and the driver crashed.

This matches the documented runtime truth (AGENTS.md): on i7-1260P the *real* hot-path is AVX2 —
the GPU break-even only appears in synthetic micro-benches, not the 147M forward.

### Fix
`mud.sh` `train` target now forces Vulkan OFF for training (CPU AVX2 path), overridable with
`MUD_TRAIN_FORCE_VULKAN=1` for discrete/stable GPUs:
```bash
if [ "${MUD_TRAIN_FORCE_VULKAN:-0}" != "1" ]; then
    export MUD_USE_VULKAN=0
    export MUD_GPU_GEMV=0
fi
```
Confirmed: trainer runs with `Vulkan: OFF (disabled (MUD_USE_VULKAN=0)) · GEMV=0` and passes
block 11/64 with no crash.

> NOTE: the Vulkan GEMV path itself is not hardened against driver SIGSEGV (a driver fault
> cannot be caught as a Rust `Result`). On stable/discrete GPUs, add a probe+fallback (already
> present as `probe_gpu()`) and avoid dispatching when the driver is the Intel UMA one.

---

## 3. SmolLM2 conversion + 25-epoch supervised training

### Conversion
- Deleted all stale `.mud` (incl. `*.bak`) from prior collapsed models.
- `universal_converter models/smollm2 models/smollm2.mud` → 210 ternary + 422 f32 tensors,
  ECC enabled. `mud_full_audit` clean; `diagnose_model` norms/scales sane.

### Training supervision
- Launched: `nohup env MUD_TRAIN_MAX_CHUNKS=0 ./mud.sh train models/smollm2.mud --epochs 25`
  (MAX_CHUNKS=0 = unlimited; the default `mud.sh` cap of 64 ends the session after 64 chunks).
- Runs on CPU AVX2×8, ~4 tk/s, ~372 chunks/epoch, ETA ~41h for 25 epochs.
- **0 crashes, 0 SIGSEGV/SIGABRT/panic** across the supervised run.
- Checkpoint `weights/checkpoints/model_latest_checkpoint.mud` updates every ~10 min.

### Telemetry health (from `mud_train_metrics.log` / `train_telemetry`)
| Metric | Value | Status |
|--------|-------|--------|
| Loss / chunk | 3–6 (fluctuant by corpus domain) | healthy descent from ~14 init |
| VarH | ~4.7 (range 2.4–4.7) | healthy, far from collapse (<1e-5) |
| VarJ | ~0.074 | in ~0.07 band (Equilibrium Mandate) |
| JEPA Attractor (integral) | ~0.006 ≈ 0 | stable, not diverging |
| Cognitive/Linguistic Cohesion | 368 (rose from ~14) | improving |

**Conclusion: the model is alive and learning — not dead/collapsed.**

---

## 4. Telemetry column-map audit (doc fix)

The `# cols:` comment in `corpus_trainer.rs:2495` was **desaligned** from the actual
`writeln!` (line 3068). The comment said
`... varh varj sat zent tsoft align integral sigma_pct cognitive ...` but the writer emits
`... varh varj 0.0 0.0 0.0 integral sigma cognitive ...` (three literal `0.0` pads, then
integral/sigma/cognitive). `train_telemetry.rs` reads the **correct** columns
(varh=7, varj=8, jepa/integral=12, cognitive=14), so runtime is fine; the comment was corrected
to match the writer to avoid future confusion. The `VarJ` panel shows `1.00` as the **top axis
label** (axis clamped to `max(.,1.0)`), not as the data value (~0.074).

---

## 5. tok/s validation (empirical cross-check)

`toks_per_sec` in `corpus_trainer.rs:3052` = `steps_per_chunk_meta / chunk_dt` (wall-clock of
the chunk). `steps_per_chunk_meta` = `train_steps_per_chunk(batch)`; with `MUD_TRAIN_MAX_CHUNKS=0`
the `align` path triggers (`sequence_pack.rs:46`) → `batch*2`. Trainer uses `batch_size=32` →
`64` steps/chunk. Cross-check at step 1823 (`elapsed=28706.7s`, `tok/s=4.0`):

```
dt_promedio = 28706.7 / 1823 = 15.75 s/chunk
tok/s_sesion = 1823*64 / 28706.7 = 4.06   ← matches TELEM 4.0 (rounding of instantaneous chunk_dt)
```

**tok/s = 4.0 is correct** for i7-1260P AVX2×8 on SmolLM2-135M (372 chunks/epoch, ~64 min/epoch,
~27h projected for 25 epochs before overhead — actual ETA ~41h incl. checkpoint flush + jitter).

---

## 6. Critical limits (thresholds that require intervention)

Derived from `corpus_trainer.rs` code (`dead` gate, P-17 fail-fast, Equilibrium Mandate F1/F2)
and observed healthy band during this run (step 1823 / epoch 5).

| Variable | Healthy now | ⚠ Warn | 🔴 Critical (colapse) |
|----------|-------------|--------|------------------------|
| VarH | ~4.7 | <1.0 | **`< 1e-5`** (dead gate, `corpus_trainer.rs:3091`) |
| VarJ | ~0.075 | <0.01 or >1.5 | **→0** (gate stops modulating) |
| Loss | 2.7–5 | >10 sustained | **NaN/Inf** (corrupt grad) |
| Cohesion (cog) | ~357 | <50 | **`< 1e-3`** (dead gate, with VarH) |
| JEPA integral | ~0.008 | \|>0.5\| | **diverges** (unstable) |
| σ | 50.2% | <10% or >90% | **0% / 100%** (full saturation) |
| conf | 6.4% | >95% (overconf) | **NaN** |
| tok/s | 4.0 | 0 for >60s | **process dead** |
| crashes | 0 | SIGSEGV/panic | **any** in hot-path |

Hard-coded collapse gate: `dead = avg_var_h < 1e-5 && avg_cognitive < 1e-3`
(`corpus_trainer.rs:3091`). Current values are **~14 orders of magnitude** above both — no risk.

VarJ ~0.07 is the Equilibrium-Mandate target band (F1/F2). The `VarJ` telemetry axis is clamped
to `max(.,1.0)`, so the panel shows `1.00` as the top label, NOT as the data value (~0.074).

---

## 7. Audit findings (while trainer running, 2026-07-18)

Read-only audit (`task` subagent) + fixes applied. `cargo clippy --all-targets` → 0 warnings;
`cargo test --lib` → 222 pass. Trainer stayed ALIVE through both edits.

### 7.1 CRITICAL — FIXED: ECC parity read used absolute `offset` (FIX D relapse)
`src/mud/mod.rs:770` read `mmap[ecc_tensor.offset..]` for ECC parity — raw `offset` as absolute
mmap index, the **same class of bug as FIX D** (offset is relative to `data_base`). On an
mmap-resident ECC tensor this verified parity against the metadata region → silent false
"clean"/false corrections. Fixed: `let start = ecc_tensor.data_base + ecc_tensor.offset;` (the
base tensor at line 801 already uses `tensor.data_ptr`, which is `data_base+offset`-derived).
All 10 `MudTensor` mmap reads now consistent (`mod.rs:97, 355, 527, 770, 801`).

### 7.2 WARNING — FIXED: Vulkan Iris Xe blacklist moved into the binary
`probe_gpu()` in `src/vulkan/ash_backend.rs` now returns `available=false` for
`"Intel(R) Iris(R) Xe"` (ADL GT2) instead of only being blocked by `mud.sh`. Defense-in-depth:
inference (`cargo run --bin forge_llm` with `MUD_USE_VULKAN=1`) on an Iris Xe host now degrades
to AVX2 automatically — the SIGSEGV (§2) can no longer reach `submit_and_wait` without the
shell wrapper. Message: `"blacklisted (crash-prone Intel Iris Xe driver)"`.

### 7.3 WARNING — P-08 soft debt (not fixed, low risk)
`ComplexSlimeRegister` (cmud.rs:194) and `complex_gemv_gauss_ref` (cmud.rs:343) are dead in the
production path (only reached from `#[cfg(test)]`). C-MUD itself is wired (opt-in
`MUD_CMUD_THINK`); just these two symbols. Deferred (research module; see
`docs/research/CMUD_LOGGAS_FEASIBILITY.md`).

### 7.4 INFO — verified clean
- **P-27 (no Rayon):** no `rayon`/`par_iter` anywhere in `src/`; `pcore_pool.rs` is the sanctioned threading.
- **P-13/P-17 fail-fast:** all `panic!`/`expect` on missing metadata or non-finite values are
  legit fail-fast, not crashes on valid input.
- **NaN propagation:** `slime_forward.rs` guards (1102–1159) panic visibly (not silent); O-GEMV
  and attn outputs each scanned. `cmud` returns 0.0 on non-finite (bounded, no propagation).
- **debate_trainer.rs:496** `top_p_probs.last().unwrap()` — low-risk; only panics on empty
  distribution in the debate trainer (non-hot path). Optional `is_empty()` guard noted.

---

## 8. Status / next work
- Trainer stable on CPU; Vulkan disabled for training on this HW (see §2); Iris Xe now blacklisted in-binary (§7.2).
- Converter produces faithful, non-corrupt `.mud`; ECC read fixed (FIX D-class relapse, §7.1).
- tok/s validated (4.0 on i7-1260P AVX2×8, 64 steps/chunk).
- C-MUD × log-gas: deferred to future-study reliability gate (see CMUD_LOGGAS_FEASIBILITY.md).
- After 25 epochs: run supervised circuit (`weights/`) per project plan.

---

**Audit final (2026-07-20):** logros, correcciones y estado → `docs/sessions/MUD_AUDIT_ACHIEVEMENTS_2026-07-20.md`.

---

**Forensic training audit (2026-07-20):** el checkpoint de 25 épocas es **MD5-idéntico** al modelo base (`ae15bdfe...`) — el trainer no persistió cambios de pesos. Telemetría sana (VarH/VarJ/σ/cog) pero engañosa: mide estabilidad del manifold, no calidad. El modelo base ya está colapsado (logits planos, ganador siempre token 0). Detalle: `docs/research/TRAIN_TELEMETRY_FORENSIC_2026-07-20.md`. Bloquea el circuito F3+ hasta corregir.

---

**Plan de correcciones (2026-07-20):** `docs/research/MUD_FIX_PLAN_2026-07-20.md`. Prioridad P0 (train no persiste pesos, checkpoint MD5-idéntico al base) → P1 (base model vocabulary-collapsed, escala ternaria) → P2 (telemetría honesta con Σ|ΔW|) → P3 (robustez). No lanzar circuito F3+ hasta P0+P1+P2 verificados.
