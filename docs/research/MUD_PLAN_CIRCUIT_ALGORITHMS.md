# MUD Circuit Trainer — Algorithm Review, Dead-RMSNorm Root Cause & Corrections

**Date:** 2026-07-18 · **Status:** ROOT CAUSE FOUND + 3 calc bugs fixed + events→circuit.log.

This document replaces the earlier "SIGSEGV + panic hardening" notes. The
circuit trainer is now reviewed end-to-end; the Dead-RMSNorm is **not** a trainer
bug — it is a *collapsed input model*. Two real calculation bugs were found and
fixed, plus the integrity gate and event logging were corrected.

---

## 1. Dead RMSNorm — Root Cause (NOT a trainer bug)

`slime_rmsnorm_i8` (`src/mud/slime_forward.rs:4`) computes

```
rms_inv   = 1/sqrt(sum(regs²)/n + eps)
xn[i]     = regs[i] * rms_inv * attn_norm_w[i]
peak      = max|xn|  (floored to EPSILON_FLOOR)
act_scale = peak / 127
```

The fail-fast (`slime_forward.rs:786`) triggers `peak_norm==0 && act_scale<1e-7`
only when **every** `xn[i] == 0`. Since `regs` had `nz_regs=576` non-zero inputs
(the panic message proved the *input* to block 0 is alive), `xn==0` ⟹
**`attn_norm_w` is all-zero**.

### Verified on disk (not a read bug)

```
$ diagnose_model models/smollm2.mud
blk.0.attn_norm.weight | Float32 | sample=[0.0,0.0,0.0,0.0,...]
blk.0.attn_q.weight    | Ternary2Bit | ternary_sample: 0/32 nonzero
blk.0.attn_q.prq_scale | Float32 | sample=[1e-8,1e-8,...]
```

`models/smollm2.mud` and `weights/checkpoints/model_latest_checkpoint.mud` are
**collapsed**: RMSNorm weights = 0, ternary weights = 0, PRQ scales = 1e-8.

### Why the trainer cannot repair it

In `materialize_for_ste_train` (`src/mud/mod.rs:619`) and
`materialize_writable`, the predicate at `mod.rs:640` is

```rust
// norms / ecc / misc: never written by STE pack → keep mmap
false
```

RMSNorm / FFN-norm weights are **frozen on read-only mmap** and never part of the
STE shadow (`run_alignment_session` only inflates `attn_q/k/v/output` + FFN
`up/gate/down`, `corpus_trainer.rs:1990`). So a 0-norm model produces 0 activations
on block 0 forever — the circuit cannot learn out of it. This is a *converter /
input-data* problem, not a circuit-trainer algorithm bug.

### Contrast: a healthy model flows fine

`models/ternary_bonsai_1.7b.mud` has `blk.0.attn_norm.weight` = non-zero
(`[0.068,0.088,...]`, `diagnose_model`). Running the circuit on it:
- health-check passes (`✓ norms alive`),
- baseline benchmark forward runs with **no Dead-RMSNorm panic**,
- `STATS|0.0163|0.0058` (finite JEPA variance — alive, not collapsed).

---

## 2. Three real calculation bugs found & fixed

### FIX A — Double token-normalization in SGD (LR shrinks by num_tokens²)

In `apply_optimizer_cpu_step_and_pack` the gradient is normalized **once**:
```rust
scale_grad_by_tokens(grad, num_tokens);   // adam_state.rs:205  → g /= num_tokens
```
But `sgd_step` / `sgd_step_avx2` then divided **again**:
```rust
g_val /= ntok;                            // ezop.rs:67  (old)
g_val = g / num_tokens;                   // avx_math.rs:102 (old)
```
Result: effective LR was `lr / num_tokens²` (e.g. `num_tokens≈64` → 4096× too
small). Training was essentially frozen. **Other optimizers (Muon/GaLore/Adam/
Chunked) only normalize via `scale_grad_by_tokens`** — so SGD was uniquely broken.

**Fix:** removed the division inside `sgd_step` (`src/mud/ezop.rs:59`) and
`sgd_step_avx2` (`forge_autograd/src/avx_math.rs:57`). `scale_grad_by_tokens` is
now the single normalization point for **all** strategies. Test
`test_sgd_matches_safe` updated to the new contract (`ntok=1`).

### FIX B — `update_prq_scales_only` sign anulled by `.abs()` (SCALES_ONLY path)

`corpus_trainer.rs` (path `MUD_TRAIN_SCALES_ONLY=1`): the least-squares fit
computes a signed scale `s_signed` when `dot<0` (shadow anti-correlated with the
frozen ternary codes T), but then wrote

```rust
*scales_ptr.add(r) = s_signed.abs().max(EPSILON_FLOOR);   // old — killed the sign
```

Since the forward applies `out = w * scale` (sign-aware, `slime_backward.rs:265`),
a negative scale is valid and *required* to flip anti-correlated rows. The `.abs()`
forced a positive scale → wrong manifold projection.

**Fix:**
```rust
let mag = s_signed.abs().clamp(EPSILON_FLOOR, 1.0);
*scales_ptr.add(r) = if s_signed < 0.0 { -mag } else { mag };
```
Sign preserved; magnitude clamped to a sane PRQ range.

### FIX C — Integrity gate ignored collapsed norms

`circuit_eval_integrity` only checked that attn/FFN *weight* tensors were
present/non-null. A collapsed model (0 norms, 0 weights) passed integrity, then
panicked with Dead-RMSNorm inside a **PCorePool worker thread** — which
`catch_unwind` in the main thread **cannot** capture (kills the whole process).

**Fix:** added `model_norms_alive(path)` — loads the model, checks
`blk.{0..3}.{attn_norm,ffn_norm,norm}.weight` Float32 tensors for any non-zero /
finite value. Used by:
- `circuit_eval_integrity` → fails the gate early with a clear message.
- A **pre-loop health-check** in `run_training_circuit` → `bail!`s with
  `"model collapsed (Dead RMSNorm risk): ... replace with a healthy .mud"`
  *before* any forward, so no worker-thread panic can kill the process.

---

## 3. All arena events now persisted to `circuit.log`

Previously only `run_training_circuit`'s `announce` wrote to `logs/circuit.log`
(via `trainer_ui::circuit_event`). The debate/arena events (`Player A:`,
`[JUEZ]`, `=== INICIANDO ARENA DE JUEGO: ...`, `match #`, `[thinking]`, `STATS|`)
went only to the optional TUI `Sender`.

**Fix:** in `run_debate_session` the sender is wrapped by a relay thread
(`corpus_trainer.rs`, start of `run_debate_session`): every message is written to
`circuit.log` (timestamped `arena` kind) **and** forwarded to the live TUI.
Verified live on bonsai:

```
[15:22:26] arena ⚔️ Starting MUD Debate Arena Session...
[15:24:35] arena === INICIANDO ARENA DE JUEGO: Math Challenge ===
[15:24:35] arena [JUEZ] max_new_tokens auto = 3 (RAM-disponible)
[15:24:35] arena [thinking] Alpha (A) generando respuesta...
[15:24:47] arena Player A: .Day,��
[15:24:47] arena STATS|0.0163|0.0058
```

---

## 4. Algorithm review — circuit trainer flow (verified correct)

### Phases (shuffled per seed via LCG, no RNG crate)
`align` (STE QAT) · `debate` (RLVR TextJudge) · `games` (verifiable
Math/TicTacToe) · `professor` (ProfessorStudent). Time-boxed by
`MUD_CIRCUIT_MAX_PER_MODE` (default 120s).

### Per-phase persistence + honors eval
1. Snapshot `.mud` → `.bak_circuit`.
2. Run phase inside `catch_unwind` (catches main-thread panics; worker-thread
   panics are now prevented by the health-check).
3. On `!ok` → rollback to `.bak_circuit`.
4. On `ok` → `circuit_eval_integrity` (now norm-aware) + `circuit_benchmark_games`
   (win-rate vs baseline, tolerance `HONORS_TOL=0.15`). Honors → keep; else rollback.

### STE QAT (align)
- Shadow inflated from ELUT+PRQ (`dequantize_ternary_row` × `prq_scale`).
- Forward/backward via `SlimeWorkspace` (AVX2×PCorePool, f32 `matmul_accum`).
- `apply_optimizer_cpu_step_and_pack`: strategy dispatch (Muon/GaLore/Chunked/
  Adam/SparseAdam, SGD fallback) → STE pack with
  `scale = (Σ|w|/cols)·(1/√2)`, `threshold = scale·0.7` (consistent across serial
  `pack_elut_prq`, parallel pack, and GPU paths — verified `ezop.rs:144` ≡
  `corpus_trainer.rs:4625`).
- Gradient normalized by `num_tokens` **once** (FIX A).

### RLVR (debate/games/professor)
Local judges only (no API, P-07): `VerifiableJudge`, `RustJudge`, `TextJudge`
(claim cosine), `ProfessorJudge`. Score → reward/penalty → SGD on shadow.

### Known limitation (documented, not a bug)
`catch_unwind` cannot capture panics raised in a PCorePool worker thread. The
pre-loop health-check (FIX C) removes the only known trigger (collapsed model).
If a *future* worker-thread panic appears, it must be caught inside the pool task
or guarded by a model health-check before the forward.

---

## 5. How to run / validate

```bash
# REQUIRES A HEALTHY .mud (smollm2.mud / the checkpoint are collapsed):
./mud.sh circuit models/ternary_bonsai_1.7b.mud

# Or headless:
cargo run --release --bin run_trainer -- models/ternary_bonsai_1.7b.mud --circuit

# Watch ALL events (circuit + arena) in the log:
tail -f logs/circuit.log
```

**Do NOT use `models/smollm2.mud` or `model_latest_checkpoint.mud`** for the
circuit — they are collapsed (FIX C makes the circuit refuse them with a clear
error instead of a Dead-RMSNorm crash).

### To repair a collapsed model
Root cause lives in the *converter* (`tools/universal_converter`): it emitted
all-zero RMSNorm weights + all-zero ternary weights + `1e-8` PRQ scales for
`smollm2.mud`. Re-run the converter from a real SmolLM2 checkpoint, or obtain a
healthy `.mud` (e.g. `ternary_bonsai_1.7b.mud`). The trainer itself is correct.

## 6b. Converter offset-corruption bug (FIX D — 2026-07-18)

While validating the converter against a *healthy* fixture
(`tools/gen_fixture.rs` → `fixtures/smollm2_mini`, SmolLM2-mini 2L/64H), the
output `.mud` was still read back as **corrupt**: every Float32 norm/scale
(except layer-1) decoded to `~±1e38`, despite the source tensors being
perfectly sane (`~1.0` for norms, `~0.05` for PRQ scales).

### Root cause
`MudFile::save` (src/mud/mod.rs) is invoked by the converter *after* Pass 2 to
write the ECC parity tensors (`ecc_generate_all` + `mud.save`). In its second
write pass it reads each mmap-backed tensor with:

```rust
let data_start = (tensor.offset + 31) & !31;   // WRONG: treats offset as absolute
let slice = &mmap[data_start..data_start + s_expected];
```

But `MudTensor::offset` — as written by `StreamingMudWriter` and read by
`MudFile::load` — is **relative to the start of the data region**, not to the
start of the mmap (the absolute data-region base = `data_start` computed in
`load` and used as `data_start + tensor.offset` for `data_ptr`). So `save`
indexed `mmap[offset..]` instead of `mmap[data_base + offset..]`, reading the
file header / neighbouring tensors → `1e38` garbage. The `MudTensor::clone`
path had the identical defect. The `StreamingMudWriter` Pass-2 path is correct;
only the post-conversion `save` rewrite was broken.

### Fix
- Added `MudTensor::data_base: usize` (absolute byte offset of the data-region
  start within the owning mmap; `0` for owned/in-code tensors).
- `MudFile::load` sets `tensor.data_base = data_start` for every loaded tensor.
- `MudFile::save` now reads `let data_start = (tensor.data_base + tensor.offset + 31) & !31;`.
- `MudTensor::clone` uses `cloned.data_base + cloned.offset`.
- All `MudTensor { .. }` constructors across the workspace updated with
  `data_base: 0`.

After the fix, `diagnose_model` on the converted fixture shows norms `≈1.0` and
PRQ scales `≈0.05` for **all** layers; forward load reports a healthy
`emb_rms≈0.1` (no Dead RMSNorm). `cargo clippy --all-targets` 0/0,
`cargo test --lib` 222 pass.

---

## 7. Future Plan: RPG Battle Circuit (Barra de Vida)

**Objective:** Introduce an evolutionary survival mechanic into the training circuit, tracking model performance like an RPG game.

### Mechanics (Planned)
1. **Health Bar (Barra de Vida):** Models will be instantiated with a persistence state including "health" or "HP" metrics (tracked across circuit boundaries).
2. **Evolution & Replacement:** When battling in the circuit against its doppelgänger (a clone or mutated snapshot), the winner becomes "Player A" (the primary model).
3. **Rewards:** The surviving model receives a "strength boost" and is rewarded with a 5-epoch training session on the core dataset.
4. **RPG Stats:** Models will save RPG-like stats (e.g. `Win Rate`, `Debate Coherence`, `Math Logic`, `HP`) in the `.mud` file's metadata or a sidecar database.
5. **Forced Study / Topic Rotation:** Each loop will rotate the RNG seed and force the models to "study" a specific topic. The sequence is:
   `Training Circuit (Study Topic) → Debate Battle → Survive/Replace → Repeat`.
6. **Goal:** This creates an evolutionary pressure, ensuring only the most coherent and factually robust checkpoints survive the rotation.


## 8. Circuit Seed, Question Determinism, and Model Persistence

**Date:** 2026-07-21

During the development of the RPG telemetry and UI integration, the behavior of the circuit's seed and the model persistence mechanism were formally documented:

### 8.1 Model Persistence
- **In-Place Writeback:** The circuit overwrites the original `.mud` file passed via the command line at the end of successful phases (`run_debate_session`). It does so safely via an atomic rename (`.tmp` -> `.mud`).
- **Phase Backups:** At the start of each phase, the model is copied to `<model_name>.bak_circuit`. If the phase corrupts the model or the model fails the *Honors-mode eval* (integrity and win-rate check), the circuit automatically rolls back from this backup.
- **Deep Alignment Checkpoints:** During the `align` phase (STE QAT), periodic checkpoints are saved to `weights/checkpoints/model_latest_checkpoint.mud`.

### 8.2 Seed vs. Question Selection
- **Deterministic Questions (P-07):** The questions asked by the Judge (e.g., `ProfessorStudent` grammar exercises, `MathChallenge`) do **NOT** depend on the random seed. They are drawn sequentially from a fixed pool using a rotating atomic counter (`fetch_add(1) % pool.len()`). This ensures the model is exposed to all exercises equally and maintains a reproducible evaluation environment without relying on RNG.
- **Phase Battery Shuffling:** The random `seed` generated in the circuit (via an LCG `seed.wrapping_mul(2).wrapping_add(1)...`) is exclusively used by `build_battery(&mut seed)` to shuffle the order of the training phases (`align`, `debate`, `games`, `professor`). This prevents the model from memorizing a predictable sequence of training modes across continuous cycles.

---

## 6. Verification done
- `cargo clippy --all-targets` → 0 warnings / 0 errors (P-06).
- `cargo test --lib` → optimizer + sgd tests pass (incl. updated
  `test_sgd_matches_safe`).
- Live: bonsai circuit → health-check OK, baseline forward OK, arena events in
  `circuit.log` (no Dead-RMSNorm, no crash).
- Live: smollm2 circuit → `❌ Modelo colapsado: 5 RMSNorm tensors all-zero…`
  then `Error: model collapsed` (clean exit 1, no panic).
