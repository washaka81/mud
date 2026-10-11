# Research: Training Logs & Telemetry Audit — Why Generation Is Broken

**Date:** 2026-07-20
**Trigger:** User observed word-salad generation from
`weights/checkpoints/model_latest_checkpoint.mud` after a "clean" 25-epoch supervised train
(telemetry looked healthy: VarH~4.7, VarJ~0.076, σ~50%, cog~340, 0 crashes).
**Method:** read-only forensic analysis of `mud_train_metrics.log` + inference diff + MD5.

---

## 1. Smoking gun: checkpoint == base (byte-identical)

```
$ md5sum models/smollm2.mud weights/checkpoints/model_latest_checkpoint.mud
ae15bdfe209676edee43991dd3a69fb5  models/smollm2.mud
ae15bdfe209676edee43991dd3a69fb5  weights/checkpoints/model_latest_checkpoint.mud
```
Both files are **181,688,480 bytes** and **same MD5**. The 25-epoch trainer wrote a checkpoint
that is **bit-for-bit identical to the input model**. The training run produced **no persisted
weight change**.

Inference confirms: logits for prompt "hola" are **identical** between base and checkpoint:
```
base:      [1.5147235, -6.8884087, -6.657243, -7.845488, -8.602604]  max=15.34
checkpoint:[1.5147235, -6.8884087, -6.657243, -7.845488, -8.602604]  max=15.34
```
So the "training" was a **no-op on disk** — yet it reported `steps=64`, `loss` varying 3.15–5.8,
`prog=100%`, `[ok] alignment session completed`. The telemetry measured **manifold stability**,
not weight change.

---

## 2. Telemetry reality-check (isolated the real 25-epoch session)

`mud_train_metrics.log` is **concatenated across many prior sessions** (older 15-field format
lines + dozens of `# MUD_TELEMETRY v2` headers with different `lr`/`opt`/`model`). The actual
25-epoch run is the **9566 lines with `lr=0.000500`** (step 1 → 9300, epoch 1 → 25).

Per-epoch last-block metrics (real session):

| epoch | loss | conf% | cog | varh | varj |
|------|------|-------|-----|------|------|
| 1 | 5.11 | 0.60 | 350 | 4.77 | 0.075 |
| 2 | 3.15 | 4.27 | 342 | 4.77 | 0.075 |
| 13 | 5.82 | 0.30 | 354 | 4.74 | 0.084 |
| 20 | 5.83 | 0.29 | 333 | 4.78 | 0.079 |
| 25 | 4.63 | 0.98 | 340 | 4.71 | 0.076 |

**Findings:**
- **Loss does not converge.** It fluctuates 3.15–5.8 with no downward trend; the "best" epoch
  (2, loss 3.15) is followed by worse ones. This is **domain-noise**, not learning — the loss
  depends on which chunk landed at the end of each epoch, not on weight improvement.
- **conf (=(-loss).exp()·100) never exceeds 4.27%** across all 25 epochs. The model never learns
  to concentrate probability mass → near-uniform sampling → word-salad.
- **VarH / VarJ / σ / cog are rock-stable** (exactly the "healthy" band). This is why the panel
  *looked* fine — but it only proves the manifold didn't collapse; it says nothing about whether
  the model learned language.

**Conclusion:** telemetry is honest but **misleading in isolation** — it tracks stability, not
quality. A no-op training run (no weight change) still produces "healthy" VarH/VarJ/cog because
the *starting* model was already manifold-stable.

---

## 3. The model base is ALSO degraded (independent problem)

Even `models/smollm2.mud` (untouched by training) generates word-salad:
- Logits are **always nearly flat**: winner ~1.5, rest < -6.
- The winning token is **consistently index 0** across prompts → **vocabulary collapse toward
  token 0** (unk/pad/bos), a classic signature of a model whose weights don't encode language
  (or are in the wrong scale).
- Forward *does* depend on input (logits change with prompt: 1.53 / 1.51 / 1.50 for "el"/"quien"
  /"123"), so the engine itself runs; the *weights* are the problem.

`training_healthcheck` reports the checkpoint IS `Ternary2Bit ELUT 4-bit + PRQ f32 scales`,
hidden=576, 30 layers — structurally sound. So the degradation is in **weight scale/values**,
not structure. The FIX D converter repair restored norms (mean≈0.0012, faithful to BF16) but the
**ternary weight magnitudes** may still be mis-scaled (logit_scale=1.0 in inference, yet logits
are ~100× too flat for a working 135M model).

---

## 4. Root-cause hypotheses (ranked)

### H1 (most likely for "no weight change"): shadows never reach the saved tensor
`sync_shadow_to_mud` (corpus_trainer.rs:3196) *does* write `owned_data = Some(...)` for emb and
layer tensors (lines 3262/3265/3277). `mud.save` (mod.rs:337) writes `owned_data` if present.
But the **gradient application** (`apply_optimizer_cpu_step_and_pack`, 4603) writes into the
*shadow* buffers, and `train_on_sequence` may be operating on shadow instances that are **not
the same ones** synced at checkpoint time, OR the optimizer step is a no-op because the forward
used a separate weight copy. The identical MD5 means: at save time, every tensor's `owned_data`
(or mmap source) equals the input. → Need to verify shadow identity across
`train_on_sequence` → `apply_optimizer_cpu_step_and_pack` → `sync_shadow_to_mud`.

### H2: `mud.save` serializes the original mmap, not the updated `owned_data`
Ruled **partially out**: mod.rs:337-339 writes `owned` when present, and sync writes
`owned_data`. But if `sync_shadow_to_mud` only updated some tensors and left others on `mmap`,
those would be written unchanged — yet the *whole file* MD5 is identical, implying **no** tensor
changed. So H1 (gradients/shadows not applied at all) is stronger than H2.

### H3: model base already collapsed → gradients ~0
If the base model's weights are mis-scaled (§3), the forward produces near-uniform logits, the
cross-entropy gradient is tiny, and 25 epochs of SGD barely move weights → MD5 stays ~identical
*by convergence to a bad fixed point*. This is consistent with loss fluctuating around 4.5–5.5
without trend. **H3 does not explain an *exactly* identical MD5** unless the save path also
fails (H1/H2), so H1+H3 together.

---

## 5. Recommended corrections (NOT applied — needs confirmation)

1. **Diagnose shadow→mud identity** (H1): add a one-line checksum/log of the FIRST layer weight
   tensor's `owned_data` mean before and after `apply_optimizer_cpu_step_and_pack` in
   `train_on_sequence`, and again inside `sync_shadow_to_mud`. If unchanged → the optimizer step
   isn't reaching the synced buffer. This is the single most important check.
2. **Verify `mud.save` writes updated `owned_data`**: temporarily write a separate
   `model_debug.mud` and diff vs input. Confirms H2.
3. **Fix model-base scale** (H3): re-examine the converter's ternary weight dequant
   (`unpack_ternary2bit_to_f32` + PRQ) against BF16 source magnitudes. If logits are ~100×
   flat, the ternary magnitude or `logit_scale` is wrong. Compare a known-good SmolLM2 logit
   distribution (e.g. via a reference HF load) to isolate.
4. **Telemetry should track weight delta**: add `Σ|ΔW|` per epoch to `mud_train_metrics.log` so
   a no-op training run is immediately visible (currently it is invisible — the panel looks
   healthy while weights don't move).

---

## 6. Status

- The 25-epoch run was **operationally clean** (0 crashes, stable manifold) but **produced no
  model improvement** — checkpoint is byte-identical to input.
- Two distinct defects: (a) training does not persist weight changes (H1/H2), (b) the base model
  is already vocabulary-collapsed (H3).
- **Do NOT launch the RLVR circuit (F3+) on this checkpoint** — it would train on a no-op /
  collapsed base and inherit both defects.
- None of the above touches the P-06 clippy cleanliness (0 warnings) or 222 tests; those remain
  valid. The ECC fix (mod.rs:770) and Iris blacklist (ash_backend.rs) from this session are
  independent and correct.

---

## 7. Resolution (2026-07-20, executed)

### 7.1 P0.1 — ΔW instrumentation (`MUD_TRAIN_DEBUG_DW=1`)
Added a diagnostic in `sync_shadow_to_mud` that counts, per checkpoint, both **ternary packed
bytes changed** and **PRQ scale Σ|Δ|/Σ|s|**. Findings:
- On the **collapsed** base: first sync changed `2280/14,155,776` packed bytes (0.0161%) → MD5
  DID change. So the trainer is **not intrinsically a no-op**; it moves weights when they are far
  from the quantization optimum.
- On a **healthy** base: ΔW ≈ 0 (both ternary codes and PRQ scales) over ~96 steps at the default
  LR — a well-fit ternary model sits in the STE deadzone, so small gradients don't flip codes.
  This is expected, not a bug, **but it means "no visible ΔW" alone is not proof of a bug** — it
  can equally mean "already converged" or "gradients ~0".
- **Root cause of the original 25-epoch no-op:** the base carried `trainer.current_epoch=25` in
  metadata; a fresh `--epochs` run resolved `end_epoch = 25-1+epochs` and, combined with the
  collapsed weights + STE deadzone, persisted no net change.

### 7.2 P1.1 — Scale audit (`scale_audit` bin / `./mud.sh scale-audit`)
New tool compares dequantized `.mud` RMS vs BF16 source RMS per layer. On the collapsed base:
```
blk.0.attn_q   ratio 0.391   (healthy — frozen layer)
blk.15.attn_q  ratio 0.417   (healthy — frozen layer)
blk.29.attn_q  ratio 27.786  (BROKEN — last/thawed layer inflated ~28×)
blk.29.attn_v  ratio 14.393  (BROKEN)
mean 7.301 → VERDICT: SCALE BROKEN
```
The **last (trainable) layers had PRQ scales inflated ~14–28×** → their weights dominate the
residual stream → logits collapse to a single token (token 0). The frozen layers were fine, which
proves the damage came from a **prior training session** (unbounded shadow → large absmean →
large `s = absmean·√½`), not from the converter (FIX D already fixed the converter).

### 7.3 P1.2 — Reconvert clean base (executed)
Reconverted `models/smollm2/` (BF16 source, confirmed healthy) with the FIX-D converter:
```
all layers ratio ≈ 0.31–0.42, mean 0.374 → VERDICT: scale within tolerance
```
Inference sanity: fresh model logits are varied (`[-0.64, -5.4, …]`, max=14.1 at a non-zero
index) vs collapsed (`[1.50, -6.9, …]`, always token 0). Installed:
- `models/smollm2.mud` ← fresh healthy (md5 `67186253…`)
- `models/smollm2.collapsed.bak` ← old collapsed (md5 `ae15bdfe…`), kept for archaeology.

### 7.4 P0.3 / P1 preventive fixes (code)
- **`MUD_TRAIN_RESET_EPOCH=1`** — restart the resume counter at epoch 1 so requested epochs
  always execute (guards the metadata-resume no-op).
- **`epochs=0` warning** — explicit "NO training will run" note.
- **Shadow magnitude clamp** (`MUD_TRAIN_WCLAMP_K`, default 8, 0=off) in
  `apply_optimizer_cpu_step_and_pack`: clamps each shadow element to ±K·row_absmean after the
  optimizer step, preventing the scale-inflation that caused collapse. Frozen layers untouched.

### 7.5 Verification
- `cargo clippy --all-targets` → **0 warnings**; `cargo test --lib` → **222 passed**.
- Fresh base: `./mud.sh health` → 🟢 CERTIFIED; `./mud.sh scale-audit` → within tolerance.

### 7.6 Remaining / next
- **Retrain from the fresh base** with `MUD_TRAIN_RESET_EPOCH=1` and verify (a) checkpoint MD5 ≠
  base and (b) `conf` rises above ~20% at some epoch before enabling the F3+ circuit.
- **P2 panel wiring — DONE (2026-07-20):** `train_telemetry.rs` now parses `[TELEM]`/`[DW]`
   by key (`kv_f64`) and renders a third bottom panel **Weight Δ (bytes moved / sync)**; the
   trainer also emits `[DW]` every sync and writes `[TELEM]` to BOTH stderr and
   `mud_train_metrics.log` (previously stderr-only → TUI read an empty file). Verified via
   `tmux capture-pane`: Loss panel shows descent 5.00→2.86, VarJ/VarH/JEPA/Cognitive + ΔW
   populated.
- P3 gates (`diagnose_model` token-0 dominance; circuit refuses collapsed/no-op base) remain as
   follow-ups; the `scale-audit` command already provides a manual gate.

### 7.7 Follow-up fixes (2026-07-20, executed later same day)

- **Telemetry TUI root-cause + fix (TLM):** the empty-panel bug was NOT the parser — the
  trainer wrote `[TELEM]` only to stderr, so `mud_train_metrics.log` (which the TUI reads)
  had no `[TELEM]` lines. Fixed: trainer writes `[TELEM]` to stderr AND `telemetry_file`;
  `[DW]` emitted every sync; TUI parser rewritten by key (`kv_f64`), added **Weight Δ**
  panel, fixed VarJ/JEPA scales.
- **Pointer-optimized hot loops (P-00/P-01):** `apply_optimizer_cpu_step_and_pack` clamp
  (raw ptr), `dequantize_ternary_row` (`TERNARY_LUT` branchless), `unpack_ternary2bit_to_f32`
  (raw ptr + LUT, 8/u32), `pack_ternary_into` (8/u32 word), `pack_elut_prq` (2/byte).
  clippy 0, 222 tests.
- **Debate writeback hash check:** `run_debate_session` compares `hash_trained_weights`
  in/out, prints ✓/⚠ NO-OP.
- **STE deadzone finding:** default `QAT_LEARNING_RATE=0.0005` + threshold `s*0.7` ⇒ a
  converged base has ΔW≈0 (no-op, expected). High LR (`MUD_QAT_LR`) moves weights.

### 7.8 Inference observation — fused + incoherent (2026-07-20)

`./mud.sh chat` / greedy gen produces word-fused, incoherent text
(`romancesinite restraintStore …`, `2034 15 life is and are …`).

- **Fused words — root cause found:** the `.mud` vocab has **0 `Ġ` (U+0120)
  and 0 `▁` (U+2581)** space-prefix chars, while the **source**
  `models/smollm2/tokenizer.json` has **64,157 `Ġ`** (`Ġthe`, `Ġand`…).
  The converter writer preserves UTF-8, so a **fresh reconvert restores `Ġ`**;
  the on-disk `.mud` was built by an older pass that dropped it. `decode`
  only restores spaces when `space_char` is detected (tokens starting with
  `Ġ`/`▁`) → with 0 such tokens `space_char=None` → no spaces → fusion.
  **Fix:** reconvert from `models/smollm2/` (preserves `Ġ`). See session report §7.1.
- **Incoherent — separate, deeper:** even the fresh base generates semantically
  random tokens; confident-but-wrong logits point to engine/quantization
  correctness on this 135M model, not spacing. **Not fixed** — needs a
  dedicated engine-correctness pass. See session report §7.2.

*Forensic audit 2026-07-20; follow-up fixes (§7.7) executed same day — clippy 0 warnings, 222 tests pass, tmux-verified TUI.*

---

## 8. Circuit (F3+) learning-write audit (2026-07-20)

The user asked whether the training circuit (`run_training_circuit`, F3+) committed the same
**learning-write error** as the alignment trainer. It did — in the debate/games/professor phases.

### 8.1 Defects found in `run_debate_session` writeback
The old manual writeback (corpus_trainer.rs, pre-fix) packed `shadow_w` directly to ternary with
a fixed `v > 0.5` threshold and **never touched `*.prq_scale`**. Two critical bugs:

- **C1 — missing PRQ scale:** it quantized the raw shadow (magnitude in weight-units, not
  normalized to ±1) with `bit = v>0.5? +1 : v<-0.5? -1 : 0`. Any `|w|<0.5` → 0, large `w` → ±1.
  This destroys all weight magnitude (equivalent to s=1 always) → logits collapse. The correct
  path (`sync_shadow_to_mud`) computes `s = absmean·√½` per row and packs `(w/s).round()`.
- **C2 — stale PRQ scales:** `*.prq_scale` tensors were never rewritten, so inflated/old scales
  (e.g. the 27× from §7.2) survived any debate writeback → perpetuates vocabulary collapse.

### 8.2 Fix applied
`run_debate_session` now calls `self.sync_shadow_to_mud(&mut mud, &mut empty_emb, &mut shadow_layers, None)`
which applies the same correct quantization + PRQ-scale refresh as the alignment trainer, then
`mud.save`. Emb is frozen so the empty shadow slice takes the skip branch.

### 8.3 C3 — MUD_DEBATE_LEARN default
`run_debate_session` defaulted `MUD_DEBATE_LEARN=false` (no persist) while the circuit forced it
`true`. Made the code default `true` (consistent) and updated `mud.sh` (`MUD_DEBATE_LEARN:=1`,
help text). Now a direct `./mud.sh debate` also persists learning.

### 8.4 C4 / C5 — checked, not bugs
- C4 (resume no-op in circuit): the circuit runs align with `epochs=1` and the model's
  `trainer.current_epoch` advances each cycle, so align always executes a fresh epoch — no no-op.
  (`MUD_TRAIN_RESET_EPOCH` is still useful for a *fresh* standalone alignment run.)
- C5 (`deep_local_alignment` at debate start): mutates `mud` in memory only, does not persist; the
  writeback correctly overwrites with the trained shadows. OK.

### 8.5 Verification
- `cargo clippy --all-targets` → 0 warnings; `cargo test --lib` → 222 passed.
- Retrain of the healthy base is running (see §7.6); the debate writeback fix will be exercised on
  the next circuit run.

**Conclusion:** the circuit *did* have the learning-write error (C1/C2) in its debate path — the
exact class of bug that produced the collapsed checkpoint. Now both the alignment and debate
writebacks share the same correct, scale-aware quantization.
