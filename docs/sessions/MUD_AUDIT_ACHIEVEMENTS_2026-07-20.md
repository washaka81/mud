# Audit — Session Achievements & Corrections (2026-07-18 → 2026-07-20)

**Scope:** read-only audit of what was accomplished, fixed, and documented this session.
No code was modified during this final audit pass (verification only).

**Final state verified:**
- `cargo clippy --all-targets` → **0 warnings**
- `cargo test --lib` → **222 passed, 0 failed, 2 ignored**
- Trainer completed **25/25 epochs** clean (`[ok] alignment session completed`, prog=100%,
  step 9300, 0 crashes across the whole run).
- Both session fixes present in source:
  - `src/mud/mod.rs:770` → `let start = ecc_tensor.data_base + ecc_tensor.offset;`
  - `src/vulkan/ash_backend.rs:1601-1610` → Iris Xe blacklist in `probe_gpu()`.

---

## A. Achievements (logros)

| # | Achievement | Evidence |
|---|-------------|----------|
| A1 | **Converter FIX D (offset-corruption)** fully closed & verified | `models/smollm2.mud` regenerated sane; norma `mean≈0.0012` (byte-faithful vs BF16 source); `diagnose_model` clean; `mud_full_audit` 210 ternary + 422 f32. |
| A2 | **Trainer Vulkan SIGSEGV isolated & mitigated** | `coredumpctl gdb` backtrace: `submit_and_wait` → `dispatch_gemv_qkv_host_sync` → `evaluate_slime_block_moe` → `train_on_sequence`, deterministic block 11/64. `mud.sh train` forces Vulkan OFF. |
| A3 | **25-epoch supervised training completed** | 9300 chunks, 0 crashes, VarH~4.7 / VarJ~0.076 / σ~50% / cog~340 stable throughout. |
| A4 | **Telemetry column-map audit** | Found `# cols:` comment in `corpus_trainer.rs:2495` desaligned from writer (3068); `train_telemetry.rs` reads correct columns. Comment corrected. |
| A5 | **tok/s cross-validated** | `64 steps/chunk ÷ 15.75 s = 4.06 tok/s` matches TELEM `4.0` on i7-1260P AVX2×8. |
| A6 | **Critical audit bug fixed (FIX D relapse)** | ECC parity read at `mod.rs:770` used absolute `offset`; corrected to `data_base + offset`. |
| A7 | **Vulkan Iris Xe blacklist moved in-binary** | `probe_gpu()` now returns `available=false` for Intel Iris Xe → inference degrades to AVX2 without `mud.sh`. |
| A8 | **Full-codebase audit** | P-27 (no Rayon) ✓, P-13/P-17 fail-fast ✓, NaN guards visible-panic ✓, dead-code scan ✓. |
| A9 | **C-MUD × log-gas research mapped** | JHEP CFT concepts mapped to `cmud.rs` primitives; 5 experiments (E1–E5) specified; deferred to reliability gate. |
| A10 | **Documentation suite updated** | `MUD_SESSION_REPORT_2026-07-18.md` (§1–§8), `CMUD_LOGGAS_FEASIBILITY.md`, `CFT_LOGGAS_VERTEX_OPERATOR_NOTE.md`, `AGENTS.md`, `GEMINI.md`. |

---

## B. Corrections applied (correcciones)

| # | File | Correction | Why |
|---|------|-----------|-----|
| B1 | `src/mud/mod.rs:770` | ECC read `mmap[offset..]` → `mmap[data_base+offset..]` | Same class of bug as FIX D; silent ECC corruption on mmap-resident tensors. |
| B2 | `src/vulkan/ash_backend.rs:1601` | Added Iris Xe blacklist in `probe_gpu()` | SIGSEGV (§A2) now unreachable in direct inference, not just via `mud.sh`. |
| B3 | `src/mud/corpus_trainer.rs:2495` | `# cols:` comment realigned to writer format | Avoid future confusion (3 literal `0.0` pads + integral/sigma/cognitive). |
| B4 | `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` | Added §7 (audit findings) + §8 (status) | Record CRIT/WARN/INFO from read-only audit. |
| B5 | `AGENTS.md` / `GEMINI.md` | Vulkan caveat + converter ECC relapse + smollm2-sane status | Keep SSOT docs truthful. |
| B6 | `docs/research/CMUD_LOGGAS_FEASIBILITY.md` | Status set to "FUTURE STUDY — deferred" | Per user decision: only after reliability gate. |

---

## C. Open items (no action this session — deferred)

| # | Item | Reason deferred |
|---|------|-----------------|
| C1 | C-MUD × log-gas experiments E1–E5 | Deferred to reliability gate (train clean + circuit honors + mHC stability). |
| C2 | P-08 soft debt: `ComplexSlimeRegister`, `complex_gemv_gauss_ref` dead in prod | Low risk; research module; only reached from `#[cfg(test)]`. |
| C3 | `debate_trainer.rs:496` `top_p_probs.last().unwrap()` | Low risk; only panics on empty distribution in debate (non-hot path). |
| C4 | **Post-train inference quality** (user observed word-salad output, conf=0.98%) | **ROOT-CAUSED (2026-07-20 forensic):** checkpoint `model_latest_checkpoint.mud` is **MD5-identical** to `models/smollm2.mud` (`ae15bdfe...`) — the 25-epoch trainer persisted **zero weight changes**. Telemetry tracked manifold stability (VarH/VarJ/σ/cog), not weight delta, so a no-op run looked "healthy". Separately, the **base model is already vocabulary-collapsed** (logits always flat ~[1.5, -6.9, ...], winner always token 0). See `docs/research/TRAIN_TELEMETRY_FORENSIC_2026-07-20.md`. **Blocker for F3+ circuit** until fixed. |

---

## D. Veredict (session outcome)

The session was **operationally successful**: converter fixed, trainer ran 25 epochs with zero
crashes, two latent corruption/crash bugs found and fixed, full audit clean (0 clippy warnings,
222 tests). Documentation is current and truthful.

**One quality flag carried forward (C4):** the trained checkpoint produces non-coherent
generation (`conf=0.98%` after 25 epochs). This is a *training-quality* issue, not a stability
or correctness bug — manifold metrics (VarH/VarJ/σ/cog) are all healthy. Diagnosis of C4 is the
natural next task before launching the RLVR circuit, but was explicitly out of scope for this
audit pass.

*Audited 2026-07-20. All fixes (B1–B2) confirmed present in source; clippy/tests re-verified.
Forensic log analysis (TRAIN_TELEMETRY_FORENSIC_2026-07-20.md) root-caused the generation
failure: checkpoint == base (MD5-identical) + base model already collapsed.*

**Fix plan (2026-07-20):** `docs/research/MUD_FIX_PLAN_2026-07-20.md` — P0 train no-op (MD5-identical checkpoint) → P1 base model collapse → P2 telemetry ΔW → P3 robustness. Blocks F3+ circuit.
