# Feasibility Study: C-MUD × Log-Gas / CFT concepts (from JHEP page)

**Date:** 2026-07-18
**Context:** User wants small future experiments connecting the complex-thinking manifold
(C-MUD, L-14) with ideas from a JHEP CFT/log-gas paper (vertex operator, Dyson circular
ensemble, contour rotation for convergence). This doc assesses feasibility and maps the JHEP
math onto the existing `src/mud/cmud.rs` kernel.

**Source analysis:** `docs/research/CFT_LOGGAS_VERTEX_OPERATOR_NOTE.md` (JHEP page: contorno
`C/C_R`, Vandermonde/ensamble circular (4.6), hipergeométrica confluente (4.7), valor de
expectación de operador de vértice (4.8)).

---

## 1. Existing C-MUD state (L-14, shipped)

`src/mud/cmud.rs` already implements the algebraic foundation — it is a **math kernel only**,
opt-in via `MUD_CMUD_THINK=1`, does NOT replace the live f32 `SlimeRegister` (P-02 SSOT).

| Piece | Status in cmud.rs |
|-------|-------------------|
| `ComplexF32` (re+im f32) | ✅ done |
| `GaussTernary` (9 states) + `gauss_mul`/`gauss_mac` | ✅ done (adds/subs only) |
| Hermitian mHC projection (`hermite_norm_sq ≤ R²`) | ✅ done (`project_hermitian`) |
| Phase ω EMA + phase-lock escape (`EMA(ω)<ε`) | ✅ done (`ThinkingState`) |
| Wave-function collapse → real (LM-head ready) | ✅ done (`wave_collapse`) |
| Hook after `apply_output_norm` | ✅ done (`maybe_cmud_think`, slime_forward.rs:1394) |
| **Full complex GEMV AVX2** | ❌ NOT done (research doc §5) |
| **Real thinking during forward** (not just post-norm stub) | ❌ stub only |
| **Vulkan complex phase compute** | ❌ NOT done |

10 cmud unit tests pass; `cargo clippy --all-targets` clean.

> CRITICAL CONSTRAINT (P-02): `SlimeRegister.matmul_accum` is f32. C-MUD is an *auxiliary*
> thinking pass, never the production accum. Do NOT dual-pack registers to f16 while P-02 holds.

---

## 2. JHEP ideas → C-MUD mapping

| JHEP concept | C-MUD analog | Feasibility |
|--------------|--------------|-------------|
| `∏_{j<k} |e^{iθ_j}−e^{iθ_k}|²` Vandermonde (log-gas repulsion) | **mHC Hermitian ball** (`hermite_norm_sq ≤ R²`) is the SAME energy: phase repulsion keeps dims apart. `mhc_radius` (√hidden) ≈ Coulomb exclusion radius. | ✅ already present; can add explicit pairwise phase-repulsion term to `think_step_stub` |
| `Im(c)` contour rotation for convergence (`e^{iβc t}`) | **Imaginary-axis thinking energy**: C-MUD seeds `h = x + i·0` then rotates via `i·h` in `think_step_stub`. The "lift to Im>0" that makes the JHEP integral converge is isomorphic to lifting activation into the imaginary (thinking) axis. | ✅ conceptual match |
| `Γ`-series → integral (4.7) resummation | **Wave collapse** `Re·(1+tanh|Im|)·cos θ`: tanh|Im| is the "renormalized" imaginary contribution injected back to real — same role as the resummed integral feeding the observable. | ✅ already present |
| Vertex operator `⟨e^{2iαϕ(0)}⟩` (charge/weight) | **Phase-lock ω ε** = the "conclusion" of the operator; α (charge) ≈ thinking-step gain. | ⚠ partial — α not yet a tunable knob |
| Circular Ensemble (CUE) normalization (4.6) | Could be used to **derive the optimal mHC radius** analytically instead of hand-tuned √hidden. | 🔬 research experiment |
| `σ ∈ [10%,90%]` saturation clamp (trainer) | = contour-rotation stability: keep exponent from diverging. | ✅ already in trainer |

---

## 3. Proposed small experiments (future, low-risk)

All opt-in, none touch the f32 production path (P-02). Run on CPU AVX2; do not enable Vulkan for
CMUD until `complex_jepa_phase.comp` exists.

1. **E1 — Phase-repulsion term (log-gas).** Add to `think_step_stub` a pairwise angular
   repulsion `Δθ_jk = phase_delta(θ_j, θ_k)` pushing dims apart (Coulomb on the circle). Measure
   VarH/VarJ deltas vs baseline. *Reuses `phase_delta`, `ComplexSlimeRegister`.*

2. **E2 — Analytic mHC radius from CUE.** Use (4.6) normalization to set `mhc_radius` from
   `β/2` partition function instead of `√hidden`. Compare cohesion (cog) and VarJ band.

3. **E3 — Contour-rotation convergence probe.** Show that seeding thinking with `Im>0`
   (lift) yields finite `omega_ema` decay vs `Im<0` divergence — directly mirrors JHEP `C_R`.
   Unit-test: `ThinkingState::from_complex(x, im_seed>0)` vs `<0`.

4. **E4 — Wave-collapse gain `α` as charge.** Expose `MUD_CMUD_ALPHA` (the `α` of vertex
   operator) into `wave_collapse` magnitude. Sweep on a tiny corpus; watch loss/perplexity.

5. **E5 — Complex GEMV AVX2 (the real blocker).** Implement `ternary_gemv_complex_avx2.s`
   (`gauss_mul` ×8 lanes, interleaved re/im). This is what makes C-MUD a *forward* path, not a
   post-norm stub. Highest effort, highest payoff.

---

## 4. Feasibility verdict

- **Math kernel:** ✅ ready. The JHEP concepts map cleanly onto existing `cmud.rs` primitives.
- **Low-effort experiments (E1–E4):** ✅ feasible now, pure Rust, opt-in, P-02-safe.
- **Full C-MUD forward (E5):** ⚠ medium effort — needs a complex AVX2 GEMV kernel + dual
  register layout. Blocked by "replace production register" item in research doc §5 (P-02).
- **Vulkan complex:** ❌ not yet (and Vulkan is unstable on this HW — see session report §2).

**Status: FUTURE STUDY — deferred.** This is research, not a scheduled task. Experiments
E1–E5 are explicitly parked until the project reaches a **reliability gate**, i.e.:

- The 25-epoch supervised train completes with 0 crashes and stable VarH/VarJ/cog bands, AND
- The RLVR training circuit (F3+) runs with integrity-honors passing on a sane `.mud`, AND
- mHC stability is observed to need less hand-tuning (or shows instability worth attacking).

Only then do we start with E1+E3 (log-gas repulsion + contour-rotation probe) as a weekend
spike, validating the physics intuition against the live `mud_train_metrics.log` columns
(VarH, VarJ, cog) without touching the trainer hot path. Do NOT begin earlier — the production
path (P-02 f32 `SlimeRegister`) must remain the SSOT and C-MUD stays opt-in only.

**Expected payoff when conditions are met** (see chat 2026-07-18): E2 → analytic mHC radius
from CUE partition function (less hand-tuning, fewer collapses); E1+E3 → thinking-loop
stability + regression-proofing for RLVR convergence detection. E5 (complex AVX2 GEMV) is a
separate long-term effort requiring P-02 rework.

---

*Captured from JHEP analysis (see CFT_LOGGAS_VERTEX_OPERATOR_NOTE.md). C-MUD kernel verified
building + 10 tests passing + clippy clean on 2026-07-18. Deferred to future-study gate
2026-07-18.*
