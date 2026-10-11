# Resumen de Sesiones — Forge LLM (MUD)

## Sesión 2026-07-16 — Docs + ASM + compute stack + tooling

**Detalle:** [`MUD_SESSION_REPORT_2026-07-16.md`](./MUD_SESSION_REPORT_2026-07-16.md)

### Cerrado
- Jerarquía canónica GEMINI / AGENTS / VISION / PLAN  
- ASM polish + `lm_head_logits_avx2` en `main.rs`; SiLU/dots en forward  
- Auditoría ELUT/FP32 + PCorePool×8 + ash (GEMV ash no live)  
- Tools con LIVE vs PLANNED (`run_trainer`, `training_healthcheck`, `avx_math_validator`, banners)  
- Docs: `MUD_COMPUTE_STACK.md`, STATUS_REPORT, VISION handoff  

### Mañana (roadmap Fase A)
1. **L-01** — wire optimizers al step  
2. L-02 Muon NS · L-03 purge workspace · L-07 P-13  

No empezar MoE/C-MUD.

---

## Sesión 2026-06-22 — JEPA Collapse & Residual Scaling

### Bugs Corregidos

1. **JEPA Collapse (VarH→0):** `jepa_stabilizer` alimentaba `y_parcial` sin normalizar al EMA → `v_jepa` crecía sin cota (E_JEPA≈1e9), `sigmoid→1`, gate roto → VarH=0. Fix: RMS norm antes del EMA.

2. **mu_ctx tracking:** Rastreaba `mean(y_final)` en vez de `mean(v_jepa)`. Corregido.

3. **Residual Overflow → i16 Saturación (Sat=100%):** 30 residuales sin escalar saturaban i16. **Fix v1:** escalar cada output por `1/num_layers`. **Fix v2:** safe_ceiling 128→256 duplica iscale. **Fix v3 (raíz):** tokens especiales (`_START`) con embeddings grandes en ciertas dimensiones agotaban todo el headroom en capas tempranas. Fix: clipping adaptativo por dimensión limita cada capa a `headroom / (num_layers - layer_idx)` i16.

### P-13 Hardcoding Audit
Auditoría sistemática de violaciones de anti-hardcoding. 10+ violaciones detectadas (ver AGENTS.md tabla completa).

### Tests
86 tests, clippy 0 warnings.

---

## Sesión 2026-06-20 — Gradient Sanitization & TUI Regression Graph

### Bugs Corregidos
1. **Deep Network Gradient Explosion:** Backprop a través de 30 capas Ternary2Bit sin alinear → gradientes explotaban a ~100,000, corrompiendo shadow_emb.
2. **Fix:** P-14 Gradient Sanitization (is_finite + L2 clamp ≤ 1.0) en `train_on_sequence_jepa`.
3. **TUI Graph:** TelemetryGraph rediseñado con scatter plot 2D ASCII + regresión de mínimos cuadrados.

---

## Sesión 2026-06-18 — Self-Play Phase Singularity

*(Contenido del reporte original)*

---

## Sesión 2026-06-10 — Phase 2: SlimeRegister Forward Pass

### Logros
1. **evaluate_slime_block:** Forward pass completo con RMSNorm(i8 Q) → QKV GEMV → Atención GQA + RoPE → O Proj → JEPA → Residual → FFN RMSNorm → Up/Gate GEMV → SiLU → Down GEMV → JEPA → Residual.
2. **AVX2 Kernels probados:** ternary_gemv_i8act, dot_product, rms_norm_scale, sum_squares, ternary_gemm_batch4.
3. **30 capas BitNet** cargadas con pesos reales (768M parámetros ternarios).

---

## Sesión 2026-06-05 — Inferencia Coherente

### Bugs Corregidos
1. **U8 centering incorrecto:** `(v-85)/85` sobre valores ya desempaquetados como {-1,0,+1}.
2. **quant_scale mal:** Usaba `ext_scale≈1.6-2.8` como denominador → colapsaba a 0.
3. **head_dim mal detectado:** No consideraba `q_out/num_heads`.
4. **Primera inferencia coherente** con tokens reales del vocabulario.

### Tests
50 tests, modelo convertido: `models/bitnet.mud` (3.0 GB), ~1.5 tok/s.
