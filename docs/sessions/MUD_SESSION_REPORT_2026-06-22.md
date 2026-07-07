# Session Report — 2026-06-22 (JEPA Collapse & Residual Scaling)

## Diagnosticado

### Bug 1: JEPA Collapse (VarH→0)
- `jepa_stabilizer` mezclaba `y_parcial` (salida del bloque en f32, típicamente cientos/miles) directamente en el EMA `z_next = 0.9*z + 0.1*y` sin normalizar
- Tras 30 capas × 2 JEPA, `v_jepa` crecía sin cota (E_JEPA ≈ 1.07e9), `sigmoid(v_jepa)→1` permanentemente
- Gate roto → residuales se descargaban sin control → todas las dimensiones colapsaban al mismo valor (VarH≈0)
- **Fix:** Normalizar `y_parcial` por su RMS antes del EMA (`slime_jepa.rs:150`). JEPA ahora en rango ~[0, 10] con gradiente útil.

### Bug 2: mu_ctx Tracking (Math Bug)
- `mu_ctx` rastreaba `mean(y_final)` pero se usaba para centrar `v_jepa` en `sum_z_delta_sq`
- Corrección: `mu_ctx` ahora rastrea `mean(v_jepa)`
- Test `test_jepa_convergence_equilibrium` actualizado (mu=1.0/32)

### Bug 3: Residual Overflow → i16 Saturation (Sat=100%, Mode=32767)
- 30 residuales sin escalar acumulaban valores fuera de rango i16 en tokens semánticos
- **Fix v1:** Escalar output de cada bloque (atención y FFN) por `1/num_layers` antes del residual. No fue suficiente — tokens especiales (`_START`, `_SP`) con embeddings altos dejaban ~0 headroom.
- **Fix v2 (iscale):** safe_ceiling 128→256 duplica iscale, dando headroom para acumulación de block outputs (~200 f32 + embedding). `ws.iscale` sincronizado entre embedding y forward pass.
- **Fix v3 (clipping adaptativo):** Por dimensión, cada capa consume como máximo `headroom / (num_layers - layer_idx)` i16. Así se garantiza headroom hasta la última capa.
- Forward: `slime_forward.rs:263-272`, `372-382`
- Backward (branch gradients): `slime_backward.rs:245`, `375`
- Añadido `num_layers: usize` a `SlimeWorkspace` y propagado a 8 callers

## Tests
- 86 tests pasan
- `cargo clippy --all-targets`: 0 warnings, 0 errors

## Archivos Modificados
- `src/mud/slime.rs` — `num_layers` field + constructor param, default iscale 256/32767
- `src/mud/slime_forward.rs` — Residual scaling en attn y FFN
- `src/mud/slime_backward.rs` — Branch gradient scaling + res_scale fuera del loop
- `src/mud/slime_jepa.rs` — RMS normalization en jepa_stabilizer
- `src/main.rs` — safe_ceiling 256, ws.iscale sincronizado, diagnostics usan ws.iscale
- `src/mud/self_play.rs` — safe_ceiling 256
- `src/mud/corpus_trainer.rs` — safe_ceiling *= 4 con floor 256
- `tools/run_trainer.rs` — num_layers desde layers.len()
- `tools/hub_api.rs` — num_layers=30 (default)
- `tools/slime_backward_bench.rs` — num_layers=30
- `src/mud/tests.rs`, `src/mud/slime_backward.rs` — tests actualizados

## P-13 Hardcoding Audit
Auditoría sistemática de violaciones P-13 (dimensiones hardcodeadas). Ver tabla en AGENTS.md.
Violaciones críticas en slime.rs (256→2*num_layers), workspace.rs, qat_dispatcher.rs, corpus_trainer.rs.

## Pendiente
- Refactor P-13: centralizar constantes en constants.rs
- safe_ceiling todavía hardcodeado a 256 (debería computarse de max(|emb|) × safety_margin)
- Eliminar fallbacks de dimensión (error si falta metadata)
- EPSILON_FLOOR triplicado (slime_jepa.rs, constants.rs, workspace.rs)
- VicReg lambdas, LR, clip, neg samples hardcodeados

