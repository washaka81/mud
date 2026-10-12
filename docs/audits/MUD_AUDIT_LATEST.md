# MUD — Auditoría vigente

> **Este fichero es un puntero.** El contenido de la auditoría vigente vive en los dos documentos siguientes; cualquier resumen aquí sería una segunda copia que se desincroniza.

## Vigente — 2026-10-11

| Documento | Rol |
|---|---|
| [`MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](./MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) | **Auditoría completa** (correctitud, seguridad, eficiencia/roofline, deriva doc↔código, estado del arte arXiv). Evidencia `file:line` + mediciones en vivo. |
| [`../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) | **Plan maestro de correcciones/limpieza/mejoras** (`R-##`, 7 olas, criterios de aceptación). |

**Estado resumido (SSOT de estado: `GEMINI.md` §0.3):**

* ✅ `cargo test --lib --release -- --test-threads=2` **PASA** (468 passed / 0 failed / 2 ignored) — UB fenwick purgado (R-03); flaky `grad_checkpoint` env-race arreglado vía `CheckpointPolicy::from_env`.
* ✅ **CI ejecutable** — `on:` movido a raíz (`push`, `pull_request`, `schedule`, `workflow_dispatch`); `actionlint .github/workflows/ci.yml` exit 0 (R-01); `./mud.sh ci` release + all-targets + all-features (R-02).
* ✅ **R-10**: el trainer SÍ entrena la cabeza evaluada — `sync_shadow_to_mud` espeja `token_embd.weight`→`output.weight` (+ `output.prq_scale`) vía `sync_tied_output_weight`. `models/ale.mud` confirma head separado.
* ✅ **R-05**: env vars SSOT (`MUD_ENV_REFERENCE.md`, 117 `R` + 7 shell + 6 retiradas); `./mud.sh env-audit` → `OK — 0 divergencias`.
* ✅ **R-11**: FFN backward SiLU — `slime_backward.rs` reemplazó ReLU² por `silu(gate)·up` vía `ezop::silu_backward_avx2` (AVX2 + escalar). `test_silu_backward_matches_reference` ✅.
* ✅ **R-12**: mHC activation backward — `grad_out *= alpha`, `grad_branch *= (1-gate)·beta` vía `ezop::mhc_activation_backward` (AVX2 + escalar). Ambas rutas (ternaria + F32). `test_mhc_activation_backward_matches_reference` ✅.
* ✅ **R-13**: deadzone SSOT `TERNARY_DEADZONE=0.7` (conversor↔trainer↔checkpoint).
* ✅ **R-14**: PRQ scale punto fijo — `stable_pack_ternary_rows` preserva `s_old` en round-trip (bit-stable) y recomputa sólo con `ΔW` real; deriva ×0.52/sesión eliminada. Tests `test_prq_scale_stable_roundtrip` (5 ciclos ±1e-6) + `test_prq_scale_recomputes_on_movement`.
* 🟠 Formato ternario = **4 bits/peso** (2.52× la entropía de 1.585 bits); **LM head en f32 denso** (77.6 MB/token en `ale.mud`), single-thread (R-42/R-40).
* 🟠 Inferencia medida: **~46 tok/s** y salida incoherente; sin validación `metadata↔shape` (OOB/OOM, R-30/R-31); parser GGUF sin cotas (R-32).

**Progreso del plan maestro:** 11/55 (R-01, R-02, R-03, R-04, R-05, R-06, R-10, R-11, R-12, R-13, R-14) · Ola 0 (F0) cerrada; Ola 1 (Correctitud) R-10…R-14 ✅, R-15…R-20 pendiente.

## Histórico

* [`MUD_AUDIT_FULL_2026-08-03.md`](./MUD_AUDIT_FULL_2026-08-03.md) — bugs BUG-1…BUG-6, riesgos, deriva doc↔código (varios ya corregidos: BUG-1/2/6 verificados corregidos el 2026-10-11).
* [`MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md`](./MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md) — CI roto por clippy (2026-07-24).
* [`MUD_AUDIT_REPORT_V01.md`](./MUD_AUDIT_REPORT_V01.md) … [`MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md`](./MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md) — serie histórica V01–V38.
* [`BUG_LOGITS_CORRUPTION.md`](./BUG_LOGITS_CORRUPTION.md) — corrupción de logits por PRQ por-fila (2026-07-23).

> **Nota (2026-10-11):** la serie histórica contiene afirmaciones de "CERTIFIED" sin evidencia ejecutable (p. ej. V37). El plan maestro introduce la regla **R-73**: toda fila de estado exige `(comando, salida, fecha)`.
