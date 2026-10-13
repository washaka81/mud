# MUD — Auditoría vigente

> **Este fichero es un puntero.** El contenido de la auditoría vigente vive en los dos documentos siguientes; cualquier resumen aquí sería una segunda copia que se desincroniza.

## Vigente — 2026-10-11

| Documento | Rol |
|---|---|
| [`MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](./MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) | **Auditoría completa** (correctitud, seguridad, eficiencia/roofline, deriva doc↔código, estado del arte arXiv). Evidencia `file:line` + mediciones en vivo. |
| [`../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) | **Plan maestro de correcciones/limpieza/mejoras** (`R-##`, 7 olas, criterios de aceptación). |

**Estado resumido (SSOT de estado: `GEMINI.md` §0.3):**

* ✅ `cargo test --lib --release --features proptest -- --test-threads=2` **PASA** (515 passed / 0 failed / 2 ignored; 2026-10-13).
* ✅ **CI ejecutable** — `on:` raíz (`push`, `pull_request`, `schedule`, `workflow_dispatch`); `./mud.sh ci` release + all-targets + all-features (R-02) + `unsafe-audit` (R-36).
* ✅ **Loader `.mud`** — `validate_model_shapes` (metadata↔shape) + aritmética checked (R-30/R-31); **GGUF** con cotas (R-32); **ECC** valida contra el extent base (R-33); **GEMV** exacto `n%8∈1..7` + strides `div_ceil(8)` (R-35).
* ✅ **R-36** — `./mud.sh unsafe-audit` → `OK - 0 sin comentario` sobre **532 `unsafe` productivos** (gate en CI + GitHub workflow).
* ✅ **R-37/R-38** — RCE feature-gated documentado estructuralmente (+ tests de gate por build); symlink escape cerrado (`symlink_metadata` + `O_NOFOLLOW`, 4 tests).
* ✅ **R-40** — LM head ternario batcheado (`ternary_head_logits`, 4/8 filas + PCorePool, sin materializar f32): greedy **4.6×**, default **2.4×** same-session / **3.5×** vs baseline congelado, RSS **201→127 MB**, logits idénticos (`models/ale.mud`, 2026-10-13).
* 🟠 Formato ternario = **4 bits/peso** (2.52× la entropía de 1.585 bits) — pendiente R-42; RSS restante por `emb_f32` (R-41); sampling nucleus ≈3.8 ms/token (R-44).
* 🟠 Salida de inferencia **incoherente** (calidad, F3/F4) — sin cambios en esta sesión.

**Progreso del plan maestro:** **28/55** · Ola 0 (F0) ✅ · Ola 1 (F1) ✅ · **Ola 2 (F2) ✅ (R-30…R-38)** · Ola 3 (F3) en curso (R-40 ✅, R-48 ✅).

> **Informe de sesión:** [`../sessions/MUD_SESSION_REPORT_2026-10-13_OLA2_OLA3.md`](../sessions/MUD_SESSION_REPORT_2026-10-13_OLA2_OLA3.md).

## Histórico

* [`MUD_AUDIT_FULL_2026-08-03.md`](./MUD_AUDIT_FULL_2026-08-03.md) — bugs BUG-1…BUG-6, riesgos, deriva doc↔código (varios ya corregidos: BUG-1/2/6 verificados corregidos el 2026-10-11).
* [`MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md`](./MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md) — CI roto por clippy (2026-07-24).
* [`MUD_AUDIT_REPORT_V01.md`](./MUD_AUDIT_REPORT_V01.md) … [`MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md`](./MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md) — serie histórica V01–V38.
* [`BUG_LOGITS_CORRUPTION.md`](./BUG_LOGITS_CORRUPTION.md) — corrupción de logits por PRQ por-fila (2026-07-23).

> **Nota (2026-10-11):** la serie histórica contiene afirmaciones de "CERTIFIED" sin evidencia ejecutable (p. ej. V37). El plan maestro introduce la regla **R-73**: toda fila de estado exige `(comando, salida, fecha)`.
