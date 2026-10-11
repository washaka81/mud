# MUD — Auditoría vigente

> **Este fichero es un puntero.** El contenido de la auditoría vigente vive en los dos documentos siguientes; cualquier resumen aquí sería una segunda copia que se desincroniza.

## Vigente — 2026-10-11

| Documento | Rol |
|---|---|
| [`MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](./MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) | **Auditoría completa** (correctitud, seguridad, eficiencia/roofline, deriva doc↔código, estado del arte arXiv). Evidencia `file:line` + mediciones en vivo. |
| [`../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) | **Plan maestro de correcciones/limpieza/mejoras** (`R-##`, 7 olas, criterios de aceptación). |

**Estado resumido (SSOT de estado: `GEMINI.md` §0.3):**

* 🔴 `cargo test --lib --release` **FALLA** (UB en `fenwick.rs:307-312`; en debug pasa ⇒ `./mud.sh ci` lo oculta).
* 🔴 El **CI no puede ejecutarse** (`on:` anidado dentro de un job, `.github/workflows/ci.yml:76-88`).
* 🔴 El **trainer no entrena la cabeza evaluada** (`token_embd` vs `output.weight`, nunca sincronizada).
* 🔴 **Backward ≠ forward** en el FFN (SiLU vs derivada ReLU²) y el mHC omite `alpha`/`(1-gate)·beta`.
* 🟠 Formato ternario = **4 bits/peso** (2.52× la entropía de 1.585 bits); **LM head en f32 denso** (77.6 MB/token en `ale.mud`), single-thread.
* 🟠 Inferencia medida: **~46 tok/s** y salida incoherente; sin validación `metadata↔shape` (OOB/OOM); parser GGUF sin cotas.

## Histórico

* [`MUD_AUDIT_FULL_2026-08-03.md`](./MUD_AUDIT_FULL_2026-08-03.md) — bugs BUG-1…BUG-6, riesgos, deriva doc↔código (varios ya corregidos: BUG-1/2/6 verificados corregidos el 2026-10-11).
* [`MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md`](./MUD_AUDIT_REPORT_V38_POST_SPRINT_5.md) — CI roto por clippy (2026-07-24).
* [`MUD_AUDIT_REPORT_V01.md`](./MUD_AUDIT_REPORT_V01.md) … [`MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md`](./MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md) — serie histórica V01–V38.
* [`BUG_LOGITS_CORRUPTION.md`](./BUG_LOGITS_CORRUPTION.md) — corrupción de logits por PRQ por-fila (2026-07-23).

> **Nota (2026-10-11):** la serie histórica contiene afirmaciones de "CERTIFIED" sin evidencia ejecutable (p. ej. V37). El plan maestro introduce la regla **R-73**: toda fila de estado exige `(comando, salida, fecha)`.
