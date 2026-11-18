# Plan de sesión — 2026-11-07 (auditoría + REA/Ghidra + ejecución)

**Rol:** plan de ejecución de esta sesión. El SSOT de backlog sigue siendo
[`MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (`R-##`);
este documento registra el alcance concreto atacado y su evidencia.

---

## 0. Baseline verificado al abrir la sesión

| Gate | Comando | Resultado |
|---|---|---|
| Tests lib release | `cargo test --lib --release -- --test-threads=2` | `475 passed / 0 failed / 0 ignored` |
| Clippy | `cargo clippy --all-targets --all-features -- -D warnings` | 0 warnings / 0 errors |
| Unsafe audit | `./mud.sh unsafe-audit` | 536 `unsafe`, 0 sin justificar |
| Env SSOT | `./mud.sh env-audit` | 0 divergencias |
| CI battery | `./mud.sh ci` | `[L-12] Health battery complete.` |
| Git | `git status --porcelain` | cambios de sesiones previas sin commitear |

---

## 1. Hallazgos de auditoría

| ID | Severidad | Hallazgo | Evidencia |
|---|---|---|---|
| **F-01** | 🟠 | `hash_trained_weights` calculaba el tamaño de un tensor Ternary2Bit mmap-backed como `rows * cols * cols.div_ceil(2) * 4` (hasta GB para `token_embd`), no como `checked_data_size()` (`rows*ceil(cols/8)*4`). Es un OOB read latente cuando el tensor no tiene `owned_data`. | lectura de `checkpoint.rs`; corregido con `checked_data_size()` + test |
| **F-02** | 🟠 | El digest de pesos iteraba `HashMap` (orden aleatorio por proceso) y no incluía `output.weight`, `output_norm.weight` ni normas/mHC. El guard de “checkpoint no-op” podía ser no determinista o ciego a cambios de cabeza/normas. | lectura de `checkpoint.rs`; corregido con snapshot ordenado |
| **F-03** | 🟠 | El ΔW por sync es 0 por diseño (el optimizer empaqueta in-place). El mensaje “no change this sync” se leía como “el entrenamiento no movió nada”. | sync QAT real: primer sync 44.2 %, segundo 0 %; hash final cambió |
| **F-04** | 🟡 | `MUD_ENV_REFERENCE.md` decía que `MUD_GEMV_ROWS` default era `4`; el código usa 8 por defecto y `4` lo desactiva. | `gemv_gpu.rs:16-29` vs doc §1.2 |
| **F-05** | 🟡 | `mud.sh tools` listaba `variance_inspector` (borrado) y el help anunciaba `inspect-var`; faltaban los tools nuevos (`next_token_eval`, `gemv_parity`, `parity_gate`, etc.). | `./mud.sh tools`; `variance_inspector` no está en `Cargo.toml` |
| **F-06** | 🟡 | `env-audit` no formaba parte de `mud.sh ci` ni del workflow de GitHub, así que la deriva documental no bloqueaba nada. | `mud.sh` ci y `.github/workflows/ci.yml` |
| **F-07** | 🟡 | `next_token_eval` y `inference_coherence` inventaban `vocab_size=32000` si faltaba metadata (violación P-13) y podían evaluar con el vocab equivocado. | `unwrap_or(32000)` en ambos tools |
| **F-08** | 🟡 | `inference_coherence` trataba cualquier línea con ≥2 números como IDs (“Hello 42 world 7” → `[42,7]`), contaminando el set de prompts. | `load_prompts` (parseo por `filter_map`) |
| **F-09** | 🟡 | El catálogo de tools de `mud.sh` no tenía verificación automática: era posible volver a tener huérfanos o `--bin` fantasma. | R-61 parcial |
| **F-10** | 🟡 | El ASM `ternary_gemv_8rows_avx2` es funcionalmente correcto pero instruction-bound: ~9 instrucciones SIMD por 8 pesos y fila (160 instrucciones por iteración de 16 columnas × 8 filas). El `adam_step_avx2` mide 0.93× vs escalar en `asm_kernel_bench` con `size=4096`. | Ghidra 12.1.2 + objdump + `asm_kernel_bench` |
| **F-11** | 🔴 | **La suite de tests ASM `src/asm/tests.rs` (1488 LOC, 45 tests) no estaba declarada** (`mod tests;` ausente), así que `cargo test` nunca la ejecutaba. Los “508/511 tests” históricos no incluían ni un solo test de kernel ASM. | `grep -n "mod tests" src/asm/mod.rs` vacío; tras declararla: 43 passed / 0 ignored (los 2 `#[ignore]` eran tests muertos de kernels ya purgados) |
| **F-12** | 🟠 | Divergencia ASM vs escalar para escalas PRQ no finitas: `apply_row_scales_8_avx2` convierte ±Inf en 0, pero el fallback escalar hacía `clamp` primero y multiplicaba por ±1e8. | lectura de `gemv_gpu.rs` + `apply_scales.s`; test nuevo `test_apply_row_scales_8_finite_guard` |

### Verificación ASM con REA/Ghidra (F-10)

- Binario no-stripped: `target/release/forge_llm` con `CARGO_PROFILE_RELEASE_STRIP=false`
  (`sha256 72cda711422d43b1158959f609f191f2c0f10ce1da9f97ab7fc93bb3152801e0`).
- `ternary_gemv_8rows_avx2`: 383 instrucciones, 1803 bytes, 22 bloques, 0 callees.
  - Loop16: 817 bytes / 160 instrucciones para 128 MACs (8 filas × 16 cols).
  - Decodifica 8 nibbles por `u32` con `vpsrlvd` + `vpand` + 2×`vpcmpeqd` y aplica
    `vpand`/`vaddps`/`vsubps` (`+1` = 0x1, `-1` = 0xF, `0` = resto).
  - `vzeroupper` y guard NaN/Inf (exponente todo-1 → 0.0) presentes.
- Call sites en el binario: 2 en `ternary_gemv_rowwise_submit` y 2 en el shim de
  closure del `PCorePool`; el kernel recibe `n_main = n & !7` y el wrapper
  `ternary_gemv_tail` cubre el resto 1..7.
- Runtime (GDB, `next_token_eval models/ale.mud ... --max 1`):
  - llamada `n=384`, `stride_words=48` (= 384/8), `scale=1.0`, `out` válido;
  - llamada mixta `n=128`, pesos `0xf11fff1f 0x11ffff01 ...`, salida con signos
    `+0.509, +9.562, -4.738, ...` (rama +1/−1 ejercitada).
- Paridad independiente: `gemv_parity models/ale.mud blk.0.attn_q.weight`
  → `asm1_vs_ref max_abs=9.54e-7 cosine=1.000000000`, `asm8_vs_ref max_abs=9.54e-7
  cosine=1.000000000`.

**Conclusión:** el motor ASM está numéricamente correcto. La optimización
propuesta (LUT/`vpdpbusd` o fusión de máscaras con `vpsubd`+`vfmadd231ps`) no se
aplica en esta sesión porque el benchmark es ruidoso y no hay una ganancia
end-to-end estable; queda como línea de trabajo medida, no como “hecho”.

---

## 2. Ejecución (P0 → P1)

### P0 — verdad de medición y gates

1. **Snapshot de pesos + ΔW de sesión** (`checkpoint.rs`, `corpus_trainer/mod.rs`,
   `session.rs`, `debate.rs`)
   - `WeightSnapshot` con hash por tensor + hashes de bloques de 1 KiB.
   - `diff_trainable_weights` → `changed_tensors`, `changed_bytes` (granularidad
     1 KiB), `total_bytes`.
   - `session_dw_report_to_log` escribe
     `[DW] session tensors=X/Y bytes=... total=... moved=...`.
   - `hash_trained_weights` = digest ordenado del snapshot; determinista e
     incluye cabeza, `output_norm` y mHC/normas.
   - Test: `test_weight_snapshot_detects_change_and_is_deterministic`.
2. **Fix OOB del hash mmap** (`tensor_bytes` + `checked_data_size`).
   - Test: `test_tensor_bytes_uses_checked_data_size`.
3. **P-13 en tools** (`next_token_eval`, `inference_coherence`): vocab desde
   metadata o shape del embedding/head; error claro si no existe.
4. **Prompt parser de coherencia**: solo líneas 100 % numéricas son IDs.
5. **Doc `MUD_GEMV_ROWS`** corregida al default real (8).
6. **`mud.sh`**: catálogo sin `variance_inspector`; comandos `predict-eval`,
   `gemv-parity`; `tools-audit` (R-61) y `env-audit`/`tools-audit` dentro de
   `mud.sh ci` y del workflow.
7. **Wiring de la suite ASM** (`src/asm/mod.rs`: `#[cfg(test)] mod tests;`):
   43 tests ASM vuelven a ejecutarse; se eliminan 2 tests muertos (`#[ignore]`)
   de kernels purgados (hadamard/sgemm no-transpuesto).
8. **Paridad ASM/escalar de escalas no finitas** (`finite_scale` en
   `gemv_gpu.rs`): NaN/±Inf → 0 en ambos caminos; outliers finitos clampados a
   ±1e8. Test ASM `test_apply_row_scales_8_finite_guard`.

### P1 — cierre de R-61 / R-74 y evidencia

- `./mud.sh tools-audit` → `OK — 0 tools huérfanos, 0 bins fantasma`.
- `./mud.sh env-audit` → `OK — 0 divergencias`.
- QAT corto real sobre copia de `ale.mud`:
  - primer sync `ΔW ternary 8,913,006 / 20,153,728 bytes (44.2251 %)`;
  - `[DW] session tensors=110/119 bytes=11,779,480 total=21,410,416 moved=1`;
  - hash final `0xe5c428a76558bc4e → 0x02d5ceef20631ae7`.
  - Conclusión: el entrenamiento **sí** movió pesos; el ΔW de sesión lo demuestra.

### Fuera de alcance (documentado, no “hecho”)

- Reescritura del kernel ASM 8-row con LUT/VNNI (F-10): requiere benchmark
  estable y A/B end-to-end.
- QAT largo de 0.5B (P1 del plan 2026-11-02): horas de CPU; no es una corrección
  de código.

---

## 3. Criterios de aceptación de la sesión

```text
cargo test --lib --release -- --test-threads=2          # 520/0/0
cargo test --lib --release --features proptest ...      # 527/0/0
cargo clippy --all-targets --all-features -- -D warnings # 0
./mud.sh env-audit                                       # 0
./mud.sh unsafe-audit                                    # 0 sin justificar
./mud.sh tools-audit                                     # 0 huérfanos / 0 fantasma
./mud.sh ci                                              # verde
```

Evidencia final en
[`docs/sessions/MUD_SESSION_REPORT_2026-11-07.md`](../sessions/MUD_SESSION_REPORT_2026-11-07.md).

---

## 4. Addendum — fórmulas y plan de calidad

Tras cerrar los gates se ejecutó el cálculo de fórmulas de cuantización y el
plan de calidad:

- Investigación/cálculos: [`../research/MUD_QUALITY_FORMULAS_2026-11-07.md`](../research/MUD_QUALITY_FORMULAS_2026-11-07.md)
- Plan de ejecución: [`PLAN_QUALITY_2026-11-07.md`](PLAN_QUALITY_2026-11-07.md)
- Tools: `./mud.sh quant-audit` (`quant_formula_audit`), `./mud.sh scale-rescale`
  (`scale_rescale`), `predict-eval --temperature-scan`.
- Hallazgo: escala actual 1.889× menor que el óptimo LSQ; NMSE 0.391→0.218;
  el reescalado post-hoc empeora PPL ⇒ corregir en conversión/QAT.
- Calibración: Qwen \(T^*=1.4\), PPL 17,788→9,055, ECE 0.264→0.046; top-1
  sigue 0/2 (la temperatura no arregla ranking).
