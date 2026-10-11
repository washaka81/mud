# MUD Audit Report V32 — Synthetic Self-Play & Dead Code Purge

**Date:** 2026-06-20
**Module:** `src/mud/corpus_trainer.rs`, `mud.sh`, Codebase Core
**Focus:** Autoentrenamiento Sintético (Prioridad 38), Error de Colapso Numérico y Limpieza de Dead Code (Prioridad 32).

## 1. Contexto & Objetivos
El objetivo principal de la **Fase 9** era activar la capacidad del modelo de autoalinearse sin depender de corpus externos. Se implementó un bucle autoregresivo para generar cadenas sintéticas de alta confianza, las cuales actúan como retroalimentación (feedback loop) para estabilizar los pesos ternarios mediante STE QAT y JEPA.
Paralelamente, se debía garantizar la directiva **P-32** (Dead Code Purge), eliminando las arquitecturas antiguas (`MudInference`, inferencia y *forward* en FP32).

## 2. Ejecución de la Prioridad 38: Auto-Play
Se reconoció que la base estructural para generar memoria al vuelo (`train_synthetic_self_play`) ya existía de forma latente en el ecosistema, pero no estaba conectada a la terminal de control, ni expuesta en las flags de ejecución.

**Acciones tomadas:**
1. Orquestación del flag `--synthetic` en el ejecutable optimizado `tools/run_trainer.rs`.
2. Habilitación del comando nativo interactivo `./mud.sh self-play`.
3. Verificación de compilación integral mediante `cargo clippy --all-targets`.

## 3. Análisis de la Anomalía Numérica: "LAYER PRODUCED ALL ZEROS"
Al realizar la primera prueba real de generación sintética (batch 4, epochs 10), el motor reportó de inmediato un apagón total:
> `LAYER PRODUCED ALL ZEROS FOR POS X`

**Diagnóstico (Causa Raíz):**
El error no provino de una asimetría matricial, sino de un colapso en la cuantización de incrustaciones (*embeddings*) iniciales.
Durante la inserción del token de semilla en el `SlimeWorkspace`, el valor del vector `f32` se divide entre `iscale` antes de convertirse en `i16`:
```rust
ws.registers[h].matmul_accum = (emb_val / iscale).clamp(-32768.0, 32767.0) as i16;
```
En el autotrainer (a diferencia de la inferencia activa en `src/main.rs`), la variable `iscale` caía por defecto en un valor bruto de `32767.0`. Por tanto, una incrustación promedio de `0.05` se dividía entre `32767.0` resultando en `0.0000015`. Al truncarlo a entero 16-bits, se perdía toda magnitud y los registros del `SlimeWorkspace` iniciaban literalmente vacíos. Un estado neuronal completamente apagado (0 * Weight = 0), capa tras capa.

**Solución aplicada (Math Rescue):**
Se homologó la constante de escala dinámica utilizada en `src/main.rs` garantizando un `safe_ceiling` de 128.0.
```rust
let safe_ceiling = 128.0;
let iscale = mud.global_metadata.get("iscale").and_then(|v| v.parse().ok()).unwrap_or(safe_ceiling / 32767.0);
```
El truncado natural ahora conserva magnitudes seguras en rango de ±12 y la propagación matricial de la generación fluye correctamente.

## 4. Validación de Integridad Estructural (Prioridad 32 y P-06)
- Se constató que las rutinas obsoletas `src/mud/inference.rs`, `src/mud/forward.rs` y `src/mud/jepa.rs` ya habían sido purgadas satisfactoriamente.
- Al correr los chequeos del compilador, se detectó una advertencia de *dead code* en `tools/iteration_validator.rs` por una macro estática mal formulada (`if test_prompts.is_empty()`).
- Se reescribió y optimizó la lógica, devolviendo al compilador el ansiado resultado **0 errores, 0 warnings**.

## 5. Conclusiones y Estado del Framework MUD
1. El modelo ahora soporta un bucle cerrado de retroalimentación donde es su propio *Teacher* y *Student*.
2. El entorno del `SlimeRegister` ha demostrado ser una estructura extremadamente resistente, pero requiere calibraciones matemáticas milimétricas desde la inyección inicial.
3. Se considera completada exitosamente la **Fase 9**.

El modelo se encuentra listo para iniciar corridas extensivas de auto-alineación mediante `./mud.sh self-play`.
