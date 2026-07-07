# MUD SESSION REPORT — 2026-06-18
## Warp-Aligner: Diagnóstico de Cuelgue + Optimizaciones ASM/Vulkan

---

## Resumen Ejecutivo

Se identificaron y corrigieron **4 bugs críticos y 1 mejora de persistencia** en el pipeline `warp-aligner`. El codebase mantiene **0 errores / 0 warnings** post-clippy.

---

## Análisis del Cuelgue — Root Causes

### Bug #1: Adam Loop con `powi()` Redundante → O(2N) cálculos innecesarios

**Archivo:** `src/mud/corpus_trainer.rs` L592–638  
**Descripción:** Las correcciones de bias `(1 - β₁ᵗ)` y `(1 - β₂ᵗ)` se recalculaban con `b1.powi(t as i32)` **dentro del loop por elemento**, causando O(n × 2) llamadas a `powi()` donde n puede ser millones de floats para capas grandes.

**Fix:** Pre-calcular `bc1` y `bc2` **una sola vez** antes del loop. Combinado con el kernel AVX2 que vectoriza 8 floats por iteración via registros YMM + FMA.

**Speedup estimado:** 4–8× en el loop Adam.

---

### Bug #2: Búsqueda O(N·M) en tape — `iter().position()` por posición

**Archivo:** `src/mud/corpus_trainer.rs` L2737–2750  
**Descripción:** Para cada posición (0..8 en `QAT_SEQ_LEN`) se hacía una búsqueda **lineal** `tape.nodes.iter().position(|n| { ... })` para encontrar el nodo embedding correspondiente. El tape crece con cada capa (~28 capas × ~50 nodos + QAT_SEQ_LEN × embeddings) resultando en búsquedas de ~1,500 nodos por posición × 8 posiciones = **12,000 comparaciones por paso**.

**Adicionalmente había un bug lógico:** la búsqueda siempre devolvía el **primer** nodo que coincide (target de la posición 0), no el de la posición actual — todos los updates de posiciones 1..7 usaban los gradientes de la posición 0.

**Fix:** Capturar el `NodeId` en el momento de creación (`all_emb_nodes.push(emb_node)`) y hacer lookup O(1) directo.

---

### Bug #3: Persistencia Incompleta — `qat.sync_to_mud()` faltante

**Archivo:** `tools/warp_aligner.rs` L259–266  
**Descripción:** Al guardar, `sync_shadow_to_mud()` solo sincronizaba los **embeddings** al formato `.mud`. Los pesos QAT de las capas de atención y FFN (que viven en `trainer.qat.master_weights`) **nunca se escribían al archivo** — cada reinicio del warp-aligner empezaba desde cero.

**Fix:** Llamar `qat.sync_to_mud(&mut mud)` antes de `mud.save()` tanto en el save final como en los checkpoints periódicos.

---

### Bug #4: Sin Checkpoints Periódicos

**Archivo:** `tools/warp_aligner.rs`  
**Descripción:** Si el proceso se colgaba o era matado externamente (OOM, kernel panic), se perdía **todo el progreso** desde el último inicio.

**Fix:** Checkpoint periódico cada `CHECKPOINT_EVERY_STEPS = 500` pasos: sincroniza QAT + embeddings y guarda el `.mud` sin interrumpir el raw mode del terminal.

---

## Nuevos Archivos Creados

### `src/asm/adam_step.s` — Kernel Adam AVX2
- Vectoriza el update EMA y SGD-Adam en 8 floats por iteración usando registros YMM
- Usa `vfmadd231ps` para m/v EMA en una instrucción FMA (m = b1*m + (1-b1)*g)
- `vsqrtps` + `vdivps` para el paso final
- Fallback escalar en `asm/mod.rs` para hardware sin AVX2
- Función pública `crate::asm::adam_step()` con dispatch ISA automático

### `assets/shaders/ternary_backward_opt.comp` — Shader Backward Optimizado
- Shared memory tiling 32×8 (1KB en L1)
- Carga coalesced de X y grad_y en tiles de `shared float s_x[TILE_Y][TILE_X]`
- Elimina los cache misses del shader original por strides variables
- Compatible con el layout actual de `VulkanContext::run_qat_backward_async`

---

## Cambios en Archivos Existentes

| Archivo | Cambio |
|---|---|
| `src/mud/corpus_trainer.rs` | `adam_update()`: pre-calc bc1/bc2, usa `crate::asm::adam_step()` |
| `src/mud/corpus_trainer.rs` | `train_on_sequence_qat()`: tracking O(1) de NodeId, elimina `iter().position()` |
| `src/asm/mod.rs` | Agrega extern `adam_step_avx2` y wrapper público `adam_step()` |
| `src/vulkan/mod.rs` | `backward_cs` apunta a `ternary_backward_opt.comp` |
| `tools/warp_aligner.rs` | Checkpoint periódico cada 500 pasos + `qat.sync_to_mud()` en save |
| `build.rs` | Registra `adam_step.s` en `cc::Build` |

---

## Pipeline de Asimilación .mud (Flujo Correcto Post-Fix)

```
warp_aligner.rs
   ↓
run_single_qat_step(mud, tokens)
   ↓ forward (CPU: embedding lookup + QAT attn + FFN)
   ↓ backward (Vulkan: ternary_backward_opt.comp → shared memory tiling)
   ↓ optimizer (Vulkan: shadow_optimizer.comp PRQ)
   ↓ adam_update (CPU: adam_step_avx2 → 8 floats/iter AVX2+FMA)
   ↓
[cada 500 pasos]
   ├── qat.sync_to_mud()    ← NUEVO: serializa todas las capas QAT
   ├── sync_shadow_to_mud() ← embeddings
   └── mud.save()           ← escribe a disco
   ↓
[final / Ctrl+C]
   ├── qat.sync_to_mud()    ← NUEVO: idem
   ├── sync_shadow_to_mud()
   └── mud.save()
```

---

## Estado Post-Optimización

- `cargo check --features="tools"`: ✅ OK
- `cargo clippy --features="tools"`: ✅ 0 warnings, 0 errors
- Política 0-Error/0-Warning del GEMINI.md: ✅ MANTENIDA

---

## Interactive Inference Live Hook (Session 11)

- **Causal LM Head Projection**: Connected `output.weight` tensor from the `core` skill dynamically inside the `InteractiveChat` dashboard logic.
- **Dynamic Decoding**: Decodes mathematical predictions against the embedded tokenizer by utilizing dot product logit projection over the full vocab (128k) and extracting the exact word text.
- **Result**: El Dashboard ya no emite strings fijos "mockeados". La interacción por chat ahora responde usando iteraciones bare-metal AVX2 puras sobre los `SlimeRegisters` + `output.weight` reales cargados desde la ROM (`.mud`).
