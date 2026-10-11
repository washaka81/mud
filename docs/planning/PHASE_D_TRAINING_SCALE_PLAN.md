# 📋 PLAN FASE D — Entrenamiento a Escala (Octubre 2026)

> **Objetivo:** De ~20h/epoch a **<2h/epoch**  
> **Duración:** 16 días  
> **Target:** Intel i7-1260P + Iris Xe (ADL GT2)  
> **Fecha de inicio:** Octubre 2026  
> **Basado en:** `ROADMAP.md` §Fase D, `GEMINI.md` §6.3, `SAMPLE_EFFICIENCY_TRAINING.md`

---

## 1. Resumen Ejecutivo

La Fase D es la fase de **escala y madureza** del entrenamiento. Tras cerrar todos los bloques fundacionales (A–E, F-K, Sprints 4-7), el foco ahora es **maximizar la eficiencia de entrenamiento** para alcanzar <2h/epoch en un modelo de 2B parámetros (1.58-bit ternario).

### Estado actual

| Sub-fase | Estado | Avance |
|----------|--------|--------|
| D1: 4-Pillar Efficiency | 🟡 3/4 DONE | Self-play pendiente |
| D2: bf16 Shadow Weights | 🟡 Parcial | KV bf16 DONE, training pending |
| D3: Gradient Checkpointing | ✅ DONE | `MUD_GRAD_CKPT=1` |
| D4: Pipeline overlap | 🟡 Parcial | Double-buffer DONE, full overlap pending |

---

## 2. D1 [5d] — 4-Pillar Efficiency

### 2.1 Pillar 1: Synthetic Self-Play (PENDING)

**Estado:** Priority 38, Partial implementation

**Código existente:**
- `src/mud/self_play.rs` — `SelfPlayConfig`, `calculate_shannon_entropy`, `apply_gradient_jitter`
- `src/mud/debate_trainer.rs` — uso en debate sessions

**Plan de implementación:**

#### Día 1-2: Self-Play Loop Integration
```rust
// src/mud/corpus_trainer/session.rs
// Integrar self-play en el training loop principal:
// 1. Generar cadena sintética via sampling autoregresivo
// 2. Feed back a SlimeBackward para QAT
// 3. Usar SelfPlayConfig para control de jitter/entropy

fn self_play_training_step(
    &mut self,
    prompt: &[u32],
    cfg: &SelfPlayConfig,
) -> TrainingResult {
    // 1. Generar cadena de alta confianza
    let synthetic_chain = self.generate_synthetic_chain(prompt, cfg);
    
    // 2. Feed back al backward pass
    let gradients = self.slime_backward(&synthetic_chain)?;
    
    // 3. Aplicar jitter para evitar over-fitting
    apply_gradient_jitter(&mut gradients, cfg, self.turn_count);
    
    // 4. Optimizer step
    self.optimizer_step(gradients)
}
```

**Entregables:**
- `self_play_training_step` en `session.rs`
- `SelfPlayConfig` extendido con parámetros de entrenamiento
- Tests: `test_self_play_chain_generation`, `test_self_play_gradient_flow`

#### Día 3: Self-Play + Circuit Integration
- Integrar self-play en `run_training_circuit` (`circuit.rs`)
- Usar `MUD_SELF_PLAY=1` para activar
- Curriculum: `align -> professor -> debate -> games -> self-play`

#### Día 4-5: Self-Play Optimization
- AOT Binary Caching para skip tokenization overhead
- Zero-copy VRAM buffers via `sync_all`
- Benchmark: self-play vs corpus training speed

### 2.2 Pillar 2: JEPA Lexical Resonance (DONE ✅)

**Estado:** Priority 34, Implemented

**Verificación:**
- `jepa_z` en `SlimeWorkspace` — ✅
- `jepa_energy` inicializado desde magnitud de embedding — ✅
- Gate multiplicativo en control — ✅

### 2.3 Pillar 3: Geometric Gradient Orthogonalization (DONE ✅)

**Estado:** Priority 41/43, Implemented

**Verificación:**
- `select_optimizer()` → Muon para matrices cuadradas — ✅
- Newton-Schulz iterations en `muon_state.rs` — ✅
- `adam_state.rs` para moments — ✅

### 2.4 Pillar 4: Sampled Softmax with Hard Negatives (DONE ✅)

**Estado:** Priority 32, Implemented

**Verificación:**
- 63 hard negatives del contexto window — ✅
- Sampled Softmax en trainer — ✅
- `loss_cert.rs` certifica convergencia — ✅

---

## 3. D2 [3d] — bf16 Shadow Weights

### 3.1 Estado actual

| Componente | Estado |
|-----------|--------|
| KV cache bf16 | ✅ `MUD_KV_DTYPE=bf16`, `kv_dtype.rs` |
| Shadow weights (FP32) | ✅ `adam_state.rs`, `ash_qat_dispatcher.rs` |
| Shadow weights (bf16) | 🟡 Pending |

### 3.2 Plan de implementación

#### Día 1: bf16 Shadow Weight Storage

```rust
// src/mud/adam_state.rs
pub struct AdamState {
    // FP32 moments (existing)
    pub exp_avg: Vec<f32>,
    pub exp_avg_sq: Vec<f32>,
    
    // NEW: bf16 shadow weights for memory reduction
    pub shadow_bf16: Option<Vec<u16>>,  // MUD_BF16_SHADOW=1
}

impl AdamState {
    pub fn store_shadow_bf16(&mut self, weights: &[f32]) {
        if std::env::var("MUD_BF16_SHADOW").is_ok() {
            self.shadow_bf16 = Some(weights.iter().map(|&w| f32_to_bf16(w)).collect());
        }
    }
    
    pub fn load_shadow_bf16(&self) -> Vec<f32> {
        self.shadow_bf16.as_ref()
            .map(|bf16| bf16.iter().map(|&b| bf16_to_f32(b)).collect())
            .unwrap_or_else(|| self.exp_avg.clone())
    }
}
```

**Funciones auxiliares:**
- `f32_to_bf16(f32) -> u16` — cast sin rounding (truncate mantissa)
- `bf16_to_f32(u16) -> f32` — shift left 16 bits

#### Día 2: bf16 Shadow in Optimizer Step

```rust
// src/mud/optim.rs
pub fn apply_optimizer_cpu_step_and_pack(
    &mut self,
    grads: &[f32],
    weights: &[f32],
    packed: &mut [u8],
) {
    // 1. Store shadow weights (bf16 if MUD_BF16_SHADOW=1)
    self.adam_state.store_shadow_bf16(weights);
    
    // 2. Adam step (FP32 moments)
    adam_step_avx2(&mut self.adam_state, grads, weights);
    
    // 3. Pack to ELUT 4-bit
    pack_ternary_into(packed, weights, &self.row_scales);
}
```

#### Día 3: Memory Reduction Benchmark

- Medir reducción de RAM: ~4GB → <2GB
- Benchmark: bf16 shadow vs FP32 shadow (speed + memory)
- Tests: `test_bf16_shadow_roundtrip`, `test_bf16_shadow_memory_reduction`

---

## 4. D3 [3d] — Gradient Checkpointing (DONE ✅)

### 4.1 Estado actual

| Feature | Estado |
|---------|--------|
| `MUD_GRAD_CKPT=1` (segmented mode) | ✅ DONE |
| `MUD_GRAD_CKPT_SEG=N` (segment size) | ✅ DONE |
| `MUD_GRAD_CKPT_RESIDUAL=1` (residual bank) | ✅ DONE |

### 4.2 Optimización adicional (Día 1-3)

#### Día 1: Batch size optimization
- Probar batch 2-4× mayor con grad ckpt
- Medir trade-off speed vs memory

#### Día 2: Residual bank optimization
- `recompute_from_residual_bank` — optimizar para menos recomputación
- Usar Fenwick tree para tracking de segmentos

#### Día 3: Benchmark
- `MUD_GRAD_CKPT=1 MUD_TRAIN_FULL_SEQ=1` vs sin ckpt
- Medir: speed, memory, convergence

---

## 5. D4 [5d] — Pipeline Overlap

### 5.1 Estado actual

| Componente | Estado |
|-----------|--------|
| L-05 True Double-Buffer | ✅ DONE (`ash_backend.rs`, `ash_qat_dispatcher.rs`) |
| Full pipeline overlap | 🟡 Pending |

### 5.2 Plan de implementación

#### Día 1-2: CPU → GPU Pipeline

```rust
// src/mud/ash_qat_dispatcher.rs
// Pipeline: CPU prepare → GPU GEMV → CPU readback → next chunk overlap

pub struct PipelineOverlap {
    // Double-buffer slots
    pub cb_slots: [CommandBuffer; 2],
    pub fence_slots: [Fence; 2],
    
    // CPU prepare buffer (async)
    pub prepare_thread: Option<JoinHandle<()>>,
    
    // GPU GEMV (async submit)
    pub gemv_pending: bool,
    
    // CPU readback (async)
    pub readback_pending: bool,
}

impl PipelineOverlap {
    pub fn step_pipeline(&mut self, chunk: &[f32]) {
        let slot = self.current_slot();
        
        // 1. CPU prepare (async, next chunk)
        self.prepare_async(chunk);
        
        // 2. GPU GEMV (submit without wait)
        self.gemv_async(slot);
        
        // 3. CPU readback (drain previous slot)
        self.readback_drain(self.prev_slot());
        
        // 4. Swap slots
        self.swap_slots();
    }
}
```

#### Día 3-4: Full Pipeline Implementation

1. **CPU prepare thread:** Prepara el siguiente chunk mientras GPU trabaja
2. **GPU GEMV:** Submit sin wait (L-05 double-buffer)
3. **CPU readback:** Drain del slot anterior
4. **Overlap:** Todos los stages en paralelo

#### Día 5: Benchmark

- Medir throughput: tokens/s con vs sin pipeline overlap
- Objetivo: ~80% overlap (vs 0% serial actual)
- `MUD_PIPELINE_OVERLAP=1` para activar

---

## 6. Cronograma

| Día | D1 (5d) | D2 (3d) | D3 (3d) | D4 (5d) |
|-----|---------|---------|---------|---------|
| 1 | Self-play loop integration | bf16 shadow storage | Batch size opt | CPU→GPU pipeline |
| 2 | Self-play + circuit | bf16 shadow optimizer | Residual bank opt | CPU→GPU pipeline |
| 3 | Self-play optimization | Memory benchmark | Benchmark | Full pipeline impl |
| 4 | Self-play optimization | — | — | Full pipeline impl |
| 5 | Self-play optimization | — | — | Benchmark |
| 6 | — | — | — | — |
| 7 | — | — | — | — |
| 8 | — | — | — | — |
| 9 | — | — | — | — |
| 10 | — | — | — | — |
| 11 | — | — | — | — |
| 12 | — | — | — | — |
| 13 | — | — | — | — |
| 14 | — | — | — | — |
| 15 | — | — | — | — |
| 16 | Final benchmark + report | | | |

---

## 7. Métricas Objetivo

| Métrica | Actual | Objetivo (Post-D) |
|---------|--------|-------------------|
| Training speed | ~20h/epoch | **<2h/epoch** |
| RAM usage | ~4GB | **<2GB** (bf16 shadows) |
| Overlap CPU/GPU | Double-buffer (L-05) | **~80%** (full pipeline) |
| Inference | ~5 tok/s | **~20 tok/s** |
| Tests | 344 | **344+** (nuevos tests D1-D4) |
| Clippy | 0 errors, 0 warnings | **0 errors, 0 warnings** |

---

## 8. Variables de Entorno Nuevas

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MUD_SELF_PLAY` | `0` | Activa self-play training loop |
| `MUD_BF16_SHADOW` | `0` | Usa bf16 shadow weights (reduce RAM 50%) |
| `MUD_PIPELINE_OVERLAP` | `0` | Activa full CPU→GPU→CPU pipeline overlap |
| `MUD_SELF_PLAY_TURNS` | `8` | Número de turns de self-play por step |
| `MUD_SELF_PLAY_TEMP` | `0.8` | Temperatura para self-play generation |

---

## 9. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Self-play genera loops de feedback | Media | Alto | `apply_gradient_jitter` + entropy monitoring |
| bf16 shadow reduce precisión | Baja | Medio | Fallback a FP32 si `MUD_BF16_SHADOW=0` |
| Pipeline overlap causa race conditions | Media | Alto | Usar L-05 double-buffer + fences |
| Vulkan SIGSEGV en Iris Xe | Alta | Alto | `MUD_USE_VULKAN=0` en training (ya fixado) |

---

## 10. Entregables

| Entregable | Archivo | Estado |
|-----------|---------|--------|
| Self-play training step | `src/mud/corpus_trainer/session.rs` | Día 2 |
| bf16 shadow weights | `src/mud/adam_state.rs` | Día 2 |
| Pipeline overlap | `src/mud/ash_qat_dispatcher.rs` | Día 4 |
| Tests D1-D4 | `src/mud/` | Día 5 |
| Benchmark report | `docs/sessions/MUD_SESSION_REPORT_2026-10_PHASE_D.md` | Día 16 |

---

## 11. Comandos de uso (post-Fase D)

```bash
# Entrenamiento con self-play + bf16 shadow + pipeline overlap
MUD_SELF_PLAY=1 MUD_BF16_SHADOW=1 MUD_PIPELINE_OVERLAP=1 \
  cargo run --release --bin run_trainer -- models/smollm2.mud --epochs 10

# Benchmark de pipeline overlap
MUD_PIPELINE_OVERLAP=1 MUD_GRAD_CKPT=1 \
  cargo run --release --bin run_trainer -- models/smollm2.mud --epochs 1

# Solo self-play
MUD_SELF_PLAY=1 MUD_SELF_PLAY_TURNS=16 \
  cargo run --release --bin run_trainer -- models/smollm2.mud --epochs 5
```

---

## 12. Conclusión

La Fase D es el puente entre el prototipo funcional y el producto de producción. Con los 4 pilares de eficiencia, bf16 shadows, gradient checkpointing, y pipeline overlap, el objetivo es reducir el tiempo de entrenamiento de ~20h/epoch a **<2h/epoch** — un **10× speedup** que hace viable el entrenamiento de modelos de 2B parámetros en un portátil de consumo.

**Documentación guardada en:** `docs/planning/PHASE_D_TRAINING_SCALE_PLAN.md`
