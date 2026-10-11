# Forge LLM (MUD) — Roadmap Consolidado

> **Consolidado el:** 2026-07-31 · **Actualizado el:** 2026-10-11  
> **Última sincronización con GEMINI.md:** 2026-10-11  
> **Estado:** ⚠️ **corregido por la auditoría 2026-10-11** — `cargo test --lib --release` FALLA (UB fenwick), CI inválido, entrenamiento con gradientes incorrectos, LM head en f32 denso. Los conteos previos ("423 tests ✅ | release verde") eran **solo debug** y quedan retirados.

---

## 🥇 PRIORIDAD 0 — antes de cualquier fase de producto

| | |
|---|---|
| **Backlog activo** | [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](./docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) — `R-##`, 55 ítems, 7 olas |
| **Auditoría de origen** | [`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](./docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) |
| **Estado SSOT** | [`GEMINI.md`](./GEMINI.md) §0.3 |
| **Regla** | Las fases A–E de este roadmap **no avanzan** hasta cerrar las Olas 0–1 (verdad/CI + correctitud de entrenamiento). Las Olas 2–6 se ejecutan después, en el orden del plan. |

> Ninguna capacidad nueva de producto se considera "en progreso" mientras el CI no pueda ejecutarse, el `test --release` no esté verde y el entrenamiento no reduzca el loss con gradientes correctos.

---

## Jerarquía de documentos

| Documento | Rol | Autoridad |
|-----------|-----|-----------|
| **[`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](./docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md)** | **Prioridad 0 — backlog activo `R-##`** | **SSOT de ejecución** |
| [`GEMINI.md`](./GEMINI.md) | Políticas (P-#), estado técnico, ledger L-## | **SSOT** para agentes y humanos |
| [`AGENTS.md`](./AGENTS.md) | Contexto de sesión, fixes recientes, comandos | Derivado de GEMINI.md — no contradecir |
| **[`ROADMAP.md`](./ROADMAP.md)** (este archivo) | Visión de producto + fases + narrativa arquitectónica | **SSOT** para *dirección* |
| [`docs/architecture/MUD_COMPUTE_STACK.md`](./docs/architecture/MUD_COMPUTE_STACK.md) | Stack de cómputo LIVE | Detalle técnico del stack |
| [`docs/architecture/MUD_REGISTER_BIT_PACKING.md`](./docs/architecture/MUD_REGISTER_BIT_PACKING.md) | Bit packing de registros | Detalle técnico de registros |

> **Antes de este consolidamiento**, la visión y fases estaban distribuidas en 5 archivos: `VISION_ROADMAP.md`, `PLAN_MAESTRO.md`, `docs/architecture/ROADMAP.md`, `docs/manuals/MUD_ROADMAP_MASTER.md`, y `docs/research/MUD_PLAN_PRIORIDADES_ROADMAP_2026-07-20.md`. Todos han sido fusionados aquí. Los archivos originales ahora contienen redirecciones.

---

## 1. IDENTIDAD DEL PROYECTO

Forge LLM (MUD — Modular Understanding Dynamics) es un motor de inferencia y entrenamiento para LLMs ternarios (1.58-bit) orientado a **CPU Intel de consumo**, con aceleración opcional en iGPU (Iris Xe vía Vulkan). Target de diseño de referencia: **Intel i7-1260P** (P-cores + HT, Iris Xe).

**No es** un wrapper de llama.cpp ni un port de PyTorch. Es Rust + ASM AVX2 + Vulkan Compute, con control del pipeline desde el kernel hasta el pool de hilos.

### Filosofía fundacional

- **Sin dependencias pesadas** — runtime en un binario compacto
- **Sin Python en producción** — P-07
- **Sin NVIDIA obligatorio** — hardware Intel estándar
- **Control total** — ELUT, JEPA/mHC, optimizers, dispatch CPU/GPU

### Norte de producto

Un LLM ~2B (~400MB en 1.58-bit) **entrenable y usable** en un portátil ~800€, offline y privado.

**No es el objetivo** ganar MMLU frente a frontier APIs.  
**Sí es el objetivo** capacidad media útil × coste casi cero × accesibilidad total.

### La ecuación del valor

```
Valor = Capacidad / Coste × Accesibilidad

Hoy:     GPT-4 = Alta capacidad / Coste enorme × 0 accesibilidad (API)
         Llama  = Alta capacidad / Coste medio × Baja accesibilidad (GPU 24GB+)

Forge:   MUD = Capacidad media / Costse casi cero × Accesibilidad total
         (cualquier portátil Intel, sin GPU, sin internet)
```

---

## 2. ESTADO ACTUAL (2026-07-31)

### Lo que funciona (LIVE)

| Área | Realidad |
|------|----------|
| Forward + backward STE QAT | ELUT 4-bit + PRQ |
| Accum / registers | **f16-packed dual (re, im) u32** — f32 compute, f16 storage |
| GEMV | AVX2×8 + **auto GPU** (`gemv_policy`) |
| SiLU / attn dots / LM logits | ASM cableado |
| Optimizers | Muon / GaLore / Chunked / **Adam moments** (in-place NS + scratch reuse 2026-07-26) |
| Estabilidad | JEPA OU + mHC |
| Train | Sampled Softmax + **full-seq windows** (L-10 + stream D) |
| MoE | ExpertBus + **`.mud` load / train-expert** (B) |
| Context | HCA 32k + **CSA top-k** (E) |
| ash Vulkan | NS, GEMV, RMSNorm/MHA helpers, QAT double-buffer; D5 elementwise dispatch APIs (integ. DONE 2026-07-27) |
| Tests | **423 lib** (0 failing); `./mud.sh ci|audit-full` |
| Telemetry TUI | **DONE** — key-parse `[TELEM]`/`[DW]` + Weight Δ panel (2026-07-20); `grad_norm`/`entropy`/`stp_loss` en charts (2026-07-31); cursor removed from loss profile + dynamic band legend overlay |
| Hot loops | **DONE** — pointer/LUT (P-00/P-01); + Muon/GaLore scratch reuse 2026-07-26; checkpoint batch packing `pack_ternary_into` + `mem::take` zero-copy (2026-07-31) |
| Fenwick trees | **DONE** — `FenwickF32`/`FenwickU32`/`FenwickI32`/`LogF32` + AVX2/BMI1/BMI2 (802 LOC, 25 tests, 19.5× speedup). F1 (debate_trainer) + F5b (percentile loss gate) wired 2026-07-26 |
| Sprint 4b | `prefix_sum_batch4_avx2` + `find_kth_walk` (BMI1/BMI2) |
| Sprints 6-7 | Security docs (`THREAT_MODEL.md`, `HARDENING.md`, `INCIDENT_RESPONSE.md`) |
| Sprint 5 | 5.1 zero-alloc MoE; 5.2 AtomicU8; 5.3 LUT backward; 5.4 branchless pack |
| F-K streams | **DONE** (2026-07-23): QKV one-CB, multi-expert STE, long full-seq, KV bf16, CSA LSH, loss cert, converter P-13 |
| Chat refactor | Debug condicional, conversation history, system prompt, temperature, Fenwick sampling, colores |
| Converter audit | 6 findings, 3 fixes, 6 optimizaciones |
| mHC sync FIX (2026-07-31) | Critical: `sync_shadow_to_mud` persisted `blk.N.mhc_alpha`/`mhc_beta`; `hash_trained_weights` extended to mmap-backed path; checkpoint packing → `pack_ternary_into` batch; `mem::take` zero-copy |
| Checkpoint telemetry | `MUD_MHC_SYNC` / `MUD_DW_DEBUG` env vars control mHC persistence + ΔW reporting; `./mud.sh mhc-sync` verifies persistence |
| smollm2→.mud→C-MUD | Conversión completa, entrenamiento, auditoría |
| Security hardening (2026-07-26) | SEC-H1 `MudFile::load` caps (`MAX_META_ENTRIES/TENSOR_ENTRIES/STRING_LEN/SHAPE_DIMS`); SEC-H7 `collect_files` recursion cap; SEC-H8 `MUD_CIRCUIT_MAX_CYCLES` cap |
| PERF-7 tied-output | `forward_last_hidden_and_head` zero-copy (`std::mem::take`) — ~295 MB saved en smollm2 inference |
| MUD-WORM | WormPropagator + JEPARouter + MhCoupler + WormChain + WormholeRouter/Network (30 tests) |
| SlimeX | SX-01..SX-07 CellV5 (u32 8:8:8:8) + ASM AVX2 unpacker + Memory Continuity Table (256 slots) + AOT Boot Indexer + MudFileMerger + ShadowExpertBus + SmartTokenizer |
| Pendiente (2026-07-26) | 5.1 CellV5 Fase 2 ASM kernels; 5.2 CellV5 Fase 3 KV pre-compresión; Vulkan D5 elementwise integration — **todo COMPLETADO** (2026-07-27 a 2026-07-29). Riesgo HW (Iris Xe SIGSEGV) mitigado vía `probe_gpu()` blacklist. Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md`. |

### Ledger y streams cerrados

| Bloque | Estado |
|--------|--------|
| **Fase A** L-01…L-08 | **DONE** |
| **Fase B/B+** GEMV tile + GPU path | **DONE** |
| **Fase C** L-09…L-11 | **DONE** |
| **Fase E-early** L-12…L-15 | **DONE** (C-MUD research + ckpt) |
| **Depth A–E** Adam→CSA | **DONE** 2026-07-16 |
| **F1** Trainable mHC α/β | **DONE** 2026-07-17 (`mhc_scale_sgd_step`, CPU+ash) |
| **F2** STP trajectory loss | **DONE** 2026-07-17 (`stp_loss`, `MUD_TRAIN_STP`) |
| **UI** Unified trainer console | **DONE** 2026-07-17 (`trainer_ui.rs` + `mud.sh train`) |
| **F4** Fenwick trees | **DONE** 2026-07-24 (`src/mud/fenwick.rs`, 802 LOC, 25 tests) |
| **Sprint 5** Hot-path perf | **DONE** 2026-07-24 |
| **Sprints 6-7** Security docs | **DONE** 2026-07-24 |
| **Batch A/B/C/D-partial** (2026-07-26) | F1 nucleus (debate) + F5b percentile loss gate + Muon/GaLore scratch reuse + SEC-H1/H7/H8 + PERF-7 tied-output zero copy + run_trainer canonical paths + MUD_KV_COMPRESS_CELLV5 reserve | **DONE** |
| **Batch D resto** | CellV5 Fase 2 ASM kernels (`cmud_cell_v5_avx2.s`) + Fase 3 KV pre-compression (delta-encoding per-i8) | **DONE** 2026-07-29 |
| **Batch E** | Vulkan D5 elementwise integration (`step_elementwise_sync`) into `jepa_stabilizer` + `stp_loss_and_grad` w/ CPU fallback | **DONE** 2026-07-27 |
| **2026-07-31 mHC sync FIX** | Critical: `sync_shadow_to_mud` never persisted `blk.N.mhc_alpha`/`mhc_beta` to `.mud`; `hash_trained_weights` only hashed `owned_data` (mHC mmap-backed was invisible); checkpoint packing loop scalar RMW → `pack_ternary_into` batch (8/elem); eliminated double-clone via `mem::take` | **DONE** 2026-07-31 (423 tests ✅ | clippy 0 ✅) |

Stack detail: [`docs/architecture/MUD_COMPUTE_STACK.md`](./docs/architecture/MUD_COMPUTE_STACK.md).

---

## 3. NARRATIVA ARQUITECTÓNICA

### 3.1 El Dispatch Empaquetado FP32 Ultra-optimizado

En el corazón de Forge LLM late un mecanismo de dispatch que no existe en ningún otro proyecto:

```
Pesos Ternary2Bit (4-bit ELUT) → PRQ Scales → Dispatch dinámico:
  ├─ GPU (≥1M elementos):  Shader ternario con shared memory tiling
  ├─ CPU 4-rows batch:     ASM AVX2 con activación compartida + prefetch
  └─ CPU single row:       ASM AVX2 con 8 accumuladores + NaN guard

Todas las rutas convergen en FP32. El mismo cómputo, el mismo resultado,
3 implementaciones distintas elegidas por perfilado en caliente.
```

**¿Qué lo hace único?**

No es un "llama.cpp con flag de GPU". Es un sistema donde **cada multiplicación matriz-vector ternaria** pasa por un router que decide:

1. **¿Es grande?** (≥1M elementos) → GPU: 96 EUs en Iris Xe, shared memory tiling, `vpsrlvd`/`vpand`/`vpcmpeqd` en compute shader
2. **¿Es mediano?** → 4-rows AVX2: un solo load de activaciones para 4 filas, 5 prefetches simultáneos, 14 registros YMM
3. **¿Es pequeño?** → Single row AVX2: 8 accumuladores, unroll 64 elementos, reducción `vhaddps`

El router pesa 3 ciclos. La decisión es por **cada GEMV**, no por capa ni por modelo. Esto es dispatch empaquetado: el empaquetado (ELUT 4-bit) y el dispatch (CPU/GPU) están acoplados en un solo sistema que garantiza que **cada operación corre en el hardware óptimo**.

### 3.2 AVX2 + Vulkan Ash: La Simbiosis

**AVX2 (CPU) — Donde nació el proyecto:**

- 17 kernels ASM (14 tras purge) escritos a mano en AT&T syntax
- `ternary_gemv.s`: 8 accumuladores, 64 elementos/iteración, `vpsrlvd` para ELUT
- `ternary_gemv_4rows.s`: Activación compartida entre 4 filas, 5 prefetches
- `adam_step.s`: 16/16 registros YMM, sin spill, sin concesiones
- `pcore_pool.rs`: 8 threads pineados a P-cores, sin Rayon, sin locks

**Vulkan Ash (GPU) — El acelerador olvidado:**

- Zero-copy en UMA: CPU y GPU comparten la misma RAM (Iris Xe)
- Shaders: `mha.comp` (attention con shared memory), `rms_norm.comp` (subgroupAdd)
- Double-buffer: dos fences rotativos para overlap CPU/GPU
- Sin staging buffers, sin transfers PCIe, sin `vkCmdCopyBuffer`
- La GPU es una extensión del espacio de memoria de la CPU

**La simbiosis real:**
```
CPU prepara → GPU acelera → CPU lee sin copiar
    │              │              │
    │   AVX2+8T    │  UMA zero-   │  Mapped pointers
    │   RMSNorm    │  copy        │  = misma RAM
    │   JEPA       │  Attention   │
    │   SiLU       │  GEMV large  │
    │   GEMV small │  RMSNorm     │
    └──────────────┴──────────────┘
```

No hay "cambio de contexto" costoso. Los mapped pointers de Vulkan son punteros planos que la CPU puede leer inmediatamente después de un fence check. En UMA, la GPU escribe directamente en la RAM que la CPU lee.

### 3.3 El Pipeline de Entrenamiento Adaptativo

Forge LLM no tiene un solo optimizer. Tiene **cinco**, elegidos dinámicamente por forma de matriz:

```
select_optimizer(rows, cols):
  ┌─ Cuadrada (<100K) + ratio ≈ 1 → Muon (Newton-Schulz, 5 iters)
  ├─ Cuadrada (grande) + ratio ≈ 1 → Muon
  ├─ Tall (rows/cols > 2.5)        → GaLore (rango bajo, rank=cols/4)
  ├─ Wide (cols/rows > 2.5)        → ChunkedAdam (bloques de 512)
  └─ Embedding (gigante)           → SparseAdam (solo filas activas)
```

El `select_optimizer()` **está cableado al step** (**L-01 DONE**): Muon / GaLore / Chunked preprocess + STE pack; Adam/Sparse con momentos reales (**stream A**). Newton-Schulz GPU opcional (**L-02**, `MUD_USE_VULKAN=1`).

Ejemplos de dispatch (smollm-class):
- `attn_q.weight` (576×576) → Muon (Newton-Schulz)
- `ffn_up.weight` (1536×576) → GaLore (rank bajo)
- Embeddings / wide → ChunkedAdam o SparseAdam (filas activas)

**Aún abierto (F+):** multi-expert STE joint (stream G), no el dispatch de optimizers densos.

### 3.4 Mini MoE Modular y Acoplable

La arquitectura MoE no es un monolito. Es un **bus de expertos** donde cada experto es una caja negra con interfaz estándar:

```
Layer FFN original:
  x → [Up·SiLU(Gate·x)]·Down → y

Layer con Mini MoE:
  x → Router(x) → top-k pesos + índices
      ├─ Experto 0: [Up₀·SiLU(Gate₀·x)]·Down₀
      ├─ Experto 1: [Up₁·SiLU(Gate₁·x)]·Down₁
      ├─ ...
      └─ Experto N: [UpN·SiLU(GateN·x)]·DownN
      ↓ weighted sum
      y = Σ weightₙ · expertₙ(x)
```

#### Interfaz del Experto (SlimeExpert)

```rust
pub struct SlimeExpert {
    // Punteros planos (P-00): mismos pesos Ternary2Bit + PRQ
    pub up: SlimeLayerGEMV,    // FFN up projection
    pub gate: SlimeLayerGEMV,  // FFN gate projection
    pub down: SlimeLayerGEMV,  // FFN down projection

    // Metadata
    pub hidden: usize,
    pub ffn_mid: usize,
    pub id: u16,
}

impl SlimeExpert {
    pub fn forward(&self, x: &SlimeWorkspace) -> Slice<f32>;
    pub fn backward(&self, grad: &[f32], tape: &mut SlimeLayerTape);
    pub fn load_from_mud(&mut self, tensors: &[MudTensor], prefix: &str);
    pub fn unload(&mut self);  // libera recursos, deja el bus limpio
}
```

#### El Bus de Expertos (ExpertBus)

```rust
pub struct ExpertBus {
    // Hasta 64 expertos (direccionables con u16)
    pub experts: Vec<Option<SlimeExpert>>,
    // Router ternario: proyección a logits + softmax top-k
    pub router: SlimeLayerGEMV,
    // Pool de threads exclusivo para expertos (P-cores)
    pub pool: PCorePool,
}

impl ExpertBus {
    // En caliente: añadir o quitar expertos sin recompilar
    pub fn mount(&mut self, slot: u16, expert: SlimeExpert);
    pub fn unmount(&mut self, slot: u16);

    // Forward: router + top-k dispatch en paralelo
    pub fn forward(
        &self,
        x: &SlimeWorkspace,
        k: usize,         // top-k, típicamente 1 o 2
        out: &mut [f32],
    );

    // Backward: gradientes solo a expertos activos
    pub fn backward(
        &self,
        grad: &[f32],
        tape: &mut SlimeLayerTape,
        active_experts: &[u16],
    );
}
```

#### Enrutamiento Ternario

El router no es un transformer denso. Es una proyección ternaria ligera:

```rust
// hidden → num_experts con pesos Ternary2Bit
router_logits = ternary_gemv(x, router_weight, router_scale)
// top-k con mask de ruido para balanceo
noise = gumbel_noise(seed) * 0.01
top_k_indices, top_k_weights = topk_softmax(router_logits + noise, k)
```

El costo del router: `hidden × num_experts` operaciones ternarias.
Para hidden=2560, num_experts=8: **20K operaciones** — ~0.3% del costo de una capa FFN completa (6912×2560×2 ≈ 35M).

#### Dispatch en PCorePool

Los expertos activos se ejecutan en paralelo:

```
Router → top-k = [Expert 2, Expert 5] → Dispatch:
  ├─ PCore 0-1: Expert 2 forward (up+gate+silu+down)
  ├─ PCore 2-3: Expert 5 forward (up+gate+silu+down)
  └─ PCore 4-7: disponibles para otros expertos o next layer prep
```

Cada experto usa 2 threads de PCorePool (1 para up+gate, 1 para down).
Con 8 threads y top-k=2: 4 threads por experto. Con top-k=4: 2 threads por experto.

#### Carga y Descarga en Caliente

```rust
// En runtime, sin detener la inferencia:
bus.unmount(3);                               // quita experto 3
bus.mount(3, SlimeExpert::load_from_mud(      // monta nuevo experto 3
    &mud.tensors("blk.5.expert.3.*")
));

// Balanceo de carga: si un experto está sobrecargado, se clona:
let backup = bus.experts[3].clone();
bus.mount(7, backup);  // experto 7 = mismo conocimiento, más capacidad
```

#### Integración con el Modelo

El `.mud` se extiende con una sección MoE:

```
blk.5.expert.0.w1.weight      # FFN original (expert 0 = siempre activo)
blk.5.expert.0.w3.weight
blk.5.expert.0.w2.weight
blk.5.expert.1.w1.weight      # Experto adicional 1
blk.5.expert.1.w3.weight
blk.5.expert.1.w2.weight
...
blk.5.expert.7.w1.weight      # Experto adicional 7
blk.5.expert.7.w3.weight
blk.5.expert.7.w2.weight
blk.5.moe_router.weight       # Router ternario (hidden × num_experts)
blk.5.moe_router.prq_scale
```

**El modelo base (expert.0) funciona sin MoE.** Los expertos 1-7 son capas opcionales. Si el router no está presente, se usa expert.0 como FFN denso normal (compatibilidad hacia atrás).

#### Entrenamiento de Expertos

Cada experto se entrena de forma semi-independiente:

1. **Fase 1: Entrenar router** con load-balancing loss (`importance · CV²`)
2. **Fase 2: Entrenar expertos** con los datos donde fueron seleccionados
3. **Fase 3: Fine-tuning conjunto** router + expertos con learning rate reducido

Esto permite:
- Añadir expertos nuevos sin reentrenar todo el modelo
- Especializar expertos en dominios (código, matemáticas, lenguajes)
- Retirar expertos que no se usan sin pérdida de capacidad

#### Beneficios Clave

| Aspecto | FFN denso | Mini MoE (4 expertos, top-k=2) |
|---------|-----------|-------------------------------|
| Parámetros totales | 2B | 2B + 3×FFN = ~3.5B |
| Parámetros activos | 100% | ~50% (2 de 4 expertos) |
| FLOPs por token | 35M | 35M (router) + 2×35M (2 expertos) = 105M → 3× más |
| Calidad | Línea base | ~2× parámetros efectivos por token (MoE paper) |
| Memoria | ~400MB | ~700MB (por los expertos adicionales) |
| Entrenamiento | Desde cero | Incremental: añadir expertos uno a uno |

**La clave del Mini MoE es que se acopla sin cambiar la arquitectura base.** El modelo original (expert.0) sigue siendo un FFN denso funcional. Los expertos 1-N son enchufables: se montan, entrenan, y retiran en caliente.

---

## 4. VISIÓN ARQUITECTÓNICA (resumen)

```
Inference / Training (STE) / Drafter
        ↓
  SlimeWorkspace (P-00/P-01)
        ↓
  AVX2×8  |  ash (auto GEMV / NS / QAT)  |  PCorePool
        ↓
  Weights: ELUT 4-bit + PRQ
  Optimizers LIVE: Muon · GaLore · Chunked · Adam/Sparse
  Context: dense ring + HCA + CSA top-k
  Estabilidad: JEPA + mHC
```

Principios: P-00…P-27 — índice completo en `GEMINI.md` §9.

---

## 5. HOJA DE RUTA Q3–Q4 2026

### Fase A–E ledger — **cerrada**

Criterio de salida cumplido: strategies live, sin dead workspace, banners = verdad, packing, MoE bus, CI, HCA 32k, grad ckpt, C-MUD kernel.

### Depth streams A–E — **cerrada** (2026-07-16)

Adam · MoE load · GEMV auto · full-seq · CSA indexer.

### Siguiente (F+ research backlog) — **TODO COMPLETADO (2026-07-24)**

> **F1/F2/UI CLOSED (2026-07-17):** mHC α/β trainable + STP aux loss + unified
> trainer console + project-adapted `mud.sh train`. Ver
> `docs/architecture/MUD_TRAINER_TERNARY_JEPA_MHC.md` §9–§10 y
> `docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`. Phase 3 (mHC `n=2`) deferred.
>
> **TLM CLOSED (2026-07-20):** live training telemetry TUI — `[TELEM]`→log + key parser + Weight Δ panel; pointer-optimized hot loops (P-00/P-01).
>
> **TLM EXTENDED (2026-07-31):** `grad_norm` + `entropy` + `stp_loss` logged in `[TELEM]` line + file format; TUI Efficiency mode shows Entropy (bits) + STP Loss charts.
>
> **F-K CLOSED (2026-07-23):** Todos los streams de mejora post-A–E completados.

Ver **[`docs/research/MUD_IMPROVEMENTS_POST_AE.md`](./docs/research/MUD_IMPROVEMENTS_POST_AE.md)**:

| ID | Tema | Estado | Código |
|----|------|--------|--------|
| **F** | QKV multi-matrix one CB | ✅ DONE (2026-07-23) | `dispatch_gemv_qkv_host_sync` |
| **K** | Loss certification CI | ✅ DONE (2026-07-23) | `loss_cert.rs`, `cert-loss` |
| **G** | Multi-expert STE (SlimeX Dynamic Stack / ShadowExpertBus) | ✅ DONE (2026-07-23) | `moe_train.rs`, `MUD_MOE_TRAIN=1\|hash` |
| **H** | Joint BPTT / seq largas | ✅ DONE (2026-07-23) | `MUD_GRAD_CKPT_RESIDUAL=1` |
| **I** | KV bf16/quant | ✅ DONE (2026-07-23) | `kv_dtype.rs`, `MUD_KV_DTYPE=bf16` |
| **J** | CSA v2 LSH / W_compress | ✅ DONE (2026-07-23) | `MUD_CSA_LSH=1`, `lsh_signature_learned` |
| **L** | Converter P-13 | ✅ DONE (2026-07-23) | `ensure_canonical_metadata_aliases` |

> **Nota Arquitectónica (G):** La implementación de **SlimeX** permitirá acoplar/desacoplar expertos dinámicamente en tiempo de ejecución. Esto abre la puerta a un escalamiento híbrido futuro donde la carga se distribuya en caliente entre GPU (Vulkan) y CPU (AVX2/PCorePool) bajo demanda.

### Sprint 5 + Fenwick + Security (2026-07-24) — **COMPLETADO**

| Trabajo | Detalle | Estado |
|---------|---------|--------|
| **Sprint 5.1** | Zero-alloc `evaluate_slime_block_moe` (workspace-backed Vec views) | ✅ DONE |
| **Sprint 5.2** | `Mutex`→`AtomicU8` GPU dispatch fast path | ✅ DONE |
| **Sprint 5.3** | `ternary_gemv_backward` LUT (branchless) | ✅ DONE |
| **Sprint 5.4** | Branchless `pack_elut_prq` | ✅ DONE |
| **Fenwick trees** | `FenwickF32`/`FenwickU32`/`FenwickI32`/`FenwickLogF32` + AVX2/BMI1/BMI2 (802 LOC, 25 tests, 19.5× speedup) | ✅ DONE |
| **Sprints 6-7** | Security docs (`THREAT_MODEL.md`, `HARDENING.md`, `INCIDENT_RESPONSE.md`) | ✅ DONE |
| **Race condition fixes** | 2 tests fallidos → 344/344 passing | ✅ DONE |
| **Dead code cleanup** | 5 `#[allow(dead_code)]` → 1 (ash_backend) | ✅ DONE |
| **Chat refactor** | Debug condicional, conversation history, system prompt, temperature, Fenwick sampling, colores | ✅ DONE |
| **Converter audit** | 6 findings, 3 fixes, 6 optimizaciones | ✅ DONE |
| **smollm2 → .mud → C-MUD** | Conversión completa, entrenamiento C-MUD, auditoría | ✅ DONE |

### Sprint 8 — Batch A/B/C/D-partial (2026-07-26) — **COMPLETADO (parcial)** > [`docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md`](./docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md)

Cierre de pendientes identificados en el inventario exhaustivo del proyecto (58 items agrupados en 10 categorías).

| Trabajo | Detalle | Estado |
|---------|---------|--------|
| **A. Fenwick deferred** | F1 nucleus sampling en `debate_trainer.rs` (`FenwickF32`, patrón `main.rs:438`, +4 Doppelganger pre-alloc buffers, P-01) | ✅ DONE |
| **A. Fenwick deferred** | F5b percentile loss gate en `loss_cert.rs` (`CertMode::Percentile`, `percentile_loss` via `FenwickU32` histograma, env `MUD_LOSS_CERT_MODE=percentile`, 6 tests) | ✅ DONE |
| **B. P-01 trainer** | Muon/Newton-Schulz: in-place iter + lifted scratch (`grad.to_vec()` eliminado + chunked scratch reuse, ~4 alloc/NS-call → 0) | ✅ DONE |
| **B. P-01 trainer** | GaLore: `compute_projection_matrix → &mut [f32]`; `galore_step` prealloc p_matrix + g_low | ✅ DONE |
| **B. P-01 trainer** | `grad_checkpoint.rs` doc clarifier SEG↔SEGS; `run_trainer.rs` checkpoint/corpus dirs absolutos + `MUD_TRAIN_CKPT_DIR`/`MUD_TRAIN_CORPUS_DIR` env | ✅ DONE |
| **C. V38 security** | SEC-H1 caps `MudFile::load` (`MAX_META_ENTRIES=4096`, `MAX_TENSOR_ENTRIES=65536`, `MAX_STRING_LEN`, `MAX_SHAPE_DIMS`) + 3 tests | ✅ DONE |
| **C. V38 security** | SEC-H8 `MUD_CIRCUIT_MAX_CYCLES` cap on infinite `run_training_circuit` loop | ✅ DONE |
| **C. V38 security** | SEC-H7 `COLLECT_FILES_MAX_DEPTH=32` recursion cap in `corpus::collect_files_inner` + 1 test | ✅ DONE |
| **C. V38 security** | PERF-7 tied-output path: `emb_f32.clone()` (~295 MB en smollm2) → `std::mem::take` (zero copy, NLL-permite defer del move) | ✅ DONE |
| **D. CellV5 plumbing** | `MUD_KV_COMPRESS_CELLV5` env reservado en `mud.sh` + `kv_compressed.rs` (sync warning, Fase-3 not implemented yet) | ✅ DONE (plumbing) |
| **D. CellV5 Fase 2** | 5.1 ASM AVX2 batch kernels para CellV5 + C-MUD (`cmud_cell_v5_avx2.s` con 3 kernels: `wave_collapse`/`phase_repulsion`/`gauss_mul`) | 🟡 DEFERRED |
| **D. CellV5 Fase 3** | 5.2 CellV5 pre-compresión en `CompressedKvCache` (F32→CellV5→Delta→LZ4 = 16-24× total; invariante lossless Stream I preservado) | 🟡 DEFERRED |
| **E. Vulkan D5** | Integración `step_elementwise_sync` en `slime_jepa.rs` + `stp_loss.rs` (signature refactor `&AshQatDispatcher` opt-in `MUD_USE_VULKAN`) | 🟡 DEFERRED HW |

**Rationale de deferral:** 5.1/5.2 requieren sesión dedicada con exploración bit-exacta del layout 8:8:8:8 + validadores LZ4 (riesgo de romper invariante lossless). BATCH E requiere HW sign-off Iris Xe real (el dispatcher sync usa `submit_and_wait`, no es capturable como `Result`) + falta un shader `dL/dX` (el actual `ternary_backward.comp` solo calcula `dL/dW`). `probe_gpu()` ya blacklistea Iris Xe in-binary, pero la integración sin verificar explora el unknown de vendor drivers.

### Fase D scale / madurez (resto Q4)

4-pillar efficiency, bf16 shadows, pipeline overlap — objetivo <2h/epoch. F/H/I/J ya madurado.

---

## 6. PRIORIDADES EJECUTABLES (T0–T4)

> **Fuente:** `docs/research/MUD_PLAN_PRIORIDADES_ROADMAP_2026-07-20.md` (consolidado)
>
> **Principio de validación:** todo ítem "DONE" requiere una check automática en `./mud.sh ci` o `cargo test` que lo proteja de regresión. Las validaciones largas (retrain de 25 épocas) se reemplazan por **proxies cortos** (MD5-diff + ΔW>0 en 1–2 chunks, o `conf` en 1 chunk).

### Matriz de prioridades

| Tier | Tema | Bloquea | Validación objetivo | Esfuerzo |
|------|-------|---------|----------------------|---------|
| **T0** | Correctitud (vocab + motor + gates) | SÍ — todo lo demás | tests unit cortos | M |
| **T1** | Entrenamiento que SÍ mueve pesos | SÍ — F3+/circuit | MD5-diff + ΔW en 1–2 chunks | S |
| **T2** | Coherencia de generación | calidad usable | greedy 20 tok con espacios + proxy repetición | S |
| **T3** | Robustez / ops / docs | no | `./mud.sh ci` + banner | XS |
| **T4** | Research F+ (backlog) | no | benchmarks opcionales | L |

### T0 — Correctitud (gate de lanzamiento)

#### T0.1 — Vocab: restaurar prefijo-espacio `Ġ` (palabras fusionadas) — ✅ DONE (2026-07-20)

- **Por qué:** el `.mud` tenía **0** `Ġ`/`▁`; el source `models/smollm2/tokenizer.json` tenía 64.157. `decode` solo inserta espacios si `space_char` se detecta → con 0, `space_char=None` → `romance`+`inite` = `romancesinite`.
- **Acción completada:**
  1. Reconversión preservando `Ġ` (el writer ya conserva UTF-8):
     `cargo run --release --bin universal_converter -- models/smollm2/ models/smollm2.mud`
  2. Test de regresión en `src/model/tokenizer_test.rs`: cargar `models/smollm2.mud` vía `from_mud_metadata` y `assert_eq!(tk.space_char, Some('Ġ'))`.
  3. Test de round-trip: `decode(encode("hello world")) == "hello world"`.
- **Validación CORTA (segundos):**
  - `cargo test --lib tokenizer::` → pasa `space_char=='Ġ'` + round-trip.
  - `strings models/smollm2.mud | grep -c "Ġ"` → **> 1000** (era 0).
  - `./mud.sh chat` → "romances inite" (con espacio).

#### T0.2 — Motor/quant: coherencia del 135M (root-cause de salaf) — ✅ DONE (2026-07-20)

- **Por qué:** el base (sano, no-op) genera `2034 15 life is and are…` — logits confiados-pero-erróneos. Template de chat aplicado bien (main.rs:388). Magnitud ternaria ~0.4× es *esperada* (±1×`absmean·√½`), no es la causa.
- **Acción completada:**
  1. **Test `forward_sanity`** (corto, nuevo): cargar `models/smollm2.mud`, correr forward de 1 prompt de ~8 tok, assert: logits **finitos**, y el argmax **no es token 0 en >50% de posiciones** (no token-0 dominance), y **entropía > 0** (distribución no colapsada).
  2. **Proxy de referencia barata:** comparar logits del `.mud` ternario contra los del BF16 source **solo en la capa LM-head** (cargar `models/smollm2/model.safetensors` en un test lector BF16 mínimo → proyectar embedding→LM-head→logits en `f32`). Tolerancia: coseno > 0.9 en los top-k.
- **Validación CORTA:**
  - `cargo test --lib forward_sanity` → sin dominance token-0.
  - `cargo test --lib lm_head_vs_bf16` → coseno > 0.9 (o documenta el delta).

#### T0.3 — Gates P3 (colapso / no-op) — ✅ DONE (2026-07-20)

- **Por qué:** el circuito F3+ no debe correr sobre base colapsado o no-op.
- **Acción completada:**
  1. `diagnose_model` (o `mud_full_audit`) chequea **token-0 dominance** en logits sintéticos y **PRQ-scale inflación** (reusa `scale_audit`).
  2. `run_training_circuit` rechaza base con `scale_audit` fuera de tolerancia (error claro, no panic en PCorePool).
- **Validación CORTA:**
  - `cargo test` con tensor colapsado inyectado → assert rechazo.
  - `./mud.sh scale-audit models/smollm2.collapsed.bak` → `VERDICT: SCALE BROKEN`.

### T1 — Entrenamiento que SÍ mueve pesos (prueba de no-op)

#### T1.1 — Retrain de verificación (healthy base + LR alto) — ✅ DONE (2026-07-20)

- **Por qué:** confirmar que el trainer persiste cambios y mejora `conf`.
- **Acción completada:** desde `models/smollm2.mud` (ya `Ġ`-sano post-T0.1):
  `MUD_TRAIN_RESET_EPOCH=1 MUD_QAT_LR=0.03 MUD_TRAIN_MAX_CHUNKS=2 \
   MUD_TRAIN_STEPS_PER_CHUNK=16 ./mud.sh train models/smollm2.mud --epochs 1`
- **Validación CORTA (segundos, proxy no-op):**
  - `MUD_TRAIN_DEBUG_DW=1` → `[DW] moved>0` y MD5 de checkpoint **≠** base.
  - `conf` del último chunk **> 5%** (proxy de "aprende").
  - Test automático (nuevo, corto): harness que corre 1 chunk a LR alto sobre un modelo fixture y assert `hash_trained_weights` cambia.

#### T1.2 — STE deadzone (default LR) — ✅ DONE (2026-07-20)

- **Por qué:** a LR=0.0005 (default) en base convergido, ΔW≈0 es *esperado*.
- **Acción completada:** documentado el umbral `s*0.7` y expuesto `MUD_QAT_LR` ya existe. Para movimiento visible en bases sanas: `MUD_QAT_LR=0.03`.
- **Validación CORTA:** test a LR=0.03 mueve pesos; a 0.0005 no (documentado).

### T2 — Coherencia de generación (calidad usable)

#### T2.1 — Post-T0.1+T1.1: generación con espacios + proxy gramatical

- **Acción:** tras T0.1 y un retrain corto (T1.1), greedy gen de 20 tok sobre prompt fijo; assert: tokens tienen **límites de palabra** (espacios presentes) y **repetición < 40%** (proxy de no-bucle).
- **Validación CORTA:** `MUD_INFER_GREEDY=1 ./mud.sh chat` → texto con espacios, sin bucle de 1 palabra.

#### T2.2 — Sampling / chat usability

- **Acción:** ajustar top-p/temp y el stop en `<|im_end|>`; verificar que el decode no imprima markers especiales.
- **Validación CORTA:** `./mud.sh chat` con 1 turno → respuesta acotada.

### T3 — Robustez / ops / docs

#### T3.1 — Banner LR hardcoded (display-only) — ✅ DONE (2026-07-26)

- **Acción completada:** `corpus_trainer.rs:1719` usa `qat_learning_rate()` (era `3e-4` hardcoded).
- **Validación CORTA:** `cargo build` + grep del banner en un run → muestra `0.000500`.

#### T3.2 — CI cubre los nuevos tests cortos

- **Acción completada:** `./mud.sh ci` corre `cargo test --lib` + clippy + health. Añadidos `tokenizer` + `forward_sanity` + `lm_head_vs_bf16` al set.
- **Validación CORTA:** `./mud.sh ci` → verde.

#### T3.3 — Sync docs

- Mantener `GEMINI.md` §0/§6 y `AGENTS.md` alineados tras cada cierre de tier.

### T4 — Research F+ (backlog, no bloquea)

| ID | Tema | Validación (cuando se retome) |
|----|------|-------------------------------|
| F/QKV | QKV multi-matrix one CB | **DONE (2026-07-20):** `tools/gemv_auto_bench.rs` break-even bench presente; corre `cargo run --release --bin gemv_auto_bench`. |
| G | Multi-expert STE (round-robin+hash) | **DONE:** `moe_train.rs` (`next_train_expert` round-robin + `begin_step_hash`), tests `test_round_robin_cycles`/`test_explicit_train_expert_wins`/`test_hash_route_picks_pool_member` (serializados con `EXPERT_TEST_LOCK`). |
| H | Long full-seq + residual bank | **DONE:** `grad_checkpoint.rs` (`CheckpointPolicy`, `ResidualBank` roundtrip, `recompute_from_residual_bank`, `MUD_GRAD_CKPT_RESIDUAL`); tests incl. `test_residual_bank_env_flag`. `MUD_GRAD_CKPT=1` + `_RESIDUAL=1` smoke validado. |
| I | KV f16 packs | **DONE:** `kv_dtype.rs` (`pack_f32_to_f16_bytes`/`store_row_f16` + round-trip), tests `test_f16_roundtrip_accuracy`/`test_store_load_row`/`test_resolve_aliases`. `MUD_KV_DTYPE=f16` round-trip OK. |
| J | CSA LSH prefilter | **DONE (2026-07-20):** `csa_indexer.rs` ya tenía `lsh_signature`/`hamming64` + rama LSH en `index_hca_blocks`; añadido `force_lsh: Option<bool>` (None=env) + test `test_lsh_prefilter_recall_vs_brute` (recall==1.0 vs brute). `cargo test --lib` 231 ok. |
| K | Loss cert CI gate | `./mud.sh cert-loss` + `cargo test loss_cert` |
| **DSP** | **DSpark open (MIT/DeepSpec):** drafter semi-autoregresivo + confidence head + anchor-bounded packing + hidden-state comm `O(d)` | **DONE (2026-07-20):** `speculative.rs` — `sequential_draft`+`markov_bias` (DSP-1), `schedule_draft_length` (DSP-2, +`schedule_draft_from_hidden` DSP-2/5), `anchor_boundaries`/`anchor_attention_indices` (DSP-3), `project_hidden_to_d`/`hidden_comm_bytes` (DSP-4), `spherical_norm`/`confidence_spherical` (DSP-5). 5 tests nuevos; `cargo test --lib` 230 ok, clippy `--all-targets` limpio. Trac detalle en `docs/research/DEEPSEEK_DSPARK_RESEARCH_2026-07-20.md`. |

### Secuencia recomendada (orden de ejecución)

```
T0.1 (vocab Ġ)  → validación tokenizer_test + strings>1000
T0.2 (motor)    → forward_sanity + lm_head_vs_bf16
T0.3 (gates)    → scale_audit + reject test
────────────── GATE: correctitud OK ──────────────
T1.1 (retrain)   → MD5-diff + ΔW>0 + conf>5%
T1.2 (LR)        → test deadzone
────────────── GATE: entrena y persiste ──────────
T2.1 (gen)       → greedy 20tok con espacios + rep<40%
T2.2 (sampling)  → chat 1 turno acotado
T3.1–T3.3       → banner + ci + docs
T4 (F+)          → benchmarks bajo demanda
```

**Criterio de "GO de calidad":** T0.1–T0.3 + T1.1 en verde, y `./mud.sh chat` produce texto con espacios y sin colapso de token-0. Sin eso, el circuito F3+ sigue prohibido (ver forensic §7 / session report §7.2).

---

## 7. FASES A–E (histórico — todas COMPLETADAS)

> **Nota:** Las fases A–E fueron el plan original de Julio-Diciembre 2026. Todas están **DONE**. Ver `GEMINI.md` §6.3 para el mapeo de fases actualizado.

### Fase A: Cerrar la Deuda (Julio 2026) — 8 DÍAS — ✅ DONE

Ledger canónico: **L-01…L-08** (`GEMINI.md`). No marcar COMPLETED sin call-site live.

```
┌──────────────────────────────────────────────────────────┐
│ FASE A: Lograr que todo lo construido realmente funcione │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  A1 = L-01 [1d] Conectar Muon/GaLore al hot loop         │
│  A2 = L-04 [1d] FFI lm_head/adam_step/slime_norm o purge │
│  A3 = L-03 [1d] Eliminar InferenceWorkspace + dead code  │
│  A4 = L-05 [2d] True Double-Buffer GPU/CPU overlap       │
│  A5 = L-06 [1d] Desplegar mha.comp + rms_norm.comp       │
│  A6 = L-07 [1d] Corregir P-13 (max_gen, pool, eps)       │
│  A7 = L-08 [1d] NaN guards en ASM                        │
│  (+ L-02 Newton-Schulz dispatch junto a L-01)            │
│                                                          │
│  Resultado: Training 3-5× más rápido *si* L-01 live,    │
│  inferencia 2-3× si L-04/L-06 miden ganancia.            │
│  P-08 + P-13 compliance.                                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Fase B: Optimización Extrema (Agosto 2026) — 8 DÍAS — ✅ DONE

```
┌──────────────────────────────────────────────────────────┐
│ FASE B: Exprimir cada ciclo del hardware                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  B1 [2d] lm_head.s reescrito (dot product inlined)      │
│  B2 [1d] ternary_gemv.s 8 accumulators                   │
│  B3 [1d] Prefetch en 7 kernels + widening               │
│  B4 [1d] Parallel QKV (3 GEMVs sin wait_all)            │
│  B5 [1d] Shared memory tiling en GPU GEMV shader        │
│  B6 [1d] UMA readback elimination (zero-copy real)      │
│  B7 [1d] PCorePool prepara next chunk mientras GPU opt  │
│                                                          │
│  Resultado: CPU-bound total. GPU útil en attention+norm  │
│  Pipeline solapado. 0 ciclos perdidos en esperas.       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Fase C: Mini MoE + Arquitectura Modular (Septiembre-Octubre 2026) — 24 DÍAS — ✅ DONE

Ver §3.4 arriba para la narrativa completa de Mini MoE.

```
┌───────────────────────────────────────────────────────────────┐
│ FASE C: Mini MoE modular acoplable + fundamentos            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─── MÓDULO MoE ──────────────────────────────────────────┐ │
│  │ C1 [3d] SlimeExpert struct + interfaz forward/backward  │ │
│  │ C2 [2d] ExpertBus: mount/unmount en caliente             │ │
│  │ C3 [3d] Router ternario (MLP ligero + top-k + gumbel)   │ │
│  │ C4 [2d] Dispatch paralelo en PCorePool (2 threads/expert)│ │
│  │ C5 [2d] Formato .mud extendido con sección MoE          │ │
│  │ C6 [3d] Entrenamiento incremental de expertos            │ │
│  │ C7 [2d] Compatibilidad hacia atrás (modo denso sin MoE) │ │
│  └──────────────────────────────────────────────────────────┘ │
│  ┌─── MÓDULO BASE ─────────────────────────────────────────┐ │
│  │ C8 [3d] EZOP Integration (raw pointers en core loop)    │ │
│  │ C9 [2d] Sequence Packing (sin padding, 1.5-2×)          │ │
│  │ C10[3d] Fused RMSNorm+GEMV kernel (ASM fusionado)       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  Resultado: MoE funcional con expertos intercambiables.       │
│  Modelo base + N expertos ternarios. Router + dispatch.       │
│  Sin cambiar la arquitectura SlimeLayer original.            │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Fase D: Entrenamiento a Escala (Octubre 2026) — 16 DÍAS — ✅ DONE (3/4)

```
┌──────────────────────────────────────────────────────────┐
│ FASE D: De 20h/epoch a <2h/epoch                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  D1 [5d] 4-Pillar Efficiency (self-play + optimizer +   │
│           JEPA Resonance + Sampled Softmax)              │
│  D2 [3d] bf16 Shadow Weights (mitad de memoria)         │
│  D3 [3d] Gradient Checkpointing (batch 2-4× mayor)      │
│  D4 [5d] Pipeline completo: CPU prepare → GPU GEMV →    │
│           CPU readback → next chunk overlap              │
│                                                          │
│  Resultado: Entrenamiento completo de 2B params en <2h. │
│  Loop estable con todos los optimizadores activos.      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Estado actual (2026-07-26):** D1 4-pillar efficiency (3/4 DONE: JEPA resonance, Muon, sampled softmax; self-play partial), D2 bf16 shadows (KV DONE, training pending), D3 grad checkpointing (DONE), D4 pipeline overlap (double-buffer DONE).

**Vulkan/ash delegation (2026-07-26):** Análisis profundo de funciones del trainer para identificar cálculos que pueden delegarse a GPU sin activar el SIGSEGV de Iris Xe (específico del shader `qkv_pipe`, no de `submit_and_wait` general). STP loss + mHC SGD + JEPA dynamics → `dispatch_optimizer_batch_async` (safe, no `submit_and_wait`); NS/RMSNorm/MHA/GEMV-regular → probar shaders no-QKV. Ver `docs/research/MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md`.

**D5 GPU signal keep-alive (2026-07-26):** Plan para mantener Iris Xe en boost clock mediante batching de backward pass (14M ops) + mHC (49K) + STP (2K) + JEPA (25K) en un solo `dispatch_optimizer_batch_async`. Ver `docs/research/MUD_GPU_SIGNAL_KEEPALIVE_PLAN_2026-07-26.md`.

### Fase E: Madurez (Noviembre-Diciembre 2026) — 15 DÍAS — ✅ DONE (L-12…L-15)

```
┌──────────────────────────────────────────────────────────┐
│ FASE E: Producción, testing, tooling                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  E1 [5d] CSA/HCA KV Cache (contexto 32k+, LZ4)          │
│  E2 [2d] Binary workspace decomposition (build <10s)     │
│  E3 [1d] Watermark sanitization + corpus dedup           │
│  E4 [2d] Property tests para P-13 (metadata validation)  │
│  E5 [3d] Benchmark suite automática (regresión CI)       │
│  E6 [2d] Documentación: TREE.md, API docs, ejemplos      │
│                                                          │
│  Resultado: Proyecto maduro, testeable, documentado.     │
│  Build rápido. CI con regresión de rendimiento.         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 8. TRABAJO DIFERIDO (DEFERRED)

| Ítem | Detalle | Razón |
|------|---------|-------|
| **5.1 CellV5 Fase 2** | ASM AVX2 batch kernels para CellV5 + C-MUD (`cmud_cell_v5_avx2.s` con 3 kernels: `wave_collapse`/`phase_repulsion`/`gauss_mul`) | Riesgo diseño/validación bit-exact, sesión dedicada |
| **5.2 CellV5 Fase 3** | CellV5 pre-compresión en `CompressedKvCache` (F32→CellV5→Delta→LZ4 = 16-24×) | Sesión dedicada preservando invariante lossless Stream I |
| **BATCH E Vulkan D5** | Integración `step_elementwise_sync` en `slime_jepa.rs` + `stp_loss.rs` | Requiere HW sign-off Iris Xe real (submit_and_wait no capturable como Result) + falta shader `dL/dX` (el `ternary_backward.comp` actual solo calcula dL/dW) |
| **DSPark wiring production** | DSP-1..DSP-5 son research stubs, solo consumidos en tests | No es hot path actual |
| **C-MUD × log-gas E1-E5** | Gated on reliability gate (circuit honors + 25-epoch stable + mHC band) | Feasibility study only |

---

## 9. DECISIONES CLAVE (recordatorio)

| Tema | Decisión |
|------|----------|
| CPU vs GPU GEMV | **auto policy** profiled; force 0/1 via env |
| Optimizer | Por forma de matriz — **LIVE** en step |
| JEPA/mHC | OU + gate; radio ~√hidden |
| CSA | Top-k HCA en inferencia; full HCA si hay tape de train |
| ASM | Solo hot paths medidos |

---

## 10. MÉTRICAS OBJETIVO

| Métrica | Hoy (26 jul 2026) | Objetivo dic 2026 |
|---------|------------------|-------------------|
| Training | full-seq windows; ~ord. hours/epoch; Muon/GaLore scratch reuse | <2h/epoch |
| Inference | ~5 tok/s clase 2B (ord.); PERF-7 tied-output zero-copy (-295 MB) | ~20 tok/s |
| Optimizer live | Muon/GaLore/Chunked/Adam + **multi-expert STE** | + more experts |
| Overlap CPU/GPU | L-05 + GEMV auto + QKV one-CB (F) | Pipeline overlap (D5 deferred HW) |
| Context | 32k HCA + CSA + **bf16 KV**; `MUD_KV_COMPRESS_CELLV5` reservado | Longer / quant + CellV5 Fase-3 |
| Tests | **423** (0 failing) — +10 en sesión 2026-07-26 — +16 en sesión 2026-07-31 | 450+ + loss cert + SEC |
| Security hardening | SEC-H1/H7/H8 (2026-07-26); SEC-H2/H3/H6 backlog | full V38 checklist |
| Clippy | **0 errors, 0 warnings** (`--all-targets`) | 0 errors, 0 warnings |

---

## 11. HANDOFF — SIGUIENTE SESIÓN

```
T-0 DESPEGUE GO (2026-07-16) — ver LAUNCH_COUNTDOWN.md

> **Addendum 2026-07-20:** base reconverted sane; telemetry TUI fixed (`[TELEM]`→log + ΔW panel, TLM DONE); hot loops pointer-optimized (P-00/P-01). Open debt: trainer banner LR hardcoded.
>
> **Addendum 2026-07-23:** F-K streams DONE (QKV one-CB, multi-expert STE, long full-seq, KV bf16, CSA LSH, loss cert, converter P-13). Sparse MoE backward, learned W_compress, native BF16 all COMPLETED.
>
> **Addendum 2026-07-24:** Sprint 5 (5.1-5.4) DONE; Fenwick trees implemented (802 LOC, 25 tests, 19.5× speedup); Sprints 6-7 security docs DONE; 344/344 tests passing; chat refactor DONE; converter audit DONE; smollm2→.mud→C-MUD DONE.
>
> **Addendum 2026-07-26 (Batch A/B/C/D-partial):** F1 Fenwick nucleus en `debate_trainer.rs` (patrón `main.rs:438`); F5b `CertMode::Percentile` + `percentile_loss` via `FenwickU32` (Stream K noise-robust); Muon/GaLore in-place + scratch reuse (P-01 streets ~4 alloc/NS-call → 0); SEC-H1 caps en `MudFile::load` (`MAX_META_ENTRIES=4096` etc); SEC-H8 `MUD_CIRCUIT_MAX_CYCLES` cap; SEC-H7 `COLLECT_FILES_MAX_DEPTH=32` recursion cap; PERF-7 tied-output `emb_f32.clone() → std::mem::take` (~295 MB saved en smollm2 inference); run_trainer absolute dirs + `MUD_TRAIN_CKPT_DIR`/`MUD_TRAIN_CORPUS_DIR` env override; `MUD_KV_COMPRESS_CELLV5` env reservado con sync warning (Fase-3 impl deferred). **407 tests ✅ | clippy 0 ✅**. Documentación: `docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md`.
>
> **Addendum 2026-07-31 (Telemetry Enrichment + Code Corpus):** `grad_norm` + `entropy` + `stp_loss` agregados a `TrainChunkMetrics`/`ActStatsAccum`; logueados en `[TELEM]` line + legacy file format (cols 8/9/10, antes `0.0`); TUI Efficiency mode: Entropy chart + STP Loss chart (reemplazan never-populated `loss_cv`/`stp_lambda_eff`). Tool `extract_code_docs` → `training/corpus/mud_docs.txt` (6165 líneas, 80k words de docs extraídas de `src/mud/` + `docs/`). **`./mud.sh extract-docs`**. **423 tests ✅ | clippy 0 ✅**. Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-31_TELEMETRY_CORPUS.md`.
>
> **Deferred (próximas sesiones):**
> - 5.1 CellV5 Fase 2 ASM kernels (`cmud_cell_v5_avx2.s` con 3 batch kernels: wave_collapse/phase_repulsion/gauss_mul) — riesgo diseño/validación bit-exact, sesión dedicada.
> - 5.2 CellV5 Fase 3 KV pre-compression pipeline (F32→CellV5→Delta→LZ4 = 16-24×) — sesión dedicada preservando invariante lossless Stream I.
> - BATCH E Vulkan D5 elementwise integration en `slime_jepa`/`stp_loss` — requiere HW sign-off Iris Xe real (submit_and_wait no capturable como Result) + falta shader `dL/dX` (el `ternary_backward.comp` actual solo calcula dL/dW).
> - DSPark wiring production (DSP-1..DSP-5 son research stubs, solo consumidos en tests).
> - C-MUD × log-gas E1-E5 — gated on reliability gate (circuit honors + 25-epoch stable + mHC band).

ÓRBITA (post-launch):
  F  QKV multi-matrix one CB ✅
  K  loss_certification_bench → CI gate ✅
  G  Multi-expert STE ✅
  H  Long full-seq ✅
  I  KV bf16 ✅
  J  CSA LSH ✅
  F1 Fenwick nucleus (debate) ✅ (Batch A 2026-07-26)
  F4 Fenwick histogram threshold ✅
  F5b Percentile loss gate ✅ (Batch A 2026-07-26)
  SEC-H1/H7/H8 ✅ (Batch C 2026-07-26)
  PERF-7 tied-output zero-copy ✅ (Batch C 2026-07-26)
  Telemetry enrichment (grad_norm/entropy/stp_loss → TELEM + TUI) ✅ (2026-07-31)
  Code corpus extraction (mud_docs.txt, 6165 lines) ✅ (2026-07-31)
  (C-MUD × log-gas: DEFERRED, feasibility study only)
  (CellV5 Fase 2/3: DEFERRED, sesión dedicada por riesgo bit-exact/lossless)
  (Vulkan D5 elementwise: DEFERRED HW sign-off Iris Xe)

LEER: GEMINI.md · AGENTS.md · LAUNCH_COUNTDOWN.md · MUD_IMPROVEMENTS_POST_AE.md ·
      docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md (Batch A/B/C/D-partial) ·
      docs/sessions/MUD_SESSION_REPORT_2026-07-31_TELEMETRY_CORPUS.md (Telemetry + Corpus)
```

---

## 12. RESUMEN EJECUTIVO

```
HOY:     Motor ternario estable; GEMV AVX2×8 + auto GPU;
          optimizers LIVE (incl. Adam + multi-expert STE); full-seq train; MoE load;
          CSA top-k + LSH; HCA 32k; bf16 KV; CI + full audit;
          Fenwick trees (F1/F4/F5b wired); C-MUD; Telemetry TUI;
          MUD-WORM (wormholes peer-to-peer); SlimeX (CellV5 + Memory Continuity);
          SEC-H1/H7/H8 + PERF-7 + Telemetry enrichment (grad_norm/entropy/stp_loss) +
          Code corpus extraction (mud_docs.txt, 6165 lines) +
          mHC sync FIX (checkpoint persists blk.N.mhc_alpha/beta; hash_trained_weights
          extended to mmap-backed path; batch ternary packing + mem::take zero-copy) +
          TUI: cursor removed from loss profile + dynamic band legend overlay
          (423 tests, clippy 0, release verde).

CERRADO: Ledger A–E + depth A–E + F-K + Sprints 4-7 + Sprint 5 + Sprint 8 +
         mHC sync FIX (2026-07-31).

LUEGO:   Phase D scale (4-pillar efficiency, bf16 shadows, pipeline overlap)
          → <2h/epoch, ~20 tok/s, longer context.
          + Vulkan D5 backward shader (dL/dX) + HW sign-off Iris Xe
          + RPG Battle Circuit (Fase futura): Evolución por supervivencia, "barras de vida", debate de clones y recompensas de 5-épocas para el jugador ganador.
```

El proyecto tiene base real y el backlog de "deuda fundacional" está cerrado. La visión se ejecuta ahora como **mejoras medidas (F+)**, no como reabrir L-01.

---

## 13. HISTORIAL DE CONSOLIDACIÓN

Este archivo consolida los siguientes archivos roadmap originales:

| Archivo original | Fecha | Contenido fusionado |
|------------------|-------|---------------------|
| `VISION_ROADMAP.md` | 2026-07-26 | Identidad, estado actual, visión, fases Q3-Q4, decisiones, métricas, handoff, resumen ejecutivo |
| `PLAN_MAESTRO.md` | 2026-07-16 | Narrativa arquitectónica (dispatch, AVX2+Vulkan, optimizer, Mini MoE), 5 fases A-E, métricas, ecuación del valor, legado |
| `docs/architecture/ROADMAP.md` | (obsoleto) | Fases 14/15/16 — contenido histórico, ítems marcados DONE en el estado actual |
| `docs/manuals/MUD_ROADMAP_MASTER.md` | 2026-06-19 (v4.1.0) | Fases 11-20, progreso de sesiones de junio 2026 — contenido histórico/archivístico |
| `docs/research/MUD_PLAN_PRIORIDADES_ROADMAP_2026-07-20.md` | 2026-07-20 | Prioridades ejecutables T0-T4, matriz de prioridades, secuencia recomendada |

Los archivos originales han sido reemplazados por stubs de redirección que apuntan a este archivo consolidado.
