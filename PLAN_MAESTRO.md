# FORGE LLM — PLAN MAESTRO
## Motor de Inferencia y Entrenamiento Ternario Ultra-optimizado para CPU

*"El que domina los punteros domina el núcleo de la programación"* — P-00
*"Cada ciclo, cada byte, cada instrucción tiene que sudar. Nada sobra, nada se repite, todo se recicla en el hot path."* — P-00b

> **Jerarquía (2026-07-16):** Políticas + ledger **L-##** + estado técnico → [`GEMINI.md`](./GEMINI.md).  
> Visión corta + fases + handoff → [`VISION_ROADMAP.md`](./VISION_ROADMAP.md). Agentes → [`AGENTS.md`](./AGENTS.md).  
> Stack LIVE (ELUT/FP32/AVX2×8/ash) → [`docs/architecture/MUD_COMPUTE_STACK.md`](./docs/architecture/MUD_COMPUTE_STACK.md).  
> Gap + env knobs → [`docs/research/MUD_GAP_ANALYSIS_POST_L15.md`](./docs/research/MUD_GAP_ANALYSIS_POST_L15.md).  
> **Mejoras F+** → [`docs/research/MUD_IMPROVEMENTS_POST_AE.md`](./docs/research/MUD_IMPROVEMENTS_POST_AE.md).  
> Este archivo: narrativa de arquitectura, Mini MoE y métricas. Si hay conflicto de *estado*, gana **GEMINI**.  
> **Estado 2026-07-16:** L-01…L-15 + streams A–E **DONE**. Próximo: backlog **F+** (QKV one-CB, multi-expert STE, …), no reabrir L-01.
> **Addendum 2026-07-20:** base reconvertido sano; telemetry TUI fija (`[TELEM]`→log + ΔW panel, TLM DONE); hot loops pointer-opt (P-00/P-01). Deuda abierta: banner LR hardcoded.

---

## Ⅰ. LA MARAVILLA ARQUITECTÓNICA

### 1.1 El Dispatch Empaquetado FP32 Ultra-optimizado

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

### 1.2 AVX2 + Vulkan Ash: La Simbiosis

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

### 1.3 El Pipeline de Entrenamiento Adaptativo

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

### 1.4 Mini MoE Modular y Acoplable

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

#### Ejemplo de Uso

```bash
# Crear modelo base (MoE-ready):
cargo run --release --bin universal_converter -- base.safetensors base.mud --enable-moe

# Añadir experto de código:
cargo run --release --bin warp_aligner -- base.mud --train-expert 1 --corpus code.txt

# Añadir experto de matemáticas:
cargo run --release --bin warp_aligner -- base.mud --train-expert 2 --corpus math.txt

# Inferencia con 2 expertos activos:
cargo run --release --bin forge_llm -- base.mud --top-k 2

# Clonar experto popular para balanceo:
cargo run --release --bin forge_llm -- base.mud --clone-expert 1 --to-slot 7
```

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

## Ⅱ. LA VISIÓN FUTURISTA

### 2.1 LLM Local, de Bajo Coste, Entrenable

La visión de Forge LLM no es "correr modelos existentes más rápido". Es:

**Un LLM de 2B parámetros que cabe en 400MB (1.58-bit) y se entrena desde cero en un portátil de 800€ en horas, no días.**

Hoy:
- GPU datacenter (H100): $30,000+ para entrenar cualquier LLM
- CPU server (Xeon): consumo de 200W+, 64 cores
- RAM: 80GB+ para modelos densos de 7B

Con Forge LLM:
- Hardware: Intel i7-1260P (portátil de 2022, 800€)
- Peso del modelo: ~400MB (2B params × 1.58 bits)
- RAM necesaria: 8GB (cuando esté optimizado)
- Tiempo de entrenamiento: <2h/epoch (con Muon + overlap GPU)
- Consumo: 28W (TDP del i7-1260P)

**Esto no es teoría.** Cada kernel está escrito para este hardware específico. Cada optimización se mide contra el reloj. No hay "soporte para CUDA" ni "compatibilidad con AMD". Hay un objetivo: **hacer que un LLM ternario entrenable quepa en un portátil Intel estándar**.

### 2.2 ¿Para Qué Sirve un LLM Así?

**No para competir con GPT-4.** Para:

1. **Asistente de código offline** — Sin enviar código a la nube. Sin conexión. Sin censura de API.
2. **Modelo ajustable por el usuario** — Entrenas con tus propios datos en 30 minutos. El modelo aprende tu estilo, tu jerga, tu código.
3. **Dispositivos sin conexión** — Raspberry Pi 5, laptop vieja, tablet. Sin GPU, sin internet.
4. **Privacidad total** — El modelo nunca sale de tu máquina. Los pesos son tuyos.
5. **Costo marginal cero** — Una vez entrenado, inference cuesta solo electricidad.

### 2.3 El Salto a MoE y Complejidad

El plan maestro contempla:

**2026-Q4: Mini MoE Modular Acoplable**
- ExpertBus con mount/unmount en caliente (sin recompilar)
- 4-8 expertos ternarios de 500M params cada uno, entrenables por separado
- Router ternario ligero (<0.3% overhead) con top-k + gumbel noise
- Dispatch paralelo: 2 expertos × 4 threads cada uno en PCorePool
- El modelo base funciona sin MoE (compatibilidad total hacia atrás)
- Entrenamiento incremental: añadir experto de código sin tocar el resto

```
Ejemplo: Modelo base 2B + 4 expertos de dominio:
  expert.0 (base):  conocimiento general (siempre activo)
  expert.1 (código): Rust/Python/C (activado por router)
  expert.2 (mates):  razonamiento matemático
  expert.3 (es):     español/literatura
  expert.4 (en):     inglés/técnico
  
  Costo por token: 2B (base) + 2×500M (2 expertos activos) = 3B params efectivos
  Parámetros totales: 2B + 4×500M = 4B en disco (~800MB)
  vs. modelo denso de 3B: ~6GB en FP16 → **8× más compacto**
```

**2027: C-MUD (Complex Manifold)**
- SlimeComplexRegister: números complejos ternarios (Gauss Integers)
- C-SiLU: activación con fase angular
- Razonamiento latente sin generar texto intermedio
- El modelo "piensa" en el espacio complejo antes de emitir tokens

**2027-2028: Entrenamiento Federado**
- Varios portátiles entrenan el mismo modelo
- Cada uno ajusta en sus datos locales
- Sincronización vía gradientes comprimidos (ternarios)
- Sin servidor central

---

## Ⅲ. EL PLAN MAESTRO: 5 FASES

### Fase A: Cerrar la Deuda (Julio 2026) — 8 DÍAS

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

### Fase B: Optimización Extrema (Agosto 2026) — 8 DÍAS

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

### Fase C: Mini MoE + Arquitectura Modular (Septiembre-Octubre 2026) — 24 DÍAS

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

**Detalle del Módulo MoE:**

| Sub-fase | Entregable | Archivos |
|----------|-----------|----------|
| C1 | `SlimeExpert` con forward/backward ternario | `src/mud/slime_expert.rs` |
| C2 | `ExpertBus` con mount/unmount, hot swap | `src/mud/expert_bus.rs` |
| C3 | Router ternario con top-k + gumbel noise | `src/mud/moe_router.rs` |
| C4 | Dispatch de expertos en PCorePool | `src/mud/pcore_pool.rs` (ext) |
| C5 | Formato MUD v2 con sección MoE | `src/mud/mod.rs` (ext) |
| C6 | Trainer con --train-expert flag | `tools/warp_aligner.rs` |
| C7 | Fallback a FFN denso si no hay MoE | `slime_forward.rs` |

**Nuevos archivos en `src/mud/`:**
- `slime_expert.rs` (~200 loc) — Experto ternario individual
- `expert_bus.rs` (~300 loc) — Bus con mount/unmount + dispatch
- `moe_router.rs` (~150 loc) — Router ternario con top-k

**Extensión a existentes:**
- `pcore_pool.rs` — Añadir `ExpertScheduler` para dispatch por experto
- `mod.rs` — Parseo de tensores MoE en MUD v2
- `slime_forward.rs` — Punto de inyección del MoE (reemplaza FFN denso)
- `corpus_trainer.rs` — Entrenamiento de expertos individuales

**Total Fase C: 24 días hábiles (~5 semanas).**

### Fase D: Entrenamiento a Escala (Octubre 2026) — 16 DÍAS

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

### Fase E: Madurez (Noviembre-Diciembre 2026) — 15 DÍAS

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

## Ⅳ. MÉTRICAS OBJETIVO

| Métrica | Hoy (Julio 2026) | Objetivo (Diciembre 2026) |
|---------|-----------------|--------------------------|
| Training speed | ~20h/epoch | <2h/epoch |
| Inference speed | ~5 tok/s (2B) | ~20 tok/s (2B) |
| RAM usage | ~4GB | <2GB (bf16 shadows + sparse) |
| Model size | ~400MB (2B, 1.58-bit) | ~400MB (mismo, optimizado) |
| Context length | 4K tokens | 32K tokens (CSA/HCA) |
| Build time (clean) | ~45s | <10s (workspace split) |
| Tests | 89 | 200+ (property tests incluidos) |
| Cobertura ASM | 6/17 testeados | 14/14 testeados |
| Optimizers activos | 1 (SGD) | 5 (Muon/GaLore/Chunked/Sparse/Adam) |
| Overlap CPU/GPU | 0% (serial) | ~80% (double-buffer real) |

---

## Ⅴ. LA ECUACIÓN DEL VALOR

Forge LLM existe porque la ecuación actual de los LLMs está rota:

```
Valor = Capacidad / Coste × Accesibilidad

Hoy:     GPT-4 = Alta capacidad / Coste enorme × 0 accesibilidad (API)
         Llama  = Alta capacidad / Coste medio × Baja accesibilidad (GPU 24GB+)

Forge:   MUD v2 = Capacidad media / Coste casi cero × Accesibilidad total
         (cualquier portátil Intel, sin GPU, sin internet)
```

**El objetivo no es ganar benchmark de MMLU. Es democratizar el acceso a LLMs entrenables.**

---

## Ⅵ. EL LEGADO

Si Forge LLM logra su visión, habrá demostrado que:

1. **No necesitas una GPU de $30,000 para entrenar un LLM.** Un portátil de 800€ basta si optimizas cada bit.
2. **El software puede superar al hardware.** Con kernels manuales, dispatch adaptativo, y solapamiento CPU/GPU, se puede extraer 10× más rendimiento del mismo silicio.
3. **La privacidad no está reñida con la capacidad.** Un modelo de 2B ternario, ajustado a tus datos, puede ser más útil que GPT-4 genérico para tus tareas específicas.
4. **El futuro de la IA no es centralizado.** Es un modelo en cada dispositivo, entrenado con tus datos, funcionando sin conexión.

---

*"El que domina los punteros domina el núcleo de la programación"*
— P-00, Forge LLM
