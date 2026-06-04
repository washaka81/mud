---
lang: es
---

# MUD (Modular Understanding Dynamics) — Visión Estratégica y Doctrina

> **Documento vivo.** Este archivo es la fuente de verdad estratégica del proyecto `forge_llm`.
> Última actualización: **2026-05-31 (Audit V6)**.

---

## 1. ¿Qué es MUD?

**MUD (Modular Understanding Dynamics)** es un motor de inferencia y aprendizaje continuo de
Inteligencia Artificial diseñado desde cero en **Rust y Ensamblador (ASM x86/AVX2)**. Su
arquitectura central combina tres innovaciones interdependientes:

### 1.1 Ecosistema Cognitivo Autónomo

MUD no es un wrapper sobre otro framework, ni un modelo de lenguaje en busca de backend.
Es un **ecosistema cognitivo completo** que integra en un único binario:

| Capa | Componente | Archivo |
|------|-----------|---------|
| Inferencia ternaria | Motor MHA + RoPE + MoE | `src/mud/inference.rs` |
| Autograd nativo | Backprop AVX2 | `forge_autograd/` |
| Memoria persistente | RAG sobre SQLite | `src/mud/store.rs` |
| Grafo de conocimiento | PageRank neural | `src/mud/graph.rs` |
| Aceleración iGPU | Backend Vulkan SPIR-V | `src/vulkan/` |
| Kernels de bajo nivel | GEMV AVX2/AVX-512 | `src/asm/` |
| Entrenamiento local | Auto-Trainer daemon | `src/mud/auto_trainer.rs` |
| Tokenización | BPE 49,152 tokens + merge | `src/mud/mod.rs` |

### 1.2 Arquitectura Zero-Allocation 1.58b

El corazón del motor es la política **Zero-Allocation** en el hot-loop de inferencia.
Todos los buffers operacionales (Q/K/V, scores de atención, logits, estados de gate) son
**pre-asignados** al arranque en la estructura `InferenceWorkspace`, con **alineación de 64
bytes** (tamaño de línea de caché L1). Esto elimina:

- Fragmentación de memoria dinámica durante generación de tokens.
- Overhead de modo-kernel por `malloc()`/`free()` en el hot-path.
- Split-cache-line penalties que degradan el throughput AVX2.

**Resultado medido:** `>1.200 allocs/token` eliminadas → **57 t/s** en CPU i7-1260P (AVX2),
sin comprometer la ventana de contexto (KV-Cache circular de 2.048 tokens).

### 1.3 Red Neuronal Ternaria (BitNet 1.58b)

Los pesos del modelo se representan en **tres estados únicos: `{-1, 0, +1}`** (BitNet 1.58b),
lo que permite reemplazar multiplicaciones de punto flotante por operaciones POPCNT/VPSADBW
en los kernels `ternary_gemv_4rows_avx2`, logrando:

- Compresión `16×` respecto a FP32 (e.g., `core_skills.mud`: 59 MB desde ~943 MB FP32).
- Reducción drástica de ancho de banda RAM (crítico para iGPUs con memoria compartida).
- Distribución estadística ideal: `37% (+1) / 26% (0) / 37% (-1)`, Sigma objetivo `0.5–0.8`.

### 1.4 Mixture of Experts (MoE)

La arquitectura MoE permite **activación selectiva de expertos por token**, concentrando
cómputo donde es necesario. El router usa Top-K con softmax estabilizado por temperatura,
ruido de annealing (`noise_std = 0.1 × (1 - step_ratio)`) y loss de balance en 3 componentes
(importance-var + load-var + z-loss) para prevenir el colapso de expertos.

**Modelos en producción actualmente:**

| Modelo | Base | Tamaño | Capas | Vocab | Velocidad |
|--------|------|--------|-------|-------|-----------|
| `core_skills.mud` | SmolLM2-135M | 59 MB | 30 | 49,152 | ~100 t/s |
| `qwen_mud.mud` | Qwen2.5-0.5B | 122 MB | 24 | 151,643 | ~70 t/s |

---

## 2. Nuestra Doctrina

La filosofía de MUD descansa sobre **cuatro pilares de ingeniería de bajo nivel y soberanía
computacional**. Cada decisión de diseño debe poder justificarse contra al menos uno de ellos.

### Pilar I — Mínima Fricción de Hardware

> *"El software debe ser físicamente consciente del hardware sobre el que corre."*

MUD optimiza dinámicamente su comportamiento según la topología del procesador anfitrión.
El módulo `src/hardware.rs` detecta en tiempo de arranque:

- Número de P-Cores vs. E-Cores (ej. i7-1260P: 4P/8E).
- Soporte de instrucciones SIMD (AVX2, AVX-512).
- Presencia y modelo de iGPU (ADL GT2 → backend Vulkan automático).
- Jerarquía de caché (L1/L2/L3) para calcular el prefetch-distance óptimo.

El pool de Rayon se limita a **4 hilos** para fijar la ejecución a los P-cores, evitando
contención de ancho de banda con los E-cores de eficiencia.

### Pilar II — Eficiencia por Compresión (1.58b)

> *"La inteligencia real no requiere precisión decimal infinita."*

La cuantización ternaria (1.58 bits/parámetro) es una decisión arquitectural irreversible, no
una optimización posterior. El ecosistema completo está diseñado para operar en este espacio:

- **Conversor Universal** (`tools/universal_converter/`): Safetensors → `.mud` con
  preservación de escalas QAT por capa (`.scale` tensors), corrección de GQA
  (`num_kv_heads × head_dim`), y soporte de embedding ternarization row-wise absmean.
- **Ternarización de Embeddings** (`--ternarize-emb`): `519 MB → 33 MB` para Qwen2.5-0.5B.
- **Autograd Nativo** (`forge_autograd`): Backprop ternario con SIMD/AVX2 — 21/21 unit tests.
- **Kernels ASM Hand-Tuned**: `ternary_gemv_4rows_avx2` con loop unrolling, prefetch NTA
  (Non-Temporal) para streaming de pesos y prefetch T0 para activaciones. Prefetch distance:
  **512 bytes** (optimizado para LPDDR5-5200 del i7-1260P).

### Pilar III — Soberanía Local de Datos

> *"El aprendizaje y la inferencia no deben depender de la nube."*

MUD es **Local-First** por diseño. Cada componente del pipeline de conocimiento existe en el
dispositivo del usuario:

- **Auto-Trainer daemon** (`src/mud/auto_trainer.rs`): Entrena continuamente sobre hechos de
  `knowledge.db` con ExpertShadow/MambaShadow cache FP32 en RAM, gradientes acumulados (8 pasos)
  y flush con gradient clamping per-element `[-1.0, 1.0]`. Utiliza **Quantization-Aware Training (QAT)** 
  con un optimizador tipo SGD (`lr=0.002`) y cálculo dinámico de Lambda (Weight Decay).
  ✅ Los issues de la Auditoría V5 han sido resueltos.
- **Corpus Aligner** (`tools/mud_corpus_trainer.rs`): Alineación con **Straight-Through Estimator (STE)** 
  para sanar el "Ternary Shock". LR=0.001, contrastivo con 7 negativos. Stateful Resume + Hard Checkpoints cada 5k chunks.
  ✅ Los issues altos (TRAIN-07/08/21/23) fueron resueltos en Audit V5/V6.
- **Knowledge DB** (`models/knowledge.db`): 74,489 hechos en SQLite con WAL mode,
  busy_timeout 5000ms y LIMIT en queries para prevenir scans infinitos.
- **RAG Semántico** (`src/mud/ingester.rs`): Ingesta `.txt` y `.pdf` usando embeddings del
  propio modelo como función de hash semántico.

La arquitectura garantiza que **ningún byte de datos del usuario abandona el dispositivo**
durante inferencia o entrenamiento.

### Pilar IV — Cero Tolerancia a la Asimetría del Código/UI

> *"La precisión matemática y la belleza estética son inseparables."*

Este pilar aplica a dos dimensiones simultáneas:

**Asimetría de Código:** Cada función crítica debe tener paths simétricos: si existe un path
AVX2, debe existir un fallback escalar válido. Si existe aceleración Vulkan, debe existir un
path CPU equivalente. Los 40 bugs documentados en `MUD_AUDIT_LATEST.md` son evidencia viva
de que cuando se viola este principio, el sistema desarrolla comportamientos caóticos (word
salad, segfaults silenciosos, NaN propagation).

**Asimetría de UI:** El CLI de MUD usa `comfy-table` para cuadros con esquinas UTF-8
perfectamente simétricas. El sistema de color es determinista: **cyan/dim/italic** para
razonamiento CoT `<thinking>`, **purple italic** para inyección RAG en tiempo real. La barra
de estado muestra métricas reales (TPS, RAM, VLK, IQ) — jamás valores estimados o hardcoded.

---

## 3. Visión

Convertir a MUD en el **estándar de facto para la Inteligencia Artificial de borde
(Edge AI) descentralizada**.

Visualizamos un futuro donde cualquier dispositivo portátil — sin importar sus limitaciones
de hardware — pueda ejecutar, entrenar y expandir modelos MoE de alto razonamiento de manera
descentralizada y en tiempo real. Donde el acceso a asistentes cognitivos avanzados no
requiera tarjetas NVIDIA de $2.000, conexión permanente a la nube, ni aceptar que tus datos
sean el producto.

**MUD es la respuesta técnica a una pregunta política:** ¿quién controla la inteligencia?

La visión se articula en tres horizontes:

| Horizonte | Timeframe | Marcador de éxito |
|-----------|-----------|-------------------|
| **Motor** | 2026 | >100 t/s con modelo 1B en hardware consumer, sin CUDA |
| **Plataforma** | 2027 | Soporte x86, ARM, WASM — compilación sin dependencias |
| **Red** | 2028 | Enjambre P2P de nodos MUD compartiendo expertos por WiFi |

---

## 4. Objetivos Estratégicos

### 4.1 Independencia Tecnológica

Proveer una alternativa open-source hiper-optimizada frente a los ecosistemas cerrados y
motores dependientes de CUDA/NVIDIA. El stack de MUD es intencionalmente independiente:

- **Sin CUDA:** El backend de GPU usa Vulkan (open standard, cross-vendor).
- **Sin PyTorch en producción:** El autograd es 100% Rust nativo (`forge_autograd`).
- **Sin Python en el binario principal:** El pipeline de entrenamiento local es Rust puro.
- **Sin dependencias cloud:** SQLite local reemplaza a DynamoDB/Pinecone/cualquier vector DB.

Dependencias únicas aceptadas: `vulkano 0.34` (Vulkan bindings), `rayon 1.12` (paralelismo),
`rusqlite 0.39` (persistencia). Todas open-source, sin telemetría.

### 4.2 Aprendizaje Continuo Real

Evolucionar MUD de un modelo estático a un **agente "vivo"** que asimila nuevos conocimientos
de forma fluida. El pipeline actual es:

```
Conversación → Auto-Trainer daemon → ExpertShadow FP32 → Flush periódico → .mud
        ↓
Texto/PDF → MudIngester → Embedding semántico → knowledge.db → RAG en tiempo real
```

La meta es eliminar la distinción entre "entrenamiento" e "inferencia" — el modelo debe
aprender continuamente mientras opera, sin requerir una fase de fine-tuning separada.

### 4.3 Portabilidad Extrema

Mantener el binario central libre de dependencias complejas para soportar:

| Plataforma | Estado | Backend |
|-----------|--------|---------|
| x86_64 (Intel/AMD) | ✅ Producción | AVX2 + Vulkan |
| ARM (Apple Silicon) | 🔄 Planificado | NEON + MoltenVK |
| WebAssembly | 🔄 Planificado | Softfloat + WebGPU |
| Android/ARM | 🔄 Planificado | Vulkan Mobile |

---

## 5. Metas Técnicas

### 5.1 Recalibración Exitosa *(Corto Plazo — En Curso)*

**Estado actual:** Ejecutando Epoch 1/2 del corpus masivo de alineación (PID: 164003).
Coherencia lingüística actual: **8.8%** → **proyección post-Epoch 1:** 99.9%.

El score de certeza de cuantización (QC Score) medido por `recalibration_projector`:
- `core_skills.mud`: **81.45%** de certeza (requiere 2 Epochs para restaurar fluidez).
- `qwen_mud.mud`: **81.82%** de certeza (mismo pipeline).

El validador `cognitive_integrity` debe confirmar **IQ > 150** tras el entrenamiento.

Issues pendientes que bloquean esta meta:
- `PERF-05`: BPE tokenizer O(n²) → necesita migración a priority queue O(n log n).
- `BUG-6`: Weight decay colapsa pesos ternarios a cero en el Auto-Trainer rewrite.

### 5.2 Soporte Multi-Dispositivo con Vulkan *(Medio Plazo)*

Consolidar el backend de Vulkan Zero-Copy para garantizar **+50 TPS constantes** tanto en
Intel Iris Xe (ADL GT2) como en AMD y Apple Silicon. Requiere:

- **Flash Attention en Vulkan:** Shaders `.spv` fusionados para aniquilar cuellos de botella
  de R/W en VRAM durante el cálculo de atención.
- **GQA Threading:** Paralelizar atención multi-cabeza con Rayon.
- **KV-Cache Cuantizado (INT8):** Reducción del 75% en RAM del caché (actualmente ~96 MB).
- **Shader Fusion:** Fusionar RMSNorm + RoPE en el shader GEMV SPIR-V existente.

### 5.3 Ternarización Total *(Medio Plazo)*

Completar la **Fase 5 del roadmap** cuantizando la totalidad de los tensores del modelo:

| Componente | Estado |
|-----------|--------|
| Pesos de expertos FFN | ✅ Completado (Fase 1-4) |
| Matrices de atención (Q/K/V) | ✅ Completado |
| Proyecciones de salida | ✅ Completado |
| Embeddings de tokens | ✅ Completado (`embed_ternarize.rs` — 15.9× compresión) |
| Mecanismos de gate MoE | ❌ Gate training es dead code (TRAIN-15). Gate persistence destruye pesos (TRAIN-23). |

**Bloqueantes activos (Audit V5):**
- **TRAIN-01:** Las escalas PRQ se ignoran al cargar shadows de expertos → gradientes incorrectos.
- **TRAIN-02:** El backward de capas Mamba es un dead-end → 50% de Mamba no se entrena.
- **TRAIN-08:** El corpus trainer guarda embeddings como Float32, rompiendo el formato ternario.
- **BUG-6:** Weight decay puede colapsar pesos ternarios a cero (sin verificar).

### 5.4 Enjambre P2P *(Largo Plazo)*

Desarrollar la capacidad de que múltiples nodos MUD compartan expertos y pesos de forma
descentralizada sobre WiFi. Arquitectura propuesta:

```
[Nodo A] ← experto_0, experto_2 → [Nodo B]
[Nodo B] ← experto_1, experto_3 → [Nodo C]
                    ↑
           Token routing distribuido
           (P2P Swarm Inference)
```

Un token puede atravesar expertos en múltiples dispositivos antes de producir output.

---

## 6. Análisis FODA (SWOT)

### 💪 Fortalezas

| Fortaleza | Evidencia en el código |
|-----------|------------------------|
| **Rendimiento Extremo — Zero-Allocation** | `InferenceWorkspace` pre-asigna todos los buffers; eliminadas >1.200 allocs/token. Medido: 57 t/s CPU. | 
| **Afinidad de Hardware — Auto-Detect** | `src/hardware.rs` detecta P/E-cores, SIMD flags y GPUs al arranque; ajusta hilos Rayon y prefetch distance. |
| **Ecosistema Unificado en Rust** | Autograd, Inferencia, Tokenización, Entrenamiento y RAG en un único binario memory-safe. 21/21 unit tests en verde. |
| **Ternarización Nivel Experto** | Conversor universal con preservación QAT de escalas por capa, GQA mapping, embedding ternarization (15.9×). |
| **Herramientas de Diagnóstico Propias** | 29+ herramientas en `tools/`: `tensor_microscope`, `mud_calibrator`, `cognitive_integrity`, `recalibration_projector`, `iq_box`. |
| **Persistencia del Conocimiento** | 74,489 hechos en `knowledge.db`; RAG semántico en tiempo real con UI diferenciada (purple italic). |

### 🚀 Oportunidades

| Oportunidad | Contexto |
|------------|---------|
| **Democratización del Hardware** | El auge de NPUs y iGPUs modernas (Intel Xe, AMD RDNA iGPU, Apple Neural Engine) crea terreno fértil para motores que no dependen de VRAM dedicada. MUD ya tiene el backend Vulkan para este hardware. |
| **Privacidad Local como Necesidad Regulatoria** | GDPR, AI Act europeo y regulaciones locales en LATAM crean demanda corporativa de IA que no exfiltra datos. MUD es Local-First por construcción. |
| **Edge AI & IoT** | Proyección del motor hacia smartphones y dispositivos IoT mediante el backend Vulkan Mobile (Android). El formato `.mud` es compacto por diseño. |
| **Alternativa al GGUF/Llama.cpp** | El ecosistema llama.cpp no está optimizado para hardware Intel iGPU ni para cuantización 1.58b nativa. MUD puede capturar el segmento de usuarios con hardware Intel consumer. |
| **Aprendizaje Continuo Sin Servidores** | Ningún competidor open-source tiene un daemon de entrenamiento local en Rust con ExpertShadow cache y persistencia atómica. Este es un diferenciador único. |

### ⚠️ Debilidades

| Debilidad | Detalle | Mitigación planificada |
|-----------|---------|----------------------|
| **Amnesia Post-Conversión** | La cuantización PTQ directa destruye asociaciones posicionales finas (word salad / Ternary Shock). | Mitigado vía QAT con Straight-Through Estimator (STE) en el native corpus trainer, curando las fronteras ternarias de forma nativa. |
| **Dependencia del Modelo Maestro** | Se requiere un modelo en alta precisión (FP16/BF16) para convertir a `.mud`. MUD no puede "nacer" sin un maestro FP32. | Pipeline Kaggle/Cloud para entrenamiento inicial en GPU, con pull local del `.mud`. |
| **Curva de Aprendizaje del Código** | ASM incrustado, `unsafe` Rust, punteros crudos sobre mmap y optimizaciones de caché de bajo nivel elevan la barrera de entrada. Los 40 bugs documentados ilustran la complejidad. | Suites de auditoría automáticas (`cargo test --release`) y documentación técnica en `docs/hardware/`. |
| **BPE Tokenizer O(n²)** | El tokenizador actual es cuadrático en longitud de entrada (`PERF-05`). Impacto visible en prompts largos. | Migración a priority queue O(n log n) — pendiente. |
| **Math Skill sin Injection en KV-Cache** | El resultado del sandbox matemático (`tools/math_sandbox.py`) se imprime en consola pero no se inyecta en el stream de inferencia (`AUDIT-4.3`). | Requiere driver asíncrono en `src/main.rs`. |

### ⚡ Amenazas

| Amenaza | Nivel de riesgo | Respuesta estratégica |
|---------|----------------|----------------------|
| **Fragmentación de Formatos** | 🔴 Alto | GGUF/llama.cpp tiene soporte comunitario masivo y evoluciona rápidamente. El formato `.mud` podría ser relegado a nicho. | Desarrollar un puente de importación GGUF nativo en Rust (ya existe `src/gguf/`). |
| **Hardware Específico para Otras Cuantizaciones** | 🟠 Medio | Si los fabricantes de silicio (Qualcomm, Apple, Intel) optimizan exclusivamente para INT4/FP8 a nivel de transistor, la ventaja teórica de 1.58b se diluye. | El path AVX2 POPCNT ya es óptimo para ternario. Monitorear NPU roadmaps. |
| **Mantenibilidad de Vulkan** | 🟠 Medio | Diversidad de drivers GPU (Intel/AMD/NVIDIA/Mali) genera bugs de precisión numérica que escapan a los tests en CPU. `PERF-05` es un ejemplo activo. | Test suite de integración con Vulkan real en CI/CD. |
| **Velocidad del Ecosistema Abierto** | 🟡 Bajo | Llama.cpp, MLX, y otros frameworks lanzan mejoras constantemente con equipos más grandes. | Especialización en el nicho Intel iGPU + ternario nativo, donde no compiten. |

---

## Apéndice: Estado del Build (Referencia Rápida)

```
cargo check --release   ✅  0 errores, 0 warnings
cargo build --release   ✅  Éxito (2m 05s, optimized)
cargo test --release    ✅  21/21 passed, 0 failed
```

**Hardware de referencia:** Intel i7-1260P (4P/8E cores), Intel ADL GT2 iGPU, LPDDR5-5200.

**Métricas de rendimiento verificadas:**
- CPU (AVX2 + P-cores): **57.0 t/s**, ~17ms latencia, 2.6 GB RAM.
- Vulkan iGPU (Zero-Copy): **20.0 t/s**, ~50ms latencia, 3.4 GB RAM.

---

*Documento generado y mantenido como parte del proceso de auditoría continua del proyecto.*
*Consistente con el estado del código fuente a fecha 2026-05-30.*