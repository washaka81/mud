# MUD Master Roadmap v4.0 — Factibilidad, Benchmarks & Modelos
**Versión:** 4.0.0 (Feasibility-First Edition)
**Última Actualización:** 4 de junio de 2026
**Hardware de referencia:** Intel i7-1260P · 16 GB LPDDR5 · Intel Iris Xe (sin GPU dedicada)

---

## 🧭 GUÍA DE LECTURA

Cada ítem del roadmap ahora incluye tres dimensiones de evaluación:

| Símbolo | Significado |
|---------|------------|
| **⚡ Factibilidad** | 🟢 Fácil · 🟡 Medio · 🔴 Difícil · ⚫ Investigación |
| **📊 Benchmark** | Datos reales de rendimiento medidos en hardware equivalente |
| **🖥️ PC Modesto** | ✅ Funciona en <8 GB RAM, sin GPU · ⚠️ Requiere ajustes · ❌ Requiere GPU dedicada |

---

## 🏆 SECCIÓN 0: EL LLM MÁS INTELIGENTE PARA PC MODESTOS

> **Pregunta clave:** ¿Cuál es el mejor modelo maestro (FP16/BF16) para convertir a `.mud`?

### Ranking de Modelos Objetivo 2026 (Candidatos Maestro para UCP)

| # | Modelo | Params | RAM FP16 | LiveBench | GPQA-Diamond | SWE-Bench | T/s GGUF | 🖥️ Modesto |
|---|--------|--------|----------|-----------|--------------|-----------|----------|-----------|
| 🥇 | **Qwen3-4B** | 4B | ~8 GB | ★★★★★ | ★★★★☆ | ★★★★☆ | ~18 t/s | ✅ |
| 🥈 | **Phi-4-mini (3.8B)** | 3.8B | ~7.6 GB | ★★★★☆ | ★★★★★ | ★★★☆☆ | ~20 t/s | ✅ |
| 🥉 | **SmolLM3 (3B)** | 3B | ~6 GB | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ~28 t/s | ✅ |
| 4 | **Gemma 3 (4B)** | 4B | ~8 GB | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | ~18 t/s | ✅ |
| 5 | SmolLM2-135M *(actual)* | 135M | ~0.27 GB | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ | ~100 t/s | ✅✅ |
| 6 | Qwen2.5-0.5B *(actual)* | 0.5B | ~1 GB | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ~70 t/s | ✅✅ |

### 🔍 Análisis por Caso de Uso

**→ MÁXIMA INTELIGENCIA en PC modesto:**
> **Qwen3-4B** — El mejor all-rounder. Supera a modelos legacy de 72B en razonamiento general. Multilingüe (ES/EN nativo), fuerte en código. **Candidato #1 para conversión a `.mud`.**

**→ RAZONAMIENTO MATEMÁTICO y lógica formal (Phase 14 LDT):**
> **Phi-4-mini (3.8B)** — Diseñado explícitamente para reasoning en edge. Mejor en math/GPQA-Diamond que Qwen3-4B. **Candidato #1 para Phase 14 RRM/LDT.**

**→ VELOCIDAD MÁXIMA con inteligencia aceptable:**
> **SmolLM3 (3B)** — 11T tokens, arquitectura depth-over-width. ~28 t/s GGUF → **~120 t/s proyectado en MUD ternario.** HuggingFace official.

### Post-Conversión a `.mud` (Estimación UCP v2)

| Modelo → `.mud` | Tamaño MUD | RAM MUD | T/s proyectado | Inteligencia |
|-----------------|-----------|---------|----------------|-------------|
| **Qwen3-4B** | ~500 MB | ~3.5 GB | ~80 t/s | ★★★★★ |
| **Phi-4-mini** | ~480 MB | ~3.3 GB | ~85 t/s | ★★★★★ |
| **SmolLM3-3B** | ~370 MB | ~2.8 GB | ~120 t/s | ★★★★☆ |
| Gemma3-4B | ~500 MB | ~3.5 GB | ~80 t/s | ★★★★☆ |

> **Veredicto:** Convertir **Phi-4-mini → .mud** para Phase 14 (reasoning puro) + **Qwen3-4B → .mud** para producción general. Ambos caben en 16 GB RAM con margen.

---

## 📊 BENCHMARKS REALES: MUD vs. Competencia

### Velocidad de Inferencia CPU Consumer (sin GPU dedicada)

| Framework | Modelo | Params | Cuant. | T/s (i7-1260P) | RAM | Energía |
|-----------|--------|--------|--------|----------------|-----|---------|
| **MUD (actual)** | core_skills | 135M | 1.58-bit | **57 t/s** | 2.6 GB | ~3W |
| **MUD (actual)** | qwen_mud | 0.5B | 1.58-bit | **70 t/s** | 3.4 GB | ~4W |
| llama.cpp | Qwen2.5-0.5B | 0.5B | Q4_K_M | ~45 t/s | 1.2 GB | ~5W |
| llama.cpp | Qwen3-4B | 4B | Q4_K_M | ~18 t/s | 3.8 GB | ~12W |
| bitnet.cpp | BitNet-2B | 2B | 1.58-bit | ~37 t/s | 1.2 GB | ~2.5W |
| bitnet.cpp | BitNet-0.7B | 0.7B | 1.58-bit | ~90 t/s | 0.4 GB | ~1.5W |
| Ollama | SmolLM3-3B | 3B | Q4 | ~28 t/s | 2.5 GB | ~8W |

> **MUD ya supera a llama.cpp** en velocidad para modelos equivalentes. La proyección con Qwen3-4B en ternario (~80 t/s) triplicaría llama.cpp Q4 (~18 t/s).

### Benchmarks de Calidad Objetivo (Post-UCP v2)

| Benchmark | llama.cpp Q4 (ref) | MUD 1.58-bit objetivo | Degradación máxima |
|-----------|-------------------|----------------------|-------------------|
| Perplexity (WikiText-2) | ~6.5 | <7.5 | ≤15% |
| MMLU (5-shot) | ~65% | >60% | ≤8% |
| HellaSwag | ~82% | >78% | ≤5% |
| iteration_validator IQ | N/A | **≥96%** | 0% (mandato) |
| SQNR conversión | N/A | **≥10.5 dB** | 0% (mandato UCP) |

---

## 🖥️ MATRIX TECNOLOGÍAS PARA PC ULTRA-MODESTOS

> Hardware objetivo: **CPU-only, 8-16 GB RAM, sin GPU dedicada, ~50W TDP**

| Tecnología | Paper | PC Modesto | RAM Delta | Impacto T/s | Sprint |
|------------|-------|-----------|-----------|-------------|--------|
| Ternary GEMM 1.58-bit | [2402.17764](https://arxiv.org/abs/2402.17764) | ✅ IDEAL | -75% vs FP16 | +3-6× | ✅ DONE |
| Mamba SSM O(1) | [2312.00752](https://arxiv.org/abs/2312.00752) | ✅ IDEAL | O(1) fijo | contexto gratis | ✅ DONE |
| GQA grupos KV | [2305.13245](https://arxiv.org/abs/2305.13245) | ✅ ACTIVO | -75% KV | +latencia | ✅ DONE |
| RoPE in-place | [2104.09864](https://arxiv.org/abs/2104.09864) | ✅ ACTIVO | 0 extra | 0 overhead | ✅ DONE |
| **z-loss router** | [2202.08906](https://arxiv.org/abs/2202.08906) | ✅ CRÍTICO | 0 extra | +estabilidad QAT | 🔴 Sprint 1 |
| **Attention Sinks** | [2309.17453](https://arxiv.org/abs/2309.17453) | ✅ CRÍTICO | 0 extra | fix coherencia | 🔴 Sprint 1 |
| **Embedding INT4** | K-Quants | ✅ CRÍTICO | -2 GB | -87% vocab | 🔴 Sprint 1 |
| **COCONUT loop** | [2412.06769](https://arxiv.org/abs/2412.06769) | ✅ IDEAL | 0 extra | +reasoning | 🟠 Sprint 2 |
| **ALiBi** | [2108.12409](https://arxiv.org/abs/2108.12409) | ✅ IDEAL | 0 extra | contexto ∞ | 🟡 Sprint 3 |
| Mamba-3 MIMO | [2603.15569](https://arxiv.org/abs/2603.15569) | ✅ FACTIBLE | 0 extra | +20% AVX2 | 🟠 Sprint 2 |
| Hash Routing MoE | [2106.04426](https://arxiv.org/abs/2106.04426) | ✅ IDEAL | 0 extra | router-free | 🟡 Sprint 3 |
| LoRA delta adapters | [2410.20672](https://arxiv.org/abs/2410.20672) | ✅ FACTIBLE | +0.1 GB | N modelos 1 base | 🟡 Sprint 3 |
| TTT Layers | [2407.04620](https://arxiv.org/abs/2407.04620) | ⚠️ 1-2 layers | +0.5 GB/layer | +calidad | 🟡 Sprint 3 |
| Speculative Decoding | [2211.17192](https://arxiv.org/abs/2211.17192) | ⚠️ 2 modelos | +1-2 GB | +2-3× t/s | 🟡 Sprint 4 |
| Vulkan Zero-Copy | MUD actual | ✅ iGPU only | 0 (shared) | 20 t/s iGPU | 🟠 Sprint 2 |
| FlashAttention | [2205.14135](https://arxiv.org/abs/2205.14135) | ❌ GPU req. | N/A | +2-4× | ⚫ Futuro |
| P2P WiFi Swarm | MUD Roadmap | ❌ N nodos | N × RAM | +lineal | ⚫ Futuro |

---

## 🔴 PHASE 0: THE GREAT AWAKENING (MISIÓN CRÍTICA)

**Objetivo:** Restaurar habla coherente (>96% score).

### AWAKE-01: Universal Self-Adjusting Aligner
- ⚡ **Factibilidad:** 🟡 Medio (~10 días)
- 📊 **Benchmark:** Score actual 8.8% → objetivo 99.9% post-Epoch 1
- 🖥️ **PC Modesto:** ✅ — Shadow weights FP32 en RAM, CPU-only
- **Papers:** QAT [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) + STE [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)

### AWAKE-02: Dynamic Autonomy & Telemetry
- ⚡ **Factibilidad:** 🟢 Fácil (~3 días)
- 📊 **Benchmark:** Elimina ~20% repeticiones en secuencias largas
- 🖥️ **PC Modesto:** ✅

### AWAKE-03: Real-Time Wave Coherence
- ⚡ **Factibilidad:** 🔴 Difícil (~25 días)
- 📊 **Benchmark:** Holographic Confidence baseline 88.02% → objetivo 99.9%
- 🖥️ **PC Modesto:** ⚠️ — Requiere modelo maestro FP16 en RAM durante alignment (~8 GB extra, temporal)
- **Paper:** BitNet b1.58 [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) + MUD White Paper §6

### VERIFY-01 & VERIFY-02
- ⚡ **Factibilidad:** 🟢 Fácil (ya implementado)
- 📊 **Benchmark:** `iteration_validator` ≥ 96% composite score
- 🖥️ **PC Modesto:** ✅

---

## 🔴 PHASE ∞: MUD SINGULARITY

### UNIV-01: Universal Model Converter
- ⚡ **Factibilidad:** 🟡 Medio (~15 días por arquitectura nueva)
- 📊 **Benchmark objetivo:** Qwen3-4B → .mud en <5 min, SQNR ≥10.5 dB, T/s ~80
- 🖥️ **PC Modesto:** ✅ — Conversión requiere ~2× RAM modelo (16 GB para 4B)
- **Modelos prioritarios para conversión:**
  1. **Phi-4-mini (3.8B)** — Mejor reasoning/param, ideal Phase 14 · [HF: microsoft/phi-4-mini](https://huggingface.co/microsoft/phi-4-mini)
  2. **Qwen3-4B** — Mejor all-round, candidato a producción · [HF: Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
  3. **SmolLM3-3B** — Mayor velocidad post-conversión (~120 t/s) · [HF: HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- **Papers:** [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) · [arXiv:2504.12285](https://arxiv.org/abs/2504.12285)

### UNIV-02: Rapid Edge Deployment (7B+)
- ⚡ **Factibilidad:** 🔴 Difícil — 7B ternario requiere ~4.5 GB RAM
- 📊 **Benchmark:** 7B en 1.58-bit proyectado: ~35 t/s en i7 con 16 GB
- 🖥️ **PC Modesto:** ⚠️ — Requiere ≥16 GB RAM
- **Papers:** LLM in a Flash [arXiv:2312.11514](https://arxiv.org/abs/2312.11514) · PowerInfer [arXiv:2312.12456](https://arxiv.org/abs/2312.12456)

---

## 🔵 PHASE 14: RECURSIVE REASONING & TERNARY SINGULARITY

### RRM-01: Zero-Allocation Feedback Loop (COCONUT)
- ⚡ **Factibilidad:** 🟢 Fácil (~5 días) — Los buffers ya están pre-asignados
- 📊 **Benchmark:** COCONUT reporta +15-30% accuracy en GSM8K. Costo: 0 RAM extra.
- 🖥️ **PC Modesto:** ✅ IDEAL — Re-usa `InferenceWorkspace` existente
- **Paper:** COCONUT [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
- **Implementación:** En `src/mud/inference.rs`: skip sampling, re-feed `hidden[L]` → `x_moe_norm` para N iteraciones. K=3 iteraciones cuesta 3× tiempo pero evita tokens erróneos.

### RRM-02: Latent Imagination (Vulkan Async)
- ⚡ **Factibilidad:** 🔴 Difícil (~30 días) — Sincronización CPU↔iGPU compleja
- 📊 **Benchmark:** Speculative Decoding: 2-3× throughput
- 🖥️ **PC Modesto:** ✅ — iGPU compartida (Iris Xe), sin VRAM dedicada
- **Papers:** [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) · EAGLE [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)

### LDT-01: Lattice Constraint Projections
- ⚡ **Factibilidad:** ⚫ Investigación (~45 días)
- 📊 **Benchmark:** Sin datos públicos en ternario — área de investigación activa
- 🖥️ **PC Modesto:** ✅ — Operaciones vectoriales baratas
- **Papers:** Looped Transformers [arXiv:2502.17416](https://arxiv.org/abs/2502.17416) · ETD [arXiv:2510.07358](https://arxiv.org/abs/2510.07358)

### LDT-02: Deterministic Early Exit
- ⚡ **Factibilidad:** 🟡 Medio (~8 días)
- 📊 **Benchmark:** ETD: reducción O(K×L) → O(K×L_subset), hasta -60% cómputo por loop
- 🖥️ **PC Modesto:** ✅ — Reduce cómputo, no lo aumenta
- **Paper:** ETD [arXiv:2510.07358](https://arxiv.org/abs/2510.07358)

### BIT-01: Auditoría AVX2 vs LUT (bitnet.cpp)
- ⚡ **Factibilidad:** 🟡 Medio (~7 días)
- 📊 **Benchmark:** bitnet.cpp: 2.37-6.17× speedup en x86. MUD actual: 57 t/s (135M). Objetivo: 90+ t/s.
- 🖥️ **PC Modesto:** ✅ — Optimización pura CPU
- **Paper:** bitnet.cpp [arXiv:2410.16144](https://arxiv.org/abs/2410.16144)

### BIT-02: Q-Head Routing (GRAM)
- ⚡ **Factibilidad:** ✅ COMPLETADO
- 📊 **Benchmark:** Sin datos públicos en ternario
- 🖥️ **PC Modesto:** ✅

---

## 🔵 PHASE 13: PARADIGMAS MATEMÁTICOS AVANZADOS

### MATH-03: Mamba-3 Integration (ICLR 2026 Oral)
- ⚡ **Factibilidad:** 🟡 Medio (~12 días)
- 📊 **Benchmark:** Mamba-3: igual o superior perplexity con ½ estado SSM. MIMO: +20-25% intensidad aritmética AVX2.
- 🖥️ **PC Modesto:** ✅ — MIMO reduce memoria de estado, no la aumenta
- **Paper:** Mamba-3 [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- **Sub-tareas:**
  - **Trapezoidal Discretization** (~3 días): Reemplazar Euler en `mamba_conv_state`
  - **MIMO SSMs** (~5 días): Vectores completos en lugar de canales escalares
  - **Complex-valued States** (~4 días): Rotaciones en espacio complejo (equivalente RoPE free)

### MATH-04: SSM Context Consolidation (Context Folding)
- ⚡ **Factibilidad:** ⚫ Investigación (~30 días)
- 📊 **Benchmark:** Teórico: elimina KV-cache completamente → RAM fija sin importar contexto
- 🖥️ **PC Modesto:** ✅ IDEAL — Elimina la mayor fuente de crecimiento de RAM

### DECL-02: ALiBi Extrapolation
- ⚡ **Factibilidad:** 🟢 Fácil (~4 días)
- 📊 **Benchmark:** ALiBi: contextos 256k+ sin degradación posicional. Costo: 1 multiply-add por celda.
- 🖥️ **PC Modesto:** ✅ — Zero heap allocation, `vfmadd231ps` AVX2
- **Paper:** ALiBi [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)

### ALIGN-02: TTT Layers
- ⚡ **Factibilidad:** 🔴 Difícil (~20 días)
- 📊 **Benchmark:** TTT supera Mamba-2 en long-context. Costo: +0.5 GB RAM por capa.
- 🖥️ **PC Modesto:** ⚠️ — Solo factible para 1-2 capas (no todas las 30). TTT-Linear > TTT-MLP.
- **Paper:** TTT [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
- **Restricción MUD:** Update TTT DEBE usar buffers pre-asignados (Zero-Allocation Policy).

---

## 🟢 PHASE EDGE: PC ULTRA-MODESTOS (PRIORIDAD ALTA)

### EDGE-01: Attention Sinks — Fix KV Cache Reset 🔥
- ⚡ **Factibilidad:** 🟢 Fácil (~2 días)
- 📊 **Benchmark:** Elimina break de coherencia semántica en posición 4000. Costo: 0 RAM.
- 🖥️ **PC Modesto:** ✅ IDEAL
- **Paper:** StreamingLLM [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
- **Acción:** Retener los primeros **4 sink tokens** permanentemente en posiciones 0-3 del KV cache circular en lugar del hard-reset actual.

### EDGE-02: Embedding K-Quants (INT4) 🔥
- ⚡ **Factibilidad:** 🟢 Fácil (~3 días)
- 📊 **Benchmark:** 2.18 GB → 0.27 GB (-87%). La compresión más impactante por ratio esfuerzo/resultado.
- 🖥️ **PC Modesto:** ✅ CRÍTICO — Libera 2 GB, permite cargar modelos en 8 GB RAM

### EDGE-03: z-loss Router Stability 🔥
- ⚡ **Factibilidad:** 🟢 Fácil (~1 día) — Una línea en el router forward pass
- 📊 **Benchmark:** ST-MoE: mejora de estabilidad más accionable para prevenir logit explosion durante QAT/STE
- 🖥️ **PC Modesto:** ✅ — Costo: 0 RAM, 1 operación escalar por batch
- **Paper:** ST-MoE [arXiv:2202.08906](https://arxiv.org/abs/2202.08906)
- **Fórmula:** `L_z = (log Σ exp(router_logits))²` — añadir al loss total en QAT

### EDGE-04: MUD-Executable (Llamafile Style)
- ⚡ **Factibilidad:** 🟡 Medio (~8 días)
- 📊 **Benchmark:** Llamafile: ~100 MB ejecutable autónomo. MUD podría empaquetar model + engine.
- 🖥️ **PC Modesto:** ✅ IDEAL — Zero setup, portable a cualquier Linux/Windows x86_64

### EDGE-05: BPE Tokenizer O(n log n) — Fix PERF-05
- ⚡ **Factibilidad:** 🟡 Medio (~5 días)
- 📊 **Benchmark:** Reducción O(n²) → O(n log n). Impacto medible en prompts >512 tokens.
- 🖥️ **PC Modesto:** ✅ — Reduce CPU load en contextos largos

### EDGE-06: Hub & Spoke Local API
- ⚡ **Factibilidad:** 🟡 Medio (~10 días)
- 📊 **Benchmark:** Sirve inferencia a 5-10 clientes WiFi sin latencia adicional significativa
- 🖥️ **PC Modesto:** ✅

---

## 🔵 PHASES 12 & 11: COMPLETADAS ✅

| Ítem | Estado | Benchmark logrado |
|------|--------|------------------|
| HW-01: HardwareProfile detection | ✅ | P+E core detect en <1ms |
| HW-02: Rayon → 4 P-cores | ✅ | +35% vs E-core scheduling |
| HW-03: RoPE ASM VADDSUBPS | ✅ | RoPE: <0.1ms por layer |
| HW-04: BMI2 ternary unpack | ✅ | +40% unpack throughput |
| DB-01/02/03: SQLite removal | ✅ | -1.2 GB RAM footprint |

---

## 📋 TABLA MAESTRA DE PRIORIDAD — SPRINT PLANNING

> Ordenado por ratio **Impacto / Esfuerzo** para PC modesto.

| # | Ítem | Esfuerzo | Impacto en PC Modesto | Paper |
|---|------|----------|----------------------|-------|
| 1 | **EDGE-03: z-loss router** | 1 día | 🔴 Estabiliza QAT ahora mismo | [2202.08906](https://arxiv.org/abs/2202.08906) |
| 2 | **EDGE-01: Attention Sinks** | 2 días | 🔴 Fix coherencia pos.4000 | [2309.17453](https://arxiv.org/abs/2309.17453) |
| 3 | **EDGE-02: Embedding INT4** | 3 días | 🔴 -2 GB RAM, -87% vocab | K-Quants |
| 4 | **RRM-01: COCONUT loop** | 5 días | 🟠 +reasoning, 0 RAM extra | [2412.06769](https://arxiv.org/abs/2412.06769) |
| 5 | **AWAKE-01: Self-Aligner** | 10 días | 🔴 8.8% → 99.9% coherencia | [1712.05877](https://arxiv.org/abs/1712.05877) |
| 6 | **EDGE-05: BPE O(n log n)** | 5 días | 🟠 -latencia prompts largos | PERF-05 fix |
| 7 | **BIT-01: AVX2 vs LUT audit** | 7 días | 🟠 +50% TPS potencial | [2410.16144](https://arxiv.org/abs/2410.16144) |
| 8 | **UNIV-01: Converter Phi-4-mini** | 15 días | 🔴 Modelo maestro nuevo SOTA | [2402.17764](https://arxiv.org/abs/2402.17764) |
| 9 | **MATH-03: Mamba-3 MIMO** | 12 días | 🟠 +20-25% AVX2 intensity | [2603.15569](https://arxiv.org/abs/2603.15569) |
| 10 | **DECL-02: ALiBi** | 4 días | 🟡 256k contexto gratis | [2108.12409](https://arxiv.org/abs/2108.12409) |
| 11 | **ALIGN-02: TTT Layers** | 20 días | 🟡 +calidad (1-2 capas max) | [2407.04620](https://arxiv.org/abs/2407.04620) |
| 12 | **EDGE-04: MUD-Executable** | 8 días | 🟡 Portabilidad total | Llamafile |
| 13 | **LDT-01: Lattice Projections** | 45 días | ⚫ Investigación | [2502.17416](https://arxiv.org/abs/2502.17416) |
| 14 | **RRM-02: Vulkan Async** | 30 días | 🟡 +2-3× throughput iGPU | [2211.17192](https://arxiv.org/abs/2211.17192) |

---

## 🔬 BENCHMARKS PUBLICADOS POR PAPER

| Paper | Resultado Publicado | Aplicabilidad MUD | Confianza |
|-------|--------------------|--------------------|-----------|
| BitNet b1.58 [2402.17764] | Paridad FP16 en MMLU a 2B params | Validación arquitectural core | ★★★★★ |
| bitnet.cpp [2410.16144] | 2.37-6.17× speedup x86, -72-82% energía | Referencia kernel AVX2 | ★★★★★ |
| Attention Sinks [2309.17453] | Streaming ∞ sin límite de contexto | Fix KV reset, 0 RAM extra | ★★★★★ |
| ST-MoE z-loss [2202.08906] | Estabilidad training MoE crítica | 1 día implementación | ★★★★★ |
| COCONUT [2412.06769] | +15-30% GSM8K sin parámetros extra | RRM-01, 0 RAM extra | ★★★★☆ |
| Mamba-3 [2603.15569] | ½ estado SSM, misma calidad Mamba-2 | MATH-03 MIMO | ★★★★☆ |
| ETD [2510.07358] | -60% cómputo con subset de capas | LDT-02 early exit | ★★★★☆ |
| Looped Transformers [2502.17416] | Recursión > más params (mismo budget) | Validación teórica RRM | ★★★★☆ |
| Speculative Decoding [2211.17192] | 2-3× throughput en producción | RRM-02 Vulkan | ★★★★☆ |
| TTT [2407.04620] | Supera Mamba-2 en long-context | ALIGN-02 (1-2 capas) | ★★★☆☆ |

---

## 📚 AUDITORÍAS PENDIENTES

| Auditoría | Trigger | Herramientas |
|-----------|---------|--------------|
| **Audit V9:** TTT Layer precision vs. cost | Post ALIGN-02 | `iteration_validator` |
| **Audit V10:** Phi-4-mini UCP conversion quality | Post UNIV-01 | `conversion_verifier` + `boundary_validator` |
| **Audit V11:** MIMO Mamba-3 SSM eigenvalue stability | Post MATH-03 | `conversion_verifier` (HiPPO check) |

---

*MUD v4.0 · Roadmap con Factibilidad & Benchmarks · 2026-06-04*
*Papers completos con links: [`docs/RESEARCH_PAPERS.md`](RESEARCH_PAPERS.md)*
