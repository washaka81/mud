# 📋 Leela = "Leer": Arquitectura de Lectura y Comprensión de MUD

> **"Leela"** (leer en español) describe la función fundamental del proyecto Forge LLM (MUD): **leer y comprender lenguaje**. El motor no solo procesa tokens — implementa una arquitectura cognitiva de lectura profunda.

---

## 1. 🧠 Arquitectura Cognitiva de Lectura (MUD Cognitive Architecture v39)

Fuente: `docs/architecture/MUD_COGNITIVE_ARCH.md`

### 1.1 Deep Reasoner Depth (2-Layer Constraint)
El poder de razonamiento no viene de la profundidad, sino de la **especialización de expertos**:
- **Chain of Thought (CoT):** Los tokens se fuerzan a través de bloques `<thinking>` que permiten múltiples pasadas sobre el workspace antes de cometer a `<answer>`
- **Expert Specialization:** 8 expertos especializados actuán como "nodos de razonamiento" que iteran sobre la lógica antes de outputar a la capa de embedding final

### 1.2 Robustness Protocols (v39)
- **Bias Injection:** Inyección obligatoria de sesgos hacia expertos de Logic, Math, y Code
- **Static Workspace Consistency:** Buffers pre-asignados eliminan memory-jitter que causaba token corruption (`<unk>`)

### 1.3 Skill Integration (habilidades de lectura)
| Skill | Función de lectura |
|-------|-------------------|
| **LogicMarkSkill** | Monólogos internos — lectura lógica |
| **CodeFormatSkill** | Lógica estructurada — lectura de código |
| **TextStylingSkill** | Información de alta densidad — lectura estilizada |
| **ResearchSkill (WebSearch)** | Ingestión autónoma de datos en tiempo real — lectura activa |

### 1.4 Cognitive Emergence (Natural Routing)
- **Learned Intent:** Routing heurístico eliminado — la arquitectura MoE discrimina autónomamente entre saludos, consultas, y resolución de problemas
- La "inteligencia social" y "conciencia de dominio" se codifican directamente en los pesos de los expertos

---

## 2. 🏗️ Arquitectura de Lectura (MUD Architecture)

Fuente: `docs/architecture/MUD_ARCHITECTURE.md`

### 2.1 Jamba Hybrid Design (lectura híbrida)
Combina dos modos de lectura:
- **Transformer Layers ($O(N^2)$):** Atención dispersa + MoE — lectura **relacional** y saltos lógicos profundos
- **Mamba Layers ($O(N)$):** State Space Models recurrentes — lectura **secuencial** con huella de memoria $O(1)$

### 2.2 Static Workspace (Zero-Allocation)
- **InferenceWorkspace:** Estructura monolítica con todos los tensores intermedios, KV-cache de atención, y estados recurrentes Mamba
- **Static Pointers:** Acceso a pesos via raw pointers a memoria mmap — elimina overhead de lookup durante la lectura

### 2.3 High-Fidelity Quantization (PRQ)
- **Per-Row Quantization:** Cada dimensión de salida de cada matriz tiene su escala f32 dedicada
- **Formato .mud:** Pesos packed u32 + vector de escala f32 separado

### 2.4 Hardware Acceleration (lectura acelerada)
- **AVX2 ASM (CPU):** Assembly para ternary GEMV, RMSNorm, SiLU, y **Mamba Parallel Scan**
- **Vulkan (iGPU):** Shaders de cómputo zero-copy con heartbeat asíncrono
- **Rayon:** Ejecución multi-thread de expertos MoE

---

## 3. 🔧 Pipeline de Restauración Lingüística

Fuente: `docs/architecture/MUD_ARCHITECTURE.md` §5

### 3.1 Align
Mapea pesos a el vocabulario bilingüe MUD — **alineación semántica** post-cuantización

### 3.2 Project
Recalibración bayesiana de escalas via análisis de activación — **restauración de precisión**

### 3.3 Train
SGD ternario en caliente (Live Hot Ternary SGD) — **minimiza desviación de cuantización**

---

## 4. 🛡️ Seguridad e Integridad (durante la lectura)

| Protocolo | Función |
|-----------|---------|
| **Pointer Alignment** | Alineación estricta `(n + 15) / 16` — previene lecturas OOB |
| **Sanitization** | Detección y neutralización de NaNs/Infs en el estado oculto |
| **Diagnostic CHI** | Índice de Salud Cognitiva — monitoreo continuo durante la inferencia |

---

## 5. 📚 Componentes de Lectura en el Código

| Componente | Archivo | Función de lectura |
|-----------|---------|-------------------|
| **SmartTokenizer** | `src/mud/smart_tokenizer.rs` | Tokenización zero-alloc con buffers reusados |
| **SlimeRegister** | `src/mud/slime.rs` | Celda de activación con `matmul_accum: f32` |
| **SlimeWorkspace** | `src/mud/slime.rs` | Workspace pre-asignado (P-01) |
| **C-MUD Thinker** | `src/mud/cmud.rs` | Razonamiento complejo (manifold, phase-lock, CUE repulsion) |
| **JEPA + mHC** | `src/mud/slime_forward.rs` | Estabilidad JEPA OU + mHC residual |
| **Fenwick trees** | `src/mud/fenwick.rs` | O(log n) sampling y búsqueda durante la lectura |
| **ASM kernels** | `src/asm/` | AVX2 FMA para GEMV, SiLU, RMSNorm, LM head |

---

## 6. 📋 Comandos de Lectura

```bash
# Lectura básica (chat interactivo)
cargo run --release --bin forge_llm -- models/smollm2.mud

# Lectura con C-MUD (razonamiento complejo)
MUD_CMUD_THINK=1 MUD_CMUD_PARAMS=models/smollm2.mud.cmud.json \
  cargo run --release --bin forge_llm -- models/smollm2.mud

# Lectura con debug (tokens, logits, top-10)
MUD_DEBUG=1 cargo run --release --bin forge_llm -- models/smollm2.mud

# Lectura con system prompt y temperatura
MUD_SYSTEM_PROMPT="Eres un asistente útil." MUD_INFER_TEMPERATURE=0.7 \
  cargo run --release --bin forge_llm -- models/smollm2.mud

# Auditoría de fidelad de lectura (Fenwick-optimized)
cargo run --release --bin precise_converter_audit -- models/smollm2 models/smollm2.mud --calibration
```

---

## 7. 📋 Conclusión

**"Leela" (leer)** describe la esencia del proyecto Forge LLM (MUD): una arquitectura de lectura y comprensión profunda que combina:

1. **Jamba Hybrid** — lectura secuencial (Mamba) + lectura relacional (Transformer MoE)
2. **Cognitive Architecture v39** — Chain of Thought, expert specialization, bias injection
3. **Zero-Allocation Workspace** — lectura sin interrupciones de memoria
4. **High-Fidelity PRQ** — lectura precisa con per-row quantization
5. **C-MUD Thinker** — razonamiento complejo durante la lectura
6. **Fenwick trees** — O(log n) sampling durante la generación
7. **ASM AVX2** — lectura acelerada por hardware

El motor no solo "lee" — **entiende, razona, y responde** con una arquitectura cognitiva diseñada para la comprensión profunda en hardware Intel de consumo.
