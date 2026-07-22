# Research Analysis: MAI-Thinking-1 Architecture for Low-End Hardware (Equipos Pobres)

**Date:** 2026-07-13
**Subject:** Integration of Microsoft's MAI-Thinking-1 concepts into the Forge LLM (MUD) Engine.
**Goal:** Extract architectural strategies to maximize inference and training efficiency on low-bandwidth, CPU-only or integrated GPU hardware (Intel i7-1260P).

## 1. LatentMoE (Compresión previa al despacho)
**Concepto Original:** A diferencia del MoE tradicional donde un vector oculto de gran dimensión ($h$) viaja completo a los expertos, *LatentMoE* aplica primero una matriz de proyección hacia abajo (compresión) compartida. Los vectores reducidos viajan a los expertos, se combinan, y luego se proyectan hacia arriba nuevamente.
**Aplicación para Equipos Pobres:**
- **Reducción de Ancho de Banda (RAM Bottleneck):** En CPUs y memorias unificadas DDR4/DDR5, el principal cuello de botella es mover los pesos desde la RAM hacia la caché L1/L2. Si reducimos la dimensionalidad *antes* de cargar a los expertos, las matrices ternarias de los expertos pueden ser minúsculas. 
- **Viabilidad:** Permite ejecutar modelos teóricos de cientos de billones de parámetros en memoria RAM convencional porque los expertos "ligeros" caben en la caché de la CPU.

## 2. Atención Híbrida Periódica (5 Locales + 1 Global)
**Concepto Original:** En lugar de calcular atención global para cada capa (O(N²)), MAI-Thinking agrupa sus capas: 5 capas usan *Sliding Window Attention* (ventana local, ej. 512 tokens), seguidas de 1 capa de atención global.
**Aplicación para Equipos Pobres:**
- **Compresión del KV-Cache:** El KV-Cache destruye la memoria en contextos largos. Al restringir las capas locales a los últimos 512 tokens, la memoria temporal necesaria no crece de forma ilimitada para esas capas. Solo la capa global necesita el KV-Cache completo.
- **Reducción de Operaciones (OPS):** La atención local es O(N), reduciendo masivamente el impacto en la iGPU o CPU.

## 3. Eliminación de RoPE en la Capa Global
**Concepto Original:** Las capas locales usan Rotary Positional Embeddings (RoPE), pero la capa global (la que procesa toda la secuencia) **no utiliza ningún Positional Encoding**.
**Aplicación para Equipos Pobres:**
- **Cero overhead trigonométrico:** Aplicar senos y cosenos en secuencias de 32,000 tokens ahoga las unidades AVX2. Eliminar RoPE en la capa más pesada acelera exponencialmente el Forward Pass sin pérdida notable matemática (según el reporte de MAI).

## 4. Zero-Bias (Sin Sesgos)
**Concepto Original:** Eliminación completa de pesos "bias" en todas las redes densas (FFN y Attention).
**Aplicación para Equipos Pobres:**
- Simplifica el kernel en Assembly AVX2 (`ternary_gemm_batch4.s`). Al quitar la adición del vector de bias, ahorramos 1 instrucción de lectura de memoria (`vmovaps`) y 1 de adición (`vaddps`) por fila.

## 5. Weight Tying Extremo (Amarrado de Pesos)
**Concepto Original:** Compartir la inmensa tabla de embeddings (ej. 200,000 vocabulario × 1024 dimensión) tanto en la entrada (`token_embd.weight`) como en la capa final de predicción (`output.weight`).
**Aplicación para Equipos Pobres:**
- **Ahorro de RAM Estática:** Para un vocabulario grande, la cabeza de salida puede consumir cientos de megabytes. Compartir la referencia del puntero (como se hace actualmente en MUD) libera espacio vital de memoria para destinarlo a los KV-caches o a lotes de entrenamiento más grandes.

---

## 6. Estrategias de Entrenamiento y Estados (Hill-Climbing RL)
El enfoque de entrenamiento por refuerzo de MAI-Thinking (conocido como *Hill-Climbing Machine*) aporta conceptos fundamentales para el diseño de nuestro entrenador, permitiendo evadir los costosos requisitos del RL tradicional:

*   **Algoritmo GRPO (Group Relative Policy Optimization):**
    A diferencia de PPO (que requiere cargar un modelo "Actor" y un gigantesco modelo "Crítico/Value"), GRPO **elimina la necesidad del modelo Crítico**. Calcula la recompensa relativa comparando las respuestas generadas en un mismo grupo.
    *   *Para MUD:* Esto es el "Santo Grial" para el RL en equipos pobres. Entrenar con GRPO significa que no necesitamos duplicar la RAM consumida durante el entrenamiento QAT, evitando un colapso por falta de memoria (OOM).

*   **Control de Entropía Adaptativo (Estados de Exploración):**
    El entrenador monitorea activamente un "estado" crítico: **la entropía de la distribución de tokens**.
    *   *El Problema:* Cuando el modelo colapsa o se vuelve muy repetitivo, su entropía cae a casi 0.
    *   *La Solución:* Si la entropía cae por debajo de un umbral (ej. 0.3), el entrenador "pulsa un interruptor" que inyecta ruido o altera la temperatura de muestreo para forzar "saltos grandes" (Exploración). En MUD, podemos mapear esto a nuestra métrica `Z_Entrop` y controlar el parámetro de repulsión dinámica.

*   **Ascensos Especializados Paralelos (Specialist Climbs):**
    En lugar de mezclar todos los datos y requerir un mega-batch (lote masivo) imposible para una sola PC, entrenan "clones" especializados (ej. uno para Código, uno para Matemáticas).
    *   *Para MUD:* Podríamos entrenar distintos `SlimeWorkspace` en diferentes tareas de forma ligera y luego fusionar sus pesos (Model Merging/Slerp), manteniendo el uso de RAM al mínimo absoluto durante el paso backward.

*   **Presupuesto Dinámico de Tokens (Dynamic Response Length):**
    Asignan un presupuesto máximo de "tokens de pensamiento" según la dificultad del problema evaluado por el sistema (de 8k a 128k). Evita desperdiciar ciclos de CPU en preguntas triviales, alineándose perfectamente con nuestro concepto de *Integral Saturation Stop-Anchor* (Early-Exit).

---
### Propuesta para el Roadmap MUD:
Añadir a la "Phase 17" la implementación experimental de **Atención Local Deslizante de 512 tokens** para las capas ternarias pares, limitando el KV-cache para reducir la latencia autoregresiva. Explorar una estructura **Ternary-LatentMoE** como reemplazo al Dense FFN actual. Además, sustituir las exploraciones de PPO por **GRPO** en el *AshQatDispatcher* para habilitar RL sin requerir un modelo Crítico en la RAM.
