# MUD: Bitácora de Consulta Tecnológica (Tech Consultation Log)
**Última Actualización:** 2 de junio de 2026

Esta bitácora registra las fuentes de investigación, repositorios y artículos técnicos utilizados para orientar el desarrollo del motor **Forge LLM (MUD)** hacia el estado del arte en inferencia local y arquitecturas de alto rendimiento.

## 📚 Fuentes de Investigación (Papers & Repos)

| Fecha | Fuente / Enlace | Paradigmas Clave | Relevancia para MUD |
| :--- | :--- | :--- | :--- |
| 02/06/2026 | [AI Papers of the Week (May 24-31)](https://github.com/dair-ai/AI-Papers-of-the-Week/blob/main/years/2026.md#top-ai-papers-of-the-week-may-24---may-31---2026) | SSM Context Consolidation, Agentic Weight Distillation | Implementar "Context Folding" para eliminar costo de KV-cache. |
| 02/06/2026 | [mcd-unison/llm](https://github.com/mcd-unison/llm) | DSPy (Declarative Programming), ALiBi, Eval-Driven Development | Transición a "Firmas Declarativas" en Rust y extrapolación de contexto masivo. |
| 02/06/2026 | [Wakoma/OfflineAI](https://github.com/Wakoma/OfflineAI) | Single-file portability, Local Hub & Spoke, Thermal Management | Paradigma de accesibilidad total y ejecutables portátiles (Llamafile style). |
| 02/06/2026 | [BitNet b1.58 (Microsoft)](https://github.com/microsoft/bitnet) | Ternary 1.58-bit, Multi-row kernels, 100B models on CPU | Validación de nuestro kernel ternario y optimización de GEMM sin multiplicaciones. |
| 02/06/2026 | [Mamba-3 (Princeton/Tri Dao)](https://github.com/state-spaces/mamba) | MIMO SSMs, Exponential-Trapezoidal Discretization | Evolución de nuestras capas Mamba para mayor intensidad aritmética en P-cores. |
| 02/06/2026 | [T-SAR: In-Register LUTs](https://github.com/microsoft/bitnet) | SIMD Register-based LUTs, 86x GEMV throughput | (Propuesto) Reemplazar LUTs de memoria por registros SIMD para evitar el Memory Wall. |
| 02/06/2026 | [FairyFuse: BMI2 PEXT Decoding](https://github.com/arxiv) | BMI2 `_pext_u32` for Ternary Masks, Masked Add/Sub | (Propuesto) Optimizar el desempaquetado de pesos usando instrucciones BMI2 avanzadas. |
| 02/06/2026 | [Rusty Penguin (Bare-Metal Rust)](https://github.com/rusty-penguin) | Sparse-skip inference, Zero-OS overhead | Validación de nuestra política de Zero-Allocation y ejecución sobre hardware crudo. |
| 03/06/2026 | [recursive-reasoning/tiny-recursive-models](https://github.com/recursive-reasoning/tiny-recursive-models) | Tiny Recursive Models (TRM), Latent Feedback Loops | Desacople de profundidad lógica vs tamaño de red (Zero-Allocation Feedback). |
| 03/06/2026 | [latent-lattice/neuro-symbolic-decoding](https://github.com/latent-lattice/neuro-symbolic-decoding) | Neuro-Symbolic Latent Lattices, Deterministic Early Exits (LDT) | Validador matemático de convergencia ($L2$ Shift / Epsilon) para detener bucles de inferencia. |
| 03/06/2026 | [open-rrm/gram-inference-kernels](https://github.com/open-rrm/gram-inference-kernels) | Width Scaling, Probabilistic Trajectories, Q-Heads | (Futuro Fase 14) Compute Shaders asíncronos en Vulkan para especulación lógica. |

## 🛠️ Tecnologías y Paradigmas Resueltos en MUD

1.  **BitNet 1.58-bit (Ternary):** Implementado mediante kernels ASM AVX2 que realizan sumas/restas vectoriales en lugar de multiplicaciones.
2.  **Hybrid SSM-Transformer (Jamba style):** Soporte nativo para capas Mamba que resumen el contexto en un estado recurrente fijo ($O(1)$ memoria).
3.  **Hardware-Aware Threading (Adler Lake):** Pool de hilos bloqueado a núcleos Performance (P-Cores) para maximizar la caché L3 y evitar hilos de eficiencia (E-Cores).
4.  **Zero-Allocation Hot-Loop:** Espacio de trabajo estático pre-asignado para eliminar latencia del gestor de memoria del SO durante la generación.
5.  **Split RoPE (AVX2):** Implementación en ensamblador de la rotación de posición usando la instrucción `VADDSUBPS` para eficiencia de un solo ciclo.

## 🚀 Próximas Investigaciones de Alta Prioridad

*   **T-SAR (In-Register LUTs):** Cómo mover las tablas de consulta directamente a los registros `%ymm` / `%zmm` para romper el cuello de botella de la RAM.
*   **Mamba-3 MIMO Architecture:** Refactorizar el escaneo SSM para procesar vectores en lugar de escalares, aprovechando el ancho de banda del bus de datos.
*   **TTT (Test-Time Training) Layers:** Investigar la implementación de mini-redes en el estado oculto que aprendan durante la inferencia.

---
*MUD: El motor de inferencia local más avanzado para hardware modesto.*
