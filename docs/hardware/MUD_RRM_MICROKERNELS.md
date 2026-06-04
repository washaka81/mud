# Arquitectura de Microkernels para SLIME ENGINE (RRM & Ternario)
**Fecha:** 3 de junio de 2026
**Contexto:** Diseño de bajo nivel para Motores de Razonamiento Recursivo (TRM/HRM) y Cuantización Ternaria (BitNet).

Para estructurar un motor de inferencia recursivo, ternario y de ultra-bajo nivel como **SLIME ENGINE**, nos alejamos de los microkernels de deep learning tradicionales (GEMM para FP32) hacia **microkernels de manipulación de bits, recursión latente y empaquetado**.

La carga computacional se divide asimétricamente aprovechando hardware heterogéneo:

---

## 1. Microkernels de CPU (Ensamblador / SIMD)
La CPU (P-Cores) se encarga de la lógica determinista, manipulación de memoria y operaciones simbólicas discretas.

### ⚡ Microkernel de Desempaquetado y Suma Ternaria (1.58-bit BitNet)
- **Función:** En lugar de multiplicar, los pesos ternarios $\{-1, 0, 1\}$ implican sumar, restar o ignorar valores de activación. Los pesos se almacenan empaquetados (2 bits por peso) para maximizar el ancho de banda.
- **Implementación (AVX2/BMI2):** Uso de operaciones de máscara de bits (`AND`, `SHR` / `VPSRLVD`) para extraer valores de 2 bits, seguido de acumulaciones vectoriales (`_mm256_add_epi32` o similares) sobre las activaciones. Cero operaciones de coma flotante puras en el cálculo del peso.

### 🔄 Microkernel de Actualización Recursiva Latente (TRM/HRM)
- **Función:** Ejecuta el bucle interno de los Tiny Recursive Models. Toma el tensor del estado latente anterior ($z_{t-1}$) y genera el nuevo estado ($z_t$) de forma cíclica.
- **Implementación (Zero-Allocation):** Diseñado bajo *Data-Oriented Design* (SoA). Mantiene los punteros de los vectores latentes fijos en caché L1/L2. Se ejecuta un bucle iterativo sin reasignaciones de memoria (sin `malloc` intermedio) hasta cumplir la condición de convergencia.

### 🌐 Microkernel de Proyección de Retícula (LDT - Neuro-Simbólico)
- **Función:** Actúa como el filtro lógico/validador. Comprueba si el vector latente continuo cumple con las restricciones booleanas/algebraicas de la retícula matemática.
- **Implementación:** Aritmética de mapas de bits (*bitmaps*). Uso de intrínsecas como `_popcnt` o `AND`/`OR` bit a bit para validar la ruta lógica en pocos ciclos de reloj, disparando un *Early Exit* si el estado converge o detecta un fallo estructural.

---

## 2. Microkernels de GPU (Vulkan Compute Shaders)
La iGPU (ej. Iris Xe) maneja el paralelismo masivo especulativo, evaluando múltiples futuros latentes simultáneamente (asincronía).

### 🎲 Microkernel de Inyección de Ruido y Caminos Paralelos (Width Scaling / GRAM)
- **Función:** Clona múltiples variantes del estado latente base inyectando ruido estocástico controlado para evaluar trayectorias paralelas.
- **Implementación:** Compute Shaders en Vulkan con generadores pseudoaleatorios ligeros (PCG o LFSR en hardware de enteros) para perturbar las activaciones espacialmente. Los grupos de hilos exploran distintas hipótesis concurrentemente.

### 🎯 Microkernel de Selección por Cabezal Q (Q-Head Scoring)
- **Función:** Evalúa cuál trayectoria probabilística de GRAM es la óptima para salir de atractores lógicos.
- **Implementación:** Reducción paralela (*Parallel Reduction*) en memoria compartida (*shared memory/workgroups*). Identifica el score máximo según la función de coste neuro-simbólica con mínima latencia de ida y vuelta.

---

## 3. Microkernel de Interfaz de Red (Topología)

### 🌉 Kernel de Enrutamiento de Grafos (Königsberg Topology)
- **Función:** Orquesta el flujo de tensores a través de los componentes del motor (Atención, Mamba, Expertos MoE), asegurando una ruta eficiente (Euleriana).
- **Implementación:** Microkernel de salto de punteros (*pointer-chasing*) ultra optimizado. Lee un arreglo indexado plano (matriz de adyacencia del grafo de cómputo) y calcula el siguiente bloque de ejecución mediante desplazamientos directos en memoria contigua, evitando rupturas de caché L1.

---

## Hoja de Ruta de Implementación (SLIME ENGINE)
1. **Fase 1 (CPU):** Escribir/Auditar el kernel de **desempaquetado ternario (BitNet)** (`pack_ternary_row` y operaciones SIMD asociadas) para lograr densidad máxima en caché AVX2. *(En proceso bajo Fase 14 del Roadmap)*.
2. **Fase 2 (CPU):** Consolidar el bucle **TRM de refinamiento latente** (implementado vía `inject_latent_feedback_moe`/`mamba` asegurando Zero-Allocation continuo).
3. **Fase 3 (Vulkan):** Trasladar el escalado de ancho probabilístico (**GRAM**) a Shaders de Vulkan, sincronizando CPU e iGPU mediante *Storage Buffers* mapeados.