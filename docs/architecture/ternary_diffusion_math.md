# Cálculos y Adaptación Matemática: Difusión Discreta en Modelos Ternarios (1.58-bit) para Hardware Restringido

**Objetivo:** Desarrollar el marco teórico y los cálculos de rendimiento para aplicar un modelo de difusión de texto discreto (tipo DiffusionGemma) sobre la arquitectura **MUD Ternaria (1.58-bit)**, optimizando para hardware de bajos recursos (solo CPU, ancho de banda de memoria pobre).

---

## 1. El Cuello de Botella del Hardware Pobre: Ancho de Banda vs Cómputo

El mayor problema del hardware pobre (ej. procesadores antiguos sin GPU dedicada, memoria DDR3/DDR4 lenta) no es la falta de FLOPS de cómputo, sino el **Ancho de Banda de Memoria (Memory Bandwidth Bottleneck)**. 

### Inferencia Autorregresiva Clásica (Token por Token)
Para generar un token, se realiza una multiplicación Matriz-Vector (GEMV).
*   **Lectura de Pesos (W):** En formato ternario empaquetado (2-bits por peso), se leen `0.25 bytes` por cada parámetro.
*   **Operaciones:** 1 operación de suma/resta por peso.
*   **Intensidad Aritmética (AR):** `1 Op / 0.25 bytes = 4 Ops/Byte`.
*   *Conclusión:* El hardware pasa la mayoría del tiempo bloqueado esperando a que la RAM entregue los pesos a la CPU. Es puramente **Memory-Bound**.

### Inferencia por Difusión de Bloque (Canvas de N tokens)
En un paso de *denoising* de difusión discreta, se reevalúa un bloque entero de $N$ tokens en paralelo (Matriz-Matriz o GEMM).
*   **Lectura de Pesos (W):** Se leen los mismos `0.25 bytes` de memoria (los pesos solo se cargan en la caché L1/L2 una vez).
*   **Operaciones:** $N$ operaciones de suma/resta por cada peso.
*   **Intensidad Aritmética (Difusión):** `N Ops / 0.25 bytes = 4N Ops/Byte`.
*   **Cálculo Práctico ($N = 256$ tokens):** `4 * 256 = 1024 Ops/Byte`.
*   *Conclusión:* La intensidad aritmética aumenta en un factor de $N$. El modelo pasa a ser **Compute-Bound**, lo que permite aprovechar al 100% las instrucciones vectoriales SIMD (AVX2/AVX-512) de procesadores antiguos sin esperar a la RAM. **¡Esto es perfecto para hardware pobre!**

---

## 2. Operadores Matemáticos Ternarios en Difusión

Dado que los pesos de MUD están restringidos a la grilla $W \in \{-1, 0, 1\}$, podemos reemplazar las costosas multiplicaciones matriciales del paso de difusión (*Reverse Process*) por adiciones condicionales.

Sea un bloque temporal oculto $X_t \in \mathbb{R}^{N \times d}$ donde $N$ es el tamaño del bloque y $d$ es el *hidden size*.
La proyección lineal estándar es $Y = X_t \cdot W^T$.

### SIMD AVX2 para Hardware Restringido
En lugar de multiplicar matrices densas, se implementa una rutina C/Rust con `_mm256_add_epi8` o análogos.
Dado que la difusión ocurre sobre $N$ secuencias (ej. $N=32$ para mantener todo en la caché L1 de la CPU):
1. Cargamos el vector de estado oculto de 32 tokens.
2. Leemos un bloque empaquetado de pesos $W$ (8 pesos por byte).
3. Usamos máscaras de bits (Bitwise AND) para determinar si sumamos o restamos a los 32 tokens en paralelo.

**Ahorro de Energía Computacional:**
*   Multiplicador FP16 estándar: ~1.5 pJ por op.
*   Suma/Resta de enteros (Ternario): ~0.1 pJ por op.
*   **Reducción de consumo:** ~93% de ahorro energético en el *hot-loop* de difusión.

---

## 3. Dinámica del Ruido (Transition Matrices) para Ternarización

Para aplicar la "Difusión Discreta" sobre la red, debemos ajustar el proceso de corrupción (ruido) a las limitaciones del modelo 1.58-bit. Un modelo ternario sufre de "Ternary Shock" si los estados latentes experimentan distribuciones continuas descontroladas.

**Matriz de Transición Categórica ($Q$):**
El proceso *forward* (añadir ruido al texto base) corrompe el token $x_0$ hacia $x_t$ según:
$$ q(x_t | x_{t-1}) = \text{Categorical}(x_t; x_{t-1}Q) $$

Para hardware pobre y mitigación de *Ternary Shock*, en lugar de usar un decaimiento gaussiano suave sobre los embeddings (que requeriría aritmética FP32 intensa), utilizaremos una política **Absorbente Categórica (Masking)**:
*   Probabilidad de mantener el token real: $\alpha_t$
*   Probabilidad de mutar a `[MASK]`: $1 - \alpha_t$
*   Probabilidad de mutar a ruido aleatorio uniforme: $0$ (se evita para no causar explosión de entropía $\Delta\sigma$ en los pesos ternarios).

### Fórmula de Agendamiento de Ruido Optimizada
El plan de difusión (*Schedule*) debe ser no-lineal para evitar saturar las sumas enteras del hardware:
$$ \bar{\alpha}_t = \cos^2 \left( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \right) $$
Donde $s = 0.008$. En $t=T$, todos los tokens son `[MASK]`.

## 4. Estrategia de Implementación en Forge LLM

Para llevar a cabo esto en `src/mud/inference.rs` en el futuro:
1.  **Reutilizar Memoria (Zero-Allocation):** Modificar `InferenceWorkspace` para alojar un `DiffusionCanvas` estático de $N=64$ o $N=128$ tokens (determinado por el tamaño de la caché L1 del hardware pobre).
2.  **Bucle de Des-Ruido (Denoising Loop):** En lugar de un bucle `for pos in 0..ctx_len`, emplearemos un bucle `for step in (0..T).rev()`.
3.  **Kernel de Suma Bidireccional:** Expandir nuestro ensamblador AVX2 para que aplique sumas ternarias sobre $N$ vectores secuenciales de manera superescalar, utilizando desenrollado de bucles (*loop unrolling*).

**Resultado esperado:** Un modelo que, aunque se ejecute en un procesador Intel i5 de 4ta generación (DDR3), puede destilar respuestas complejas iterando todo el texto en memoria caché sin tocar la RAM principal en cada paso, superando ampliamente los *tokens per second (t/s)* teóricos de una inferencia autorregresiva en el mismo hardware.
