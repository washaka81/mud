# HMP Viability: Mathematical Proof of Zero-Bottleneck Impact

Para demostrar matemáticamente que la arquitectura HMP (Heterogeneous Multi-Processing) aislando los P-Cores de la GPU (Iris Xe) tiene un impacto de cuello de botella tendiente a cero, utilizaremos el concepto de **Intensidad Aritmética (Arithmetic Intensity)**.

## 1. El Escenario de Hardware (La Frontera Física)
- **Ancho de banda RAM (DDR4 2666MHz):** $\approx 42.6 \text{ GB/s}$ (Límite crítico compartido).
- **Potencia GPU (Iris Xe 96 EUs):** $\approx 1.4 \text{ TFLOPS}$ (Cálculo FP32).
- **Caché Iris Xe / L3:** Intel Iris Xe tiene acceso al Smart Cache unificado (hasta 18MB), suficiente para mantener tensores pequeños sin tocar la RAM externa.

## 2. El Problema Actual (P-Cores en GEMV)
El Forward Pass y QAT normal se basan en multiplicaciones Matriz-Vector (GEMV).
- **Lectura Matriz:** $N^2$ bytes.
- **Operaciones:** $2N^2$ FLOPs.
- **Intensidad Aritmética:** $\frac{2N^2}{N^2} \approx 2 \text{ FLOPs/Byte}$.

Con solo 2 operaciones por cada byte leído, la CPU es **totalmente esclava de la memoria (Memory-Bound)**. Chupa todo el ancho de banda posible (necesita desesperadamente los 42.6 GB/s).

## 3. La Prueba del Optimizador Muon en Vulkan (Iris Xe)
Evaluemos la viabilidad de enviar el Optimizador Muon (Newton-Schulz) a la GPU. El algoritmo purifica la matriz de gradiente haciendo 5 iteraciones de: $X_{k+1} = 1.5 X_k - 0.5 X_k (X_k^T X_k)$.

Tomemos una matriz del modelo de ejemplo: $N = 576$ (`attn_q.weight`).

### A) Costo de Transferencia (Bus de Memoria)
La CPU envía el gradiente inicial a Vulkan, y Vulkan devuelve el final.
- Matriz 576×576 en FP32 = $576 \times 576 \times 4 \text{ bytes} \approx 1.32 \text{ MB}$.
- Transferencia total (Ida + Vuelta) = $1.32 \text{ MB} \times 2 = \mathbf{2.64 \text{ MB}}$.

### B) Costo Computacional (FLOPs internos en la GPU)
Una iteración requiere dos multiplicaciones densas de matrices Matriz-Matriz (GEMM).
- 1 GEMM ($N \times N \times N$) = $2N^3$ FLOPs.
- 2 GEMMs por iteración = $4N^3 = 4 \times (576)^3 \approx 764.4 \text{ MFLOPs}$.
- 5 iteraciones = $5 \times 764.4 = \mathbf{3.82 \text{ GFLOPs}}$.

### C) La Intensidad Aritmética de Vulkan
$$\text{Intensidad} = \frac{\text{Operaciones}}{\text{Bytes Transferidos}} = \frac{3.82 \times 10^9 \text{ FLOPs}}{2.64 \times 10^6 \text{ Bytes}} \approx \mathbf{1447 \text{ FLOPs/Byte}}$$

## 4. Conclusión del Impacto (Minimizado a 0)
Fíjate en la diferencia: La CPU trabaja a **2 FLOPs/Byte**, mientras que la GPU trabajará a **1447 FLOPs/Byte**.

Al enviar la matriz a la GPU (Iris Xe):
1. La GPU ejecuta 3.82 GFLOPs. A 1.4 TFLOPS de potencia teórica, y sumando el factor de eficiencia real en iGPUs, le toma solo unos **~2.7 a 5.0 milisegundos**.
2. Al estar todo el tensor de 1.32 MB metido dentro de la caché, el cálculo de las 5 iteraciones *no toca la RAM externa*.
3. El ancho de banda *real* consumido durante esa fracción de tiempo es apenas la transferencia de ida y vuelta: $2.64 \text{ MB} / 0.0027 \text{ s} \approx \mathbf{0.97 \text{ GB/s}}$.

**Veredicto Final:**
Enviar el optimizador a los Shaders de Vulkan consumirá **apenas un ~2.2% del ancho de banda (0.97 de 42.6 GB/s)**. Esto deja libre el 97.8% del bus para que los P-Cores continúen trabajando ininterrumpidamente. Al ser un proceso asíncrono y de altísima intensidad aritmética, **el impacto en el cuello de botella se minimiza a ~0**.
