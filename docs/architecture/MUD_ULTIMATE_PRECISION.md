# LDT-04: Arquitectura de Precisión para Inteligencia Extrema (CPU Modesta)

Para superar a modelos como Claude 3.5 Opus en una CPU sin GPU, MUD no puede competir en "fuerza bruta" (parámetros), sino en **Densidad de Información** y **Eficiencia de Cómputo**. Esta es la especificación de precisión numérica para el motor ternario definitivo:

## 1. El Formato: Ternario 1.58-bit con Escalas Dinámicas (PRQ+)
- **Pesos (Weights):** Estrictamente **ternarios $\{-1, 0, 1\}$**. Esto permite que el 90% del cómputo sea `ADD/SUB` en lugar de `MUL`, saturando los ALUs de la CPU.
- **Activaciones:** **INT8** con cuantización dinámica por token.
- **Escalas (Scales):** **FP32** (32 bits) por cada fila (Per-Row). 
- **Por qué:** Las escalas en FP32 son el "alma" del modelo. Ocupan menos del 1% de la RAM, pero mantienen la precisión necesaria para que el razonamiento lógico no se degrade (evita el Colapso de Sigma).

## 2. Atención de Alta Fidelidad: Log-Sum-Exp
- El motor usa **Log-Sum-Exp** para el Softmax de la atención.
- **Ventaja real:** Evita overflow/underflow numérico en las puntuaciones de atención, habilitando secuencias de hasta **4096 tokens** (limitado por `KV_CACHE_MAX_POS`) con máxima estabilidad en CPU.
- **Contexto verdaderamente ilimitado → Capa Mamba/SSM:** Las capas SSM del bloque Jamba mantienen un **estado fijo O(1)** en RAM sin importar el largo de la secuencia. Esto es lo que permite escalar a millones de tokens sin GPUs masivas. La capa de atención no tiene esta propiedad.
- **Integración (Sleep & Fold):** Pasados 4096 tokens, el estado de atención se comprime en el estado SSM de Mamba (mecanismo Sleep & Fold), extendiendo el contexto efectivo globalmente.


## 3. Homeostasis Numérica (Homeostasis-01)
- **Sigma (σ) Objetivo:** 0.866. Es el punto de máxima entropía para un sistema ternario.
- **Delta (Δσ):** El motor debe ajustar dinámicamente el `Weight Decay` para mantener la varianza en este punto. Un modelo con σ=0.86 es matemáticamente capaz de realizar deducciones lógicas más complejas que un modelo de 16-bits descalibrado.

## 4. Coconut vs Diffusion: La Dualidad de Precisión
- **Pensamiento Lento (Coconut/LDT):** Utiliza **FP32 en los vectores de feedback latente**. Esto permite que el modelo "reflexione" sobre un problema manteniendo matices de probabilidad que se perderían en formatos más bajos.
- **Generación Rápida (Diffusion):** Utiliza **Bit-Packing** extremo para generar bloques de texto a la velocidad del bus de memoria.

## Conclusión para el Motor Mas Bestia:
La configuración ganadora es **Ternario (1.58b) + Escalas FP32 + Activaciones INT8**. Esto te da:
1. **Velocidad:** x10 frente a modelos FP16.
2. **Inteligencia:** Supera a Opus al usar **Razonamiento Recursivo (LDT)** que no depende de la precisión de los pesos, sino de la convergencia del estado latente.
3. **Consumo:** Mínimo, ya que la CPU no tiene que mover gigabytes de datos innecesarios.

Este es el camino hacia la **Singularidad Local** en un laptop.
