# Investigación: Entrenamiento Pulido, Eficiente y de Bajo Coste para LLMs

**Fecha:** 10 de Junio de 2026
**Materia:** Optimización de Entrenamiento, PEFT, GaLore, y QLoRA
**Objetivo:** Reducir drásticamente el coste y la memoria VRAM/RAM requerida para alinear y afinar LLMs, aplicable al pipeline de restauración (QAT) del motor MUD.

---

## 1. El Paisaje del Entrenamiento de Bajo Coste (2026)

Entrenar o afinar (fine-tuning) modelos de lenguaje grandes históricamente requería clústeres de GPUs carísimas (como las H100). Sin embargo, la convergencia de algoritmos matemáticos para la reducción de rango y la cuantización de hardware ha democratizado este proceso, permitiendo afinar modelos de más de 7B de parámetros en una simple tarjeta gráfica de consumidor (ej. RTX 4090) o incluso en CPUs con memoria unificada.

### 1.1 QLoRA (Quantized Low-Rank Adaptation)
El estándar de facto de la industria. 
*   **Mecanismo:** Congela los pesos originales del modelo en 4-bit (NF4) y entrena únicamente dos matrices pequeñas (adaptadores de bajo rango, $A$ y $B$).
*   **Impacto:** Reduce el consumo de memoria hasta en 20 veces. Permite al modelo aprender nuevos conocimientos o estilos sin alterar la estructura fundamental del modelo base.

### 1.2 DoRA (Weight-Decomposed LoRA)
Una mejora directa y "gratuita" sobre LoRA estándar.
*   **Mecanismo:** Descompone el entrenamiento en dos componentes: **Magnitud** y **Dirección**. DoRA aplica el concepto de LoRA exclusivamente a la dirección del peso, mientras que entrena un vector de magnitud separadamente.
*   **Impacto:** Logra una convergencia más rápida y evita el sobreajuste (*overfitting*), imitando mucho mejor el comportamiento de un entrenamiento completo (*Full Fine-Tuning*) pero con el coste computacional y de memoria de LoRA.

### 1.3 GaLore (Gradient Low-Rank Projection)
A diferencia de LoRA que entrena "adaptadores paralelos" y mantiene el modelo congelado, GaLore permite **entrenamiento de parámetros completos (Full-Parameter Learning)** pero con uso de memoria PEFT.
*   **Mecanismo:** En lugar de calcular y almacenar matrices de gradientes enormes y estados del optimizador (como Adam) del tamaño exacto de la red, GaLore proyecta los gradientes hacia un espacio de bajo rango *durante* el cálculo de la actualización.
*   **Impacto:** Permite pre-entrenar y alinear modelos desde cero o realizar un aprendizaje profundo sin quedarse sin memoria. 

### 1.4 GRPO (Group Relative Policy Optimization)
Para alinear modelos y dotarlos de capacidades de "razonamiento" (similar a la serie OpenAI o Gemma), el aprendizaje por refuerzo tradicional (RLHF) requería modelos críticos (Critic Models) masivos. GRPO elimina el modelo crítico evaluando estadísticamente grupos de respuestas relativas, abaratando drásticamente el coste de alineación matemática y lógica.

---

## 2. Aplicación Directa al Motor MUD (Curar el "Ternary Shock")

Actualmente, el comando `restore-iq` de MUD utiliza un **Modelo Sombra Completo (Full-Shadow Model)** en FP32 para calcular los gradientes del estimador directo (STE) y curar la "afasia" provocada por la reducción a 1.58-bits. Esto es extremadamente ineficiente en memoria (ej. `Mem: 15.4G` observado en nuestras pruebas).

Para bajar el coste y hacerlo ejecutable en "hardware pobre", debemos implementar las siguientes estrategias:

1.  **GaLore para QAT:** Al aplicar *Gradient Low-Rank Projection* sobre nuestro Modelo Sombra, no necesitaremos almacenar tensores de momentum u optimizadores del tamaño FP32 real. Podremos actualizar los pesos sombra proyectados ocupando una fracción de la RAM.
2.  **DoRA Ternario:** Si en lugar de actualizar todos los pesos, congelamos la grilla ternaria estricta `[-1, 0, 1]` y entrenamos adaptadores DoRA (Magnitud/Dirección) on-the-fly, el modelo podría recuperar su cohesión lingüística en muchísimas menos *epochs* y sin riesgo de corromper la matemática básica ya validada.
3.  **Frameworks Nativos:** Inspirarnos en el código ultra-optimizado de librerías como **Unsloth**, re-escribiendo los kernels de retropropagación del "Corpus Aligner" de MUD usando ensamblador puro (AVX2) para los cálculos de descenso de gradiente proyectado.

---
*Fin del reporte. Estas metodologías son esenciales si el roadmap prioriza hardware restringido.*
