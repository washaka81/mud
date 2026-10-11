# Investigación: Modelos de Lenguaje por Difusión Discreta y DiffusionGemma

**Fecha:** 10 de Junio de 2026
**Materia:** Modelos de Difusión de Texto sin Tokens (Token-Free/Discrete Text Diffusion)
**Referencia de Modelo Analizado:** DiffusionGemma (Google DeepMind)

---

## 1. Introducción: El Paradigma de la Difusión Discreta

Históricamente, los grandes modelos de lenguaje (LLMs) como GPT o la serie original de Gemma han operado bajo un paradigma **autorregresivo (AR)**. En los modelos AR, el texto se genera de forma estrictamente secuencial, de izquierda a derecha, prediciendo el siguiente token basado en la probabilidad condicional de los tokens anteriores.

La **Difusión de Texto Discreta** propone una alternativa radical: en lugar de predecir un token a la vez, el modelo parte de un bloque o "lienzo" (canvas) lleno de ruido (tokens aleatorios o enmascarados) y aprende a refinar iterativamente todo el bloque de texto en paralelo.

### Reto Matemático
A diferencia de las imágenes, que existen en un espacio continuo de píxeles donde el ruido gaussiano es natural, el texto es categórico y discreto. La difusión de texto moderna aborda esto usando:
1. **Enmascaramiento de Tokens (Masking):** Destruir la información reemplazando tokens por un token especial `[MASK]`.
2. **Matrices de Transición Categóricas:** Modelar la probabilidad de que un token mute aleatoriamente a otro token del vocabulario durante el proceso *forward* (corrupción), para luego entrenar la red a revertir ese proceso probabilístico (*reverse denoising*).

## 2. Caso de Estudio: DiffusionGemma

El 10 de junio de 2026, Google DeepMind lanzó **DiffusionGemma**, un modelo fundacional que aplica estos principios a gran escala.

### 2.1 Arquitectura Base
*   **Backbone:** Construido sobre la familia **Gemma 4**, específicamente usando un enfoque de *Mixture-of-Experts* (MoE) de 26B parámetros.
*   **Eficiencia:** Solo activa **3.8B de parámetros** por paso de inferencia.
*   **Cabezal de Difusión (Diffusion Head):** La capa final no es un simple clasificador softmax para el siguiente token, sino un cabezal diseñado para predecir la distribución original de un bloque entero de tokens que actualmente están corruptos.

### 2.2 Modos de Operación (Atención Dual)
El modelo recicla los mismos pesos para operar en dos modalidades distintas:
1.  **Modo Encoder (Prefilling):** Usa atención causal estándar para leer rápidamente el prompt del usuario y consolidar el caché KV.
2.  **Modo Denoising (Decodificador):** Usa **atención bidireccional** sobre el lienzo de generación (hasta 256 tokens simultáneos). Al poder ver el bloque completo (pasado y futuro de la oración que se está formando), el modelo puede aplicar autocorrección iterativa.

### 2.3 Beneficios y Trade-offs
*   **Velocidad:** Al procesar lienzos de 256 tokens en paralelo, DiffusionGemma logra velocidades hasta **4 veces mayores** en GPUs dedicadas en cargas de trabajo de baja latencia/un solo usuario.
*   **Trade-off Computacional:** Intercambia la presión sobre el ancho de banda de memoria (el cuello de botella clásico de los LLMs AR) por un mayor uso intensivo de cómputo puro.

## 3. Estado del Arte y Literatura Clave

Para profundizar en la difusión discreta, la comunidad académica ha consolidado varios papers fundamentales:

1.  **D3PM (Discrete Denoising Diffusion Probabilistic Models) - Austin et al.**
    El paper fundacional que introdujo los modelos de difusión de ruido estructurado en espacios de estados discretos, sentando las bases matemáticas usando matrices de transición.
2.  **SEDD (Score Entropy Discrete Diffusion) - Lou et al.**
    Ganador de premio en ICML 2024, este paper adaptó enfoques basados en *Score-Matching* para datos discretos, estabilizando enormemente el entrenamiento.
3.  **LLaDA (Large Language Diffusion Models) - Li et al.**
    Demostró empíricamente que la difusión discreta puede escalar a tamaños de modelos de miles de millones de parámetros compitiendo con el rendimiento de LLMs autorregresivos.

## 4. Relevancia para el Engine MUD
Actualmente, nuestro motor **MUD (Static, Ternary, High-Fidelity)** utiliza una arquitectura híbrida Mamba/Transformer fuertemente optimizada para inferencia O(1) en caché KV con pesos a 1.58-bits. 

La integración de la **Difusión Discreta** en el motor MUD supondría un cambio de paradigma:
*   En lugar de la predicción secuencial, podríamos alojar un *canvas* preasignado en el `InferenceWorkspace`.
*   El bucle de generación dejaría de ser un ciclo de *append* token a token, pasando a ser un bucle de refinamiento de $T$ pasos sobre el buffer estático.
*   **Desafío MUD:** Evaluar si el ruido categórico iterativo es compatible con los pesos fuertemente cuantizados (PRQ) y la severa restricción de entropía (Ternary Shock) observada en nuestra calibración QAT.

---
*Reporte generado de forma autónoma. Se recomiendan pruebas empíricas antes de iniciar cualquier pivote arquitectónico en Forge LLM.*
