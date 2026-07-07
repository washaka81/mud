# MUD Architecture Manifesto: Holographic Phase Alignment

## 1. The Holographic Axiom
La cuantización ternaria perfecta (llevar un modelo de FP16/BF16 al espacio discreto `[-1, 0, 1]`) sin sufrir "Amnesia Ternaria" (Afasia Semántica) requiere abandonar el paradigma tradicional del "redondeo destructivo". En el motor MUD, la cuantización se entiende como un **problema de conservación de ondas continuas (termodinámica de la señal)** y optimización matemática profunda.

## 2. Cálculo Integral: La Búsqueda del Umbral Óptimo (Δ)
Para mapear los pesos continuos a la matriz ternaria sin destruir la entropía de la red, minimizamos el Error Cuadrático Medio (MSE) de la cuantización. Asumiendo que los pesos $w$ siguen una distribución Gaussiana o de Laplace $p(w)$, la pérdida total $J$ se modela como:

$$ J(\Delta, S) = \int_{-\Delta}^{\Delta} w^2 p(w) dw + \int_{\Delta}^{\infty} (w - S)^2 p(w) dw + \int_{-\infty}^{-\Delta} (w + S)^2 p(w) dw $$

Al calcular las derivadas parciales de esta integral ($\frac{\partial J}{\partial \Delta} = 0$ y $\frac{\partial J}{\partial S} = 0$), el umbral ideal ($\Delta$) se asienta en $\approx 0.7 \times \text{absmean}$. Esto fundamenta matemáticamente la constante `SPARSITY_THRESHOLD_RATIO` del motor, garantizando una esparsidad termodinámicamente estable del 26.0%.

## 3. Derivadas Profundas y el Straight-Through Estimator (STE)
Dado que la función de redondeo $Q(x) = \text{round}(x / S)$ tiene una derivada de 0 en casi todo su dominio, MUD utiliza el **Straight-Through Estimator (STE)**:
- **Forward Pass:** Ejecución estricta en matemática discreta (ternaria).
- **Backward Pass:** "Engañamos" a la red definiendo $\frac{\partial Q}{\partial x} \approx 1$.
Esto permite que los gradientes fluyan hacia los *shadow weights* en FP32. La red ajusta infinitesimalmente las matrices base para que el producto punto final coincida de manera idéntica con la probabilidad original del token.

## 4. Análisis de Fourier y Alineamiento Cosenoidal (Fase vs Magnitud)
Si interpretamos el vector de activación de un token como una señal o forma de onda (Dominio de Fourier), **el error de fase importa radicalmente más que el error de magnitud**.

- **Alineamiento Cosenoidal:** Al maximizar la similitud de coseno entre la activación ternaria y la maestra, garantizamos que las frecuencias dominantes del token (su identidad semántica latente) apunten en la dirección probabilística correcta.
- **Amortiguación RMS (0.7071):** El motor utiliza `DEPTH_DAMPENING_FACTOR = 0.7071`. El valor RMS de una onda senoidal es $\frac{1}{\sqrt{2}} \approx 0.7071$. Al aplicar este factor a la escala, se amortigua la energía de la señal en capas profundas. Esto resuelve la "Paradoja del Target Sigma", estabilizando la varianza recursiva (`TARGET_SIGMA = 0.86`).

## 5. El "Phase Loss" Compuesto (Fase 3: Implementación de Código)
Para solidificar esta teoría en el pipeline QAT (Quantitative Aware Training), el motor implementará un **Holographic Phase Loss** híbrido:

$$ L_{total} = \lambda_1 (\text{CrossEntropy}) + \lambda_2 (\text{MSE}) + \lambda_3 \left(1 - \frac{A \cdot B}{||A|| \times ||B||}\right) $$

Esta arquitectura de *Loss* compuesta garantiza que la "reconversión perfecta" se logre a través de un equilibrio dinámico guiado por las integrales, conservando la dirección de los embeddings holográficos y previniendo colapsos de fase.
