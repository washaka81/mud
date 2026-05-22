# MUD Mathematical Audit (v1.0)

Este documento centraliza el análisis de las propiedades estadísticas y matemáticas del motor MUD, garantizando que el modelo mantenga un comportamiento dinámico y saludable a nivel numérico durante el entrenamiento.

## 1. Distribución de Pesos y Desviaciones (Sigmas)

El motor MUD se basa en tensores ternarios (-1, 0, 1). La correcta dispersión de los pesos es vital para asegurar que la red no colapse. Usando la herramienta `deep_math_audit`, monitoreamos cuatro momentos estadísticos:
- **Media (Expectation):** Debe estar cercana a `0.0`. Una media alta (>0.1) indica un sesgo severo (Bias Detected), donde la red depende abrumadoramente de un solo valor (1 o -1), destruyendo la dispersividad del conocimiento.
- **Varianza y Desviación Estándar (Sigma):** Mide la dispersión. Un Sigma muy bajo indica que casi todos los pesos se han vuelto `0`, lo que representa una "amnesia ternaria". 
- **Asimetría (Skewness):** Una asimetría alta (>0.5) significa que los pesos positivos y negativos no están balanceados. MUD compensa esto con la pérdida de balance en los enrutadores.
- **Curtosis (Kurtosis):** Mide la pesadez de las colas. Si el modelo es "Leptokurtic" (Curtosis > 1.0), los valores extremos (-1 y 1) predominan anormalmente frente al 0, sugiriendo una cuantización demasiado rígida que puede impedir la generalización.

## 2. Autocorrelaciones Espaciales y Temporales

La herramienta `deep_statistics` mide ahora la **Autocorrelación (Lag 2)**.
- **Objetivo:** Detectar si el sistema de predicción entra en bucles repetitivos ("ciclos recurrentes").
- **Diagnóstico:** Si la autocorrelación es `> 0.1`, el modelo está reciclando patrones. Esto puede originarse en el algoritmo Mixture of Experts (MoE) si un grupo pequeño de expertos asume toda la carga e impide la exploración de los demás, causando que la ruta de inferencia se auto-alimente y produzca bucles en el texto.
- **Corrección Propuesta:** Penalización de autocorrelación. Agregar a la función de pérdida del modelo una penalidad si las secuencias decodificadas muestran picos recurrentes tempranos.

## 3. Trazas de Confianza (Confidence Traces)

Basándonos en la **Entropía de Shannon**, hemos definido una métrica continua denominada **Traza de Confianza**.
- Se calcula como `1.0 - (Entropía Media / Entropía Máxima del Vocabulario)`.
- Si la entropía del lote de salida es muy alta, el modelo está adivinando y la confianza decae (< 60%).
- **Aplicación:** El muestreo guiado por confianza (Confidence-Guided Sampling) podrá intervenir en tiempo real. Cuando la traza de confianza decae, se disminuirá la `Temperatura` y el parámetro `Top-K` temporalmente para forzar al modelo a decodificar usando solo los tokens más probables, impidiendo alucinaciones profundas y la selección de tokens residuales (`<unk>`).

## 4. Deltas de Entrenamiento (Pérdidas y Gradientes)

En el MoE distribuido, la función de pérdida por sí sola no cuenta toda la historia. Observar el **Delta** (la tasa de cambio de la pérdida o gradiente a lo largo del tiempo) revela "mesetas" de aprendizaje (Plateaus).
- Si el Delta se aproxima a cero, pero la función de pérdida general aún no es satisfactoria, los expertos están estancados en óptimos locales.
- **Mecanismo Dynamic Delta Compensation:** Proponemos ajustar dinámicamente el coeficiente auxiliar de balance `aux_coeff`. Si el Delta de pérdida es bajo (meseta), se inyecta ruido adicional en el `Top-K gating` o se eleva el `aux_coeff` para forzar a que expertos "inactivos" comiencen a procesar gradientes de nuevo y saquen a la red del estancamiento.
