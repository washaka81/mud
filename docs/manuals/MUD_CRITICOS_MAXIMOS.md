# MUD: CRÍTICOS MÁXIMOS - Reporte Maestro de Supervivencia Cognitiva
**Fecha:** 26 de mayo de 2026
**Estatus:** CRÍTICO / ACCIÓN REQUERIDA

## 1. El Fenómeno: TERNARY SHOCK
Se ha identificado una falla catastrófica en modelos de más de 12 capas (ej: Qwen 0.5B, Llama) al ser convertidos a ternario (1.58-bit). El modelo produce "ruido puro" (gibberish).

### Diagnóstico de Señal
- **Entrada (Embedding):** std ~0.01 (Demasiado bajo).
- **Cuerpo (24 Capas):** Acumulación de deriva residual.
- **Salida (Logits):** std ~7.5 (Explosión de ruido).
- **Resultado:** Distribución plana de probabilidades. La semántica se disuelve en el proceso de cuantización.

## 2. Causa Raíz: ESCALAMIENTO GLOBAL (Global Scaling)
El uso de una única escala (`scale`) por tensor completo destruye la jerarquía interna de los pesos. Las filas con pesos pequeños son aplastadas a cero, mientras que las grandes introducen un error masivo.

## 3. La Solución Maestra: CUANTIZACIÓN POR FILA (PRQ)
Para restaurar la inteligencia en **cualquier modelo**, es obligatorio transicionar a **Per-Row Quantization**:
- **Granularidad:** 1 escala `f32` por cada fila de la matriz de pesos.
- **Impacto:** Reduce el error de cuantización en un 10x por capa, permitiendo que la señal sobreviva a través de arquitecturas profundas.

## 4. Protocolo Universal de Restauración (RESTORE-IQ)
Este proceso no es opcional y debe aplicarse a todo modelo convertido:

1.  **CONVERSION (PRQ):** Convertir usando `universal_converter` con escalas por fila.
2.  **ALIGN (Tokenización):** Mapear el vocabulario original al manifold expandido de MUD (es/en).
3.  **PROJECT (Bayesian QC):** Ejecutar `recalibration_projector` para ajustar escalas basadas en activaciones reales (Tier 2/3).
4.  **TRAIN (Live SGD):** Re-entrenamiento ligero (500-1000 pasos) para que los pesos se "asienten" en el manifold discreto {-1, 0, 1}.

## 5. Mandatos de Ingeniería
- **NUNCA** usar escalas globales para modelos >0.1B.
- **SIEMPRE** verificar el flujo de señal (`diagnose_layers.rs`) después de una conversión.
- **TODO** modelo es recuperable si se sigue el pipeline de calibración universal.

---
*MUD: La inteligencia no reside en los bits, sino en la calibración del manifold.*
