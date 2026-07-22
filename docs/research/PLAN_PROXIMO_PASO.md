# PLAN MAESTRO: Próximos Pasos (2026-07-20)

## 1. Análisis del Estado Actual

El proyecto Forge LLM (MUD) acaba de lanzar su versión foundation `T-0 = GO` y ha cerrado exitosamente todos los ítems del backlog principal (L-01 a L-15), así como los streams de optimización A-E (Adam moments, MoE load, GEMV auto, Full-seq train, y CSA top-k).

Posteriormente, también se han cerrado los objetivos satelitales en la órbita F-L:
- **F**: QKV unificado en un solo Command Buffer.
- **K**: Certificación de validación de Loss (Unit tests y CI).
- **G**: Multi-expert STE (Round-robin y Hash routing para el Top-1).
- **H**: Long full-seq BPTT + Recompute desde Residual Bank.
- **I**: Cuantización de KV cache en f16 (IEEE half).
- **J**: CSA LSH (SimHash prefilter antes del top-k).
- **L**: Alias canónicos en la conversión.
- **Fixes 2026-07-20**: Telemetría reparada para guardar en logs, tests de gradiente sin cuellos de botella (timeout de `cmud_gradtest` optimizado gracias a caché de forward).

El entorno de pruebas, entrenamiento y evaluación está completamente estable.

## 2. Plan de Desarrollo: "Deep MoE & C-MUD"

Con la fundación asegurada y sana (0 clippy warnings, tests pasando, sin degradación silenciosa), el objetivo para avanzar a la verdadera "fase compleja" se define en el documento `MUD_IMPROVEMENTS_POST_AE.md`. 

A continuación se presenta la lista de tareas priorizada para ejecutar:

### Tarea 1: Multi-expert Weighted STE (Joint Training)
Actualmente, el modo `MUD_MOE_TRAIN=hash` descubre los pesos del enrutador pero **solo entrena al Top-1** experto seleccionado en ese paso.
- **Objetivo**: Integrar `weighted_expert_deltas` (disponible en `moe_train.rs`) directamente en `corpus_trainer.rs`. 
- **Mecanismo**: Cuando `top_k > 1` y estamos en modo Hash, el forward debe rutear a través de los $K$ expertos y combinar sus salidas ponderadas. El backward pass debe generar gradientes para los $K$ expertos y actualizarlos concurrentemente usando sus `route weights`.
- **Restricción**: Mantener intacto el camino denso cuando `MUD_MOE_TRAIN` no es `hash` para no romper el comportamiento verificado.

### Tarea 2: Matrices `W_compress` entrenables para CSA
El filtro SimHash implementado en J es estático.
- **Objetivo**: Añadir parámetros de compresión (`W_compress`) en `.mud` para que el SimHash proyecte los estados ocultos de manera aprendida.

### Tarea 3: Experimentos C-MUD "Log-Gas"
- **Objetivo**: Activar las repulsiones de fase E1 y medir si el modelo en espacio complejo (manifold hermítico) evita el colapso a largo plazo.

## 3. Ejecución

Este plan se activará en cascada, comenzando por el desarrollo y cableado de **Tarea 1**.
