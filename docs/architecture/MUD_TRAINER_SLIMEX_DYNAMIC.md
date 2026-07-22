# MUD Trainer: SlimeX Dynamic Stack (ShadowExpertBus)

**Fecha:** 2026-07-20
**Tema:** Diseño Arquitectónico para Multi-expert Weighted STE (`MUD_MOE_TRAIN=hash`)
**Decisión:** Uso de `SlimeX` dinámico para acoplar y desacoplar sombras de expertos (Shadows) como una pila.

---

## 1. El Problema

El entrenador actual (`corpus_trainer.rs`) asume una estructura FFN única por capa en el `SlimeWorkspace` y `SlimeLayerShadowF32`.
Para habilitar el **Multi-expert Weighted STE** (donde entrenamos los `top-K` expertos juntos, como dicta `MUD_IMPROVEMENTS_POST_AE.md` para la Fase F+), requerimos calcular el paso hacia adelante (forward) y el paso hacia atrás (backward) para múltiples expertos por capa, fusionando sus deltas según el peso de enrutamiento.

Modificar `SlimeLayerShadowF32` estáticamente para contener N expertos multiplicaría drásticamente el consumo de memoria en entrenamiento (RAM), lo cual va en contra de la política **P-01** y la **Ecuación de Valor**.

## 2. La Solución: SlimeX Dinámico

Dado que trabajamos con punteros directamente mapeados a memoria/disco y módulos acoplables, el entrenador adoptará la filosofía de un **SlimeX dinámico (ShadowExpertBus)**.

### Características del SlimeX

- **Montaje/Desmontaje en Caliente (Stack-like):** Durante el `forward` y `backward` pass de una capa específica, el entrenador montará los `top-k` expertos seleccionados por el enrutador en el bus.
- **Punteros Ligeros:** En lugar de duplicar arreglos masivos de ADAM states y pesos en la capa base de shadow, `SlimeX` operará como una pila de contextos por experto, que se pueden acoplar y desacoplar sobre la marcha leyendo directamente los tensores mapeados en memoria del `.mud`.
- **Integración con `weighted_expert_deltas`:** Una vez recolectados los gradientes de los expertos activos en el bus `SlimeX`, se aplicará la función `weighted_expert_deltas` (ya implementada en `moe_train.rs`) para escalar los deltas según la probabilidad del enrutador antes de realizar el paso del optimizador.
- **Sin Recompute Masivo:** Al montar el contexto de forma dinámica en la memoria pre-asignada, evitamos el sobrecosto de un *Backpropagation Through Time (BPTT)* o recompute completo (Gradient Checkpointing puro) para cada experto.

### Flujo de Ejecución Esperado (Fase G+)

1. **Routing:** `begin_step_hash` devuelve `route = [(expert_id, weight)]`.
2. **Mounting:** El entrenador acopla en el bus `SlimeX` los contextos sombra para los `expert_ids` de la ruta.
3. **Forward (Joint):** Calcula el output de cada experto y emite la suma ponderada (`weighted_sum`).
4. **Backward (Joint):** Recibe el error residual. Pasa el error a cada experto acoplado en el `SlimeX` para calcular sus gradientes.
5. **Weighted STE:** Se llama a `weighted_expert_deltas(grads, weights, lr)` para generar el paso final.
6. **Unmounting:** Se sincronizan los momentos de Adam (si aplican) y se desacoplan de la pila para liberar caché L1/L2.

## 3. Próximos Pasos

- Extender `SlimeLayerShadowF32` para delegar la FFN a un `ShadowExpertBus` opcional.
- Crear la estructura `SlimeX` para manejar los punteros y buffers temporales de `ffn_mid` por experto sin violar **P-01** (cero-alocaciones en loops calientes).

---

## 4. Visión a Futuro: Escalamiento Híbrido

*Nota arquitectónica:* Al estar basado en el manejo directo de punteros mapeados y montaje en caliente (hot-mounting), el diseño del `SlimeX` abre la puerta a un escalamiento masivo sin precedentes en fases futuras del proyecto. Esto permitirá que la pila de expertos se ensamble y desacople bajo demanda **distribuyendo la carga dinámicamente entre la GPU (Vulkan/UMA) y la CPU (AVX2/PCorePool)**. Con este esquema, el entramado físico de cómputo del LLM puede transformarse dinámicamente en tiempo de ejecución, moviendo expertos al acelerador que se encuentre más libre sin cuellos de botella de VRAM estática.
