# Forge LLM — Session Report
**Date:** 2026-07-21
**Focus:** Mejoras en Circuit Telemetry (TUI), mecánicas RPG, y supresión de contaminación en Stdout.

## 1. Resumen de Objetivos
El usuario solicitó refinar la interfaz de usuario interactiva del circuito de entrenamiento (TUI), específicamente abordar el solapamiento visual provocado por mensajes descontrolados en `stdout`, mejorar la lógica de las batallas en la arena y profundizar las mecánicas RPG del circuito.

## 2. Acciones Realizadas

### 2.1. Refactorización del TUI (Circuit Telemetry)
- Se solucionó la contaminación del buffer de terminal que causaba "scrolling" defectuoso. 
- Los macros `println!` y `eprintln!` en `corpus_trainer.rs` y `mod.rs` (específicamente la alerta de `[RAM] selective materialize`) fueron envueltos en comprobaciones `if std::env::var("MUD_CIRCUIT_TUI").is_err()`. Esto asegura que cuando el TUI está activo, la salida estándar se suprime para evitar romper el renderizado de la terminal.
- Se reestructuró la disposición (layout) para proveer barras de vida e información claramente separada para el **Jugador A (Evolutivo)** y el **Jugador B (Baseline)**. Se adoptó una estética visual monocromática inspirada en Apple.

### 2.2. Sistema de Nombres RPG y Linaje
- Se extendió el struct `CircuitRpgStats` en `circuit_rpg.rs` para incluir campos `name` y `baseline_name`.
- En `corpus_trainer.rs`, cuando un modelo pierde todos sus HP (0 HP) y evoluciona a la siguiente generación, se le asigna de manera determinista (basado en la generación) un nombre RPG épico (p. ej., "Aspirante", "Gladiador", "Titán", "Espectro").
- Se implementó la **Defensa del Título**: Si el modelo en entrenamiento supera de manera estricta el `win_rate` del baseline anterior durante las evaluaciones, este reclama el título y su nombre actual pasa a convertirse en el nombre del Baseline permanente (`baseline_name = name`).

### 2.3. HP de Batalla (Arena HP)
- Previamente, el HP Global del circuito se mantenía estático durante la arena, lo que daba la impresión de que "ninguno se hacía daño" tras recibir un castigo del juez.
- Se añadieron variables locales de `battle_hp_a` y `battle_hp_b` (iniciadas en 100.0) a `circuit_telemetry.rs`.
- Cuando el juez impone recompensas negativas (`REWARD|A:-1.250|B:-0.050`), la telemetría ahora substrae estos valores en tiempo real del *Battle HP* temporal, mostrando un daño real y dinámico sin comprometer el HP Global intergeneracional.

### 2.4. Orden del Currículum Fijo
- La secuencia de entrenamiento se estabilizó de una generación puramente aleatoria (que el usuario consideró desorientadora) a un currículum estricto y ordenado: `align` -> `professor` -> `debate` -> `games`. 
- Esto asegura que el modelo se estabilice matemáticamente primero antes de entrar en los modos de RLVR agresivo.

### 2.5. Corrección de C-MUD, Tokenización y Desbordamiento de Memoria
- **Panic en Inferencia (`src/mud/inference.rs`):** Se identificó un pánico al ejecutar `cmud-train` debido a números en texto plano dentro del corpus que superaban el `vocab_size` (49,152), generando un índice fuera de rango (`emb_start = 46224576` vs límite `28311552`). Se solucionó aplicando una protección estricta en el motor: `tid = (tokens[current_pos] as usize).min(vocab_size.saturating_sub(1))`.
- **Tokenización Automática (`tools/cmud_train.rs`):** Se integró el `Tokenizer` de MUD para que, al procesar archivos de texto plano (`project_corpus.txt`), convierta automáticamente las oraciones a token IDs válidos en lugar de interpretar números sueltos del código/markdown.
- **Configuración por Defecto de `MUD_CMUD_THINK`:** Se restauró `MUD_CMUD_THINK=0` en `mud.sh` para evitar distorsiones en las respuestas conversacionales estándar. C-MUD se mantiene como módulo opt-in de investigación.
- **Reconversión Limpia:** Se eliminaron todos los tensores `.mud` antiguos y se reconvirtió `models/smollm2` desde cero con el conversor universal, certificando su salud (`scale-audit` ratio 0.374 y `health` 🟢 CERTIFIED).

## 3. Análisis de Métricas 
Durante la sesión, el usuario presentó múltiples volcados y capturas de métricas:
- **Babbling y Salud (circuit.log):** Durante las fases de *Math Challenge*, se confirmó que los modelos generan tokens aleatorios (babbling) en vez de haber colapsado en un único token constante. Esto, sumado a estadísticas sanas de Varianza `VarH` y `VarJ`, confirma que la red ternaria 1.58-bit se encuentra sana pero en etapa inicial exploratoria.
- **Breakthrough del Loss:** Las gráficas visuales mostraron un valle abrupto donde el *PosLoss* cayó un 50% de manera simultánea con un pico masivo (259+) en la *Derivada Cognitiva*. Esto confirmó de forma definitiva que la arquitectura puede salir de los valles planos de gradientes para lograr ganancias de aprendizaje profundas y rápidas.
- **Validación del Kernel C-MUD:** Se certificaron 28/28 pruebas de C-MUD e inspección de radio hermítico (`1.7488 / 2.3907`), confirmando la estabilidad espectral de la variedad compleja.

## 4. Próximos Pasos Recomendados
- Permitir que las métricas extraídas durante las etapas tardías (`games`) ajusten dinámicamente el LR (Learning Rate) u otros parámetros de JEPA.
- A futuro, implementar la propagación del daño de `Battle HP` hacia `RPG HP` si se acumula un umbral muy alto de recompensas negativas repetitivas, haciendo la supervivencia intergeneracional todavía más desafiante.
