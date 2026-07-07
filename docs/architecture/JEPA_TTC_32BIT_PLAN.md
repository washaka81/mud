# JEPA Test-Time Compute (TTC) & 32-bit Unification Plan

Este documento detalla el plan maestro para transformar MUD en un motor de inferencia con **Test-Time Compute determinista** usando la integral de energía de JEPA, permitiendo comprimir la estructura cognitiva de vuelta a un registro puro de 32-bits (`SlimeRegister32`).

---

## 1. Fundamento Matemático (El Controlador Integral)

El objetivo es predecir la estabilización semántica para detener el cómputo en el momento exacto.

**Ecuación Diferencial (Disonancia Cognitiva):**
En cada capa $L$, calculamos el delta (fuerza del resorte JEPA):
$$ \Delta E_L = | \mu_{ctx} - y_{parcial} | $$

**Integral de Acción (Certeza Acumulada):**
Mantenemos un estado acumulativo a lo largo del tiempo de inferencia del token:
$$ S_L = S_{L-1} + \Delta E_L $$

**Condición de Detención (Asíntota):**
Calculamos la derivada de la integral (cuánto aportó la última capa a la certeza global). Si el cambio es minúsculo, el razonamiento convergió:
$$ \text{Si } \left( \frac{\Delta E_L}{S_L} \right) < \epsilon \implies \text{HALT (Emitir Token)} $$

---

## 2. Reestructuración del Hardware (SlimeRegister32)

Dado que la condición de detención garantiza que el bucle terminará antes del desbordamiento, podemos eliminar el costoso `f32` y usar un registro perfecto de 32 bits.

**Diseño Físico (`src/mud/slime.rs`):**
```rust
#[repr(C, packed)]
pub struct SlimeRegister32 {
    // Hemisferio Izquierdo (Discreto / Lógica de Trabajo)
    pub ternary_accum: i16,  // Bits 0-15: Acumulador rápido AVX2

    // Hemisferio Derecho (Continuo / Energía de JEPA)
    pub jepa_energy: u16,    // Bits 16-31: Estado Z y umbral integral
}
```
* **Impacto:** Reduce el consumo de RAM/VRAM a la mitad. Permite que las instrucciones vectoriales AVX2 procesen el doble de registros por ciclo.

---

## 3. Dinámica del Bucle de Inferencia (Test-Time Compute)

El archivo `src/main.rs` (bucle autorregresivo) y `slime_forward.rs` (propagación) dejarán de ejecutar exactamente $N$ capas.

**El Nuevo Bucle "Fluido":**
1. **Inicio de Token:** Inyectamos el embedding y ponemos la integral $S = 0$.
2. **Evaluación de Capa (Hardware):** Ejecutamos bloques GEMV ternarios.
3. **Cálculo de JEPA (Controlador):** Computamos $\Delta E_L$ y actualizamos la integral $S$.
4. **Decisión de Routing:**
    * Si el gradiente es casi 0 $\rightarrow$ **Early Exit**: Rompemos el bucle prematuramente (ej. Capa 5) y escupimos el token al instante.
    * Si alcanzamos la última capa física (Capa 30) y el gradiente sigue alto $\rightarrow$ **Recurrencia Oculta**: Enviamos el registro de vuelta a la Capa 1 y seguimos dando vueltas (Test-Time Compute real) hasta estabilizar.

---

## 4. Fases de Implementación (Roadmap)

### Fase A: Telemetría de la Integral (Modo Pasivo)
* **Objetivo:** No alterar el comportamiento actual, solo medir.
* **Acción:** Modificar `slime_jepa.rs` para calcular la integral $S$ y loguear en qué capa el diferencial cae por debajo de $\epsilon = 0.005$.
* **Validación:** Comprobar que en tokens de relleno (como espacios o puntuación), la integral satura antes de la capa 10.

### Fase B: El Anclaje de Detención (Early Exit)
* **Objetivo:** Acelerar el modelo cortando cálculos innecesarios.
* **Acción:** Modificar el bucle en `src/main.rs` para que haga `break` y retorne los logits si el controlador JEPA da la señal de convergencia.
* **Validación:** El throughput (ops/s) de inferencia debe dispararse enormemente para respuestas conversacionales simples.

### Fase C: Recurrencia de Test-Time Compute (Deep Thinking)
* **Objetivo:** Permitir razonamiento prolongado para tokens difíciles.
* **Acción:** Implementar el bucle `while !converged` que reconduzca los `SlimeRegisters` a través del transformer repetidas veces si la integral no ha saturado tras la última capa.

### Fase D: Compresión a 32-bits (El Santo Grial AVX2)
* **Objetivo:** Aprovechar la barrera térmica de la integral para encoger el tipo de datos.
* **Acción:** Cambiar `SlimeRegister` a `SlimeRegister32`. Reescribir la macro AVX2 en `ternary_gemv.s` para operar con `i16` sabiendo que matemáticamente es imposible que desborde gracias al controlador JEPA.

---

## 5. Próximo Paso Recomendado

Antes de romper el código actual, la forma más segura de empezar es la **Fase A**. 
Consiste en añadir unas pocas líneas de código a `slime_jepa.rs` para calcular la integral y ver empíricamente en qué capa se estabilizan los distintos tipos de tokens (matemáticos vs conversacionales) en el modelo Phi-4-mini.
