# Apunte: CFT / Log-Gas — contorno complejo, Vandermonde y operador de vértice (JHEP)

**Origen:** lectura de una página de artículo de física teórica (marca de agua JHEP — Journal of
High Energy Physics). CFT / mecánica estadística / gas de Coulomb unidimensional.

**Propósito de este doc:** referencia de conceptos útiles para razonar sobre dinámicas
estocásticas en manifold (JEPA/mHC) y regularización de integrales en el trainer de Forge LLM.
No es trabajo de ingeniería del proyecto; es trasferencia de ideas.

---

## 1. La página analizada

Los autores calculan un valor de expectación de un operador de vértice:
`⟨e^{2iαϕ(0)}⟩` en una teoría de campos conforme, usando un gas de Coulomb / ensamble circular.

### Figura 2 — deformación de contorno
- Contorno original `C = C(1) ∪ C(2) ∪ C(3)`: sube por el eje imaginario desde `−∞i` hasta 0,
  recorre el eje real `0 → β/(2π)`, y baja de vuelta a `−∞i`.
- Contorno deformado `C_R`: las líneas se "suben" a `Im(c) = +R` (esquinas en `iR` y
  `β/(2π)+iR`).

### Ecuación (4.6) — medida de Vandermonde / ensamble circular de Dyson
```
∫ ∏_i dθ_i ∏_{j<k} |e^{iθ_j} − e^{iθ_k}|^{2β/2}
   = (2π)^n · Γ(1 + nβ/2) / Γ(1 + β/2)^n
```
Es el módulo cuadrado del Jacobiano (factor de Vandermonde) del **Circular Ensemble (CUE) de
Dyson** con parámetro de acoplamiento `β/2`. No es "relacionado con" matrices aleatorias — *es*
la medida exacta del ensamble circular. Normalización canónica del log-gas.

### Ecuación (4.7) — serie hipergeométrica confluente → integral
```
∑_{l=0}^∞ (2π Λ̃_B / Γ(1+β/2))^l · Γ(1 + lβ/2) / Γ(l+1)
   = ∫_{R>0} dt e^{− (2π Λ̃_B / Γ(1+β/2)) · t^{β/2} − t}
```
Truco: `Γ(1 + lβ/2) = ∫_0^∞ t^{lβ/2} e^{−t} dt` lineariza la `l` en el exponente → permite
sumar la serie geométrica `∑ (algo · t^{β/2})^l` e intercambiar suma/integral. Es una
`₁F₁` (hipergeométrica confluente) estándar.

### Ecuación (4.8) — objetivo: valor de expectación
```
⟨e^{2iαϕ(0)}⟩ = ∫_C dc e^{i c (2α − β + β^{-1})}
                  ∫_{R>0} dt e^{− (2π Λ_B / Γ(1+β/2)) · e^{iβ c} · t^{β/2} − t}
```
Integral doble: interna sobre `t` (de 4.7, con `e^{iβc}` en el exponente por el operador de
vértice), externa sobre `c` a lo largo de `C`. `α` = carga del operador; `β, Λ_B` = parámetros
físicos.

---

## 2. Correcciones / precisiones a la lectura ingenua

1. **La deformación `C → C_R` NO es estética.** El integrando tiene `e^{i c (2α−β+β⁻¹)}` y la
   integral de `t` introduce `e^{iβ c t}` con `t > 0`. En el eje imaginario **negativo**
   (`Im(c) < 0`) ese factor **diverge**; subir a `Im(c) = +R` **cambia el signo de la parte
   imaginaria del exponente** y da decaimiento exponencial → la integral de `t` *converge*.
   Es un contour rotation / Wick rotation, no solo teorema de Cauchy. Sin `C_R` la integral no
   existe.

2. **(4.6) es la medida exacta del Circular Ensemble de Dyson**, no una integral "a menudo
   relacionada". El factor `∏_{j<k} |e^{iθ_j}−e^{iθ_k}|²` es el Vandermonde del log-gas 1D.

3. **(4.7) es `₁F₁` confluente**; la conversión serie→integral es legítima y estándar.

4. **(4.8)** `⟨e^{2iαϕ(0)}⟩` en CFT = coeficiente de dos-punto / dimensión de operador (o
   anomalía de carga). `α` controla la carga del inserto en el origen.

---

## 3. Puente con Forge LLM (ideas transferibles)

- **Estabilización del attractor JEPA (OU/EMA):** el `z_next = 0.9·z + 0.1·y_norm` y el gate
  `sigmoid(v_jepa)` son análogos a un proceso de Ornstein-Uhlenbeck en manifold. El log-gas
  (4.6) también es un OU en el espacio de ángulos con repulsión de Coulomb — la "cohesión"
  (`cog`) del trainer es isomorfa a la energía del gas. VarJ ~0.07 = temperatura del ensemble.
- **Regularización de integrales en el trainer:** el contour-rotation que hace converger (4.8)
  es el mismo principio que el `scale_up` adaptivo (fix 37 del histórico) y el clamp de sigma
  (`σ ∈ [10%,90%]`): desplazar el "contorno" de cómputo para que el exponente no diverja.
- **mHC radius `√hidden`:** análogo al radio de exclusión de Coulomb entre partículas del gas.

> No implementar nada de esto aún — es mapa conceptual para futuras mejoras de estabilidad
> (stream F/K/G/H de `MUD_IMPROVEMENTS_POST_AE.md`).

---

*Capturado 2026-07-18. Lectura validada por opencode; correcciones de interpretación anotadas en §2.*
