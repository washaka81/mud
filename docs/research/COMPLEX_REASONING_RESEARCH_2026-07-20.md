# Research: Complex-Valued Calculations for MUD Reasoning (2026-07-20)

**Fuente:** web research (arxiv 2025–2026) + estado actual de `src/mud/cmud.rs` (L-14).
**Objetivo:** aterrizar cálculos complejos concretos para potenciar el *razonamiento* de MUD
sobre el kernel C-MUD ya existente, mapeando el item de orbit **C-MUD × log-gas** (Dyson
circular ensemble / CFT) a `cmud.rs` (E1 phase-repulsion, E3 contour-rotation).

---

## 1. Estado actual (kernel C-MUD, `cmud.rs`)

El kernel ya tiene:
- `ComplexF32 { re, im }`, `GaussTernary { wr, wi } ∈ {-1,0,1}` (9 estados).
- `gauss_mul` (⇒ `Y_R = X_R W_R − X_I W_I`, `Y_I = X_R W_I + X_I W_R`), `project_hermitian` (mHC bala `‖h‖_C ≤ R`).
- `phase_delta`, `wave_collapse(h) = Re·(1+tanh|Im|)·cos θ` (lectura a LM-head).
- `ThinkingState` con bucle de thinking `think_step_stub` (rotación 90° + proyección Hermitiana) y phase-lock por EMA de `ω`.

**Faltante para razonamiento** (lo que la literatura 2026 resuelve y este doc propone):
1. Atención fase-coherente compleja (en vez de competencia softmax).
2. Repulsión de fase tipo CUE (E1) para evitar colapso de fase.
3. Composición de capas por free probability / R-transform (pieza CFT).
4. Rotación de contorno / tiempo-complejo (E3) para "continuar" el razonamiento.

---

## 2. Estado del arte (web, 2026)

| Trabajo | Idea central | Dato duro |
|---------|--------------|-----------|
| **Phase-Coherent Transformer (PCT)** arxiv 2605.10123 | QK complejo L2-normalizado + compuerta real suave; atención *no competitiva* (preserva fase) | supera softmax Transformer y su par complejo en retrieval largo; sin colapso con profundidad |
| **ComplexFormer (CMHA)** arxiv 2505.10222 | Euler per-head: `exp[i(Adapt(AS)+ΔP)]`; rotación diferencial adaptiva | GSM8K 21.3% vs RoPE 18.5%; MBPP/HumanEval ↑ |
| **PRISM** arxiv 2512.01208 | tokens como fasores `z=re^{iθ}`; *interferencia sustractiva* (destructiva) como primitiva de razonamiento; colapso = wavefunction→vocab | PPL 6.06 (fase-constreñido) vs FNet 9.87; "semantic phase compass" |
| **CAWN / COLM / PAM** | LM nativo complejo; razonamiento lógico discreto *emerge* de interferencia de ondas | CAWN 150M bate Pythia-160M en PIQA/ARC-E; PAM escala con exponente −0.15 vs −0.12 |
| **Complex-Time NN (CTNN)** algorithms 19050334 | `T = t + iτ`; ImT<0 memoria, ImT=0 presente, ImT>0 imaginación | Teorema Separación Expresiva: O(1) vs Ω(Δ) acceso temporal |
| **Dyson log-gas / CUE** (Forrester, Mehta) | `λⱼ=e^{iθⱼ}`, densidad `∝|Δ(e^{iθ})|²`; gas de Coulomb inverso-temperatura β | base de la repulsión de fase y free probability |

---

## 3. Matemática concreta a adoptar

### 3.1 Atención fase-coherente compleja (PCT / CMHA)
Codifica Q,K como fasores sobre la esfera L2: `q = e^{iθ_q}`, `k = e^{iθ_k}` (norma 1).
Score independiente de magnitud (token-non-competing):
```
score(m,n) = Re[ q_m^* · k_n ] = cos(θ_q_m − θ_k_n)          (1)
```
y se aplica una **compuerta real suave** `g(·)` (PCT) o rotación adaptiva
`exp[i(Adapt(AS_{mn}) + ΔP_{mn})]` (CMHA, ComplexFormer):
```
Score = Σ_j Re[ exp(i·φ_{mn,j}) ] ,  φ_{mn,j} = Adapt_j(AS_{mn}) + ΔP_{mn,j}   (2)
```
Salida por cabeza: mezcla fase-preservante de V complejo. **Mapeo MUD:** Q,K,V complejos
se derivan del estado `ComplexF32` del bucle de thinking; la salida se proyecta Hermitiana
antes del siguiente paso. Esto reemplaza (en modo `MUD_CMUD_THINK`) la atención real por
una donde la *relación* vive en la fase, no en la magnitud.

### 3.2 Repulsión de fase tipo CUE (E1 — anti-colapso)
En el Circular Unitary Ensemble (β=2) los autovalores `λⱼ=e^{iθⱼ}` tienen densidad conjunta
```
p(θ) ∝ ∏_{j<k} |e^{iθ_j} − e^{iθ_k}|²
      = ∏_{j<k} 4·sin²((θ_j − θ_k)/2)                      (3)
```
El factor de Vandermonde **repeles fases cercanas** (dos modos con la misma fase "colisionan"
⇒ energía infinita). Esto es exactamente el antídoto al *rejection cascade* / colapso de modo
que sufre MUD. **Regularizador de repulsión de fase** sobre el vector de fases `{θ_i}` del
estado oculto:
```
R(θ) = −∑_{i<j} log |e^{iθ_i} − e^{iθ_j}|²
     = −∑_{i<j} [ log 4 + 2·log|sin((θ_i−θ_j)/2)| ]        (4)
```
Gradiente (empuja fases aparte):
```
∂R/∂θ_i = −∑_{j≠i} cot((θ_i − θ_j)/2)                      (5)
```
En el bucle de thinking (`think_step`) se añade un paso de *phase-repulsion* con tasa `η_rep`
antes de `project_hermitian`, para que los modos no colapsen en un solo punto de fase. Junto con
la bala Hermitiana (mHC) se mantiene el estado distribuido en el círculo.

### 3.3 Free probability / R-transform (pieza CFT / Virasoro)
Dos capas complejas (circulares) **libres** A,B se componen por convolución libre aditiva:
```
R_{A⊕B}(z) = R_A(z) + R_B(z)                  (R-transform aditivo)   (6)
S_{AB}(z)    = S_A(z) · S_B(z)                 (S-transform multiplic.) (7)
```
donde el **transform de Cauchy** del estado oculto (autovalores `λⱼ` de la matriz de
correlación del hidden) es
```
G(z) = (1/N)·Σ_j 1/(z − λ_j),    R(G(z) + 1/z) = z            (8)
```
**Uso en MUD:** tras `τ` pasos de thinking, la distribución espectral límite del hidden se
calcula *analíticamente* vía R-additividad (en vez de sumar realmente capas). Esto da una
"lectura de salud" del bucle: si `G(z)` se acerca a un solo polo ⇒ colapso (detectable antes de
`wave_collapse`). Conexión CFT: las Dyson-Schwinger del modelo de matriz = restricciones de
Virasoro ⇒ la invariancia conforme del bucle de thinking (las rotaciones de fase son justo
transformaciones conformes del círculo).

### 3.4 Rotación de contorno / continuación analítica (E3 — imaginación)
La continuación analítica por rotación de contorno (Trefethen; ej. `∫e^{−az²−z}dz`):
```
f(a>0) = ∫_0^∞ e^{−az²−z}dz  ──(rotar contorno π/4)──▶  f(a<0) válido
```
i.e. rotar el contorno en el plano complejo extiende el dominio de validez sin perder información
(Cauchy: la integral a lo largo del contorno es igual a la original). **Mapeo MUD:** el estado
complejo `h` del thinking loop vive en un disco; para "continuar" el razonamiento a un régimen
imaginado (prospección, CTNN ImT>0) se rota el contorno:
```
h → h · e^{iφ_rot}                                 (9)
```
`φ_rot` es un ángulo de paso (fijo o `learnable`). Por construcción `‖h·e^{iφ_rot}‖ = ‖h‖`
(rotación preserva norma Hermitiana ⇒ no rompe mHC). Esto es el "contour rotation" del orbit:
explora una rama de razonamiento analíticamente continuada antes del colapso a vocab.

### 3.5 Tiempo-complejo (CTNN) para memoria/imaginación
Extiende el eje de thinking a tiempo-complejo `T = τ + i·σ`:
- `σ<0` (ImT<0): recuperación / memoria (fases pasadas).
- `σ=0`: cómputo presente.
- `σ>0` (ImT>0): imaginación / prospección (rama continuada, E3).

MUD puede añadir `sigma` al `ThinkingState` y dejar que `think_step` recorra `σ` (un eje
ortogonal) para *explorar* alternativas sin comprometer el hidden real (un manifold de lookahead).

---

## 4. Adiciones propuestas a `cmud.rs` (firmas concretas)

```rust
/// (3.1) Atención fase-coherente compleja: score = cos(Δθ) por cabeza + gate suave.
pub fn phase_coherent_attn(
    q: &[ComplexF32], k: &[ComplexF32], v: &[ComplexF32],
    gate: impl Fn(f32) -> f32, out: &mut [ComplexF32],
);

/// (3.2, E1) Repulsión de fase tipo CUE; devuelve R(θ) y aplica gradiente ∇R.
pub fn cue_phase_repulsion(phases: &mut [f32], eta_rep: f32) -> f32;

/// (3.3, CFT) Transform de Cauchy + R-transform de una lista de autovalores λ∈ℂ.
pub fn cauchy_transform(lambdas: &[ComplexF32], z: ComplexF32) -> ComplexF32;
pub fn r_transform_add(ra: &[f32], rb: &[f32]) -> Vec<f32>; // composición libre

/// (3.4, E3) Rotación de contorno (continuación analítica) de todo el estado.
pub fn contour_rotate(h: &mut ComplexF32, phi: f32); // h *= e^{i phi}

/// (3.5, CTNN) Paso de tiempo-complejo: añade eje σ (imaginación/memoria).
pub fn complex_time_step(st: &mut ThinkingState, alpha: f32, sigma: f32);
```

`wave_collapse` (ya existente) queda como el *readout* (wavefunction → vocab), coincidiendo
con PRISM.

---

## 5. Plan de validación CORTA (cada ítem con test unitario)

| Ítem | Test (segundos) |
|------|-----------------|
| `phase_coherent_attn` | `cos(π/2)=0` para fases ortogonales; `cos(0)=1` para iguales; norma de salida finita |
| `cue_phase_repulsion` | dos fases iguales ⇒ `R` grande (penaliza); fases equidistribuidas ⇒ `R` pequeño; ∇ empuja aparte |
| `cauchy_transform` | para `λ={e^{iθ_j}}` uniformes, `G(i0)` ≈ media; `r_transform_add` con duas CUE da densidad límite conocida |
| `contour_rotate` | `‖h·e^{iφ}‖ == ‖h‖` (preserva mHC); fase rota `φ` |
| `complex_time_step` | cambiar `σ` no altera `‖h‖`; converge phase-lock con `σ>0` (prospección) |

Todos corren en `cargo test --lib` (sin entrenamiento largo). `MUD_CMUD_THINK=1` sigue siendo
el gate opt-in; el production path (`SlimeRegister` f32, P-02) NO se toca.

---

## 6. Referencias
- PCT: arxiv 2605.10123 (Hioki, 2026).
- ComplexFormer / CMHA: arxiv 2505.10222.
- PRISM: arxiv 2512.01208.
- CAWN: arxiv 2604.04250; COLM: zenodo 20118033; PAM: arxiv 2604.05030.
- CTNN: Algorithms 2026, 19(5), 334.
- Dyson log-gas / CUE / CFT: Forrester *Log-Gases and Random Matrices*; Mehta *Random Matrices*; Dumitriu-Edelman (matrix models for circular ensembles); Weyl integration formula; Dyson-Schwinger = Virasoro.
- Contour rotation / analytic continuation: Trefethen *Numerical analytic continuation* (AAA); math.stackexchange 2060291.
- Base MUD: `src/mud/cmud.rs` (L-14), `docs/research/COMPLEX_THINKING_MANIFOLD.md`, `docs/research/CMUD_LOGGAS_FEASIBILITY.md`.

---

*Integrable en `GEMINI.md` / `AGENTS.md` como orbit F+ (C-MUD log-gas). No contradice políticas
(P-02 SSOT f32 producción; C-MUD es kernel de investigación opt-in). No toca ratatui ni el path
de inferencia real salvo bajo `MUD_CMUD_THINK=1`.*
