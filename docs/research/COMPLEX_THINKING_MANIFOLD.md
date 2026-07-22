# COMPLEX_THINKING_MANIFOLD.md
# Phase 17: Complex-Valued Thinking Manifold (C-MUD)

## 1. Fundamentos Algebraicos: Enteros de Gauss Ternarios
Para preservar la regla de oro de la inferencia en 1.58 bits sin latencia masiva, los pesos de la red ya no son escalares $W \in \{-1, 0, 1\}$, sino **Enteros de Gauss Ternarios** $W \in \mathbb{C}$:
$$ W = W_R + iW_I \quad \text{donde} \quad W_R, W_I \in \{-1, 0, 1\} $$
Esto permite a un solo peso aprender **9 estados discretos**. La multiplicación de la activación (que fluye en $f32$ como $X = X_R + iX_I$) por un peso ternario se reduce matemáticamente a sumas y restas cruzadas, perfectas para AVX2:
$$ Y_R = X_R W_R - X_I W_I $$
$$ Y_I = X_R W_I + X_I W_R $$
El hardware (P-Cores) procesaría $Y_R$ y $Y_I$ en registros paralelos, o bien alternando la instrucción `VPMADDUBSW` de AVX2, costando computacionalmente solo un factor de $\sim 2x$ en vez de $4x$ (aprovechando que los pesos siguen siendo enteros).

## 2. El Ciclo de Oscilación Latente (Thinking Loop)
En el *Inference Bucle Autoregresivo* tradicional, procesar la capa $L_{30}$ escupe los *logits* para el token $T_{n+1}$.
Bajo el **C-MUD**, introducimos la dimensión temporal interna $\tau$. Al recibir un prompt difuso, el modelo desvía su flujo al componente Imaginario:
$$ h_{\tau=0} = \text{Embed}(T_n) + i\cdot 0 $$
Durante $\tau = 1, 2, ..., N_{max}$:
1. $h_{\tau} = \text{Layer}(h_{\tau-1})$ (El Transformer evalúa el estado complejo)
2. **mHC Projection (Manifold-Constrained Hyper-Connections):** En lugar de un radio euclidiano unidimensional, proyectamos sobre el límite Hermitiano para evitar que la energía imaginaria explote:
   $$ ||h_\tau||_C^2 = h_\tau h_\tau^* = \text{Re}(h_\tau)^2 + \text{Im}(h_\tau)^2 \le \text{radius}^2 $$

## 3. Dinámica del JEPA: Detección Termodinámica de "Phase-Lock"
¿Cómo escapamos del bucle de pensamiento de forma matemática, sin tokens especiales que generen falsos positivos?
La puerta termodinámica JEPA monitoreará la **Fase (Ángulo)** de cada dimensión latente:
$$ \theta_{\tau} = \arctan2(\text{Im}(h_\tau), \text{Re}(h_\tau)) $$
El JEPA mide la *Derivada de la Entropía Angular*:
$$ \omega_{\tau} = || \theta_{\tau} - \theta_{\tau-1} ||_1 $$
Cuando la red está "pensando", la fase de los vectores rota salvajemente ($\omega$ alta). A medida que el gradiente interno converge a una solución latente estática, las rotaciones se acoplan. 
**Condición de Escape (Resonancia):**
$$ \text{EMA}(\omega_\tau) < \varepsilon \quad (\text{ej. } \varepsilon = 10^{-3}) $$
Cuando esto se cumple, el oscilador ha alcanzado una conclusión matemática.

## 4. Colapso de Función de Onda (Wave-Function Collapse)
Para regresar a un estado verbal y determinista, aplicamos una contracción unitaria que inyecte la información estructural que se construyó en el eje imaginario ($i$) directamente sobre el espectro semántico real ($\mathbb{R}$):

$$ h_{\text{final}} = \text{Re}(h_{\text{lock}}) \cdot (1 + \tanh(|\text{Im}(h_{\text{lock}})|)) \cdot \cos(\theta_{\text{lock}}) $$

**Propiedades de este colapso:**
1. $\tanh(|\text{Im}|)$ asegura que los descubrimientos abstractos modifiquen la magnitud semántica sin saturar el formato $f16$.
2. $\cos(\theta)$ penaliza fuertemente a los vectores cuya fase haya colapsado ortogonalmente al vocabulario real (ruido o incertidumbre) e impulsa aquellos vectores que se alinearon cerca del eje Real ($\pm 0^\circ$ o $\pm 180^\circ$).
3. La variable colapsada entra ahora al $LM\_Head$ (capa final RMS Norm + Projection) **garantizando** una distribución de Logits estable que el vocabulario `BPE` puede decodificar deterministamente.

## 5. Integración (L-14 foundation — 2026-07-16)

**Shipped in `src/mud/cmud.rs` (does not replace live f32 `SlimeRegister`):**
- [x] `ComplexF32`, `GaussTernary` (9 states), `gauss_mul` / `gauss_mac`
- [x] Hermitian mHC projection, phase $\omega$ EMA, wave-function collapse
- [x] `ThinkingState` stub loop + `maybe_think_collapse` (`MUD_CMUD_THINK=1`)
- [x] Hook after `apply_output_norm` (opt-in only)

**Still future:**
- [ ] Full-network complex GEMV AVX2 (`ternary_gemv_complex_avx2.s`)
- [ ] Replace production register with dual-f16 packing (not while f32 SSOT holds)
- [ ] Vulkan `complex_jepa_phase.comp`
