# Bug: Corrupción de Logits en Generate()

## Síntoma
Los logits se calculan correctamente en el path Float32 pero se corrompen inmediatamente después, resultando en valores garbage (~-1.2×10⁹) que impiden inferencia coherente.

## Evidencia Experimental

```
DEBUG: Float32 path, hidden_size=2560
DEBUG: i=0, val=-3.6889
DEBUG: i=1, val=-7.4281
DEBUG: After Float32 loop, logits[0..5] = [-3.6889, -7.4281, -2.2421, -6.8472, -3.7072]  ✓ CORRECTO

DEBUG: energy=3.4550, temp=0.8211, rep_pen=2.3483

DEBUG: Temp loop start, logits.len()=128000, first=-1000000000.0  ✗ GARbage!
DEBUG: After temp, logits[0..5] = [-1217861900.0, ...]  ✗ GARbage!
```

## Cronología de la Corrupción

1. **Float32 path** (líneas 2178-2208): Escribe logits correctamente
   - `logits_guard[i] = val` funciona perfecto
   - Valores confirmados: [-3.69, -7.43, -2.24, -6.85, -3.71]

2. **Sanitize logits** (líneas 2227-2235): ADQUIERE NUEVO GUARD
   - `let mut logits_guard = ws.logits.write()`
   - Itera sobre `logits_guard.iter_mut()`
   - Debería solo modificar NaN/Inf

3. **Temp loop start**: PRIMERA LECTIÓN DESPUÉS DE SANITIZE
   - `logits[0] = -1000000000.0` ← ¡YA CORRUPTO!

## Hipótesis

### H1: Doble adquisición de write lock corrompe memoria
El path Float32 mantiene un `UnifiedWriteGuard::Cpu` que crea un slice desde raw pointer.
Cuando se libera y se adquiere otro guard inmediatamente, algo en la creación del slice
(`std::slice::from_raw_parts_mut(ptr, b.len)`) falla.

**Evidencia a favor:**
- La corrupción ocurre ENTRE el drop del primer guard y la adquisición del segundo
- `UnifiedBuffer::write()` usa `unsafe` para crear slices desde raw pointers
- El valor -1e9 sugiere memoria no inicializada o reutilizada

**Evidencia en contra:**
- El AlignedBuffer debería mantener memoria estable mientras exista
- No hay deallocation entre guards

### H2: Sanitize itera sobre elementos no escritos
El path Float32 escribe `logits_guard.len()` elementos (vocab_size=128000).
Sanitize itera sobre TODOS los elementos del buffer.

Si el buffer tiene MÁS de 128000 elementos, los elementos extra podrían contener
garbage que al ser iterado corrompe los elementos válidos.

**Evidencia a favor:**
- `logits.len()=128000` confirmado por debug
- Pero ¿qué pasa si el AlignedBuffer fue allocado con capacidad mayor?

**Evidencia en contra:**
- `AlignedBuffer::new(size)` allocata exactamente `size * 4` bytes
- `b.len = size` es exacto

### H3: Alias de mutable references
El código tiene múltiples `let mut logits_guard = ws.logits.write()` en scopes anidados.
Si el primer guard no se libera completamente antes de adquirir el segundo, podría
haber aliasing de referencias mutables.

**Evidencia a favor:**
- Rust permite esto con RwLock, pero el comportamiento unsafe podría violarlo
- `UnifiedWriteGuard::Cpu(&'a mut [f32])` tiene lifetime `'a` que podría extenderse

### H4: Memoria no inicializada en AlignedBuffer
`AlignedBuffer::new()` usa `alloc_zeroed`, pero ¿qué pasa si hay un bug en el layout?

```rust
let layout = std::alloc::Layout::from_size_align(size * 4, 64).unwrap();
let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
```

Si `size * 4` overflowea usize, el layout sería incorrecto.

**Evidencia a favor:**
- 128000 * 4 = 512000 bytes, no hay overflow
- Pero ¿y si el size es incorrecto en otro lado?

## Próximos Pasos

1. **Verificar que logits tiene exactamente 128000 elementos:**
   ```rust
   eprintln!("DEBUG: logits addr={:p}, len={}", logits_guard.as_ptr(), logits_guard.len());
   ```

2. **Imprimir dirección del pointer en cada guard:**
   ```rust
   eprintln!("DEBUG: Guard A ptr={:p}", logits_guard.as_ptr());
   // Después del drop
   eprintln!("DEBUG: Guard B ptr={:p}", logits_guard2.as_ptr());
   ```

3. **Verificar si Sanitize está modificando elementos:**
   ```rust
   let mut modified = 0;
   for logit in logits_guard.iter_mut() {
       if logit.is_nan() || logit.is_infinite() {
           *logit = -1e4;
           modified += 1;
       }
   }
   eprintln!("DEBUG: Sanitize modified {} logits", modified);
   ```

4. **Probar deshabilitando Sanitize:**
   Comentar líneas 2227-2235 para ver si la corrupción persiste.

5. **Probar con Mutex en vez de RwLock:**
   Si es un bug de RwLock, Mutex debería funcionar.

## Impacto

- **Crítico:** Impide cualquier inferencia con modelos BitNet
- **Scope:** Solo afecta path Float32 (output projection)
- **Modelos afectados:** BitNet-b1.58-2B-4T.mud y cualquier modelo denso no-MoE

## Workaround Temporal

Ninguno conocido. La inferencia produce garbage irrecuperable.