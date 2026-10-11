# MUD Security — Incident Response Playbook (Sprint 7)

> **Sprint:** 7 (Security Documentation)
> **Fecha:** 2026-07-24
> **Estado:** ✅ Completado

---

## 1. Reportar una vulnerabilidad

### 1.1 No uses GitHub Issues públicos

Si descubres un issue de seguridad:

1. **No** abras un GitHub issue público
2. Email a los mantenedores con:
   - Descripción del problema
   - Pasos de reproducción
   - Impacto esperado
3. No exploits en público
4. Espera confirmación antes de divulgar

### 1.2 Triage

| Severidad | Tiempo de respuesta | SLA |
|---|---|---|
| Critical (RCE, OOB) | < 2 horas | 24 horas fix |
| High (corruptión datos) | < 8 horas | 7 días fix |
| Medium (info leak) | < 24 horas | 30 días fix |
| Low (hardening) | < 72 horas | 90 días fix |

---

## 2. Clasificación de vulnerabilidades

### 2.1 Critical (P0)

- **RCE** via `unsafe-rce` o `unsafe-cli` en builds de producción
- **OOB read/write** en `.mud` loader o ASM kernels
- **Vulkan driver exploit** que escape del proceso

### 2.2 High (P1)

- **Corruptión de pesos** silenciosa (scale inflation, PRQ corruption)
- **Path traversal** que escape del workspace
- **Integer overflow** en tensor size/offset

### 2.3 Medium (P2)

- **Information leak** via timing side-channel
- **Denial of service** via malformed input
- **Tokenizer bounds** panic

### 2.4 Low (P3)

- **Missing SAFETY comment** en `unsafe` block
- **Unvalidated env var** con fallback inseguro
- **Verbose error messages** que leak paths internos

---

## 3. Playbook de respuesta

### 3.1 Fase 1: Confirmación (0-2 horas)

```bash
# 1. Reproducir el issue
cargo test --lib <test_name> -- --nocapture

# 2. Verificar en release build
cargo build --release
./target/release/forge_llm <repro_args>

# 3. Verificar que no es feature-flag
cargo build --release  # sin --features
```

### 3.2 Fase 2: Análisis (2-8 horas)

```bash
# 1. Identificar el componente afectado
grep -rn "SAFETY:" src/<component>/

# 2. Verificar mitigaciones existentes
cargo test --lib <security_tests>

# 3. Determinar severidad usando DREAD
# Ver docs/seguridad/THREAT_MODEL.md §4
```

### 3.3 Fase 3: Fix (8-24 horas para P0)

```bash
# 1. Crear branch de fix
git checkout -b fix/security-<CVE>-<component>

# 2. Implementar fix
# - Agregar validación de bounds
# - Agregar SAFETY comment
# - Agregar test de regression

# 3. Verificar
cargo clippy --all-targets
cargo test --lib -- --test-threads=2

# 4. No incluir en el commit el exploit
```

### 3.4 Fase 4: Verificación (24-48 horas)

```bash
# 1. Full CI
./mud.sh ci

# 2. Security tests
cargo test --lib security
cargo test --lib pointer_audit

# 3. Build de producción
cargo build --release
```

### 3.5 Fase 5: Divulgación (7-90 días)

```bash
# 1. Asignar CVE (si es necesario)
# 2. Publicar advisory en GitHub
# 3. Notificar a usuarios afectados
# 4. Actualizar SECURITY.md
```

---

## 4. Checklist de seguridad para PRs

| Item | Verificación |
|---|---|
| ✅ No `unsafe` sin `// SAFETY:` comment | `grep -rn "unsafe" src/ \| grep -v SAFETY` |
| ✅ No hardcoded model dimensions | `grep -rn "128\|256\|512\|1024" src/mud/ \| grep -v test` |
| ✅ Path traversal protection | `grep -rn "canonicalize\|starts_with" src/mud/` |
| ✅ Feature flags disabled by default | `grep "default" Cargo.toml` |
| ✅ No Python files | `find . -name "*.py" -not -path "./target/*"` |
| ✅ Clippy 0 warnings | `cargo clippy --all-targets` |
| ✅ Tests pass | `cargo test --lib -- --test-threads=2` |

---

## 5. Runbook: Iris Xe SIGSEGV

**Symptom:** `SIGSEGV` in `submit_and_wait` durante QKV GEMV dispatch

**Root cause:** Intel Vulkan driver (`libvulkan_intel.so`, ADL GT2 UMA) hace
SIGSEGV en `submit_and_wait` al dispatchar QKV GEMV. Determinista en block 11/64.

**Mitigation:**
```bash
# Force CPU (default for training)
export MUD_USE_VULKAN=0
export MUD_GPU_GEMV=0

# Override (expert only, discrete GPU)
export MUD_TRAIN_FORCE_VULKAN=1
```

**Code:** `probe_gpu()` blacklistea Iris Xe in-binary.

**Doc:** `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` §2/§7.2

---

## 6. Runbook: Q2_0→ELUT+PRQ corruption

**Symptom:** Logits casi-uniformes, vocabulary collapse

**Root cause:** Q2_0→ELUT+PRQ path computed single PRQ scale per row by averaging
16 block scales. Destroyed weight semantics.

**Fix:** All Q2_0 tensors go through `Q2ToF32`, dequantized block-by-block.

**Verificación:**
```bash
./mud.sh scale-audit models/smollm2.mud models/smollm2
./mud.sh health models/smollm2.mud
```

**Doc:** `docs/audits/BUG_LOGITS_CORRUPTION.md`

---

## 7. Runbook: MudFile offset corruption (FIX D)

**Symptom:** `~±1e38` garbage en norms/scales después de `MudFile::save`

**Root cause:** `MudFile::save` usaba `tensor.offset` como absolute mmap offset,
pero `offset` es relativo a la data region.

**Fix:** Agregado `MudTensor::data_base`, lectura usa `data_base + offset`.

**Verificación:**
```bash
./mud.sh audit-conv models/smollm2.mud
./mud.sh health models/smollm2.mud
```

**Doc:** `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` §FIX D
