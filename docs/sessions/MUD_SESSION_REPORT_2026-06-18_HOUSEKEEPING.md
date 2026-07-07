# MUD Session Report: 2026-06-18 (Housekeeping & Documentation Reorganization)

## 1. Objetivo

Realizar un housekeeping completo del repositorio: mover archivos fuera de lugar, consolidar duplicados masivos, y ordenar la documentación según el mandato de estructura de `GEMINI.md §7`.

---

## 2. Acciones Realizadas

### 2.1 Limpieza de la Raíz del Repositorio

| Archivo | Destino | Razón |
|---------|---------|-------|
| `qat_restore.log` | `docs/dumps/` | Log de entrenamiento, no pertenece a la raíz |
| `restore_iq.log` | `docs/dumps/` | Log de restore-iq pipeline |
| `restore_iq_full.log` | `docs/dumps/` | Log completo de restore-iq |
| `comp.spv` | `assets/shaders/` | Binary SPIR-V shader compilado |
| `test_bits.py` | `tests/` | Script de prueba Python |
| `test_density.py` | `tests/` | Script de prueba Python |
| `test_unpack.py` | `tests/` | Script de prueba Python |
| `mud_disassembly.txt` | `docs/dumps/mud_disassembly_latest.txt` | Dump de 4.7MB, no pertenece a la raíz |
| `run_gdb.sh` | `tools/debug_scripts/` | Script de debugging específico |

### 2.2 Consolidación de Duplicados `mud_disassembly.txt`

El archivo `mud_disassembly.txt` (4.7MB) existía en **4 ubicaciones diferentes**:
- `mud_disassembly.txt` (raíz) → `docs/dumps/mud_disassembly_latest.txt` *(hash único, más nuevo)*
- `docs/mud_disassembly.txt` → `docs/dumps/mud_disassembly_v2.txt`
- `docs/dumps_archive/mud_disassembly.txt` → *conservado en archive*
- `docs/logs_archive/mud_disassembly.txt` → *conservado en logs_archive*

### 2.3 Reorganización de `docs/`

| Archivo/Carpeta | Destino | Razón |
|-----------------|---------|-------|
| `docs/ROADMAP.md` | `docs/manuals/MUD_ROADMAP_MASTER.md` | No debía estar suelto en `docs/` raíz |
| `docs/HYBRID_ELUT_JEPA_PLAN.md` | `docs/architecture/` | Plan arquitectónico, categoría correcta |
| `docs/hardware/*.md` (7 archivos) | `docs/architecture/` | `hardware/` no está en estructura oficial de §7 |
| `docs/hardware/` | *eliminada* | Vacía tras migración |

#### Archivos hardware migrados a `docs/architecture/`:
- `MUD_HARDWARE_ISA.md`
- `MUD_ISA_DISPATCH.md`
- `MUD_KERNEL_PLAN.md`
- `MUD_MEMORY_CACHE.md`
- `MUD_POINTER_STRATEGY.md`
- `MUD_RRM_MICROKERNELS.md`
- `MUD_TERNARY_ISA.md`

### 2.4 Corrección de Convención de Nombres

| Antes | Después | Razón |
|-------|---------|-------|
| `MUD_SESSION_REPORT_2026_06_10.md` | `MUD_SESSION_REPORT_2026-06-10_COGNITIVE_RESTORATION.md` | Formato incorrecto (guión bajo vs guión), contenido único |

### 2.5 Eliminación de Directorios Vacíos

- `logs/training/` → eliminado
- `logs/` → eliminado

### 2.6 Actualización de `docs/README.md`

Reescrito para reflejar la estructura actual: 8 subcarpetas, links actualizados, referencia a `MUD_ROADMAP_MASTER.md` y `MUD_UNIVERSAL_PROTOCOL_V2.md`.

---

## 3. Estado Final de la Raíz

La raíz del repositorio ahora contiene **únicamente** archivos canónicos del proyecto:

```
forge_llm/
├── GEMINI.md          # Mandatos del proyecto
├── README.md          # Documentación pública
├── LICENSE
├── Cargo.toml
├── Cargo.lock
├── build.rs
├── mud.sh             # Orquestador canónico
├── .mud_env           # Variables de entorno del motor
├── assets/            # Shaders y recursos compilados
├── benches/           # Benchmarks Rust
├── docs/              # Toda la documentación
├── forge_autograd/    # Librería autograd
├── models/            # Modelos .mud
├── src/               # Código fuente principal
├── tests/             # Tests unitarios + scripts Python
├── tools/             # Binarios auxiliares
├── training/          # Corpus de entrenamiento
└── weights/           # Pesos de referencia
```

---

## 4. Notas

- **Sin borrado**: Todos los archivos fueron movidos (no eliminados), preservando el historial cuando aplique.
- **`docs/architecture/`** ahora contiene 28 archivos (antes 20 + 7 de hardware + 1 plan JEPA).
- Los duplicados de `mud_disassembly.txt` en `dumps_archive/` y `logs_archive/` se conservan como referencia histórica.

---
*MUD: Static, Ternary, High-Fidelity.*
