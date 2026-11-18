# Sesión 2026-11-18 — Mensajes de GitHub y fix de CI (CMake 4)

## Objetivo

Rescatar los mensajes pendientes de GitHub y resolver la causa de los fallos
de CI que llegaron entre el 13 y el 18 de noviembre de 2026.

## Mensajes rescatados

Cuenta `washaka81`, repositorio `washaka81/mud`. Seis notificaciones
`ci_activity` (`CheckSuite`), todas sin leer:

| Fecha (UTC) | Run | Título |
|---|---|---|
| 2026-11-13 09:02 | [43397944895](https://github.com/washaka81/mud/actions/runs/43397944895) | CI workflow run failed for main branch |
| 2026-11-14 08:32 | [43576278518](https://github.com/washaka81/mud/actions/runs/43576278518) | CI workflow run failed for main branch |
| 2026-11-15 08:46 | [43696076946](https://github.com/washaka81/mud/actions/runs/43696076946) | CI workflow run failed for main branch |
| 2026-11-16 08:57 | [43835524175](https://github.com/washaka81/mud/actions/runs/43835524175) | CI workflow run failed for main branch |
| 2026-11-17 10:06 | [44061731907](https://github.com/washaka81/mud/actions/runs/44061731907) | CI workflow run failed for main branch |
| 2026-11-18 09:41 | [44278278069](https://github.com/washaka81/mud/actions/runs/44278278069) | CI workflow run failed for main branch |

No había menciones, PRs ni issues pendientes.

## Causa raíz

El runner `ubuntu-latest` ahora resuelve a **ubuntu-26.04** con **CMake 4.x**.
`shaderc-sys 0.8.3` (dependencia transitiva de `vulkano-shaders`) declara
`cmake_minimum_required(VERSION 2.8.12)`, y CMake 4 eliminó la compatibilidad
con versiones anteriores a 3.5:

```text
CMake Error at CMakeLists.txt:6 (cmake_minimum_required):
  Compatibility with CMake < 3.5 has been removed from CMake.
```

Fallaban los dos jobs que compilan el workspace:

- `L-12 health battery` → `Clippy (all-targets, all-features, -D warnings)`.
- `Nightly cert-loss (loss-must-fall gate)` → `Build loss certification bench`.

## Fix

En `.github/workflows/ci.yml`:

```yaml
env:
  ...
  CMAKE_POLICY_VERSION_MINIMUM: "3.5"
```

No se fija CMake 3.x: la variable permite que CMake 4 acepte el mínimo
antiguo de `shaderc-sys` y es ignorada por CMake 3.x.

### Segundo fallo: glslang antiguo vs GCC de Ubuntu 26.04

El primer intento superó CMake, pero el `glslang` empaquetado en
`shaderc-sys 0.8.3` no compila con el GCC de Ubuntu 26.04:

```text
glslang/SPIRV/SpvBuilder.h:236:30: error: 'uint32_t' has not been declared
```

El runner del commit `0ae8da8` falló con ese error en
[44379943075](https://github.com/washaka81/mud/actions/runs/44379943075).
La solución definitiva es no compilar el `glslang` empaquetado: se instala
`libshaderc-dev` del sistema y `shaderc-sys` enlaza
`/usr/lib/x86_64-linux-gnu/libshaderc_combined.a` (el paquete existe en
`resolute/universe`, versión 2026.1-1). El workflow añade en ambos jobs:

```yaml
- name: Install system shaderc
  run: |
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends libshaderc-dev
```

Así se evita por completo la compilación C++ antigua y el problema de CMake.

## Evidencia

| Gate | Comando | Resultado |
|---|---|---|
| Reproducción del fallo | `cmake` 4.5.0 sobre `shaderc-sys 0.8.3/build` sin la variable | exit 1, error de compatibilidad |
| Fix CMake | mismo `cmake` con `CMAKE_POLICY_VERSION_MINIMUM=3.5` | exit 0, configuración completa |
| CI con CMake fix | run 44379943075 | falla después, en `glslang` (`uint32_t` no declarado) |
| Paquete del sistema | `libshaderc-dev` 2026.1-1 en `resolute/universe` | provee `libshaderc_combined.a` |
| Link sistema | `SHADERC_LIB_DIR=/usr/lib cargo clippy --all-targets --all-features -- -D warnings` | 0 warnings, sin compilar glslang |
| Clippy | `CMAKE_POLICY_VERSION_MINIMUM=3.5 cargo clippy --all-targets --all-features -- -D warnings` | 0 warnings |
| Battery | `CMAKE_POLICY_VERSION_MINIMUM=3.5 ./mud.sh ci` | `[L-12] Health battery complete.` (exit 0) |
| Tests lib | `cargo test --lib --release -- --test-threads=2` | **520/0/0** |
| Proptest | `cargo test --lib --release --features proptest -- --test-threads=2` | **527/0/0** |
| Unsafe | `./mud.sh unsafe-audit` | 536 · 0 sin justificar |
| Env | `./mud.sh env-audit` | 0 divergencias |
| Tools | `./mud.sh tools-audit` | 0 huérfanos / 0 fantasmas |

## Notas

- Los seis avisos de GitHub siguen sin marcar como leídos; el contenido quedó
  rescatado en este informe y en los logs locales de la sesión.
- El commit `ci: allow shaderc-sys CMake 2.8 policy under CMake 4` dejó el
  primer fix; el commit `ci: use system libshaderc on ubuntu-26.04` añade el
  paquete del sistema y evita el `glslang` empaquetado.
- El push a `main` vuelve a disparar la battery con `ubuntu-26.04`.
