# Security Policy

## Threat Model

Forge LLM is a local, offline LLM inference and training engine. It runs entirely on the user's machine — no network calls are made during inference or training by default.

### Attack Surface

| Component | Risk | Mitigation |
|---|---|---|
| `.mud` file loader | Malicious file could trigger OOB reads | `MudFile::load` validates tensor **counts, ranks, string lengths and per-tensor extent vs mmap**, plus **metadata↔shape coherence** (`validate_model_shapes`) and **checked arithmetic** (`MAX_TENSOR_DIM`, no wrap on 32/64-bit overflow) — R-30/R-31, 2026-10-12. `diagnose_model` para health checks |
| GGUF parser (`src/gguf/mod.rs`) | Malicious `.gguf` → OOB / OOM / stack overflow | **R-32 (2026-10-13):** cursor con bounds-checks (truncado ⇒ `Err`, nunca panic), caps SEC-H1 (counts, key/name/string, rank, dim ≤ 1<<28), arrays ≤ 4M con reserva incremental, nesting ≤ 8, `general.alignment` validado (≠0/pot2/≤1M), offsets con `checked_add` + tamaño exacto por tipo. Proptest fuzz sobre bytes arbitrarios |
| ASM kernels | Raw pointer misuse | **R-36 (2026-10-13):** 532 sitios `unsafe` productivos justificados (contratos `//! SAFETY:` de módulo + `// SAFETY:` per-site; alto riesgo siempre adyacente). Gate `./mud.sh unsafe-audit` ⇒ **0 sin comentario**, en `mud.sh ci` y GitHub workflow. Nota: `unsafe-rce`/`unsafe-cli` son *nombres de feature flags*, no operaciones `unsafe` |
| Vulkan backend | Driver exploit (SIGSEGV on Iris Xe) | `probe_gpu()` blacklists Iris Xe in-binary; `MUD_USE_VULKAN=0` forces CPU |
| Feature flags | RCE via `unsafe-rce` / `unsafe-cli` | **Riesgo ESTRUCTURAL, no "0 RCE":** ambos RCE están feature-gated (`default = []` en `Cargo.toml`); el build de producción/CI usa features por defecto y compila stubs que devuelven `PermissionDenied`. Habilitarlos exige build explícito (`--features unsafe-rce` / `unsafe-cli`), sólo research/eval |
| Skills autónomos | Lectura de ficheros derivada de texto generado | **R-37 (2026-10-13):** `skills/` **no se compila** hoy (dead code, decisión en R-60), por lo que la superficie no está activa. `data_analysis.rs` ya no usa `fs::read_to_string`: lee exclusivamente vía `AgentWorkspace::read_file` (confinado al root; sin workspace montado ⇒ inerte). Al reactivar `skills/` (R-60) el confinamiento ya está en el código |

> **Revisión 2026-10-13:** R-30/R-31 (loader), R-32 (GGUF), R-33 (ECC), R-34 (worm), R-35 (GEMV), R-36 (`unsafe-audit` = 0), R-37 (RCE estructural + confinamiento), R-38 (symlink escape) cerrados con evidencia en [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) §12. La revisión 2026-10-11 de la auditoría [`MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) §4 queda cubierta por las olas 1-2.

## Feature Flags (CRITICAL)

Production builds MUST be `cargo build --release` (no feature flags).

| Feature | What it enables | Risk |
|---|---|---|
| `unsafe-rce` | `RlvrCritic::evaluate_rust_code` — compiles & executes Rust code via `rustc` | **RCE** — only for research/eval |
| `unsafe-cli` | `TerminalSandbox::execute_command` — runs `bash -c` | **RCE** — only for research/eval |

```bash
# ✅ Production build (safe)
cargo build --release

# ❌ Research only (enables RCE paths)
cargo build --release --features unsafe-rce
cargo build --release --features unsafe-cli
```

## Path Traversal Protection

`AgentWorkspace` (used by `cmud_train` and skills) validates all file paths (R-38, 2026-10-13):
- Rejects absolute paths and any `..` component before touching the FS
- Canonicalizes the **deepest existing ancestor** (follows symlinks) and requires it to stay under the workspace root; a **dangling symlink is rejected** explicitly instead of being treated as a "new file"
- Leaf opens use **`O_NOFOLLOW`** on Unix (defence-in-depth against a TOCTOU symlink swap)
- Tests: `test_absolute_path_rejected`, `test_path_traversal_blocked`, `test_dangling_symlink_rejected`, `test_symlink_to_outside_rejected`, `test_symlink_dir_ancestor_escape_rejected`, `test_symlink_inside_root_ok`
- Symlinks that resolve **inside** the root remain allowed

## Reporting Vulnerabilities

If you discover a security issue:
1. **Do not** open a public GitHub issue
2. Email the maintainers with a description and reproduction steps
3. Do not exploit the vulnerability in public

## Known Issues

- **Iris Xe SIGSEGV (2026-07-18):** Intel Vulkan driver crashes in `submit_and_wait` when dispatching QKV GEMV. Blacklisted in `probe_gpu()`. Use `MUD_USE_VULKAN=0` to force CPU. See `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` §2.

## Sprint 6-7 Documentation

Security documentation has been expanded in `docs/seguridad/`:

| File | Sprint | Descripción |
|---|---|---|
| `THREAT_MODEL.md` | 6 | Threat model, attack surface, DREAD analysis |
| `HARDENING.md` | 6 | Build/runtime hardening, input validation, audit trail |
| `INCIDENT_RESPONSE.md` | 7 | Playbook de respuesta, runbooks, checklist de PRs |

```bash
# Security tests
cargo test --lib security
cargo test --lib pointer_audit
cargo test --lib test_path_traversal
cargo test --lib test_absolute_path
cargo test --lib test_mud_file_load_rejects_invalid_magic
```
