# Security Policy

## Threat Model

Forge LLM is a local, offline LLM inference and training engine. It runs entirely on the user's machine — no network calls are made during inference or training by default.

### Attack Surface

| Component | Risk | Mitigation |
|---|---|---|
| `.mud` file loader | Malicious file could trigger OOB reads | `MudFile::load` validates tensor **counts, ranks, string lengths and per-tensor extent vs mmap**. ⚠️ **No valida la magnitud de las dims ni el cotejo `metadata ↔ tensor.shape`** (SEC-MUD-01/02, R-30/R-31): un fichero con metadata incoherente puede producir OOB read/OOM. `diagnose_model` para health checks |
| GGUF parser (`src/gguf/mod.rs`) | Malicious `.gguf` → OOB / OOM / stack overflow | ⚠️ **Sin cotas hoy** (SEC-GGUF-01, R-32); sólo se usa en herramientas de conversión |
| ASM kernels | Raw pointer misuse | ⚠️ **No todo `unsafe` tiene `// SAFETY:`** — 19 sitios sin comentario en código no-test (mod.rs, ash_backend.rs, kv_compressed.rs, ecc.rs, worm.rs); deuda R-36 |
| Vulkan backend | Driver exploit (SIGSEGV on Iris Xe) | `probe_gpu()` blacklists Iris Xe in-binary; `MUD_USE_VULKAN=0` forces CPU |
| Feature flags | RCE via `unsafe-rce` / `unsafe-cli` | **Disabled by default** (`default = []` in Cargo.toml) — el "0 RCE" es **feature-gated, no estructural** (R-37) |
| Skills autónomos | Lectura de ficheros derivada de texto generado | ⚠️ `skills/data_analysis.rs` lee cualquier `*.csv` existente nombrado en el contexto (SEC-SKILL-01, R-37) |

> **Revisión 2026-10-11:** la auditoría [`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) §4 documenta los hallazgos SEC-MUD-01…SEC-SKILL-01 con evidencia `file:line`. Las filas marcadas ⚠️ son **afirmaciones retiradas** hasta que se ejecute el plan `R-##`.

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

`AgentWorkspace` (used by `cmud_train` and skills) validates all file paths:
- `fs::canonicalize()` resolves symlinks and `..`
- `starts_with(&root_canon)` ensures paths stay within workspace root
- Tests: `test_absolute_path_rejected`, `test_path_traversal_blocked`
- ⚠️ **Known gap (SEC-WS-01, R-38):** un **symlink colgante** dentro del root pasa la validación (el leaf no existe ⇒ no se canonicaliza) y `fs::write` sigue el enlace fuera del workspace. Latente (sin llamadores de producción hoy).

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
