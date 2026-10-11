# MUD Security — Hardening Guide (Sprint 6)

> **Sprint:** 6 (Security Documentation)
> **Fecha:** 2026-07-24
> **Estado:** ✅ Completado

---

## 1. Build hardening

### 1.1 Production builds

```bash
# ✅ Production build (safe — no unsafe features)
cargo build --release

# ❌ Research only (enables RCE paths)
cargo build --release --features unsafe-rce
cargo build --release --features unsafe-cli
```

### 1.2 Cargo.toml feature flags

```toml
# CRITICAL: default = [] (no unsafe features)
[features]
default = []
unsafe-rce = []    # RlvrCritic::evaluate_rust_code
unsafe-cli = []    # TerminalSandbox::execute_command
```

### 1.3 Compiler flags

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"  # No unwinding — prevents exception-based exploits
```

---

## 2. Runtime hardening

### 2.1 Environment variables

| Variable | Default | Seguridad |
|---|---|---|
| `MUD_USE_VULKAN` | `1` | Iris Xe blacklisted in `probe_gpu()` |
| `MUD_GPU_GEMV` | `0` | Disabled by default (Iris Xe SIGSEGV) |
| `MUD_TRAIN_FORCE_VULKAN` | `0` | Must be explicitly `1` to override |
| `MUD_CMUD_THINK` | `0` | Disabled by default (avoid distortions) |
| `MUD_DEBATE_LEARN` | `1` | Persists shadow→MUD (data integrity) |

### 2.2 Memory safety (P-00/P-01)

- **P-00:** Raw pointer mastery in hot path
  - Todas las operaciones ASM usan `// SAFETY:` comments
  - `AlignedArray<T>` 64-byte aligned para SIMD
  - `MemoryPointerEntry` 64-byte aligned (Continuity Table)

- **P-01:** Zero allocation en hot loops
  - `SlimeWorkspace` pre-alloca todos los buffers
  - `FenwickF32::with_capacity` + `reset_zeroed` para reuso
  - `MUD_SLIMEX=1` → `slimex_cells` pre-alloc (zero-size si ≠ 1)

### 2.3 Path traversal protection

```rust
// src/mud/workspace_agent.rs
let root_canon = fs::canonicalize(&root_dir)?;
// check is a simple starts_with (no syscall).
if !path.starts_with(&root_canon) {
    return Err(anyhow!("path traversal blocked"));
}
```

---

## 3. Input validation

### 3.1 `.mud` file format

```rust
// src/mud/mod.rs
// Magic bytes: "MUD\x01"
if header.magic != MUD_MAGIC {
    return Err(MudError::InvalidMagic);
}
// Validate tensor counts, shapes, offsets
for tensor in &header.tensors {
    if tensor.offset + tensor.size > data_region.len() {
        return Err(MudError::OffsetOutOfBounds);
    }
}
```

### 3.2 Model dimensions (P-13)

```rust
// NEVER hardcode model dimensions
// Infer from metadata: layers, hidden, ffn, heads
let hidden = header.metadata.hidden_size
    .ok_or(MudError::MissingDimension("hidden_size"))?;
```

### 3.3 Tokenizer bounds

```rust
// src/mud/inference.rs
// Prevent panics out of range
let tid = token_id.min(vocab_size - 1);
```

---

## 4. ASM kernel safety checklist

| Kernel | Safety check | Status |
|---|---|---|
| `ternary_gemv` | `n` must be multiple of 8 | ✅ |
| `ternary_gemv_4rows` | `stride` must be ≥ n_u32 | ✅ |
| `ternary_gemv_8rows` | `stride` must be ≥ n_u32 | ✅ |
| `adam_step` | All arrays same length `n` | ✅ |
| `sgemm_abt` | `c` must be m×n | ✅ |
| `slimex_cell_v5` | `cells` must be 64-byte aligned | ✅ |
| `apply_scales` | `scales` must be 8-aligned | ✅ |

---

## 5. GPU safety

### 5.1 Iris Xe blacklist (2026-07-18)

```rust
// src/vulkan/ash_backend.rs
fn probe_gpu() -> Option<GpuInfo> {
    let device_name = get_device_name();
    if device_name.contains("Intel") && device_name.contains("Iris Xe") {
        // Blacklisted: SIGSEGV in submit_and_wait
        return None;
    }
    Some(GpuInfo { ... })
}
```

### 5.2 Vulkan SIGSEGV mitigation

- `MUD_USE_VULKAN=0` forces CPU
- `MUD_TRAIN_FORCE_VULKAN=1` overrides (expert only)
- `probe_gpu()` returns `None` for blacklisted devices

---

## 6. Audit trail

### 6.1 Security-relevant commits

| Fecha | Commit | Descripción |
|---|---|---|
| 2026-07-18 | FIX D | `MudFile::save` offset absoluto → `data_base + offset` |
| 2026-07-18 | FIX D | Read de paridad ECC también usaba offset absoluto |
| 2026-07-18 | FIX | Iris Xe SIGSEGV → blacklist en `probe_gpu()` |
| 2026-07-20 | FIX | STE deadzone → `MUD_QAT_LR` banner |
| 2026-07-20 | FIX | Telemetry TUI → `[TELEM]` a archivo + stderr |
| 2026-07-23 | FIX | Q2_0→ELUT+PRQ per-row scale averaging → F32 dequant |
| 2026-07-24 | FIX | Sprint 5 clippy warnings (19) |

### 6.2 Tests de seguridad

```bash
# Run all security-related tests
cargo test --lib security
cargo test --lib pointer_audit
cargo test --lib test_path_traversal
cargo test --lib test_absolute_path
cargo test --lib test_mud_file_load_rejects_invalid_magic
```

---

## 7. Incident response

Ver `docs/seguridad/INCIDENT_RESPONSE.md` para el playbook de respuesta a vulnerabilidades.
