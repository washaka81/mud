# MUD Standard Project Layout (V1-MASTER)

Este es el árbol de directorios oficial del proyecto Forge LLM (MUD). Todos los scripts y binarios esperan esta estructura.

```text
/ (Raíz del Proyecto)
├── mud.sh                # Orquestador Maestro (CLI Único)
├── build.rs              # Configuración de compilación ASM
├── Cargo.toml            # Metadatos del proyecto Rust
├── Cargo.lock
├── .gitignore
├── LICENSE
├── README.md
├── .venv/                 # Entorno virtual Python
│
├── src/                  # Código Fuente (Rust/ASM)
│   ├── main.rs
│   ├── lib.rs
│   ├── asm/              # Kernels AVX2 Ternarios + tests
│   ├── model/            # Transformer, Tokenizer, Inferencia
│   ├── gguf/             # Carga e inspección GGUF
│   ├── vulkan/           # Backend Vulkan zero-copy, clear_caches
│   └── mud/              # Motor MUD, Router MoE, Skills
│       ├── skills/       # 14 skills: language, memory, web_search, etc.
│       ├── inference.rs  # Orquestación de skills
│       ├── auto_trainer.rs
│       ├── routing.rs, store.rs, graph.rs, ingester.rs
│       └── mod.rs
│
├── training/             # Pipeline de Entrenamiento (Python)
│   ├── mud_fast_trainer.py    # Trainer principal (3-comp balance, noisy gating)
│   ├── mud_language_trainer.py
│   ├── mud_cognitive_trainer.py
│   ├── mud_ultra_trainer.py
│   ├── mud_final_trainer.py   # ⚠️ SIN balance loss
│   ├── kaggle_trainer.py
│   ├── distillation_trainer.py
│   ├── bench_trainer.py
│   ├── vulkan_trainer_benchmark.py  # ⚠️ SIN balance loss
│   ├── auto_config.py        # Config automática por RAM
│   ├── vulkan_backend.py     # FFI wrapper + clear_caches()
│   ├── exporter.py, corpus.py, build_vocab.py, ...
│   ├── README.md
│   └── KAGGLE_COMMANDS.md
│
├── weights/              # Pesos y checkpoints (.pt/.mud)
│   └── checkpoints/
│
├── models/               # Despliegue y persistencia
│   └── knowledge.db      # SQLite con perfiles, IQ, curiosidad
│
├── checkpoints_vulkan/   # Checkpoints exportados formato Vulkan
│
├── tools/                # Utilidades
│
├── docs/                 # Documentación (26 archivos)
│   ├── hardware/         # ISA, Vulkan, kernels, punteros, memoria
│   ├── MUD_ARCHITECTURE.md, MUD_ROADMAP.md, MUD_OPTIMIZATION_LOG.md
│   ├── MUD_MOE_EXPERTS.md, MUD_AUDIT.md, MUD_AUDIT_REPORT_V1.md
│   └── MUD_V1_MASTER_REPORT.md, MUD_USER_MANUAL.md, etc.
│
├── logs/                 # Registros de Ejecución
│   └── training/
│
└── tests/                # Tests (vacíos, por implementar)
```

### Reglas de Mantenimiento
1. **Limpieza:** Usar `./mud.sh clean` para re-organizar archivos fuera de lugar.
2. **Nuevas Skills:** Deben colocarse en `src/mud/skills/` y registrarse en `src/mud/inference.rs`.
3. **Checkpoints:** Se rotan automáticamente en `weights/checkpoints/`.
4. **Nuevos trainers:** Deben propagar `aux_coeff` por constructor y setear `_step_ratio` en el loop.
5. **Auditoría MoE:** Verificar balance >85% después de 600 pasos antes de marcar como estable.
