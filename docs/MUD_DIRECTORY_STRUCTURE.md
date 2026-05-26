---
lang: es
---

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
│       ├── skills/       # 15 skills (14 cargadas en runtime + CodingExpert registrada pero inactiva)
│       ├── inference.rs  # Orquestación de skills
│       ├── auto_trainer.rs
│       ├── routing.rs, store.rs, graph.rs, ingester.rs
│       └── mod.rs
│
├── training/             # Pipeline de Entrenamiento (Rust/Shell nativo)
│   ├── push_to_kaggle.sh, pull_from_kaggle.sh   # Sincronización cloud
│   ├── kaggle_config.sh / .example              # Config Kaggle
│   ├── dataset-metadata.json / kernel-metadata.json
│   ├── KAGGLE_COMMANDS.md    # Guía de setup Kaggle
│   ├── README.md
│   ├── *.txt                # Corpora de conocimiento (vocab, rae, statistics, etc.)
│   ├── logs/                # Logs de entrenamiento
│   ├── models/              # Modelos exportados
│   └── kaggle_push/         # Artefactos para push a Kaggle
│
├── weights/              # Pesos y checkpoints (.pt/.mud)
│   └── checkpoints/
│
├── models/               # Despliegue y persistencia
│   └── knowledge.db      # SQLite con perfiles, IQ, curiosidad
│
├── checkpoints_vulkan/   # Checkpoints exportados formato Vulkan
│
├── tools/                # Utilidades y Herramientas Nanométricas
│   ├── tensor_microscope.rs # Análisis estadístico nanométrico y de dispersión
│   ├── mud_calibrator.rs    # Inyección precisa de hiperparámetros en el archivo
│   ├── universal_converter/ # Transpilador de Tensores y Metadatos a .mud
│   └── ...
│
├── docs/                 # Documentación (31+ archivos)
│   ├── MUD_OVERVIEW.md, MUD_ARCHITECTURE.md, MUD_ROADMAP.md
│   ├── MUD_USER_MANUAL.md, MUD_GUIDELINES.md, MUD_MASTER_MANIFESTO.md
│   ├── MUD_MOE_EXPERTS.md, MUD_COGNITIVE_ARCH.md, MUD_DATA_ARCHITECTURE.md
│   ├── MUD_ORCHESTRATION.md, MUD_TRAINING_PROTOCOLS.md, MUD_DELEGATION_ROUTER.md
│   ├── MUD_SYSTEM_UPGRADE_V1.5.md, MUD_V1_MASTER_REPORT.md
│   ├── MUD_OPTIMIZATION_ANALYSIS.md, MUD_OPTIMIZATION_LOG.md
│   ├── MUD_AUDIT_LATEST.md  # Audit consolidado vivo (reemplaza 6 docs históricos)
│   ├── MUD_AUDIT.md (HISTÓRICO), MUD_AUDIT_REPORT_V1.md (HISTÓRICO)
│   ├── MUD_AUDIT_RESOLUTION.md (HISTÓRICO), MUD_CONVERSION_AUDIT.md (HISTÓRICO)
│   ├── MUD_STATISTICAL_AUDIT.md (HISTÓRICO), MUD_MATHEMATICAL_AUDIT.md (HISTÓRICO)
│   ├── hardware/         # ISA, Vulkan, kernels, punteros, memoria
│   └── *.txt             # Dumps de desensamblado y tokens
│
├── logs/                 # Registros de Ejecución
│   └── training/
│
└── tests/                # Tests y datos de prueba
    ├── data/             # CSV, documentos de prueba
    └── ...               # Tests unitarios Rust (21/21 passing)
```

### Reglas de Mantenimiento
1. **Limpieza:** Usar `./mud.sh clean` para re-organizar archivos fuera de lugar.
2. **Nuevas Skills:** Deben colocarse en `src/mud/skills/`, declararse en `mod.rs`, y registrarse en `src/mud/inference.rs`.
3. **Checkpoints:** Se rotan automáticamente en `weights/checkpoints/`.
4. **Documentación:** Los cambios arquitectónicos deben reflejarse en `docs/` antes de implementarse.
5. **Auditoría MoE:** Ejecutar `cargo run --release --bin moe_audit` para verificar balance de carga.
