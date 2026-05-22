#!/bin/bash
# ==============================================================================
# MUD INTELLIGENT TRAINING ORCHESTRATOR v2.0
# ==============================================================================
# MoE Hiper-Granular: 256 micro-expertos / 16 clústeres funcionales
# Ciclo completo de entrenamiento:
#   1. Auditoría de hardware (AVX2/AVX512/Vulkan)
#   2. Deep Audit MoE (salud, balance, coherencia ternaria)
#   3. Calibración de pipeline
#   4. Entrenamiento (local o Kaggle)
#   5. Exportación y validación cognitiva
# ==============================================================================

set -euo pipefail  # Fail-fast + variables no declaradas = error

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME="MUD-256MOE-V2"
LOG_DIR="logs/training"
mkdir -p "$LOG_DIR" models weights
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SESSION_LOG="$LOG_DIR/session_${TIMESTAMP}.log"

# MoE config (sincronizado con mud_fast_trainer.py)
NUM_EXPERTS=256
CLUSTER_SIZE=16
NUM_CLUSTERS=16
TOP_K=4
TARGET_STEPS=100000

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
MODE="local"
TEST_MODE=0
QUICK_TEST=0
DRY_RUN=0
RESUME=""
STEPS_OVERRIDE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --kaggle)       MODE="kaggle";          shift ;;
        --test-all)     TEST_MODE=1;            shift ;;
        --quick)        QUICK_TEST=1;           shift ;;
        --resume)       RESUME="--resume";      shift ;;
        --steps)        STEPS_OVERRIDE="$2";    shift 2 ;;
        --experts)      NUM_EXPERTS="$2";       shift 2 ;;
        --top-k)        TOP_K="$2";             shift 2 ;;
        --dry-run)      DRY_RUN=1;              shift ;;
        *) echo "❌ Parámetro desconocido: $1"; exit 1 ;;
    esac
done

STEPS=${STEPS_OVERRIDE:-$TARGET_STEPS}

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$SESSION_LOG"
echo -e "\x1b[1;35m╔══════════════════════════════════════════════════════════════════════╗" | tee -a "$SESSION_LOG"
echo -e "║     MUD SLIME ENGINE — TRAINING ORCHESTRATOR v2.0                    ║" | tee -a "$SESSION_LOG"
echo -e "║     MoE: ${NUM_EXPERTS} expertos → ${NUM_CLUSTERS} clústeres × ${CLUSTER_SIZE}  | Top-K: ${TOP_K}          ║" | tee -a "$SESSION_LOG"
echo -e "╚══════════════════════════════════════════════════════════════════════╝\x1b[0m" | tee -a "$SESSION_LOG"
echo "  Modo: $MODE | Steps: $STEPS | Session: $SESSION_LOG" | tee -a "$SESSION_LOG"
echo "" | tee -a "$SESSION_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# MODO KAGGLE
# ─────────────────────────────────────────────────────────────────────────────
if [ "$MODE" == "kaggle" ]; then
    echo -e "\x1b[1;34m[CLOUD] Despachando a Kaggle...\x1b[0m" | tee -a "$SESSION_LOG"

    if ! command -v kaggle &> /dev/null; then
        echo "  ❌ Kaggle API no encontrada. Instalar con: pip install kaggle" | tee -a "$SESSION_LOG"
        exit 1
    fi

    # Verificar credenciales
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "  ❌ Credenciales Kaggle no encontradas en ~/.kaggle/kaggle.json" | tee -a "$SESSION_LOG"
        exit 1
    fi

    # Actualizar metadata del kernel con la nueva configuración MoE
    python3 -c "
import json, sys
meta_path = 'training/kernel-metadata.json'
try:
    with open(meta_path) as f:
        meta = json.load(f)
    # Inyectar variables de entorno en el kernel
    meta['environment_variables'] = {
        'NUM_EXPERTS': '${NUM_EXPERTS}',
        'CLUSTER_SIZE': '${CLUSTER_SIZE}',
        'TOP_K': '${TOP_K}',
        'STEPS': '${STEPS}',
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print('  ✅ kernel-metadata.json actualizado con config MoE')
except Exception as e:
    print(f'  ⚠ No se pudo actualizar kernel-metadata.json: {e}')
" | tee -a "$SESSION_LOG"

    # Verificar syntax Python antes de push
    python3 -m py_compile training/kaggle_trainer.py && \
        echo "  ✅ Syntax Python: OK" | tee -a "$SESSION_LOG"

    kaggle kernels push -p training/ | tee -a "$SESSION_LOG"

    KAGGLE_USER=$(kaggle config view 2>/dev/null | grep username | awk '{print $3}')
    echo "" | tee -a "$SESSION_LOG"
    echo -e "\x1b[1;32m✅ KERNEL PUSHED — ${NUM_EXPERTS} expertos MoE\x1b[0m" | tee -a "$SESSION_LOG"
    echo "  Monitor: https://www.kaggle.com/code/${KAGGLE_USER}/mud-ternary-moe-training-es-en" | tee -a "$SESSION_LOG"
    echo "  Cuando termine: ./training/pull_from_kaggle.sh" | tee -a "$SESSION_LOG"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# [1/6] HARDWARE PROFILER — detección + benchmark + DB
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\x1b[1;34m[1/6] Hardware Profiler (auto-detección)...\x1b[0m" | tee -a "$SESSION_LOG"

python3 tools/hardware_profiler.py 2>&1 | tee -a "$SESSION_LOG"

# Cargar config recomendada desde auto_config
AUTO_CFG=$(python3 -c "
import sys; sys.path.insert(0, 'training')
from auto_config import load_training_config
cfg = load_training_config()
print(f'{cfg[\"num_experts\"]}|{cfg[\"hidden\"]}|{cfg[\"num_layers\"]}|{cfg[\"top_k\"]}|{cfg[\"mode\"]}')
" 2>/dev/null || echo "256|512|4|4|big")

IFS='|' read -r AUTO_EXPERTS AUTO_HIDDEN AUTO_LAYERS AUTO_TOP_K AUTO_MODE <<< "$AUTO_CFG"

# Si no se especificaron expertos manualmente, usar auto-detected
if [ "$NUM_EXPERTS" -eq 256 ] && [ "$AUTO_EXPERTS" -lt 256 ]; then
    echo -e "\x1b[1;33m   ↳ Auto-config sugiere modo '$AUTO_MODE' ($AUTO_EXPERTS expertos)\x1b[0m" | tee -a "$SESSION_LOG"
    NUM_EXPERTS=$AUTO_EXPERTS
    CLUSTER_SIZE=$(( NUM_EXPERTS / 4 ))
    NUM_CLUSTERS=$(( NUM_EXPERTS / CLUSTER_SIZE ))
    TOP_K=$AUTO_TOP_K
fi

NUM_CORES=$(nproc)
echo "  ✅ Config: ${NUM_EXPERTS} expertos × ${AUTO_HIDDEN} hidden × ${AUTO_LAYERS} capas (modo $AUTO_MODE)" | tee -a "$SESSION_LOG"

# Mostrar reporte detallado
python3 -c "
import sys; sys.path.insert(0, 'training')
from auto_config import load_training_config, print_config_report
cfg = load_training_config('$AUTO_MODE')
cfg['num_experts'] = ${NUM_EXPERTS}
cfg['hidden'] = ${AUTO_HIDDEN}
cfg['num_layers'] = ${AUTO_LAYERS}
cfg['top_k'] = ${TOP_K}
print_config_report(cfg)
" 2>&1 | tee -a "$SESSION_LOG"

# Dry-run: solo mostrar config sin entrenar
if [ $DRY_RUN -eq 1 ]; then
    echo -e "\x1b[1;33m   🏁 Dry-run: configurado, no se ejecuta entrenamiento\x1b[0m" | tee -a "$SESSION_LOG"
    echo -e "\x1b[1;33m   Quita --dry-run para entrenar realmente\x1b[0m" | tee -a "$SESSION_LOG"
    exit 0
fi

# Compilación Rust con target-cpu=native
echo "  🦀 Compilando Rust (target-cpu=native)..." | tee -a "$SESSION_LOG"
RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet 2>&1 | tee -a "$SESSION_LOG"
echo "  ✅ Compilación Rust: completa" | tee -a "$SESSION_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# [2/6] DEEP AUDIT MOE — 256 expertos / 16 clústeres
# ─────────────────────────────────────────────────────────────────────────────
if [ $TEST_MODE -eq 1 ]; then
    echo -e "\n\x1b[1;33m[2/5] Deep Audit MoE (256 expertos / 16 clústeres)...\x1b[0m" | tee -a "$SESSION_LOG"

    # Suite de auditoría ordenada por dependencia lógica
    declare -A AUDIT_TOOLS=(
        ["tokenizer_audit"]="Verifica tokenizador y vocabulario"
        ["weight_audit"]="Magnitud global de pesos ternarios"
        ["expert_anatomy"]="Salud anatómica por experto/capa"
        ["moe_audit"]="Balance de carga por clúster funcional"
        ["ternary_audit"]="Coherencia del cuantizador {-1,0,1}"
        ["attention_audit"]="Estabilidad de atención multi-cabeza"
        ["deep_math_audit"]="Razonamiento matemático"
        ["language_audit"]="Competencia lingüística"
        ["neural_autopsy"]="Autopsia profunda de activaciones"
        ["truth_auditor"]="Verificación de hechos y consistencia"
    )

    AUDIT_PASS=0
    AUDIT_FAIL=0
    MODEL_PATH="${1:-models/core_skills.mud}"

    for tool in tokenizer_audit weight_audit expert_anatomy moe_audit \
                ternary_audit attention_audit deep_math_audit language_audit \
                neural_autopsy truth_auditor; do
        desc="${AUDIT_TOOLS[$tool]:-$tool}"
        echo -e "\n  \x1b[1;34m▶ $tool\x1b[0m — $desc" | tee -a "$SESSION_LOG"
        if [ -f "./target/release/$tool" ]; then
            if ./target/release/"$tool" "$MODEL_PATH" 2>&1 | tee -a "$SESSION_LOG"; then
                echo "  \x1b[32m✅ PASS\x1b[0m" | tee -a "$SESSION_LOG"
                AUDIT_PASS=$((AUDIT_PASS + 1))
            else
                echo "  \x1b[31m❌ FAIL\x1b[0m" | tee -a "$SESSION_LOG"
                AUDIT_FAIL=$((AUDIT_FAIL + 1))
            fi
        else
            echo "  \x1b[33m⚠ SKIP (binario no encontrado)\x1b[0m" | tee -a "$SESSION_LOG"
        fi
    done

    echo "" | tee -a "$SESSION_LOG"
    echo -e "\x1b[1;35m[AUDIT SUMMARY] PASS: $AUDIT_PASS | FAIL: $AUDIT_FAIL\x1b[0m" | tee -a "$SESSION_LOG"
    echo "  📄 Reporte completo: $SESSION_LOG" | tee -a "$SESSION_LOG"
    exit 0

elif [ $QUICK_TEST -eq 1 ]; then
    echo -e "\n\x1b[1;33m[2/5] Quick Audit (moe_audit + weight_audit)...\x1b[0m" | tee -a "$SESSION_LOG"
    MODEL_PATH="${1:-models/core_skills.mud}"
    for tool in moe_audit weight_audit; do
        [ -f "./target/release/$tool" ] && \
            ./target/release/"$tool" "$MODEL_PATH" 2>&1 | tee -a "$SESSION_LOG" || true
    done
else
    echo -e "  ℹ️  Usa --test-all para ejecutar la suite de auditoría completa" | tee -a "$SESSION_LOG"
    echo -e "  ℹ️  Usa --quick para auditoría rápida MoE\n" | tee -a "$SESSION_LOG"
fi

# ─────────────────────────────────────────────────────────────────────────────
# [3/6] CALIBRACIÓN DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\x1b[1;34m[3/6] Calibración del Pipeline...\x1b[0m" | tee -a "$SESSION_LOG"

# Verificar syntax Python
python3 -m py_compile training/mud_fast_trainer.py && \
    echo "  ✅ mud_fast_trainer.py: syntax OK" | tee -a "$SESSION_LOG"

# Verificar que el vocabulario existe
VOCAB_PATH=""
for p in training/vocab_es_en.txt vocab_es_en.txt; do
    if [ -f "$p" ]; then
        VOCAB_PATH="$p"
        VOCAB_SIZE=$(wc -l < "$p")
        echo "  ✅ Vocabulario: $p (${VOCAB_SIZE} tokens)" | tee -a "$SESSION_LOG"
        break
    fi
done
if [ -z "$VOCAB_PATH" ]; then
    echo "  ❌ vocab_es_en.txt no encontrado — ejecutar: python3 training/build_vocab_bilingual.py" | tee -a "$SESSION_LOG"
    exit 1
fi

# Verificar corpus
CORPUS_PATH=""
for p in training/synthetic_knowledge.txt training/massive_knowledge_corpus.txt; do
    if [ -f "$p" ]; then
        CORPUS_PATH="$p"
        CORPUS_MB=$(du -sm "$p" | cut -f1)
        echo "  ✅ Corpus: $p (${CORPUS_MB} MB)" | tee -a "$SESSION_LOG"
        break
    fi
done
if [ -z "$CORPUS_PATH" ]; then
    echo "  ⚠️  Corpus no encontrado — el trainer usará corpus de arranque mínimo" | tee -a "$SESSION_LOG"
fi

# Configurar OpenMP
export OMP_NUM_THREADS=$NUM_CORES
export MKL_NUM_THREADS=$NUM_CORES
export MKL_DEBUG_CPU_TYPE=5
echo "  ⚡ OpenMP: ${NUM_CORES} threads | MKL: AVX2 enforced" | tee -a "$SESSION_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# [4/6] VERIFICACIÓN DE MEMORIA — evitar OOM
# ─────────────────────────────────────────────────────────────────────────────
RAM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)
# Estimación conservadora: 256 experts ~28GB, 64 experts ~7GB, 32 experts ~3.5GB, 16 experts ~1.8GB
_MIN_RAM_FOR_EXPERTS=$(( NUM_EXPERTS * 120 / 1024 + 4 ))  # ~120MB por expert + overhead
if [ "$RAM_GB" -lt "$_MIN_RAM_FOR_EXPERTS" ]; then
    echo -e "\x1b[1;33m⚠️  ADVERTENCIA: ${RAM_GB}GB RAM detectados, ${NUM_EXPERTS} expertos pueden requerir ~${_MIN_RAM_FOR_EXPERTS}GB\x1b[0m" | tee -a "$SESSION_LOG"
    echo -e "\x1b[1;33m   Recomendación: usa --experts 64, 32 o 16 para local\x1b[0m" | tee -a "$SESSION_LOG"
    if [ "$RAM_GB" -lt 12 ] && [ "$NUM_EXPERTS" -gt 64 ]; then
        echo -e "\x1b[1;31m   ❌ Forzando reducción a 64 expertos para evitar OOM\x1b[0m" | tee -a "$SESSION_LOG"
        NUM_EXPERTS=64
        CLUSTER_SIZE=8
        NUM_CLUSTERS=8
        TOP_K=2
    fi
    if [ "$RAM_GB" -lt 8 ] && [ "$NUM_EXPERTS" -gt 16 ]; then
        echo -e "\x1b[1;31m   ❌ Forzando reducción a 16 expertos para evitar OOM\x1b[0m" | tee -a "$SESSION_LOG"
        NUM_EXPERTS=16
        CLUSTER_SIZE=4
        NUM_CLUSTERS=4
        TOP_K=2
    fi
fi
echo "  ✅ Memoria: ${RAM_GB}GB RAM — ${NUM_EXPERTS} expertos MoE" | tee -a "$SESSION_LOG"

# ─────────────────────────────────────────────────────────────────────────────
# [5/6] DETECCIÓN DE CHECKPOINT Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n\x1b[1;34m[5/6] Entrenamiento MoE (${NUM_EXPERTS} expertos, ${STEPS} pasos)...\x1b[0m" | tee -a "$SESSION_LOG"

LATEST_CKPT=$(ls -t weights/mud_last_checkpoint.pt models/mud_fast_ckpt.pt \
              weights/checkpoints/ckpt_step_*.pt 2>/dev/null | head -n 1 || echo "")

if [ -n "$LATEST_CKPT" ]; then
    echo "  📂 Checkpoint encontrado: $LATEST_CKPT" | tee -a "$SESSION_LOG"
    RESUME="--resume"
else
    echo "  🆕 Sin checkpoints — entrenamiento desde cero" | tee -a "$SESSION_LOG"
    RESUME=""
fi

# Lanzar entrenamiento
python3 training/mud_fast_trainer.py \
    --steps "$STEPS" \
    --experts "$NUM_EXPERTS" \
    --top-k "$TOP_K" \
    --log-balance \
    ${RESUME} \
    2>&1 | tee -a "$SESSION_LOG"

# Verificar exportación
if [ -f "models/core_skills.mud" ]; then
    MUD_SIZE=$(du -sm models/core_skills.mud | cut -f1)
    echo "  ✅ Exportación: models/core_skills.mud (${MUD_SIZE} MB)" | tee -a "$SESSION_LOG"

    # Guardar resultado en training_history
    python3 -c "
import sys; sys.path.insert(0, 'training')
from auto_config import save_training_result
import re
log = open('$SESSION_LOG').read()
loss_m = re.search(r'loss=([0-9.]+)', log)
it_m  = re.search(r'([0-9.]+) it/s', log)
tok_m = re.search(r'([0-9.]+) tok/s', log)
time_m = re.search(r'finalizado en ([0-9.]+)s', log)
save_training_result(
    session_id='${TIMESTAMP}',
    mode='$AUTO_MODE',
    num_experts=${NUM_EXPERTS},
    num_layers=${AUTO_LAYERS},
    hidden=${AUTO_HIDDEN},
    steps=${STEPS},
    loss_final=float(loss_m.group(1)) if loss_m else 0.0,
    avg_it_s=float(it_m.group(1)) if it_m else 0.0,
    avg_tok_s=float(tok_m.group(1)) if tok_m else 0.0,
    total_time_s=float(time_m.group(1)) if time_m else 0.0,
)
print('  ✅ Resultado guardado en training_history')
" 2>&1 | tee -a "$SESSION_LOG" || true
else
    echo "  ❌ Error: models/core_skills.mud no generado" | tee -a "$SESSION_LOG"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# [6/6] VALIDACIÓN COGNITIVA FINAL + LOG HISTÓRICO
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n\x1b[1;34m[6/6] Validación Cognitiva Final...\x1b[0m" | tee -a "$SESSION_LOG"

if [ -f "tools/cognitive_dashboard.py" ]; then
    python3 tools/cognitive_dashboard.py 2>&1 | tee -a "$SESSION_LOG"
fi

# Post-entrenamiento: quick audit si el binario existe
if [ -f "./target/release/moe_audit" ]; then
    echo "  🔬 Post-training MoE Audit..." | tee -a "$SESSION_LOG"
    ./target/release/moe_audit models/core_skills.mud 2>&1 | tee -a "$SESSION_LOG" || true
fi

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
echo "" | tee -a "$SESSION_LOG"
echo -e "\x1b[1;32m╔══════════════════════════════════════════════════════════════════════╗" | tee -a "$SESSION_LOG"
echo -e "║     ✅ $MODEL_NAME — ENTRENAMIENTO COMPLETADO$(printf '%*s' $((26 - ${#MODEL_NAME})) '')║" | tee -a "$SESSION_LOG"
echo -e "╚══════════════════════════════════════════════════════════════════════╝\x1b[0m" | tee -a "$SESSION_LOG"
echo "  Modelo: models/core_skills.mud" | tee -a "$SESSION_LOG"
echo "  Log:    $SESSION_LOG" | tee -a "$SESSION_LOG"
echo "  Fecha:  $(date)" | tee -a "$SESSION_LOG"
echo "" | tee -a "$SESSION_LOG"
