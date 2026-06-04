#!/bin/bash

# Deshabilitar compilación para no tardar tanto en CPU local
export TORCH_COMPILE_DISABLE=1

echo "=========================================================="
echo "Fase 1: Entrenamiento Kaggle (mud_ultra_trainer) - 200 pasos"
echo "=========================================================="
# Ejecutamos 201 pasos para que el paso 200 genere el guardado (empieza en 0)
python3 training/mud_ultra_trainer.py --steps 201

echo ""
echo "=========================================================="
echo "Sincronizando Checkpoint hacia Local..."
echo "=========================================================="
if [ -f "weights/checkpoints/ckpt_step_200.pt" ]; then
    cp weights/checkpoints/ckpt_step_200.pt models/mud_fast_ckpt.pt
    echo "✅ Checkpoint copiado exitosamente a models/mud_fast_ckpt.pt"
else
    echo "❌ Error: El checkpoint de Kaggle no se generó."
    exit 1
fi

echo ""
echo "=========================================================="
echo "Fase 2: Entrenamiento Local (mud_fast_trainer) - 200 pasos adicionales"
echo "=========================================================="
# Ejecutamos 401 pasos para que vaya del 200 al 400.
python3 training/mud_fast_trainer.py --steps 401 --resume

echo ""
echo "=========================================================="
echo "✅ PRUEBA DE HANDOFF KAGGLE <-> LOCAL FINALIZADA"
echo "=========================================================="
