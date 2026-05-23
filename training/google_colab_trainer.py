"""
google_colab_trainer.py — MUD Engine Google Colab Launcher
===========================================================
Ejecutar en Google Colab (GPU T4/A100). 
Soporta montaje de Google Drive para checkpoints persistentes.

INSTRUCCIONES PARA COLAB:
-------------------------
1. Abre un notebook en Colab.
2. ⚠️ MUY IMPORTANTE: Selecciona Entorno de ejecución -> Cambiar tipo de entorno -> GPU T4.
   Si ves "CPU (Lento)" en los logs, el entrenamiento no avanzará.
3. Pega y ejecuta este bloque:

   !rm -rf mud
   !git clone https://github.com/washaka81/mud.git
   %cd mud
   !./mud.sh colab --drive
...
"""

import os, sys, subprocess, time, argparse
from auto_config import load_training_config, print_config_report

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN POR DEFECTO (Alineada con Kaggle v1-Master)
# ─────────────────────────────────────────────────────────────────────────────
_cfg = load_training_config()

STEPS           = 100000
EXPERTS         = _cfg["num_experts"]
TOP_K           = _cfg["top_k"]
BATCH_SIZE      = _cfg["batch_size"]
LR              = _cfg["lr"]
RESUME          = True
NO_COMPILE      = True  # Deshabilitado por defecto para ahorrar RAM en Colab Free

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS COLAB
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_MOUNT_POINT = "/content/drive"
DRIVE_MUD_PATH    = "/content/drive/MyDrive/MUD_Checkpoints"
LOCAL_MODELS_DIR  = "models"
LOCAL_LOGS_DIR    = "logs/training"

# ─────────────────────────────────────────────────────────────────────────────
# COLORES ANSI
# ─────────────────────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; B = "\033[94m"; X = "\033[0m"

def ok(msg):  print(f"  {G}✅{X} {msg}")
def warn(msg):print(f"  {Y}⚠️ {X} {msg}")
def err(msg): print(f"  {R}❌{X} {msg}")
def info(msg):print(f"  {B}ℹ️ {X} {msg}")
def hdr(msg): print(f"\n{B}{'='*60}{X}\n  {msg}\n{B}{'='*60}{X}")

# ─────────────────────────────────────────────────────────────────────────────
# MONTAJE DE GOOGLE DRIVE
# ─────────────────────────────────────────────────────────────────────────────
def mount_drive():
    if os.path.exists(DRIVE_MUD_PATH):
        ok(f"Drive ya detectado en: {DRIVE_MUD_PATH}")
        return DRIVE_MUD_PATH

    try:
        from google.colab import drive
        hdr("📂 Intentando montar Google Drive")
        # Forzar mount interactivo
        drive.mount(DRIVE_MOUNT_POINT, force_remount=False)
        os.makedirs(DRIVE_MUD_PATH, exist_ok=True)
        ok(f"Drive montado exitosamente.")
        return DRIVE_MUD_PATH
    except Exception as e:
        warn("No se pudo montar Drive automáticamente (posible ejecución en subshell).")
        info("INSTRUCCIONES MANUALES:")
        print(f"  1. Ejecuta esta celda en Colab antes de lanzar el script:")
        print(f"     from google.colab import drive; drive.mount('{DRIVE_MOUNT_POINT}')")
        print(f"  2. Luego vuelve a ejecutar el comando de entrenamiento.")
        print("-" * 60)
        
        # Fallback a local si el usuario decide continuar sin Drive
        if os.environ.get("MUD_FORCE_DRIVE") == "1":
            err("Error crítico: Drive es obligatorio pero no se pudo montar.")
            sys.exit(1)
        
        warn("Usando almacenamiento LOCAL temporal.")
        return LOCAL_MODELS_DIR

# ─────────────────────────────────────────────────────────────────────────────
# LANZAR ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
def launch_training(models_dir, steps, experts, top_k, batch_size, lr, resume, no_compile):
    hdr("🚀 Lanzando Entrenamiento MUD (Google Colab Mode)")
    
    import torch
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (Lento)"
    
    info(f"Hardware detectado: {device_name}")
    if not has_cuda:
        warn("No se detectó GPU. El entrenamiento será extremadamente lento.")
        warn("Asegúrate de ir a 'Entorno de ejecución' -> 'Cambiar tipo de entorno' -> 'GPU T4'")
    
    trainer_path = "training/mud_fast_trainer.py"
    if not os.path.exists(trainer_path):
        err(f"No se encuentra {trainer_path}")
        return

    # Preparar comando
    cmd = [
        sys.executable, trainer_path,
        "--steps",      str(steps),
        "--experts",    str(experts),
        "--top-k",      str(top_k),
        "--batch-size", str(batch_size),
        "--lr",         str(lr),
    ]
    if resume:
        cmd.append("--resume")
    if no_compile:
        cmd.append("--no-compile")
        
    # Inyectar variables de entorno para el trainer
    env = os.environ.copy()
    env["MUD_MODELS_DIR"] = models_dir
    env["MUD_CKPT_PATH"]  = os.path.join(models_dir, "mud_fast_ckpt.pt")
    env["MUD_EXPORT_PATH"] = os.path.join(models_dir, "core_skills.mud")
    
    # Sincronizar checkpoint actual si existe en models/ pero no en Drive
    local_ckpt = os.path.join(LOCAL_MODELS_DIR, "mud_fast_ckpt.pt")
    drive_ckpt = os.path.join(models_dir, "mud_fast_ckpt.pt")
    if os.path.exists(local_ckpt) and not os.path.exists(drive_ckpt):
        info("Sincronizando checkpoint local -> Drive...")
        import shutil
        shutil.copy(local_ckpt, drive_ckpt)
        ok("Checkpoints sincronizados.")

    info(f"Modelos -> {models_dir}")
    info(f"Config: {experts} expertos | Top-K: {top_k} | Batch: {batch_size}")
    print("-" * 60)

    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        warn("Entrenamiento interrumpido.")
    except subprocess.CalledProcessError as e:
        err(f"El entrenamiento falló con código {e.returncode}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MUD Engine — Google Colab Launcher")
    parser.add_argument("--drive",      action="store_true", help="Usar Google Drive para checkpoints")
    parser.add_argument("--steps",      type=int,   default=STEPS)
    parser.add_argument("--experts",    type=int,   default=EXPERTS)
    parser.add_argument("--top-k",      type=int,   default=TOP_K)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--no-resume",  action="store_true")
    parser.add_argument("--compile",    action="store_true", help="Habilitar torch.compile (Usa más RAM)")
    
    args = parser.parse_args()

    # Logo
    print(f"""
{G}╔══════════════════════════════════════════════════════════════╗
║   MUD SLIME ENGINE — Google Colab Launcher v1.0              ║
║   Optimizaciones: CUDA + FP16 + Neural Kick                  ║
╚══════════════════════════════════════════════════════════════╝{X}""")

    models_path = LOCAL_MODELS_DIR
    if args.drive:
        models_path = mount_drive()
    else:
        os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
        
    launch_training(
        models_path, 
        args.steps, 
        args.experts, 
        args.top_k, 
        args.batch_size, 
        args.lr, 
        not args.no_resume,
        not args.compile
    )
