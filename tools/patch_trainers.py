import os
import glob
import re

TRAINERS = [
    "training/mud_language_trainer.py",
    "training/mud_cognitive_trainer.py",
    "training/mud_ultra_trainer.py",
    "training/mud_final_trainer.py",
    "training/kaggle_trainer.py",
    "training/distillation_trainer.py",
]

VULKAN_IMPORT = """
VULKAN_AVAILABLE = False
try:
    if os.environ.get("MUD_USE_VULKAN") == "1":
        sys.path.append(os.getcwd())
        from training import vulkan_backend
        vulkan_backend._load_lib()
        VULKAN_AVAILABLE = vulkan_backend._vulkan_available
except Exception:
    pass
"""

OOM_RECOVERY_TEMPLATE = """
        try:
            if scaler:
                scaler.scale(total_loss).backward()
                # Numerical guard
                if not torch.isfinite(total_loss):
                    print(f"\\n⚠️  Salto de emergencia: Loss no finita. Ignorando lote.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if not torch.isfinite(total_loss):
                    print(f"\\n⚠️  Salto de emergencia: Loss no finita. Ignorando lote.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                print(f"\\n⚠️  OOM en paso. Limpiando cachés y omitiendo lote.")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                import gc; gc.collect()
                try:
                    if VULKAN_AVAILABLE:
                        from training import vulkan_backend
                        vulkan_backend.clear_caches()
                except: pass
                continue
            else:
                raise e
"""

for t in TRAINERS:
    if not os.path.exists(t):
        continue
    with open(t, "r") as f:
        content = f.read()

    # Add VULKAN_IMPORT if not present
    if "VULKAN_AVAILABLE =" not in content and "vulkan_backend" not in content:
        content = content.replace("import torch", "import torch\nimport sys\n" + VULKAN_IMPORT)

    # Patch backward block for language, kaggle, distillation, final, ultra
    # Most have a block like:
    #         if scaler:
    #             scaler.scale(total_loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #         else:
    #             total_loss.backward()
    #             optimizer.step()
    
    # regex to find backward block:
    pattern_scaler = re.compile(r"(\s*)if scaler:\s+scaler\.scale\([a-zA-Z_]+\)\.backward\(\)\s+scaler\.step\(optimizer\)\s+scaler\.update\(\)\s+else:\s+[a-zA-Z_]+\.backward\(\)\s+optimizer\.step\(\)")
    
    if pattern_scaler.search(content):
        match = pattern_scaler.search(content)
        indent = match.group(1)
        replacement = OOM_RECOVERY_TEMPLATE.replace("\n        ", "\n" + indent)
        content = pattern_scaler.sub(replacement, content)
        print(f"Patched scaler backward in {t}")
    else:
        # cognitive trainer uses:
        #         loss.backward()
        #         optimizer.step()
        pattern_simple = re.compile(r"(\s*)loss\.backward\(\)\s+optimizer\.step\(\)")
        if pattern_simple.search(content):
            match = pattern_simple.search(content)
            indent = match.group(1)
            replacement = OOM_RECOVERY_TEMPLATE.replace("\n        ", "\n" + indent).replace("total_loss", "loss")
            content = pattern_simple.sub(replacement, content)
            print(f"Patched simple backward in {t}")

    with open(t, "w") as f:
        f.write(content)
