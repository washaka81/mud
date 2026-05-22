import os
for f in ["training/mud_language_trainer.py", "training/mud_cognitive_trainer.py", "training/distillation_trainer.py"]:
    with open(f, "r") as file:
        content = file.read()
    content = content.replace('print(f"\\n', 'print(f"\\\\n')
    with open(f, "w") as file:
        file.write(content)
