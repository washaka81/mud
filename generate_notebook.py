import json

# Read files
with open("training/auto_config.py", "r") as f:
    auto_config_code = f.read()

with open("training/mud_fast_trainer.py", "r") as f:
    mud_fast_trainer_code = f.read()

with open("training/vocab_es_en.txt", "r") as f:
    vocab_content = f.read()

notebook = {
  "cells": [
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "# Prepare directory structure\n",
        "os.makedirs('models', exist_ok=True)\n",
        "os.makedirs('logs/training', exist_ok=True)\n",
        "os.makedirs('training', exist_ok=True)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "%%writefile auto_config.py\n" + auto_config_code
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "%%writefile vocab_es_en.txt\n" + vocab_content
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "%%writefile mud_fast_trainer.py\n" + mud_fast_trainer_code
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# En Kaggle usamos el almacenamiento local (/kaggle/working)\n",
        "os.environ['MUD_USE_VULKAN'] = '0'\n",
        "os.environ['MUD_NO_COMPILE'] = '0'\n",
        "\n",
        "# Intentar localizar el corpus masivo en los inputs de Kaggle\n",
        "corpus_found = False\n",
        "for root, dirs, files in os.walk('/kaggle/input'):\n",
        "    if 'massive_knowledge_corpus.txt' in files:\n",
        "        path = os.path.join(root, 'massive_knowledge_corpus.txt')\n",
        "        print(f'✅ Corpus detectado en: {path}')\n",
        "        # Linkear para que el trainer lo encuentre en la carpeta local\n",
        "        if not os.path.exists('massive_knowledge_corpus.txt'):\n",
        "            os.symlink(path, 'massive_knowledge_corpus.txt')\n",
        "        corpus_found = True\n",
        "        break\n",
        "\n",
        "if not corpus_found:\n",
        "    print('⚠️  AVISO: No se encontró massive_knowledge_corpus.txt en /kaggle/input.')\n",
        "    print('Asegúrate de haber añadido el dataset mud-master-training-data al kernel.')\n",
        "\n",
        "!python3 mud_fast_trainer.py --steps 100000 --resume"
      ]
    }

  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

with open("training/notebook9dbdee419a.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated!")
