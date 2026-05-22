import json

# Read files
with open("training/auto_config.py", "r") as f:
    auto_config_code = f.read()

with open("training/mud_fast_trainer.py", "r") as f:
    mud_fast_trainer_code = f.read()

notebook = {
  "cells": [
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "!mkdir -p /content/drive/MyDrive/MUD_Checkpoints"
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
        "%%writefile mud_fast_trainer.py\n" + mud_fast_trainer_code
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!mkdir -p models logs training",
        "env = {'MUD_MODELS_DIR': '/content/drive/MyDrive/MUD_Checkpoints'}",
        "!python mud_fast_trainer.py --steps 100000"
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
