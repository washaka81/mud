---
lang: en
---

# Extended User Manual: Forge LLM (MUD) Engine

Welcome to the MUD operating console. This manual details how to operate, train, and expand the system's knowledge base.

---

## 1. Console Commands (REPL)

Run `./mud.sh chat` to enter the interactive MUD interface.

### `/ingest <path>`
The most important command for AI expansion.
- **Usage:** `/ingest tests/data/books/`
- **Function:** Scans the folder, reads `.txt` and `.pdf` files (via `pdftotext`), chunks them, and saves them to the knowledge database (`knowledge.db`).
- **Algorithm:** Applies **PageRank** to rank fact importance and manage RAM residency.

### `/status`
- Shows detailed statistics of the knowledge base: unassimilated facts vs. integrated facts.

### `/reset`
- Resets all knowledge chunks to 'pending' state, allowing for a complete re-training cycle.

### `/exit` or `/quit`
- Safely exits the inference session and restores terminal scrolling regions.

---

## 2. Model Preparation: The High-Fidelity Pipeline (v1.5)

To run a model in MUD, it must be converted from Safetensors to the `.mud` format using the high-fidelity protocol.

### 2.1 Conversion (PRQ)
MUD utilizes **Per-Row Quantization (PRQ)**. This stores a unique scale factor for every row of the weight matrix, preventing semantic decay in deep models.

- **Command Syntax:**
  ```bash
  ./mud.sh convert [input_safetensors_path] [output_mud_path] [--ternarize-emb]
  ```
  *(Example: `./mud.sh convert models/qwen2_0.5b/model.safetensors models/qwen2_0.5b.mud --ternarize-emb`)*

- **Agnostic Support:** The converter handles LLaMA, Qwen, Mistral, and DeepSeek architectures through automatic `config.json` parsing and structural mapping.

### 2.2 Linguistic Restoration (restore-iq)
Models undergo **"Ternary Shock"** after conversion, where weights are mathematically correct but semantically unseated. You MUST run the restoration pipeline:

1.  **Align:** Automated during conversion.
2.  **Project:** Applies Bayesian recalibration and signal boosting.
    ```bash
    ./mud.sh project models/my_model.mud --boost
    ```
3.  **Train:** Fine-tune the ternary manifold and inject recent facts from `knowledge.db`.
    ```bash
    ./mud.sh restore-iq models/my_model.mud
    ```

---

## 3. Training Pipeline (Cloud Evolution)

MUD can be re-trained to absorb new books into its weights:

1.  **Ingestion:** Use `/ingest` to populate your local `knowledge.db`.
2.  **Syncing:** Run `bash training/push_to_kaggle.sh` to upload the new dataset to the cloud.
3.  **Retrieval:** Once Kaggle finishes training, use `./training/pull_from_kaggle.sh` to fetch your upgraded, more intelligent AI.

For local training, use `./mud.sh train` to launch the native Rust auto-trainer (`MudAutoTrainer`), which processes knowledge database chunks directly without Python dependencies.

---

## 4. Troubleshooting

- **"Error: May not be a PDF file":** Ensure the PDF is not password protected or a renamed HTML landing page.
- **Ingestion feels frozen:** MUD is calculating semantic bridges between thousands of nodes. Wait for the progress counter to finish.
- **Low Token Speed (t/s):** Ensure no heavy background processes are using the iGPU or CPU. MUD requires priority access to Vulkan Subgroups.
- **SegFaults on start:** Ensure the model was converted with the current PRQ standard. Re-run conversion if the engine was updated after May 26, 2026.

---

## 5. Key File Structure
- `models/core_skills.mud`: The current ternary brain.
- `models/knowledge.db`: Persistent database of all ingested books.
- `src/vulkan/`: GPU kernel source code.
- `training/`: Model evolution and Kaggle sync scripts.

---

## 6. Unified Diagnostics Dashboard & Cognitive Health Index (CHI)

MUD integrates a deep, native console health auditing system (`./mud.sh diag`) that monitors system parameters and evaluates model health:

- **Cognitive Health Index (CHI):** A weighted metric representing model integrity:
  - **Retención de Memoria Coherente (Standard Deviation $\sigma$):** Monitors ternary distribution. A value below `0.10` triggers a ternary shock warning.
  - **Especialización del Razonamiento:** Checks for "Dead Experts" in MoE layers (weights collapsed to zero).
