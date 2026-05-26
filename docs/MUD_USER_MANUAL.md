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

### `/exit` or `Ctrl+Q`
- Safely exits the inference session and restores terminal settings.

---

## 2. Operating Modes

### Thinking Mode (Reasoning)
MUD is built for **Chain of Thought (CoT)** reasoning.
- When the model generates `<thinking>` tags, the terminal switches to **Dim Cyan Italic**.
- You can observe the logical reasoning process before receiving the final answer in the `<answer>` tag.

### Hybrid Inference (Hardware)
The system automatically manages hardware resources:
- **Status Bar:** Always visible at the bottom, showing CPU/RAM usage and **Tokens per Second (t/s)**.
- **Vulkan Acceleration:** If an Intel Iris Xe iGPU is detected, MUD will use it to accelerate ternary multiplications automatically via Subgroup Arithmetic.

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

---

## 5. Key File Structure
- `models/core_skills.mud`: The current ternary brain.
- `models/knowledge.db`: Persistent database of all ingested books.
- `src/vulkan/`: GPU kernel source code.
- `training/`: Model evolution and Kaggle sync scripts.
