# Ornith-1.0-9B: Hybrid Architecture Analysis

## Overview

**Ornith-1.0-9B** is a specialized hybrid model architecture integrated into the MUD (Modular Understanding Dynamics) engine. According to project intelligence, Ornith serves as a synthesis of **Gemma-4** and **Qwen-3.5** architectural design patterns. 

This hybrid topology introduces unique characteristics that require specialized parsing and memory handling during the UCP v2 (Universal Calibration Protocol) conversion and inference pipelines.

## Architectural Composition

### Qwen-3.5 Lineage
- **Configuration Nesting:** The model adopts the Qwen-3.5 standard where core parameters (`num_attention_heads`, `rope_theta`, etc.) are nested deeply inside `"text_config"` and `"rope_parameters"`.
- **Rotary Position Embeddings (RoPE):** Utilizes the complex multi-section `mrope` parameters native to Qwen's advanced reasoning models.
- **Attention Topology:** Maintains Qwen-like standard projections, utilizing `q_proj`, `k_proj`, and `v_proj` inside its linear and full attention blocks.

### Gemma-4 Lineage
- **State-Space / Linear Attention:** Integrates sub-quadratic sequence modeling resembling Gemma-4's efficient linear attention implementations. 
- **Block Regularization:** MUD's UCP pipeline identifies the transition matrices within these linear blocks and audits them under the strict **HiPPO Recurrence Stability** requirements (`ssm_a` state-transition audits).
- **Logit Scaling & Embeddings:** Inherits deep embedding norm characteristics and tying logic often seen in Gemma variants, requiring careful Sub-LayerNorm synthesis during conversion to prevent Ternary Shock.

## Implications for MUD Engine V2 (SlimeRegister)

The fusion of Gemma-4 and Qwen-3.5 inside Ornith requires the MUD engine to handle:
1. **Dynamic Metadata Extraction:** The `universal_converter` must proactively drill into nested JSON fields (`text_config`, `rope_parameters`) to extract `hidden_size`, `rope_theta`, and `num_heads`.
2. **Eigenvalue Auditing:** The `boundary_validator` applies a relaxed A-log stability check for `ssm_a` layers in Ornith's linear blocks to avoid false-positive eigenvalue collapse flags.
3. **SlimeRegister Saturation:** The large 9B parameter count pushes the limits of the i16 partial-accumulation strategy. The engine utilizes the recent **f32 `matmul_accum` upgrade** inside the `SlimeRegister` to ensure Ornith's deep semantic tokens do not overflow the registers during the forward pass.
