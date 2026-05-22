# MUD Ternary Logic ISA & Computation Standards

This document defines the fundamental ternary operations that the MUD (Modular Understanding Dynamics) engine must implement or simulate for its 1.58-bit Ternary MoE architecture.

## 1. Core Ternary Primitives (Balanced Ternary: {-1, 0, 1})

### Unary Operations
- **BUF (Identity):** $A \rightarrow A$.
- **NOT (Inversion):** $1 \leftrightarrow -1, 0 \rightarrow 0$. Used for weight sign reversal.
- **ABS (Absolute):** $|A|$. Used in importance-based expert routing.
- **ISZ (Is Zero):** $0 \rightarrow 1, \text{else} \rightarrow 0$. Critical for sparse MoE gating.

### Binary Operations
- **MUL (Ternary Multiplier):** $A \otimes B$. Standard multiplication for BitNet 1.58b.
- **SUM (Ternary Add):** $A \oplus B$. Balanced ternary addition with carry simulation.
- **ANY (Maximum):** $\max(A, B)$. Used in Knowledge Graph activation.
- **AND (Minimum):** $\min(A, B)$. Used for conditional logic and pruning.

## 2. LLM-Specific Implementations

### Ternary GEMV (General Matrix-Vector Multiplication)
The kernel must implement the product $\sum(a_i \otimes b_i)$ where $w \in \{-1, 0, 1\}$.
- **Optimization:** Use bit-masks to skip multiplications by 0 (zero-skipping) and simple addition/subtraction for 1 and -1.

### Expert Routing (MoE)
Routing decisions should transition from Softmax to **Ternary Gating**:
- **ISP (Is Positive):** High-pass filter for activating experts.
- **Clamp Up/Down:** Saturation at +1 or -1 to prevent gradient explosion in sparse architectures.

## 3. Physical & Software Mapping
Currently, trits are mapped to 2-bit or FP32 representations for compatibility:
- `-1` $\rightarrow$ `0b11` (binary 2's complement or specific flag)
- `0` $\rightarrow$ `0b00`
- `1` $\rightarrow$ `0b01`

Future hardware abstraction layers (Vulkan/ASM) will prioritize **Balanced Ternary** to reduce memory bandwidth by 30-40% compared to standard binary quantization.
