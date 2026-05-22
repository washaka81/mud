# MUD Mathematical Delegation & Tool-Use Architecture

As a ternary-quantized assistant, MUD prioritizes **efficiency in generation** over **symbolic precision**. For exact mathematical and logical results, the engine uses a **Delegation Router**.

## 1. The Delegation Principle
- **Ternary Model (LLM):** Understands context, identifies intent, and generates syntactically correct code.
- **External Engine (Sandbox):** Performs exact arithmetic, symbolic algebra, and code verification.

## 2. Delegation Flow
1. **Detection:** The `LogicMathSkill` monitors the context and generated tokens for complex mathematical signatures (e.g., `sin`, `log`, `**`, large number multiplication).
2. **Interception:** If the "Confidence Score" for a mathematical prediction falls below a threshold, the engine triggers an autonomous action.
3. **Execution:**
   - **Simple Math:** Evaluated via a Rust-native expression parser.
   - **Complex Math:** Delegated to a Python sandbox (`SymPy`/`NumPy`).
   - **Code Verification:** Delegated to `ast.parse` or language-specific linters.
4. **Injection:** The result is injected back into the inference stream as a "Fact" or a corrected token.

## 3. Implementation Checklist
- [ ] **Restricted Python Sandbox:** A secure environment to execute `numexpr` or `sympy` without system access.
- [ ] **Symbolic Resolver:** Mapping natural language queries (e.g., "derivada de x^2") to formal tool-calls.
- [ ] **Validation Layer:** Post-generation check that ensures generated code with math actually runs.

## 4. Why Ternary?
Ternary quantization is uniquely suited for the **Router**:
- **ISZ (Is Zero) gates** can be used to deterministically mute expert branches that are prone to mathematical hallucinations.
- **Sign-based logic** aligns with the binary nature of "Delegate vs. Generate" decisions.
