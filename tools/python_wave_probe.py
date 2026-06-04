import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    if len(sys.argv) < 3:
        print("Usage: python wave_probe.py <model_path> <word>")
        return

    model_path = sys.argv[1]
    word = sys.argv[2]

    print("==================================================")
    print("🔬 ORIGINAL MODEL SEMANTIC PROPAGATION PROBE (PyTorch)")
    print("==================================================")
    print(f"  Cargando Modelo Original: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        output_hidden_states=True, 
        torch_dtype=torch.float32, # We force float32 to compute exact variance without underflow
        device_map="cpu"
    )
    
    tokens = tokenizer.encode(word, add_special_tokens=False)
    if not tokens:
        print("Error: La palabra no produjo tokens.")
        return
        
    token_id = tokens[0]
    print(f"\n[Inyectando Semántica: '{word}' (Token ID: {token_id})]")
    print("\n\033[1;36m>> TRAZANDO HUELLA DE LA PALABRA EN MODELO BASE\033[0m")

    # Disable grad and run forward pass
    with torch.no_grad():
        inputs = torch.tensor([[token_id]])
        outputs = model(inputs)
    
    hidden_states = outputs.hidden_states
    
    # hidden_states is a tuple of (embedding_output, layer_1_out, layer_2_out, ...)
    for l, h in enumerate(hidden_states):
        # h shape is (batch, seq_len, hidden_size)
        x = h[0, 0, :].float()
        
        min_v = torch.min(x).item()
        max_v = torch.max(x).item()
        var_v = torch.var(x, unbiased=False).item()
        sigma = var_v ** 0.5
        
        if l == 0:
            print(f"    [SONDA Pytorch] Embedding IN | Pico Mín: {min_v:>8.4f} | Pico Máx: {max_v:>8.4f} | Sigma (Ola): {sigma:>8.4f}")
        else:
            print(f"    [SONDA Pytorch] Capa {l-1:>2} OUT | Pico Mín: {min_v:>8.4f} | Pico Máx: {max_v:>8.4f} | Sigma (Ola): {sigma:>8.4f}")

    print(f"\n  \033[1;32m[FIRMA FINAL DE '{word}' (Pytorch)] Pico Mín: {min_v:.4f} | Pico Máx: {max_v:.4f} | Sigma: {sigma:.4f}\033[0m")

if __name__ == "__main__":
    main()
