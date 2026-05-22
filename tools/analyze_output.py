import sys
import re
from collections import Counter

def analyze_text(text):
    words = re.findall(r"\w+", text.lower())
    total_words = len(words)
    unique_words = len(set(words))
    
    # 1. Repetition Ratio (0.0 = all unique, 1.0 = all same)
    repetition_ratio = 1.0 - (unique_words / total_words) if total_words > 0 else 0
    
    # 2. Vocabulary Concentration (Top 5 words)
    counts = Counter(words)
    top_5 = counts.most_common(5)
    
    # 3. Character Corruption Detection (Non-ASCII count)
    corruption_count = len(re.findall(r'[^\x00-\x7F]+', text))
    
    # 4. Sentence Stop (EOS presence)
    has_stop = "</s>" in text or "¡Listo!" in text # Placeholder for our markers
    
    return {
        "total": total_words,
        "unique": unique_words,
        "rep_ratio": repetition_ratio,
        "top_5": top_5,
        "corruption": corruption_count,
        "has_stop": has_stop
    }

if __name__ == "__main__":
    # Análisis de la última interacción proporcionada por el usuario
    sample_outputs = [
        "Abrazo excelente import reaction problema grave fundamental great you think of am am feel me the boy salvaje dice there there there cuatro tragos presentarme justice batteries jugando pop esto contains receptor necesario holdem google combat all arriba ests MUD tops is beautiful ataque is life pregnant seek",
        "funcionando funcionando funcionando funcionando feeds grosero lastimarte battlefield there there there cuatro busquen dice pasa is beautiful ataque is life pregnant actividades MUD tops problema grave fundamental great you think of am am feel me the boy salvaje hurra restore nuevo entiendo qu pasa is beautiful he engine",
        "abrazo excelente import reaction problema grave fundamental great you think of am am ests there there there cuatro tragos presentarme justice batteries jugando pop esto contains receptor necesario holdem google combat all arriba piensas de la vida es great you think is beautiful ataque is life pregnant seek"
    ]
    
    print("=== MUD INFERENCE AUTOPSY (STATISTICS) ===")
    for i, out in enumerate(sample_outputs):
        res = analyze_text(out)
        print(f"\n[Test {i+1}] Word Count: {res['total']}")
        print(f"  > Repetition Ratio: {res['rep_ratio']:.2%}")
        print(f"  > Unique Vocabulary: {res['unique']}")
        print(f"  > Top Words: {res['top_5']}")
        print(f"  > Corruption Detected: {'YES' if res['corruption'] > 0 else 'NO'} ({res['corruption']} instances)")
        print(f"  > Knows when to stop? {'NO (Infinite loop)' if not res['has_stop'] else 'YES'}")

    print("\n--- DIAGNOSIS ---")
    print("El motor actual sufre de 'Logit Saturation'. Al no tener un token EOS entrenado,")
    print("el modelo cae en atractores de baja energía (palabras frecuentes del corpus old)")
    print("como 'there', 'beautiful' y 'funcionando'. La v1-MASTER resolverá esto mediante")
    print("el fin de secuencia forzado y el muestreo Top-P.")
