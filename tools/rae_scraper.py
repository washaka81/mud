import requests
import re
import time
import os
import random

# Lista de agentes de usuario para evitar bloqueos
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
]

def get_rae_definition(word):
    url = f"https://dle.rae.es/{word}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        # Regex simple para extraer el contenido de las acepciones
        # La RAE usa etiquetas <p class="j" ...> para las acepciones
        html = response.text
        # Limpieza básica de HTML
        def clean_html(raw_html):
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', raw_html)
            return cleantext.strip()

        # Buscar acepciones reales
        # Las acepciones suelen estar en <p class="j"> o dentro de un <article>
        html = response.text
        if "La palabra «" in html and "no está en el Diccionario" in html:
            return None

        matches = re.findall(r'<p class="j".*?>(.*?)</p>', html, re.DOTALL)
        if not matches:
            matches = re.findall(r'<p class="k".*?>(.*?)</p>', html, re.DOTALL)
        
        # Limpieza de tags y entidades HTML
        def clean_html(raw_html):
            cleanr = re.compile('<.*?>')
            cleantext = re.sub(cleanr, '', raw_html)
            # Eliminar números de acepción al inicio (ej. "1. f.")
            cleantext = re.sub(r'^\d+\.\s+[a-z]+\.\s+', '', cleantext)
            return cleantext.strip()

        definitions = [clean_html(m) for m in matches]
        # Filtrar textos basura de la RAE
        bad_texts = ["Consulta posible gracias al compromiso", "Real Academia Española", "Aviso:"]
        definitions = [d for d in definitions if not any(bt in d for b_t in bad_texts for bt in [b_t])]
        
        return " | ".join(definitions) if definitions else None
    except Exception as e:
        print(f"Error fetching {word}: {e}")
        return None

def main():
    vocab_path = "training/vocab_es_en.txt"
    output_path = "training/rae_knowledge.txt"
    
    if not os.path.exists(vocab_path):
        print("Error: vocab_es_en.txt not found.")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if len(line.strip()) > 3]

    print(f"🚀 Iniciando ingesta masiva de la RAE ({len(words)} palabras potenciales)...")
    
    # Cargar progreso previo si existe
    processed_words = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("RAE:"):
                    w = line.split(":")[1].split("|")[0].strip()
                    processed_words.add(w)

    with open(output_path, "a", encoding="utf-8") as f:
        count = 0
        for word in words:
            if word in processed_words:
                continue
            
            print(f"  > Buscando: {word}...", end="\r")
            definition = get_rae_definition(word)
            
            if definition:
                f.write(f"RAE: {word} | Definición: {definition}\n")
                f.flush()
                count += 1
            
            # Delay para evitar baneo
            time.sleep(random.uniform(0.5, 1.5))
            
            if count >= 100: # Límite por sesión para no abusar
                print(f"\n✅ Lote de 100 definiciones completado.")
                break

    print(f"\n✅ Ingesta finalizada. Total nuevas: {count}")

if __name__ == "__main__":
    main()
