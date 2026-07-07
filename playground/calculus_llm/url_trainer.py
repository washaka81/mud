#!/usr/bin/env python3
import urllib.request
import urllib.parse
import json
import re
import sys
import os
import subprocess

def fetch_wikipedia_text(topic_or_url):
    # Extraer el título de la página si es un enlace completo de Wikipedia
    title = topic_or_url
    if "wikipedia.org/wiki/" in topic_or_url:
        parts = topic_or_url.split("/wiki/")
        if len(parts) > 1:
            title = urllib.parse.unquote(parts[1])
            
    # Reemplazar espacios por guiones bajos (estándar de Wikipedia)
    title = title.replace(" ", "_")
    
    print(f"[API] Solicitando artículo a la API de Wikipedia (es): '{title}'")
    
    # Construir la URL oficial de la API de Wikipedia (Action API)
    # action=query: Consulta de contenido
    # prop=extracts: Extrae el texto del artículo
    # explaintext=1: Devuelve texto plano libre de etiquetas HTML o sintaxis Wiki
    # redirects=1: Resuelve automáticamente redirecciones de páginas
    base_url = "https://es.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "titles": title,
        "redirects": "1",
        "origin": "*"
    }
    
    query_string = urllib.parse.urlencode(params)
    api_url = f"{base_url}?{query_string}"
    
    # Wikipedia exige declarar un User-Agent identificable de forma obligatoria
    headers = {
        'User-Agent': 'SLIMETrainer/2.0 (contact@example.com; https://github.com/slime-engine)'
    }
    req = urllib.request.Request(api_url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        pages = data.get("query", {}).get("pages", {})
        if not pages or list(pages.keys())[0] == "-1":
            print(f"[ERROR] No se encontró el artículo '{title}' en Wikipedia.")
            return None
            
        page_id = list(pages.keys())[0]
        actual_title = pages[page_id].get("title", title)
        extract = pages[page_id].get("extract", "")
        
        if not extract.strip():
            print(f"[WARN] El artículo '{actual_title}' está vacío o no contiene texto.")
            return None
            
        print(f"[API] Descargado artículo '{actual_title}' ({len(extract)} caracteres).")
        return extract
        
    except Exception as e:
        print(f"[ERROR] Fallo al consultar la API de Wikipedia: {e}")
        
        # Base de conocimiento local (Mock Fallback) en caso de rate-limiting (HTTP 429) o fallos de red
        local_db = {
            "lenguaje_c": "El lenguaje de programación C es un estándar desarrollado en los laboratorios Bell por Dennis Ritchie. C es un lenguaje de propósito general ampliamente utilizado en el desarrollo de sistemas operativos, compiladores y sistemas integrados. Su diseño proporciona construcciones que se mapean eficientemente a instrucciones de máquina típicas, sirviendo de base para lenguajes modernos como C++ y Java.",
            "calculo_diferencial": "El cálculo diferencial es una parte del análisis matemático que consiste en el estudio de cómo cambian las funciones continuas cuando sus variables cambian. El principal objeto de estudio en el cálculo diferencial es la derivada, que mide la tasa de cambio instantánea de una función. Históricamente, fue desarrollado de forma independiente por Isaac Newton y Gottfried Leibniz en el siglo diecisiete.",
            "teorema_de_stokes": "El teorema de Stokes generalizado en geometría diferencial es una afirmación sobre la integración de formas diferenciales en variedades con frontera. Establece que la integral de la derivada exterior de una forma diferencial sobre una variedad orientable equivale a la integral de la forma sobre la frontera de dicha variedad. Es un principio unificador del cálculo vectorial."
        }
        
        normalized_title = title.lower().replace(" ", "_")
        if normalized_title in local_db:
            print(f"[FALLBACK] Activando base de conocimiento local (Mock Offline) para '{title}'...")
            return local_db[normalized_title]
            
        return None

def normalize_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Mantener solo letras castellanas con acentos y eliminar puntuaciones/caracteres wiki
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Colapsar múltiples espacios
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text_into_pairs(text, max_pairs=100):
    words = text.split()
    if len(words) < 2:
        print("[WARN] Texto demasiado corto para trocear.")
        return []
        
    print(f"[CHUNKER] Procesando {len(words)} palabras normalizadas...")
    
    # Filtro de palabras funcionales comunes para evitar ruido en los gradientes semánticos de la Neural ODE
    stop_words = {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con", 
        "y", "o", "que", "para", "por", "su", "sus", "como", "cuando", "desde", "hasta", "sobre",
        "este", "esta", "estos", "estas", "eso", "esa", "esos", "esas", "pero", "sino"
    }
    
    pairs = []
    seen = set()
    
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i+1]
        
        # Evitar preposiciones y artículos puros como entrada o salida para mayor densidad conceptual
        if w1 in stop_words or w2 in stop_words:
            continue
        if len(w1) < 3 or len(w2) < 3:
            continue
            
        pair = (w1, w2)
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) >= max_pairs:
                break
                
    print(f"[CHUNKER] Generados {len(pairs)} pares semánticos únicos (input, target).")
    return pairs

def update_dataset_and_vocab(pairs, dataset_path="dataset.csv", vocab_path="vocabulario_es.txt"):
    if not pairs:
        return 0
        
    print(f"[INTEGRATION] Fusionando datos con '{dataset_path}'...")
    
    # Leer pares existentes para no duplicar en dataset.csv
    existing_pairs = set()
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    existing_pairs.add((parts[0], parts[1]))
                    
    # Añadir nuevos pares al dataset.csv
    added_count = 0
    with open(dataset_path, "a", encoding="utf-8") as f:
        for w1, w2 in pairs:
            if (w1, w2) not in existing_pairs:
                f.write(f"{w1},{w2}\n")
                added_count += 1
                
    # Leer palabras del vocabulario
    vocab_words = set()
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                vocab_words.add(line.strip())
                
    # Añadir palabras nuevas al vocabulario general
    new_words = set()
    for w1, w2 in pairs:
        new_words.add(w1)
        new_words.add(w2)
        
    added_vocab_count = 0
    with open(vocab_path, "a", encoding="utf-8") as f:
        for w in sorted(list(new_words)):
            if w not in vocab_words:
                f.write(f"{w}\n")
                added_vocab_count += 1
                
    print(f"[INTEGRATION] +{added_count} nuevos pares agregados a dataset.csv.")
    print(f"[INTEGRATION] +{added_vocab_count} nuevas palabras añadidas a vocabulario_es.txt.")
    return added_count

def run_cpp_retraining():
    possible_dirs = ["build", ".", "../build"]
    trainer_bin = None
    work_dir = "."
    
    for d in possible_dirs:
        path = os.path.join(d, "trainer")
        if os.path.exists(path) and os.path.isfile(path):
            trainer_bin = "./trainer"
            work_dir = d
            break
            
    if not trainer_bin:
        print("[ERROR] No se encontró el binario ejecutable 'trainer' en las rutas de build. Recompila con 'make'.")
        return False
        
    print(f"[TRAIN] Ejecutando reentrenamiento C++ en '{work_dir}' usando '{trainer_bin}'...")
    try:
        # Ejecutar el reentrenamiento de los vectores continuos en la Neural ODE
        result = subprocess.run([trainer_bin], cwd=work_dir, capture_output=True, text=True, check=True)
        print("[TRAIN] Reentrenamiento finalizado con éxito.")
        lines = result.stdout.split('\n')
        for line in lines:
            if "[TRAIN]" in line or "Loss" in line:
                print(f"  -> {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Falló la ejecución del trainer C++: {e.stderr}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Uso: ./url_trainer.py <TÍTULO_ARTÍCULO_O_URL_WIKIPEDIA> [max_pares]")
        print("Ejemplos:")
        print("  ./url_trainer.py Geometría_diferencial 80")
        print("  ./url_trainer.py https://es.wikipedia.org/wiki/Teorema_de_Stokes 100")
        sys.exit(1)
        
    input_topic = sys.argv[1]
    max_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    raw_text = fetch_wikipedia_text(input_topic)
    if not raw_text:
        print("[ABORT] No se pudo obtener contenido.")
        sys.exit(1)
        
    clean_text = normalize_text(raw_text)
    pairs = chunk_text_into_pairs(clean_text, max_pairs)
    if not pairs:
        print("[ABORT] No se generaron pares de entrenamiento.")
        sys.exit(1)
        
    added = update_dataset_and_vocab(pairs)
    if added > 0:
        run_cpp_retraining()
    else:
        print("[INFO] Todos los pares extraídos ya existían en el dataset. No se requiere entrenamiento adicional.")

if __name__ == "__main__":
    main()
