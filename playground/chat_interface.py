import subprocess
import os
import sys

class CalculusLLMChat:
    def __init__(self, binary_path):
        self.binary_path = binary_path
        
        if not os.path.exists(binary_path):
            print(f"Error: No se encuentra el binario en {binary_path}")
            print("Asegúrate de haber compilado el proyecto con CMake.")
            sys.exit(1)

    def ask(self, prompt):
        # Ejecutamos el binario y le pasamos el prompt por stdin
        # Usamos un proceso separado para cada inferencia para demostrar la convergencia limpia
        try:
            # Preparamos el comando para que termine después de una respuesta
            # El binario espera 'salir' para terminar, así que enviamos el prompt y luego 'salir'
            input_data = f"{prompt}\nsalir\n"
            
            process = subprocess.Popen(
                [self.binary_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(self.binary_path) # Ejecutar en el build dir por el vocabulario
            )
            
            stdout, stderr = process.communicate(input=input_data)
            
            if process.returncode != 0:
                return f"Error en el motor: {stderr}"

            # Post-procesamiento para extraer solo la respuesta del LLM
            lines = stdout.split('\n')
            response = ""
            for line in lines:
                if "[LLM] Respuesta final:" in line:
                    response = line.replace("[LLM] Respuesta final:", "").strip()
                    break
            
            return response if response else "El motor no pudo converger en una respuesta."

        except Exception as e:
            return f"Excepción al llamar al motor: {str(e)}"

def find_binary(base_dir):
    candidates = [
        os.path.join(base_dir, "calculus_llm", "build", "calculus_llm"),
        os.path.join(base_dir, "calculus_llm", "build", "calculus_llm", "calculus_llm"),
    ]
    env_bin = os.environ.get("CALCULUS_LLM_BIN")
    if env_bin and os.path.exists(env_bin):
        return env_bin
    for c in candidates:
        if os.path.exists(c):
            return c
    # Intentar encontrar con which
    import shutil
    return shutil.which("calculus_llm") or candidates[0]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    binary = find_binary(base_dir)

    print("\033[95m" + "="*56 + "\033[0m")
    print("\033[96m   SLIME: SELECTIVE LATENT INTEGRAL MODEL ENGINE (v2.0) \033[0m")
    print("\033[95m" + "="*56 + "\033[0m")
    print("Inferencia híbrida basada en Neural ODEs (RK45) y Lógica Positrónica.")
    print("Escribe 'salir' o 'exit' para terminar.\n")

    chat = CalculusLLMChat(binary)

    while True:
        try:
            user_input = input("\033[92mTú > \033[0m")
            
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\033[93mCerrando conexión con el motor matemático...\033[0m")
                break
            
            if not user_input.strip():
                continue

            print("\033[90m[Calculando gradientes...]\033[0m", end="\r")
            response = chat.ask(user_input)
            
            print("\033[94mLLM > \033[0m" + response)
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nCerrando...")
            break

if __name__ == "__main__":
    main()
