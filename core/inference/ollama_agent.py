import ollama
from typing import Optional

class OllamaSummarizerAgent:
    """
    Agente SLM especializado em sumarização.
    Conecta-se ao Ollama local para inferência restrita.
    """
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def summarize(self, text: str) -> Optional[str]:
        prompt = (
            "Resuma a seguinte notícia em no máximo 3 parágrafos, extraindo os pontos principais. "
            "Responda em Português do Brasil.\n\n"
            f"Notícia:\n{text}"
        )
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    "role": "system", 
                    "content": "Você é um assistente hiper-especializado em sumarizar dados e remover ruídos, seguindo princípios on-premise estritos. Produza resumos curtos e diretos."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ])
            return response['message']['content']
        except ollama.ResponseError as e:
            if "not found" in str(e).lower():
                print(f"\n[ERRO] O modelo '{self.model_name}' não foi encontrado no seu Ollama local.")
                print(f"Por favor, rode no seu terminal: ollama run {self.model_name}\n")
            else:
                print(f"Erro na API do Ollama: {e}")
            return None
        except Exception as e:
            print(f"Erro ao conectar com Ollama (ele está rodando?): {e}")
            return None
