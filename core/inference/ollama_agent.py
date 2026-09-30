import ollama
from typing import Optional, List

class OllamaSummarizerAgent:
    """
    Agente SLM especializado em sumarização INDIVIDUAL.
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
        except Exception as e:
            print(f"Erro no Agente de Sumarização ({self.model_name}): {e}")
            return None


class OllamaMasterAgent:
    """
    Agente SLM Master (Map-Reduce).
    Lê os resumos individuais e cria um boletim global consolidado.
    """
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def generate_global_bulletin(self, summaries: List[str]) -> Optional[str]:
        context = "\n\n---\n\n".join(summaries)
        prompt = (
            "Abaixo estão vários resumos individuais de notícias recentes.\n"
            "Escreva um único Boletim Diário conectando essas informações de forma coesa "
            "e estruturada. Use formatação Markdown, crie um título atrativo, destaque as principais "
            "tendências em tópicos (bullet points) e finalize com um parágrafo rápido de conclusão.\n\n"
            f"RESUMOS:\n{context}"
        )
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    "role": "system", 
                    "content": "Você é o Orquestrador Chefe de Notícias (Agente Master), especializado em redigir relatórios executivos (Briefings) em Português a partir de múltiplos resumos."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Erro no Agente Master ({self.model_name}): {e}")
            return None

    def generate_daily_master(self, bulletins_content: List[str], date_str: str) -> Optional[str]:
        """Lê os vários boletins de um dia e funde num Mega-Boletim Final."""
        context = "\n\n=== BOLETIM PARCIAL ===\n\n".join(bulletins_content)
        prompt = (
            f"Abaixo estão os relatórios parciais gerados durante o dia {date_str}.\n"
            "Escreva o **Sumário Geral Definitivo do Dia**. Conecte as notícias de forma inteligente, "
            "elimine qualquer informação repetida entre os relatórios parciais, destaque as notícias mais importantes, "
            "e crie uma conclusão fechando as tendências de tecnologia do dia.\n"
            "Use Markdown e estruture bem o texto para leitura executiva.\n\n"
            f"RELATÓRIOS PARCIAIS:\n{context}"
        )
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    "role": "system", 
                    "content": "Você é o Orquestrador Chefe de Notícias (Agente Master Definitivo). Sua função é consolidar múltiplos boletins diários em um único mega-relatório executivo em Português."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Erro no Agente Master (Daily Summarizer) ({self.model_name}): {e}")
            return None
