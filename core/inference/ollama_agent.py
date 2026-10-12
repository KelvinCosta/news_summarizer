"""Agentes SLM Locais para Map-Reduce (Clean Architecture).

Conectam-se ao Ollama via OllamaClient HTTP puro desacoplado,
permitindo injeção de dependência e testes unitários determinísticos.
"""

from typing import List, Optional

from core.infrastructure.adapters.ollama_client import OllamaClient


class OllamaSummarizerAgent:
    """Agente SLM especializado em sumarização INDIVIDUAL (Etapa MAP)."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        client: Optional[OllamaClient] = None,
    ) -> None:
        self.model_name = model_name
        self._client = client or OllamaClient()

    def summarize(self, text: str) -> Optional[str]:
        prompt = (
            "Extraia os fatos desta notícia em até 3 parágrafos objetivos no padrão jornalístico brasileiro.\n\n"
            "DIRETRIZES DE EXTRAÇÃO:\n"
            "1. Identifique claramente os sujeitos, as empresas ou entidades citadas e as ações que realizaram.\n"
            "2. Se a notícia relatar um crime, golpe ou fraude cometida por terceiros, deixe claro quem é o autor e quem é a vítima/empresa afetada.\n"
            "3. Se for uma newsletter com vários temas curtos, resuma apenas as notícias principais sobre tecnologia e IA.\n"
            "4. Responda em Português do Brasil de forma direta, sem introduções vazias como 'Aqui está o resumo'.\n\n"
            f"TEXTO DA NOTÍCIA:\n{text}"
        )
        system_instruction = (
            "Você é um extrator de inteligência jornalística de alta precisão. "
            "Sua função é transformar artigos brutos em fatos claros e objetivos com atribuição correta."
        )
        try:
            response = self._client.generate(
                prompt=prompt,
                model=self.model_name,
                system=system_instruction,
                options={"temperature": 0.2},
            )
            return response.strip() if response else None
        except Exception as e:
            print(f"Erro no Agente de Sumarização ({self.model_name}): {e}")
            return None


class OllamaMasterAgent:
    """Agente SLM Master (Map-Reduce).

    Lê os resumos individuais e cria um boletim global consolidado com
    análise de tendências e um radar ativo de veracidade/desinformação (Etapa REDUCE).
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        client: Optional[OllamaClient] = None,
    ) -> None:
        self.model_name = model_name
        self._client = client or OllamaClient()

    def generate_global_bulletin(self, summaries: List[str]) -> Optional[str]:
        context = "\n\n---\n\n".join(summaries)
        prompt = (
            "Com base estritamente nos resumos jornalísticos abaixo, componha o Boletim Diário de Inteligência Tecnológica.\n\n"
            "REGRAS DE CONDICIONAMENTO:\n"
            "1. ANCORAGEM NO CONTEXTO: Todos os dados abaixo são notícias reais e recentes apuradas pela imprensa. Nunca classifique as notícias como falsas nem as refute com base em conhecimentos prévios.\n"
            "2. ATRIBUIÇÃO JORNALÍSTICA: Sempre que citar declarações ou propostas mencione como uma declaração pública reportada pelos veículos de imprensa.\n"
            "3. ESTRUTURA DO BOLETIM (Siga rigidamente este padrão Markdown):\n\n"
            "# 📰 BOLETIM DIÁRIO: INTELIGÊNCIA TECNOLÓGICA & MERCADO\n\n"
            "## 🚀 Destaques Principais & Tendências\n"
            "(Apresente as 3 a 5 principais notícias do dia em tópicos com negrito no título e explicação concisa do fato)\n\n"
            "## ⚠️ Radar Crítico & Riscos Reportados\n"
            "(Aponte aqui fraudes, golpes com IA, declarações controversas ou desafios regulatórios relatados nas notícias. Se não houver relatos de risco, escreva: 'Nenhum risco ou fraude tecnológica reportada neste lote.')\n\n"
            "## 💡 Conclusão Executiva\n"
            "(Parágrafo final de 3 a 4 linhas sintetizando o impacto dos acontecimentos no mercado.)\n\n"
            f"RESUMOS DAS FONTES:\n{context}"
        )
        system_instruction = (
            "Você é o Editor-Chefe de Inteligência Executiva de Tecnologia. "
            "Você redige boletins analíticos e concisos em Português corporativo, baseando-se estritamente no material fornecido, "
            "com postura séria, imparcial e sem emitir opiniões pessoais de debunker."
        )
        try:
            response = self._client.generate(
                prompt=prompt,
                model=self.model_name,
                system=system_instruction,
                options={"temperature": 0.2},
            )
            return response.strip() if response else None
        except Exception as e:
            print(f"Erro no Agente Master ({self.model_name}): {e}")
            return None

    def generate_daily_master(
        self, bulletins_content: List[str], date_str: str
    ) -> Optional[str]:
        """Lê os vários boletins de um dia e funde num Mega-Boletim Final."""
        context = "\n\n=== BOLETIM PARCIAL ===\n\n".join(bulletins_content)
        prompt = (
            f"Consolide os relatórios parciais do dia {date_str} em um único Mega-Boletim Executivo Definitivo.\n\n"
            "REGRAS:\n"
            "1. Elimine redundâncias entre as sessões parciais e selecione os fatos de maior impacto.\n"
            "2. Mantenha as seções: Destaques Principais, Radar Crítico & Riscos Reportados, e Conclusão Executiva.\n"
            "3. Redija em Português profissional estruturado em Markdown, sem refutar notícias apuradas pelas fontes.\n\n"
            f"RELATÓRIOS DO DIA:\n{context}"
        )
        system_instruction = (
            "Você é o Editor-Chefe Definitivo de Inteligência de Mercado. Sua função é fundir múltiplos relatórios diários "
            "em um sumário executivo conciso, de leitura rápida e alto valor analítico."
        )
        try:
            response = self._client.generate(
                prompt=prompt,
                model=self.model_name,
                system=system_instruction,
                options={"temperature": 0.2},
            )
            return response.strip() if response else None
        except Exception as e:
            print(f"Erro no Agente Master (Daily Summarizer) ({self.model_name}): {e}")
            return None
