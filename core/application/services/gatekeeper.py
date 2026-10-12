"""Serviço de Triagem e Classificação de Intenções (Estágio 1 do Pipeline).

Classifica a entrada do usuário estritamente em uma das quatro categorias canônicas:
- FACTUAL_QUERY: Perguntas sobre fatos e notícias arquivadas na base.
- TRENDS_SUMMARY: Solicitações de boletim diário, panorama ou tendências consolidadas.
- OUT_OF_DOMAIN: Assuntos fora do escopo de notícias/tecnologia (receitas, piadas, curiosidades).
- HUMAN_INTERVENTION: Solicitação de operador humano ou correção de diretriz (HITL).

Utiliza prompting few-shot otimizado, saída JSON forçada via gramática nativa e
fallback determinístico de segurança para HUMAN_INTERVENTION.
"""

import json
import re
from typing import Optional

from core.domain.models.intents import NewsIntentType
from core.infrastructure.adapters.ollama_client import OllamaClient


def is_temporal_system_query(query: str) -> bool:
    """Verifica deterministicamente se a mensagem é uma consulta sobre a data/dia atual."""
    if not query or not query.strip():
        return False
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    joined = " ".join(q.split())
    temporal_patterns = [
        r"^que dia (e|é) hoje$",
        r"^hoje (e|é) que dia$",
        r"^qual (a|é a|e a) data (de hoje|atual)$",
        r"^qual data (e|é) hoje$",
        r"^que data (e|é) hoje$",
        r"^qual (o|é o|e o) dia (de hoje|atual)$",
        r"^(em )?que dia estamos$",
        r"^que dia do mes (e|é) hoje$",
        r"^qual e a data de hoje$",
    ]
    for pat in temporal_patterns:
        if re.search(pat, joined):
            return True
    return False


def is_catalog_meta_query(query: str) -> Optional[str]:
    """Detecta se a mensagem é uma consulta sobre o catálogo de notícias (mais recente, mais antiga).
    
    Retorna 'recency', 'antiquity' ou None.
    """
    if not query or not query.strip():
        return None
    import re
    q = re.sub(r"[^\w\s]", "", query.lower().strip())

    recency_patterns = [
        r".*not[ií]cia mais recente.*",
        r".*not[ií]cia mais atual.*",
        r".*[uú]ltima not[ií]cia.*",
        r".*[uú]ltimo artigo.*",
        r".*artigo mais recente.*",
        r".*artigo mais atual.*",
    ]
    if any(re.match(p, q) for p in recency_patterns):
        return "recency"

    antiquity_patterns = [
        r".*not[ií]cia mais antiga.*",
        r".*artigo mais antig[ao].*",
        r".*primeira not[ií]cia.*",
        r".*primeiro artigo.*",
    ]
    if any(re.match(p, q) for p in antiquity_patterns):
        return "antiquity"

    return None


GATEKEEPER_SYSTEM_PROMPT = """Você é o classificador de intenções do News Summarizer.
Classifique a mensagem do usuário estritamente em uma das 4 categorias:
- FACTUAL_QUERY: Perguntas pontuais sobre fatos, notícias arquivadas, tecnologias, empresas ou acontecimentos (ex: "O que a Apple lançou?", "Qual sua notícia mais recente?", "Qual a notícia mais antiga?").
- TRENDS_SUMMARY: Pedidos de resumo geral, boletim diário, panorama executivo ou tendências (ex: "Me dê o resumo de hoje", "Quais as tendências de IA?", "Gere um briefing do dia").
- OUT_OF_DOMAIN: Assuntos alheios a notícias, tecnologia e atualidades (ex: piadas, receitas, previsão do tempo genérica, futebol). Perguntas sobre o acervo de notícias do sistema NUNCA são OUT_OF_DOMAIN.
- HUMAN_INTERVENTION: Pedido de intervenção humana, suporte técnico manual ou correção de regras (ex: "Quero falar com um humano", "Alguém pode me ajudar?", "Ajuste a diretriz").

Responda exclusivamente no formato JSON com a chave "intent":
{"intent": "<CATEGORIA>"}

Exemplos:
Entrada: "O que aconteceu com a OpenAI ontem?"
Resposta: {"intent": "FACTUAL_QUERY"}

Entrada: "Qual sua notícia mais recente?"
Resposta: {"intent": "FACTUAL_QUERY"}

Entrada: "Qual a notícia mais antiga?"
Resposta: {"intent": "FACTUAL_QUERY"}

Entrada: "Qual sua notícia mais atual arquivada?"
Resposta: {"intent": "FACTUAL_QUERY"}

Entrada: "Quais os novos avanços em chips de IA reportados nas notícias?"
Resposta: {"intent": "FACTUAL_QUERY"}

Entrada: "Qual empresa comprou a startup de robótica?"
Resposta: {"intent": "FACTUAL_QUERY"}

Entrada: "Me dê um resumo das principais notícias de hoje"
Resposta: {"intent": "TRENDS_SUMMARY"}

Entrada: "Gere o boletim executivo diário de tendências"
Resposta: {"intent": "TRENDS_SUMMARY"}

Entrada: "Quais são as principais tendências de tecnologia da semana?"
Resposta: {"intent": "TRENDS_SUMMARY"}

Entrada: "Como fazer um bolo de chocolate?"
Resposta: {"intent": "OUT_OF_DOMAIN"}

Entrada: "Conte uma piada engraçada"
Resposta: {"intent": "OUT_OF_DOMAIN"}

Entrada: "Qual a capital da Finlândia?"
Resposta: {"intent": "OUT_OF_DOMAIN"}

Entrada: "Quero falar com um operador humano"
Resposta: {"intent": "HUMANO"}

Entrada: "Preciso de intervenção manual da equipe"
Resposta: {"intent": "HUMAN_INTERVENTION"}

Entrada: "Não invente fatos, ajuste a regra de resposta"
Resposta: {"intent": "HUMAN_INTERVENTION"}
"""


class GatekeeperService:
    """Serviço de classificação de intenções do usuário (Estágio 1 do pipeline)."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        model: str = "llama3.2",
    ) -> None:
        """Inicializa o Gatekeeper com cliente Ollama e modelo configuráveis."""
        self._client = client or OllamaClient()
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def classify(self, user_message: str) -> NewsIntentType:
        """Classifica deterministicamente a intenção da mensagem.

        Em qualquer falha de rede, timeout ou payload malformado, retorna
        o fallback seguro NewsIntentType.HUMAN_INTERVENTION.
        """
        if not user_message or not user_message.strip():
            return NewsIntentType.HUMAN_INTERVENTION

        msg_clean = user_message.strip()
        msg_lower = msg_clean.lower()

        # Guardrails heurísticos rápidos (Fail-Fast pré-LLM para economia de ciclos)
        if is_temporal_system_query(msg_clean) or is_catalog_meta_query(msg_clean) is not None:
            return NewsIntentType.FACTUAL_QUERY

        human_indicators = ["falar com humano", "operador humano", "atendente", "intervenção manual", "ajuste a regra"]
        if any(h in msg_lower for h in human_indicators):
            return NewsIntentType.HUMAN_INTERVENTION

        summary_indicators = ["resumo de hoje", "boletim diário", "boletim diario", "mega-boletim", "mega boletim", "briefing do dia", "panorama de hoje"]
        if any(s in msg_lower for s in summary_indicators):
            return NewsIntentType.TRENDS_SUMMARY

        prompt = f'Entrada: "{msg_clean}"\nResposta:'

        try:
            raw_response = self._client.generate(
                prompt=prompt,
                model=self._model,
                require_json=True,
                system=GATEKEEPER_SYSTEM_PROMPT,
                options={"temperature": 0.0},
            )
            return self._parse_intent(raw_response)
        except Exception:
            return NewsIntentType.HUMAN_INTERVENTION

    def _parse_intent(self, raw_response: str) -> NewsIntentType:
        """Processa a resposta JSON do modelo e valida contra o enum NewsIntentType."""
        try:
            data = json.loads(raw_response.strip())
            if not isinstance(data, dict):
                return NewsIntentType.HUMAN_INTERVENTION

            raw_intent = data.get("intent") or data.get("intent_type")
            if not isinstance(raw_intent, str):
                return NewsIntentType.HUMAN_INTERVENTION

            normalized = raw_intent.strip().upper()
            if normalized in ["HUMANO", "HUMAN"]:
                return NewsIntentType.HUMAN_INTERVENTION

            return NewsIntentType(normalized)
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError):
            return NewsIntentType.HUMAN_INTERVENTION

    def __call__(self, user_message: str) -> NewsIntentType:
        return self.classify(user_message)

