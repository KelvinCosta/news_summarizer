"""Serviço de Modo Oráculo Estrito e Grounded Topic Retrieval (Estágio 3 e 4).

Executa a consulta na memória semântica vetorial (ChromaDB) ou Lake de Notícias (Parquet),
isola o contexto recuperado e sintetiza respostas factuais blindadas contra alucinações.
Aplica guardrail determinístico: caso a resposta não esteja nos dados recuperados,
emite a resposta padronizada de recusa sem inventar fatos.
"""

from typing import Any, Dict, List, Optional
from core.domain.ports import ISemanticMemory
from core.infrastructure.adapters.ollama_client import OllamaClient


ORACLE_SYSTEM_PROMPT = """Você é o Oráculo Estrito do News Summarizer.
Seu objetivo é responder à pergunta do usuário utilizando EXCLUSIVAMENTE o conteúdo contido na <base_conhecimento>.

Regras inegociáveis:
1. Responda APENAS com base nos fatos contidos em <base_conhecimento>.
2. Se a <base_conhecimento> NÃO contiver os dados necessários para responder com certeza absoluta, responda EXATAMENTE a seguinte frase:
"Não possuo dados suficientes em minha base interna para responder a esta pergunta."
3. Não use conhecimento prévio ou externo sobre a internet. Não invente datas, fatos, nomes ou acontecimentos.
4. Responda de forma direta, clara e sucinta em Português do Brasil.
"""


class GroundedOracleService:
    """Serviço de resposta factual ancorada em memória semântica e guardrails."""

    STRICT_REFUSAL_MESSAGE: str = (
        "Não possuo dados suficientes em minha base interna para responder a esta pergunta."
    )

    def __init__(
        self,
        memory: Optional[ISemanticMemory] = None,
        client: Optional[OllamaClient] = None,
        model: str = "llama3.2",
    ) -> None:
        self._memory = memory
        self._client = client or OllamaClient()
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def answer(
        self,
        user_query: str,
        limit: int = 4,
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        """Gera resposta grounded a partir da memória semântica.

        Returns:
            Tupla contendo:
            - response_text: Resposta gerada ou frase de recusa padrão
            - contexts: Lista de contextos recuperados para rastreabilidade/auditoria
            - is_refusal: Booleano indicando se a resposta foi uma recusa por falta de dados
        """
        if not user_query or not user_query.strip():
            return "Por favor, formule uma pergunta sobre as notícias arquivadas.", [], False

        # Se a memória não estiver configurada, recusa com segurança
        if self._memory is None:
            return self.STRICT_REFUSAL_MESSAGE, [], True

        try:
            contexts = self._memory.recall_context(user_query, limit=limit)
        except Exception:
            return self.STRICT_REFUSAL_MESSAGE, [], True

        # Guardrail 1: Se a busca vetorial não encontrou memórias relevantes
        if not contexts:
            return self.STRICT_REFUSAL_MESSAGE, [], True

        context_body = "\n\n---\n\n".join(
            [c.get("content", "") for c in contexts if c.get("content")]
        )
        if not context_body.strip():
            return self.STRICT_REFUSAL_MESSAGE, [], True

        system_instruction = ORACLE_SYSTEM_PROMPT
        prompt = (
            f"<base_conhecimento>\n{context_body}\n</base_conhecimento>\n\n"
            f"Pergunta do usuário: {user_query.strip()}\n"
            "Resposta:"
        )

        try:
            raw_response = self._client.generate(
                prompt=prompt,
                model=self._model,
                require_json=False,
                system=system_instruction,
                options={"temperature": 0.0},
            )
            response_text = raw_response.strip()

            # Guardrail 2: Validação de recusa ou alucinação
            refusal_indicators = [
                "não possuo dados suficientes",
                "nao possuo dados suficientes",
                "não tenho dados suficientes",
                "não consta na base",
                "base de conhecimento não menciona",
            ]
            is_refusal = any(ref in response_text.lower() for ref in refusal_indicators)
            if is_refusal or not response_text:
                return self.STRICT_REFUSAL_MESSAGE, contexts, True

            return response_text, contexts, False
        except Exception:
            return self.STRICT_REFUSAL_MESSAGE, contexts, True

    def __call__(
        self, user_query: str, limit: int = 4
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        return self.answer(user_query, limit=limit)

