"""Serviço de Modo Oráculo Estrito e Grounded Topic Retrieval (Estágio 3 e 4).

Executa a consulta na memória semântica vetorial (ChromaDB) ou Lake de Notícias (Parquet),
formata o contexto com metadados (Data, Origem, Título) e sintetiza respostas factuais
blindadas contra alucinações.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from core.domain.models.intents import NewsIntentType
from core.domain.ports import ISemanticMemory
from core.infrastructure.adapters.ollama_client import OllamaClient


FACTUAL_SYSTEM_PROMPT = """Você é o Oráculo Estrito do News Summarizer.
Seu objetivo é responder à pergunta do usuário utilizando EXCLUSIVAMENTE o conteúdo contido na <base_conhecimento>.

Regras inegociáveis:
1. Responda APENAS com base nos fatos contidos em <base_conhecimento>.
2. Se a <base_conhecimento> mencionar a entidade, projeto, empresa ou pessoa perguntada, relate tudo o que constar no texto sobre ela de forma objetiva, direta e sucinta.
3. Se a <base_conhecimento> NÃO contiver NENHUMA informação ou menção sobre o assunto perguntado, responda EXATAMENTE a seguinte frase:
"Não possuo dados suficientes em minha base interna para responder a esta pergunta."
4. Não use conhecimento prévio ou externo sobre a internet. Não invente datas, fatos, nomes ou acontecimentos.
5. Responda de forma direta, clara e sucinta em Português do Brasil.
"""

SUMMARY_SYSTEM_PROMPT = """Você é o Oráculo de Notícias do News Summarizer.
Seu objetivo é sintetizar as notícias e tendências com base EXCLUSIVAMENTE no conteúdo de <base_conhecimento>.

Diretrizes:
1. Apresente um resumo claro, estruturado e em tópicos dos acontecimentos e tendências das notícias mais recentes presentes na <base_conhecimento>.
2. Se o usuário perguntar pelas notícias de hoje, novidades ou resumo geral, sintetize as notícias mais recentes fornecidas, destacando seus títulos ou tópicos.
3. NÃO invente notícias que não constem na <base_conhecimento>.
4. Se a <base_conhecimento> estiver vazia ou for sobre assunto completamente diferente sem relação, responda:
"Não possuo dados suficientes em minha base interna para responder a esta pergunta."
5. Responda em Português do Brasil de forma executiva, objetiva e fluida.
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
        limit: int = 5,
        provided_contexts: Optional[List[Dict[str, Any]]] = None,
        is_summary_mode: bool = False,
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        """Gera resposta grounded a partir da memória semântica.

        Args:
            user_query: Consulta do usuário.
            limit: Quantidade de documentos a recuperar se provided_contexts for None.
            provided_contexts: Contextos pré-carregados (ex: boletins consolidados).
            is_summary_mode: Se True, aplica prompt de síntese executiva em vez de checagem pontual.

        Returns:
            Tupla contendo:
            - response_text: Resposta gerada ou frase de recusa padrão
            - contexts: Lista de contextos recuperados para rastreabilidade/auditoria
            - is_refusal: Booleano indicando se a resposta foi uma recusa por falta de dados
        """
        if not user_query or not user_query.strip():
            return "Por favor, formule uma pergunta sobre as notícias arquivadas.", [], False

        # Recupera contextos caso não tenham sido passados
        if provided_contexts is not None:
            contexts = provided_contexts
        elif self._memory is not None:
            try:
                contexts = self._memory.recall_context(user_query, limit=limit)
            except Exception:
                return self.STRICT_REFUSAL_MESSAGE, [], True
        else:
            return self.STRICT_REFUSAL_MESSAGE, [], True

        # Guardrail 1: Se não há documentos recuperados
        if not contexts:
            return self.STRICT_REFUSAL_MESSAGE, [], True

        # Formata o corpo do contexto com metadados estruturados (Data, Fonte, Título)
        formatted_items: List[str] = []
        for i, c in enumerate(contexts, start=1):
            meta = c.get("metadata", {})
            dt = meta.get("date") or meta.get("processed_at", "")
            if dt and len(dt) >= 10:
                dt_str = dt[:10]
            else:
                dt_str = "Data recente"

            title = meta.get("title") or meta.get("source_url") or f"Artigo {i}"
            tipo = meta.get("type", "noticia")

            header = f"[Documento {i} | Tipo: {tipo} | Data: {dt_str} | Título/Fonte: {title}]"
            content = c.get("content", "").strip()
            if content:
                formatted_items.append(f"{header}\n{content}")

        if not formatted_items:
            return self.STRICT_REFUSAL_MESSAGE, [], True

        context_body = "\n\n---\n\n".join(formatted_items)
        now_str = datetime.now().strftime("%d/%m/%Y")

        # Detecta se a pergunta pede resumo/visão geral
        q_lower = user_query.lower()
        summary_terms = ["resumo", "hoje", "notícias de hoje", "noticias de hoje", "principais notícias", "principais noticias", "panorama", "tendências", "tendencias"]
        use_summary_prompt = is_summary_mode or any(term in q_lower for term in summary_terms)

        system_instruction = SUMMARY_SYSTEM_PROMPT if use_summary_prompt else FACTUAL_SYSTEM_PROMPT

        prompt = (
            f"<base_conhecimento>\n{context_body}\n</base_conhecimento>\n\n"
            f"[Data de referência do sistema: {now_str}]\n"
            f"Pergunta do usuário: {user_query.strip()}\n"
            "Resposta:"
        )

        try:
            raw_response = self._client.generate(
                prompt=prompt,
                model=self._model,
                require_json=False,
                system=system_instruction,
                options={"temperature": 0.2 if use_summary_prompt else 0.0},
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
        self,
        user_query: str,
        limit: int = 5,
        provided_contexts: Optional[List[Dict[str, Any]]] = None,
        is_summary_mode: bool = False,
    ) -> tuple[str, List[Dict[str, Any]], bool]:
        return self.answer(
            user_query,
            limit=limit,
            provided_contexts=provided_contexts,
            is_summary_mode=is_summary_mode,
        )
