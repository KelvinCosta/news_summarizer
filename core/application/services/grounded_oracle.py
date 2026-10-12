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


FACTUAL_SYSTEM_PROMPT = """Você é o Oráculo Estrito de Notícias do News Summarizer.
Sua função é sintetizar e responder EXCLUSIVAMENTE relatando os acontecimentos, declarações e anúncios noticiados nos documentos em <base_conhecimento>.

Regras inegociáveis:
1. Responda apenas com base nas notícias fornecidas em <base_conhecimento>.
2. É terminantemente PROIBIDO responder com conhecimentos gerais pré-treinados ou fatos de enciclopédia (como explicar o que é a Casa Branca, sua localização, ou história geral). Relate apenas os acontecimentos noticiados nos documentos.
3. Comece diretamente pelos fatos noticiados (ex: "De acordo com as notícias arquivadas...").
4. Termos e nomes próprios em português e inglês referem-se à mesma entidade (ex: White House = Casa Branca, EUA = US/USA, IA = AI).
5. Se a pergunta envolver a data de hoje ou calendário atual, use a [Data de referência do sistema].
6. Se nenhuma das notícias em <base_conhecimento> contiver fatos ou menções sobre o tema consultado, declare claramente que não há informações sobre esse assunto nas notícias arquivadas.
7. Responda de forma direta, clara e sucinta em Português do Brasil.
"""

SUMMARY_SYSTEM_PROMPT = """Você é o Oráculo de Notícias do News Summarizer.
Seu objetivo é sintetizar as notícias com base EXCLUSIVAMENTE no conteúdo de <base_conhecimento>.

Diretrizes:
1. Apresente um resumo claro e objetivo dos acontecimentos, anúncios e tendências noticiados em <base_conhecimento>.
2. É terminantemente PROIBIDO adicionar conhecimentos pré-treinados ou fatos gerais de enciclopédia. Limite-se estritamente aos fatos descritos nos documentos.
3. Nomes próprios e entidades bilíngues em inglês e português referem-se à mesma entidade (ex: White House = Casa Branca, EUA = US/USA, IA = AI).
4. Se os documentos não contiverem o assunto, informe claramente que as notícias arquivadas não contêm essa informação.
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
        now_dt = datetime.now()
        dias_semana = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
        meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
        dia_sem = dias_semana[now_dt.weekday()]
        mes_nome = meses[now_dt.month - 1]
        now_full = f"{now_dt.day} de {mes_nome} de {now_dt.year} ({dia_sem})"
        now_str = now_dt.strftime("%d/%m/%Y")

        # Detecta se a pergunta pede resumo/visão geral
        q_lower = user_query.lower()
        summary_terms = ["resumo", "hoje", "notícias de hoje", "noticias de hoje", "principais notícias", "principais noticias", "panorama", "tendências", "tendencias"]
        use_summary_prompt = is_summary_mode or any(term in q_lower for term in summary_terms)

        system_instruction = SUMMARY_SYSTEM_PROMPT if use_summary_prompt else FACTUAL_SYSTEM_PROMPT

        prompt = (
            f"<base_conhecimento>\n{context_body}\n</base_conhecimento>\n\n"
            f"[Data de referência do sistema: {now_full} - {now_str}]\n\n"
            f"Consulta do usuário: \"{user_query.strip()}\"\n"
            f"Equivalência de entidades: Nomes em inglês e português referem-se à mesma entidade (ex: White House = Casa Branca, EUA = US/USA, IA = AI).\n\n"
            f"Tarefa: Sintetize o que as notícias arquivadas acima em <base_conhecimento> relatam sobre o assunto consultado.\n"
            f"Apresente os fatos, anúncios, acordos e acontecimentos noticiados nos documentos.\n"
            f"PROIBIDO: Não defina prédios, história geral ou conhecimentos externos de enciclopédia. Comece relatando diretamente os acontecimentos noticiados. Se os documentos não contiverem o assunto, informe que não há informações nas notícias arquivadas.\n\n"
            f"Resposta baseada nas notícias:"
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
                "não há nenhuma menção",
                "não há menção",
                "não foram encontradas informações",
                "não contêm informações",
                "não contém informações",
                "não há informações nas notícias",
                "não há informações sobre",
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
