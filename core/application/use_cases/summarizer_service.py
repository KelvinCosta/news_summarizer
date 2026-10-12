"""Caso de uso: Pipeline Map-Reduce com Agentes SLM Locais.

Fase 1 (MAP): Processa artigos brutos inéditos gerando resumos individuais
              e indexando no Hipocampo (ChromaDB) e summaries.parquet.
Fase 2 (REDUCE): Sintetiza o Boletim Diário Global a partir dos resumos
                 e grava em bulletins.parquet.
"""

from datetime import datetime, timezone
import logging
from typing import Callable, List, Optional

from core.domain.article import GlobalBulletin, SummarizedArticle
from core.domain.ports import ISemanticMemory
from core.fsm.states import SystemState
from core.inference.ollama_agent import OllamaMasterAgent, OllamaSummarizerAgent
from core.persistence.chroma_adapter import ChromaDBAdapter
from core.persistence.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class SummarizerService:
    """Orquestrador do fluxo Map-Reduce de notícias."""

    def __init__(
        self,
        parquet_store: Optional[ParquetStore] = None,
        semantic_memory: Optional[ISemanticMemory] = None,
        summarizer_agent: Optional[OllamaSummarizerAgent] = None,
        master_agent: Optional[OllamaMasterAgent] = None,
        model_name: str = "llama3.2",
    ) -> None:
        self.parquet_store = parquet_store or ParquetStore("data/parquet")
        self.semantic_memory = semantic_memory or ChromaDBAdapter("data/chroma")
        self.summarizer_agent = summarizer_agent or OllamaSummarizerAgent(model_name=model_name)
        self.master_agent = master_agent or OllamaMasterAgent(model_name=model_name)

    def run(
        self,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> dict:
        """Executa a esteira Map-Reduce sobre os artigos não resumidos.

        Args:
            limit: Limite opcional de artigos a processar nesta rodada (evita bloqueios prolongados).
            progress_callback: Callback opcional de progresso (indice_atual, total_lote, titulo).
        """
        articles = self.parquet_store.get_unsummarized_articles()

        if not articles:
            logger.info("Não há notícias novas aguardando sumarização hoje.")
            return {
                "status": "empty",
                "processed_count": 0,
                "bulletin": None,
                "message": "Nenhum artigo pendente de processamento.",
            }

        total_pending = len(articles)
        if limit is not None and limit > 0:
            articles = articles[:limit]

        logger.info(
            f"[{SystemState.RETRIEVING_CONTEXT.value}] Detectados {total_pending} artigos inéditos. "
            f"Processando lote de {len(articles)} artigos."
        )

        summarized_objects: List[SummarizedArticle] = []

        # FASE 1: MAP (Sumarizações Individuais)
        logger.info("--- FASE 1: MAP (Resumos Individuais) ---")
        total_batch = len(articles)
        for idx, article in enumerate(articles, 1):
            title = article["title"]
            content = article["content"]
            url = article["url"]

            if progress_callback:
                try:
                    progress_callback(idx, total_batch, title)
                except Exception:
                    pass

            logger.info(f"[{SystemState.INFERENCING.value}] [{idx}/{total_batch}] Agente 1 (Summarizer): Lendo '{title}'")
            summary = self.summarizer_agent.summarize(f"{title}\n\n{content}")

            if summary:
                summarized_obj = SummarizedArticle(
                    original_url=url,
                    title=title,
                    summary=summary,
                    processed_at=datetime.now(timezone.utc),
                )
                summarized_objects.append(summarized_obj)

                # Injeção no Hipocampo (ChromaDB)
                if self.semantic_memory:
                    self.semantic_memory.store_individual_summary(summarized_obj)

                logger.info(f"[{SystemState.COMPLETED.value}] Resumo gerado e vetorizado: '{title}'")
            else:
                logger.error(f"[{SystemState.FAILED.value}] Falha na inferência para '{title}'")

        if not summarized_objects:
            return {
                "status": "failed",
                "processed_count": 0,
                "bulletin": None,
                "message": "Nenhum resumo pôde ser gerado pelos agentes.",
            }

        # Salvando resumos no Parquet
        logger.info(
            f"[{SystemState.CONSOLIDATING.value}] Gravando {len(summarized_objects)} resumos em summaries.parquet..."
        )
        self.parquet_store.append_summaries(summarized_objects)

        # FASE 2: REDUCE (Boletim Global)
        logger.info("--- FASE 2: REDUCE (Boletim Global) ---")
        summary_texts = [s.summary for s in summarized_objects]
        global_bulletin = self.master_agent.generate_global_bulletin(summary_texts)

        if global_bulletin:
            bulletin_obj = GlobalBulletin(
                content=global_bulletin,
                processed_at=datetime.now(timezone.utc),
            )
            self.parquet_store.append_bulletin(bulletin_obj)
            logger.info(
                f"[{SystemState.CONSOLIDATING.value}] Boletim Final salvo em 'bulletins.parquet' com sucesso!"
            )

        return {
            "status": "success",
            "processed_count": len(summarized_objects),
            "bulletin": global_bulletin,
            "summaries": summarized_objects,
        }

