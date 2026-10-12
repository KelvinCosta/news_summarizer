"""Caso de uso: Pipeline Map-Reduce com Agentes SLM Locais.

Fase 1 (MAP): Processa artigos brutos inéditos gerando resumos individuais
              e indexando no Hipocampo (ChromaDB) e summaries.parquet.
Fase 2 (REDUCE): Sintetiza o Boletim Diário Global a partir dos resumos
                 e grava em bulletins.parquet.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Callable, List, Optional

from core.domain.article import GlobalBulletin, SummarizedArticle
from core.domain.ports import ISemanticMemory
from core.fsm.states import SystemState
from core.inference.ollama_agent import OllamaMasterAgent, OllamaSummarizerAgent
from core.persistence.chroma_adapter import ChromaDBAdapter
from core.persistence.parquet_store import ParquetStore
from core.filters.bayesian import BayesianFilter

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
        bayesian_filter: Optional[BayesianFilter] = None,
    ) -> None:
        self.parquet_store = parquet_store or ParquetStore("data/parquet")
        self.semantic_memory = semantic_memory or ChromaDBAdapter("data/chroma")
        self.summarizer_agent = summarizer_agent or OllamaSummarizerAgent(model_name=model_name)
        self.master_agent = master_agent or OllamaMasterAgent(model_name=model_name)
        self.bayesian_filter = bayesian_filter or BayesianFilter("vocabulary_seed.json")

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

        # Prioriza no topo os artigos com maior relevância analítica (Score Bayesiano)
        if self.bayesian_filter:
            articles.sort(
                key=lambda a: self.bayesian_filter.score_article(f"{a.get('title', '')} {a.get('content', '')}"),
                reverse=True,
            )

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
        try:
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

                    # CHECKPOINT IMEDIATO: Salva cada resumo no Parquet assim que concluído.
                    # Se o usuário cancelar ou houver interrupção, o progresso nunca é perdido.
                    try:
                        self.parquet_store.append_summaries([summarized_obj])
                    except Exception as e:
                        logger.warning(f"Erro ao salvar checkpoint incremental no Parquet: {e}")

                    logger.info(f"[{SystemState.COMPLETED.value}] Resumo gerado e gravado no Parquet: '{title}'")
                else:
                    logger.error(f"[{SystemState.FAILED.value}] Falha na inferência para '{title}'")

                # Arrefecimento breve (100ms) para liberar I/O e evitar congelamento de GPU/TDR no Windows
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.warning("Interrupção solicitada pelo usuário (Ctrl+C). Finalizando lote com os resumos concluídos...")
            print("\n⚠️ Interrupção manual (Ctrl+C) detectada! Prosseguindo para consolidar os resumos já concluídos...")

        if not summarized_objects:
            return {
                "status": "failed",
                "processed_count": 0,
                "bulletin": None,
                "message": "Nenhum resumo pôde ser gerado pelos agentes.",
            }

        # FASE 2: INDEXAÇÃO NO HIPOCAMPO (ChromaDB)
        # Executada após a finalização de todos os resumos para evitar "ping-pong" de modelos na GPU
        logger.info("--- FASE 2: INDEXAÇÃO NO HIPOCAMPO (ChromaDB) ---")
        if self.semantic_memory:
            try:
                for obj in summarized_objects:
                    self.semantic_memory.store_individual_summary(obj)
                logger.info(f"Vetorizados {len(summarized_objects)} resumos no Hipocampo com sucesso.")
            except Exception as e:
                logger.warning(f"Aviso ao indexar resumos no ChromaDB: {e}")

        # FASE 3: REDUCE (Boletim Global)
        logger.info("--- FASE 3: REDUCE (Boletim Global com Agente Master) ---")
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

