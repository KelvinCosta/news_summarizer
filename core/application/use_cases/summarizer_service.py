"""Caso de uso: Pipeline Map-Reduce com Agentes SLM Locais.

Fase 1 (MAP): Processa artigos brutos inéditos gerando resumos individuais
              e indexando no Hipocampo (ChromaDB) e summaries.parquet.
Fase 2 (REDUCE): Sintetiza o Boletim Diário Global a partir dos resumos
                 e grava em bulletins.parquet.
"""

from datetime import datetime, timezone
import hashlib
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
        generate_bulletin: bool = True,
    ) -> dict:
        """Executa a esteira Map-Reduce sobre os artigos não resumidos.

        Args:
            limit: Limite opcional de artigos a processar nesta rodada (evita bloqueios prolongados).
            progress_callback: Callback opcional de progresso (indice_atual, total_lote, titulo).
            generate_bulletin: Se True, executa a Fase 3 (REDUCE) consolidando o boletim do lote.
                               Se False, apenas resume e indexa no Parquet e ChromaDB.
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

        # Se não foi solicitada a geração de boletim, encerra aqui (modo Indexação Pura)
        if not generate_bulletin:
            return {
                "status": "success",
                "processed_count": len(summarized_objects),
                "bulletin": None,
                "summaries": summarized_objects,
            }

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

    def generate_strategic_bulletin(
        self,
        top_k: Optional[int] = 10,
        theme: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> dict:
        """Gera ou recupera do cache o Boletim Consolidado por quantidade (Top K) ou tema filtrado.

        Args:
            top_k: Quantidade de notícias de maior peso a selecionar (se theme for None).
            theme: Palavra-chave/termo para filtrar notícias (utiliza TODAS as encontradas).
            force_regenerate: Se True, ignora o cache e força nova inferência na GPU.
        """
        summaries = self.parquet_store.get_all_summaries()
        if not summaries:
            return {
                "status": "empty",
                "message": "Nenhum resumo encontrado no banco de dados. Execute a opção de indexação primeiro.",
                "bulletin": None,
                "articles_used": [],
                "cached": False,
            }

        # Deduplica por URL para evitar artigos duplicados
        seen_urls = set()
        unique_summaries = []
        for s in summaries:
            url = s.get("original_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_summaries.append(s)

        if theme:
            term = theme.strip().lower()
            topic = f"tema:{term}"
            matching = [
                s for s in unique_summaries
                if term in s.get("title", "").lower() or term in s.get("summary", "").lower()
            ]
            if not matching:
                return {
                    "status": "empty",
                    "message": f"Nenhuma notícia encontrada contendo o tema '{theme}'.",
                    "bulletin": None,
                    "articles_used": [],
                    "cached": False,
                }
            if self.bayesian_filter:
                matching.sort(
                    key=lambda s: self.bayesian_filter.score_article(f"{s.get('title', '')} {s.get('summary', '')}"),
                    reverse=True,
                )
            target_articles = matching
        else:
            k = top_k or 10
            topic = f"top_{k}"
            if self.bayesian_filter:
                unique_summaries.sort(
                    key=lambda s: self.bayesian_filter.score_article(f"{s.get('title', '')} {s.get('summary', '')}"),
                    reverse=True,
                )
            target_articles = unique_summaries[:k]

        # Calcula o hash determinístico baseado nas URLs das notícias selecionadas e no tópico
        sorted_urls = sorted([a.get("original_url", "") for a in target_articles if a.get("original_url")])
        articles_hash = hashlib.sha256(f"{topic}::".encode() + "||".join(sorted_urls).encode()).hexdigest()

        # IDEMPOTÊNCIA / CACHE INTELIGENTE:
        # Se já existe um boletim gerado com exatamente este lote de notícias, retorna o existente
        if not force_regenerate:
            cached = self.parquet_store.find_bulletin_by_hash(articles_hash)
            if cached:
                logger.info(f"Boletim existente em cache recuperado para '{topic}' (hash={articles_hash[:8]}).")
                return {
                    "status": "cached",
                    "bulletin": cached["content"],
                    "processed_at": cached["processed_at"],
                    "topic": topic,
                    "articles_used": target_articles,
                    "count": len(target_articles),
                    "cached": True,
                }

        # REDUÇÃO (MAP-REDUCE)
        logger.info(f"Gerando novo Boletim para '{topic}' com {len(target_articles)} notícias...")
        summary_texts = [s["summary"] for s in target_articles]

        # Se for um lote muito grande (ex: tema com mais de 20 notícias), faz redução hierárquica
        if len(summary_texts) > 20:
            chunk_size = 15
            partial_bulletins = []
            for i in range(0, len(summary_texts), chunk_size):
                chunk = summary_texts[i:i + chunk_size]
                partial = self.master_agent.generate_global_bulletin(chunk)
                if partial:
                    partial_bulletins.append(partial)

            if partial_bulletins:
                global_bulletin = self.master_agent.generate_daily_master(
                    partial_bulletins,
                    date_str=datetime.now().strftime("%d/%m/%Y"),
                )
            else:
                global_bulletin = None
        else:
            global_bulletin = self.master_agent.generate_global_bulletin(summary_texts)

        if global_bulletin:
            bulletin_obj = GlobalBulletin(
                content=global_bulletin,
                processed_at=datetime.now(timezone.utc),
                topic=topic,
                articles_hash=articles_hash,
            )
            self.parquet_store.append_bulletin(bulletin_obj)
            logger.info(f"Novo boletim '{topic}' gerado e salvo em 'bulletins.parquet'.")

        return {
            "status": "success" if global_bulletin else "failed",
            "bulletin": global_bulletin,
            "processed_at": datetime.now(timezone.utc),
            "topic": topic,
            "articles_used": target_articles,
            "count": len(target_articles),
            "cached": False,
        }

    def generate_top_bulletin(self, top_k: int = 10) -> dict:
        """Compatibilidade retroativa: chama generate_strategic_bulletin."""
        return self.generate_strategic_bulletin(top_k=top_k)

