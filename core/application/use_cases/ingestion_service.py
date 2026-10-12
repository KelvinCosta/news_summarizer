"""Caso de uso: Ingestão de Notícias e Funil de Triagem O(1).

Executa o fetch de feeds RSS, aplica o funil matemático de descarte:
1. SimHash (Hamming Distance) para duplicatas léxicas em O(1)
2. Classificador Naive Bayes (Cold Start via vocabulary_seed.json) para relevância
3. Persistência append-only no Data Lake Parquet
4. Emissão de eventos imutáveis no EventStore
"""

import logging
from typing import List, Optional
from uuid import uuid4

from core.domain.article import Article
from core.domain.models.events import ArticleFiltered, ArticleIngested, ImmutableEventStore
from core.filters.bayesian import BayesianFilter
from core.filters.lexical import LexicalFilter
from core.fsm.states import SystemState
from core.ingestion.rss_parser import RSSParser
from core.persistence.parquet_store import ParquetStore

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "http://rss.cnn.com/rss/edition_technology.rss",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
]


class IngestionService:
    """Orquestra o ciclo de ingestão e funil O(1)."""

    def __init__(
        self,
        parquet_store: Optional[ParquetStore] = None,
        lexical_filter: Optional[LexicalFilter] = None,
        bayesian_filter: Optional[BayesianFilter] = None,
        event_store: Optional[ImmutableEventStore] = None,
        feeds: Optional[List[str]] = None,
    ) -> None:
        self.parquet_store = parquet_store or ParquetStore("data/parquet")
        self.lexical_filter = lexical_filter or LexicalFilter(tolerance=3)
        self.bayesian_filter = bayesian_filter or BayesianFilter("vocabulary_seed.json")
        self.event_store = event_store or ImmutableEventStore()
        self.feeds = feeds or DEFAULT_FEEDS

    def run(self, custom_articles: Optional[List[Article]] = None) -> dict:
        """Executa o ciclo completo de ingestão e filtragem.

        Args:
            custom_articles: Lista opcional de artigos brutos (útil para testes unitários/offline).
        """
        session_id = str(uuid4())
        logger.info(f"[{SystemState.INGESTING.value}] Iniciando ciclo de ingestão de feeds...")

        if custom_articles is not None:
            articles = custom_articles
        else:
            parser = RSSParser(self.feeds)
            try:
                articles = parser.fetch_all()
            except Exception as e:
                logger.error(f"Erro ao capturar feeds RSS: {e}")
                articles = []

        logger.info(f"Total de artigos brutos extraídos: {len(articles)}")

        existing_hashes = self.parquet_store.get_all_simhashes()
        accepted_articles: List[Article] = []
        discarded_duplicates = 0
        discarded_irrelevant = 0

        for article in articles:
            # Evento de recepção bruta
            self.event_store.append(
                ArticleIngested(
                    session_id=session_id,
                    url=article.url,
                    title=article.title,
                    source=article.source,
                )
            )

            # A. Filtro Lexical (SimHash)
            new_hash = self.lexical_filter.compute_hash(article.content)
            object.__setattr__(article, "simhash_value", new_hash)

            if self.lexical_filter.is_duplicate(new_hash, existing_hashes):
                discarded_duplicates += 1
                logger.warning(
                    f"[{SystemState.DISCARDED_LEXICALDUPLICATE.value}] Descartado (Cópia): {article.title}"
                )
                self.event_store.append(
                    ArticleFiltered(
                        session_id=session_id,
                        url=article.url,
                        title=article.title,
                        status="discarded_lexical_duplicate",
                    )
                )
                continue

            existing_hashes.append(new_hash)

            # B. Filtro Bayesiano (Naive Bayes)
            full_text = f"{article.content} {article.title}"
            score = self.bayesian_filter.score_article(full_text)
            if not self.bayesian_filter.is_relevant(full_text, threshold=1.0):
                discarded_irrelevant += 1
                logger.warning(
                    f"[{SystemState.DISCARDED_IRRELEVANT.value}] Descartado (Irrelevante/Score {score}): {article.title}"
                )
                self.event_store.append(
                    ArticleFiltered(
                        session_id=session_id,
                        url=article.url,
                        title=article.title,
                        status="discarded_irrelevant",
                        score=score,
                    )
                )
                continue

            logger.info(f"[{SystemState.ROUTING.value}] Artigo aceito (Score {score}): {article.title}")
            self.event_store.append(
                ArticleFiltered(
                    session_id=session_id,
                    url=article.url,
                    title=article.title,
                    status="accepted",
                    score=score,
                )
            )
            accepted_articles.append(article)

        # Salva no Parquet
        if accepted_articles:
            logger.info(
                f"[{SystemState.CONSOLIDATING.value}] Gravando {len(accepted_articles)} artigos válidos no Parquet..."
            )
            self.parquet_store.append(accepted_articles)

        return {
            "session_id": session_id,
            "total_raw": len(articles),
            "accepted": len(accepted_articles),
            "discarded_duplicates": discarded_duplicates,
            "discarded_irrelevant": discarded_irrelevant,
            "articles": accepted_articles,
        }

