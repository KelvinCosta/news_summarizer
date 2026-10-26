"""Caso de uso: Ingestão de Notícias e Funil de Triagem O(1).

Executa o fetch de feeds RSS, aplica o funil matemático de descarte:
1. SimHash (Hamming Distance) para duplicatas léxicas em O(1)
2. Classificador Naive Bayes (Cold Start via vocabulary_seed.json) para relevância
3. Persistência append-only no Data Lake Parquet
4. Emissão de eventos imutáveis no EventStore
"""

import logging
from typing import Callable, List, Optional
from uuid import uuid4

from core.domain.article import Article
from core.domain.models.events import ArticleFiltered, ArticleIngested, ImmutableEventStore
from core.filters.bayesian import BayesianFilter
from core.filters.lexical import LexicalFilter
from core.fsm.states import SystemState
from core.ingestion.rss_parser import RSSParser
from core.persistence.parquet_store import ParquetStore
from pathlib import Path
import json

logger = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "http://rss.cnn.com/rss/edition_technology.rss",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
]


def load_configured_feeds(feeds_path: str = "feeds.json") -> List[str]:
    """Carrega as URLs de feeds RSS a partir de feeds.json ou feeds.example.json."""
    path = Path(feeds_path)
    if not path.exists():
        example_path = Path("feeds.example.json")
        if example_path.exists():
            path = example_path
        else:
            return DEFAULT_FEEDS

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            valid = [url.strip() for url in data if isinstance(url, str) and url.strip()]
            if valid:
                return valid
        elif isinstance(data, dict):
            feeds_list = data.get("feeds", [])
            valid = [url.strip() for url in feeds_list if isinstance(url, str) and url.strip()]
            if valid:
                return valid
    except Exception as e:
        logger.warning(f"Erro ao ler feeds em {path}: {e}. Utilizando feeds padrão.")

    return DEFAULT_FEEDS


class IngestionService:
    """Orquestra o ciclo de ingestão e funil O(1)."""

    def __init__(
        self,
        parquet_store: Optional[ParquetStore] = None,
        lexical_filter: Optional[LexicalFilter] = None,
        bayesian_filter: Optional[BayesianFilter] = None,
        event_store: Optional[ImmutableEventStore] = None,
        feeds: Optional[List[str]] = None,
        threshold: float = 0.5,
    ) -> None:
        self.parquet_store = parquet_store or ParquetStore("data/parquet")
        self.lexical_filter = lexical_filter or LexicalFilter(tolerance=3)
        self.bayesian_filter = bayesian_filter or BayesianFilter("vocabulary_seed.json")
        self.event_store = event_store or ImmutableEventStore()
        self.feeds = feeds or load_configured_feeds()
        self.threshold = threshold

    def run(
        self,
        custom_articles: Optional[List[Article]] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> dict:
        """Executa o ciclo completo de ingestão e filtragem.

        Args:
            custom_articles: Lista opcional de artigos brutos (útil para testes unitários/offline).
            progress_callback: Callback opcional de progresso para download concorrente.
        """
        session_id = str(uuid4())
        logger.info(f"[{SystemState.INGESTING.value}] Iniciando ciclo de ingestão de feeds...")

        if custom_articles is not None:
            articles = custom_articles
        else:
            parser = RSSParser(self.feeds)
            try:
                articles = parser.fetch_all(progress_callback=progress_callback)
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
            if not self.bayesian_filter.is_relevant(full_text, threshold=self.threshold):
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

