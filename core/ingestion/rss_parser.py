"""Adaptador de Ingestão de Feeds RSS Concorrente.

Utiliza ThreadPoolExecutor com HTTPX assíncrono/concorrente para processar
dezenas ou centenas de feeds em paralelo com timeouts estritos (fail-fast),
evitando gargalos de I/O de rede e travamentos indefinidos.
"""

import concurrent.futures
from datetime import datetime, timezone
import logging
from time import mktime
from typing import Callable, List, Optional

import feedparser
import httpx

from core.domain.article import Article

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


class RSSParser:
    """Adaptador de Ingestão: Faz fetch paralelo de feeds RSS com fail-fast timeout."""

    def __init__(
        self,
        feeds: List[str],
        max_workers: int = 16,
        timeout_seconds: float = 7.0,
    ) -> None:
        # Higieniza URLs removendo espaços em branco acidentais
        self.feeds = [f.strip() for f in feeds if f and f.strip()]
        self.max_workers = max(1, min(max_workers, 32))
        self.timeout_seconds = timeout_seconds

    def _fetch_single_feed(self, client: httpx.Client, feed_url: str) -> List[Article]:
        try:
            response = client.get(
                feed_url,
                timeout=self.timeout_seconds,
                headers=DEFAULT_HEADERS,
                follow_redirects=True,
            )
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
        except Exception as e:
            logger.warning(f"Falha ao capturar feed [{feed_url}]: {e}")
            return []

        articles: List[Article] = []
        for entry in parsed.entries:
            dt = datetime.now(timezone.utc)
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    dt = datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)

            content = ""
            if hasattr(entry, "content"):
                content = " ".join([c.value for c in entry.content])
            elif hasattr(entry, "summary"):
                content = entry.summary

            url = getattr(entry, "link", "")
            title = getattr(entry, "title", "")
            if not url or not title:
                continue

            articles.append(
                Article(
                    url=url,
                    title=title,
                    content=content,
                    published_at=dt,
                    source=feed_url,
                )
            )
        return articles

    def fetch_all(
        self,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[Article]:
        """Captura todos os feeds em paralelo usando ThreadPoolExecutor.

        Args:
            progress_callback: Função opcional chamada a cada feed finalizado
                               (concluidos, total_feeds, total_artigos_acumulados).
        """
        if not self.feeds:
            return []

        all_articles: List[Article] = []
        total_feeds = len(self.feeds)
        completed = 0

        # Pool de conexões HTTP reutilizável com limites adequados
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
        with httpx.Client(limits=limits, timeout=self.timeout_seconds) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_url = {
                    executor.submit(self._fetch_single_feed, client, url): url
                    for url in self.feeds
                }

                for future in concurrent.futures.as_completed(future_to_url):
                    completed += 1
                    try:
                        feed_articles = future.result()
                        all_articles.extend(feed_articles)
                    except Exception as e:
                        feed_url = future_to_url.get(future, "unknown")
                        logger.warning(f"Exceção no processamento de {feed_url}: {e}")

                    if progress_callback:
                        try:
                            progress_callback(completed, total_feeds, len(all_articles))
                        except Exception:
                            pass

        return all_articles
