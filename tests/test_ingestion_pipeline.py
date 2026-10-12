from datetime import datetime, timezone
import pytest

from core.application.use_cases.ingestion_service import IngestionService
from core.domain.article import Article
from core.domain.models.events import ArticleFiltered, ArticleIngested, ImmutableEventStore
from core.filters.bayesian import BayesianFilter
from core.filters.lexical import LexicalFilter
from core.persistence.parquet_store import ParquetStore


class TestIngestionPipeline:
    """Testes unitários para o caso de uso de ingestão e funil O(1)."""

    @pytest.fixture
    def tmp_env(self, tmp_path):
        parquet_dir = tmp_path / "parquet"
        vocab_path = tmp_path / "vocab.json"
        vocab_path.write_text(
            '{"positive": {"ai": 2.0, "python": 2.0, "tecnologia": 1.5}, "negative": {"fofoca": -2.0}}'
        )

        store = ParquetStore(str(parquet_dir))
        lex_filter = LexicalFilter(tolerance=3)
        bay_filter = BayesianFilter(str(vocab_path))
        event_store = ImmutableEventStore()

        service = IngestionService(
            parquet_store=store,
            lexical_filter=lex_filter,
            bayesian_filter=bay_filter,
            event_store=event_store,
        )
        return service, store, event_store

    def test_ingestion_funnel_accepts_valid_and_filters_duplicates(self, tmp_env):
        service, store, event_store = tmp_env

        articles = [
            Article(
                url="https://news.com/ai-1",
                title="Novo modelo de AI lançado",
                content="Engenharia de tecnologia e inteligência artificial revolucionária.",
                published_at=datetime.now(timezone.utc),
                source="TechNews",
            ),
            Article(
                url="https://news.com/ai-duplicate",
                title="Novo modelo de AI lançado!",
                content="Engenharia de tecnologia e inteligência artificial revolucionária!",
                published_at=datetime.now(timezone.utc),
                source="MirrorNews",
            ),
            Article(
                url="https://news.com/gossip",
                title="Fofoca do dia",
                content="Muita fofoca sobre celebridades sem tecnologia.",
                published_at=datetime.now(timezone.utc),
                source="GossipNews",
            ),
        ]

        result = service.run(custom_articles=articles)

        assert result["total_raw"] == 3
        assert result["accepted"] == 1
        assert result["discarded_duplicates"] == 1
        assert result["discarded_irrelevant"] == 1

        # Verifica persistência no Parquet
        saved = store.get_unsummarized_articles()
        assert len(saved) == 1
        assert saved[0]["title"] == "Novo modelo de AI lançado"

        # Verifica Event Sourcing
        events = event_store.get_events()
        assert any(isinstance(e, ArticleIngested) for e in events)
        assert any(isinstance(e, ArticleFiltered) and e.status == "accepted" for e in events)
        assert any(isinstance(e, ArticleFiltered) and e.status == "discarded_lexical_duplicate" for e in events)

    def test_rss_parser_concurrent_fetch_with_mock(self, monkeypatch):
        from unittest.mock import MagicMock
        from core.ingestion.rss_parser import RSSParser

        rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Mock Tech Feed</title>
                <item>
                    <title>AI Revolution 2026</title>
                    <link>https://tech.com/ai-2026</link>
                    <description>Summary of AI news</description>
                </item>
            </channel>
        </rss>"""

        mock_resp = MagicMock()
        mock_resp.content = rss_xml
        mock_resp.raise_for_status = MagicMock()

        # Mock httpx.Client.get
        monkeypatch.setattr("httpx.Client.get", lambda self, url, **kwargs: mock_resp)

        progress_calls = []
        def track_progress(done, total, count):
            progress_calls.append((done, total, count))

        parser = RSSParser(
            feeds=["https://feed1.com/rss", "https://feed2.com/rss "],
            max_workers=2,
            timeout_seconds=2.0,
        )
        articles = parser.fetch_all(progress_callback=track_progress)

        assert len(articles) == 2
        assert articles[0].title == "AI Revolution 2026"
        assert len(progress_calls) == 2
        assert progress_calls[-1][0] == 2
        assert progress_calls[-1][1] == 2


