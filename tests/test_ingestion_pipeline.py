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

