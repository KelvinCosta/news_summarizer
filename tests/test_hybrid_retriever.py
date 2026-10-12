"""Testes unitários para o HybridRetrieverService (Estágio 3 - Busca Híbrida)."""

from unittest.mock import MagicMock
import pytest

from core.application.services.hybrid_retriever import HybridRetrieverService
from core.domain.models.schemas import NewsFilterSchema
from core.domain.ports import ISemanticMemory
from core.persistence.parquet_store import ParquetStore
from core.fsm.orchestrator import FSMOrchestrator
from core.domain.models.intents import NewsIntentType
from core.application.services.gatekeeper import GatekeeperService
from core.application.services.query_extractor import QueryExtractorService
from core.application.services.grounded_oracle import GroundedOracleService
from core.domain.models.events import ImmutableEventStore, ContextRetrieved


class TestHybridRetrieverService:
    """Testes do recuperador híbrido com fusão léxica e vetorial."""

    @pytest.fixture
    def mock_parquet(self):
        store = MagicMock(spec=ParquetStore)
        store.search_summaries_by_keyword.return_value = []
        store.search_bulletins_by_keyword.return_value = []
        return store

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock(spec=ISemanticMemory)
        memory.recall_context.return_value = []
        return memory

    def test_lexical_exact_match_prioritized_over_vector(self, mock_parquet, mock_memory):
        # O Parquet encontra o artigo exato da Genesis Mission
        mock_parquet.search_summaries_by_keyword.return_value = [
            {
                "title": "Exclusive: Inside Trump's AI science summit",
                "summary": "Compromissos para o Genesis Mission com AMD e OpenAI.",
                "original_url": "https://axios.com/genesis",
                "processed_at": "2026-10-12T10:00:00",
            }
        ]

        # A memória vetorial retorna outro artigo sem relação direta
        mock_memory.recall_context.return_value = [
            {
                "content": "Guia de missões do jogo Arc Raiders.",
                "metadata": {"title": "Arc Raiders", "source_url": "https://gamespot.com/arc"},
            }
        ]

        retriever = HybridRetrieverService(semantic_memory=mock_memory, parquet_store=mock_parquet)
        filters = NewsFilterSchema(palavra_chave="Genesis Mission", limite=5)

        contexts = retriever.retrieve(query="me fale sobre a Genesis Mission", filters=filters, limit=5)

        assert len(contexts) == 2
        # O primeiro documento DEVE ser o casamento léxico exato
        assert contexts[0]["metadata"]["title"] == "Exclusive: Inside Trump's AI science summit"
        assert contexts[0]["metadata"]["match_strategy"] == "lexical_exact"
        # O segundo documento é o resultado vetorial
        assert contexts[1]["metadata"]["title"] == "Arc Raiders"

    def test_vector_fallback_when_no_lexical_match(self, mock_parquet, mock_memory):
        mock_parquet.search_summaries_by_keyword.return_value = []
        mock_parquet.search_bulletins_by_keyword.return_value = []

        mock_memory.recall_context.return_value = [
            {
                "content": "Novidades em semicondutores e GPUs.",
                "metadata": {"title": "Hardware News", "source_url": "https://tech.com/gpu"},
            }
        ]

        retriever = HybridRetrieverService(semantic_memory=mock_memory, parquet_store=mock_parquet)
        filters = NewsFilterSchema(topico="semicondutores", limite=3)

        contexts = retriever.retrieve(query="quais as novidades em hardware?", filters=filters, limit=3)

        assert len(contexts) == 1
        assert contexts[0]["metadata"]["title"] == "Hardware News"

    def test_deduplication_of_identical_sources(self, mock_parquet, mock_memory):
        same_url = "https://tech.com/same-article"
        mock_parquet.search_summaries_by_keyword.return_value = [
            {
                "title": "Mesmo Artigo",
                "summary": "Conteúdo vindo do Parquet.",
                "original_url": same_url,
                "processed_at": "2026-10-12T10:00:00",
            }
        ]
        mock_memory.recall_context.return_value = [
            {
                "content": "Conteúdo vindo do ChromaDB.",
                "metadata": {"title": "Mesmo Artigo", "source_url": same_url},
            }
        ]

        retriever = HybridRetrieverService(semantic_memory=mock_memory, parquet_store=mock_parquet)
        filters = NewsFilterSchema(palavra_chave="Mesmo", limite=5)

        contexts = retriever.retrieve(query="Mesmo", filters=filters, limit=5)

        # Não deve haver duplicata: o match léxico prioritário vence
        assert len(contexts) == 1
        assert contexts[0]["metadata"]["source_url"] == same_url
        assert contexts[0]["metadata"]["match_strategy"] == "lexical_exact"

    def test_empty_sources_graceful_handling(self):
        retriever = HybridRetrieverService(semantic_memory=None, parquet_store=None)
        contexts = retriever.retrieve(query="qualquer coisa")
        assert contexts == []

    def test_orchestrator_integration_with_hybrid_retriever(self, mock_parquet, mock_memory):
        mock_parquet.search_summaries_by_keyword.return_value = [
            {
                "title": "Genesis Mission News",
                "summary": "A White House e AMD apoiam a Genesis Mission.",
                "original_url": "https://axios.com/genesis",
                "processed_at": "2026-10-12T10:00:00",
            }
        ]

        mock_gatekeeper = MagicMock(spec=GatekeeperService)
        mock_gatekeeper.classify.return_value = NewsIntentType.FACTUAL_QUERY

        mock_extractor = MagicMock(spec=QueryExtractorService)
        mock_extractor.extract.return_value = NewsFilterSchema(palavra_chave="Genesis Mission", limite=3)

        mock_oracle = MagicMock(spec=GroundedOracleService)
        mock_oracle.answer.return_value = (
            "A Genesis Mission é uma parceria com AMD e Casa Branca.",
            [{"content": "A White House e AMD apoiam a Genesis Mission."}],
            False,
        )
        mock_oracle.model = "llama3.2"

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(
            semantic_memory=mock_memory,
            parquet_store=mock_parquet,
            event_store=event_store,
            gatekeeper=mock_gatekeeper,
            query_extractor=mock_extractor,
            oracle_service=mock_oracle,
        )

        result = orchestrator.handle_message("me fale sobre a Genesis Mission")

        assert result.intent == NewsIntentType.FACTUAL_QUERY
        assert "Genesis Mission" in result.response
        assert not result.handoff_triggered

        # Verifica evento ContextRetrieved emitido com estratégia hybrid_rag
        events = event_store.get_events()
        retrieval_events = [e for e in events if isinstance(e, ContextRetrieved)]
        assert len(retrieval_events) == 1
        assert retrieval_events[0].strategy == "hybrid_rag"
        assert retrieval_events[0].result_count >= 1

    def test_bilingual_expansion_casa_branca_searches_white_house(self, mock_parquet, mock_memory):
        """Verifica se uma busca por 'casa branca' aciona a busca léxica de 'white house'."""
        mock_parquet.search_summaries_by_keyword.side_effect = lambda kw, limit: [
            {"title": "White House AI summit", "summary": "White House AI summit details", "original_url": "https://wh.gov"}
        ] if kw.lower() == "white house" else []

        retriever = HybridRetrieverService(semantic_memory=mock_memory, parquet_store=mock_parquet)
        filters = NewsFilterSchema(palavra_chave="casa branca", limite=5)

        contexts = retriever.retrieve(query="me fale sobre a casa branca", filters=filters, limit=5)

        assert len(contexts) >= 1
        assert contexts[0]["metadata"]["title"] == "White House AI summit"
        # Garante que search_summaries_by_keyword foi chamado tanto para 'casa branca' quanto 'white house'
        called_keywords = [call.args[0].lower() for call in mock_parquet.search_summaries_by_keyword.call_args_list]
        assert "casa branca" in called_keywords
        assert "white house" in called_keywords

    def test_bilingual_expansion_query_direct_entity_detection(self, mock_parquet, mock_memory):
        """Verifica se entidades bilíngues são detectadas diretamente na query mesmo sem palavra_chave extraída."""
        mock_parquet.search_summaries_by_keyword.side_effect = lambda kw, limit: [
            {"title": "White House AI summit", "summary": "White House AI summit details", "original_url": "https://wh.gov"}
        ] if kw.lower() == "white house" else []

        retriever = HybridRetrieverService(semantic_memory=mock_memory, parquet_store=mock_parquet)

        contexts = retriever.retrieve(query="quais as noticias da casa branca?", filters=None, limit=5)

        assert len(contexts) >= 1
        called_keywords = [call.args[0].lower() for call in mock_parquet.search_summaries_by_keyword.call_args_list]
        assert "casa branca" in called_keywords
        assert "white house" in called_keywords


