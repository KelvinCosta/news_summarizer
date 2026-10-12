from unittest.mock import MagicMock
import pytest

from core.domain.models.events import (
    ContextRetrieved,
    HandoffTriggered,
    ImmutableEventStore,
    IntentClassified,
    ResponseGenerated,
    SearchFiltersExtracted,
    UserMessageReceived,
)
from core.domain.models.intents import NewsIntentType
from core.domain.models.schemas import NewsFilterSchema, OrchestratorResult
from core.domain.ports import ISemanticMemory
from core.fsm.orchestrator import FSMOrchestrator
from core.fsm.states import SystemState
from core.application.services.gatekeeper import GatekeeperService
from core.application.services.query_extractor import QueryExtractorService
from core.application.services.grounded_oracle import GroundedOracleService
from core.persistence.parquet_store import ParquetStore


class TestFSMOrchestrator:
    """Testes unitários para o FSMOrchestrator e transições de estado."""

    @pytest.fixture
    def mock_gatekeeper(self):
        return MagicMock(spec=GatekeeperService)

    @pytest.fixture
    def mock_extractor(self):
        return MagicMock(spec=QueryExtractorService)

    @pytest.fixture
    def mock_oracle(self):
        return MagicMock(spec=GroundedOracleService)

    @pytest.fixture
    def mock_memory(self):
        return MagicMock(spec=ISemanticMemory)

    def test_handle_factual_query_pipeline(
        self, mock_gatekeeper, mock_extractor, mock_oracle, mock_memory
    ):
        mock_gatekeeper.classify.return_value = NewsIntentType.FACTUAL_QUERY
        mock_extractor.extract.return_value = NewsFilterSchema(topico="IA", limite=3)
        mock_oracle.answer.return_value = (
            "Notícia confirmada com 100% de groundedness.",
            [{"content": "Trecho da notícia"}],
            False,
        )
        mock_oracle.model = "llama3.2"

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(
            semantic_memory=mock_memory,
            event_store=event_store,
            gatekeeper=mock_gatekeeper,
            query_extractor=mock_extractor,
            oracle_service=mock_oracle,
        )

        result = orchestrator.handle_message("Quais são as notícias de IA?")

        assert result.response == "Notícia confirmada com 100% de groundedness."
        assert result.intent == NewsIntentType.FACTUAL_QUERY
        assert not result.handoff_triggered
        assert orchestrator.current_state == SystemState.IDLE

        # Verificação do Event Sourcing
        events = event_store.get_events()
        event_types = [type(e) for e in events]
        assert UserMessageReceived in event_types
        assert IntentClassified in event_types
        assert SearchFiltersExtracted in event_types
        assert ContextRetrieved in event_types
        assert ResponseGenerated in event_types

    def test_handle_out_of_domain(
        self, mock_gatekeeper, mock_extractor, mock_oracle, mock_memory
    ):
        mock_gatekeeper.classify.return_value = NewsIntentType.OUT_OF_DOMAIN

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(
            semantic_memory=mock_memory,
            event_store=event_store,
            gatekeeper=mock_gatekeeper,
            query_extractor=mock_extractor,
            oracle_service=mock_oracle,
        )

        result = orchestrator.handle_message("Como fazer torta de morango?")

        assert result.intent == NewsIntentType.OUT_OF_DOMAIN
        assert result.handoff_triggered is True
        assert result.handoff_reason == "out_of_domain"
        assert result.response == FSMOrchestrator.OUT_OF_DOMAIN_MESSAGE

        # Verifica que o HandoffTriggered foi registrado
        events = event_store.get_events()
        assert any(isinstance(e, HandoffTriggered) for e in events)

    def test_handle_human_intervention(
        self, mock_gatekeeper, mock_extractor, mock_oracle, mock_memory
    ):
        mock_gatekeeper.classify.return_value = NewsIntentType.HUMAN_INTERVENTION

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(
            semantic_memory=mock_memory,
            event_store=event_store,
            gatekeeper=mock_gatekeeper,
            query_extractor=mock_extractor,
            oracle_service=mock_oracle,
        )

        result = orchestrator.handle_message("Quero falar com um humano")

        assert result.intent == NewsIntentType.HUMAN_INTERVENTION
        assert result.handoff_triggered is True
        assert result.handoff_reason == "user_requested"
        assert result.response == FSMOrchestrator.HUMAN_ESCALATION_MESSAGE

    def test_fail_fast_and_human_correction(self):
        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(event_store=event_store)

        # Dispara fail-fast: o estado transita para HUMAN_INTERVENTION
        orchestrator.trigger_fail_fast(reason="Alucinação detectada")
        assert orchestrator.current_state == SystemState.HUMAN_INTERVENTION

        # Aplicando a correção humana
        orchestrator.handle_human_correction(
            correction_data="Apenas notícias reais da base",
            session_id="test_sess",
        )
        assert orchestrator.current_state == SystemState.IDLE

        corrections = event_store.get_unprocessed_corrections()
        assert len(corrections) == 1
        assert corrections[0].correction_text == "Apenas notícias reais da base"

    def test_handle_human_correction_outside_state_raises_error(self):
        orchestrator = FSMOrchestrator()
        orchestrator.current_state = SystemState.IDLE

        with pytest.raises(ValueError):
            orchestrator.handle_human_correction("correção indevida")

    def test_handle_temporal_system_query_fast_path(self):
        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(event_store=event_store)

        result = orchestrator.handle_message("que dia é hoje?")

        assert result.intent == NewsIntentType.FACTUAL_QUERY
        assert "Hoje é" in result.response
        assert not result.handoff_triggered
        assert result.handoff_reason is None
        assert len(result.contexts) == 1
        assert result.contexts[0]["metadata"]["type"] == "relogio_do_sistema"
        assert orchestrator.current_state == SystemState.IDLE

        # Verifica Event Sourcing
        events = event_store.get_events()
        event_types = [type(e) for e in events]
        assert ContextRetrieved in event_types
        assert ResponseGenerated in event_types

    def test_handle_catalog_recency_and_antiquity_queries(self):
        mock_parquet = MagicMock(spec=ParquetStore)
        mock_parquet.get_latest_summaries.return_value = [
            {
                "title": "Notícia Super Recente",
                "summary": "Resumo do que acabou de acontecer.",
                "original_url": "https://tech.com/new",
                "processed_at": "2026-10-12T13:00:00",
            }
        ]
        mock_parquet.get_oldest_summaries.return_value = [
            {
                "title": "Notícia Bem Antiga",
                "summary": "Resumo do primeiro artigo do mês.",
                "original_url": "https://tech.com/old",
                "processed_at": "2026-10-01T10:00:00",
            }
        ]

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(parquet_store=mock_parquet, event_store=event_store)

        # 1. Recente
        res_rec = orchestrator.handle_message("qual sua noticia mais recente?")
        assert res_rec.intent == NewsIntentType.FACTUAL_QUERY
        assert not res_rec.handoff_triggered
        assert "Notícia Super Recente" in res_rec.response
        assert res_rec.contexts[0]["metadata"]["source_url"] == "https://tech.com/new"

        # 2. Antiga
        res_ant = orchestrator.handle_message("qual sua noticia mais antiga?")
        assert res_ant.intent == NewsIntentType.FACTUAL_QUERY
        assert not res_ant.handoff_triggered
        assert "Notícia Bem Antiga" in res_ant.response
        assert res_ant.contexts[0]["metadata"]["source_url"] == "https://tech.com/old"

        # 3. Mais atual
        res_atu = orchestrator.handle_message("qual sua noticia mais atual?")
        assert res_atu.intent == NewsIntentType.FACTUAL_QUERY
        assert not res_atu.handoff_triggered
        assert "Notícia Super Recente" in res_atu.response

