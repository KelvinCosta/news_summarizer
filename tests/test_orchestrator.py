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

