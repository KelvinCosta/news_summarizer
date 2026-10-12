from unittest.mock import MagicMock
import pytest

from core.domain.models.events import HumanCorrectionProvidedEvent, ImmutableEventStore
from core.domain.models.intents import NewsIntentType
from core.fsm.orchestrator import FSMOrchestrator
from core.fsm.states import SystemState
from core.application.services.gatekeeper import GatekeeperService
from core.application.services.grounded_oracle import GroundedOracleService


class TestSimulationDemo:
    """Testes unitários automatizados para os cenários demonstrados pelo sistema."""

    def test_happy_path_factual_query(self):
        mock_gatekeeper = MagicMock(spec=GatekeeperService)
        mock_gatekeeper.classify.return_value = NewsIntentType.FACTUAL_QUERY

        mock_oracle = MagicMock(spec=GroundedOracleService)
        mock_oracle.answer.return_value = (
            "Notícia confirmada: avanços na tecnologia de chips de IA.",
            [{"content": "Trecho verificado"}],
            False,
        )
        mock_oracle.model = "llama3.2"

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(
            event_store=event_store,
            gatekeeper=mock_gatekeeper,
            oracle_service=mock_oracle,
        )

        res = orchestrator.handle_message("Quais são as notícias sobre inteligência artificial?")
        assert res.intent == NewsIntentType.FACTUAL_QUERY
        assert "Notícia confirmada" in res.response
        assert not res.handoff_triggered

    def test_guardrail_out_of_domain(self):
        mock_gatekeeper = MagicMock(spec=GatekeeperService)
        mock_gatekeeper.classify.return_value = NewsIntentType.OUT_OF_DOMAIN

        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(
            event_store=event_store,
            gatekeeper=mock_gatekeeper,
        )

        res = orchestrator.handle_message("Como preparar uma receita de bolo de cenoura?")
        assert res.intent == NewsIntentType.OUT_OF_DOMAIN
        assert res.handoff_triggered is True
        assert res.response == FSMOrchestrator.OUT_OF_DOMAIN_MESSAGE

    def test_fail_fast_circuit_breaker(self):
        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(event_store=event_store)

        orchestrator.trigger_fail_fast(reason="Alucinação detectada por teste heurístico")
        assert orchestrator.current_state == SystemState.HUMAN_INTERVENTION

        events = event_store.get_events()
        assert any(e.session_id == "system_fail_fast" for e in events)

    def test_human_in_the_loop_resolution(self):
        event_store = ImmutableEventStore()
        orchestrator = FSMOrchestrator(event_store=event_store)

        orchestrator.trigger_fail_fast()
        assert orchestrator.current_state == SystemState.HUMAN_INTERVENTION

        orchestrator.handle_human_correction(
            correction_data="Sintetize apenas fatos estritamente contidos nas notícias arquivadas.",
            session_id="sessao_demo",
        )

        assert orchestrator.current_state == SystemState.IDLE
        corrections = event_store.get_unprocessed_corrections()
        assert len(corrections) == 1
        assert corrections[0].correction_text == "Sintetize apenas fatos estritamente contidos nas notícias arquivadas."
        assert corrections[0].session_id == "sessao_demo"
