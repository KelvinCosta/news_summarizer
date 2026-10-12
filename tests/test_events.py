from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from core.domain.models.events import (
    DomainEvent,
    ImmutableEventStore,
    IntentClassified,
    UserMessageReceived,
    SearchFiltersExtracted,
    GuardrailTriggered,
    HandoffTriggered,
    HumanCorrectionProvidedEvent,
    InferenceFailedEvent,
)
from core.domain.models.intents import NewsIntentType
from core.domain.models.schemas import NewsFilterSchema


class TestDomainEvents:
    """Testes unitários para eventos de domínio e Event Sourcing."""

    def test_domain_event_is_frozen(self):
        event = UserMessageReceived(session_id="s1", text="Olá")
        with pytest.raises(ValidationError):
            event.text = "Modificado"  # Imutabilidade estrita via frozen=True

    def test_event_has_uuid_and_utc_timestamp(self):
        event = UserMessageReceived(session_id="s1", text="Teste")
        assert len(event.event_id) > 10
        assert event.timestamp.tzinfo is not None

    def test_immutable_event_store_append_and_query(self):
        store = ImmutableEventStore()
        e1 = UserMessageReceived(session_id="s1", text="Msg 1")
        e2 = IntentClassified(session_id="s1", intent=NewsIntentType.FACTUAL_QUERY)
        e3 = UserMessageReceived(session_id="s2", text="Msg 2")

        store.append(e1)
        store.append(e2)
        store.append(e3)

        assert len(store.get_events()) == 3
        s1_events = store.get_events(session_id="s1")
        assert len(s1_events) == 2
        assert s1_events[0].text == "Msg 1"
        assert s1_events[1].intent == NewsIntentType.FACTUAL_QUERY

    def test_event_store_corrections(self):
        store = ImmutableEventStore()
        corr = HumanCorrectionProvidedEvent(
            session_id="s1",
            failed_event_id="err_1",
            correction_text="Use apenas fatos arquivados",
        )
        store.append(corr)

        unprocessed = store.get_unprocessed_corrections()
        assert len(unprocessed) == 1
        assert unprocessed[0].correction_text == "Use apenas fatos arquivados"

    def test_to_dict_list_serialization(self):
        store = ImmutableEventStore()
        e = UserMessageReceived(session_id="s1", text="Teste serialização")
        store.append(e)

        data = store.to_dict_list("s1")
        assert len(data) == 1
        assert data[0]["text"] == "Teste serialização"
        assert data[0]["session_id"] == "s1"

