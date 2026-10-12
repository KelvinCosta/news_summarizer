"""Reexport para compatibilidade com o padrão CQRS legado.

Aponta diretamente para a camada de Domínio rica em Event Sourcing baseada em Pydantic v2.
"""

from core.domain.models.events import (
    DomainEvent,
    DomainEvent as Event,
    InferenceFailedEvent,
    HumanCorrectionProvidedEvent,
    ImmutableEventStore,
    UserMessageReceived,
    IntentClassified,
    SearchFiltersExtracted,
    ContextRetrieved,
    ResponseGenerated,
    GuardrailTriggered,
    HandoffTriggered,
    ArticleIngested,
    ArticleFiltered,
)

__all__ = [
    "Event",
    "DomainEvent",
    "InferenceFailedEvent",
    "HumanCorrectionProvidedEvent",
    "ImmutableEventStore",
    "UserMessageReceived",
    "IntentClassified",
    "SearchFiltersExtracted",
    "ContextRetrieved",
    "ResponseGenerated",
    "GuardrailTriggered",
    "HandoffTriggered",
    "ArticleIngested",
    "ArticleFiltered",
]
