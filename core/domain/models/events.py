"""Modelos de Event Sourcing baseados em Pydantic v2 (Clean Architecture).

Todos os eventos de domínio são imutáveis (frozen=True), possuem carimbo temporal UTC,
identificadores únicos e session_id para garantir auditoria, rastreabilidade e
reconstituição determinística de estado sem efeitos colaterais.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field

from core.domain.models.intents import NewsIntentType
from core.domain.models.schemas import NewsFilterSchema


class DomainEvent(BaseModel):
    """Classe base imutável para eventos de domínio de sessão (Event Sourcing)."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Identificador universal único do evento.",
    )
    session_id: str = Field(
        description="Identificador da sessão ou ciclo de execução à qual o evento pertence.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Carimbo temporal em UTC da ocorrência do evento.",
    )


class UserMessageReceived(DomainEvent):
    """Disparado quando uma requisição ou mensagem do usuário é recepcionada."""

    text: str = Field(description="Conteúdo textual bruto enviado pelo usuário.")


class IntentClassified(DomainEvent):
    """Disparado pelo Gatekeeper (Estágio 1) após classificação da intenção."""

    intent: NewsIntentType = Field(description="Tipo de intenção classificada pelo modelo/guardrail.")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Grau opcional de confiança da inferência.",
    )


class SearchFiltersExtracted(DomainEvent):
    """Disparado pelo Extrator de Parâmetros (Estágio 2)."""

    filters: NewsFilterSchema = Field(
        description="Filtros estruturados extraídos a partir da pergunta do usuário.",
    )


class ContextRetrieved(DomainEvent):
    """Disparado ao consultar a Memória Semântica (ChromaDB) ou Lake de Notícias (Parquet)."""

    result_count: int = Field(ge=0, description="Quantidade de trechos ou notícias retornadas.")
    strategy: str = Field(description="Estratégia empregada (ex: 'chroma_similarity', 'polars_exact').")


class ResponseGenerated(DomainEvent):
    """Disparado quando a resposta final para o usuário é sintetizada."""

    response: str = Field(description="Texto final gerado e entregue ao usuário.")
    model: str = Field(description="Nome do modelo ou mecanismo que gerou a resposta.")


class GuardrailTriggered(DomainEvent):
    """Disparado quando um guardrail em código ou heurística detecta desvio ou inconsistência."""

    guardrail_name: str = Field(description="Nome ou tipo do guardrail acionado.")
    reason: str = Field(description="Justificativa do acionamento.")


class HandoffTriggered(DomainEvent):
    """Disparado quando ocorre transbordo seguro determinístico ou acionamento de fail-fast."""

    reason: str = Field(description="Motivo do transbordo (ex: 'out_of_domain', 'no_data', 'user_requested').")
    details: Optional[str] = Field(default=None, description="Metadados para auditoria ou operador.")


class InferenceFailedEvent(DomainEvent):
    """Compatibilidade com modelo legado: Registrado quando o Guardrail aciona Fail-Fast."""

    original_prompt: str = ""
    error_reason: str = ""


class HumanCorrectionProvidedEvent(DomainEvent):
    """Compatibilidade com modelo legado: Registra correções humanas manuais (HITL)."""

    failed_event_id: str = ""
    correction_text: str = ""


class ArticleIngested(DomainEvent):
    """Disparado quando um artigo é coletado via feed RSS."""

    url: str
    title: str
    source: str


class ArticleFiltered(DomainEvent):
    """Disparado no funil O(1) com a decisão de filtro lexical ou Bayesiano."""

    url: str
    title: str
    status: str  # 'accepted', 'discarded_lexical_duplicate', 'discarded_irrelevant'
    score: Optional[float] = None


class ImmutableEventStore:
    """Log Imutável no padrão Event Sourcing (Append-Only).

    Garante integridade e rastreabilidade de todas as transições de estado do sistema.
    """

    def __init__(self) -> None:
        self._log: List[DomainEvent] = []

    def append(self, event: DomainEvent) -> None:
        """Adiciona um evento de domínio ao log de forma estritamente imutável."""
        self._log.append(event)

    def get_events(self, session_id: Optional[str] = None) -> List[DomainEvent]:
        """Retorna os eventos registrados, opcionalmente filtrados por session_id."""
        if session_id is None:
            return list(self._log)
        return [e for e in self._log if e.session_id == session_id]

    def get_unprocessed_corrections(self) -> List[HumanCorrectionProvidedEvent]:
        """Retorna eventos de correção humana pendentes de consolidação."""
        return [e for e in self._log if isinstance(e, HumanCorrectionProvidedEvent)]

    def to_dict_list(self, session_id: Optional[str] = None) -> List[dict[str, Any]]:
        """Serializa os eventos para formato de dicionário."""
        events = self.get_events(session_id)
        return [e.model_dump() for e in events]

