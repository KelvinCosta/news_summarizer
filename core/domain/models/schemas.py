from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from core.domain.models.intents import NewsIntentType


class NewsFilterSchema(BaseModel):
    """Esquema fortemente tipado para extração estruturada de parâmetros de busca de notícias.

    Garante validação via Pydantic v2 antes de qualquer consulta ao Parquet ou ChromaDB.
    """

    model_config = ConfigDict(frozen=True)

    topico: Optional[str] = Field(
        default=None,
        description="Assunto ou área temática da notícia (ex: 'inteligencia artificial', 'startups').",
    )
    palavra_chave: Optional[str] = Field(
        default=None,
        description="Termo específico ou entidade mencionada (ex: 'NVIDIA', 'OpenAI', 'Python').",
    )
    fonte: Optional[str] = Field(
        default=None,
        description="Veículo ou origem da notícia (ex: 'CNN', 'BBC').",
    )
    data_referencia: Optional[str] = Field(
        default=None,
        description="Data no formato 'YYYY-MM-DD' caso o usuário especifique um dia.",
    )
    limite: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Quantidade máxima de contextos ou artigos a retornar.",
    )

    def is_empty(self) -> bool:
        """Verifica se nenhum critério de busca semântica/léxica foi preenchido."""
        return (
            self.topico is None
            and self.palavra_chave is None
            and self.fonte is None
            and self.data_referencia is None
        )


class OrchestratorResult(BaseModel):
    """Resultado imutável gerado por um ciclo de orquestração do News Summarizer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    response: str
    intent: NewsIntentType
    events: List[Any] = Field(default_factory=list)
    contexts: List[dict] = Field(default_factory=list)
    handoff_triggered: bool = False
    handoff_reason: Optional[str] = None

