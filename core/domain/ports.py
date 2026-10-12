from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from core.domain.article import Article, SummarizedArticle, DailySummary, GlobalBulletin
from core.domain.models.intents import NewsIntentType
from core.domain.models.schemas import NewsFilterSchema


class ISemanticMemory(Protocol):
    """Interface de domínio para a Memória Semântica (Vector DB / Hipocampo).

    Isola a regra de negócio da implementação concreta do ChromaDB.
    """

    def store_daily_summary(self, daily: DailySummary) -> None:
        ...

    def store_individual_summary(self, summary: SummarizedArticle) -> None:
        ...

    def store_individual_summaries_batch(self, summaries: List[SummarizedArticle]) -> None:
        ...

    def recall_context(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        ...


class ILLMClient(Protocol):
    """Porta abstrata para clientes de inferência LLM (Ollama HTTP puro)."""

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        require_json: bool = False,
        system: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        ...

    def embeddings(
        self,
        text: str,
        model: str = "nomic-embed-text",
    ) -> List[float]:
        ...


class IArticleStore(Protocol):
    """Porta abstrata para a camada de persistência de notícias (Parquet Data Lake)."""

    def append(self, articles: List[Article]) -> None:
        ...

    def append_summaries(self, summaries: List[SummarizedArticle]) -> None:
        ...

    def append_bulletin(self, bulletin: GlobalBulletin) -> None:
        ...

    def get_all_simhashes(self) -> List[int]:
        ...

    def get_latest_summaries(self, limit: int = 5) -> List[dict]:
        ...

    def get_unsummarized_articles(self) -> List[dict]:
        ...


class IGatekeeper(Protocol):
    """Porta para classificação determinística de intenções do usuário."""

    def classify(self, user_message: str) -> NewsIntentType:
        ...
