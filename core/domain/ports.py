from typing import Protocol, List, Dict, Any
from core.domain.article import SummarizedArticle, DailySummary

class ISemanticMemory(Protocol):
    """
    Interface de domínio (Clean Architecture) para a Memória Semântica.
    Isola a regra de negócios da implementação tecnológica do banco vetorial.
    """
    def store_daily_summary(self, daily: DailySummary) -> None:
        ...
        
    def store_individual_summary(self, summary: SummarizedArticle) -> None:
        ...
        
    def recall_context(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        ...
