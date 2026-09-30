from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class Article:
    """Entidade imutável que representa uma notícia bruta."""
    url: str
    title: str
    content: str
    published_at: datetime
    source: str
    simhash_value: Optional[int] = None

@dataclass(frozen=True)
class SummarizedArticle:
    """Entidade que representa uma notícia processada individualmente pelo Agente SLM."""
    original_url: str
    title: str
    summary: str
    processed_at: datetime

@dataclass(frozen=True)
class GlobalBulletin:
    """Entidade que representa o boletim consolidado de uma sessão (Agente Master)."""
    content: str
    processed_at: datetime

@dataclass(frozen=True)
class DailySummary:
    """Entidade que representa o sumário definitivo de um dia (Mega-Boletim)."""
    target_date: str # Formato 'YYYY-MM-DD'
    content: str
    processed_at: datetime
