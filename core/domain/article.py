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
