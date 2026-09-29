from dataclasses import dataclass
from typing import List

@dataclass
class ConsolidatedRule:
    """Regra consolidada ('Regras Z') gerada pelo Ciclo de Sono."""
    rule_id: str
    content: str
    tags: List[str] # Isolamento Multitenant (Asymmetric Tagging)

class VectorReadModel:
    """
    Representação do Banco Vetorial (CQRS Read Model).
    Este modelo é altamente mutável e armazenará as heurísticas consolidadas.
    A integração direta com ChromaDB será feita na etapa 2.
    """
    def __init__(self):
        self.consolidated_rules: List[ConsolidatedRule] = []

    def upsert_rule(self, rule: ConsolidatedRule):
        """Realiza o upsert com coalescência da regra no banco vetorial."""
        # Lógica de coalescência futura para substituir regras antigas
        self.consolidated_rules.append(rule)

    def search_context(self, query: str, active_tags: List[str]) -> List[ConsolidatedRule]:
        """
        Busca semântica com bloqueio prévio baseado em tags para evitar Data Bleeding.
        """
        # Implementação futura com ChromaDB + Tag Filtering
        pass
