from dataclasses import dataclass, field
from typing import Set

@dataclass
class SessionContext:
    """
    Gerencia o contexto de uma sessão, incluindo as tags de isolamento.
    """
    session_id: str
    active_tags: Set[str] = field(default_factory=set)
    
    def add_tag(self, tag: str):
        self.active_tags.add(tag)
        
    def can_access(self, target_tag: str) -> bool:
        """
        Verifica a herança direcional para prevenir contaminação cruzada.
        """
        # Exemplo simples: Se a sessão tem a tag, ela pode acessar.
        # Uma lógica de árvore de herança pode ser implementada aqui.
        return target_tag in self.active_tags
