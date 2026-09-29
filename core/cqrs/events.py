from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Event:
    """Evento base para o Event Sourcing."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class InferenceFailedEvent(Event):
    """Registrado quando o Guardrail aciona o Fail-Fast."""
    session_id: str = ""
    original_prompt: str = ""
    error_reason: str = "" # Ex: Alucinação, Violação de Regra

@dataclass
class HumanCorrectionProvidedEvent(Event):
    """Registra as correções brutas do usuário (Append-Only)."""
    session_id: str = ""
    failed_event_id: str = ""
    correction_text: str = ""

class ImmutableEventStore:
    """
    Log Imutável no padrão Event Sourcing.
    Opera estritamente em append-only.
    """
    def __init__(self):
        self._log = []

    def append(self, event: Event):
        self._log.append(event)
        
    def get_unprocessed_corrections(self):
        """Retorna eventos para o 'Ciclo de Sono' processar."""
        return [e for e in self._log if isinstance(e, HumanCorrectionProvidedEvent)]
