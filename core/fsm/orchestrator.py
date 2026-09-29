import logging
from typing import Any, Dict
from core.fsm.states import SystemState

logger = logging.getLogger(__name__)

class FSMOrchestrator:
    """
    Orquestrador principal governado por uma Máquina de Estados Finita.
    Garante que o fluxo seja imutável e sem loops infinitos entre agentes.
    """
    def __init__(self):
        self.current_state = SystemState.IDLE
        self.context: Dict[str, Any] = {}

    def transition_to(self, new_state: SystemState):
        logger.info(f"Transição de estado: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state

    def process_request(self, user_input: str):
        """Fluxo principal de processamento de uma requisição."""
        self.transition_to(SystemState.ROUTING)
        # TODO: Implementar Lobo Frontal (Heurística/RegEx para seleção de SLM)
        
        self.transition_to(SystemState.RETRIEVING_CONTEXT)
        # TODO: Implementar Hipocampo (Busca no Read Model / ChromaDB)
        
        self.transition_to(SystemState.INFERENCING)
        # TODO: Implementar chamada sequencial do SLM via Ollama
        
        self.transition_to(SystemState.VALIDATING)
        # TODO: Implementar Guardrail
        # Exemplo de lógica Fail-Fast:
        # if guardrail_detects_hallucination():
        #     self.transition_to(SystemState.HUMAN_INTERVENTION)
        #     return
        
        self.transition_to(SystemState.COMPLETED)
        self.transition_to(SystemState.IDLE)

    def trigger_fail_fast(self):
        """Abre o circuito instantaneamente em caso de pensamento intrusivo."""
        logger.error("Guardrail acionado! Interrompendo execução (Zero retentativas).")
        self.transition_to(SystemState.HUMAN_INTERVENTION)

    def handle_human_correction(self, correction_data: str):
        """Retoma o controle após a intervenção manual do usuário."""
        if self.current_state != SystemState.HUMAN_INTERVENTION:
            raise ValueError("Correções só podem ser aplicadas no estado HUMAN_INTERVENTION.")
        logger.info("Processando correção humana e registrando no Event Store...")
        # O registro real ocorrerá na camada de Event Sourcing
        self.transition_to(SystemState.IDLE)
