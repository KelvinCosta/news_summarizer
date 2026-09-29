from enum import Enum

class SystemState(Enum):
    IDLE = "IDLE"
    ROUTING = "ROUTING" # Lobo Frontal: roteamento de intenção
    RETRIEVING_CONTEXT = "RETRIEVING_CONTEXT" # Hipocampo: busca no banco vetorial
    INFERENCING = "INFERENCING" # Execução do SLM (Ollama)
    VALIDATING = "VALIDATING" # Guardrail: verificação de alucinações
    HUMAN_INTERVENTION = "HUMAN_INTERVENTION" # HITL / Fail-Fast
    CONSOLIDATING = "CONSOLIDATING" # Ciclo de Sono: processamento background
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
