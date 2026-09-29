import logging
from core.fsm.orchestrator import FSMOrchestrator
from core.cqrs.events import ImmutableEventStore, InferenceFailedEvent, HumanCorrectionProvidedEvent
from core.cqrs.read_models import VectorReadModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    print("Inicializando News Summarizer - Arquitetura On-Premise")
    
    # Inicialização da Infraestrutura Base
    event_store = ImmutableEventStore()
    vector_read_model = VectorReadModel()
    orchestrator = FSMOrchestrator()
    
    print("\n--- Simulando Fluxo Feliz ---")
    orchestrator.process_request("Resuma as notícias de hoje sobre IA.")
    
    print("\n--- Simulando Falha (Guardrail acionado) ---")
    orchestrator.process_request("Resumo de notícias.")
    # Simulando disparo manual do fail-fast durante inferência
    orchestrator.trigger_fail_fast()
    
    # Registrando no Immutable Log
    failed_event = InferenceFailedEvent(session_id="123", original_prompt="Resumo", error_reason="Alucinação detectada")
    event_store.append(failed_event)
    
    print("\n--- Simulando Intervenção Humana (HITL) ---")
    correction_event = HumanCorrectionProvidedEvent(
        session_id="123", 
        failed_event_id=failed_event.event_id,
        correction_text="Não invente notícias, use apenas o contexto."
    )
    event_store.append(correction_event)
    orchestrator.handle_human_correction(correction_event.correction_text)
    
    print("\n--- Log Imutável (Event Sourcing) ---")
    for event in event_store._log:
        print(f"[{event.timestamp}] {type(event).__name__}")

if __name__ == "__main__":
    main()
