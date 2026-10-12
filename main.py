"""Ponto de entrada do News Summarizer (Clean Architecture & SLM On-Premise).

Executa o fluxo do sistema via FSMOrchestrator com roteamento em estágios desacoplados,
Event Sourcing auditável e guardrails contra alucinações.
"""

import logging
import sys
from typing import Optional

from core.fsm.orchestrator import FSMOrchestrator
from core.domain.models.events import ImmutableEventStore, InferenceFailedEvent, HumanCorrectionProvidedEvent
from core.persistence.chroma_adapter import ChromaDBAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_demo() -> None:
    """Executa demonstração dos estágios da arquitetura."""
    print("=" * 70)
    print("  News Summarizer - Arquitetura Clean em Estágios (SLM On-Premise)")
    print("  Event Sourcing, Guardrails Determinísticos e FSM")
    print("=" * 70)

    event_store = ImmutableEventStore()

    try:
        memory = ChromaDBAdapter("data/chroma")
    except Exception as e:
        logging.warning(f"ChromaDB não inicializado ou vazio: {e}. Executando em modo desacoplado.")
        memory = None

    orchestrator = FSMOrchestrator(semantic_memory=memory, event_store=event_store)

    print("\n[1] --- Teste: Consulta Factual Ancorada (FACTUAL_QUERY) ---")
    res1 = orchestrator.handle_message("Quais são as notícias sobre inteligência artificial?")
    print(f"Resposta:\n{res1.response}\n[Intenção: {res1.intent.value}]")

    print("\n[2] --- Teste: Consulta Fora de Escopo (OUT_OF_DOMAIN Guardrail) ---")
    res2 = orchestrator.handle_message("Como preparar uma receita de bolo de cenoura?")
    print(f"Resposta:\n{res2.response}\n[Intenção: {res2.intent.value} | Handoff: {res2.handoff_triggered}]")

    print("\n[3] --- Teste: Disparo Manual de Fail-Fast (Circuito Aberto) ---")
    orchestrator.trigger_fail_fast(reason="Alucinação detectada por teste heurístico")

    print("\n[4] --- Teste: Intervenção Humana (HITL) e Resolução ---")
    correction = HumanCorrectionProvidedEvent(
        session_id="sessao_demo",
        failed_event_id="system_fail_fast",
        correction_text="Sintetize apenas fatos estritamente contidos nas notícias arquivadas.",
    )
    event_store.append(correction)
    orchestrator.handle_human_correction(correction.correction_text)
    print("Correção manual registrada no Event Store imutável com sucesso.")

    print("\n[5] --- Trilha de Auditoria (Event Sourcing) ---")
    for event in event_store.get_events():
        print(f"[{event.timestamp.strftime('%H:%M:%S')}] {type(event).__name__} (session: {event.session_id})")


def interactive_chat() -> None:
    """Modo interativo no terminal similar ao padrão de produção."""
    print("=" * 70)
    print("  Terminal Interativo - News Summarizer (Modo Oráculo Estrito)")
    print("  Digite 'sair' para encerrar.")
    print("=" * 70)

    try:
        memory = ChromaDBAdapter("data/chroma")
    except Exception:
        memory = None

    orchestrator = FSMOrchestrator(semantic_memory=memory)

    while True:
        try:
            user_input = input("\nVocê: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["sair", "exit", "quit", "q"]:
                print("\nEncerrando terminal. Até logo!")
                break

            result = orchestrator.handle_message(user_input)
            print(f"\nOráculo [{result.intent.value}]:\n{result.response}")

            if result.handoff_triggered:
                print(f"[Transbordo / Aviso]: Motivo={result.handoff_reason}")

            if result.contexts:
                print(f"\n[Auditoria: {len(result.contexts)} fontes consultadas]")

        except KeyboardInterrupt:
            print("\nEncerrado pelo usuário.")
            break
        except Exception as e:
            print(f"[Erro]: {e}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        interactive_chat()
    else:
        run_demo()


if __name__ == "__main__":
    main()
