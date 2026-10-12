"""Ponto de Entrada Oficial do News Summarizer (On-Premise SLM Architecture).

Unifica Ingestão O(1), Processamento Map-Reduce, Visualização de Boletins,
Terminal de RAG com o Oráculo Estrito e Auditoria de Event Sourcing sob uma única interface.

Uso:
    python main.py
"""

import logging
import os
import sys
from typing import Optional

from core.application.use_cases.bulletin_viewer_service import BulletinViewerService
from core.application.use_cases.ingestion_service import IngestionService
from core.application.use_cases.summarizer_service import SummarizerService
from core.domain.models.events import ImmutableEventStore
from core.fsm.orchestrator import FSMOrchestrator
from core.persistence.chroma_adapter import ChromaDBAdapter
from core.persistence.parquet_store import ParquetStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class NewsSummarizerApp:
    """Aplicação principal unificada do News Summarizer."""

    def __init__(self) -> None:
        self.parquet_store = ParquetStore("data/parquet")
        self.event_store = ImmutableEventStore()

        try:
            self.memory = ChromaDBAdapter("data/chroma")
        except Exception as e:
            logging.warning(f"ChromaDB não pôde ser inicializado completamente: {e}")
            self.memory = None

        self.ingestion_service = IngestionService(
            parquet_store=self.parquet_store,
            event_store=self.event_store,
        )
        self.summarizer_service = SummarizerService(
            parquet_store=self.parquet_store,
            semantic_memory=self.memory,
        )
        self.bulletin_viewer = BulletinViewerService(
            parquet_store=self.parquet_store,
            semantic_memory=self.memory,
        )
        self.orchestrator = FSMOrchestrator(
            semantic_memory=self.memory,
            event_store=self.event_store,
            parquet_store=self.parquet_store,
        )

    def print_header(self) -> None:
        print("\n" + "=" * 75)
        print("  📰 NEWS SUMMARIZER - Inteligência Autônoma de Notícias (SLM On-Premise)")
        print("  Arquitetura em Estágios, Clean Pipeline, CQRS & Event Sourcing")
        print("=" * 75)

    def print_menu(self) -> None:
        print("\n=== MENU PRINCIPAL ===")
        print("[1] Ingestão de Notícias & Funil O(1) (RSS -> SimHash -> Naive Bayes)")
        print("[2] Executar Pipeline Map-Reduce (Resumos SLM + Vetorização + Boletim)")
        print("[3] Visualizar Mega-Boletins por Data (Visão Executiva)")
        print("[4] Visualizar Boletins Parciais (Sessões Individuais)")
        print("[5] Conversar com o Oráculo RAG (Modo Estrito Anti-Alucinação)")
        print("[6] Trilha de Auditoria (Event Sourcing da Sessão)")
        print("[0] Sair")

    def run_ingestion(self) -> None:
        total_feeds = len(self.ingestion_service.feeds)
        print("\n" + "=" * 60)
        print(f"🚀 Ingestão Concorrente: Baixando {total_feeds} Feeds RSS em paralelo...")
        print("   (Fail-fast timeout: 7s por feed | 16 workers assíncronos)")
        print("=" * 60)

        def on_feed_progress(done: int, total: int, count: int) -> None:
            pct = int((done / total) * 100) if total > 0 else 100
            sys.stdout.write(
                f"\r  [📡 Rede] Feeds: {done}/{total} ({pct}%) | Artigos capturados: {count} "
            )
            sys.stdout.flush()

        result = self.ingestion_service.run(progress_callback=on_feed_progress)
        print()

        print("\n=== Resumo do Funil Matemático O(1) ===")
        print(f"Total de Artigos Brutos : {result['total_raw']}")
        print(f"Descartados por Cópia   : {result['discarded_duplicates']} (SimHash O(1))")
        print(f"Descartados Irrelevantes: {result['discarded_irrelevant']} (Naive Bayes)")
        print(f"Artigos Aceitos e Salvos: {result['accepted']}")
        if result["accepted"] > 0:
            print("Artigos gravados em 'data/parquet/articles.parquet'.")

    def run_summarization(self) -> None:
        print("\n" + "=" * 60)
        print("🧠 Pipeline Map-Reduce (Agentes SLM Locais)...")
        print("=" * 60)

        unsummarized = self.parquet_store.get_unsummarized_articles()
        total_pending = len(unsummarized)

        if total_pending == 0:
            print("\n[!] Não há notícias novas aguardando sumarização hoje.")
            print("Dica: Execute a opção [1] para ingerir novos artigos.")
            return

        print(f"\nDetectadas {total_pending} notícias inéditas pendentes no Parquet.")
        print("Deseja resumir quantas nesta rodada? (SLMs locais levam ~3-6s por artigo)")
        print("  [Enter] Top 10 mais recentes (recomendado para resposta rápida)")
        print("  [número] Digite uma quantidade personalizada (ex: 5, 20)")
        print("  [todas]  Processar todas as notícias pendentes")

        choice = input("\nEscolha: ").strip().lower()
        limit: Optional[int] = 10
        if choice in ["todas", "all", "t"]:
            limit = None
        elif choice.isdigit() and int(choice) > 0:
            limit = int(choice)
        elif choice == "":
            limit = 10

        batch_desc = f"{limit} artigos" if limit else f"todos os {total_pending} artigos"
        print(f"\n🚀 Iniciando síntese Map-Reduce de {batch_desc}...")

        def on_summary_progress(current: int, total: int, title: str) -> None:
            short_title = (title[:35] + "...") if len(title) > 35 else title
            sys.stdout.write(f"\r  [🧠 SLM] Processando ({current}/{total}): '{short_title}' ")
            sys.stdout.flush()

        result = self.summarizer_service.run(limit=limit, progress_callback=on_summary_progress)
        print()

        if result["status"] == "empty":
            print(f"\n[!] {result['message']}")
            return

        if result["status"] == "failed":
            print(f"\n[X] Falha no processamento: {result['message']}")
            return

        print(f"\n✅ Concluído! {result['processed_count']} artigos resumidos e vetorizados.")
        if result["bulletin"]:
            print("\n" + "=" * 70)
            print("📰 BOLETIM DIÁRIO GLOBAL CONSOLIDADO")
            print("=" * 70)
            print(result["bulletin"])
            print("=" * 70)

    def view_by_date(self) -> None:
        dates = self.bulletin_viewer.get_dates()
        if not dates:
            print("\n[!] Nenhum boletim encontrado no banco de dados.")
            return

        print("\n--- DATAS DISPONÍVEIS ---")
        for i, date_str in enumerate(dates):
            print(f"[{i}] {date_str}")

        choice = input("\nSelecione o número da data desejada (ou 'v' para voltar): ").strip()
        if choice.lower() in ["v", "q", "voltar"]:
            return

        try:
            idx = int(choice)
            if 0 <= idx < len(dates):
                selected_date = dates[idx]
                content, origin = self.bulletin_viewer.get_daily_content(selected_date)

                if content:
                    orig_map = {
                        "cached": "Recuperado do Cache / Parquet",
                        "single_session": "Sessão Única do Dia",
                        "fused_now": "Fundido Agora pelo Agente Master",
                        "partial_fallback": "Boletim Parcial",
                    }
                    print("\n" + "=" * 75)
                    print(f"🌟 MEGA-BOLETIM DO DIA: {selected_date} ({orig_map.get(origin, origin)})")
                    print("=" * 75)
                    print(f"\n{content}\n")
                    print("=" * 75)
                else:
                    print("Nenhum conteúdo pôde ser recuperado para esta data.")
            else:
                print("Número fora do intervalo.")
        except ValueError:
            print("Entrada inválida.")

    def view_by_session(self) -> None:
        bulletins = self.bulletin_viewer.get_all_bulletins()
        if not bulletins:
            print("\n[!] Nenhum boletim parcial encontrado no banco de dados.")
            return

        print("\n--- BOLETINS PARCIAIS (Sessões Individuais) ---")
        for i, b in enumerate(bulletins):
            dt = b["processed_at"]
            dt_str = dt.strftime("%d/%m/%Y às %H:%M:%S") if hasattr(dt, "strftime") else str(dt)
            print(f"[{i}] Boletim de: {dt_str}")

        choice = input("\nSelecione o número do boletim (ou 'v' para voltar): ").strip()
        if choice.lower() in ["v", "q", "voltar"]:
            return

        try:
            idx = int(choice)
            if 0 <= idx < len(bulletins):
                selected = bulletins[idx]
                dt = selected["processed_at"]
                dt_str = dt.strftime("%d/%m/%Y às %H:%M:%S") if hasattr(dt, "strftime") else str(dt)
                print("\n" + "=" * 75)
                print(f"📰 BOLETIM PARCIAL DE {dt_str}")
                print("=" * 75)
                print(f"\n{selected['content']}\n")
                print("=" * 75)
            else:
                print("Número fora do intervalo.")
        except ValueError:
            print("Entrada inválida.")

    def run_rag_chat(self) -> None:
        print("\n" + "=" * 75)
        print("🧠 Terminal de RAG - Modo Oráculo Estrito (Clean Architecture)")
        print("Faça perguntas sobre as notícias arquivadas. Digite 'v' para voltar ao menu.")
        print("=" * 75)

        while True:
            try:
                query = input("\nVocê: ").strip()
                if not query:
                    continue
                if query.lower() in ["v", "voltar", "q", "sair"]:
                    break

                print(" (Processando pelos 4 estágios do pipeline...)")
                result = self.orchestrator.handle_message(query)

                print(f"\nOráculo [{result.intent.value}]:\n{result.response}")

                if result.handoff_triggered:
                    print(f"\n[Transbordo / Aviso]: Motivo={result.handoff_reason}")

                if result.contexts:
                    print("\n[Auditoria de Origem]")
                    for c in result.contexts:
                        meta = c.get("metadata", {})
                        tipo = meta.get("type", "noticia")
                        orig = meta.get("source_url") or meta.get("title") or "Base interna"
                        data_str = meta.get("date") or meta.get("processed_at", "")
                        print(f" - [{tipo}] {orig} (Data: {data_str})")

            except KeyboardInterrupt:
                print("\nChat interrompido.")
                break
            except Exception as e:
                print(f"Erro no processamento: {e}")

    def view_audit_log(self) -> None:
        session_events = self.event_store.get_events()
        history = self.event_store.load_persistent_history()

        print("\n" + "=" * 75)
        print("📋 TRILHA DE AUDITORIA & FORENSE - EVENT SOURCING")
        print("=" * 75)
        print(f"• Eventos na sessão atual (memória): {len(session_events)}")
        print(f"• Total histórico gravado em disco  : {len(history)} (data/events/audit_log.jsonl)")

        if history:
            guardrails_count = sum(1 for e in history if e.get("event_type") == "GuardrailTriggered")
            handoffs_count = sum(1 for e in history if e.get("event_type") == "HandoffTriggered")
            queries_count = sum(1 for e in history if e.get("event_type") == "UserMessageReceived")
            corrections_count = sum(1 for e in history if e.get("event_type") == "HumanCorrectionProvidedEvent")

            print("\n--- Estatísticas Forenses do Sistema ---")
            print(f"  Total de Mensagens Recebidas : {queries_count}")
            print(f"  Guardrails Factuais Acionados: {guardrails_count}")
            print(f"  Transbordos / Recusas Seguras: {handoffs_count}")
            print(f"  Intervenções Humanas (HITL)  : {corrections_count}")

        if not session_events and not history:
            print("\n[!] Nenhum evento registrado ainda.")
            print("=" * 75)
            return

        print("\nÚltimos 10 eventos registrados:")
        recent = history[-10:] if history else [
            {"event_type": type(e).__name__, "timestamp": e.timestamp.isoformat(), "session_id": e.session_id}
            for e in session_events[-10:]
        ]
        for rec in recent:
            etype = rec.get("event_type", "Event")
            ts = rec.get("timestamp", "")[:19]
            sid = rec.get("session_id", "")
            extra = ""
            if "intent" in rec:
                extra = f" | intent={rec['intent']}"
            elif "guardrail_name" in rec:
                extra = f" | guardrail={rec['guardrail_name']} ({rec.get('reason', '')})"
            elif "reason" in rec:
                extra = f" | reason={rec['reason']}"
            elif "status" in rec:
                extra = f" | {rec.get('title', '')} ({rec['status']})"
            print(f" [{ts}] {etype} | session={sid}{extra}")
        print("=" * 75)

    def start(self) -> None:
        self.print_header()

        while True:
            self.print_menu()
            try:
                choice = input("\nEscolha uma opção: ").strip()
                if choice == "1":
                    self.run_ingestion()
                elif choice == "2":
                    self.run_summarization()
                elif choice == "3":
                    self.view_by_date()
                elif choice == "4":
                    self.view_by_session()
                elif choice == "5":
                    self.run_rag_chat()
                elif choice == "6":
                    self.view_audit_log()
                elif choice in ["0", "sair", "exit", "quit", "q"]:
                    print("\nEncerrando o News Summarizer. Até logo! 👋\n")
                    break
                else:
                    print("\n[!] Opção inválida. Digite um número de 0 a 6.")
            except KeyboardInterrupt:
                print("\n\nSessão encerrada pelo usuário. Até logo!")
                break
            except EOFError:
                break


def main() -> None:
    app = NewsSummarizerApp()
    app.start()


if __name__ == "__main__":
    main()
