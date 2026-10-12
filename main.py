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
        print("[2] Processar & Indexar Notícias (SLM Map -> Parquet -> ChromaDB)")
        print("[3] Gerar Boletim (Top 10 / Personalizado / Por Tema)")
        print("[4] Visualizar Histórico de Boletins Gerados")
        print("[5] Visualizar Sumários Individuais de Notícias (Navegar / Buscar)")
        print("[6] Conversar com o Oráculo RAG (Modo Estrito Anti-Alucinação)")
        print("[7] Trilha de Auditoria (Event Sourcing da Sessão)")
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
        print("🧠 Processamento & Indexação de Notícias (SLM Map + ChromaDB)...")
        print("=" * 60)

        unsummarized = self.parquet_store.get_unsummarized_articles()
        total_pending = len(unsummarized)

        if total_pending == 0:
            print("\n[!] Não há notícias novas aguardando sumarização hoje.")
            print("Dica: Execute a opção [1] para ingerir novos artigos.")
            return

        print(f"\nDetectadas {total_pending} notícias inéditas pendentes no Parquet.")
        print("Deseja resumir e indexar quantas nesta rodada? (SLMs locais levam ~2-4s por artigo)")
        print("  [Enter] Top 10 mais relevantes para IA/Tech (~30s)")
        print("  [número] Digite uma quantidade personalizada (ex: 25, 50)")
        print(f"  [todas]  Processar todas as {total_pending} notícias pendentes")

        choice = input("\nEscolha: ").strip().lower()
        limit: Optional[int] = 10
        if choice in ["todas", "all", "t"]:
            if total_pending > 50:
                est_min = int((total_pending * 3) / 60)
                confirm = input(
                    f"\n⚠️  Atenção: Processar {total_pending} notícias levará ~{est_min} minutos ininterruptos "
                    f"de GPU/VRAM.\nDeseja mesmo processar todas agora? (s/N): "
                ).strip().lower()
                if confirm not in ["s", "sim", "y", "yes"]:
                    print("Operação cancelada para segurança da GPU. Experimente um lote de 25 ou 50 artigos.")
                    return
            limit = None
        elif choice.isdigit() and int(choice) > 0:
            limit = int(choice)
        elif choice == "":
            limit = 10

        batch_desc = f"{limit} artigos prioritários" if limit else f"todos os {total_pending} artigos"
        print(f"\n🚀 Iniciando processamento e indexação de {batch_desc}...")

        def on_summary_progress(current: int, total: int, title: str) -> None:
            short_title = (title[:35] + "...") if len(title) > 35 else title
            sys.stdout.write(f"\r  [🧠 SLM] Processando ({current}/{total}): '{short_title}' ")
            sys.stdout.flush()

        result = self.summarizer_service.run(
            limit=limit,
            progress_callback=on_summary_progress,
            generate_bulletin=False,
        )
        print()

        if result["status"] == "empty":
            print(f"\n[!] {result['message']}")
            return

        if result["status"] == "failed":
            print(f"\n[X] Falha no processamento: {result['message']}")
            return

        print(f"\n✅ Concluído! {result['processed_count']} artigos resumidos e indexados no Hipocampo (ChromaDB) com sucesso.")
        print("💡 Dica: Use a opção [3] a qualquer momento para compor o Boletim das Top 10 notícias!")

    def generate_bulletin_menu(self) -> None:
        print("\n" + "=" * 60)
        print("📰 GERAR BOLETIM (Map-Reduce & Cache Inteligente)")
        print("=" * 60)
        print("Escolha a modalidade:")
        print("  [1] Top 10 mais relevantes (Padrão)")
        print("  [2] Quantidade personalizada (ex: 5, 15, 25)")
        print("  [3] Filtrar por tema (utiliza todas as notícias encontradas)")
        print("  [v] Voltar")

        sub = input("\nEscolha uma opção: ").strip().lower()
        if sub in ["v", "voltar", "q", "sair"]:
            return

        top_k: Optional[int] = 10
        theme: Optional[str] = None

        if sub in ["1", ""]:
            top_k = 10
            theme = None
        elif sub == "2":
            q_str = input("Quantidade desejada de notícias (ex: 5, 15, 25): ").strip()
            if not q_str.isdigit() or int(q_str) <= 0:
                print("[!] Quantidade inválida.")
                return
            top_k = int(q_str)
            theme = None
        elif sub == "3":
            t_str = input("Digite o tema desejado (ex: Trump, Apple, Chips, Segurança): ").strip()
            if not t_str:
                print("[!] Tema não informado.")
                return
            theme = t_str
            top_k = None
        else:
            print("[!] Opção inválida.")
            return

        print("\n🔍 Analisando notícias indexadas...")
        result = self.summarizer_service.generate_strategic_bulletin(
            top_k=top_k,
            theme=theme,
            force_regenerate=False,
        )

        if result["status"] == "empty":
            print(f"\n[!] {result['message']}")
            return

        if result["status"] == "failed" or not result["bulletin"]:
            print("\n[X] Falha ao gerar o boletim.")
            return

        articles = result.get("articles_used", [])
        is_cached = result.get("cached", False)
        topic_name = result.get("topic", "Geral")

        if is_cached:
            dt = result.get("processed_at")
            dt_str = dt.strftime("%d/%m/%Y às %H:%M:%S") if hasattr(dt, "strftime") else str(dt)
            print("\n" + "-" * 70)
            print(f"⚡ [CACHE ATIVO] Boletim idêntico já existente para este conjunto de notícias!")
            print(f"   Modalidade: {topic_name} | Gerado em: {dt_str} | Base: {len(articles)} notícias")
            print(f"   (Retornando versão em cache sem reprocessamento de GPU)")
            print("-" * 70)
        else:
            print(f"\n🎯 Novo Boletim Gerado com {len(articles)} Notícias ({topic_name}):")
            for idx, a in enumerate(articles[:10], 1):
                score = self.summarizer_service.bayesian_filter.score_article(f"{a.get('title', '')} {a.get('summary', '')}")
                print(f"  [{idx:02d}] (Score: {score:4.1f}) {a.get('title')[:65]}")
            if len(articles) > 10:
                print(f"  ... e mais {len(articles) - 10} notícias sintetizadas na esteira Map-Reduce.")

        print("\n" + "=" * 70)
        print(f"📰 BOLETIM CONSOLIDADO: {topic_name.upper()}")
        print("=" * 70)
        print(result["bulletin"])
        print("=" * 70)
        if not is_cached:
            print("💾 Novo boletim gravado em 'data/parquet/bulletins.parquet'.")

    # Alias para retrocompatibilidade
    generate_top_bulletin = generate_bulletin_menu

    def view_saved_bulletins(self) -> None:
        bulletins = self.bulletin_viewer.get_all_bulletins()
        if not bulletins:
            print("\n[!] Nenhum boletim encontrado no banco de dados.")
            print("Dica: Use a opção [3] para gerar um boletim primeiro.")
            return

        print("\n--- HISTÓRICO DE BOLETINS GERADOS ---")
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
                print(f"📰 BOLETIM DE {dt_str}")
                print("=" * 75)
                print(f"\n{selected['content']}\n")
                print("=" * 75)
            else:
                print("Número fora do intervalo.")
        except ValueError:
            print("Entrada inválida.")

    def view_individual_summaries(self) -> None:
        all_summaries = self.parquet_store.get_all_summaries()
        if not all_summaries:
            print("\n[!] Nenhum resumo encontrado no banco de dados.")
            print("Dica: Use a opção [2] para processar notícias primeiro.")
            return

        current_list = all_summaries
        filter_term = ""
        page_size = 10
        page = 0

        while True:
            total = len(current_list)
            total_pages = max(1, (total + page_size - 1) // page_size)
            if page >= total_pages:
                page = total_pages - 1

            start_idx = page * page_size
            end_idx = min(start_idx + page_size, total)
            page_items = current_list[start_idx:end_idx]

            header_info = f"[🔍 Filtro: '{filter_term}' - {total} encontrados]" if filter_term else f"[Total: {total}]"
            print("\n" + "=" * 75)
            print(f"📚 SUMÁRIOS INDIVIDUAIS DE NOTÍCIAS {header_info} (Página {page + 1}/{total_pages})")
            print("=" * 75)

            if total == 0:
                print(f"\n  Nenhum resumo encontrado para o filtro '{filter_term}'.")
            else:
                for i, s in enumerate(page_items, start=start_idx):
                    dt = s.get("processed_at")
                    dt_str = dt.strftime("%d/%m %H:%M") if hasattr(dt, "strftime") else str(dt)[:16]
                    title = s.get("title", "Sem título")
                    short_title = (title[:58] + "...") if len(title) > 58 else title
                    print(f"[{i:03d}] [{dt_str}] {short_title}")

            nav_commands = "[número] Ler resumo | [p] Próxima | [a] Anterior | [f] Filtrar/Buscar"
            if filter_term:
                nav_commands += " | [c] Limpar filtro"
            nav_commands += " | [v] Voltar"

            print(f"\nComandos: {nav_commands}")
            choice = input("Escolha: ").strip().lower()

            if choice in ["v", "voltar", "q", "sair"]:
                break
            elif choice in ["p", "prox", "proxima", "next"]:
                if page < total_pages - 1:
                    page += 1
                else:
                    print("\n[!] Você já está na última página.")
            elif choice in ["a", "ant", "anterior", "prev"]:
                if page > 0:
                    page -= 1
                else:
                    print("\n[!] Você já está na primeira página.")
            elif choice in ["c", "limpar", "clear"]:
                current_list = all_summaries
                filter_term = ""
                page = 0
                print("\n[✓] Filtro removido. Exibindo todos os resumos.")
            elif choice in ["f", "buscar", "filtro"]:
                term = input("\nDigite o termo de busca (ou Enter para cancelar): ").strip()
                if term:
                    matches = [
                        s for s in all_summaries
                        if term.lower() in s.get("title", "").lower() or term.lower() in s.get("summary", "").lower()
                    ]
                    current_list = matches
                    filter_term = term
                    page = 0
                    print(f"\n🔍 {len(matches)} resultados encontrados para '{term}'.")
            elif choice.isdigit():
                idx = int(choice)
                if 0 <= idx < total:
                    self._print_single_summary(current_list[idx])
                else:
                    print("\n[!] Número fora do intervalo da lista atual.")
            else:
                print("\n[!] Entrada inválida.")

    def _print_single_summary(self, item: dict) -> None:
        dt = item.get("processed_at")
        dt_str = dt.strftime("%d/%m/%Y às %H:%M:%S") if hasattr(dt, "strftime") else str(dt)
        print("\n" + "=" * 75)
        print(f"📰 {item.get('title', 'Sem título')}")
        print(f"📅 Data de Processamento: {dt_str}")
        if item.get("original_url"):
            print(f"🔗 Link da Fonte: {item['original_url']}")
        print("=" * 75)
        print(f"\n{item.get('summary', '')}\n")
        print("=" * 75)
        input("Pressione Enter para voltar à navegação...")

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
                    self.generate_bulletin_menu()
                elif choice == "4":
                    self.view_saved_bulletins()
                elif choice == "5":
                    self.view_individual_summaries()
                elif choice == "6":
                    self.run_rag_chat()
                elif choice == "7":
                    self.view_audit_log()
                elif choice in ["0", "sair", "exit", "quit", "q"]:
                    print("\nEncerrando o News Summarizer. Até logo! 👋\n")
                    break
                else:
                    print("\n[!] Opção inválida. Digite um número de 0 a 7.")
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
