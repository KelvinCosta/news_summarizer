import sys
from datetime import datetime
from core.persistence.parquet_store import ParquetStore
from core.persistence.chroma_adapter import ChromaDBAdapter
from core.domain.article import DailySummary
from core.inference.ollama_agent import OllamaMasterAgent
import ollama

def chat_with_kelvin(memory: ChromaDBAdapter):
    print("\n" + "="*80)
    print("🧠 Terminal de RAG (Busca Semântica)")
    print("Faça perguntas sobre as notícias já processadas.")
    print("="*80)
    
    while True:
        query = input("\nVocê: ")
        if query.lower() in ['q', 'sair', 'exit']:
            break
            
        print("Buscando memórias...")
        contexts = memory.recall_context(query, limit=3)
        
        if not contexts:
            print("Não encontrei nenhuma memória consolidada sobre este assunto.")
            continue
            
        # Montar o Prompt com o Contexto Injetado
        context_text = "\n\n---\n\n".join([c["content"] for c in contexts])
        
        prompt = (
            f"Use ESTRITAMENTE as informações abaixo para responder a pergunta do usuário.\n"
            f"Se a informação não estiver no contexto, diga que não possui dados suficientes em memória.\n\n"
            f"MEMÓRIA VETORIAL:\n{context_text}\n\n"
            f"PERGUNTA: {query}"
        )
        
        try:
            response = ollama.chat(model="llama3.2", messages=[
                {"role": "system", "content": "Você é um agente inteligente operando um sistema de Retrieval-Augmented Generation (RAG)."},
                {"role": "user", "content": prompt}
            ])
            print(f"\n{response['message']['content']}")
            
            # Auditoria (Rastreabilidade)
            print("\n[Auditoria de Memória]")
            for c in contexts:
                meta = c['metadata']
                tipo = meta.get('type', 'desconhecido')
                print(f" - Origem: {tipo} | Data: {meta.get('date', meta.get('processed_at', ''))}")
        except Exception as e:
            print(f"Erro na Inferência: {e}")

def view_by_date(store: ParquetStore, memory: ChromaDBAdapter):
    dates = store.get_available_bulletin_dates()
    if not dates:
        print("Nenhum boletim encontrado no banco de dados.")
        return
        
    print(f"\n--- MEGA-BOLETINS POR DATA ---")
    for i, date_str in enumerate(dates):
        print(f"[{i}] {date_str}")
        
    print("\nSelecione o número do dia (ou 'q' para voltar):")
    choice = input("> ")
    if choice.lower() == 'q':
        return
        
    try:
        idx = int(choice)
        if 0 <= idx < len(dates):
            selected_date = dates[idx]
            daily_summary = store.get_daily_summary(selected_date)
            
            if daily_summary:
                print("\n" + "="*80)
                print(f"🌟 MEGA-BOLETIM DO DIA: {selected_date} (Carregado do Banco)")
                print("="*80)
                print(f"\n{daily_summary['content']}\n")
                print("="*80 + "\n")
            else:
                bulletins = store.get_bulletins_by_date(selected_date)
                if len(bulletins) == 0:
                    print("Nenhum boletim encontrado para esta data.")
                elif len(bulletins) == 1:
                    print("\n" + "="*80)
                    print(f"📰 BOLETIM DO DIA: {selected_date} (Apenas 1 sessão registrada hoje)")
                    print("="*80)
                    print(f"\n{bulletins[0]['content']}\n")
                    print("="*80 + "\n")
                else:
                    print(f"\n[!] Detectados {len(bulletins)} boletins parciais para o dia {selected_date}.")
                    print("[!] Mega-Boletim não encontrado. Acionando Agente Master para fusão...")
                    
                    agent = OllamaMasterAgent(model_name="llama3.2")
                    content_list = [b['content'] for b in bulletins]
                    master_content = agent.generate_daily_master(content_list, selected_date)
                    
                    if master_content:
                        new_daily = DailySummary(
                            target_date=selected_date, 
                            content=master_content, 
                            processed_at=datetime.utcnow()
                        )
                        store.append_daily_summary(new_daily)
                        memory.store_daily_summary(new_daily) # SALVANDO NO CHROMA
                        
                        print("\n" + "="*80)
                        print(f"🌟 MEGA-BOLETIM DO DIA: {selected_date} (Gerado agora, salvo e vetorizado!)")
                        print("="*80)
                        print(f"\n{master_content}\n")
                        print("="*80 + "\n")
                    else:
                        print("Falha ao gerar o Mega-Boletim.")
        else:
            print("Número inválido.")
    except ValueError:
        print("Entrada inválida.")

def view_by_bulletin(store: ParquetStore):
    bulletins = store.get_all_bulletins()
    if not bulletins:
        print("Nenhum boletim encontrado no banco de dados.")
        return
        
    print(f"\n--- BOLETINS PARCIAIS (Sessões Individuais) ---")
    for i, b in enumerate(bulletins):
        dt = b['processed_at']
        print(f"[{i}] Boletim Parcial de: {dt.strftime('%d/%m/%Y às %H:%M:%S')}")
        
    print("\nSelecione o número do boletim (ou 'q' para voltar):")
    choice = input("> ")
    if choice.lower() == 'q':
        return
        
    try:
        idx = int(choice)
        if 0 <= idx < len(bulletins):
            selected = bulletins[idx]
            dt_str = selected['processed_at'].strftime('%d/%m/%Y às %H:%M:%S')
            
            print("\n" + "="*80)
            print(f"📰 LENDO BOLETIM PARCIAL DE {dt_str}")
            print("="*80)
            print(f"\n{selected['content']}\n")
            print("="*80 + "\n")
        else:
            print("Número inválido.")
    except ValueError:
        print("Entrada inválida.")

def main():
    store = ParquetStore("data/parquet")
    memory = ChromaDBAdapter("data/chroma")
    
    while True:
        print("\n=== MENU INTERATIVO ===")
        print("[1] Visualizar Mega-Boletins por Data (Visão Diária)")
        print("[2] Visualizar Boletins Parciais (Sessões Específicas)")
        print("[3] Fazer uma pergunta (RAG / Memória Semântica)")
        print("[4] Sair")
        
        try:
            choice = input("> ")
            if choice == '1':
                view_by_date(store, memory)
            elif choice == '2':
                view_by_bulletin(store)
            elif choice == '3':
                chat_with_kelvin(memory)
            elif choice == '4' or choice.lower() == 'q':
                print("Finalizando. Até logo!")
                break
            else:
                print("Opção inválida.")
        except KeyboardInterrupt:
            print("\nSaindo.")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
