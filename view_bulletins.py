import sys
from datetime import datetime
from core.persistence.parquet_store import ParquetStore
from core.domain.article import DailySummary
from core.inference.ollama_agent import OllamaMasterAgent

def view_by_date(store: ParquetStore):
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
                        print("\n" + "="*80)
                        print(f"🌟 MEGA-BOLETIM DO DIA: {selected_date} (Gerado agora e salvo!)")
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
    while True:
        print("\n=== MENU DE NAVEGAÇÃO DE BOLETINS ===")
        print("[1] Visualizar Mega-Boletins por Data (Visão Diária)")
        print("[2] Visualizar Boletins Parciais (Sessões Específicas)")
        print("[3] Sair")
        
        try:
            choice = input("> ")
            if choice == '1':
                view_by_date(store)
            elif choice == '2':
                view_by_bulletin(store)
            elif choice == '3' or choice.lower() == 'q':
                print("Saindo.")
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
