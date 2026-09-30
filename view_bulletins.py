import sys
from datetime import datetime
from core.persistence.parquet_store import ParquetStore
from core.domain.article import DailySummary
from core.inference.ollama_agent import OllamaMasterAgent

def main():
    store = ParquetStore("data/parquet")
    dates = store.get_available_bulletin_dates()
    
    if not dates:
        print("Nenhum boletim encontrado no banco de dados.")
        return
        
    print(f"\n=== HISTÓRICO DE BOLETINS ===")
    print("Dias com atividades registradas:")
    for i, date_str in enumerate(dates):
        print(f"[{i}] {date_str}")
        
    print("\nSelecione o número do dia que deseja ler (ou 'q' para sair):")
    
    while True:
        try:
            choice = input("> ")
            if choice.lower() == 'q':
                print("Saindo.")
                break
                
            idx = int(choice)
            if 0 <= idx < len(dates):
                selected_date = dates[idx]
                
                # 1. Verifica se já existe um Mega-Boletim para esse dia
                daily_summary = store.get_daily_summary(selected_date)
                
                if daily_summary:
                    print("\n" + "="*80)
                    print(f"🌟 MEGA-BOLETIM DO DIA: {selected_date} (Carregado do Banco)")
                    print("="*80)
                    print(f"\n{daily_summary['content']}\n")
                    print("="*80 + "\n")
                else:
                    # 2. Se não existe, busca todos os boletins parciais desse dia
                    bulletins = store.get_bulletins_by_date(selected_date)
                    
                    if len(bulletins) == 0:
                        print("Erro inesperado: Nenhum boletim encontrado para esta data.")
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
                            # Salva o novo mega-boletim no Parquet
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
                            
                print("Digite outro número para ler um dia diferente ou 'q' para sair.")
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite apenas o número listado ou 'q' para sair.")
        except KeyboardInterrupt:
            print("\nSaindo.")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
