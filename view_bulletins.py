import sys
from core.persistence.parquet_store import ParquetStore

def main():
    store = ParquetStore("data/parquet")
    bulletins = store.get_all_bulletins()
    
    if not bulletins:
        print("Nenhum boletim encontrado no banco de dados (bulletins.parquet está vazio ou não existe).")
        return
        
    print(f"\n=== Você tem {len(bulletins)} boletim(ns) salvo(s) ===")
    
    for i, b in enumerate(bulletins):
        # Formatando a data para facilitar a leitura
        dt = b['processed_at']
        print(f"[{i}] Boletim gerado em: {dt.strftime('%d/%m/%Y às %H:%M:%S')}")
        
    print("\nDigite o número do boletim que deseja ler (ou 'q' para sair):")
    
    while True:
        try:
            choice = input("> ")
            if choice.lower() == 'q':
                print("Saindo do leitor de boletins.")
                break
                
            idx = int(choice)
            if 0 <= idx < len(bulletins):
                selected = bulletins[idx]
                dt_str = selected['processed_at'].strftime('%d/%m/%Y às %H:%M:%S')
                
                print("\n" + "="*80)
                print(f"📰 LENDO BOLETIM DE {dt_str}")
                print("="*80)
                print(f"\n{selected['content']}\n")
                print("="*80 + "\n")
                print("Digite outro número para ler um boletim anterior ou 'q' para sair.")
            else:
                print("Número inválido. Tente novamente.")
        except ValueError:
            print("Entrada inválida. Digite apenas o número listado ou 'q' para sair.")
        except KeyboardInterrupt:
            print("\nSaindo do leitor de boletins.")
            break
        except EOFError:
            break

if __name__ == "__main__":
    main()
