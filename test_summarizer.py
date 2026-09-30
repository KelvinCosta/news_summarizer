import logging
from datetime import datetime
from core.persistence.parquet_store import ParquetStore
from core.persistence.chroma_adapter import ChromaDBAdapter
from core.domain.article import SummarizedArticle, GlobalBulletin
from core.inference.ollama_agent import OllamaSummarizerAgent, OllamaMasterAgent
from core.fsm.states import SystemState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    print("=== Iniciando Pipeline Map-Reduce Dinâmico (Agentes SLM Locais) ===")
    
    parquet_store = ParquetStore("data/parquet")
    semantic_memory = ChromaDBAdapter("data/chroma")
    
    # 1. Recuperando TODOS os artigos brutos inéditos
    articles = parquet_store.get_unsummarized_articles()
    
    if not articles:
        logging.info("Não há notícias novas aguardando sumarização hoje.")
        return
        
    logging.info(f"[{SystemState.RETRIEVING_CONTEXT.value}] Detectados {len(articles)} artigos brutos inéditos no Parquet.")
    
    summarizer_agent = OllamaSummarizerAgent(model_name="llama3.2")
    master_agent = OllamaMasterAgent(model_name="llama3.2")
    
    summarized_objects = []
    
    # FASE 1: MAP (Sumarizações Individuais)
    print("\n--- FASE 1: Processando Resumos Individuais ---")
    for article in articles:
        title = article['title']
        content = article['content']
        url = article['url']
        
        logging.info(f"[{SystemState.INFERENCING.value}] Agente 1 (Summarizer): Lendo '{title}'")
        summary = summarizer_agent.summarize(f"{title}\n\n{content}")
        
        if summary:
            summarized_obj = SummarizedArticle(
                original_url=url,
                title=title,
                summary=summary,
                processed_at=datetime.utcnow()
            )
            summarized_objects.append(summarized_obj)
            
            # INJEÇÃO NO HIPOCAMPO (ChromaDB)
            semantic_memory.store_individual_summary(summarized_obj)
            
            logging.info(f"[{SystemState.COMPLETED.value}] Resumo gerado e vetorizado com sucesso.")
        else:
            logging.error(f"[{SystemState.FAILED.value}] Falha na inferência.")
            return

    # Salvando os resumos no Parquet
    logging.info(f"[{SystemState.CONSOLIDATING.value}] Salvando {len(summarized_objects)} resumos em summaries.parquet...")
    parquet_store.append_summaries(summarized_objects)
    
    # FASE 2: REDUCE (Sumário Global)
    print("\n--- FASE 2: Agente Master (Boletim Global) ---")
    
    summary_texts = [s.summary for s in summarized_objects]
    logging.info(f"[{SystemState.INFERENCING.value}] Agente 2 (Master): Sintetizando o Boletim Diário a partir dos resumos...")
    
    global_bulletin = master_agent.generate_global_bulletin(summary_texts)
    
    if global_bulletin:
        print("\n" + "="*80)
        print("📰 BOLETIM DIÁRIO GLOBAL")
        print("="*80)
        print(f"\n{global_bulletin}\n")
        print("="*80 + "\n")
        
        # Salvando no Event Store (Parquet)
        bulletin_obj = GlobalBulletin(content=global_bulletin, processed_at=datetime.utcnow())
        parquet_store.append_bulletin(bulletin_obj)
        logging.info(f"[{SystemState.CONSOLIDATING.value}] Boletim Final salvo em 'bulletins.parquet' com sucesso!")
        logging.info(f"[{SystemState.COMPLETED.value}] Pipeline concluído com sucesso!")
    else:
        logging.error(f"[{SystemState.FAILED.value}] Agente Master falhou na geração do boletim.")

if __name__ == "__main__":
    run_pipeline()
