import logging
from core.persistence.parquet_store import ParquetStore
from core.inference.ollama_agent import OllamaSummarizerAgent
from core.fsm.states import SystemState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_summarizer():
    print("=== Iniciando Agente Sumarizador (Ollama Local) ===")
    
    # 1. Recuperando Contexto (Hipocampo rudimentar acessando o Parquet)
    parquet_store = ParquetStore("data/parquet")
    articles = parquet_store.get_latest_articles(limit=3)
    
    if not articles:
        logging.info("Nenhum artigo encontrado no Parquet para sumarizar.")
        return
        
    logging.info(f"[{SystemState.RETRIEVING_CONTEXT.value}] Recuperados {len(articles)} artigos recentes do Parquet.")
    
    # 2. Inicializando SLM (Small Language Model)
    # Por padrão, vamos tentar usar o Llama 3.2 que é um SLM fantástico e rápido.
    agent = OllamaSummarizerAgent(model_name="llama3.2")
    
    # 3. Executando a Inferência
    for article in articles:
        title = article['title']
        content = article['content']
        logging.info(f"[{SystemState.INFERENCING.value}] Gerando resumo para: {title}")
        
        # O agente fará a chamada ao Ollama local
        summary = agent.summarize(f"{title}\n\n{content}")
        
        if summary:
            print("\n" + "="*60)
            print(f"📰 TÍTULO: {title}")
            print(f"🤖 RESUMO (Llama):\n{summary}")
            print("="*60 + "\n")
            logging.info(f"[{SystemState.COMPLETED.value}] Resumo gerado com sucesso.")
        else:
            logging.error(f"[{SystemState.FAILED.value}] Falha na inferência. Verifique se o Ollama está rodando e o modelo baixado.")
            # Fail-fast action here if needed

if __name__ == "__main__":
    run_summarizer()
