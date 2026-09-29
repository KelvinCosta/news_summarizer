import logging
from core.ingestion.rss_parser import RSSParser
from core.persistence.parquet_store import ParquetStore
from core.filters.lexical import LexicalFilter
from core.filters.bayesian import BayesianFilter
from core.fsm.states import SystemState

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_funnel():
    print("=== Iniciando Teste do Funil de Ingestão e Triagem O(1) ===")
    
    # 1. Adaptador de Ingestão (Fetch RSS)
    # Feeds de teste (podem não funcionar em sandbox sem rede, mas instanciam para o teste)
    feeds = [
        "http://rss.cnn.com/rss/edition_technology.rss",
        "http://feeds.bbci.co.uk/news/technology/rss.xml"
    ]
    parser = RSSParser(feeds)
    
    try:
        logging.info(f"[{SystemState.INGESTING.value}] Puxando dados de {len(feeds)} feeds...")
        articles = parser.fetch_all()
        logging.info(f"Total de artigos brutos extraídos: {len(articles)}")
    except Exception as e:
        logging.error(f"Erro no parser: {e}")
        return

    # 2. Configurando Filtros e Armazenamento
    parquet_store = ParquetStore("data/parquet")
    lexical_filter = LexicalFilter(tolerance=3)
    bayesian_filter = BayesianFilter("vocabulary_seed.json")
    
    existing_hashes = parquet_store.get_all_simhashes()
    
    accepted_articles = []
    
    # 3. O Funil Determinístico
    for article in articles:
        # A. Filtro Lexical (SimHash)
        new_hash = lexical_filter.compute_hash(article.content)
        object.__setattr__(article, 'simhash_value', new_hash)
        
        if lexical_filter.is_duplicate(new_hash, existing_hashes):
            logging.warning(f"[{SystemState.DISCARDED_LEXICALDUPLICATE.value}] Descartado (Cópia Lexical): {article.title}")
            continue
            
        existing_hashes.append(new_hash)
        
        # B. Filtro Bayesiano (Naive Bayes com Cold Start)
        score = bayesian_filter.score_article(article.content + " " + article.title)
        if not bayesian_filter.is_relevant(article.content + " " + article.title, threshold=1.0):
            logging.warning(f"[{SystemState.DISCARDED_IRRELEVANT.value}] Descartado (Irrelevante/Score {score}): {article.title}")
            continue
            
        logging.info(f"[{SystemState.ROUTING.value}] Artigo aceito (Score {score}): {article.title}")
        accepted_articles.append(article)
        
    # 4. Primeira Camada de Persistência (Event Store em Parquet)
    logging.info(f"[{SystemState.CONSOLIDATING.value}] Salvando {len(accepted_articles)} artigos válidos no Parquet...")
    parquet_store.append(accepted_articles)
    
    print("\n=== Funil Concluído ===")
    print(f"Brutos: {len(articles)} | Aceitos: {len(accepted_articles)}")

if __name__ == "__main__":
    test_funnel()
