import chromadb
from chromadb.config import Settings
import ollama
from typing import List, Dict, Any
from core.domain.ports import ISemanticMemory
from core.domain.article import SummarizedArticle, DailySummary

class OllamaEmbeddingFunction:
    """Função customizada para gerar embeddings via Ollama local (On-Premise)."""
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"[ERRO] Falha ao gerar embedding com {self.model_name}. Ele está instalado? (ollama pull {self.model_name})")
                raise e
        return embeddings

class ChromaDBAdapter(ISemanticMemory):
    """
    Adaptador de Infraestrutura: Implementa ISemanticMemory conectando-se ao ChromaDB.
    """
    def __init__(self, storage_path: str = "data/chroma"):
        # Instanciamos o client com a telemetria desativada (Privacidade On-Premise)
        self.client = chromadb.PersistentClient(
            path=storage_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_fn = OllamaEmbeddingFunction("nomic-embed-text")
        
        # O Hipocampo: Uma única coleção para memórias consolidadas
        self.collection = self.client.get_or_create_collection(
            name="hippocampus_memory",
            embedding_function=self.embedding_fn
        )

    def store_daily_summary(self, daily: DailySummary) -> None:
        doc_id = f"daily_{daily.target_date}"
        
        metadata = {
            "type": "mega_bulletin",
            "date": daily.target_date,
            "processed_at": daily.processed_at.isoformat()
        }
        
        self.collection.upsert(
            documents=[daily.content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def store_individual_summary(self, summary: SummarizedArticle) -> None:
        import hashlib
        url_hash = hashlib.md5(summary.original_url.encode()).hexdigest()
        doc_id = f"summary_{url_hash}"
        
        metadata = {
            "type": "individual_summary",
            "source_url": summary.original_url,
            "title": summary.title,
            "processed_at": summary.processed_at.isoformat()
        }
        
        self.collection.upsert(
            documents=[summary.summary],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def recall_context(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        contexts = []
        if not results["documents"] or not results["documents"][0]:
            return contexts
            
        for i in range(len(results["documents"][0])):
            contexts.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
            
        return contexts
