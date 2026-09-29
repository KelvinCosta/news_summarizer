import polars as pl
from pathlib import Path
from core.domain.article import Article
from typing import List

class ParquetStore:
    """
    Primeira Camada de Persistência (Event Store).
    Grava o log bruto em formato Parquet no padrão append-only.
    """
    def __init__(self, storage_dir: str = "data/parquet"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.storage_dir / "articles.parquet"

    def append(self, articles: List[Article]):
        if not articles:
            return
            
        data = [{
            "url": a.url,
            "title": a.title,
            "content": a.content,
            "published_at": a.published_at,
            "source": a.source,
            "simhash_value": a.simhash_value
        } for a in articles]
        
        df_new = pl.DataFrame(data)
        
        if self.file_path.exists():
            df_existing = pl.read_parquet(self.file_path)
            # Append-only (concatenando para manter a Fonte da Verdade)
            df_combined = pl.concat([df_existing, df_new])
            df_combined.write_parquet(self.file_path)
        else:
            df_new.write_parquet(self.file_path)

    def get_all_simhashes(self) -> List[int]:
        if not self.file_path.exists():
            return []
        df = pl.read_parquet(self.file_path, columns=["simhash_value"])
        return df["simhash_value"].drop_nulls().to_list()
