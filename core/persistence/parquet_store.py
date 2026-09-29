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
            "simhash_value": str(a.simhash_value) if a.simhash_value is not None else None
        } for a in articles]
        
        schema = {
            "url": pl.Utf8,
            "title": pl.Utf8,
            "content": pl.Utf8,
            "published_at": pl.Datetime,
            "source": pl.Utf8,
            "simhash_value": pl.Utf8
        }
        
        df_new = pl.DataFrame(data, schema=schema)
        
        if self.file_path.exists():
            df_existing = pl.read_parquet(self.file_path)
            if df_existing["simhash_value"].dtype != pl.Utf8:
                df_existing = df_existing.with_columns(pl.col("simhash_value").cast(pl.Utf8))
            df_combined = pl.concat([df_existing, df_new])
            df_combined.write_parquet(self.file_path)
        else:
            df_new.write_parquet(self.file_path)

    def get_all_simhashes(self) -> List[int]:
        if not self.file_path.exists():
            return []
        df = pl.read_parquet(self.file_path, columns=["simhash_value"])
        return [int(float(h)) if '.' in str(h) else int(h) for h in df["simhash_value"].drop_nulls().to_list()]

    def get_latest_articles(self, limit: int = 5) -> List[dict]:
        """Recupera os artigos mais recentes com base na data de publicação."""
        if not self.file_path.exists():
            return []
        df = pl.read_parquet(self.file_path)
        if df.height == 0:
            return []
        
        # Ordena pelos mais recentes
        df_sorted = df.sort("published_at", descending=True)
        return df_sorted.head(limit).to_dicts()
