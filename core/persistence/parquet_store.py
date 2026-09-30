import polars as pl
from pathlib import Path
from core.domain.article import Article, SummarizedArticle, GlobalBulletin, DailySummary
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
        self.summaries_path = self.storage_dir / "summaries.parquet"
        self.bulletins_path = self.storage_dir / "bulletins.parquet"

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

    def append_summaries(self, summaries: List[SummarizedArticle]):
        if not summaries:
            return
            
        data = [{
            "original_url": s.original_url,
            "title": s.title,
            "summary": s.summary,
            "processed_at": s.processed_at
        } for s in summaries]
        
        schema = {
            "original_url": pl.Utf8,
            "title": pl.Utf8,
            "summary": pl.Utf8,
            "processed_at": pl.Datetime
        }
        
        df_new = pl.DataFrame(data, schema=schema)
        
        if self.summaries_path.exists():
            df_existing = pl.read_parquet(self.summaries_path)
            df_combined = pl.concat([df_existing, df_new])
            df_combined.write_parquet(self.summaries_path)
        else:
            df_new.write_parquet(self.summaries_path)

    def append_bulletin(self, bulletin: GlobalBulletin):
        if not bulletin:
            return
            
        data = [{
            "content": bulletin.content,
            "processed_at": bulletin.processed_at
        }]
        
        schema = {
            "content": pl.Utf8,
            "processed_at": pl.Datetime
        }
        
        df_new = pl.DataFrame(data, schema=schema)
        
        if self.bulletins_path.exists():
            df_existing = pl.read_parquet(self.bulletins_path)
            df_combined = pl.concat([df_existing, df_new])
            df_combined.write_parquet(self.bulletins_path)
        else:
            df_new.write_parquet(self.bulletins_path)

    def get_all_simhashes(self) -> List[int]:
        if not self.file_path.exists():
            return []
        df = pl.read_parquet(self.file_path, columns=["simhash_value"])
        return [int(float(h)) if '.' in str(h) else int(h) for h in df["simhash_value"].drop_nulls().to_list()]

    def get_latest_articles(self, limit: int = 5) -> List[dict]:
        if not self.file_path.exists():
            return []
        df = pl.read_parquet(self.file_path)
        if df.height == 0:
            return []
        
        df_sorted = df.sort("published_at", descending=True)
        return df_sorted.head(limit).to_dicts()

    def get_latest_summaries(self, limit: int = 5) -> List[dict]:
        if not self.summaries_path.exists():
            return []
        df = pl.read_parquet(self.summaries_path)
        if df.height == 0:
            return []
        
        df_sorted = df.sort("processed_at", descending=True)
        return df_sorted.head(limit).to_dicts()

    def get_unsummarized_articles(self) -> List[dict]:
        """Retorna todos os artigos brutos que ainda não possuem um resumo associado."""
        if not self.file_path.exists():
            return []
            
        df_articles = pl.read_parquet(self.file_path)
        
        if not self.summaries_path.exists():
            # Se a tabela de resumos não existe, todos os artigos são inéditos
            return df_articles.sort("published_at", descending=True).to_dicts()
            
        df_summaries = pl.read_parquet(self.summaries_path)
        
        if df_summaries.height == 0:
            return df_articles.sort("published_at", descending=True).to_dicts()
            
        # Extrai a lista de URLs que já foram processadas
        summarized_urls = df_summaries["original_url"].to_list()
        
        # Filtra os artigos brutos rejeitando aqueles cuja URL já está na lista dos processados
        df_unsummarized = df_articles.filter(~pl.col("url").is_in(summarized_urls))
        
        # Ordena os inéditos pelos mais recentes
        df_unsummarized = df_unsummarized.sort("published_at", descending=True)
        
        return df_unsummarized.to_dicts()

    def get_all_bulletins(self) -> List[dict]:
        """Recupera todos os boletins salvos, ordenados do mais recente para o mais antigo."""
        if not self.bulletins_path.exists():
            return []
        df = pl.read_parquet(self.bulletins_path)
        if df.height == 0:
            return []
        
        df_sorted = df.sort("processed_at", descending=True)
        return df_sorted.to_dicts()

    def append_daily_summary(self, daily: DailySummary):
        if not daily:
            return
            
        self.daily_path = self.storage_dir / "daily_summaries.parquet"
        
        data = [{
            "target_date": daily.target_date,
            "content": daily.content,
            "processed_at": daily.processed_at
        }]
        
        schema = {
            "target_date": pl.Utf8,
            "content": pl.Utf8,
            "processed_at": pl.Datetime
        }
        
        df_new = pl.DataFrame(data, schema=schema)
        
        if self.daily_path.exists():
            df_existing = pl.read_parquet(self.daily_path)
            df_combined = pl.concat([df_existing, df_new])
            df_combined.write_parquet(self.daily_path)
        else:
            df_new.write_parquet(self.daily_path)

    def get_daily_summary(self, target_date: str) -> dict:
        """Retorna o mega-boletim de uma data específica, se existir."""
        self.daily_path = self.storage_dir / "daily_summaries.parquet"
        if not self.daily_path.exists():
            return None
            
        df = pl.read_parquet(self.daily_path)
        if df.height == 0:
            return None
            
        # Filtra pela data alvo
        df_filtered = df.filter(pl.col("target_date") == target_date)
        if df_filtered.height == 0:
            return None
            
        # Pega a versão mais recente caso haja reprocessamento
        return df_filtered.sort("processed_at", descending=True).head(1).to_dicts()[0]

    def get_available_bulletin_dates(self) -> List[str]:
        """Retorna uma lista única das datas (YYYY-MM-DD) que possuem boletins salvos."""
        if not self.bulletins_path.exists():
            return []
            
        df = pl.read_parquet(self.bulletins_path)
        if df.height == 0:
            return []
            
        # Converte a coluna processed_at para string YYYY-MM-DD e pega os únicos
        dates = df.with_columns(pl.col("processed_at").dt.to_string("%Y-%m-%d").alias("date_str"))
        unique_dates = dates["date_str"].unique().sort(descending=True).to_list()
        return unique_dates
        
    def get_bulletins_by_date(self, target_date: str) -> List[dict]:
        """Recupera todos os boletins gerados em um dia específico."""
        if not self.bulletins_path.exists():
            return []
            
        df = pl.read_parquet(self.bulletins_path)
        if df.height == 0:
            return []
            
        # Cria uma coluna temporária para facilitar o filtro
        df = df.with_columns(pl.col("processed_at").dt.to_string("%Y-%m-%d").alias("date_str"))
        df_filtered = df.filter(pl.col("date_str") == target_date)
        
        return df_filtered.sort("processed_at", descending=False).to_dicts()
