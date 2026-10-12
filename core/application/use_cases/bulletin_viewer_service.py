"""Caso de uso: Leitura e Gestão de Boletins Diários e Parciais.

Permite consultar boletins consolidados por data, listar boletins de sessões
individuais e disparar a fusão sob demanda para Mega-Boletim via Agente Master.
"""

from datetime import datetime, timezone
from typing import List, Optional

from core.domain.article import DailySummary
from core.domain.ports import ISemanticMemory
from core.inference.ollama_agent import OllamaMasterAgent
from core.persistence.chroma_adapter import ChromaDBAdapter
from core.persistence.parquet_store import ParquetStore


class BulletinViewerService:
    """Serviço para visualização e fusão de boletins executivos."""

    def __init__(
        self,
        parquet_store: Optional[ParquetStore] = None,
        semantic_memory: Optional[ISemanticMemory] = None,
        master_agent: Optional[OllamaMasterAgent] = None,
    ) -> None:
        self.store = parquet_store or ParquetStore("data/parquet")
        self.memory = semantic_memory or ChromaDBAdapter("data/chroma")
        self.master_agent = master_agent or OllamaMasterAgent()

    def get_dates(self) -> List[str]:
        """Retorna lista única de datas disponíveis (YYYY-MM-DD)."""
        return self.store.get_available_bulletin_dates()

    def get_daily_content(self, date_str: str) -> tuple[Optional[str], str]:
        """Recupera ou gera o Mega-Boletim para uma data.

        Returns:
            Tupla (conteudo_markdown, tipo_origem) onde tipo_origem pode ser:
            'cached', 'single_session', 'fused_now', 'not_found'.
        """
        daily = self.store.get_daily_summary(date_str)
        if daily:
            return daily["content"], "cached"

        bulletins = self.store.get_bulletins_by_date(date_str)
        if not bulletins:
            return None, "not_found"

        if len(bulletins) == 1:
            return bulletins[0]["content"], "single_session"

        # Fusão sob demanda via Agente Master
        content_list = [b["content"] for b in bulletins]
        master_content = self.master_agent.generate_daily_master(content_list, date_str)

        if master_content:
            new_daily = DailySummary(
                target_date=date_str,
                content=master_content,
                processed_at=datetime.now(timezone.utc),
            )
            self.store.append_daily_summary(new_daily)
            if self.memory:
                self.memory.store_daily_summary(new_daily)
            return master_content, "fused_now"

        return bulletins[0]["content"], "partial_fallback"

    def get_all_bulletins(self) -> List[dict]:
        """Retorna todos os boletins parciais de sessões individuais."""
        return self.store.get_all_bulletins()

