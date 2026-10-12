from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest

from core.application.use_cases.summarizer_service import SummarizerService
from core.domain.article import Article
from core.domain.ports import ISemanticMemory
from core.inference.ollama_agent import OllamaMasterAgent, OllamaSummarizerAgent
from core.persistence.parquet_store import ParquetStore


class TestSummarizerPipeline:
    """Testes unitários para o caso de uso de Map-Reduce (SummarizerService)."""

    @pytest.fixture
    def mock_env(self, tmp_path):
        store = ParquetStore(str(tmp_path / "parquet"))
        memory = MagicMock(spec=ISemanticMemory)
        summarizer_agent = MagicMock(spec=OllamaSummarizerAgent)
        master_agent = MagicMock(spec=OllamaMasterAgent)

        service = SummarizerService(
            parquet_store=store,
            semantic_memory=memory,
            summarizer_agent=summarizer_agent,
            master_agent=master_agent,
        )
        return service, store, memory, summarizer_agent, master_agent

    def test_run_empty_articles(self, mock_env):
        service, store, memory, summarizer_agent, master_agent = mock_env
        result = service.run()

        assert result["status"] == "empty"
        assert result["processed_count"] == 0
        summarizer_agent.summarize.assert_not_called()
        master_agent.generate_global_bulletin.assert_not_called()

    def test_run_map_reduce_pipeline_success(self, mock_env):
        service, store, memory, summarizer_agent, master_agent = mock_env

        # Insere artigo bruto no Parquet
        store.append([
            Article(
                url="https://tech.com/article1",
                title="Novo processador quântico",
                content="Conteúdo longo do artigo sobre computação quântica.",
                published_at=datetime.now(timezone.utc),
                source="TechSource",
            )
        ])

        summarizer_agent.summarize.return_value = "Resumo executivo do processador quântico em 3 parágrafos."
        master_agent.generate_global_bulletin.return_value = "📰 Boletim Consolidado do Dia sobre Computação Quântica."

        result = service.run()

        assert result["status"] == "success"
        assert result["processed_count"] == 1
        assert "Boletim Consolidado" in result["bulletin"]

        # Verifica chamada ao agente MAP
        summarizer_agent.summarize.assert_called_once()
        # Verifica armazenamento semântico no ChromaDB
        memory.store_individual_summary.assert_called_once()
        # Verifica chamada ao agente REDUCE
        master_agent.generate_global_bulletin.assert_called_once()

        # Verifica persistência no Parquet
        summaries = store.get_latest_summaries()
        assert len(summaries) == 1
        bulletins = store.get_all_bulletins()
        assert len(bulletins) == 1

