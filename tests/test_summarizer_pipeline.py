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

    def test_run_map_reduce_with_limit(self, mock_env):
        service, store, memory, summarizer_agent, master_agent = mock_env

        store.append([
            Article(
                url="https://tech.com/article1",
                title="Artigo 1",
                content="Conteúdo 1",
                published_at=datetime.now(timezone.utc),
                source="Source1",
            ),
            Article(
                url="https://tech.com/article2",
                title="Artigo 2",
                content="Conteúdo 2",
                published_at=datetime.now(timezone.utc),
                source="Source2",
            ),
        ])

        summarizer_agent.summarize.return_value = "Resumo válido."
        master_agent.generate_global_bulletin.return_value = "Boletim de teste."

        result = service.run(limit=1)

        assert result["status"] == "success"
        assert result["processed_count"] == 1
        assert summarizer_agent.summarize.call_count == 1

    def test_ollama_summarizer_agent_truncation_and_sanitization(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "Resumo gerado."
        agent = OllamaSummarizerAgent(client=mock_client)

        huge_html_text = "<p>Texto com <b>HTML</b></p> " + ("A" * 6000)
        summary = agent.summarize(huge_html_text)

        assert summary == "Resumo gerado."
        call_args = mock_client.generate.call_args[1]
        prompt = call_args["prompt"]
        # Garante que as tags HTML foram removidas e o texto foi truncado
        assert "<p>" not in prompt
        assert "<b>" not in prompt
        assert len(prompt) < 5500

    def test_run_indexing_without_bulletin(self, mock_env):
        service, store, memory, summarizer_agent, master_agent = mock_env

        store.append([
            Article(
                url="https://tech.com/article1",
                title="Novo processador quântico",
                content="Conteúdo longo do artigo sobre computação quântica.",
                published_at=datetime.now(timezone.utc),
                source="TechSource",
            )
        ])

        summarizer_agent.summarize.return_value = "Resumo executivo do processador."

        result = service.run(generate_bulletin=False)

        assert result["status"] == "success"
        assert result["processed_count"] == 1
        assert result["bulletin"] is None
        summarizer_agent.summarize.assert_called_once()
        memory.store_individual_summary.assert_called_once()
        master_agent.generate_global_bulletin.assert_not_called()

        # Verifica persistência no Parquet
        summaries = store.get_all_summaries()
        assert len(summaries) == 1
        bulletins = store.get_all_bulletins()
        assert len(bulletins) == 0

    def test_generate_top_bulletin(self, mock_env):
        service, store, memory, summarizer_agent, master_agent = mock_env

        # Insere artigo e resumo no Parquet
        store.append([
            Article(
                url="https://tech.com/ai_super",
                title="Super Inteligência e IA",
                content="Conteúdo sobre IA e modelos.",
                published_at=datetime.now(timezone.utc),
                source="TechSource",
            )
        ])
        summarizer_agent.summarize.return_value = "Resumo sobre IA e tecnologia."
        service.run(generate_bulletin=False)

        master_agent.generate_global_bulletin.return_value = "📰 Boletim Top 10 Consolidado."

        result = service.generate_top_bulletin(top_k=5)

        assert result["status"] == "success"
        assert "Boletim Top 10" in result["bulletin"]
        assert result["count"] == 1
        master_agent.generate_global_bulletin.assert_called_once()
        assert len(store.get_all_bulletins()) == 1

    def test_generate_top_bulletin_empty(self, mock_env):
        service, store, memory, summarizer_agent, master_agent = mock_env
        result = service.generate_top_bulletin(top_k=10)
        assert result["status"] == "empty"
        assert result["bulletin"] is None




