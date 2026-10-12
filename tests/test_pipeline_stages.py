import json
from unittest.mock import MagicMock
import pytest

from core.application.services.gatekeeper import GatekeeperService
from core.application.services.query_extractor import QueryExtractorService
from core.application.services.grounded_oracle import GroundedOracleService
from core.domain.models.intents import NewsIntentType
from core.domain.models.schemas import NewsFilterSchema
from core.domain.ports import ISemanticMemory
from core.infrastructure.adapters.ollama_client import OllamaClient, OllamaClientError


class TestGatekeeperService:
    """Testes unitários para o GatekeeperService (Estágio 1)."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock(spec=OllamaClient)

    def test_classify_factual_query(self, mock_client):
        mock_client.generate.return_value = '{"intent": "FACTUAL_QUERY"}'
        service = GatekeeperService(client=mock_client)

        intent = service.classify("O que a NVIDIA lançou?")
        assert intent == NewsIntentType.FACTUAL_QUERY

    def test_classify_trends_summary(self, mock_client):
        mock_client.generate.return_value = '{"intent": "TRENDS_SUMMARY"}'
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Me dê o panorama de tendências")
        assert intent == NewsIntentType.TRENDS_SUMMARY

    def test_classify_out_of_domain(self, mock_client):
        mock_client.generate.return_value = '{"intent": "OUT_OF_DOMAIN"}'
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Como fazer um bolo de chocolate?")
        assert intent == NewsIntentType.OUT_OF_DOMAIN

    def test_classify_human_intervention(self, mock_client):
        mock_client.generate.return_value = '{"intent": "HUMAN_INTERVENTION"}'
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Quero falar com um humano")
        assert intent == NewsIntentType.HUMAN_INTERVENTION

    def test_heuristic_guardrail_human_bypass_llm(self, mock_client):
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Preciso falar com um operador humano imediatamente")
        assert intent == NewsIntentType.HUMAN_INTERVENTION
        mock_client.generate.assert_not_called()

    def test_heuristic_guardrail_summary_bypass_llm(self, mock_client):
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Qual o resumo de hoje?")
        assert intent == NewsIntentType.TRENDS_SUMMARY
        mock_client.generate.assert_not_called()

    def test_fallback_on_malformed_json(self, mock_client):
        mock_client.generate.return_value = '{"intent": "QUEBRADO'
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Pergunta qualquer")
        assert intent == NewsIntentType.HUMAN_INTERVENTION

    def test_fallback_on_client_exception(self, mock_client):
        mock_client.generate.side_effect = OllamaClientError("Erro de conexão")
        service = GatekeeperService(client=mock_client)

        intent = service.classify("Pergunta qualquer")
        assert intent == NewsIntentType.HUMAN_INTERVENTION

    def test_fallback_on_empty_message(self, mock_client):
        service = GatekeeperService(client=mock_client)
        assert service.classify("") == NewsIntentType.HUMAN_INTERVENTION
        assert service.classify("   ") == NewsIntentType.HUMAN_INTERVENTION


class TestQueryExtractorService:
    """Testes unitários para o QueryExtractorService (Estágio 2)."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock(spec=OllamaClient)

    def test_extract_valid_schema(self, mock_client):
        mock_client.generate.return_value = json.dumps({
            "topico": "inteligencia artificial",
            "palavra_chave": "OpenAI",
            "fonte": "TechCrunch",
            "data_referencia": "2026-10-10",
            "limite": 3,
        })
        service = QueryExtractorService(client=mock_client)

        filters = service.extract("Notícias da TechCrunch sobre OpenAI em 2026-10-10")
        assert isinstance(filters, NewsFilterSchema)
        assert filters.topico == "inteligencia artificial"
        assert filters.palavra_chave == "OpenAI"
        assert filters.fonte == "TechCrunch"
        assert filters.data_referencia == "2026-10-10"
        assert filters.limite == 3
        assert not filters.is_empty()

    def test_extract_empty_on_exception(self, mock_client):
        mock_client.generate.side_effect = Exception("Falha de rede")
        service = QueryExtractorService(client=mock_client)

        filters = service.extract("Qualquer coisa")
        assert filters.is_empty()

    def test_extract_empty_on_empty_input(self, mock_client):
        service = QueryExtractorService(client=mock_client)
        assert service.extract("").is_empty()


class TestGroundedOracleService:
    """Testes unitários para o GroundedOracleService (Estágio 3 e 4)."""

    @pytest.fixture
    def mock_client(self):
        return MagicMock(spec=OllamaClient)

    @pytest.fixture
    def mock_memory(self):
        return MagicMock(spec=ISemanticMemory)

    def test_grounded_answer_success(self, mock_client, mock_memory):
        mock_memory.recall_context.return_value = [
            {"content": "A empresa X lançou o modelo Y com 70B de parâmetros.", "metadata": {"source": "TechNews"}}
        ]
        mock_client.generate.return_value = "O modelo Y possui 70B de parâmetros."

        service = GroundedOracleService(memory=mock_memory, client=mock_client)
        resp, contexts, is_refusal = service.answer("Quantos parâmetros tem o modelo Y?")

        assert resp == "O modelo Y possui 70B de parâmetros."
        assert len(contexts) == 1
        assert not is_refusal

    def test_refusal_when_no_memory_contexts(self, mock_client, mock_memory):
        mock_memory.recall_context.return_value = []

        service = GroundedOracleService(memory=mock_memory, client=mock_client)
        resp, contexts, is_refusal = service.answer("Qual a notícia sobre Z?")

        assert resp == GroundedOracleService.STRICT_REFUSAL_MESSAGE
        assert contexts == []
        assert is_refusal
        mock_client.generate.assert_not_called()

    def test_refusal_when_model_states_insufficient_data(self, mock_client, mock_memory):
        mock_memory.recall_context.return_value = [
            {"content": "Artigo sobre biologia marinha.", "metadata": {}}
        ]
        mock_client.generate.return_value = "Não possuo dados suficientes em minha base para responder."

        service = GroundedOracleService(memory=mock_memory, client=mock_client)
        resp, contexts, is_refusal = service.answer("O que a Apple lançou?")

        assert resp == GroundedOracleService.STRICT_REFUSAL_MESSAGE
        assert is_refusal

