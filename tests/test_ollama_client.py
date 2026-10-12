from unittest.mock import MagicMock, patch
import pytest

from core.infrastructure.adapters.ollama_client import OllamaClient, OllamaClientError


class TestOllamaClient:
    """Testes unitários para o OllamaClient (HTTP puro)."""

    def test_default_initialization(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.generate_endpoint == "http://localhost:11434/api/generate"
        assert client.embeddings_endpoint == "http://localhost:11434/api/embeddings"
        assert client.timeout == 60.0

    def test_custom_initialization(self):
        client = OllamaClient(base_url="http://ollama-server:8000/", timeout=15.0)
        assert client.base_url == "http://ollama-server:8000"
        assert client.generate_endpoint == "http://ollama-server:8000/api/generate"
        assert client.timeout == 15.0

    @patch("core.infrastructure.adapters.ollama_client.httpx.post")
    def test_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Texto sintetizado com sucesso."}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        client = OllamaClient()
        result = client.generate(
            prompt="Resuma isto",
            model="llama3.2",
            require_json=True,
            system="Instrução do sistema",
            options={"temperature": 0.0},
        )

        assert result == "Texto sintetizado com sucesso."
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["model"] == "llama3.2"
        assert payload["prompt"] == "Resuma isto"
        assert payload["format"] == "json"
        assert payload["system"] == "Instrução do sistema"
        assert payload["options"] == {"temperature": 0.0}

    @patch("core.infrastructure.adapters.ollama_client.httpx.post")
    def test_generate_error_raises_ollama_client_error(self, mock_post):
        mock_post.side_effect = Exception("Conexão recusada")

        client = OllamaClient()
        with pytest.raises(OllamaClientError) as exc_info:
            client.generate(prompt="Olá")

        assert "Falha na comunicação com a API do Ollama" in str(exc_info.value)

    @patch("core.infrastructure.adapters.ollama_client.httpx.post")
    def test_embeddings_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        client = OllamaClient()
        vec = client.embeddings("Texto para vetorizar")

        assert vec == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

