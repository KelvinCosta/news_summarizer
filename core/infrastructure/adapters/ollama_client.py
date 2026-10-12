"""Cliente HTTP puro para comunicação com a API do Ollama (Clean Architecture).

Elimina dependência de bibliotecas de terceiros monolíticas (LangChain, LlamaIndex),
executando chamadas HTTP diretas com suporte a gramática JSON forçada a nível de backend
(llama.cpp) e controle de timeout e retry.
"""

import os
from typing import Any, List, Optional

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:  # pragma: no cover
    import requests

    _HAS_HTTPX = False


class OllamaClientError(Exception):
    """Exceção de infraestrutura para falhas de comunicação com o servidor Ollama."""

    pass


class OllamaClient:
    """Cliente HTTP puro para invocação da API de geração e embeddings do Ollama."""

    DEFAULT_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "llama3.2"

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        """Inicializa o cliente com URL base e tempo limite configuráveis.

        Args:
            base_url: URL base do servidor Ollama (ex: 'http://localhost:11434').
                Se omitido, consulta a variável de ambiente OLLAMA_BASE_URL.
            timeout: Tempo limite da requisição HTTP em segundos (default: 60.0).
        """
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL", self.DEFAULT_BASE_URL)
        self._base_url = raw_url.rstrip("/")
        self._generate_endpoint = f"{self._base_url}/api/generate"
        self._embeddings_endpoint = f"{self._base_url}/api/embeddings"
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """Retorna a URL base do serviço Ollama."""
        return self._base_url

    @property
    def generate_endpoint(self) -> str:
        """Retorna o endpoint de geração."""
        return self._generate_endpoint

    @property
    def embeddings_endpoint(self) -> str:
        """Retorna o endpoint de embeddings."""
        return self._embeddings_endpoint

    @property
    def timeout(self) -> float:
        """Retorna o timeout configurado em segundos."""
        return self._timeout

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        require_json: bool = False,
        system: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        """Executa uma requisição de geração textual junto à API do Ollama.

        Args:
            prompt: Texto do prompt enviado para geração.
            model: Nome do modelo de linguagem (default: DEFAULT_MODEL).
            require_json: Se True, injeta `"format": "json"` no payload para forçar
                o backend (llama.cpp) a utilizar amostragem com gramática JSON estrita.
            system: Instrução de sistema (System Prompt) opcional.
            options: Parâmetros adicionais de amostragem (ex: {"temperature": 0.0}).

        Returns:
            String contendo o texto gerado retornado pelo modelo.

        Raises:
            OllamaClientError: Em caso de falha de conexão, timeout ou status HTTP de erro.
        """
        target_model = model or self.DEFAULT_MODEL
        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system

        if require_json:
            payload["format"] = "json"

        if options:
            payload["options"] = options

        try:
            if _HAS_HTTPX:
                response = httpx.post(
                    self._generate_endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
            else:  # pragma: no cover
                response = requests.post(
                    self._generate_endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

            return data.get("response", "")
        except Exception as exc:
            raise OllamaClientError(
                f"Falha na comunicação com a API do Ollama em {self._generate_endpoint}: {exc}"
            ) from exc

    def embeddings(
        self,
        text: str,
        model: str = "nomic-embed-text",
    ) -> List[float]:
        """Gera o vetor de embedding para o texto fornecido via Ollama.

        Args:
            text: Texto de entrada para gerar a representação vetorial.
            model: Nome do modelo de embeddings (default: 'nomic-embed-text').

        Returns:
            Lista de floats representando o vetor gerado.

        Raises:
            OllamaClientError: Em caso de falha de conexão, timeout ou erro retornado.
        """
        payload: dict[str, Any] = {
            "model": model,
            "prompt": text,
        }

        try:
            if _HAS_HTTPX:
                response = httpx.post(
                    self._embeddings_endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
            else:  # pragma: no cover
                response = requests.post(
                    self._embeddings_endpoint,
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

            return data.get("embedding", [])
        except Exception as exc:
            raise OllamaClientError(
                f"Falha ao gerar embeddings no Ollama em {self._embeddings_endpoint}: {exc}"
            ) from exc

