"""Serviço de Extração de Parâmetros Estruturados de Busca (Estágio 2 do Pipeline).

Converte a consulta em linguagem natural do usuário em uma instância validada de NewsFilterSchema,
garantindo integridade de tipos, restrições e higienização prévia à consulta na memória semântica.
"""

import json
import os
from typing import Optional
from pydantic import ValidationError

from core.domain.models.schemas import NewsFilterSchema
from core.infrastructure.adapters.ollama_client import OllamaClient


QUERY_EXTRACTOR_SYSTEM_PROMPT = """Você é o extrator de parâmetros estruturados para busca de notícias e inteligência de tecnologia.
Analise a mensagem do usuário e extraia exclusivamente os critérios de filtragem em formato JSON.

O JSON gerado deve conter EXCLUSIVAMENTE as seguintes 5 chaves:
- "topico": Tema ou área temática (ex: "inteligencia artificial", "chips", "startups", "computação quântica") ou null
- "palavra_chave": Entidade, empresa, pessoa ou termo específico (ex: "OpenAI", "NVIDIA", "Sam Altman", "Rust") ou null
- "fonte": Veículo jornalístico mencionado (ex: "CNN", "BBC", "TechCrunch") ou null
- "data_referencia": Data mencionada no formato "YYYY-MM-DD" ou null
- "limite": Número inteiro entre 1 e 20 (default: 5)

Regras estritas:
1. Para qualquer informação ausente ou não especificada, defina o valor como null.
2. Em "limite", utilize um número inteiro (default 5 se não especificado).
3. Não inclua nenhuma chave extra além das especificadas.
4. Responda exclusivamente o objeto JSON.

Exemplos:
Entrada: "O que a Apple anunciou sobre o chip M4?"
Resposta: {"topico": "chips", "palavra_chave": "Apple M4", "fonte": null, "data_referencia": null, "limite": 5}

Entrada: "Notícias da BBC sobre inteligência artificial"
Resposta: {"topico": "inteligencia artificial", "palavra_chave": null, "fonte": "BBC", "data_referencia": null, "limite": 5}

Entrada: "O que aconteceu em 2026-10-10?"
Resposta: {"topico": null, "palavra_chave": null, "fonte": null, "data_referencia": "2026-10-10", "limite": 5}

Entrada: "Me mostre os 3 principais artigos sobre Rust"
Resposta: {"topico": "linguagens de programacao", "palavra_chave": "Rust", "fonte": null, "data_referencia": null, "limite": 3}
"""


class QueryExtractorService:
    """Serviço de extração de parâmetros estruturados de notícias (Estágio 2)."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        model: Optional[str] = None,
    ) -> None:
        self._client = client or OllamaClient()
        self._model = model or os.getenv("OLLAMA_MODEL", "llama3.2")

    @property
    def model(self) -> str:
        return self._model

    def extract(self, user_message: str) -> NewsFilterSchema:
        """Extrai os filtros estruturados a partir da consulta do usuário.

        Em caso de erro de validação (ValidationError) ou falha de infraestrutura,
        retorna deterministicamente NewsFilterSchema() vazio.
        """
        if not user_message or not user_message.strip():
            return NewsFilterSchema()

        prompt = f'Entrada: "{user_message.strip()}"\nResposta:'

        try:
            raw_response = self._client.generate(
                prompt=prompt,
                model=self._model,
                require_json=True,
                system=QUERY_EXTRACTOR_SYSTEM_PROMPT,
                options={"temperature": 0.0},
            )
            return self._parse_and_validate(raw_response)
        except Exception:
            return NewsFilterSchema()

    def _parse_and_validate(self, raw_response: str) -> NewsFilterSchema:
        """Valida a resposta gerada contra o modelo Pydantic NewsFilterSchema."""
        try:
            data = json.loads(raw_response.strip())
            if not isinstance(data, dict):
                return NewsFilterSchema()

            return NewsFilterSchema(**data)
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError, Exception):
            return NewsFilterSchema()

    def __call__(self, user_message: str) -> NewsFilterSchema:
        return self.extract(user_message)

