from enum import Enum


class NewsIntentType(str, Enum):
    """Categorias determinísticas de intenção para o orquestrador do News Summarizer.

    Seguindo a Clean Architecture e os estágios desacoplados:
    - FACTUAL_QUERY: Busca ou pergunta pontual sobre fatos de notícias já arquivadas.
    - TRENDS_SUMMARY: Solicitação de panorama diário, boletim consolidado ou tendências globais.
    - OUT_OF_DOMAIN: Pergunta fora do escopo de notícias ou tecnologia (ex: receitas, piadas).
    - HUMAN_INTERVENTION: Solicitação explícita de atendente humano ou correção de diretriz (HITL).
    """

    FACTUAL_QUERY = "FACTUAL_QUERY"
    TRENDS_SUMMARY = "TRENDS_SUMMARY"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    HUMAN_INTERVENTION = "HUMAN_INTERVENTION"

