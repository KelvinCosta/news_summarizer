"""Serviço de Recuperação Híbrida de Contexto (Estágio 3 do Pipeline).

Combina busca léxica de alta precisão (Palavra-chave/Entidade exata em Parquet Data Lake)
com busca semântica vetorial (ChromaDB / ISemanticMemory), aplicando fusão
e reranking para mitigar a diluição semântica de entidades (Entity Drowning).
"""

import re
from typing import Any, Dict, List, Optional
from core.domain.ports import ISemanticMemory
from core.domain.models.schemas import NewsFilterSchema
from core.persistence.parquet_store import ParquetStore


BILINGUAL_ENTITY_MAP = {
    "casa branca": ["white house", "casa branca"],
    "white house": ["casa branca", "white house"],
    "eua": ["us", "usa", "united states", "estados unidos"],
    "estados unidos": ["us", "usa", "united states", "eua"],
    "inteligencia artificial": ["ai", "ia", "artificial intelligence", "inteligência artificial"],
    "inteligência artificial": ["ai", "ia", "artificial intelligence", "inteligencia artificial"],
    "uniao europeia": ["eu", "european union", "união europeia"],
    "união europeia": ["eu", "european union", "uniao europeia"],
    "reino unido": ["uk", "united kingdom"],
}


class HybridRetrieverService:
    """Recuperador híbrido de notícias (Léxico + Vetorial + Reranking)."""

    def __init__(
        self,
        semantic_memory: Optional[ISemanticMemory] = None,
        parquet_store: Optional[ParquetStore] = None,
    ) -> None:
        self._memory = semantic_memory
        self._parquet_store = parquet_store

    def retrieve(
        self,
        query: str,
        filters: Optional[NewsFilterSchema] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recupera contextos relevantes unificando casamento léxico e proximidade semântica.

        Estratégia:
        1. Identifica palavras-chave prioritárias (do schema de filtros ou da query).
        2. Realiza varredura léxica exata no Parquet Lake (resumos e boletins).
        3. Realiza busca vetorial no Hipocampo (ChromaDB).
        4. Funde e reordena os contextos, priorizando menções exatas da entidade.
        5. Aplica deduplicação por URL/conteúdo.
        """
        contexts: List[Dict[str, Any]] = []
        seen_identifiers: set[str] = set()

        def _add_context(item: Dict[str, Any], priority_push: bool = False) -> None:
            meta = item.get("metadata", {})
            src = meta.get("source_url") or meta.get("title") or item.get("content", "")[:50]
            if src in seen_identifiers:
                return
            seen_identifiers.add(src)
            if priority_push:
                contexts.insert(0, item)
            else:
                contexts.append(item)

        # 1. Identificação de termos e entidades candidatas
        keywords_to_check: List[str] = []
        if filters:
            if filters.palavra_chave and filters.palavra_chave.strip():
                keywords_to_check.append(filters.palavra_chave.strip())
            if (
                filters.topico
                and filters.topico.strip()
                and filters.topico.strip() not in keywords_to_check
            ):
                keywords_to_check.append(filters.topico.strip())

        # Varredura direta na query por entidades do mapa bilíngue
        query_norm = query.lower()
        for entity_key, aliases in BILINGUAL_ENTITY_MAP.items():
            if entity_key in query_norm:
                if entity_key not in [k.lower() for k in keywords_to_check]:
                    keywords_to_check.append(entity_key)
                for alias in aliases:
                    if alias.lower() not in [k.lower() for k in keywords_to_check]:
                        keywords_to_check.append(alias)

        # Heurística complementar caso o extractor não tenha isolado palavra-chave
        if not keywords_to_check and query:
            clean_q = re.sub(r"[^\w\s-]", " ", query)
            words = clean_q.strip().split()
            stop_starters = {
                "me", "fale", "o", "a", "os", "as", "quem", "qual", "quais",
                "como", "onde", "quando", "sobre", "noticias", "notícias", "da", "do", "de",
                "em", "na", "no", "por", "para", "com", "um", "uma"
            }
            candidates = [
                w for w in words
                if len(w) > 2
                and w.lower() not in stop_starters
            ]
            if candidates:
                keywords_to_check.append(" ".join(candidates))

        # Expansão bilíngue e limpeza de pontuação das palavras-chave
        expanded: List[str] = []
        for kw in keywords_to_check:
            kw_clean = re.sub(r"[^\w\s-]", "", kw).strip()
            if kw_clean and kw_clean not in expanded:
                expanded.append(kw_clean)
            kw_lower = kw_clean.lower()
            for entity_key, aliases in BILINGUAL_ENTITY_MAP.items():
                if entity_key in kw_lower or kw_lower in entity_key:
                    for alias in aliases:
                        if alias not in expanded and alias.lower() != kw_lower:
                            expanded.append(alias)
        keywords_to_check = expanded

        # 2. Busca Léxica no Parquet Lake (Alta Precisão para Entidades)
        if self._parquet_store and keywords_to_check:
            for kw in keywords_to_check:
                # Busca em resumos individuais
                matched_summaries = self._parquet_store.search_summaries_by_keyword(kw, limit=limit)
                for s in matched_summaries:
                    _add_context({
                        "content": s.get("summary", ""),
                        "metadata": {
                            "type": "individual_summary",
                            "source_url": s.get("original_url") or s.get("url", ""),
                            "title": s.get("title", ""),
                            "processed_at": str(s.get("processed_at", "")),
                            "match_strategy": "lexical_exact",
                        },
                    })

                # Busca em boletins consolidados
                matched_bulletins = self._parquet_store.search_bulletins_by_keyword(kw, limit=2)
                for b in matched_bulletins:
                    _add_context({
                        "content": b.get("content", ""),
                        "metadata": {
                            "type": "mega_boletim",
                            "title": f"Boletim Consolidado ({b.get('topic', 'Geral')})",
                            "processed_at": str(b.get("processed_at", "")),
                            "match_strategy": "lexical_bulletin",
                        },
                    })

        # 3. Busca Semântica Vetorial no Hipocampo (ChromaDB)
        if self._memory:
            try:
                vector_results = self._memory.recall_context(query, limit=limit)
                for v in vector_results:
                    v_meta = v.get("metadata", {})
                    v_meta.setdefault("match_strategy", "semantic_vector")
                    v["metadata"] = v_meta
                    _add_context(v)
            except Exception:
                pass

            # Expansão vetorial por termos e aliases bilíngues caso ainda tenhamos vagas no limite
            if len(contexts) < limit and keywords_to_check:
                for kw in keywords_to_check:
                    if len(contexts) >= limit:
                        break
                    try:
                        kw_results = self._memory.recall_context(kw, limit=limit)
                        for kv in kw_results:
                            kv_meta = kv.get("metadata", {})
                            kv_meta.setdefault("match_strategy", "semantic_vector_keyword")
                            kv["metadata"] = kv_meta
                            _add_context(kv)
                    except Exception:
                        pass

        return contexts[:limit]

