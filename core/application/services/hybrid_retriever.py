"""Serviço de Recuperação Híbrida de Contexto (Estágio 3 do Pipeline).

Combina busca léxica de alta precisão (Palavra-chave/Entidade exata em Parquet Data Lake)
com busca semântica vetorial (ChromaDB / ISemanticMemory), aplicando fusão
e reranking para mitigar a diluição semântica de entidades (Entity Drowning).
"""

from typing import Any, Dict, List, Optional
from core.domain.ports import ISemanticMemory
from core.domain.models.schemas import NewsFilterSchema
from core.persistence.parquet_store import ParquetStore


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

        # Heurística complementar caso o extractor não tenha isolado palavra-chave
        if not keywords_to_check and query:
            words = query.strip().split()
            stop_starters = {
                "me", "fale", "o", "a", "os", "as", "quem", "qual", "quais",
                "como", "onde", "quando", "sobre", "noticias", "notícias", "da", "do", "de"
            }
            candidates = [
                w for i, w in enumerate(words)
                if (i > 0 or w.lower() not in stop_starters)
                and len(w) > 3
                and w.lower() not in stop_starters
            ]
            if candidates:
                keywords_to_check.append(" ".join(candidates))

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

            # Expansão vetorial por palavra_chave caso ainda tenhamos vagas no limite
            if filters and filters.palavra_chave and len(contexts) < limit:
                try:
                    kw_results = self._memory.recall_context(filters.palavra_chave, limit=limit)
                    for kv in kw_results:
                        kv_meta = kv.get("metadata", {})
                        kv_meta.setdefault("match_strategy", "semantic_vector_keyword")
                        kv["metadata"] = kv_meta
                        _add_context(kv)
                except Exception:
                    pass

        return contexts[:limit]

