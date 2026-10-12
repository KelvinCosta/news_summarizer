"""Orquestrador Central governado por Máquina de Estados Finita e Event Sourcing (Clean Architecture).

Coordena os 4 estágios do pipeline desacoplado:
1. Classificação de Intenção (GatekeeperService)
2. Extração Estruturada de Parâmetros (QueryExtractorService)
3. Recuperação de Memória Semântica (ISemanticMemory / ChromaDB / Parquet)
4. Síntese Factual Ancorada (GroundedOracleService) com Guardrails e Fail-Fast.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from core.application.services.gatekeeper import GatekeeperService, is_temporal_system_query
from core.application.services.query_extractor import QueryExtractorService
from core.application.services.grounded_oracle import GroundedOracleService
from core.application.services.hybrid_retriever import HybridRetrieverService
from core.domain.models.events import (
    ContextRetrieved,
    DomainEvent,
    GuardrailTriggered,
    HandoffTriggered,
    HumanCorrectionProvidedEvent,
    ImmutableEventStore,
    InferenceFailedEvent,
    IntentClassified,
    ResponseGenerated,
    SearchFiltersExtracted,
    UserMessageReceived,
)
from core.domain.models.intents import NewsIntentType
from core.domain.models.schemas import OrchestratorResult
from core.domain.ports import ISemanticMemory
from core.fsm.states import SystemState
from core.infrastructure.adapters.ollama_client import OllamaClient

from core.persistence.parquet_store import ParquetStore

logger = logging.getLogger(__name__)


class FSMOrchestrator:
    """Orquestrador do News Summarizer governado por FSM e Event Sourcing imutável."""

    OUT_OF_DOMAIN_MESSAGE: str = (
        "Essa pergunta está fora do meu escopo de atuação. Sou um sistema especializado em "
        "notícias, artigos tecnológicos e inteligência consolidada de mercado."
    )

    HUMAN_ESCALATION_MESSAGE: str = (
        "Sua solicitação foi registrada para intervenção humana e ajuste de diretrizes (HITL). "
        "Um operador pode revisar as regras ou o histórico de eventos imutáveis."
    )

    def __init__(
        self,
        semantic_memory: Optional[ISemanticMemory] = None,
        event_store: Optional[ImmutableEventStore] = None,
        parquet_store: Optional[ParquetStore] = None,
        client: Optional[OllamaClient] = None,
        gatekeeper: Optional[GatekeeperService] = None,
        query_extractor: Optional[QueryExtractorService] = None,
        oracle_service: Optional[GroundedOracleService] = None,
        retriever: Optional[HybridRetrieverService] = None,
    ) -> None:
        """Inicializa o orquestrador com injeção de dependências desacopladas."""
        self.current_state: SystemState = SystemState.IDLE
        self.context: Dict[str, Any] = {}

        self._client = client or OllamaClient()
        self._semantic_memory = semantic_memory
        self._event_store = event_store or ImmutableEventStore()
        self._parquet_store = parquet_store

        self._gatekeeper = gatekeeper or GatekeeperService(client=self._client)
        self._query_extractor = query_extractor or QueryExtractorService(client=self._client)
        self._oracle_service = oracle_service or GroundedOracleService(
            memory=self._semantic_memory, client=self._client
        )
        self._retriever = retriever or HybridRetrieverService(
            semantic_memory=self._semantic_memory,
            parquet_store=self._parquet_store,
        )

    @property
    def event_store(self) -> ImmutableEventStore:
        return self._event_store

    def transition_to(self, new_state: SystemState) -> None:
        """Transita deterministicamente para um novo estado da FSM."""
        logger.info(f"Transição de estado: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state

    def handle_message(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> OrchestratorResult:
        """Executa o ciclo completo de processamento de uma mensagem do usuário.

        Coordena os estágios, emite os eventos de domínio no ImmutableEventStore e
        retorna uma instância validada de OrchestratorResult.
        """
        sid = session_id or str(uuid4())
        session_events: List[DomainEvent] = []

        # Registro inicial do evento de mensagem
        msg_event = UserMessageReceived(session_id=sid, text=user_message)
        session_events.append(msg_event)
        self._event_store.append(msg_event)

        # 1. Estágio: Roteamento / Classificação de Intenção
        self.transition_to(SystemState.ROUTING)
        intent = self._gatekeeper.classify(user_message)

        intent_event = IntentClassified(session_id=sid, intent=intent)
        session_events.append(intent_event)
        self._event_store.append(intent_event)

        # Roteamento determinístico por intenção
        if intent == NewsIntentType.HUMAN_INTERVENTION:
            return self._handle_human_intervention(sid, user_message, session_events)

        if intent == NewsIntentType.OUT_OF_DOMAIN:
            return self._handle_out_of_domain(sid, session_events)

        if intent == NewsIntentType.TRENDS_SUMMARY:
            return self._handle_trends_summary(sid, user_message, session_events)

        # Padrão: FACTUAL_QUERY (Modo Oráculo Estrito)
        return self._handle_factual_query(sid, user_message, session_events)

    def _handle_human_intervention(
        self,
        session_id: str,
        user_message: str,
        events: List[DomainEvent],
    ) -> OrchestratorResult:
        """Trata pedidos manuais ou acionamento de suporte humano."""
        self.transition_to(SystemState.HUMAN_INTERVENTION)
        handoff_event = HandoffTriggered(
            session_id=session_id,
            reason="user_requested",
            details="Intervenção solicitada explicitamente pelo usuário",
        )
        events.append(handoff_event)
        self._event_store.append(handoff_event)

        self.transition_to(SystemState.IDLE)
        return OrchestratorResult(
            session_id=session_id,
            response=self.HUMAN_ESCALATION_MESSAGE,
            intent=NewsIntentType.HUMAN_INTERVENTION,
            events=events,
            handoff_triggered=True,
            handoff_reason="user_requested",
        )

    def _handle_out_of_domain(
        self,
        session_id: str,
        events: List[DomainEvent],
    ) -> OrchestratorResult:
        """Trata consultas fora do domínio de notícias e tecnologia."""
        self.transition_to(SystemState.VALIDATING)
        handoff_event = HandoffTriggered(
            session_id=session_id,
            reason="out_of_domain",
            details="Pergunta fora do escopo de notícias",
        )
        events.append(handoff_event)
        self._event_store.append(handoff_event)

        self.transition_to(SystemState.IDLE)
        return OrchestratorResult(
            session_id=session_id,
            response=self.OUT_OF_DOMAIN_MESSAGE,
            intent=NewsIntentType.OUT_OF_DOMAIN,
            events=events,
            handoff_triggered=True,
            handoff_reason="out_of_domain",
        )

    def _handle_trends_summary(
        self,
        session_id: str,
        user_message: str,
        events: List[DomainEvent],
    ) -> OrchestratorResult:
        """Processa pedidos de síntese de tendências diárias ou boletim executivo."""
        self.transition_to(SystemState.RETRIEVING_CONTEXT)

        contexts: List[dict] = []

        # 1. Prioriza Boletim Consolidado recente do Parquet se disponível
        if self._parquet_store:
            try:
                bulletins = self._parquet_store.get_all_bulletins()
                if bulletins:
                    latest_b = bulletins[0]
                    contexts.append({
                        "content": latest_b["content"],
                        "metadata": {
                            "type": "mega_boletim",
                            "processed_at": str(latest_b.get("processed_at", "")),
                            "title": "Boletim Consolidado do Dia",
                        },
                    })
            except Exception:
                pass

        # 2. Recupera resumos semânticos complementares
        if self._semantic_memory:
            limit = 5 - len(contexts)
            if limit > 0:
                try:
                    extra = self._semantic_memory.recall_context(user_message, limit=limit)
                    contexts.extend(extra)
                except Exception:
                    pass

        retrieval_event = ContextRetrieved(
            session_id=session_id,
            result_count=len(contexts),
            strategy="semantic_trends",
        )
        events.append(retrieval_event)
        self._event_store.append(retrieval_event)

        self.transition_to(SystemState.INFERENCING)
        response_text, retrieved_contexts, is_refusal = self._oracle_service.answer(
            user_message,
            provided_contexts=contexts if contexts else None,
            is_summary_mode=True,
        )

        self.transition_to(SystemState.VALIDATING)
        if is_refusal:
            guardrail_event = GuardrailTriggered(
                session_id=session_id,
                guardrail_name="StrictFactualRefusal",
                reason="Sem notícias suficientes para sintetizar panorama de tendências",
            )
            events.append(guardrail_event)
            self._event_store.append(guardrail_event)

        resp_event = ResponseGenerated(
            session_id=session_id,
            response=response_text,
            model=self._oracle_service.model,
        )
        events.append(resp_event)
        self._event_store.append(resp_event)

        self.transition_to(SystemState.COMPLETED)
        self.transition_to(SystemState.IDLE)

        return OrchestratorResult(
            session_id=session_id,
            response=response_text,
            intent=NewsIntentType.TRENDS_SUMMARY,
            events=events,
            contexts=retrieved_contexts,
            handoff_triggered=is_refusal,
            handoff_reason="no_data" if is_refusal else None,
        )

    def _generate_temporal_response(self) -> tuple[str, List[Dict[str, Any]]]:
        """Gera resposta contextualizada para consultas temporais do sistema com integridade do Lake."""
        from datetime import datetime
        now_dt = datetime.now()
        dias_semana = [
            "segunda-feira", "terça-feira", "quarta-feira",
            "quinta-feira", "sexta-feira", "sábado", "domingo"
        ]
        dia_semana_str = dias_semana[now_dt.weekday()]
        meses = [
            "janeiro", "fevereiro", "março", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"
        ]
        mes_str = meses[now_dt.month - 1]
        data_formatada = f"{now_dt.day} de {mes_str} de {now_dt.year} ({dia_semana_str})"

        info_base = ""
        last_url = ""
        if self._parquet_store:
            try:
                sums = self._parquet_store.get_latest_summaries(limit=1)
                if sums:
                    dt_proc = sums[0].get("processed_at")
                    dt_proc_str = dt_proc.strftime("%d/%m/%Y") if hasattr(dt_proc, "strftime") else str(dt_proc)[:10]
                    last_url = sums[0].get("original_url") or sums[0].get("url") or ""
                    info_base = f" Minha base de inteligência contém notícias arquivadas e atualizadas até {dt_proc_str}."
            except Exception:
                pass

        resp_text = f"Hoje é {data_formatada}.{info_base}"
        context = [{
            "content": resp_text,
            "metadata": {
                "type": "relogio_do_sistema",
                "title": "Relógio do Sistema e Frescor do Lake",
                "source_url": last_url or "system://clock",
                "processed_at": now_dt.strftime("%d/%m/%Y"),
            }
        }]
        return resp_text, context

    def _handle_factual_query(
        self,
        session_id: str,
        user_message: str,
        events: List[DomainEvent],
    ) -> OrchestratorResult:
        """Processa perguntas factuais pontuais no Modo Oráculo Estrito."""
        # 1. Verificação rápida de meta-consulta temporal do sistema (Zero GPU)
        if is_temporal_system_query(user_message):
            response_text, contexts = self._generate_temporal_response()

            self.transition_to(SystemState.RETRIEVING_CONTEXT)
            retrieval_event = ContextRetrieved(
                session_id=session_id,
                result_count=len(contexts),
                strategy="system_clock",
            )
            events.append(retrieval_event)
            self._event_store.append(retrieval_event)

            self.transition_to(SystemState.INFERENCING)
            resp_event = ResponseGenerated(
                session_id=session_id,
                response=response_text,
                model="system_clock",
            )
            events.append(resp_event)
            self._event_store.append(resp_event)

            self.transition_to(SystemState.COMPLETED)
            self.transition_to(SystemState.IDLE)

            return OrchestratorResult(
                session_id=session_id,
                response=response_text,
                intent=NewsIntentType.FACTUAL_QUERY,
                events=events,
                contexts=contexts,
                handoff_triggered=False,
                handoff_reason=None,
            )

        # Estágio 2: Extração de filtros estruturados
        filters = self._query_extractor.extract(user_message)
        filter_event = SearchFiltersExtracted(session_id=session_id, filters=filters)
        events.append(filter_event)
        self._event_store.append(filter_event)

        # Estágio 3: Recuperação Híbrida de Contexto (Léxico + Vetorial + Reranking)
        self.transition_to(SystemState.RETRIEVING_CONTEXT)
        limit = filters.limite if filters.limite else 5
        contexts = self._retriever.retrieve(
            query=user_message,
            filters=filters,
            limit=limit,
        )

        retrieval_event = ContextRetrieved(
            session_id=session_id,
            result_count=len(contexts),
            strategy="hybrid_rag",
        )
        events.append(retrieval_event)
        self._event_store.append(retrieval_event)

        # Estágio 4: Inferência e Validação Factual Grounded
        self.transition_to(SystemState.INFERENCING)
        response_text, retrieved_contexts, is_refusal = self._oracle_service.answer(
            user_message,
            provided_contexts=contexts,
        )

        self.transition_to(SystemState.VALIDATING)
        if is_refusal:
            guardrail_event = GuardrailTriggered(
                session_id=session_id,
                guardrail_name="StrictOracleGuardrail",
                reason="Fatos solicitados ausentes na base ou divergentes",
            )
            events.append(guardrail_event)
            self._event_store.append(guardrail_event)

        resp_event = ResponseGenerated(
            session_id=session_id,
            response=response_text,
            model=self._oracle_service.model,
        )
        events.append(resp_event)
        self._event_store.append(resp_event)

        self.transition_to(SystemState.COMPLETED)
        self.transition_to(SystemState.IDLE)

        return OrchestratorResult(
            session_id=session_id,
            response=response_text,
            intent=NewsIntentType.FACTUAL_QUERY,
            events=events,
            contexts=retrieved_contexts,
            handoff_triggered=is_refusal,
            handoff_reason="no_data" if is_refusal else None,
        )

    def process_request(self, user_input: str) -> OrchestratorResult:
        """Compatibilidade com o método legado: executa processamento da requisição."""
        return self.handle_message(user_input)

    def trigger_fail_fast(self, reason: str = "Alucinação detectada") -> None:
        """Abre o circuito instantaneamente (Fail-Fast) e transita para intervenção humana."""
        logger.error(f"Guardrail acionado! Fail-Fast disparado: {reason}")
        self.transition_to(SystemState.HUMAN_INTERVENTION)
        failed_event = InferenceFailedEvent(
            session_id="system_fail_fast",
            original_prompt="",
            error_reason=reason,
        )
        self._event_store.append(failed_event)

    def handle_human_correction(self, correction_data: str, session_id: str = "123") -> None:
        """Aplica correção humana manual para fechar o loop do circuito (HITL)."""
        if self.current_state != SystemState.HUMAN_INTERVENTION:
            raise ValueError("Correções só podem ser aplicadas no estado HUMAN_INTERVENTION.")
        logger.info("Processando correção humana e registrando no Event Store...")
        correction_event = HumanCorrectionProvidedEvent(
            session_id=session_id,
            failed_event_id="system_fail_fast",
            correction_text=correction_data,
        )
        self._event_store.append(correction_event)
        self.transition_to(SystemState.IDLE)

    def __call__(
        self, user_message: str, session_id: Optional[str] = None
    ) -> OrchestratorResult:
        return self.handle_message(user_message, session_id=session_id)
