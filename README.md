# News Summarizer - On-Premise SLM Architecture

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Ollama](https://img.shields.io/badge/ollama-local-orange.svg)
![Clean Architecture](https://img.shields.io/badge/architecture-clean-green.svg)
![Tests](https://img.shields.io/badge/tests-41%20passed-brightgreen.svg)

News Summarizer é um sistema autônomo de inteligência de notícias baseado em *Small Language Models* (SLMs) projetado para rodar 100% On-Premise (local) em hardwares de consumo (ex: RTX 3060 12GB). 

O projeto adota princípios avançados de Engenharia de Software, incluindo **Clean Architecture desacoplada de frameworks monolíticos, CQRS, Event Sourcing com Pydantic v2, Funis O(1), FSM (Finite State Machine) e Arquitetura Map-Reduce**, para extrair, filtrar, resumir e vetorizar informações da internet de forma determinística e com privacidade absoluta.

---

## 🏗️ Arquitetura em Estágios (Clean Architecture)

Seguindo os padrões comprovados de engenharia de software com SLMs, o pipeline abandona wrappers monolíticos e adota estágios especializados e fortemente tipados:

```mermaid
flowchart TD
    User([Usuário / Consulta]) --> Orchestrator[FSMOrchestrator\nMáquina de Estados + Event Sourcing]

    subgraph Estágio 1 - Classificação
        Orchestrator --> Gatekeeper[GatekeeperService\nRoteamento Determinístico]
        Gatekeeper --> OllamaClient[Ollama HTTP Client\nllama3.2]
    end

    Gatekeeper -->|NewsIntentType| Decision{Intenção?}

    Decision -->|FACTUAL_QUERY| Extractor[Estágio 2: QueryExtractorService\nNewsFilterSchema Pydantic]
    Decision -->|TRENDS_SUMMARY| OracleTrends[Estágio 3 e 4: Grounded Oracle\nSíntese de Tendências]
    Decision -->|OUT_OF_DOMAIN| OOD[Resposta Institucional de Escopo]
    Decision -->|HUMAN_INTERVENTION| Escalation[Fail-Fast / Transbordo para HITL]

    subgraph Estágio 3 - Hipocampo e Memória
        Extractor --> Memory[ISemanticMemory\nChromaDB nomic-embed-text]
        Memory --> GroundedAnswering[Estágio 4: GroundedOracleService\nValidação Factual Estrita]
    end

    OracleTrends --> EventStore[(ImmutableEventStore\nEvent Sourcing Imutável)]
    GroundedAnswering --> EventStore
    OOD --> EventStore
    Escalation --> EventStore

    EventStore --> Result([Resposta Final + Domain Events])
```

### Os 4 Estágios do Pipeline

1. **Estágio 1 — Gatekeeper (`GatekeeperService`):**
   - Classifica deterministicamente a entrada em 4 categorias (`FACTUAL_QUERY`, `TRENDS_SUMMARY`, `OUT_OF_DOMAIN`, `HUMAN_INTERVENTION`).
   - Prompt few-shot otimizado com formatação JSON forçada no backend (`format: json`) e guardrails heurísticos pré-LLM para fail-fast instantâneo.
   - Fallback determinístico de segurança para `HUMAN_INTERVENTION` em caso de erro de conexão ou timeout.

2. **Estágio 2 — Extrator Estruturado (`QueryExtractorService`):**
   - Transforma a linguagem natural do usuário em filtros validados via Pydantic v2 (`NewsFilterSchema`): tópico, palavra-chave, fonte, data de referência e limite.
   - Elimina alucinações e garante consultas higienizadas.

3. **Estágio 3 — Memória Semântica (`ChromaDBAdapter` via `ISemanticMemory`):**
   - Atua como o *Read Model* da arquitetura CQRS.
   - Vetoriza a inteligência destilada usando o modelo leve `nomic-embed-text`.
   - Recuperação isolada do contexto semântico, eliminando contaminação cruzada.

4. **Estágio 4 — Modo Oráculo Estrito & Guardrails Factuais (`GroundedOracleService`):**
   - Injeta estritamente o contexto das notícias na instrução do modelo.
   - Guardrail determinístico contra alucinação: se os fatos não constarem na base ou a busca vetorial estiver vazia, aciona a recusa segura canônica (*"Não possuo dados suficientes em minha base interna para responder a esta pergunta."*) sem inventar informações.

5. **Event Sourcing Auditável (`ImmutableEventStore`):**
   - Todos os eventos (`UserMessageReceived`, `IntentClassified`, `SearchFiltersExtracted`, `ContextRetrieved`, `ResponseGenerated`, `GuardrailTriggered`, `HandoffTriggered`, `HumanCorrectionProvidedEvent`) são imutáveis (`frozen=True`) e registrados para auditoria de ponta a ponta.

---

## ⚡ Funil O(1) e Map-Reduce

1. **Ingestão e Funil O(1) (Filtro Bayesiano e Lexical):**
   - Captura dados de feeds RSS.
   - Utiliza **SimHash (Hamming Distance)** para descartar redundâncias jornalísticas em tempo constante O(1).
   - Utiliza um **Classificador Naive Bayes** (*Cold Start* via `vocabulary_seed.json`) para barrar tópicos irrelevantes antes que consumam ciclos de inferência.

2. **Event Sourcing em Parquet:**
   - A verdade bruta e todos os estados intermediários (Notícias -> Resumos Individuais -> Mega Boletins) são salvos de forma imutável (*append-only*) na camada de persistência via **Polars** (arquivos `.parquet`).

3. **Map-Reduce (Agentes SLM Locais):**
   - **MAP (`OllamaSummarizerAgent`):** Lê artigos extensos e extrai sínteses de 3 parágrafos.
   - **REDUCE (`OllamaMasterAgent`):** Coleta resumos do dia e consolida o Mega-Boletim Executivo cruzando tendências.

---

## 🚀 Instalação e Execução

### Pré-requisitos
* Python 3.12+
* [Ollama](https://ollama.com/) rodando localmente.
* Modelos necessários no Ollama:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### Instalação
```bash
git clone https://github.com/SEU_USUARIO/news_summarizer.git
cd news_summarizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configurações de Feeds e Preferências

O projeto utiliza modelos de exemplo para manter as configurações pessoais fora do versionamento:

1. **Feeds RSS (`feeds.json`):**
   - Copie o arquivo modelo:
     ```bash
     cp feeds.example.json feeds.json
     ```
   - Adicione suas URLs de feeds RSS no formato de array JSON:
     ```json
     [
       "https://rss.cnn.com/rss/edition_technology.rss",
       "https://feeds.bbci.co.uk/news/technology/rss.xml"
     ]
     ```

2. **Filtro Bayesiano (`vocabulary_seed.json`):**
   - Personalize os pesos positivos e negativos para triagem matemática *Cold Start*:
     ```bash
     cp vocabulary_seed.example.json vocabulary_seed.json
     ```

### Como Rodar

#### 1. Iniciar o Sistema (Menu Principal Unificado)
```bash
python main.py
```
O terminal apresentará um menu interativo completo:
- **[1]** Ingestão de Notícias & Funil O(1) (RSS -> SimHash -> Naive Bayes)
- **[2]** Executar Pipeline Map-Reduce (Resumos SLM + Vetorização + Boletim)
- **[3]** Visualizar Mega-Boletins por Data (Visão Executiva)
- **[4]** Visualizar Boletins Parciais (Sessões Individuais)
- **[5]** Conversar com o Oráculo RAG (Modo Estrito Anti-Alucinação)
- **[6]** Trilha de Auditoria (Event Sourcing da Sessão)
- **[0]** Sair

#### 2. Executar a Suíte de Testes Unitários
```bash
pytest -v
```

#### 3. Executar Bateria de Medição e Benchmark
```bash
python tests/medir_pipeline.py --reps 1
```

---

## 🤝 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
