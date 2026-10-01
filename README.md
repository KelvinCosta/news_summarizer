# News Summarizer - News Summarizer (On-Premise SLM Architecture)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Ollama](https://img.shields.io/badge/ollama-local-orange.svg)

News Summarizer é um sistema autônomo de inteligência de notícias baseado em *Small Language Models* (SLMs) projetado para rodar 100% On-Premise (local) em hardwares de consumo (ex: RTX 3060 12GB). 

O projeto adota princípios avançados de Engenharia de Software, incluindo **CQRS, Event Sourcing, Funis O(1) e Arquitetura Map-Reduce**, para extrair, filtrar, resumir e vetorizar informações não-estruturadas da internet de forma determinística e com privacidade absoluta.

## 🧠 Arquitetura

O sistema foi concebido para ser blindado contra alucinações e estouros de memória (OOM), dividindo o fluxo de processamento em etapas assíncronas isoladas:

1. **Ingestão e Funil O(1) (Filtro Bayesiano e Lexical):**
   - Captura dados de feeds RSS.
   - Utiliza **SimHash (Hamming Distance)** para descartar redundâncias jornalísticas em tempo constante O(1).
   - Utiliza um **Classificador Naive Bayes** (*Cold Start* via `vocabulary_seed.json`) para barrar tópicos irrelevantes antes que consumam ciclos da GPU.

2. **Event Sourcing (Parquet):**
   - A verdade bruta e todos os estados intermediários (Notícias -> Resumos Individuais -> Mega Boletins) são salvos de forma imutável (*append-only*) na camada de persistência via **Polars** (arquivos `.parquet`).

3. **Map-Reduce (Agentes Locais via Ollama):**
   - **MAP (Agente 1):** Lê artigos extensos e extrai a síntese (3 parágrafos).
   - **REDUCE (Agente Master):** Coleta todos os resumos do dia e consolida um Mega-Boletim Executivo cruzando as tendências.

4. **Hipocampo RAG (ChromaDB):**
   - Atua como o *Read Model* da arquitetura CQRS.
   - Vetoriza a inteligência destilada usando o modelo leve `nomic-embed-text`.
   - Permite consultas semânticas no "Modo Oráculo Estrito": o LLM (`llama3.2`) responde a perguntas cirúrgicas com 100% de rastreabilidade (auditoria da origem) e sem alucinações baseadas em pesos de pré-treinamento da internet.

## 🚀 Instalação e Execução

### Pré-requisitos
* Python 3.10+
* [Ollama](https://ollama.com/) rodando localmente.
* Modelos necessários no Ollama:
  ```bash
  ollama pull llama3.2
  ollama pull nomic-embed-text
  ```

### Como configurar
1. Clone o repositório e crie um ambiente virtual:
   ```bash
   git clone https://github.com/SEU_USUARIO/news_summarizer.git
   cd news_summarizer
   python -m venv .venv
   source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

*(Certifique-se de preencher o arquivo `vocabulary_seed.json` com suas palavras de interesse para o Filtro Bayesiano).*

### Como rodar a esteira
A arquitetura foi desenhada em módulos isolados que podem ser orquestrados por uma *cronjob*:

```bash
# 1. Busca feeds, aplica o funil matemático O(1) e salva a verdade bruta no Parquet
python test_ingestion.py

# 2. Executa a Injeção de LLM (Map-Reduce) e retroalimenta o ChromaDB
python test_summarizer.py

# 3. Interage com a Interface (Menu CLI) para ler boletins ou consultar o RAG
python view_bulletins.py
```

## 🤝 Contribuição
Este projeto tem arquitetura aberta sob licença MIT. A intenção é amadurecer a estrutura de Agentes SLM Locais. Contribuições na construção de interfaces Web (React/Vue), conteinerização (Docker) e refatoração de Workers assíncronos são muito bem-vindas!

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.
