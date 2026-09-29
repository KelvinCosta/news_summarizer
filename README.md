# News Summarizer

An autonomous agent system focused on processing confidential data with an on-premise architecture. The system is designed to run locally using Ollama as the inference engine on consumer-grade hardware (e.g., 12GB VRAM), utilizing Hyper-specialized Small Language Models (SLMs) loaded sequentially to prevent Out-of-Memory (OOM) errors and context window overflows.

## Architecture

This project is built around strict architectural patterns:

*   **Finite State Machine (FSM) Orchestration:** The flow is governed by a rigid orchestrator ensuring immutable transitions through routing, retrieval, inference, and validation, preventing infinite loops between agents.
*   **CQRS and Event Sourcing:** Memory management is strictly divided. An *Immutable Log* (Event Sourcing) acts as append-only for inference failures and human corrections. The vector database acts as the *Read Model* (CQRS), storing consolidated heuristics.
*   **Fail-Fast and Human-in-the-Loop (HITL):** A guardrail module instantly opens the circuit (zero retries) upon detecting hallucinations, returning control to the user.
*   **Asymmetric Tagging:** Multitenant isolation via directional inheritance tags in the vector DB to prevent data bleeding across sessions.

## Prerequisites

*   Python 3.10+
*   Ollama (must be installed and running locally for Phase 2)

## Setup and Execution

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd news_summarizer
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the basic architecture simulation (Sprint 1):**
    To run the base FSM orchestrator and CQRS simulation:
    ```bash
    python3 main.py
    ```

5.  **Run the Ingestion and O(1) Funnel (Sprint 2):**
    To test the RSS fetching, Parquet Event Store, SimHash and Bayesian filters:
    ```bash
    python3 test_ingestion.py
    ```
    *Note: This script will generate a `vocabulary_seed.json` file on its first run and a `data/parquet/` directory to store the accepted articles.*
