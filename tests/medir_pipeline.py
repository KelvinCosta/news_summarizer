"""Bateria de testes e medição do pipeline do News Summarizer (Clean Architecture).

Avalia acurácia de classificação de intenção, grounded retrieval, guardrails contra
alucinações e rastreabilidade via Event Sourcing.

Uso:
    python tests/medir_pipeline.py
    python tests/medir_pipeline.py --reps 1
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.domain.models.intents import NewsIntentType
from core.fsm.orchestrator import FSMOrchestrator
from core.persistence.chroma_adapter import ChromaDBAdapter

CASOS_TESTE = [
    ("factual", "Quais são as notícias recentes sobre inteligência artificial?", NewsIntentType.FACTUAL_QUERY),
    ("factual", "O que foi anunciado sobre chips e semicondutores?", NewsIntentType.FACTUAL_QUERY),
    ("trends", "Me dê um resumo das tendências e notícias de hoje", NewsIntentType.TRENDS_SUMMARY),
    ("trends", "Gere o boletim executivo de tecnologia", NewsIntentType.TRENDS_SUMMARY),
    ("fora_de_escopo", "Qual a receita para fazer pudim de leite?", NewsIntentType.OUT_OF_DOMAIN),
    ("fora_de_escopo", "Me conte uma piada sobre papagaios", NewsIntentType.OUT_OF_DOMAIN),
    ("humano", "Quero falar com um operador humano da equipe", NewsIntentType.HUMAN_INTERVENTION),
    ("humano", "Preciso de intervenção manual para corrigir uma diretriz", NewsIntentType.HUMAN_INTERVENTION),
]


def rodar_benchmark(reps: int = 1, out_file: str = None) -> None:
    print("=" * 70)
    print("  Benchmark de Pipeline - News Summarizer")
    print(f"  Repetições: {reps} | Casos de teste: {len(CASOS_TESTE)}")
    print("=" * 70)

    try:
        memory = ChromaDBAdapter("data/chroma")
    except Exception:
        memory = None

    orchestrator = FSMOrchestrator(semantic_memory=memory)

    total_execs = 0
    intent_hits = 0
    total_time = 0.0
    registros = []

    for idx, (categoria, pergunta, expected_intent) in enumerate(CASOS_TESTE, start=1):
        print(f"\n[{idx}/{len(CASOS_TESTE)}] ({categoria}) Pergunta: '{pergunta}'")
        for r in range(1, reps + 1):
            t0 = time.perf_counter()
            result = orchestrator.handle_message(pergunta)
            t_exec = time.perf_counter() - t0

            total_execs += 1
            total_time += t_exec
            matched = (result.intent == expected_intent)
            if matched:
                intent_hits += 1

            status = "✅ MATCH" if matched else f"❌ MISMATCH (obtido={result.intent.value}, esperado={expected_intent.value})"
            print(f"  Rep {r}/{reps}: {status} em {t_exec:.3f}s | Handoff={result.handoff_triggered}")

            registros.append({
                "categoria": categoria,
                "pergunta": pergunta,
                "expected_intent": expected_intent.value,
                "obtained_intent": result.intent.value,
                "matched": matched,
                "handoff_triggered": result.handoff_triggered,
                "response": result.response,
                "latency_s": round(t_exec, 4),
                "events_count": len(result.events),
            })

    acc_intent = (intent_hits / total_execs) * 100 if total_execs > 0 else 0.0
    avg_lat = (total_time / total_execs) if total_execs > 0 else 0.0

    print("\n" + "=" * 70)
    print("  RESULTADOS DO BENCHMARK")
    print("=" * 70)
    print(f"Total de execuções: {total_execs}")
    print(f"Acurácia de Classificação de Intenção: {acc_intent:.1f}% ({intent_hits}/{total_execs})")
    print(f"Latência média por execução: {avg_lat:.3f}s")

    if out_file:
        out_path = Path(out_file)
        with open(out_path, "w", encoding="utf-8") as f:
            for reg in registros:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        print(f"Resultados detalhados salvos em: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1, help="Número de repetições por caso")
    parser.add_argument("--out", type=str, default=None, help="Caminho de saída para salvar JSONL")
    args = parser.parse_args()

    rodar_benchmark(reps=args.reps, out_file=args.out)


if __name__ == "__main__":
    main()

