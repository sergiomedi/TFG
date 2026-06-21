#!/usr/bin/env python3
"""evaluate_ragas.py — Fase B de la evaluación con RAGAS (entorno aislado).

Lee el .jsonl producido por ``scripts/export_for_ragas.py`` (pregunta, respuesta y
contextos recuperados) y calcula métricas RAGAS usando un LLM local (Ollama) como
juez. NO importa ``graia`` ni torch/faiss/sentence-transformers: por eso debe
ejecutarse en un entorno virtual SEPARADO (``.venv-ragas``), de modo que las
dependencias de RAGAS (datasets, langchain, pyarrow, transformers…) nunca toquen el
``.venv`` de la aplicación.

Métricas (sin respuesta de referencia, acordes al dataset):
  - **faithfulness**     : fracción de afirmaciones de la respuesta que se
                           sustentan en los contextos recuperados. Es la medida
                           directa de fidelidad / ausencia de alucinación (H2).
  - **answer_relevancy** : cuán pertinente es la respuesta a la pregunta. Requiere un
                           modelo de *embeddings* en Ollama (por defecto
                           ``nomic-embed-text``; descárgalo con ``ollama pull``).

Ambas se calculan por defecto (``--metrics faithfulness,answer_relevancy``).

Solo se evalúan las preguntas **factuales realmente respondidas** (no abstenciones
ni fuera de ámbito): faithfulness y answer_relevancy no tienen sentido sobre una
respuesta de "no dispongo de información".

Uso (en el .venv-ragas):
    python scripts/evaluate_ragas.py --in eval/ragas_input_llama.jsonl --out eval/ragas_llama.json
    python scripts/evaluate_ragas.py --in eval/ragas_input_qwen.jsonl  --out eval/ragas_qwen.json --judge qwen2.5:14b-instruct
    python scripts/evaluate_ragas.py --in eval/ragas_input_llama.jsonl --out eval/ragas_llama.json --metrics faithfulness,answer_relevancy

Requiere Ollama en ejecución con el modelo juez descargado (y, para
answer_relevancy, el de embeddings).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación RAGAS sobre salidas exportadas")
    ap.add_argument("--in", dest="inp", required=True, help="ragas_input_*.jsonl (de export_for_ragas.py)")
    ap.add_argument("--out", required=True, help="Ruta del informe JSON de salida")
    ap.add_argument("--judge", default="llama3.1:8b-instruct-q4_K_M",
                    help="Modelo Ollama usado como juez por RAGAS")
    ap.add_argument("--embed-model", default="nomic-embed-text",
                    help="Modelo Ollama de embeddings (solo para answer_relevancy)")
    ap.add_argument("--metrics", default="faithfulness,answer_relevancy",
                    help="Lista separada por comas. Por defecto: faithfulness,answer_relevancy. "
                         "answer_relevancy requiere un modelo de embeddings en Ollama (--embed-model)")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    # --- Imports de RAGAS aislados (mensaje claro si falta el entorno) ---
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.run_config import RunConfig
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from langchain_ollama import ChatOllama, OllamaEmbeddings
    except ImportError as e:
        print("ERROR: faltan dependencias de RAGAS. ¿Estás en el .venv-ragas?", file=sys.stderr)
        print(f"  Detalle: {e}", file=sys.stderr)
        print("  Instala con:  pip install -r requirements-ragas.txt", file=sys.stderr)
        return 2

    rows = load_rows(Path(args.inp))
    # Solo factuales respondidas: la fidelidad de una abstención no es informativa.
    evald = [r for r in rows
             if r.get("type") != "no_info" and not r.get("should_refuse")
             and not r.get("refused")
             and r.get("retrieved_contexts")]
    skipped = len(rows) - len(evald)
    print(f"Filas totales: {len(rows)} | evaluadas: {len(evald)} | "
          f"omitidas (abstención/no_info/sin contexto): {skipped}")
    if not evald:
        print("No hay filas evaluables.", file=sys.stderr)
        return 1

    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    judge = LangchainLLMWrapper(ChatOllama(model=args.judge, base_url=args.ollama_url,
                                           temperature=0.0))
    metrics = []
    embeddings = None
    if "faithfulness" in wanted:
        metrics.append(Faithfulness(llm=judge))
    if "answer_relevancy" in wanted:
        embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=args.embed_model, base_url=args.ollama_url))
        metrics.append(ResponseRelevancy(llm=judge, embeddings=embeddings))
    if not metrics:
        print("No se reconoció ninguna métrica en --metrics.", file=sys.stderr)
        return 1

    dataset = EvaluationDataset.from_list([
        {"user_input": r["user_input"],
         "response": r["response"],
         "retrieved_contexts": r["retrieved_contexts"]}
        for r in evald
    ])

    print(f"Evaluando con juez={args.judge} | métricas={[m.name for m in metrics]} ...")
    # max_workers=1: Ollama local sirve secuencialmente; evita saturar la GPU.
    run_config = RunConfig(max_workers=1, timeout=args.timeout)
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge,
                      embeddings=embeddings, run_config=run_config)

    df = result.to_pandas()
    metric_cols = [m.name for m in metrics]
    means = {c: (float(df[c].mean()) if c in df.columns else None) for c in metric_cols}

    per_item = []
    for r, (_, row) in zip(evald, df.iterrows()):
        rec = {"id": r["id"], "type": r.get("type")}
        for c in metric_cols:
            rec[c] = (None if c not in df.columns else
                      (None if row[c] != row[c] else float(row[c])))  # NaN -> None
        per_item.append(rec)

    summary = {"judge": args.judge, "n_evaluated": len(evald), "n_skipped": skipped,
               "metrics_mean": means}

    print("\n" + "=" * 64)
    print(f"RESUMEN RAGAS — juez: {args.judge}")
    print("=" * 64)
    for c, v in means.items():
        print(f"  {c:20s}: {v:.3f}" if v is not None else f"  {c:20s}: -")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": per_item},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInforme guardado en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
