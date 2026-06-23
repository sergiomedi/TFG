#!/usr/bin/env python3
"""export_for_ragas.py — Fase A de la evaluación con RAGAS.

Ejecuta el MISMO pipeline RAG de :mod:`scripts.evaluate` (recuperación → gate de
ámbito → generación → post-procesado) sobre el dataset y vuelca, por pregunta, los
campos que RAGAS necesita:

  - ``user_input``         : la consulta del usuario.
  - ``response``           : la respuesta final del sistema (ya post-procesada).
  - ``retrieved_contexts`` : el texto de los fragmentos recuperados (``chunk.text``),
                             tal cual constituyen la evidencia que el modelo vio.

Se ejecuta en el **entorno global** (el que tiene torch/faiss/
sentence-transformers y corre la app), porque reutiliza ``graia``. NO importa RAGAS:
así el entorno nunca se contamina. El cálculo de las métricas RAGAS lo hace después
``scripts/evaluate_ragas.py`` en un entorno aislado (``.venv-ragas``), leyendo el
.jsonl que aquí se genera.

Uso:
    cd codigo/
    python scripts/export_for_ragas.py --model llama3.1:8b-instruct-q4_K_M --out eval/ragas_input_llama.jsonl
    python scripts/export_for_ragas.py --model qwen2.5:14b-instruct        --out eval/ragas_input_qwen.jsonl
    python scripts/export_for_ragas.py --model gemma2:9b                   --out eval/ragas_input_gemma.jsonl

Requiere índice construido (index_corpus.py) y Ollama en ejecución, igual que
``evaluate.py``.
"""

from __future__ import annotations

# Forzar codificación UTF-8 en la salida estándar. En Windows la consola usa
# por defecto cp1252, que no puede representar caracteres Unicode como los de
# dibujo de cajas (U+2500 «─») ni algunos acentos, provocando UnicodeEncodeError.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from graia.generacion.ollama_client import OllamaClient
from graia.generacion.postprocess import clean_answer, is_abstention
from graia.generacion.prompt_builder import build_messages
from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.query_router import expand_abbreviations
from graia.recuperacion.retriever import retrieve
from graia.recuperacion.scope_classifier import OUT_OF_SCOPE_ANSWER, is_in_scope


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Exporta entradas para la evaluación RAGAS")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default="eval/eval_dataset.jsonl",
                    help="Dataset de evaluación de 74 preguntas, "
                         "el mismo que usa la evaluación del Capítulo 7 (coherencia de denominador)")
    ap.add_argument("--model", default=None, help="Sobrescribe generacion.model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="Ruta del .jsonl de salida")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rec = cfg["recuperacion"]
    gen = cfg["generacion"]
    model = args.model or gen["model"]

    print(f"Cargando índice y modelos (generación: {model}) ...")
    emb = cfg["embeddings"]
    embedder = Embedder(model_name=emb["model_name"], query_prefix=emb["query_prefix"],
                        passage_prefix=emb["passage_prefix"], batch_size=emb["batch_size"],
                        normalize=emb["normalize"])
    store = VectorStore.load(cfg["paths"]["index"])
    bm25_path = Path(cfg["paths"]["index"]) / "bm25_index.pkl"
    bm25 = BM25Index.load(cfg["paths"]["index"]) if bm25_path.exists() else None
    client = OllamaClient(model=model, temperature=gen["temperature"], top_p=gen["top_p"],
                          max_tokens=gen["max_tokens"], stop=gen.get("stop", []))

    dataset = load_dataset(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]

    sg = cfg.get("scope_gate", {})
    rows: list[dict] = []
    for item in dataset:
        q = expand_abbreviations(item["query"])
        chunks = retrieve(
            q, embedder, store, bm25_index=bm25,
            use_hybrid=rec.get("use_hybrid", False) and bm25 is not None,
            rrf_k=rec.get("rrf_k", 60), k_candidates=rec["k_candidates"],
            k_final=rec["k_final"], k_final_listing=rec.get("k_final_listing"),
            similarity_threshold=rec["similarity_threshold"],
            mmr_lambda=rec["mmr_lambda"], use_reranker=rec.get("use_reranker", False),
            off_category_penalty=rec.get("off_category_penalty", 0.0),
        )
        if not is_in_scope(
            item["query"], chunks, client,
            enabled=sg.get("enabled", True),
            reranker_used=rec.get("use_reranker", False),
            high_margin=sg.get("high_margin", 2.0),
            low_margin=sg.get("low_margin", -1.0),
        ):
            answer = OUT_OF_SCOPE_ANSWER
        else:
            sys_p, usr_p = build_messages(item["query"], chunks)
            answer = clean_answer(client.generate(sys_p, usr_p).text)

        rows.append({
            "id": item["id"],
            "type": item.get("type", "factual"),
            "should_refuse": bool(item.get("should_refuse")),
            "refused": is_abstention(answer),
            "user_input": item["query"],
            "response": answer,
            "retrieved_contexts": [c.text for c in chunks],
            # Referencia débil a partir de las palabras clave esperadas (informativa;
            # las métricas por defecto, faithfulness y answer_relevancy, NO la usan).
            "expected_keywords": item.get("expected_keywords", []),
        })
        print(f"  [{item['id']}] ctxs={len(rows[-1]['retrieved_contexts'])} "
              f"refused={rows[-1]['refused']}")

    out = Path(args.out) if args.out else Path("eval") / f"ragas_input_{model.split(':')[0]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nExportadas {len(rows)} filas a {out}")
    print("Siguiente paso (en el .venv-ragas):")
    print(f"  python scripts/evaluate_ragas.py --in {out} --out eval/ragas_{model.split(':')[0]}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
