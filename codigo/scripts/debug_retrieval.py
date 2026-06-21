#!/usr/bin/env python3
"""debug_retrieval.py — inspecciona qué fragmentos recupera GRAIA para una consulta.

Muestra, para una pregunta dada, los fragmentos que el sistema entregaría al LLM
(con su categoría, tipo, score y un extracto). Sirve para diagnosticar fallos de
recuperación: por ejemplo, comprobar si la línea «Defensa del TFG: del 19 al 21…»
llega realmente al contexto y en qué posición, y si el reranker está activo.

Uso:
    cd codigo/
    python scripts/debug_retrieval.py "¿qué días es la defensa del TFG?"
    python scripts/debug_retrieval.py "en qué aula es Cálculo"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.query_router import expand_abbreviations
from graia.recuperacion.retriever import retrieve

# Mostrar los INFO del retriever (routing, inyección, reranking) para diagnóstico.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "¿qué días es la defensa del TFG?"
    # k opcional (2.º argumento): amplía el nº de fragmentos mostrados para ver
    # dónde cae en el ranking un chunk que no entra en el top-5 habitual.
    k_override = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with open("config/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    emb = cfg["embeddings"]
    rec = cfg["recuperacion"]

    embedder = Embedder(
        model_name=emb["model_name"], query_prefix=emb["query_prefix"],
        passage_prefix=emb["passage_prefix"], batch_size=emb["batch_size"],
        normalize=emb["normalize"],
    )
    store = VectorStore.load(cfg["paths"]["index"])
    bm25_path = Path(cfg["paths"]["index"]) / "bm25_index.pkl"
    bm25 = BM25Index.load(cfg["paths"]["index"]) if bm25_path.exists() else None

    eq = expand_abbreviations(query)
    print("\n" + "=" * 70)
    print(f"Consulta:  {query!r}")
    print(f"Expandida: {eq!r}")
    print(f"use_reranker={rec.get('use_reranker')}  use_hybrid={rec.get('use_hybrid')}")
    print("=" * 70)

    chunks = retrieve(
        eq, embedder, store, bm25_index=bm25,
        use_hybrid=rec.get("use_hybrid", False) and bm25 is not None,
        rrf_k=rec.get("rrf_k", 60), k_candidates=rec["k_candidates"],
        k_final=k_override or rec["k_final"],
        k_final_listing=None if k_override else rec.get("k_final_listing"),
        similarity_threshold=rec["similarity_threshold"], mmr_lambda=rec["mmr_lambda"],
        use_reranker=rec.get("use_reranker", False),
        off_category_penalty=rec.get("off_category_penalty", 0.0),
    )

    print(f"\n{len(chunks)} fragmentos entregados al LLM:\n")
    for c in chunks:
        cat = c.metadata.get("category", "?")
        tipo = c.metadata.get("tipo", "-")
        print(f"[rank {c.rank}] score={c.similarity:.3f}  cat={cat}  tipo={tipo}")
        print("   ", c.text[:180].replace("\n", " "))
        print("    fuente:", c.source_url[:90])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
