#!/usr/bin/env python3
"""debug_gate.py — diagnostica el gate de ámbito y la generación para una consulta.

Aísla dónde falla una consulta: (1) margen del reranker, (2) veredicto del
clasificador LLM, (3) decisión final del gate, y (4) si pasa, la respuesta
generada. Útil para distinguir un falso rechazo del gate de una abstención del
modelo en generación.

Uso:
    cd codigo/
    python scripts/debug_gate.py "¿Cuándo abre la secretaría?"
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graia.generacion.ollama_client import OllamaClient
from graia.generacion.postprocess import clean_answer
from graia.generacion.prompt_builder import build_messages
from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.query_router import expand_abbreviations, route_query
from graia.recuperacion.retriever import retrieve
from graia.recuperacion.scope_classifier import _classify_with_llm, is_in_scope


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "¿Cuándo abre la secretaría?"
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    emb, rec, gen, sg = cfg["embeddings"], cfg["recuperacion"], cfg["generacion"], cfg.get("scope_gate", {})

    embedder = Embedder(model_name=emb["model_name"], query_prefix=emb["query_prefix"],
                        passage_prefix=emb["passage_prefix"], batch_size=emb["batch_size"],
                        normalize=emb["normalize"])
    store = VectorStore.load(cfg["paths"]["index"])
    bm25 = BM25Index.load(cfg["paths"]["index"]) if (Path(cfg["paths"]["index"]) / "bm25_index.pkl").exists() else None
    client = OllamaClient(model=gen["model"], temperature=gen["temperature"], top_p=gen["top_p"],
                          max_tokens=gen["max_tokens"], stop=gen.get("stop", []))

    eq = expand_abbreviations(query)
    route = route_query(query)
    chunks = retrieve(eq, embedder, store, bm25_index=bm25,
                      use_hybrid=rec.get("use_hybrid", False) and bm25 is not None,
                      rrf_k=rec.get("rrf_k", 60), k_candidates=rec["k_candidates"],
                      k_final=rec["k_final"], k_final_listing=rec.get("k_final_listing"),
                      similarity_threshold=rec["similarity_threshold"], mmr_lambda=rec["mmr_lambda"],
                      use_reranker=rec.get("use_reranker", False),
                      off_category_penalty=rec.get("off_category_penalty", 0.0))

    top = max((c.similarity for c in chunks), default=float("-inf"))
    print("=" * 70)
    print(f"Consulta: {query!r}")
    print(f"Router enrutado: {route.is_routed}  categorías: {list(route.categories.keys())}")
    print(f"Margen del mejor fragmento (reranker): {top:.3f}   "
          f"[high={sg.get('high_margin')}, low={sg.get('low_margin')}]")
    print(f"Clasificador LLM (aislado) dice ACADÉMICO: {_classify_with_llm(query, client)}")
    decision = is_in_scope(query, chunks, client, enabled=sg.get("enabled", True),
                           reranker_used=rec.get("use_reranker", False),
                           high_margin=sg.get("high_margin", 2.0), low_margin=sg.get("low_margin", -1.0))
    print(f"DECISIÓN DEL GATE (is_in_scope): {decision}")
    if decision:
        sys_p, usr_p = build_messages(query, chunks)
        ans = clean_answer(client.generate(sys_p, usr_p).text)
        print(f"\nRESPUESTA GENERADA:\n{ans[:400]}")
    else:
        print("\n→ El gate rechaza la consulta (no genera): abstención canónica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
