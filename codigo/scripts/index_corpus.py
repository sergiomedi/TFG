#!/usr/bin/env python3
"""index_corpus.py — Chunking + Embedding + Indexación FAISS para GRAIA.

Lee el corpus validado (data/processed/corpus.jsonl), lo fragmenta en chunks,
genera embeddings con E5 y construye el índice FAISS en data/index/.

Ejecución:
    cd codigo/
    python scripts/index_corpus.py
    python scripts/index_corpus.py --input data/processed/corpus.jsonl
    python scripts/index_corpus.py --dry-run   # solo muestra estadísticas sin indexar

Requisitos:
    - GPU recomendada (RTX 4070 Super → ~2 min para 5000 chunks)
    - Sin GPU funciona en CPU (~10 min)
    - El corpus debe haber sido revisado con review_corpus.py
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
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graia.ingesta.chunker import chunk_document
from graia.ingesta.structured import chunk_structured_records, is_structured
from graia.ingesta.models import ParsedDocument, SourceType
from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index_corpus")


def load_corpus(path: Path) -> list[dict]:
    """Carga el corpus JSONL."""
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def refine_category(url: str, title: str, original_category: str) -> str:
    """Refina la categoría del documento basándose en URL y título.

    La categoría original del crawler (``build_corpus.py``) agrupa bajo
    ``guias_docentes`` documentos heterogéneos: guías docentes individuales,
    fichas de asignatura de la ETSIIT, el Plan de Estudios y la Presentación
    del Grado. Esta función los separa en categorías más precisas para que
    el Query Router pueda enrutar consultas al subconjunto correcto.

    Categorías resultantes (10):
      plan_estudios, presentacion_grado, guias_docentes,
      tfg, calendario, movilidad, normativa, tramites,
      estudiantes, profesorado
    """
    title_lower = (title or "").lower()
    url_lower = url.lower()

    # Horarios de clase (tabla reconstruida por HorarioParser): categoría propia
    # 'horarios', separada del calendario de exámenes (CalendarioExamenes...) y
    # del calendario académico, que permanecen en 'calendario'. Esto permite que
    # el Query Router envíe las consultas de clase/aula/hora al documento correcto.
    if "horario" in url_lower or "horario" in title_lower:
        return "horarios"

    # Plan de estudios (separado de guias_docentes)
    if "plan-estudios" in url_lower or "plan de estudios" in title_lower:
        return "plan_estudios"

    # Secretaría (debe evaluarse ANTES de presentacion_grado
    # porque la URL contiene "/presentacion/secretaria")
    if "/secretaria" in url_lower and "etsiit" in url_lower:
        return "tramites"

    # Presentación del grado (separado de guias_docentes)
    # Solo la página raíz de presentación, no subpáginas como /secretaria
    if (
        url_lower.endswith("/presentacion")
        or url_lower.endswith("/presentacion/")
        or "presentación del grado" in title_lower
    ):
        return "presentacion_grado"

    return original_category


def corpus_to_parsed_docs(corpus: list[dict]) -> list[ParsedDocument]:
    """Convierte los registros JSONL a objetos ParsedDocument para el chunker."""
    parsed = []
    for record in corpus:
        try:
            doc = ParsedDocument(
                url=record["url"],
                source_type=SourceType(record.get("source_type", "html")),
                title=record.get("title"),
                text=record["text"],
                fetched_at=datetime.fromisoformat(record["fetched_at"]),
                metadata=record.get("metadata", {}),
            )
            # Categoría refinada para query routing (Sección 5.8.4)
            raw_category = record.get("category", "general")
            doc.metadata["category"] = refine_category(
                record["url"], record.get("title", ""), raw_category,
            )
            parsed.append(doc)
        except Exception as e:
            logger.warning("Error convirtiendo registro %s: %s", record.get("url", "?"), e)
    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Indexación del corpus de GRAIA: chunk → embed → FAISS"
    )
    parser.add_argument("--input", default="data/processed/corpus.jsonl",
                        help="Corpus JSONL de entrada")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Configuración general de GRAIA")
    parser.add_argument("--output", default=None,
                        help="Directorio de salida del índice (default: config paths.index)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo chunking + estadísticas, sin generar embeddings")
    args = parser.parse_args()

    # Cargar configuración
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunk_cfg = cfg["chunking"]
    emb_cfg = cfg["embeddings"]
    output_dir = Path(args.output or cfg["paths"]["index"])

    # ── Fase 1: Cargar corpus ────────────────────────────────────────────
    corpus_path = Path(args.input)
    if not corpus_path.exists():
        logger.error("Corpus no encontrado: %s", corpus_path)
        logger.error("Ejecuta primero: python scripts/build_corpus.py")
        sys.exit(1)

    logger.info("Cargando corpus desde %s ...", corpus_path)
    corpus = load_corpus(corpus_path)
    parsed_docs = corpus_to_parsed_docs(corpus)
    logger.info("Documentos cargados: %d", len(parsed_docs))

    # NOTA: el post-procesado tabular con pdfplumber que existía aquí en v1 se
    # ha eliminado. Ahora el parser (graia.ingesta.parser.PdfParser) extrae las
    # tablas a Markdown y aplica OCR durante build_corpus.py, de modo que el
    # corpus.jsonl ya llega limpio y estructurado. Esto elimina la re-descarga
    # de PDFs en tiempo de indexación (más rápido y reproducible).

    # ── Fase 2: Chunking ─────────────────────────────────────────────────
    logger.info("Fragmentando documentos (chunk_size=%d, overlap=%d, min=%d) ...",
                chunk_cfg["chunk_size_tokens"],
                chunk_cfg["chunk_overlap_tokens"],
                chunk_cfg["min_chunk_tokens"])

    all_chunks = []
    n_structured = 0
    for doc in parsed_docs:
        # De-ruido (P3): la página HTML índice de horarios/calendarios solo lista
        # enlaces a PDFs por año y compite con el PDF real; se descarta.
        url_l = (doc.url or "").lower()
        if doc.source_type.value == "html" and url_l.rstrip("/").endswith("/horarios"):
            logger.info("Descartada página índice de horarios (ruido): %s", doc.url)
            continue

        # Datos estructurados (horarios): un registro = un chunk + metadatos
        # (best practice para datos tabulares; recuperación de alta precisión).
        if is_structured(doc):
            chunks = chunk_structured_records(doc)
            n_structured += len(chunks)
        else:
            chunks = chunk_document(
                doc,
                chunk_size_tokens=chunk_cfg["chunk_size_tokens"],
                chunk_overlap_tokens=chunk_cfg["chunk_overlap_tokens"],
                min_chunk_tokens=chunk_cfg["min_chunk_tokens"],
            )
        all_chunks.extend(chunks)

    if n_structured:
        logger.info("Chunks estructurados (1 registro = 1 chunk): %d", n_structured)

    logger.info("Total de chunks generados: %d", len(all_chunks))
    logger.info("  Tamaño medio: %d chars",
                sum(len(c.text) for c in all_chunks) // max(len(all_chunks), 1))

    if args.dry_run:
        logger.info("── DRY RUN: no se generan embeddings ni índice ──")
        print(f"\nResumen dry-run:")
        print(f"  Documentos:  {len(parsed_docs)}")
        print(f"  Chunks:      {len(all_chunks)}")
        print(f"  Chars total: {sum(len(c.text) for c in all_chunks):,}")
        return

    # ── Fase 2b: Contextual Retrieval (opcional) ─────────────────────────
    # Antepone a cada chunk un breve contexto generado por el LLM que lo sitúa
    # en su documento (Anthropic, 2024). Mejora la recuperación de fragmentos
    # huérfanos. Se cachea en disco para reproducibilidad. Tras este paso, el
    # texto de cada chunk pasa a ser "<contexto>\n\n<texto>" y ese texto es el
    # que se embebe (Fase 3) y se indexa en BM25 (Fase 5).
    cr_cfg = cfg.get("contextual_retrieval", {}) or {}
    if cr_cfg.get("enabled", False):
        from graia.generacion.ollama_client import OllamaClient
        from graia.ingesta.contextualizer import ContextCache, contextualize_chunks

        logger.info("Contextual Retrieval ACTIVADO (modelo=%s) ...",
                    cr_cfg.get("model"))
        doc_texts = {d.url: d.text for d in parsed_docs}
        cr_client = OllamaClient(
            model=cr_cfg.get("model", "llama3.1:8b-instruct-q4_K_M"),
            temperature=0.0,
            max_tokens=160,
            stop=(),
        )
        cache = ContextCache(cr_cfg.get(
            "cache_path", "data/processed/context_cache.json"))
        all_chunks = contextualize_chunks(
            all_chunks, doc_texts, cr_client,
            cache=cache,
            max_doc_chars=cr_cfg.get("max_doc_chars", 6000),
            progress=lambda i, n: logger.info("  Contextualizando: %d/%d", i, n),
        )
    else:
        logger.info("Contextual Retrieval desactivado (contextual_retrieval.enabled=false)")

    # ── Fase 3: Embeddings ───────────────────────────────────────────────
    logger.info("Inicializando Embedder (%s) ...", emb_cfg["model_name"])
    embedder = Embedder(
        model_name=emb_cfg["model_name"],
        query_prefix=emb_cfg["query_prefix"],
        passage_prefix=emb_cfg["passage_prefix"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg["normalize"],
    )

    # NOTA: Se evaluó title prepending para embeddings (anteponer título al
    # texto antes de codificar). Resultado: empeoró el ranking del Plan de
    # Estudios (rank #240 → #524) porque enriqueció uniformemente las 841
    # guías docentes, haciéndolas aún más competitivas. El título se usa
    # solo en BM25 (Fase 5), donde el matching léxico sí se beneficia.
    # La solución real es query routing con filtrado por categoría (Sec. 5.8).
    texts = [c.text for c in all_chunks]
    logger.info("Generando embeddings para %d chunks (batch_size=%d) ...",
                len(texts), emb_cfg["batch_size"])

    t0 = time.perf_counter()

    # Procesar en batches con barra de progreso
    batch_size = emb_cfg["batch_size"]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = embedder.encode_passages(batch)
        all_embeddings.append(batch_vecs)
        done = min(i + batch_size, len(texts))
        logger.info("  Embeddings: %d/%d (%.0f%%)", done, len(texts), 100 * done / len(texts))

    embeddings = np.vstack(all_embeddings)
    elapsed = time.perf_counter() - t0
    logger.info("Embeddings generados en %.1fs (%.0f chunks/s)",
                elapsed, len(texts) / elapsed)

    # ── Fase 4: Indexación FAISS ─────────────────────────────────────────
    logger.info("Construyendo índice FAISS (IndexFlatIP, dim=%d) ...", embedder.dim)
    store = VectorStore(dim=embedder.dim)
    store.add(embeddings, all_chunks)

    logger.info("Guardando índice FAISS en %s ...", output_dir)
    store.save(output_dir)

    # ── Fase 5: Índice BM25 (recuperación léxica) ───────────────────────
    logger.info("Construyendo índice BM25 para recuperación híbrida (title prepending=ON) ...")
    bm25 = BM25Index()
    # Title prepending también para BM25: permite matching léxico con términos
    # del título (e.g. "plan de estudios", "calendario TFG") que pueden no
    # aparecer en el cuerpo del chunk.
    bm25_texts = [
        f"{c.title}\n{c.text}" if c.title else c.text
        for c in all_chunks
    ]
    bm25_ids = [c.chunk_id for c in all_chunks]
    bm25.build(bm25_texts, bm25_ids)
    bm25.save(output_dir)
    bm25_size_kb = (output_dir / "bm25_index.pkl").stat().st_size / 1024

    # ── Resumen ──────────────────────────────────────────────────────────
    index_size_mb = (output_dir / "index.faiss").stat().st_size / 1_048_576
    meta_size_mb = (output_dir / "index.meta.json").stat().st_size / 1_048_576

    print(f"\n{'=' * 60}")
    print("¡INDEXACIÓN COMPLETADA!")
    print(f"  Documentos:    {len(parsed_docs)}")
    print(f"  Chunks:        {len(all_chunks)}")
    print(f"  Dimensión:     {embedder.dim}")
    print(f"  Índice FAISS:  {index_size_mb:.1f} MB")
    print(f"  Índice BM25:   {bm25_size_kb:.1f} KB")
    print(f"  Metadatos:     {meta_size_mb:.1f} MB")
    print(f"  Tiempo embed:  {elapsed:.1f}s")
    print(f"  Directorio:    {output_dir}")
    print(f"\nSiguiente paso: streamlit run graia/interfaz/app.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
