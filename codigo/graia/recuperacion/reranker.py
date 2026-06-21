"""Reranker — reordenación con cross-encoder para mejorar la precisión.

Implementa una etapa de reranking entre la recuperación y la generación
(Sección 5.X del diseño). Mientras que la búsqueda por embeddings
(bi-encoder) codifica query y documento de forma independiente — lo que
permite búsqueda rápida pero pierde interacciones cruzadas —, un
cross-encoder evalúa el par (query, documento) conjuntamente, capturando
relaciones semánticas más finas.

El modelo seleccionado es ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``:
  - Arquitectura MiniLM (Wang et al., 2020), ligera (~120 MB)
  - Entrenado sobre mMARCO (Bonifacio et al., 2021), versión multilingüe de
    MS MARCO que incluye español, por lo que discrimina bien pares
    (consulta, fragmento) en español
  - Respaldo automático a ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (solo inglés)
    si el modelo multilingüe no puede descargarse
Fuentes (model cards en Hugging Face):
  - https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

Se aplica sobre los top-k_candidates tras el filtrado por umbral τ y
ANTES del MMR, de modo que MMR diversifique sobre un pool ya reordenado
por relevancia precisa.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Importación lazy para no requerir sentence-transformers si no se usa
_cross_encoder = None
# Cross-encoder MULTILINGÜE: el corpus de la ETSIIT está en español, por lo que
# un reranker entrenado solo en inglés (ms-marco-MiniLM) discrimina mal entre
# pares (consulta, fragmento) en español. mMiniLMv2 está entrenado sobre mMARCO
# (versión multilingüe de MS MARCO, con español), manteniendo un tamaño ligero.
_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# Respaldo en inglés por si el modelo multilingüe no puede descargarse (sin red,
# caché ausente): preferible un reranking en inglés a no tener reranking.
_FALLBACK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model():
    """Carga el cross-encoder de forma lazy (una sola vez), con respaldo."""
    global _cross_encoder, _MODEL_NAME
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        try:
            logger.info("Cargando cross-encoder multilingüe: %s", _MODEL_NAME)
            _cross_encoder = CrossEncoder(_MODEL_NAME)
        except Exception as exc:  # noqa: BLE001 (queremos degradar con cualquier error)
            logger.warning(
                "No se pudo cargar el cross-encoder multilingüe (%s); "
                "se usa el respaldo en inglés %s.", exc, _FALLBACK_MODEL_NAME,
            )
            _cross_encoder = CrossEncoder(_FALLBACK_MODEL_NAME)
            _MODEL_NAME = _FALLBACK_MODEL_NAME
        logger.info("Cross-encoder cargado correctamente: %s", _MODEL_NAME)
    return _cross_encoder


def rerank(
    query: str,
    texts: Sequence[str],
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Reordena *texts* por relevancia respecto a *query* usando cross-encoder.

    Parameters
    ----------
    query : str
        Consulta del usuario.
    texts : Sequence[str]
        Textos candidatos a reordenar.
    top_k : int | None
        Si se especifica, devuelve solo los top_k más relevantes.

    Returns
    -------
    list[tuple[int, float]]
        Lista de (índice_original, score) ordenada por score descendente.
    """
    if not texts:
        return []

    model = _get_model()

    # El cross-encoder evalúa pares (query, passage) conjuntamente
    pairs = [(query, text) for text in texts]
    scores = model.predict(pairs)

    # Convertir a lista indexada y ordenar por score
    indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        indexed_scores = indexed_scores[:top_k]

    logger.debug(
        "Reranking: %d candidatos → top score=%.4f, bottom score=%.4f",
        len(texts),
        indexed_scores[0][1] if indexed_scores else 0,
        indexed_scores[-1][1] if indexed_scores else 0,
    )
    return indexed_scores
