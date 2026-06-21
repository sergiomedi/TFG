"""Subsistema de recuperación de GRAIA.

Componentes principales:
  - :func:`retrieve`        — flujo completo (búsqueda → [RRF] → umbral → MMR → top-k)
  - :func:`mmr_rerank`      — algoritmo MMR de Carbonell & Goldstein (1998)
  - :class:`BM25Index`      — índice léxico BM25 para recuperación híbrida
  - :class:`RetrievedChunk` — chunk anotado con score y rank
"""

from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.mmr import mmr_rerank
from graia.recuperacion.retriever import RetrievedChunk, retrieve

__all__ = ["retrieve", "mmr_rerank", "BM25Index", "RetrievedChunk"]
