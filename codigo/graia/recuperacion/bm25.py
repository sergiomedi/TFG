"""BM25 — recuperación léxica para el canal híbrido de GRAIA.

Implementa un índice BM25 (Best Matching 25) basado en la librería
``rank_bm25`` como canal léxico complementario al canal denso (FAISS).
La combinación de ambos canales se realiza mediante Reciprocal Rank
Fusion (RRF) en el módulo :mod:`graia.recuperacion.retriever`.

Decisión de diseño (Cap. 5, Sección 5.8):
  BM25 complementa la búsqueda densa capturando coincidencias léxicas
  exactas (nombres propios, códigos de asignatura, siglas) que los
  embeddings densos pueden diluir al proyectar a un espacio semántico.

Persistencia:
  El índice BM25 se serializa con ``pickle`` en ``bm25_index.pkl``
  junto al índice FAISS, permitiendo carga simultánea en tiempo de
  consulta sin re-tokenizar el corpus.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Tokenización simple para español: lowercase + split por no-alfanumérico
_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Tokenización básica: lowercase + tokens alfanuméricos.

    No se aplica stemming para preservar la capacidad de matching exacto
    con códigos de asignatura (ej. ``IC``, ``FFT``) y siglas académicas.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Índice BM25 sobre un corpus de chunks textuales.

    Almacena internamente la instancia de ``BM25Okapi`` y una lista
    paralela de identificadores (``chunk_ids``) para mapear resultados
    de BM25 a los metadatos del VectorStore.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    # ---- Construcción ----

    def build(self, texts: Sequence[str], chunk_ids: Sequence[str]) -> None:
        """Construye el índice BM25 a partir de los textos del corpus.

        Parameters
        ----------
        texts : Sequence[str]
            Textos de los chunks (mismo orden que en el VectorStore).
        chunk_ids : Sequence[str]
            Identificadores de los chunks (paralelo a *texts*).
        """
        if len(texts) != len(chunk_ids):
            raise ValueError(
                f"Longitudes no coinciden: {len(texts)} textos vs {len(chunk_ids)} chunk_ids"
            )
        tokenized_corpus = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_ids = list(chunk_ids)
        logger.info("Índice BM25 construido con %d documentos", len(texts))

    # ---- Búsqueda ----

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        """Busca los *k* chunks más relevantes según BM25.

        Parameters
        ----------
        query : str
            Consulta en lenguaje natural.
        k : int
            Número de resultados a devolver.

        Returns
        -------
        list[tuple[str, float]]
            Lista de (chunk_id, score_bm25) ordenada por score descendente.
        """
        if self._bm25 is None:
            raise RuntimeError("El índice BM25 no ha sido construido. Llama a build() primero.")

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        # Obtener top-k índices por score
        top_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # Descartar chunks sin ningún token en común
                results.append((self._chunk_ids[idx], score))
        return results

    # ---- Persistencia ----

    def save(self, directory: str | Path) -> None:
        """Guarda el índice BM25 serializado en *directory*/bm25_index.pkl."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "bm25_index.pkl"
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)
        logger.info("Índice BM25 guardado en %s (%.1f KB)",
                     path, path.stat().st_size / 1024)

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        """Carga un índice previamente guardado con :meth:`save`."""
        path = Path(directory) / "bm25_index.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._bm25 = data["bm25"]
        instance._chunk_ids = data["chunk_ids"]
        logger.info("Índice BM25 cargado desde %s (%d documentos)",
                     path, len(instance._chunk_ids))
        return instance

    @property
    def size(self) -> int:
        """Número de documentos indexados."""
        return len(self._chunk_ids)
