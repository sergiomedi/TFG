"""VectorStore — índice FAISS para búsqueda por similitud.

Implementa el componente de almacenamiento vectorial descrito en la
Sección 5.7.2 del diseño:
  - FAISS IndexFlatIP (producto interno sobre vectores normalizados ≡ coseno)
  - Persistencia: índice binario (``.faiss``) + metadatos JSON (``.meta.json``)
  - Operaciones: build, save, load, search (top-k)

La separación entre índice y metadatos permite reconstruir la correspondencia
vector → chunk sin duplicar el texto dentro de FAISS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np

from graia.ingesta.models import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Almacén vectorial basado en FAISS IndexFlatIP.

    Mantiene en memoria el índice FAISS y una lista paralela de metadatos
    (uno por vector) que permite recuperar el ``Chunk`` original tras la
    búsqueda.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.metadata: list[dict[str, Any]] = []

    # ---- Construcción ----

    def add(
        self,
        vectors: np.ndarray,
        chunks: Sequence[Chunk],
    ) -> None:
        """Añade vectores y sus metadatos asociados al índice.

        Parameters
        ----------
        vectors : np.ndarray
            Matriz (n, dim) de embeddings ya normalizados.
        chunks : Sequence[Chunk]
            Lista paralela de chunks; se almacenan sus metadatos serializables.
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Longitudes no coinciden: {len(vectors)} vectores vs {len(chunks)} chunks"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Dimensión incorrecta: esperaba {self.dim}, recibió {vectors.shape[1]}"
            )

        self.index.add(vectors.astype(np.float32))
        for chunk in chunks:
            entry = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_url": chunk.source_url,
                "source_type": chunk.source_type.value,
                "title": chunk.title,
                "position": chunk.position,
            }
            # Propagar campos del metadata del chunk (incluye category para
            # query routing — Sección 5.8.4 del diseño)
            if chunk.metadata:
                for key, value in chunk.metadata.items():
                    if key not in entry:  # no sobreescribir campos base
                        entry[key] = value
            self.metadata.append(entry)
        logger.info("Añadidos %d vectores al índice (total: %d)", len(vectors), self.index.ntotal)

    # ---- Búsqueda ----

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        """Busca los *k* vectores más similares a *query_vector*.

        Returns
        -------
        list[tuple[dict, float]]
            Lista de (metadatos_chunk, score) ordenada por score descendente.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results: list[tuple[dict[str, Any], float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # FAISS devuelve -1 si no hay suficientes resultados
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    # ---- Persistencia ----

    def save(self, directory: str | Path) -> None:
        """Guarda el índice FAISS y los metadatos en *directory*.

        Se generan dos ficheros:
          - ``index.faiss``   — índice binario de FAISS
          - ``index.meta.json`` — metadatos JSON (uno por vector)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        meta_path = directory / "index.meta.json"

        faiss.write_index(self.index, str(index_path))
        meta_path.write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Índice guardado en %s (%d vectores, %.1f MB)",
            directory, self.index.ntotal,
            index_path.stat().st_size / 1_048_576,
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Carga un índice previamente guardado con :meth:`save`.

        Returns
        -------
        VectorStore
            Instancia con el índice y los metadatos restaurados.
        """
        directory = Path(directory)
        index_path = directory / "index.faiss"
        meta_path = directory / "index.meta.json"

        index = faiss.read_index(str(index_path))
        dim = index.d
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        store = cls(dim=dim)
        store.index = index
        store.metadata = metadata
        logger.info(
            "Índice cargado desde %s (%d vectores, dim=%d)",
            directory, index.ntotal, dim,
        )
        return store

    @property
    def size(self) -> int:
        """Número de vectores almacenados."""
        return self.index.ntotal
