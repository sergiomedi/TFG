"""Embedder — generación de vectores densos con E5 multilingüe.

Implementa el componente de embeddings descrito en la Sección 5.7.1 del diseño:
  - Modelo: ``intfloat/multilingual-e5-base`` (768 dimensiones)
  - Prefijos obligatorios: ``"query: "`` para consultas, ``"passage: "`` para chunks
  - Normalización L2 de los vectores resultantes (requisito de IndexFlatIP)
  - Codificación por lotes para eficiencia en GPU/CPU

Los prefijos ``"query: "`` / ``"passage: "`` no son una decisión propia: son una
convención EXIGIDA por el modelo E5, que se entrenó con ellos para distinguir el
rol de cada texto. Omitirlos degrada la calidad de los embeddings.
Fuente: Wang et al. (2022/2024) y la *model card* oficial:
https://huggingface.co/intfloat/multilingual-e5-base

Se emplea ``sentence-transformers`` como wrapper de alto nivel sobre
Hugging Face Transformers, lo que simplifica la gestión del tokenizador,
el pooling y la normalización sin sacrificar control sobre los prefijos.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Tipo de retorno explícito para claridad
EmbeddingMatrix = np.ndarray  # shape (n, dim), dtype float32


class Embedder:
    """Genera embeddings densos normalizados con E5 multilingüe.

    Parameters
    ----------
    model_name : str
        Identificador del modelo en Hugging Face Hub.
    query_prefix : str
        Prefijo antepuesto a las consultas del usuario.
    passage_prefix : str
        Prefijo antepuesto a los chunks del corpus.
    batch_size : int
        Tamaño de lote para ``model.encode()``.
    normalize : bool
        Si ``True``, los vectores se normalizan a norma L2 unitaria.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.normalize = normalize
        # Dimensión del embedding. El nombre del método cambia entre versiones de
        # sentence-transformers: `get_sentence_embedding_dimension()` (clásico, 2.x–5.x)
        # y `get_embedding_dimension()` (alias en versiones recientes). Soportamos ambos.
        get_dim = getattr(self.model, "get_sentence_embedding_dimension", None) \
            or getattr(self.model, "get_embedding_dimension")
        self.dim: int = int(get_dim())  # type: ignore[assignment]
        logger.info(
            "Embedder inicializado: %s (dim=%d, normalize=%s)",
            model_name, self.dim, normalize,
        )

    def encode_passages(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Codifica una lista de chunks del corpus (con prefijo ``passage:``)."""
        prefixed = [f"{self.passage_prefix}{t}" for t in texts]
        return self._encode(prefixed)

    def encode_query(self, query: str) -> np.ndarray:
        """Codifica una consulta individual (con prefijo ``query:``)."""
        prefixed = f"{self.query_prefix}{query}"
        vec = self._encode([prefixed])[0]
        return vec

    def _encode(self, texts: Sequence[str]) -> EmbeddingMatrix:
        """Codificación interna con normalización opcional."""
        embeddings: np.ndarray = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
