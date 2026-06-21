"""Tests del subsistema de indexación de GRAIA.

Se emplean vectores sintéticos (np.random) para aislar las pruebas del modelo
de embeddings real, que requiere ~500 MB de pesos y GPU. El Embedder se testea
solo a nivel de interfaz (existencia de métodos, tipos de retorno) y con un
mock ligero; el VectorStore se testea completamente con FAISS real.
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import faiss
from graia.ingesta.models import Chunk, SourceType
from graia.indexacion.vector_store import VectorStore


# ---- Helpers ----

def _make_chunks(n: int) -> list[Chunk]:
    """Genera *n* chunks sintéticos."""
    return [
        Chunk(
            text=f"Texto del chunk {i} con contenido académico de prueba.",
            source_url=f"https://etsiit.ugr.es/page_{i // 3}",
            source_type=SourceType.HTML,
            position=i,
            char_start=i * 100,
            char_end=(i + 1) * 100,
            fetched_at=datetime.now(timezone.utc),
        )
        for i in range(n)
    ]


def _random_vectors(n: int, dim: int = 768, normalized: bool = True) -> np.ndarray:
    """Genera *n* vectores aleatorios de dimensión *dim*."""
    vecs = np.random.default_rng(42).standard_normal((n, dim)).astype(np.float32)
    if normalized:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
    return vecs


# ---- Tests del VectorStore ----

class TestVectorStore:
    def test_init_creates_empty_index(self):
        store = VectorStore(dim=768)
        assert store.size == 0
        assert store.dim == 768

    def test_add_vectors(self):
        store = VectorStore(dim=768)
        chunks = _make_chunks(10)
        vectors = _random_vectors(10)
        store.add(vectors, chunks)
        assert store.size == 10

    def test_add_validates_length_mismatch(self):
        store = VectorStore(dim=768)
        chunks = _make_chunks(5)
        vectors = _random_vectors(10)
        with pytest.raises(ValueError, match="Longitudes no coinciden"):
            store.add(vectors, chunks)

    def test_add_validates_dimension_mismatch(self):
        store = VectorStore(dim=768)
        chunks = _make_chunks(5)
        vectors = _random_vectors(5, dim=512)
        with pytest.raises(ValueError, match="Dimensión incorrecta"):
            store.add(vectors, chunks)

    def test_search_returns_correct_k(self):
        store = VectorStore(dim=768)
        chunks = _make_chunks(50)
        vectors = _random_vectors(50)
        store.add(vectors, chunks)

        query = _random_vectors(1)[0]
        results = store.search(query, k=5)
        assert len(results) == 5

    def test_search_returns_scores_descending(self):
        store = VectorStore(dim=768)
        chunks = _make_chunks(50)
        vectors = _random_vectors(50)
        store.add(vectors, chunks)

        query = _random_vectors(1)[0]
        results = store.search(query, k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_results_have_metadata(self):
        store = VectorStore(dim=768)
        chunks = _make_chunks(10)
        vectors = _random_vectors(10)
        store.add(vectors, chunks)

        query = _random_vectors(1)[0]
        results = store.search(query, k=3)
        for meta, score in results:
            assert "chunk_id" in meta
            assert "text" in meta
            assert "source_url" in meta
            assert isinstance(score, float)

    def test_search_k_exceeds_total(self):
        """Pedir más resultados que vectores indexados no debe fallar."""
        store = VectorStore(dim=768)
        chunks = _make_chunks(3)
        vectors = _random_vectors(3)
        store.add(vectors, chunks)

        results = store.search(_random_vectors(1)[0], k=100)
        assert len(results) == 3

    def test_incremental_add(self):
        """Se pueden hacer múltiples add() y el índice crece."""
        store = VectorStore(dim=768)
        store.add(_random_vectors(5), _make_chunks(5))
        store.add(_random_vectors(5), _make_chunks(5))
        assert store.size == 10

    def test_save_and_load(self, tmp_path: Path):
        store = VectorStore(dim=768)
        chunks = _make_chunks(20)
        vectors = _random_vectors(20)
        store.add(vectors, chunks)

        # Guardar
        save_dir = tmp_path / "test_index"
        store.save(save_dir)
        assert (save_dir / "index.faiss").exists()
        assert (save_dir / "index.meta.json").exists()

        # Cargar
        loaded = VectorStore.load(save_dir)
        assert loaded.size == 20
        assert loaded.dim == 768

        # Mismos resultados de búsqueda
        query = _random_vectors(1)[0]
        r1 = store.search(query, k=5)
        r2 = loaded.search(query, k=5)
        assert [m["chunk_id"] for m, _ in r1] == [m["chunk_id"] for m, _ in r2]

    def test_known_vector_is_top_result(self):
        """Un vector buscado contra sí mismo debe ser el resultado #1."""
        store = VectorStore(dim=768)
        chunks = _make_chunks(10)
        vectors = _random_vectors(10)
        store.add(vectors, chunks)

        # Buscar el vector 7 → debe aparecer primero
        results = store.search(vectors[7], k=3)
        top_meta, top_score = results[0]
        assert top_meta["chunk_id"] == chunks[7].chunk_id
        assert top_score > 0.99  # coseno ≈ 1.0 consigo mismo


# ---- Tests del Embedder (interfaz sin modelo pesado) ----

class TestEmbedderInterface:
    def test_embedder_module_importable(self):
        from graia.indexacion.embedder import Embedder
        assert Embedder is not None

    def test_embedder_has_required_methods(self):
        from graia.indexacion.embedder import Embedder
        assert hasattr(Embedder, "encode_passages")
        assert hasattr(Embedder, "encode_query")
