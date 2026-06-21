"""Tests del módulo BM25 y la recuperación híbrida con RRF.

Verifica:
  - Construcción y búsqueda del índice BM25
  - Tokenización en español
  - Persistencia (save/load)
  - Reciprocal Rank Fusion (RRF)
  - Integración del retriever con modo híbrido
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from graia.recuperacion.bm25 import BM25Index, _tokenize
from graia.recuperacion.retriever import _reciprocal_rank_fusion, retrieve


# ---- Datos de prueba ----

CORPUS_TEXTS = [
    "Horarios de clase del Grado en Ingeniería Informática para el curso 2025-2026",
    "Normativa de Trabajo Fin de Grado de la ETSIIT Universidad de Granada",
    "Guía docente de la asignatura Inteligencia Artificial IC curso 2025",
    "Procedimiento de solicitud de beca MEC para estudiantes universitarios",
    "Calendario académico oficial de la Universidad de Granada 2025-2026",
    "Requisitos de matrícula para el Grado en Telecomunicación ETSIIT",
    "Plan de estudios del Doble Grado Informática y Matemáticas",
    "Normativa de evaluación y calificación de la UGR vigente",
]

CORPUS_IDS = [f"chunk_{i}" for i in range(len(CORPUS_TEXTS))]


# ---- Tests de tokenización ----

class TestTokenize:
    def test_lowercase(self):
        tokens = _tokenize("ETSIIT Universidad de Granada")
        assert all(t == t.lower() for t in tokens)

    def test_spanish_chars(self):
        tokens = _tokenize("matrícula evaluación año calificación")
        assert "matrícula" in tokens
        assert "evaluación" in tokens
        assert "año" in tokens

    def test_strips_punctuation(self):
        tokens = _tokenize("Hola, ¿qué tal? (bien)")
        assert "," not in tokens
        assert "?" not in tokens
        assert "hola" in tokens
        assert "qué" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_preserved(self):
        tokens = _tokenize("curso 2025-2026")
        assert "2025" in tokens
        assert "2026" in tokens


# ---- Tests del índice BM25 ----

class TestBM25Index:
    def test_build_and_search(self):
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)
        assert idx.size == len(CORPUS_TEXTS)

        results = idx.search("Trabajo Fin de Grado normativa", k=3)
        assert len(results) > 0
        assert len(results) <= 3
        # El chunk sobre TFG debería estar en los resultados
        ids_returned = [cid for cid, _ in results]
        assert "chunk_1" in ids_returned  # normativa TFG

    def test_search_returns_scores_descending(self):
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)
        results = idx.search("beca universitarios", k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_no_match(self):
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)
        results = idx.search("xyzzyqwert", k=5)
        # Sin tokens en común, no debería haber resultados con score > 0
        assert len(results) == 0

    def test_search_empty_query(self):
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)
        results = idx.search("", k=5)
        assert results == []

    def test_build_mismatched_lengths(self):
        idx = BM25Index()
        with pytest.raises(ValueError, match="Longitudes"):
            idx.build(CORPUS_TEXTS, CORPUS_IDS[:3])

    def test_search_before_build_raises(self):
        idx = BM25Index()
        with pytest.raises(RuntimeError, match="no ha sido construido"):
            idx.search("algo")

    def test_save_and_load(self):
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)

        with tempfile.TemporaryDirectory() as tmpdir:
            idx.save(tmpdir)
            loaded = BM25Index.load(tmpdir)

            assert loaded.size == idx.size
            # Los resultados de búsqueda deben ser idénticos
            r_orig = idx.search("horarios clase", k=3)
            r_loaded = loaded.search("horarios clase", k=3)
            assert [cid for cid, _ in r_orig] == [cid for cid, _ in r_loaded]

    def test_k_larger_than_corpus(self):
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)
        results = idx.search("Granada universidad", k=100)
        assert len(results) <= len(CORPUS_TEXTS)

    def test_exact_code_match(self):
        """BM25 debe encontrar coincidencias exactas con códigos de asignatura."""
        idx = BM25Index()
        idx.build(CORPUS_TEXTS, CORPUS_IDS)
        results = idx.search("IC", k=3)
        ids_returned = [cid for cid, _ in results]
        # chunk_2 contiene "IC"
        assert "chunk_2" in ids_returned


# ---- Tests de Reciprocal Rank Fusion ----

class TestRRF:
    def test_single_ranking(self):
        ranking = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        fused = _reciprocal_rank_fusion([ranking], k_rrf=60)
        # Orden debe mantenerse
        assert [cid for cid, _ in fused] == ["a", "b", "c"]

    def test_two_rankings_agreement(self):
        """Si ambos rankings coinciden, el orden se mantiene."""
        r1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        r2 = [("a", 5.0), ("b", 3.0), ("c", 1.0)]
        fused = _reciprocal_rank_fusion([r1, r2], k_rrf=60)
        assert fused[0][0] == "a"

    def test_two_rankings_disagreement(self):
        """Documento presente en ambos rankings debe tener mayor RRF score."""
        r1 = [("a", 0.9), ("b", 0.8)]
        r2 = [("c", 5.0), ("b", 3.0)]
        fused = _reciprocal_rank_fusion([r1, r2], k_rrf=60)
        fused_ids = [cid for cid, _ in fused]
        # "b" aparece en ambos rankings → mayor RRF score
        assert fused_ids[0] == "b"

    def test_empty_rankings(self):
        fused = _reciprocal_rank_fusion([], k_rrf=60)
        assert fused == []

    def test_rrf_scores_positive(self):
        r1 = [("a", 0.9), ("b", 0.5)]
        fused = _reciprocal_rank_fusion([r1], k_rrf=60)
        for _, score in fused:
            assert score > 0

    def test_different_k_rrf(self):
        """k_rrf mayor suaviza las diferencias entre posiciones."""
        r1 = [("a", 0.9), ("b", 0.8)]
        fused_low = _reciprocal_rank_fusion([r1], k_rrf=1)
        fused_high = _reciprocal_rank_fusion([r1], k_rrf=1000)
        # Con k alto, la diferencia de score entre a y b es menor
        diff_low = fused_low[0][1] - fused_low[1][1]
        diff_high = fused_high[0][1] - fused_high[1][1]
        assert diff_low > diff_high
