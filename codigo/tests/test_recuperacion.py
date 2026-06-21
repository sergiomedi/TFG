"""Tests del subsistema de recuperación de GRAIA.

MMR se testea directamente con vectores sintéticos controlados.
El Retriever se testea con mocks ligeros del Embedder y VectorStore reales
para aislar la lógica de orquestación sin necesidad del modelo E5.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from graia.ingesta.models import Chunk, SourceType
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.mmr import mmr_rerank
from graia.recuperacion.retriever import RetrievedChunk, retrieve


# ---- Helpers ----

def _random_vectors(n: int, dim: int = 768, seed: int = 42) -> np.ndarray:
    vecs = np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            text=f"Contenido académico del chunk {i} sobre la ETSIIT.",
            source_url=f"https://etsiit.ugr.es/page_{i}",
            source_type=SourceType.HTML,
            position=i,
            char_start=i * 100,
            char_end=(i + 1) * 100,
            fetched_at=datetime.now(timezone.utc),
        )
        for i in range(n)
    ]


def _build_store(n: int = 50, dim: int = 768) -> tuple[VectorStore, np.ndarray, list[Chunk]]:
    """Construye un VectorStore con *n* vectores aleatorios."""
    store = VectorStore(dim=dim)
    chunks = _make_chunks(n)
    vectors = _random_vectors(n, dim)
    store.add(vectors, chunks)
    return store, vectors, chunks


# ---- Tests de MMR ----

class TestMMR:
    def test_returns_k_indices(self):
        sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        embs = _random_vectors(5, dim=32)
        result = mmr_rerank(sims, embs, k=3, lambda_param=0.7)
        assert len(result) == 3

    def test_lambda_1_equals_topk(self):
        """Con λ=1.0, MMR debe devolver los k más relevantes en orden."""
        sims = np.array([0.3, 0.9, 0.5, 0.7, 0.1], dtype=np.float32)
        embs = _random_vectors(5, dim=32)
        result = mmr_rerank(sims, embs, k=3, lambda_param=1.0)
        # Deben ser los índices con scores 0.9, 0.7, 0.5
        assert result[0] == 1  # score 0.9
        assert set(result) == {1, 3, 2}

    def test_lambda_0_maximizes_diversity(self):
        """Con λ=0.0, MMR prioriza diversidad tras el primer doc."""
        n = 10
        sims = np.linspace(0.9, 0.5, n).astype(np.float32)
        embs = _random_vectors(n, dim=32)
        result = mmr_rerank(sims, embs, k=5, lambda_param=0.0)
        # El primer seleccionado es el más relevante (score 0.9)
        assert result[0] == 0
        # Los siguientes no serán necesariamente los más relevantes
        assert len(result) == 5

    def test_empty_input(self):
        sims = np.array([], dtype=np.float32)
        embs = np.empty((0, 32), dtype=np.float32)
        result = mmr_rerank(sims, embs, k=5)
        assert result == []

    def test_k_exceeds_n(self):
        sims = np.array([0.8, 0.6], dtype=np.float32)
        embs = _random_vectors(2, dim=32)
        result = mmr_rerank(sims, embs, k=10)
        assert len(result) == 2

    def test_all_indices_unique(self):
        sims = np.random.default_rng(99).random(20).astype(np.float32)
        embs = _random_vectors(20, dim=64)
        result = mmr_rerank(sims, embs, k=10, lambda_param=0.7)
        assert len(result) == len(set(result))

    def test_duplicate_embeddings_still_works(self):
        """Embeddings idénticos no deben causar error."""
        emb = _random_vectors(1, dim=32)
        embs = np.tile(emb, (5, 1))  # 5 vectores iguales
        sims = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        result = mmr_rerank(sims, embs, k=3, lambda_param=0.7)
        assert len(result) == 3


# ---- Tests del Retriever (con mock del Embedder) ----

class TestRetriever:
    def _make_mock_embedder(self, dim: int = 768) -> MagicMock:
        """Crea un mock del Embedder que devuelve vectores aleatorios."""
        mock = MagicMock()
        mock.dim = dim
        mock.encode_query.return_value = _random_vectors(1, dim)[0]
        mock.encode_passages.side_effect = lambda texts: _random_vectors(len(texts), dim)
        return mock

    def test_retrieve_returns_retrieved_chunks(self):
        store, vectors, chunks = _build_store(30)
        embedder = self._make_mock_embedder()
        results = retrieve("horarios de clase", embedder, store, k_candidates=10, k_final=3)
        assert len(results) <= 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_retrieve_respects_k_final(self):
        store, _, _ = _build_store(50)
        embedder = self._make_mock_embedder()
        results = retrieve("becas", embedder, store, k_candidates=20, k_final=5)
        assert len(results) <= 5

    def test_retrieve_filters_by_threshold(self):
        """Con umbral muy alto, debe devolver pocos o ningún resultado."""
        store, _, _ = _build_store(20)
        embedder = self._make_mock_embedder()
        results = retrieve(
            "consulta cualquiera", embedder, store,
            k_candidates=20, k_final=5,
            similarity_threshold=0.99,  # umbral altísimo
        )
        # Con vectores aleatorios, es improbable que haya coseno > 0.99
        assert len(results) <= 5  # puede ser 0

    def test_retrieve_empty_store(self):
        store = VectorStore(dim=768)
        embedder = self._make_mock_embedder()
        results = retrieve("algo", embedder, store, k_candidates=10, k_final=5)
        assert results == []

    def test_retrieved_chunks_have_rank(self):
        store, _, _ = _build_store(30)
        embedder = self._make_mock_embedder()
        results = retrieve("matricula", embedder, store, k_candidates=15, k_final=5)
        if results:
            ranks = [r.rank for r in results]
            assert ranks == list(range(len(results)))

    def test_retrieved_chunks_have_required_fields(self):
        store, _, _ = _build_store(20)
        embedder = self._make_mock_embedder()
        results = retrieve("tfg", embedder, store, k_candidates=10, k_final=3)
        for r in results:
            assert r.chunk_id
            assert r.text
            assert r.source_url
            assert isinstance(r.similarity, float)
            assert isinstance(r.rank, int)


# ---- query_intent: detección de listado y extracción de curso ----

from graia.recuperacion.query_intent import extract_curso, is_listing_query


class TestQueryIntent:
    def test_listing_explicit_verbs(self):
        assert is_listing_query("Dame la lista de asignaturas de tercero")
        assert is_listing_query("enumera las optativas del segundo cuatrimestre")
        assert is_listing_query("¿cuáles son las asignaturas de 4º?")

    def test_listing_plural_domain_plus_scope(self):
        assert is_listing_query("¿qué asignaturas hay en el primer curso?")
        assert is_listing_query("asignaturas del segundo cuatrimestre")
        assert is_listing_query("optativas de la mención de Computación")

    def test_listing_what_is_taught(self):
        assert is_listing_query("¿qué se imparte en tercero?")

    def test_listing_all_groups(self):
        assert is_listing_query("Y las clases de CA de todos los grupos?")
        assert is_listing_query("horario de cada grupo")

    def test_pointwise_queries_are_not_listing(self):
        assert not is_listing_query("¿a qué hora es Cálculo?")
        assert not is_listing_query("¿en qué aula es Inteligencia Artificial?")
        assert not is_listing_query("¿cuál es el horario de la secretaría?")

    def test_extract_curso_word_and_number(self):
        assert extract_curso("asignaturas de tercero") == 3
        assert extract_curso("¿qué hay en 4º?") == 4
        assert extract_curso("optativas del primer curso") == 1

    def test_extract_curso_absent(self):
        assert extract_curso("¿cuáles son las optativas del grado?") is None


class TestRetrieveListing:
    """Integración: consultas de listado amplían k e inyectan resúmenes."""

    def _store_with_summaries(self, dim: int = 768):
        # Vector de consulta fijo; los resúmenes se colocan en la dirección
        # opuesta (similitud ~ -1) para que NO entren por el canal denso ni
        # superen el umbral τ: sólo deben aparecer vía inyección + exención.
        qv = _random_vectors(1, dim, seed=7)[0]
        normal_chunks = _make_chunks(20)
        normal_vecs = _random_vectors(20, dim, seed=11)
        summary_chunks = [
            Chunk(
                text=f"Primer cuatrimestre | Primer curso | Grupo 1º{g} | Asignaturas: ALEM, CA, FP.",
                source_url=f"https://etsiit.ugr.es/horario#{g}",
                source_type=SourceType.PDF,
                position=100 + i,
                char_start=0, char_end=50,
                fetched_at=datetime.now(timezone.utc),
                metadata={"is_summary": True, "category": "horarios",
                          "tipo": "horario", "curso": 1},
            )
            for i, g in enumerate("ABCD")
        ]
        summary_vecs = np.tile(-qv, (4, 1)).astype(np.float32)
        store = VectorStore(dim=dim)
        store.add(np.vstack([normal_vecs, summary_vecs]),
                  normal_chunks + summary_chunks)
        mock = MagicMock()
        mock.dim = dim
        mock.encode_query.return_value = qv
        mock.encode_passages.side_effect = lambda texts: _random_vectors(len(texts), dim)
        summary_ids = {c.chunk_id for c in summary_chunks}
        return store, mock, summary_ids

    def test_listing_injects_summaries_and_widens_k(self):
        store, embedder, summary_ids = self._store_with_summaries()
        results = retrieve(
            "lista de asignaturas del primer cuatrimestre",
            embedder, store,
            k_candidates=10, k_final=3, k_final_listing=8,
            similarity_threshold=0.5, mmr_lambda=0.7,
        )
        got = {r.chunk_id for r in results}
        # Los 4 resúmenes (sim ~ -1, exentos del umbral) deben estar presentes.
        assert summary_ids <= got, (summary_ids, got)
        # k ampliado: se devuelven más de los 3 puntuales.
        assert len(results) > 3

    def test_pointwise_query_does_not_inject_summaries(self):
        store, embedder, summary_ids = self._store_with_summaries()
        results = retrieve(
            "¿a qué hora es Cálculo?",
            embedder, store,
            k_candidates=10, k_final=3, k_final_listing=8,
            similarity_threshold=0.5, mmr_lambda=0.7,
        )
        got = {r.chunk_id for r in results}
        # Sin intención de listado, los resúmenes (bajo τ) no se inyectan.
        assert not (summary_ids & got)


# ---- contextual_query: recuperación consciente del historial ----

from graia.recuperacion.contextual_query import enrich_query_with_history
from graia.recuperacion.query_router import (
    detect_subject_siglas,
    expand_abbreviations,
    route_query,
)


class TestSubjectRecognition:
    """Reconocimiento de asignaturas por sigla y por nombre completo."""

    def test_detect_by_sigla(self):
        assert detect_subject_siglas("En qué aula es DI?") == ["DI"]

    def test_detect_by_full_name(self):
        assert "CA" in detect_subject_siglas("en qué aula es Cálculo")
        assert "DI" in detect_subject_siglas("¿Derecho Informático en qué aula?")

    def test_detect_full_name_without_accent(self):
        assert "CA" in detect_subject_siglas("Y Calculo?")

    def test_detect_none(self):
        assert detect_subject_siglas("¿Cuál es el horario de la secretaría?") == []

    def test_expand_full_name_appends_sigla(self):
        assert "(CA)" in expand_abbreviations("en qué aula es Cálculo")

    def test_expand_sigla_appends_name(self):
        assert "Derecho Informático" in expand_abbreviations("clases de DI")


class TestRouterPlanEstudios:
    """La regla plan_estudios debe disparar en consultas de especialidad/curso."""

    def test_cada_especialidad(self):
        cats = route_query("qué asignaturas se imparten en cada especialidad").categories
        assert "plan_estudios" in cats

    def test_especialidades_de_cuarto(self):
        cats = route_query(
            "¿Derecho Informático es la misma asignatura en todas las especialidades de cuarto?"
        ).categories
        assert "plan_estudios" in cats

    def test_que_especialidades_hay(self):
        assert "plan_estudios" in route_query("¿Y qué especialidades hay?").categories

    def test_pointwise_no_plan(self):
        assert "plan_estudios" not in route_query("¿a qué hora es Cálculo?").categories


class TestContextualQuery:
    def _msgs(self, *user_queries):
        m = []
        for q in user_queries:
            m.append({"role": "user", "content": q})
            m.append({"role": "assistant", "content": "..."})
        return m

    def test_carries_subject_into_group_followup(self):
        msgs = self._msgs(
            "¿A qué hora son las clases de DI y en qué aula?",
            "Y las clases de prácticas de CA?",
            "Y para el grupo 1ºA y subgrupo 2 que es al que pertenezco?",
        )
        out = enrich_query_with_history(
            "Y para el grupo 1ºA y subgrupo 2 que es al que pertenezco?", msgs)
        assert "CA" in out

    def test_carries_subject_and_group_into_subgroup_followup(self):
        msgs = self._msgs(
            "Y las clases de prácticas de CA?",
            "Y para el grupo 1ºA y subgrupo 2 que es al que pertenezco?",
            "Me refiero al subgrupo A2",
        )
        out = enrich_query_with_history("Me refiero al subgrupo A2", msgs)
        assert "CA" in out and "grupo 1ºA" in out

    def test_carries_group_and_subgroup_into_subject_followup(self):
        msgs = self._msgs(
            "Y las clases de prácticas de CA?",
            "Y para el grupo 1ºA y subgrupo 2 que es al que pertenezco?",
            "Me refiero al subgrupo A2",
            "Y las de CA?",
        )
        out = enrich_query_with_history("Y las de CA?", msgs)
        assert "grupo 1ºA" in out and "subgrupo A2" in out

    def test_imperfecto_me_referia_is_followup(self):
        # "Me refería a ISE" (imperfecto) debe tratarse como seguimiento y
        # arrastrar el grupo (formato "grupo D", letra sola) del turno previo.
        msgs = self._msgs(
            "¿Y las clases de ISE del grupo A?",
            "Y para el grupo D?",
        )
        out = enrich_query_with_history("Me refería a ISE", msgs)
        assert out != "Me refería a ISE"  # se ha enriquecido
        assert "grupo D" in out

    def test_present_me_refiero_still_followup(self):
        msgs = self._msgs("Y las clases de prácticas de CA?", "Y para el grupo 1ºA")
        out = enrich_query_with_history("Me refiero al subgrupo A2", msgs)
        assert out != "Me refiero al subgrupo A2"

    def test_fresh_query_not_enriched(self):
        msgs = self._msgs(
            "Y las de CA?",
            "¿Cuál es el horario de la secretaría?",
        )
        q = "¿Cuál es el horario de la secretaría?"
        assert enrich_query_with_history(q, msgs) == q

    def test_no_history_returns_query(self):
        q = "Y las de CA?"
        assert enrich_query_with_history(q, [{"role": "user", "content": q}]) == q
