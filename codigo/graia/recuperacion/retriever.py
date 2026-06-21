"""Retriever — orquestación del flujo de recuperación.

Implementa el flujo completo descrito en la Sección 5.8 del diseño:

  1. Codificar la consulta con el Embedder (prefijo ``query:``)
  2. [Query Routing] Clasificar la consulta en categorías del corpus
  3. Buscar los *k_candidates* vecinos más próximos en el VectorStore (canal denso)
  4. [Opcional] Buscar los *k_candidates* más relevantes con BM25 (canal léxico)
  5. [Si híbrido] Fusionar ambos rankings con Reciprocal Rank Fusion (RRF)
  6. Aplicar *category boosting*: multiplicar score de chunks cuya categoría
     coincide con las predichas por el router
  7. Filtrar por umbral de similitud τ (descartar chunks poco relevantes)
  8. Reordenar con MMR (λ) para diversificar el contexto
  9. Devolver los *k_final* chunks seleccionados

La recuperación híbrida (densa + léxica) se activa mediante el parámetro
``use_hybrid=True``. El query routing se aplica siempre que los chunks
contengan metadata de categoría.

Cada paso es configurable vía el diccionario de parámetros del YAML.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.mmr import mmr_rerank
from graia.recuperacion.query_intent import (
    desired_structured_tipo,
    extract_curso,
    is_listing_query,
)
from graia.recuperacion.query_router import RouteResult, detect_subject_siglas, route_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Chunk recuperado con su score de similitud y posición en el ranking."""
    chunk_id: str
    text: str
    source_url: str
    source_type: str
    title: str | None
    position: int
    similarity: float
    rank: int              # posición tras MMR (0-indexed)
    metadata: dict = field(default_factory=dict)


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k_rrf: int = 60,
) -> list[tuple[str, float]]:
    """Fusiona múltiples rankings mediante Reciprocal Rank Fusion (RRF).

    Fórmula (Cormack et al., 2009):
        RRF_score(d) = Σ_{r ∈ rankings} 1 / (k + rank_r(d))

    donde *k* es una constante de suavizado (por defecto 60, valor estándar
    de la literatura) y rank_r(d) es la posición 1-indexed del documento *d*
    en el ranking *r*.

    Parameters
    ----------
    rankings : list[list[tuple[str, float]]]
        Lista de rankings; cada ranking es una lista de (chunk_id, score)
        ya ordenada por score descendente.
    k_rrf : int
        Constante de suavizado RRF. Valores típicos: 60 (estándar).

    Returns
    -------
    list[tuple[str, float]]
        Ranking fusionado, ordenado por RRF score descendente.
    """
    rrf_scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_0, (chunk_id, _original_score) in enumerate(ranking):
            rank_1 = rank_0 + 1  # RRF usa ranks 1-indexed
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank_1)

    # Ordenar por RRF score descendente
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


_SIGLA_Q_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,6})\b")
_ASIG_INTENT_RE = re.compile(r"asignaturas?|materias?|qué\s+se\s+imparte", re.IGNORECASE)


def _inject_summary_records(
    query_vector: np.ndarray,
    store: VectorStore,
    route: RouteResult,
    existing_ids: set[str],
    curso: int | None,
) -> list[tuple[dict, float]]:
    """Reúne los registros-resumen pertinentes para una consulta de listado.

    Escanea el índice en busca de chunks marcados ``is_summary`` (registros que
    enumeran las asignaturas de un curso/grupo) que (a) pertenezcan a una de las
    categorías del router —si la consulta fue enrutada— y (b) coincidan con el
    curso pedido, si la consulta menciona uno. Calcula su score denso para que
    participen del ranking. Garantiza que el dato agregado por curso/cuatrimestre
    esté disponible en el contexto aunque FAISS no lo hubiera traído al top-k.
    """
    injected: list[tuple[dict, float]] = []
    for idx, meta in enumerate(store.metadata):
        if not meta.get("is_summary"):
            continue
        if meta["chunk_id"] in existing_ids:
            continue
        if route.is_routed and meta.get("category") not in route.categories:
            continue
        # Si la consulta especifica curso, restringir a ese curso (los resúmenes
        # sin campo 'curso' no se filtran, para no perder agregados globales).
        if curso is not None and meta.get("curso") not in (None, curso):
            continue
        vec = store.index.reconstruct(idx).reshape(1, -1)
        dense_score = float(np.dot(query_vector.reshape(1, -1), vec.T)[0][0])
        injected.append((meta, dense_score))
    return injected


def _structured_boost(query: str, meta: dict) -> float:
    """Boost por coincidencia de metadatos en chunks estructurados (horarios).

    Seguro y acotado: solo afecta a chunks de horario (``tipo == 'horario'``);
    nunca penaliza. Favorece (a) el registro cuya sigla coincide exactamente con
    una sigla escrita en la consulta (p.ej. «DI», «IA»), y (b) los registros-
    resumen cuando la consulta pide listar asignaturas.
    """
    if meta.get("tipo") != "horario":
        return 1.0
    boost = 1.0
    qsig = set(_SIGLA_Q_RE.findall(query))
    if qsig & set(meta.get("siglas") or []):
        boost *= 1.6
    if meta.get("is_summary") and _ASIG_INTENT_RE.search(query):
        boost *= 1.3
    return boost


def _chunk_subject_siglas(meta: dict) -> set[str]:
    """Asignatura(s) a la(s) que pertenece un chunk, para el filtro de coherencia.

    Estrategia por capas:
      - Catálogos multi-asignatura (plan de estudios / registros-resumen) NO
        tienen una única asignatura: se devuelven vacíos para NUNCA filtrarlos.
      - Si el chunk trae ``siglas`` en metadatos (horarios, plan), se usan.
      - En su defecto (guías docentes, calendario), se deriva la asignatura del
        TÍTULO (que en las guías es inequívoco: «Guía docente de …») y, si no hay
        título, de la primera línea del texto (calendario: «… Derecho
        Informático (DI): examen …»). Reconoce sigla y nombre completo.

    Devuelve un conjunto vacío cuando el chunk no identifica una asignatura
    concreta, en cuyo caso queda EXENTO del filtro (no se descarta).
    """
    if meta.get("is_summary") or meta.get("tipo") == "plan_estudios":
        return set()
    sig = set(meta.get("siglas") or [])
    if sig:
        return sig
    text = meta.get("title") or (meta.get("text") or "").split("\n", 1)[0]
    return set(detect_subject_siglas(text))


def retrieve(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    *,
    bm25_index: BM25Index | None = None,
    use_hybrid: bool = False,
    rrf_k: int = 60,
    k_candidates: int = 20,
    k_final: int = 5,
    k_final_listing: int | None = None,
    similarity_threshold: float = 0.50,
    mmr_lambda: float = 0.7,
    use_reranker: bool = False,
    off_category_penalty: float = 0.0,
) -> list[RetrievedChunk]:
    """Ejecuta el flujo completo de recuperación para *query*.

    Parameters
    ----------
    query : str
        Consulta del usuario en lenguaje natural.
    embedder : Embedder
        Instancia inicializada del modelo de embeddings.
    store : VectorStore
        Índice FAISS con los chunks del corpus.
    bm25_index : BM25Index | None
        Índice BM25 para recuperación léxica. Requerido si ``use_hybrid=True``.
    use_hybrid : bool
        Activa la recuperación híbrida (densa + léxica + RRF).
    rrf_k : int
        Constante de suavizado para RRF (por defecto 60).
    k_candidates : int
        Número de candidatos iniciales por canal.
    k_final : int
        Número de chunks devueltos tras MMR (consultas puntuales).
    k_final_listing : int | None
        Si se indica, ``k_final`` efectivo para consultas de listado/agregación
        (detectadas con :func:`query_intent.is_listing_query`). Permite ampliar
        el contexto sólo cuando la respuesta requiere agregar muchos registros,
        sin penalizar las consultas puntuales. ``None`` → sin ampliación.
    similarity_threshold : float
        Umbral τ: se descartan chunks con score < τ (solo canal denso).
    mmr_lambda : float
        Parámetro λ de MMR (1.0 = pura relevancia, 0.0 = pura diversidad).

    Returns
    -------
    list[RetrievedChunk]
        Chunks seleccionados, ordenados por ranking MMR.
    """
    # 1. Codificar consulta
    query_vector = embedder.encode_query(query)

    # 1b. Query routing: clasificar la consulta en categorías
    route = route_query(query)

    # 1c. Intención de listado/agregación: amplía k_final y activa la inyección
    # forzada de registros-resumen (ver pasos 4c y 5). Sólo afecta a consultas
    # de listado; las puntuales conservan su k_final reducido.
    listing = bool(k_final_listing) and is_listing_query(query)
    if listing:
        k_final = k_final_listing  # type: ignore[assignment]
        logger.info(
            "Intención de listado detectada: k_final ampliado a %d para '%s'",
            k_final, query[:60],
        )

    # 2. Búsqueda densa: top-k_candidates en FAISS
    raw_results = store.search(query_vector, k=k_candidates)
    if not raw_results:
        logger.info("Sin resultados densos para la consulta: %s", query[:80])
        return []

    # 3. Recuperación híbrida (si está activada)
    if use_hybrid and bm25_index is not None:
        # Canal léxico: BM25
        bm25_results = bm25_index.search(query, k=k_candidates)

        # Preparar rankings para RRF
        # Canal denso: (chunk_id, score)
        dense_ranking = [(meta["chunk_id"], score) for meta, score in raw_results]
        # Canal léxico: ya viene como (chunk_id, score)
        lexical_ranking = bm25_results

        # Fusión RRF
        fused = _reciprocal_rank_fusion([dense_ranking, lexical_ranking], k_rrf=rrf_k)

        # Reconstruir raw_results en orden RRF, incluyendo chunks de AMBOS canales.
        # Bug fix: la versión anterior solo conservaba chunks del canal denso,
        # descartando silenciosamente los que BM25 rescataba. Ahora se buscan
        # en el store completo y se calcula su score denso bajo demanda.
        meta_by_id = {meta["chunk_id"]: (meta, score) for meta, score in raw_results}

        # Índice invertido chunk_id → posición FAISS para chunks BM25-only
        store_id_to_idx = {
            m["chunk_id"]: i for i, m in enumerate(store.metadata)
        }

        reordered = []
        bm25_rescued = 0
        for chunk_id, rrf_score in fused:
            if chunk_id in meta_by_id:
                # Chunk ya presente en el canal denso: usar su score FAISS
                meta, dense_score = meta_by_id[chunk_id]
                reordered.append((meta, dense_score))
            elif chunk_id in store_id_to_idx:
                # Chunk BM25-only: obtener metadatos del store y calcular
                # score denso reconstruyendo el vector desde FAISS
                faiss_idx = store_id_to_idx[chunk_id]
                meta = store.metadata[faiss_idx]
                vec = store.index.reconstruct(faiss_idx).reshape(1, -1)
                dense_score = float(
                    np.dot(query_vector.reshape(1, -1), vec.T)[0][0]
                )
                reordered.append((meta, dense_score))
                bm25_rescued += 1
        raw_results = reordered[:k_candidates]

        logger.debug(
            "Híbrido: denso=%d, léxico=%d → fusionados RRF=%d → "
            "retenidos=%d (BM25-only rescatados: %d)",
            len(dense_ranking), len(lexical_ranking), len(fused),
            len(raw_results), bm25_rescued,
        )
    elif use_hybrid and bm25_index is None:
        logger.warning(
            "use_hybrid=True pero no se proporcionó bm25_index; "
            "procediendo con recuperación puramente densa."
        )

    # 4. Recuperación focalizada por categoría (query routing)
    # Problema: el pool de k_candidates de FAISS puede no contener chunks de
    # la categoría correcta (e.g., "Plan de Estudios" rank #240, fuera del
    # top-50). Solución en dos fases:
    #   Fase A: pool global (ya obtenido en paso 2/3)
    #   Fase B: escanear TODOS los chunks de las categorías del router,
    #           calcular su score denso, y añadir los mejores al pool
    # Esto inyecta chunks de la categoría relevante que FAISS no seleccionó.
    if route.is_routed:
        existing_ids = {meta["chunk_id"] for meta, _ in raw_results}

        # Fase B: búsqueda focalizada en categorías del router
        category_candidates: list[tuple[dict, float]] = []
        for idx, chunk_meta in enumerate(store.metadata):
            chunk_cat = chunk_meta.get("category", "")
            if chunk_cat in route.categories and chunk_meta["chunk_id"] not in existing_ids:
                vec = store.index.reconstruct(idx).reshape(1, -1)
                dense_score = float(
                    np.dot(query_vector.reshape(1, -1), vec.T)[0][0]
                )
                category_candidates.append((chunk_meta, dense_score))

        # Tomar los mejores k_final*2 de la búsqueda focalizada
        category_candidates.sort(key=lambda x: x[1], reverse=True)
        injected = category_candidates[:k_final * 2]

        # Merge: pool global + chunks focalizados
        raw_results = raw_results + injected

        # Aplicar boosting al pool combinado
        boosted_results = []
        for meta, score in raw_results:
            chunk_category = meta.get("category", "")
            boost = route.categories.get(chunk_category, 1.0)
            boosted_results.append((meta, score * boost))
        raw_results = sorted(boosted_results, key=lambda x: x[1], reverse=True)

        logger.info(
            "Query routing: %d chunks inyectados de categorías %s "
            "(pool total: %d)",
            len(injected), list(route.categories.keys()), len(raw_results),
        )

    # 4b. Boost por metadatos estructurados (solo afecta a chunks de horario)
    if any(meta.get("tipo") == "horario" for meta, _ in raw_results):
        raw_results = sorted(
            ((meta, score * _structured_boost(query, meta)) for meta, score in raw_results),
            key=lambda x: x[1], reverse=True,
        )

    # 4b-bis. Filtro de coherencia de asignatura (anti-contaminación cruzada).
    # Si la consulta nombra una o varias asignaturas concretas (p.ej. «DI»), los
    # fragmentos que pertenecen a OTRA asignatura son ruido: el LLM tiende a
    # fusionarlos o a citarlos como si fueran de la asignatura preguntada (p.ej.
    # presentar el horario de TID como un cuatrimestre de DI, o citar la guía
    # docente de IA al preguntar la nota mínima de DI). Se descartan aquí, de
    # forma determinista, comparando la asignatura del chunk (horarios, calendario
    # y guías docentes; ver _chunk_subject_siglas) con la(s) de la consulta. Los
    # chunks sin asignatura concreta (catálogos, planes, genéricos) quedan exentos.
    q_subjects = set(detect_subject_siglas(query))
    if q_subjects:
        before = len(raw_results)
        kept: list[tuple[dict, float]] = []
        for meta, score in raw_results:
            subs = _chunk_subject_siglas(meta)
            if subs and not (subs & q_subjects):
                continue  # chunk de otra asignatura → descartar
            kept.append((meta, score))
        raw_results = kept
        if len(raw_results) != before:
            logger.info(
                "Filtro de asignatura %s: %d→%d chunks (descartados fragmentos de otras asignaturas)",
                sorted(q_subjects), before, len(raw_results),
            )

    # 4b-ter. Filtro de coherencia de TIPO (anti-contaminación entre registros
    # estructurados). El horario de clase y el calendario de exámenes comparten
    # casi todo el vocabulario (asignatura, curso, cuatrimestre), por lo que se
    # desplazan y mezclan mutuamente: una pregunta de "horario de DI" arrastra sus
    # exámenes, y una de "examen de Cálculo" arrastra su horario, restando sitio en
    # el top-k al registro correcto. Cuando la consulta pide INEQUÍVOCAMENTE uno de
    # los dos tipos, se descarta el otro. Es simétrico al filtro de asignatura y
    # solo afecta a registros estructurados (tipo horario/calendario); guías,
    # normativa, planes y demás quedan intactos. Si la intención es ambigua
    # (menciona ambos o ninguno), no se filtra, para no provocar falsos rechazos.
    desired_tipo = desired_structured_tipo(query)
    if desired_tipo is not None:
        conflicting = {"horario", "calendario"} - {desired_tipo}
        before = len(raw_results)
        raw_results = [
            (meta, score) for meta, score in raw_results
            if meta.get("tipo") not in conflicting
        ]
        if len(raw_results) != before:
            logger.info(
                "Filtro de tipo (%s): %d→%d chunks (descartados registros de tipo %s)",
                desired_tipo, before, len(raw_results), sorted(conflicting),
            )

    # 4c. Inyección forzada de registros-resumen (sólo consultas de listado).
    # Garantiza que el dato agregado por curso/cuatrimestre entre en el pool,
    # con independencia del ranking denso/léxico. Sus chunks quedan exentos del
    # umbral τ (paso 5) para que sobrevivan hasta el MMR.
    # La inyección de registros-resumen solo tiene sentido para consultas sobre
    # ASIGNATURAS/materias (no para "clases de X de todos los grupos", que es un
    # listado de horarios y no debe contaminarse con líneas de resumen). El
    # k ampliado (paso 1c) sí se mantiene para cualquier listado.
    summary_ids: set[str] = set()
    if listing and _ASIG_INTENT_RE.search(query):
        existing_ids = {meta["chunk_id"] for meta, _ in raw_results}
        curso = extract_curso(query)
        summaries = _inject_summary_records(
            query_vector, store, route, existing_ids, curso,
        )
        raw_results = raw_results + summaries
        # El conjunto de exención del umbral se construye sobre TODO el pool, no
        # solo sobre los recién inyectados: los registros-resumen pueden haber
        # entrado ya en la Fase B (inyección por categoría), y también deben
        # quedar exentos. Se respetan los filtros de categoría y curso.
        for meta, _ in raw_results:
            if not meta.get("is_summary"):
                continue
            if route.is_routed and meta.get("category") not in route.categories:
                continue
            if curso is not None and meta.get("curso") not in (None, curso):
                continue
            summary_ids.add(meta["chunk_id"])
        logger.info(
            "Listado: %d resúmenes inyectados, %d exentos de umbral (curso=%s)",
            len(summaries), len(summary_ids), curso,
        )

    # 5. Filtrado por umbral τ
    # En modo híbrido se aplica un umbral reducido (τ * 0.7) para no descartar
    # chunks que BM25 rescató por coincidencia léxica exacta pero cuyo score
    # denso es moderado. La presencia en el ranking RRF ya indica relevancia.
    # Los registros-resumen inyectados para un listado quedan EXENTOS del umbral.
    effective_threshold = (
        similarity_threshold * 0.7
        if (use_hybrid and bm25_index is not None)
        else similarity_threshold
    )
    filtered = [
        (meta, score) for meta, score in raw_results
        if score >= effective_threshold or meta["chunk_id"] in summary_ids
    ]
    if not filtered:
        logger.info(
            "Todos los candidatos bajo el umbral τ=%.2f (efectivo=%.2f) para: %s",
            similarity_threshold, effective_threshold, query[:80],
        )
        return []

    logger.debug(
        "Candidatos: %d → tras umbral τ=%.2f (efectivo=%.2f): %d",
        len(raw_results), similarity_threshold, effective_threshold, len(filtered),
    )

    # 5b. Reranking con cross-encoder (opcional)
    # El cross-encoder evalúa (query, passage) conjuntamente, capturando
    # interacciones semánticas que el bi-encoder pierde. Se aplica ANTES
    # del MMR para que la diversificación opere sobre un pool ya
    # reordenado por relevancia precisa.
    if use_reranker:
        try:
            from graia.recuperacion.reranker import rerank as ce_rerank

            rerank_texts = [meta["text"] for meta, _ in filtered]
            reranked = ce_rerank(query, rerank_texts, top_k=len(filtered))

            # Reordenar filtered según el ranking del cross-encoder
            filtered = [(filtered[idx][0], ce_score) for idx, ce_score in reranked]

            logger.info(
                "Cross-encoder reranking: %d candidatos reordenados "
                "(top=%.4f, bottom=%.4f)",
                len(filtered),
                filtered[0][1] if filtered else 0,
                filtered[-1][1] if filtered else 0,
            )
        except Exception as exc:  # noqa: BLE001
            # Si el reranker falla (modelo no disponible, etc.) se degrada con
            # elegancia: se mantiene el orden previo en lugar de romper la consulta.
            logger.warning("Reranking omitido por error: %s", exc)

    # 5c. Penalización por categoría no enrutada (control de ruido). Cuando el
    # router ha clasificado la consulta con confianza, los fragmentos de
    # categorías AJENAS a las enrutadas suelen ser ruido (p.ej. boletines
    # oficiales o FAQ de prácticas colándose en «¿en qué cursos van las
    # especialidades?»). Se RESTA una penalización al score del reranker para
    # demotarlos en la selección final, SIN eliminarlos: si fueran el único
    # candidato, sobreviven (no se provocan falsos rechazos). Solo se aplica con
    # reranker activo (escala de score conocida) y router confiable.
    if use_reranker and off_category_penalty > 0 and route.is_routed:
        penalized = 0
        adjusted = []
        for meta, score in filtered:
            if meta.get("category") not in route.categories:
                score -= off_category_penalty
                penalized += 1
            adjusted.append((meta, score))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        filtered = adjusted
        if penalized:
            logger.info(
                "Penalización de ruido: %d/%d chunks de categoría no enrutada (−%.1f)",
                penalized, len(filtered), off_category_penalty,
            )

    # 6. Reordenación MMR
    candidate_texts = [meta["text"] for meta, _ in filtered]
    candidate_embeddings = embedder.encode_passages(candidate_texts)
    query_sims = np.array([score for _, score in filtered], dtype=np.float32)

    selected_indices = mmr_rerank(
        query_similarities=query_sims,
        doc_embeddings=candidate_embeddings,
        k=k_final,
        lambda_param=mmr_lambda,
    )

    # 7. Construir resultado final
    results: list[RetrievedChunk] = []
    for rank, idx in enumerate(selected_indices):
        meta, score = filtered[idx]
        results.append(
            RetrievedChunk(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                source_url=meta["source_url"],
                source_type=meta["source_type"],
                title=meta.get("title"),
                position=meta["position"],
                similarity=score,
                rank=rank,
                metadata={k: v for k, v in meta.items()
                          if k not in {"chunk_id", "text", "source_url",
                                       "source_type", "title", "position"}},
            )
        )

    logger.info(
        "Recuperados %d chunks para: '%s' (de %d candidatos, %d tras umbral%s)",
        len(results), query[:60], len(raw_results), len(filtered),
        ", híbrido RRF" if (use_hybrid and bm25_index is not None) else "",
    )
    return results
