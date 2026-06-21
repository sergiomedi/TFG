"""MMR — Maximal Marginal Relevance (Carbonell & Goldstein, 1998).

Implementa el algoritmo de reordenación descrito en la Sección 5.8.3 del
diseño. MMR selecciona iterativamente el documento que maximiza:

    MMR(d) = λ · sim(q, d) − (1 − λ) · max_{d_j ∈ S} sim(d, d_j)

donde λ controla el equilibrio entre relevancia y diversidad:
  - λ = 1.0 → pura relevancia (equivale a top-k estándar)
  - λ = 0.0 → pura diversidad (elige los documentos más distintos entre sí)
  - λ = 0.7 → valor por defecto justificado en el Cap. 5

Se trabaja directamente con las matrices de similitud precalculadas
(query→docs y docs→docs) para evitar re-computar embeddings.
"""

from __future__ import annotations

import numpy as np


def mmr_rerank(
    query_similarities: np.ndarray,
    doc_embeddings: np.ndarray,
    k: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """Selecciona *k* índices mediante MMR.

    Parameters
    ----------
    query_similarities : np.ndarray
        Vector de similitudes (n,) entre la query y cada candidato.
    doc_embeddings : np.ndarray
        Matriz (n, dim) de embeddings de los candidatos (normalizados).
    k : int
        Número de documentos a seleccionar.
    lambda_param : float
        Peso de relevancia vs. diversidad (0–1).

    Returns
    -------
    list[int]
        Índices de los *k* documentos seleccionados, en orden de selección.
    """
    n = len(query_similarities)
    if n == 0:
        return []
    k = min(k, n)

    # Similitud entre todos los pares de documentos (producto interno,
    # ya normalizados → equivale a coseno)
    doc_sim_matrix = doc_embeddings @ doc_embeddings.T

    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_score = -float("inf")

        for idx in remaining:
            relevance = float(query_similarities[idx])

            # Máxima similitud con los ya seleccionados
            if selected:
                redundancy = float(
                    max(doc_sim_matrix[idx, s] for s in selected)
                )
            else:
                redundancy = 0.0

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected
