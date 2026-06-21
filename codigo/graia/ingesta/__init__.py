"""Subsistema de ingesta de GRAIA.

Pipeline secuencial: Fetcher → Parser → Cleaner → Chunker.
Cada componente puede usarse de forma independiente o encadenado
mediante la función de conveniencia :func:`ingest_url`.
"""

from graia.ingesta.chunker import chunk_document
from graia.ingesta.cleaner import clean
from graia.ingesta.fetcher import fetch
from graia.ingesta.models import Chunk, ParsedDocument, RawDocument, SourceType
from graia.ingesta.parser import parse

__all__ = [
    "fetch",
    "parse",
    "clean",
    "chunk_document",
    "Chunk",
    "ParsedDocument",
    "RawDocument",
    "SourceType",
]


def ingest_url(
    url: str,
    *,
    user_agent: str = "GRAIA-academic-crawler/0.1",
    timeout_s: int = 30,
    max_retries: int = 3,
    respect_robots: bool = True,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 50,
    min_chunk_tokens: int = 64,
) -> list[Chunk]:
    """Función de conveniencia: ejecuta el pipeline completo para una URL.

    Retorna una lista vacía si el documento es bloqueado por robots.txt,
    no se puede descargar, o resulta demasiado corto tras la limpieza.
    """
    raw = fetch(
        url,
        user_agent=user_agent,
        timeout_s=timeout_s,
        max_retries=max_retries,
        respect_robots=respect_robots,
    )
    if raw is None:
        return []

    parsed = parse(raw)
    cleaned = clean(parsed)
    if cleaned is None:
        return []

    return chunk_document(
        cleaned,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        min_chunk_tokens=min_chunk_tokens,
    )
