"""Subsistema de generación de GRAIA.

Orquesta la construcción del prompt, la invocación al LLM y la validación
de citas. Componentes:
  - :class:`OllamaClient`       — interfaz con Ollama (streaming y batch)
  - :func:`build_messages`      — prompt de sistema GRAIA + contexto con markers
  - :func:`validate_citations`  — validación post-hoc de citas [n]
  - :func:`generate`            — flujo completo de generación
"""

from graia.generacion.citation_validator import (
    CitationReport,
    format_sources_block,
    validate_citations,
)
from graia.generacion.dedup import deduplicate_sentences
from graia.generacion.ollama_client import GenerationResult, OllamaClient
from graia.generacion.prompt_builder import build_messages, get_source_map
from graia.recuperacion.retriever import RetrievedChunk

__all__ = [
    "OllamaClient",
    "GenerationResult",
    "CitationReport",
    "build_messages",
    "get_source_map",
    "validate_citations",
    "format_sources_block",
    "deduplicate_sentences",
    "generate",
]


def generate(
    query: str,
    chunks: list[RetrievedChunk],
    client: OllamaClient,
) -> tuple[GenerationResult, CitationReport]:
    """Flujo completo de generación: prompt → LLM → validación de citas.

    Parameters
    ----------
    query : str
        Consulta del usuario.
    chunks : list[RetrievedChunk]
        Chunks recuperados y reordenados por MMR.
    client : OllamaClient
        Instancia del cliente Ollama.

    Returns
    -------
    tuple[GenerationResult, CitationReport]
        Resultado de la generación y el informe de validación de citas.
    """
    system_prompt, user_message = build_messages(query, chunks)
    source_map = get_source_map(chunks)

    result = client.generate(system_prompt, user_message)
    # Deduplicación de contenido antes de validar citas (coherente con la
    # interfaz): elimina frases redundantes y, con ellas, citas sobrantes.
    result.text = deduplicate_sentences(result.text)
    report = validate_citations(result.text, source_map)

    # Reemplazar texto con versión limpia (sin citas alucinadas)
    result.text = report.clean_text

    return result, report
