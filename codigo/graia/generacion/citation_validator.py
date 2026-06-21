"""CitationValidator — validación post-hoc de citas emitidas por el LLM.

Implementa el sistema de citación híbrido descrito en la Sección 5.10:
  - Extrae los markers ``[n]`` de la respuesta generada
  - Verifica que cada marker corresponde a un chunk real del contexto
  - Marca las citas inválidas (hallucinated) y las válidas (grounded)
  - Genera un bloque de fuentes verificadas para el pie de la respuesta

La validación post-hoc complementa la inyección in-context de markers: aunque
el prompt instruye al LLM a citar solo chunks reales, no hay garantía de que
lo haga. Este validador actúa como red de seguridad.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from graia.recuperacion.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Patrón para detectar markers de citación [1], [2], [3], etc.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class CitationReport:
    """Resultado de la validación de citas de una respuesta."""
    valid_markers: list[int] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    clean_text: str = ""

    @property
    def all_valid(self) -> bool:
        """True si no hay citas alucinadas."""
        return len(self.invalid_markers) == 0

    @property
    def total_citations(self) -> int:
        return len(self.valid_markers) + len(self.invalid_markers)


def validate_citations(
    response_text: str,
    source_map: dict[int, RetrievedChunk],
) -> CitationReport:
    """Valida los markers ``[n]`` de *response_text* contra *source_map*.

    Parameters
    ----------
    response_text : str
        Texto generado por el LLM.
    source_map : dict[int, RetrievedChunk]
        Mapa marker → chunk devuelto por ``prompt_builder.get_source_map()``.

    Returns
    -------
    CitationReport
        Informe con citas válidas, inválidas y bloque de fuentes.
    """
    found_markers = sorted(set(int(m) for m in _CITATION_PATTERN.findall(response_text)))

    valid: list[int] = []
    invalid: list[int] = []

    for marker in found_markers:
        if marker in source_map:
            valid.append(marker)
        else:
            invalid.append(marker)

    if invalid:
        logger.warning(
            "Citas alucinadas detectadas: %s (válidas: %s)",
            invalid, valid,
        )

    # Construir bloque de fuentes verificadas
    sources: list[dict[str, str]] = []
    for marker in valid:
        chunk = source_map[marker]
        sources.append({
            "marker": f"[{marker}]",
            "title": chunk.title or "Sin título",
            "url": chunk.source_url,
        })

    # Limpiar citas inválidas del texto (reemplazar [n] inválido por texto vacío)
    clean_text = response_text
    for inv in invalid:
        clean_text = clean_text.replace(f"[{inv}]", "")
    # Eliminar marcadores MALFORMADOS que el LLM inventa a veces ([5.1-3], [1-3],
    # [1, 2]…): cualquier corchete con dígitos y separadores que no sea un [n] limpio.
    clean_text = re.sub(
        r"\[[\d][\d.,\-\s]*\]",
        lambda m: m.group(0) if re.fullmatch(r"\[\d+\]", m.group(0)) else "",
        clean_text,
    )
    # Normalizar espacios sobrantes (incluido antes de signos de puntuación)
    clean_text = re.sub(r"\s+([.,;:])", r"\1", clean_text)
    clean_text = re.sub(r"  +", " ", clean_text).strip()

    return CitationReport(
        valid_markers=valid,
        invalid_markers=invalid,
        sources=sources,
        clean_text=clean_text,
    )


def recover_sources(
    answer_text: str,
    retrieved_chunks: list[RetrievedChunk],
    source_map: dict[int, RetrievedChunk],
    threshold: float = 0.3,
) -> list[dict[str, str]]:
    """Recupera las fuentes de una respuesta sin marcadores ``[n]``.

    Red de seguridad frente a modelos pequeños que responden con datos del
    contexto pero omiten los marcadores de cita (dejando al usuario sin saber la
    fuente). Para cada documento recuperado, mide el solapamiento léxico entre la
    respuesta y los fragmentos de ese documento; si supera *threshold*, se
    atribuye la fuente. La atribución se **deduplica por URL** (una entrada por
    documento) y nunca inventa fuentes: solo cita documentos realmente
    recuperados cuyo contenido aparece en la respuesta.

    Parameters
    ----------
    answer_text : str
        Respuesta ya limpia (sin marcadores válidos).
    retrieved_chunks : list[RetrievedChunk]
        Todos los fragmentos recuperados (no colapsados por URL).
    source_map : dict[int, RetrievedChunk]
        Mapa marker → chunk (uno por URL) de ``get_source_map``.
    threshold : float
        Coeficiente de solapamiento mínimo (tokens compartidos / tokens del
        fragmento) para atribuir la fuente.

    Returns
    -------
    list[dict[str, str]]
        Fuentes recuperadas (marker, title, url), deduplicadas por URL y
        ordenadas por solapamiento descendente.
    """
    from graia.generacion.dedup import salient_tokens

    ans = salient_tokens(answer_text)
    if not ans:
        return []

    url_to_marker = {chunk.source_url: marker for marker, chunk in source_map.items()}

    # Mejor solapamiento por URL (deduplica documentos).
    best: dict[str, tuple[float, RetrievedChunk]] = {}
    for chunk in retrieved_chunks:
        ct = salient_tokens(chunk.text)
        if not ct:
            continue
        shared = ans & ct
        # Cobertura de la RESPUESTA: qué parte de la respuesta procede del
        # fragmento. Es robusto a respuestas cortas (p.ej. "Noviembre, Junio y
        # Septiembre"): si están contenidas en el fragmento, la cobertura es
        # alta aunque el fragmento sea largo. Se exige además un mínimo de
        # tokens compartidos para no atribuir por coincidencias triviales.
        if len(shared) < 3:
            continue
        score = len(shared) / len(ans)
        if score < threshold:
            continue
        url = chunk.source_url
        if url not in best or score > best[url][0]:
            best[url] = (score, chunk)

    sources: list[dict[str, str]] = []
    for url, (score, chunk) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        marker = url_to_marker.get(url)
        sources.append({
            "marker": f"[{marker}]" if marker else "",
            "title": chunk.title or "Sin título",
            "url": url,
        })
    return sources


def format_sources_block(report: CitationReport) -> str:
    """Genera un bloque de texto con las fuentes verificadas para mostrar al usuario."""
    if not report.sources:
        return ""

    lines = ["\n---", "**Fuentes:**"]
    for src in report.sources:
        url = src["url"]
        marker = src.get("marker", "")
        prefix = f"{marker} " if marker else ""  # las fuentes recuperadas pueden no tener marcador
        lines.append(f"- {prefix}{src['title']} — [{url}]({url})")
    return "\n".join(lines)
