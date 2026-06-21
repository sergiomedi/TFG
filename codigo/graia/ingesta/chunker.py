"""Chunker — fragmentación recursiva de texto con solapamiento y
propagación de cabeceras contextuales (*contextual chunk headers*).

Implementa la estrategia de chunking recursivo descrita en la Sección 5.6.1
del diseño, con la extensión de *contextual chunk headers*:

  - Tamaño objetivo: 512 tokens (aprox. 4 caracteres/token en español)
  - Solapamiento: 50 tokens entre chunks consecutivos (intra-sección)
  - Separadores jerárquicos: ``\\n\\n`` → ``\\n`` → ``. `` → `` ``
  - Chunks por debajo de ``min_chunk_tokens`` se fusionan con el anterior

Contextual Chunk Headers (inspirado en Anthropic, 2024):
  - El ``HtmlParser`` inyecta marcadores ``§H<n>§ <texto>`` para cabeceras.
  - Este módulo también detecta cabeceras en texto plano (fallback).
  - Cada sección se fragmenta de forma independiente y sus chunks
    reciben como prefijo la ruta jerárquica de cabeceras vigente.
  - El solapamiento solo se aplica entre sub-chunks de la MISMA sección,
    evitando mezclar contenido de distintas secciones.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from graia.ingesta.models import Chunk, ParsedDocument

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]
_HEADER_MARKER_RE = re.compile(r"^§H(\d)§\s*(.+)$", re.MULTILINE)
_PLAIN_HEADER_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(Primer|Segundo|Tercer|Cuarto)\s+(curso|semestre)$", re.MULTILINE),
]


def _token_len(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _recursive_split(text: str, max_tokens: int, separators: list[str]) -> list[str]:
    """Divide *text* en fragmentos de a lo sumo *max_tokens* tokens.

    Versión **verdaderamente recursiva** (corrige el bug de v1): cuando un
    fragmento resultante de un separador sigue excediendo ``max_tokens``, se
    vuelve a dividir con los separadores *restantes*, y como último recurso se
    trocea por caracteres. Así se garantiza que **ningún** fragmento de salida
    supere el tamaño objetivo (la v1 devolvía fragmentos sobredimensionados en
    cuanto encontraba el primer separador presente, generando los 71 chunks de
    >3000 caracteres observados en el análisis del corpus).
    """
    if _token_len(text) <= max_tokens:
        return [text]

    # Sin más separadores: troceo duro por caracteres (garantía de tamaño)
    if not separators:
        hard = max_tokens * _CHARS_PER_TOKEN
        return [text[i:i + hard] for i in range(0, len(text), hard)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _recursive_split(text, max_tokens, rest)

    parts = text.split(sep)
    fragments: list[str] = []
    current = ""

    def _flush(piece: str) -> None:
        """Añade *piece*; si aún excede el tamaño, lo subdivide con *rest*."""
        piece = piece.strip()
        if not piece:
            return
        if _token_len(piece) <= max_tokens:
            fragments.append(piece)
        else:
            fragments.extend(_recursive_split(piece, max_tokens, rest))

    for part in parts:
        candidate = part if not current else current + sep + part
        if _token_len(candidate) <= max_tokens:
            current = candidate
        else:
            _flush(current)
            current = part
    _flush(current)
    return fragments


def _extract_sections(text: str) -> list[tuple[str | None, str]]:
    """Extrae secciones del texto, cada una con su header path.

    Returns
    -------
    list[tuple[str | None, str]]
        Lista de (header_path_o_None, texto_de_la_sección).
        El preámbulo (texto antes de cualquier cabecera) tiene header=None.
    """
    active_headers: dict[int, str] = {}
    # Acumulamos líneas por sección
    sections: list[tuple[str | None, list[str]]] = []
    current_header: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        # Check §Hn§ markers
        marker = _HEADER_MARKER_RE.match(line.strip())
        detected_level = None
        detected_heading = None

        if marker:
            detected_level = int(marker.group(1))
            detected_heading = marker.group(2).strip()
        else:
            # Check plain-text header patterns
            for pattern in _PLAIN_HEADER_PATTERNS:
                m = pattern.match(line.strip())
                if m:
                    heading = m.group(0).strip()
                    if "curso" in heading.lower():
                        detected_level = 2
                    elif "semestre" in heading.lower():
                        detected_level = 3
                    else:
                        detected_level = 2
                    detected_heading = heading
                    break

        if detected_level is not None and detected_heading is not None:
            # Actualizar headers activos
            active_headers[detected_level] = detected_heading
            for lvl in list(active_headers):
                if lvl > detected_level:
                    del active_headers[lvl]
            new_path = " > ".join(active_headers[k] for k in sorted(active_headers))

            # ¿Es un cambio de sección significativo?
            # Solo crear nueva sección si cambia la ruta completa
            if new_path != current_header:
                # Guardar sección anterior si tiene contenido
                if current_lines:
                    sections.append((current_header, current_lines))
                current_header = new_path
                current_lines = []

            # Para markers §Hn§: no añadir la línea al texto
            # Para plain-text headers: sí añadirla (es contenido visible)
            if not marker:
                current_lines.append(line)
        else:
            current_lines.append(line)

    # Última sección
    if current_lines:
        sections.append((current_header, current_lines))

    # Convertir a (header, texto)
    result: list[tuple[str | None, str]] = []
    for header, lines in sections:
        text_block = "\n".join(lines).strip()
        if text_block:
            result.append((header, text_block))

    return result


def chunk_document(
    doc: ParsedDocument,
    *,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 50,
    min_chunk_tokens: int = 64,
) -> list[Chunk]:
    """Fragmenta un ``ParsedDocument`` en ``Chunk``s con solapamiento.

    Cada sección (delimitada por cabeceras) se fragmenta de forma
    independiente. El solapamiento solo se aplica entre sub-chunks de
    la misma sección. Cada chunk recibe como prefijo la ruta de
    cabeceras de su sección (ej. ``[Primer curso > Primer semestre]``).
    """
    overlap_chars = chunk_overlap_tokens * _CHARS_PER_TOKEN

    # Fase 1: Extraer secciones con sus headers
    sections = _extract_sections(doc.text)

    # Fase 2: Fragmentar cada sección de forma independiente
    all_fragments: list[tuple[str | None, str]] = []  # (header, text)

    for header, section_text in sections:
        frags = _recursive_split(section_text, chunk_size_tokens, _SEPARATORS)

        # Solapamiento INTRA-sección
        overlapped: list[str] = []
        for i, frag in enumerate(frags):
            if i > 0 and overlap_chars > 0:
                prev_tail = frags[i - 1][-overlap_chars:]
                frag = prev_tail + " " + frag
            overlapped.append(frag)

        # Fusionar sub-chunks cortos dentro de la misma sección
        section_merged: list[str] = []
        for frag in overlapped:
            if section_merged and _token_len(frag) < min_chunk_tokens:
                section_merged[-1] = section_merged[-1] + " " + frag
            else:
                section_merged.append(frag)

        for frag in section_merged:
            all_fragments.append((header, frag))

    # Fase 3: Construir objetos Chunk
    chunks: list[Chunk] = []
    for position, (header, text) in enumerate(all_fragments):
        # Prepender header contextual
        if header:
            contextualized_text = f"[{header}]\n{text}"
        else:
            contextualized_text = text

        chunks.append(
            Chunk(
                text=contextualized_text,
                source_url=doc.url,
                source_type=doc.source_type,
                title=doc.title,
                position=position,
                char_start=0,   # offset ya no es preciso con secciones
                char_end=len(text),
                fetched_at=doc.fetched_at,
                metadata=doc.metadata,
            )
        )

    n_contextualized = sum(1 for c in chunks if c.text.startswith("["))
    logger.info(
        "Chunked %s -> %d chunks (%d con header contextual) "
        "(cfg: %d tok, %d overlap, %d min)",
        doc.url, len(chunks), n_contextualized,
        chunk_size_tokens, chunk_overlap_tokens, min_chunk_tokens,
    )
    return chunks
