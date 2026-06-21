"""Cleaner — normalización y limpieza de texto extraído.

Implementa las transformaciones de limpieza descritas en la Sección 5.6.1
del diseño, aplicadas *después* del parsing y *antes* del chunking:

  1. Normalización Unicode NFC (canónica compuesta)
  2. Colapso de espacios en blanco redundantes
  3. Eliminación de líneas de boilerplate (patrones recurrentes en las webs
     de la ETSIIT/UGR: menús de cookies, «leer más», breadcrumbs)
  4. Descarte de documentos cuyo texto limpio sea demasiado corto para
     aportar un chunk útil
"""

from __future__ import annotations

import logging
import re
import unicodedata

from graia.ingesta.models import ParsedDocument

logger = logging.getLogger(__name__)

# Patrones de boilerplate habituales en webs institucionales UGR/ETSIIT
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Aceptar cookies?", re.IGNORECASE),
    re.compile(r"Pol[ií]tica de privacidad", re.IGNORECASE),
    re.compile(r"© Universidad de Granada", re.IGNORECASE),
    re.compile(r"Aviso legal", re.IGNORECASE),
    re.compile(r"Mapa del sitio", re.IGNORECASE),
    re.compile(r"Breadcrumb", re.IGNORECASE),
    re.compile(r"Ir al contenido principal", re.IGNORECASE),
    re.compile(r"^Inicio\s*>", re.MULTILINE),
]

# Umbral mínimo: documentos con menos caracteres útiles se descartan
_MIN_CLEAN_CHARS = 80


def _normalize_unicode(text: str) -> str:
    """Normalización NFC para unificar representaciones de acentos y eñes."""
    return unicodedata.normalize("NFC", text)


def _collapse_whitespace(text: str) -> str:
    """Reduce secuencias de espacios/tabuladores a un solo espacio y
    secuencias de más de dos saltos de línea a exactamente dos."""
    text = re.sub(r"[^\S\n]+", " ", text)       # espacios horizontales
    text = re.sub(r"\n{3,}", "\n\n", text)       # saltos verticales excesivos
    return text.strip()


def _remove_boilerplate(text: str) -> str:
    """Elimina líneas que coinciden con patrones conocidos de boilerplate."""
    lines = text.split("\n")
    clean_lines = [
        line for line in lines
        if not any(pat.search(line) for pat in _BOILERPLATE_PATTERNS)
    ]
    return "\n".join(clean_lines)


def _dedupe_consecutive_lines(text: str) -> str:
    """Colapsa líneas consecutivas idénticas (anti-duplicación).

    Red de seguridad redundante con el parser: garantiza que, sea cual sea el
    origen (HTML, PDF, OCR), el texto que llega al *chunker* no contiene
    repeticiones consecutivas como ``Asignatura\\nAsignatura`` que distorsionan
    los *embeddings* y confunden al generador. Las filas de tablas Markdown
    (``| ... |``) se preservan aunque parezcan similares, pues su contenido
    rara vez es idéntico carácter a carácter.
    """
    out: list[str] = []
    prev: str | None = None
    for line in text.split("\n"):
        key = line.strip()
        if key and key == prev and not key.startswith("|"):
            continue
        out.append(line)
        prev = key
    return "\n".join(out)


def clean(doc: ParsedDocument, *, min_chars: int = _MIN_CLEAN_CHARS) -> ParsedDocument | None:
    """Aplica la cadena completa de limpieza sobre *doc*.

    Devuelve ``None`` si el documento limpio no supera el umbral mínimo de
    caracteres (*min_chars*), señal de que no producirá chunks útiles.
    """
    text = doc.text
    text = _normalize_unicode(text)
    text = _remove_boilerplate(text)
    text = _dedupe_consecutive_lines(text)
    text = _collapse_whitespace(text)

    if len(text) < min_chars:
        logger.info(
            "Documento descartado por brevedad (%d < %d chars): %s",
            len(text), min_chars, doc.url,
        )
        return None

    logger.debug("Limpio: %s (%d → %d caracteres)", doc.url, len(doc.text), len(text))
    return doc.model_copy(update={"text": text})
