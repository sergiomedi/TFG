"""Fetcher — descarga de documentos desde URLs web.

Implementa el componente Fetcher descrito en la Sección 5.6.1 del diseño:
  - Peticiones HTTP con ``requests`` + User-Agent configurable
  - Reintentos con backoff exponencial
  - Respeto a ``robots.txt`` (urllib.robotparser)
  - Clasificación automática del tipo de fuente (HTML / PDF)
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from graia.ingesta.models import RawDocument, SourceType

logger = logging.getLogger(__name__)

# Caché de parsers de robots.txt por dominio (evita re-descarga en la misma ejecución)
_robots_cache: dict[str, RobotFileParser] = {}


def _get_robots_parser(url: str, user_agent: str, timeout: int) -> RobotFileParser:
    """Obtiene (y cachea) el parser de robots.txt para el dominio de *url*."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    if base not in _robots_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base}/robots.txt")
        try:
            rp.read()
        except Exception:
            # Si no se puede leer, se asume permiso total (estándar de facto)
            logger.warning("No se pudo leer robots.txt de %s; se asume permiso.", base)
            rp.allow_all = True
        _robots_cache[base] = rp
    return _robots_cache[base]


def _classify_source(content_type: str, url: str) -> SourceType:
    """Determina el tipo de fuente a partir del Content-Type o la extensión."""
    ct = content_type.lower()
    if "pdf" in ct or url.lower().endswith(".pdf"):
        return SourceType.PDF
    if "text/plain" in ct or url.lower().endswith(".txt"):
        return SourceType.TXT
    return SourceType.HTML


def fetch(
    url: str,
    *,
    user_agent: str = "GRAIA-academic-crawler/0.1",
    timeout_s: int = 30,
    max_retries: int = 3,
    respect_robots: bool = True,
) -> Optional[RawDocument]:
    """Descarga un documento desde *url* respetando las reglas configuradas.

    Devuelve ``None`` si la URL está bloqueada por ``robots.txt`` o si se
    agotan los reintentos.  El backoff entre reintentos es exponencial
    (2^intento segundos) para no saturar servidores institucionales.
    """
    # --- Comprobación de robots.txt ---
    if respect_robots:
        rp = _get_robots_parser(url, user_agent, timeout_s)
        if not rp.can_fetch(user_agent, url):
            logger.info("Bloqueada por robots.txt: %s", url)
            return None

    headers = {"User-Agent": user_agent}

    # --- Descarga con reintentos y backoff exponencial ---
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            source_type = _classify_source(
                resp.headers.get("Content-Type", ""), url
            )
            logger.info(
                "Descargado %s (%s, %d bytes, intento %d)",
                url, source_type.value, len(resp.content), attempt,
            )
            return RawDocument(
                url=url,
                content=resp.content,
                content_type=resp.headers.get("Content-Type", ""),
                source_type=source_type,
                http_status=resp.status_code,
            )
        except requests.RequestException as exc:
            wait = 2 ** attempt
            logger.warning(
                "Error descargando %s (intento %d/%d): %s. Reintento en %ds.",
                url, attempt, max_retries, exc, wait,
            )
            if attempt < max_retries:
                time.sleep(wait)

    logger.error("Agotados los reintentos para %s", url)
    return None
