"""Modelos de datos del subsistema de ingesta.

Define las estructuras que fluyen entre los componentes del pipeline:
  RawDocument  → salida del Fetcher, entrada del Parser
  ParsedDocument → salida del Parser, entrada del Cleaner
  Chunk        → salida del Chunker, entrada del módulo de indexación
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class SourceType(str, Enum):
    """Tipo de fuente original del documento."""
    HTML = "html"
    PDF = "pdf"
    TXT = "txt"


class RawDocument(BaseModel):
    """Documento descargado sin procesar."""
    url: str
    content: bytes
    content_type: str           # MIME type (text/html, application/pdf)
    source_type: SourceType
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    http_status: int = 200


class ParsedDocument(BaseModel):
    """Documento con texto plano extraído y metadatos preservados."""
    url: str
    source_type: SourceType
    title: Optional[str] = None
    text: str                   # texto plano completo
    fetched_at: datetime
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """Fragmento de texto listo para ser embebido e indexado.

    El campo ``chunk_id`` se calcula como SHA-256 de (url + position) para
    garantizar idempotencia: re-ejecutar la ingesta sobre el mismo corpus
    produce los mismos identificadores, facilitando la detección de duplicados
    sin necesidad de comparar contenido carácter a carácter.
    """
    text: str
    source_url: str
    source_type: SourceType
    title: Optional[str] = None
    position: int               # índice ordinal dentro del documento de origen
    char_start: int             # offset en caracteres dentro del texto completo
    char_end: int
    fetched_at: datetime
    metadata: dict = Field(default_factory=dict)

    @computed_field  # type: ignore[misc]
    @property
    def chunk_id(self) -> str:
        """Identificador determinista basado en origen + posición."""
        raw = f"{self.source_url}:::{self.position}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
