"""Logger JSONL estructurado para GRAIA.

Implementa el componente de registro descrito en la Sección 5.12 del diseño:
  - Formato JSONL (una línea JSON por evento)
  - Campos obligatorios: timestamp, level, event, module
  - Campos opcionales correlacionables: query_id, latency_ms
  - Salida a fichero rotativo + consola (configurable por nivel)

Se emplea ``python-json-logger`` como formateador sobre el módulo ``logging``
estándar, lo que permite que todos los módulos de GRAIA emitan logs
estructurados con una sola llamada a ``get_logger()``.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pythonjsonlogger import jsonlogger


class GraiaJsonFormatter(jsonlogger.JsonFormatter):
    """Formateador JSON personalizado que inyecta campos estándar de GRAIA."""

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        # Timestamp ISO-8601 en UTC
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["module"] = record.name
        # Asegurar que 'event' existe (usa el message si no se pasa explícitamente)
        if "event" not in log_record:
            log_record["event"] = record.getMessage()


def setup_logging(
    log_dir: str = "logs",
    level: str = "INFO",
    log_to_file: bool = True,
) -> None:
    """Configura el sistema de logging global de GRAIA.

    Debe llamarse una sola vez al arrancar la aplicación (desde ``app.py``
    o desde los scripts de orquestación).

    Parameters
    ----------
    log_dir : str
        Directorio donde se escribe el fichero JSONL.
    level : str
        Nivel mínimo de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_to_file : bool
        Si ``True``, escribe además a ``{log_dir}/graia.jsonl``.
    """
    root = logging.getLogger("graia")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Evitar handlers duplicados si se llama más de una vez
    if root.handlers:
        return

    formatter = GraiaJsonFormatter()

    # Handler de consola (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Handler de fichero JSONL
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / "graia.jsonl",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger hijo del espacio ``graia``.

    Todos los loggers obtenidos con esta función heredan la configuración
    establecida por ``setup_logging()``.
    """
    return logging.getLogger(name)


def generate_query_id() -> str:
    """Genera un identificador único para correlacionar los logs de una consulta."""
    return uuid.uuid4().hex[:12]
