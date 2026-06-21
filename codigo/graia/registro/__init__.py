"""Subsistema de registro de GRAIA.

Proporciona logging JSONL estructurado para todos los módulos del sistema.
Uso típico::

    from graia.registro.logger import get_logger, setup_logging
    setup_logging(level="INFO")
    logger = get_logger("graia.mi_modulo")
    logger.info("evento", extra={"query_id": "abc123", "latency_ms": 42.5})
"""

from graia.registro.logger import generate_query_id, get_logger, setup_logging

__all__ = ["get_logger", "setup_logging", "generate_query_id"]
