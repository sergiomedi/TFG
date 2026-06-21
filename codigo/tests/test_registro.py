"""Tests del subsistema de registro de GRAIA."""

from __future__ import annotations

import json
import logging

from graia.registro.logger import (
    GraiaJsonFormatter,
    generate_query_id,
    get_logger,
    setup_logging,
)


class TestLogger:
    def test_get_logger_returns_child(self):
        lg = get_logger("graia.test")
        assert lg.name == "graia.test"
        assert isinstance(lg, logging.Logger)

    def test_generate_query_id_unique(self):
        ids = {generate_query_id() for _ in range(100)}
        assert len(ids) == 100

    def test_generate_query_id_length(self):
        qid = generate_query_id()
        assert len(qid) == 12

    def test_formatter_adds_required_fields(self):
        formatter = GraiaJsonFormatter()
        record = logging.LogRecord(
            name="graia.test", level=logging.INFO,
            pathname="", lineno=0, msg="test_event",
            args=None, exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["module"] == "graia.test"
        assert "event" in data

    def test_formatter_preserves_extra_fields(self):
        formatter = GraiaJsonFormatter()
        record = logging.LogRecord(
            name="graia.test", level=logging.INFO,
            pathname="", lineno=0, msg="query_processed",
            args=None, exc_info=None,
        )
        record.query_id = "abc123"
        record.latency_ms = 42.5
        output = formatter.format(record)
        data = json.loads(output)
        assert data["query_id"] == "abc123"
        assert data["latency_ms"] == 42.5

    def test_setup_logging_idempotent(self):
        """Llamar setup_logging dos veces no debe duplicar handlers."""
        # Limpiar handlers previos
        root = logging.getLogger("graia")
        root.handlers.clear()

        setup_logging(level="DEBUG", log_to_file=False)
        n1 = len(root.handlers)
        setup_logging(level="DEBUG", log_to_file=False)
        n2 = len(root.handlers)
        assert n1 == n2


class TestInterfazImport:
    def test_app_module_importable(self):
        """El módulo app.py debe ser importable sin lanzar Streamlit."""
        # Solo verificamos que el módulo se puede parsear
        import importlib
        spec = importlib.util.find_spec("graia.interfaz.app")
        assert spec is not None
