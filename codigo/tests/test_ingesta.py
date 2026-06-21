"""Tests de la ingesta modernizada (parser sin duplicación, tablas→Markdown,
chunker con tamaño garantizado, dedup del cleaner y Contextual Retrieval).

Cubren las garantías introducidas en la reestructuración del subsistema de
ingesta y motivadas por el análisis de errores del corpus v1:

  - El parser HTML no duplica texto y serializa tablas a Markdown.
  - El chunker no produce fragmentos por encima del tamaño objetivo.
  - El cleaner elimina líneas consecutivas duplicadas.
  - El contextualizador antepone contexto, cachea y degrada con elegancia.

Las pruebas que dependen de binarios pesados (PyMuPDF, Tesseract) se omiten
automáticamente si no están instalados, para no romper la CI ligera.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graia.ingesta.models import Chunk, ParsedDocument, RawDocument, SourceType
from graia.ingesta.parser import HtmlParser
from graia.ingesta.cleaner import _dedupe_consecutive_lines, clean
from graia.ingesta.chunker import chunk_document, _token_len, _recursive_split


def _raw_html(body: str, url: str = "https://etsiit.ugr.es/p") -> RawDocument:
    html = f"<html><head><title>T</title></head><body>{body}</body></html>"
    return RawDocument(url=url, content=html.encode("utf-8"),
                       content_type="text/html", source_type=SourceType.HTML)


# ---------- Parser HTML: sin duplicación ----------

class TestHtmlNoDuplication:
    def test_consecutive_duplicate_lines_collapsed(self):
        body = ("<main><div><a title='Web'>Web del Grado</a>"
                "<span>Web del Grado</span></div></main>")
        doc = HtmlParser().parse(_raw_html(body))
        assert doc.text.count("Web del Grado") == 1

    def test_table_to_markdown(self):
        body = ("<main><table>"
                "<tr><th>Asignatura</th><th>Tipo</th><th>Cr</th></tr>"
                "<tr><td>Calculo</td><td>Troncal</td><td>6</td></tr>"
                "</table></main>")
        doc = HtmlParser().parse(_raw_html(body))
        assert "| Calculo | Troncal | 6 |" in doc.text
        assert "| --- | --- | --- |" in doc.text
        # La cabecera no aparece duplicada
        assert doc.text.count("Asignatura") == 1

    def test_header_markers_present(self):
        body = "<main><h1>Grado</h1><h2>Primer curso</h2><p>texto</p></main>"
        doc = HtmlParser().parse(_raw_html(body))
        assert "§H1§ Grado" in doc.text
        assert "§H2§ Primer curso" in doc.text


# ---------- Chunker: tamaño garantizado ----------

class TestChunkerSizeGuarantee:
    @pytest.mark.parametrize("text", [
        "Frase corta. " * 400,            # muchas frases, sin \n\n
        "x" * 9000,                        # sin ningún separador
        ("parrafo " * 300 + "\n\n") * 3,  # secciones con \n\n
    ])
    def test_no_fragment_exceeds_max(self, text):
        frags = _recursive_split(text, 512, ["\n\n", "\n", ". ", " "])
        assert all(_token_len(f) <= 512 for f in frags)

    def test_chunk_document_respects_size(self):
        doc = ParsedDocument(
            url="https://etsiit.ugr.es/x", source_type=SourceType.HTML,
            title="T", text="Contenido académico relevante. " * 800,
            fetched_at=datetime.now(timezone.utc),
        )
        chunks = chunk_document(doc, chunk_size_tokens=512, chunk_overlap_tokens=50)
        # Con solapamiento, se permite un margen: ningún chunk gigante (>2x)
        assert all(_token_len(c.text) <= 512 * 2 for c in chunks)
        assert len(chunks) > 1


# ---------- Cleaner: dedup ----------

class TestCleanerDedup:
    def test_dedupe_consecutive(self):
        text = "Asignatura\nAsignatura\nTipo\nTipo\nCalculo"
        out = _dedupe_consecutive_lines(text)
        assert out == "Asignatura\nTipo\nCalculo"

    def test_dedupe_preserves_markdown_rows(self):
        text = "| a | b |\n| a | b |"
        out = _dedupe_consecutive_lines(text)
        # Las filas de tabla NO se colapsan (su contenido suele diferir)
        assert out.count("| a | b |") == 2


# ---------- Contextual Retrieval ----------

class _Result:
    """Imita a generacion.ollama_client.GenerationResult (solo el campo .text)."""
    def __init__(self, text):
        self.text = text


class _StubLLM:
    """Cliente LLM simulado compatible con la interfaz de OllamaClient."""
    def __init__(self, text="Contexto del Plan de Estudios."):
        self._text = text
        self.calls = 0

    def generate(self, system_prompt, user_message):
        self.calls += 1
        return _Result(self._text)


class _FailLLM:
    def generate(self, system_prompt, user_message):
        raise RuntimeError("ollama no disponible")


def _chunk(text, url="https://etsiit.ugr.es/plan"):
    return Chunk(text=text, source_url=url, source_type=SourceType.HTML,
                 position=0, char_start=0, char_end=len(text),
                 fetched_at=datetime.now(timezone.utc))


class TestContextualizer:
    def test_prepends_context_and_traces(self, tmp_path):
        from graia.ingesta.contextualizer import contextualize_chunks, ContextCache
        ch = _chunk("| Calculo | Troncal | 6 |")
        cache = ContextCache(tmp_path / "ctx.json")
        out = contextualize_chunks([ch], {ch.source_url: "Plan de Estudios..."},
                                   _StubLLM(), cache=cache)
        assert out[0].text.startswith("Contexto del Plan de Estudios.")
        assert out[0].metadata["context"] == "Contexto del Plan de Estudios."

    def test_cache_avoids_second_call(self, tmp_path):
        from graia.ingesta.contextualizer import contextualize_chunks, ContextCache
        ch = _chunk("texto")
        cache = ContextCache(tmp_path / "ctx.json")
        llm = _StubLLM()
        contextualize_chunks([ch], {}, llm, cache=cache)
        contextualize_chunks([ch], {}, llm, cache=cache)
        assert llm.calls == 1  # la segunda pasada usa la caché

    def test_graceful_degradation(self):
        from graia.ingesta.contextualizer import contextualize_chunks
        ch = _chunk("texto")
        out = contextualize_chunks([ch], {}, _FailLLM())
        assert out[0].text == "texto"  # chunk intacto si el LLM falla


# ---------- PDF (omitido si faltan binarios) ----------

class TestPdfParser:
    def test_pdf_table_to_markdown_and_ocr(self, tmp_path):
        fitz = pytest.importorskip("fitz", reason="PyMuPDF no instalado")
        from graia.ingesta.parser import PdfParser

        # PDF con tabla dibujada (líneas + texto)
        doc = fitz.open()
        pg = doc.new_page(width=400, height=300)
        cols = [50, 180, 300, 360]
        rows = [60, 90, 120]
        data = [["Asignatura", "Tipo", "Cr"], ["Calculo", "Troncal", "6"]]
        for ri, row in enumerate(data):
            for ci, val in enumerate(row):
                pg.insert_text((cols[ci] + 3, rows[ri] + 18), val, fontsize=10)
        for x in cols:
            pg.draw_line((x, rows[0]), (x, rows[-1] + 30))
        for y in rows + [rows[-1] + 30]:
            pg.draw_line((cols[0], y), (cols[-1], y))
        pdf_bytes = doc.tobytes()
        doc.close()

        raw = RawDocument(url="https://etsiit.ugr.es/t.pdf", content=pdf_bytes,
                          content_type="application/pdf", source_type=SourceType.PDF)
        parsed = PdfParser(enable_ocr=False).parse(raw)
        assert "Calculo" in parsed.text and "Troncal" in parsed.text
        # Debe contener al menos una fila Markdown
        assert "|" in parsed.text


class TestHorarioParser:
    def test_grid_reconstruction(self, tmp_path):
        fitz = pytest.importorskip("fitz", reason="PyMuPDF no instalado")
        from graia.ingesta.horario_parser import HorarioParser

        doc = fitz.open()
        pg = doc.new_page(width=600, height=260)
        colx = {"Lunes": 120, "Martes": 200, "Miércoles": 280,
                "Jueves": 360, "Viernes": 440}

        def put(x, y, s):
            pg.insert_text((x, y), s, fontsize=9)

        put(40, 40, "4º Grado en Ingeniería Informática (Ingeniería del Software)")
        put(40, 58, "2º cuatrimestre")
        for d, x in colx.items():
            put(x, 80, d)
        put(40, 110, "11:30-12:30"); put(120, 110, "DI"); put(200, 110, "PPR")
        put(120, 124, "1.4"); put(200, 124, "2.6")
        # Leyenda DESPUÉS de las filas (caso real): exige dos pasadas
        put(480, 200, "DI."); put(505, 200, "Derecho"); put(545, 200, "Informático.")
        pdf_bytes = doc.tobytes()
        doc.close()

        raw = RawDocument(url="https://etsiit.ugr.es/Horarios%20GII.pdf",
                          content=pdf_bytes, content_type="application/pdf",
                          source_type=SourceType.PDF)
        parsed = HorarioParser().parse(raw)

        # DI se expande con la leyenda y se asocia a su día/hora/aula
        assert "Lunes 11:30-12:30: Derecho Informático (DI), aula 1.4" in parsed.text
        # Sigla sin leyenda se mantiene; celda vacía (Viernes) no genera registro
        assert "Martes 11:30-12:30: PPR, aula 2.6" in parsed.text
        assert "Viernes" not in parsed.text
        # Registro-resumen de agregación por curso/especialidad/cuatrimestre
        assert "Asignaturas de" in parsed.text
        assert "Derecho Informático (DI)" in parsed.text

    def test_routing_via_url(self):
        """parse() debe enrutar URLs con 'horario' al HorarioParser."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF no instalado")
        from graia.ingesta.parser import parse as parse_factory

        doc = fitz.open()
        pg = doc.new_page(width=600, height=200)
        pg.insert_text((40, 40), "1º A Grado en Ingeniería Informática", fontsize=9)
        pg.insert_text((40, 55), "1er. cuatrimestre", fontsize=9)
        for i, d in enumerate(["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]):
            pg.insert_text((120 + i * 80, 80), d, fontsize=9)
        pg.insert_text((40, 110), "8:30-9:30", fontsize=9)
        pg.insert_text((120, 110), "CA", fontsize=9)
        pg.insert_text((120, 124), "0.3", fontsize=9)
        pdf_bytes = doc.tobytes()
        doc.close()

        raw = RawDocument(url="https://etsiit.ugr.es/Horarios%20GII%20(25-26).pdf",
                          content=pdf_bytes, content_type="application/pdf",
                          source_type=SourceType.PDF)
        parsed = parse_factory(raw)
        assert parsed.metadata.get("tipo") == "horario"
        assert "Lunes 8:30-9:30" in parsed.text


class TestStructuredIndexing:
    def test_parse_horario_record_fields(self):
        from graia.ingesta.structured import parse_horario_record
        line = ("Segundo cuatrimestre | Cuarto curso | Asignaturas globales "
                "(todas las ramas) | Derecho Informático (DI) — Teoría (aula 1.4): "
                "Lunes 12:30-14:30, Martes 11:30-13:30.")
        m = parse_horario_record(line)
        assert m["curso"] == 4 and m["cuatrimestre"] == 2
        assert "DI" in m["siglas"] and m["self_contained"] is True

    def test_is_summary_curated_format(self):
        """El registro-resumen del fichero verificado ('| Asignaturas:') se marca."""
        from graia.ingesta.structured import parse_horario_record
        line = ("Primer cuatrimestre | Primer curso | Grupo 1ºA | Asignaturas: "
                "Álgebra Lineal y Estructuras Matemáticas (ALEM), Cálculo (CA).")
        m = parse_horario_record(line)
        assert m["is_summary"] is True
        assert m["curso"] == 1

    def test_is_summary_parser_format(self):
        from graia.ingesta.structured import parse_horario_record
        assert parse_horario_record("Asignaturas de 3º (rama Computación): IA, MC.")["is_summary"] is True

    def test_class_record_is_not_summary(self):
        from graia.ingesta.structured import parse_horario_record
        line = ("Primer cuatrimestre | Primer curso | Grupo 1ºA | Cálculo (CA) — "
                "Teoría (aula 0.3): Martes 10:30-11:30.")
        assert parse_horario_record(line)["is_summary"] is False

    def test_parse_plan_estudios_specialty_record(self):
        from graia.ingesta.structured import parse_plan_estudios_record
        line = ("Plan de estudios GII 2025-2026 | Tercer curso, especialidad "
                "Computación y Sistemas Inteligentes (CSI) | Asignaturas de la "
                "especialidad Computación y Sistemas Inteligentes (CSI) en tercer "
                "curso (segundo cuatrimestre): Aprendizaje Automático (AA), "
                "Metaheurísticas (MH), Modelos de Computación Avanzada (MCA).")
        m = parse_plan_estudios_record(line)
        assert m["tipo"] == "plan_estudios" and m["category"] == "plan_estudios"
        assert m["is_summary"] is True
        assert m["curso"] == 3 and m["especialidad"] == "CSI"
        assert "AA" in m["siglas"] and "MCA" in m["siglas"]

    def test_parse_plan_estudios_curso_record(self):
        from graia.ingesta.structured import parse_plan_estudios_record
        line = ("Plan de estudios GII 2025-2026 | Primer curso | Asignaturas de "
                "primer curso (comunes a todos los grupos). Primer cuatrimestre: "
                "Cálculo (CA). Segundo cuatrimestre: Estadística (ES).")
        m = parse_plan_estudios_record(line)
        assert m["curso"] == 1 and "especialidad" not in m
        assert m["is_summary"] is True

    def test_merge_contiguous_teoria(self):
        from graia.ingesta.structured import merge_contiguous_slots
        line = ("Derecho Informático (DI) — Teoría (aula 1.4): Lunes 12:30-13:30, "
                "Lunes 13:30-14:30, Martes 11:30-12:30, Martes 12:30-13:30.")
        out = merge_contiguous_slots(line)
        assert "Lunes 12:30-14:30" in out and "Martes 11:30-13:30" in out
        assert "13:30-14:30," not in out  # ya no hay franja suelta

    def test_merge_contiguous_practicas_same_room(self):
        from graia.ingesta.structured import merge_contiguous_slots
        line = ("subgrupo A2 Martes 11:30-12:30 (aula 3.6), Martes 12:30-13:30 (aula 3.6)")
        assert merge_contiguous_slots(line) == "subgrupo A2 Martes 11:30-13:30 (aula 3.6)"

    def test_no_merge_different_room(self):
        from graia.ingesta.structured import merge_contiguous_slots
        line = "Martes 11:30-12:30 (aula 3.6), Martes 12:30-13:30 (aula 3.7)"
        assert merge_contiguous_slots(line) == line  # aulas distintas: no fusiona

    def test_no_merge_non_contiguous(self):
        from graia.ingesta.structured import merge_contiguous_slots
        line = "Teoría (aula 0.3): Lunes 10:30-11:30, Martes 9:30-10:30"
        assert merge_contiguous_slots(line) == line

    def test_merge_chain_three(self):
        from graia.ingesta.structured import merge_contiguous_slots
        line = "Lunes 9:30-10:30, Lunes 10:30-11:30, Lunes 11:30-12:30"
        assert merge_contiguous_slots(line) == "Lunes 9:30-12:30"

    def test_is_structured_plan_estudios(self):
        from datetime import datetime, timezone
        from graia.ingesta.structured import is_structured
        doc = ParsedDocument(
            url="https://grados.ugr.es/informatica/docencia/plan-estudios",
            source_type=SourceType.HTML, title="Plan de estudios",
            fetched_at=datetime.now(timezone.utc),
            metadata={"tipo": "plan_estudios"}, text="...")
        assert is_structured(doc) is True

    def test_one_record_per_chunk(self):
        from datetime import datetime, timezone
        from graia.ingesta.structured import chunk_structured_records
        doc = ParsedDocument(
            url="https://etsiit.ugr.es/Horarios.pdf", source_type=SourceType.PDF,
            title="Horarios", fetched_at=datetime.now(timezone.utc),
            metadata={"tipo": "horario"},
            text=("Primer cuatrimestre | Primer curso | Grupo 1ºA | Cálculo (CA) — Teoría (aula 0.3): Martes 10:30-11:30.\n"
                  "Primer cuatrimestre | Primer curso | Grupo 1ºA | Fundamentos de Programación (FP) — Teoría (aula 0.1): Lunes 9:30-10:30."),
        )
        chunks = chunk_structured_records(doc)
        assert len(chunks) == 2                      # una línea = un chunk
        assert chunks[0].metadata["curso"] == 1
        assert chunks[0].metadata["self_contained"] is True
        assert all(c.chunk_id for c in chunks)


class TestPostprocess:
    def test_strips_preamble_and_trailing(self):
        from graia.generacion.postprocess import clean_answer
        txt = ("La pregunta del usuario es sobre el horario.\n"
               "La Secretaría atiende de 9:00 a 14:00 horas [1].\n"
               "Espero que esto le ayude.")
        out = clean_answer(txt)
        assert out == "La Secretaría atiende de 9:00 a 14:00 horas [1]."

    def test_keeps_clean_answer(self):
        from graia.generacion.postprocess import clean_answer
        txt = "El grado tiene 240 créditos [1]."
        assert clean_answer(txt) == txt
