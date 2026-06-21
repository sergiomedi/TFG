"""Tests del subsistema de generación de GRAIA.

Se testean PromptBuilder y CitationValidator de forma unitaria (sin LLM).
El OllamaClient se testea a nivel de interfaz y con mock del SDK.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graia.recuperacion.retriever import RetrievedChunk
from graia.generacion.prompt_builder import (
    build_context_block,
    build_messages,
    get_source_map,
    _SYSTEM_PROMPT,
    _NO_CONTEXT_PROMPT,
)
from graia.generacion.citation_validator import (
    validate_citations,
    format_sources_block,
    CitationReport,
    _CITATION_PATTERN,
)
from graia.generacion.ollama_client import OllamaClient, GenerationResult
from graia.generacion.dedup import deduplicate_sentences


# ---- Helpers ----

def _make_retrieved(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"abc{i:04d}",
            text=f"El plazo de matrícula para el curso {2025 + i} es del 1 al 15 de julio.",
            source_url=f"https://etsiit.ugr.es/matricula_{i}",
            source_type="html",
            title=f"Matrícula {2025 + i}",
            position=i,
            similarity=0.9 - i * 0.05,
            rank=i,
        )
        for i in range(n)
    ]


# ---- Tests del PromptBuilder ----

class TestPromptBuilder:
    def test_build_context_block_with_chunks(self):
        chunks = _make_retrieved(3)
        block = build_context_block(chunks)
        assert "[1]" in block
        assert "[2]" in block
        assert "[3]" in block
        assert "Matrícula 2025" in block
        assert "https://etsiit.ugr.es/matricula_0" in block

    def test_build_context_block_empty(self):
        block = build_context_block([])
        assert block == ""

    def test_build_messages_with_context(self):
        chunks = _make_retrieved(2)
        system, user = build_messages("¿Cuándo es la matrícula?", chunks)
        assert system == _SYSTEM_PROMPT
        assert "<contexto>" in user and "</contexto>" in user
        assert "<fragmento>" in user
        assert "[1]" in user
        assert "<pregunta>" in user and "¿Cuándo es la matrícula?" in user

    def test_build_messages_without_context(self):
        system, user = build_messages("¿Cuándo es la matrícula?", [])
        assert system == _NO_CONTEXT_PROMPT
        assert user == "¿Cuándo es la matrícula?"

    def test_source_map_numbering(self):
        chunks = _make_retrieved(3)
        smap = get_source_map(chunks)
        assert 1 in smap
        assert 2 in smap
        assert 3 in smap
        assert 0 not in smap
        assert smap[1].chunk_id == chunks[0].chunk_id

    def test_system_prompt_contains_graia_rules(self):
        assert "GRAIA" in _SYSTEM_PROMPT
        assert "CONTEXTO" in _SYSTEM_PROMPT
        assert "[1]" in _SYSTEM_PROMPT or "marcadores" in _SYSTEM_PROMPT


# ---- Tests de deduplicación de contenido ----

class TestDeduplicateSentences:
    def test_real_case_collapses_to_single_source(self):
        """El caso reportado: mismo dato repetido con [1] y [2] → una sola frase."""
        text = ("La Secretaría de la ETSIIT atiende al público de 9:00 a 14:00 horas, "
                "de lunes a viernes [1]. El horario de atención al público es de 9 a 14 "
                "horas, de lunes a viernes [2].")
        out = deduplicate_sentences(text)
        assert out.count("[") == 1
        assert "[1]" in out and "[2]" not in out

    def test_genuine_redundancy_merges(self):
        text = ("El TFG se solicita hasta el 3 de octubre [1]. "
                "El plazo para solicitar el TFG es hasta el 3 de octubre [2].")
        assert deduplicate_sentences(text).count("[") == 1

    def test_distinct_subjects_not_merged(self):
        text = ("Cálculo se imparte los lunes de 9:00 a 11:00 en el aula 0.3 [1]. "
                "Álgebra se imparte los martes de 10:00 a 12:00 en el aula 0.6 [2].")
        assert deduplicate_sentences(text).count("[") == 2

    def test_complementary_info_preserved(self):
        text = ("Cálculo tiene teoría los lunes de 9 a 11 [1]. "
                "Cálculo tiene prácticas los martes de 11 a 12 [2].")
        assert deduplicate_sentences(text).count("[") == 2

    def test_non_duplicate_text_unchanged(self):
        text = "El plazo de matrícula es del 1 al 15 de julio [1]."
        assert deduplicate_sentences(text) == text

    def test_empty_input(self):
        assert deduplicate_sentences("") == ""
        assert deduplicate_sentences("   ") == "   "

    def test_parallel_list_items_not_collapsed(self):
        """Regresión: filas con MISMA estructura pero NÚMEROS distintos (horario
        de cada grupo) no deben fusionarse aunque compartan mucho vocabulario."""
        text = (
            "Grupo 1ºA: Martes 10:30-11:30, Teoría en aula 0.3; Prácticas subgrupo A1 (aula 3.3). "
            "Grupo 1ºB: Martes 11:30-12:30, Teoría en aula 0.6; Prácticas subgrupo B1 (aula 3.9). "
            "Grupo 1ºC: Jueves 11:30-12:30, Teoría en aula 0.1; Prácticas subgrupo C1 (aula 3.1). "
            "Grupo 1ºD: Martes 16:30-18:30, Teoría en aula 0.2; Prácticas subgrupo D1 (aula 2.8)."
        )
        out = deduplicate_sentences(text)
        assert out.count("Grupo") == 4  # ningún grupo eliminado

    def test_three_sentences_only_redundant_dropped(self):
        text = ("Atiende de 9 a 14 de lunes a viernes [1]. "
                "El horario es de 9 a 14 horas de lunes a viernes [2]. "
                "El TFG se entrega el 22 de junio [3].")
        out = deduplicate_sentences(text)
        assert out.count("[") == 2 and "[3]" in out


# ---- Tests del CitationValidator ----

class TestCitationValidator:
    def test_all_valid_citations(self):
        chunks = _make_retrieved(3)
        smap = get_source_map(chunks)
        text = "Según [1], el plazo es en julio. Además [2] confirma lo anterior."
        report = validate_citations(text, smap)
        assert report.valid_markers == [1, 2]
        assert report.invalid_markers == []
        assert report.all_valid
        assert report.total_citations == 2

    def test_invalid_citations_detected(self):
        chunks = _make_retrieved(2)
        smap = get_source_map(chunks)
        text = "Según [1] y [5], el plazo es en julio."
        report = validate_citations(text, smap)
        assert 1 in report.valid_markers
        assert 5 in report.invalid_markers
        assert not report.all_valid

    def test_invalid_citations_removed_from_text(self):
        chunks = _make_retrieved(2)
        smap = get_source_map(chunks)
        text = "Info [1] y alucinación [99] aquí."
        report = validate_citations(text, smap)
        assert "[99]" not in report.clean_text
        assert "[1]" in report.clean_text

    def test_no_citations(self):
        smap = get_source_map(_make_retrieved(2))
        text = "No hay citas en esta respuesta."
        report = validate_citations(text, smap)
        assert report.valid_markers == []
        assert report.invalid_markers == []
        assert report.all_valid
        assert report.total_citations == 0

    def test_duplicate_markers_counted_once(self):
        smap = get_source_map(_make_retrieved(3))
        text = "Dato [1]. Otro dato [1]. Y [2]."
        report = validate_citations(text, smap)
        assert report.valid_markers == [1, 2]  # [1] aparece solo una vez

    def test_sources_block_format(self):
        chunks = _make_retrieved(2)
        smap = get_source_map(chunks)
        text = "Info [1] y [2]."
        report = validate_citations(text, smap)
        block = format_sources_block(report)
        assert "[1]" in block
        assert "[2]" in block
        assert "etsiit.ugr.es" in block

    def test_sources_block_empty_when_no_sources(self):
        report = CitationReport()
        block = format_sources_block(report)
        assert block == ""

    def test_citation_pattern_matches_correctly(self):
        text = "Ref [1], [23], [456] pero no [abc] ni []."
        matches = _CITATION_PATTERN.findall(text)
        assert matches == ["1", "23", "456"]


# ---- Tests del OllamaClient (interfaz) ----

class TestOllamaClientInterface:
    def test_client_importable(self):
        assert OllamaClient is not None

    def test_client_has_required_methods(self):
        assert hasattr(OllamaClient, "generate")
        assert hasattr(OllamaClient, "generate_stream")

    def test_generation_result_fields(self):
        r = GenerationResult(text="Hola", model="test")
        assert r.text == "Hola"
        assert r.model == "test"
        assert r.total_tokens == 0


class TestOllamaClientHistory:
    """Verifica el cableado del historial conversacional (sin LLM real)."""

    class _FakeOllama:
        """Mock mínimo del cliente ollama: captura los mensajes recibidos."""
        def __init__(self):
            self.captured = None

        def chat(self, model, messages, options=None, stream=False):
            self.captured = messages
            if stream:
                return iter([{"message": {"content": "ok"}}])
            return {"message": {"content": "ok"}, "prompt_eval_count": 1, "eval_count": 1}

    def _client_with_fake(self):
        client = OllamaClient.__new__(OllamaClient)
        client.model = "test"
        client.temperature = 0.1
        client.top_p = 0.9
        client.max_tokens = 128
        client.stop = []
        client.client = self._FakeOllama()
        return client

    def test_generate_without_history(self):
        client = self._client_with_fake()
        client.generate("SYS", "PREGUNTA")
        msgs = client.client.captured
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == "SYS"
        assert msgs[-1]["content"] == "PREGUNTA"

    def test_generate_inserts_history_between_system_and_user(self):
        client = self._client_with_fake()
        history = [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Buenas"},
        ]
        client.generate("SYS", "PREGUNTA", history=history)
        msgs = client.client.captured
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
        assert msgs[0]["content"] == "SYS"
        assert msgs[1]["content"] == "Hola"
        assert msgs[2]["content"] == "Buenas"
        assert msgs[3]["content"] == "PREGUNTA"

    def test_generate_stream_inserts_history(self):
        client = self._client_with_fake()
        history = [{"role": "user", "content": "Previa"}]
        list(client.generate_stream("SYS", "ACTUAL", history=history))
        msgs = client.client.captured
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        assert msgs[1]["content"] == "Previa"
        assert msgs[2]["content"] == "ACTUAL"

    def test_none_history_equivalent_to_empty(self):
        client = self._client_with_fake()
        client.generate("SYS", "Q", history=None)
        assert len(client.client.captured) == 2


# ---- Memoria conversacional (módulo interfaz.history, sin Streamlit) ----

from graia.interfaz.history import (
    CLOSINGS,
    build_history,
    is_meta_query,
    is_no_info_answer,
    is_subjective_decline,
    sanitize_for_history,
)
from graia.generacion.postprocess import clean_answer, normalize_markdown_lists


class TestHedgeAndBullets:
    def test_strips_leading_hedge(self):
        out = clean_answer("Según la información proporcionada, los plazos son dos.")
        assert out.startswith("Los plazos son dos")
        assert "información proporcionada" not in out

    def test_keeps_useful_segun(self):
        text = "Según el Calendario TFG 2025-2026 [5], puede defender del 19 al 21."
        assert clean_answer(text) == text  # no se recorta: aporta la fuente

    def test_normalize_inline_bullets(self):
        text = "* Grupo 1ºA: Martes 10:30-11:30. * Grupo 1ºC: Jueves 11:30-12:30."
        out = normalize_markdown_lists(text)
        assert out.count("\n- ") == 2
        assert "10:30-11:30" in out and "*" not in out

    def test_normalize_preserves_time_ranges(self):
        text = "Teoría: Martes 11:30-12:30, Miércoles 12:30-13:30."
        assert normalize_markdown_lists(text) == text  # sin viñetas, intacto

    def test_normalize_dash_separator(self):
        text = "Plazos: Primero hasta octubre. - Segundo hasta febrero."
        out = normalize_markdown_lists(text)
        assert "\n- Segundo" in out


class TestIsMetaQuery:
    def test_identity_questions(self):
        assert is_meta_query("Quién eres?")
        assert is_meta_query("¿Qué eres?")
        assert is_meta_query("¿qué puedes hacer?")
        assert is_meta_query("¿quién te creó?")

    def test_non_meta(self):
        assert not is_meta_query("¿Cuál es el horario de la secretaría?")
        assert not is_meta_query("¿Cuáles son las asignaturas de primero?")


class TestSubjectiveDecline:
    def test_detects_decline(self):
        assert is_subjective_decline(
            "Lo siento, pero no puedo emitir opiniones personales ni recomendar una opción.")
        assert is_subjective_decline("No puedo recomendar una especialidad.")

    def test_factual_answer_not_decline(self):
        assert not is_subjective_decline("La Secretaría atiende de 9 a 14 horas.")


class TestNoInfoDetection:
    def test_canonical(self):
        assert is_no_info_answer("No dispongo de información suficiente sobre este tema.")

    def test_variants(self):
        assert is_no_info_answer("Lo siento, pero no hay información sobre la nota de corte.")
        assert is_no_info_answer("No tengo información sobre el menú de la cafetería.")
        assert is_no_info_answer("No tengo acceso a información específica sobre las plazas.")

    def test_factual_not_no_info(self):
        assert not is_no_info_answer("Las clases de Cálculo son los martes de 10:30 a 11:30.")


class TestIsNoInfoAnswer:
    def test_detects_no_info_message(self):
        assert is_no_info_answer(
            "No dispongo de información suficiente sobre este tema. "
            "Le recomiendo consultar con la Secretaría de la ETSIIT."
        )

    def test_case_insensitive(self):
        assert is_no_info_answer("no dispongo de informacion suficiente")

    def test_normal_answer_is_not_no_info(self):
        assert not is_no_info_answer("La secretaría abre de 9 a 14 horas.")


class TestSanitizeForHistory:
    def test_strips_sources_block(self):
        text = "Atiende de 9 a 14 [1].\n---\n**Fuentes:**\n- [1] Secretaría — url"
        out = sanitize_for_history(text)
        assert "Fuentes" not in out
        assert "url" not in out
        assert "Atiende de 9 a 14" in out

    def test_strips_citation_markers(self):
        out = sanitize_for_history("El plazo es julio [1] y agosto [2].")
        assert "[1]" not in out and "[2]" not in out
        assert "El plazo es julio" in out

    def test_strips_known_closings(self):
        text = f"La secretaría abre de 9 a 14. {CLOSINGS[0]}"
        out = sanitize_for_history(text)
        assert CLOSINGS[0] not in out
        assert "La secretaría abre" in out

    def test_normalizes_spaces_before_punctuation(self):
        out = sanitize_for_history("El plazo [1] , es julio.")
        assert " ," not in out


class TestBuildHistory:
    def _msgs(self):
        return [
            {"role": "assistant", "content": "Bienvenida"},          # se excluye
            {"role": "user", "content": "¿Horario secretaría?"},
            {"role": "assistant", "content": "De 9 a 14 [1].\n---\n**Fuentes:** url"},
            {"role": "user", "content": "¿Y el TFG?"},                # consulta actual: se excluye
        ]

    def test_excludes_welcome_and_current_query(self):
        hist = build_history(self._msgs(), max_turns=3)
        contents = [m["content"] for m in hist]
        assert "Bienvenida" not in contents
        assert "¿Y el TFG?" not in contents
        assert hist[0]["content"] == "¿Horario secretaría?"

    def test_assistant_messages_are_sanitized(self):
        hist = build_history(self._msgs(), max_turns=3)
        asst = [m for m in hist if m["role"] == "assistant"][0]
        assert "[1]" not in asst["content"]
        assert "Fuentes" not in asst["content"]

    def test_zero_turns_disables_memory(self):
        assert build_history(self._msgs(), max_turns=0) == []

    def test_only_welcome_returns_empty(self):
        assert build_history([{"role": "assistant", "content": "Bienvenida"}], 3) == []

    def test_limits_to_max_turns(self):
        msgs = [{"role": "assistant", "content": "Bienvenida"}]
        for i in range(5):
            msgs.append({"role": "user", "content": f"P{i}"})
            msgs.append({"role": "assistant", "content": f"R{i}"})
        msgs.append({"role": "user", "content": "ACTUAL"})
        hist = build_history(msgs, max_turns=2)
        assert len(hist) == 4  # 2 turnos = 4 mensajes
        assert hist[0]["content"] == "P3"  # solo los 2 últimos turnos previos


# ---- Tests de integración (sin LLM) ----

class TestGenerationIntegration:
    def test_generate_function_importable(self):
        from graia.generacion import generate
        assert callable(generate)

    def test_full_prompt_pipeline(self):
        """Verifica que el pipeline prompt → context → citation funciona end-to-end."""
        chunks = _make_retrieved(3)

        # Construir prompt
        system, user = build_messages("¿Cuándo me matriculo?", chunks)
        assert "CONTEXTO:" in user

        # Simular respuesta del LLM
        fake_response = (
            "El plazo de matrícula es del 1 al 15 de julio [1]. "
            "Para el siguiente curso, consulte [2]. "
            "No tengo información sobre [7]."
        )

        # Validar citas
        smap = get_source_map(chunks)
        report = validate_citations(fake_response, smap)
        assert 1 in report.valid_markers
        assert 2 in report.valid_markers
        assert 7 in report.invalid_markers
        assert "[7]" not in report.clean_text


# ---- Citas: deduplicación por URL y marcadores malformados ----

def _shared_url_chunks() -> list[RetrievedChunk]:
    """Dos fragmentos de la MISMA URL (secretaría) + uno de otra (TFG)."""
    def mk(cid, url, title, pos):
        return RetrievedChunk(chunk_id=cid, text="texto " + cid, source_url=url,
                              source_type="html", title=title, position=pos,
                              similarity=0.9 - pos * 0.1, rank=pos)
    return [
        mk("s1", "https://etsiit.ugr.es/sec", "Secretaría", 0),
        mk("s2", "https://etsiit.ugr.es/sec", "Secretaría", 1),
        mk("t1", "https://grados.ugr.es/tfg", "TFG", 0),
    ]


class TestRecoverSources:
    """Recuperación de citas cuando el modelo no emite marcadores [n]."""

    def _chunks(self):
        cal = "https://etsiit.ugr.es/cal.pdf"
        sec = "https://etsiit.ugr.es/sec"

        def mk(cid, url, text, title):
            return RetrievedChunk(chunk_id=cid, text=text, source_url=url,
                                  source_type="pdf", title=title, position=0,
                                  similarity=0.9, rank=0)
        return [
            mk("c1", cal, "Calendario TFG 2025-2026 | Defensa del TFG: del 19 al 21 "
               "de noviembre de 2025, y del 29 de junio al 3 de julio de 2026.",
               "Calendario académico y de TFG 2025-2026"),
            mk("c2", cal, "Calendario TFG 2025-2026 | Solicitud de evaluación del TFG: "
               "3 y 4 de noviembre de 2025, 15 y 16 de junio de 2026.",
               "Calendario académico y de TFG 2025-2026"),
            mk("s1", sec, "La Secretaría atiende de 9:00 a 14:00 de lunes a viernes.",
               "Secretaría"),
        ]

    def _source_map(self, chunks):
        # un marker por URL (como get_source_map)
        smap, seen = {}, {}
        for ch in chunks:
            if ch.source_url not in seen:
                seen[ch.source_url] = len(seen) + 1
            smap.setdefault(seen[ch.source_url], ch)
        return smap

    def test_recovers_relevant_source(self):
        from graia.generacion.citation_validator import recover_sources
        chunks = self._chunks()
        smap = self._source_map(chunks)
        answer = ("La defensa del TFG se realizará del 19 al 21 de noviembre de 2025 "
                  "y del 29 de junio al 3 de julio de 2026.")
        out = recover_sources(answer, chunks, smap)
        assert len(out) == 1
        assert out[0]["url"] == "https://etsiit.ugr.es/cal.pdf"

    def test_dedup_by_url(self):
        from graia.generacion.citation_validator import recover_sources
        chunks = self._chunks()
        smap = self._source_map(chunks)
        # respuesta que solapa con AMBAS líneas del calendario (misma URL)
        answer = ("Solicitud del TFG: 3 y 4 de noviembre de 2025. Defensa del TFG: "
                  "del 19 al 21 de noviembre de 2025.")
        out = recover_sources(answer, chunks, smap)
        urls = [s["url"] for s in out]
        assert urls.count("https://etsiit.ugr.es/cal.pdf") == 1  # una sola entrada

    def test_recovers_terse_answer(self):
        """Respuesta muy corta contenida en el fragmento (cobertura alta)."""
        from graia.generacion.citation_validator import recover_sources
        cal = "https://etsiit.ugr.es/cal.pdf"
        chunks = [RetrievedChunk(
            chunk_id="c", source_url=cal, source_type="pdf",
            title="Calendario académico y de TFG 2025-2026", position=0,
            similarity=0.9, rank=0,
            text="Calendario TFG 2025-2026 | Convocatorias anuales de evaluación del "
                 "TFG: noviembre de 2025, junio de 2026 y septiembre de 2026.")]
        smap = {1: chunks[0]}
        out = recover_sources("Noviembre, Junio y Septiembre.", chunks, smap)
        assert len(out) == 1 and out[0]["url"] == cal

    def test_no_attribution_when_no_overlap(self):
        from graia.generacion.citation_validator import recover_sources
        chunks = self._chunks()
        smap = self._source_map(chunks)
        assert recover_sources("El máster de ciberseguridad tiene 60 plazas.", chunks, smap) == []

    def test_single_shared_token_not_enough(self):
        """Un solo token compartido no basta para atribuir (evita coincidencias triviales)."""
        from graia.generacion.citation_validator import recover_sources
        cal = "https://etsiit.ugr.es/cal.pdf"
        chunks = [RetrievedChunk(
            chunk_id="c", source_url=cal, source_type="pdf", title="Calendario",
            position=0, similarity=0.9, rank=0,
            text="Calendario TFG 2025-2026 | Defensa del TFG: del 19 al 21 de noviembre de 2025.")]
        # 'noviembre' es el único token compartido → no se atribuye
        assert recover_sources("El examen es en noviembre.", chunks, {1: chunks[0]}) == []

    def test_empty_answer(self):
        from graia.generacion.citation_validator import recover_sources
        chunks = self._chunks()
        assert recover_sources("", chunks, self._source_map(chunks)) == []


class TestCitationDeduplication:
    def test_same_url_single_marker(self):
        smap = get_source_map(_shared_url_chunks())
        assert len(smap) == 2  # dos URLs únicas, no tres fragmentos
        assert len({c.source_url for c in smap.values()}) == 2

    def test_context_block_shares_marker_per_url(self):
        block = build_context_block(_shared_url_chunks())
        # la secretaría aparece dos veces pero con el MISMO marcador, nunca [3]
        assert "[1]" in block and "[2]" in block and "[3]" not in block

    def test_sources_block_no_duplicate_urls(self):
        smap = get_source_map(_shared_url_chunks())
        report = validate_citations("Atiende de 9 a 14 [1] y [2].", smap)
        block = format_sources_block(report)
        # una sola entrada de fuente para la secretaría (el enlace Markdown
        # repite la URL dos veces dentro de la MISMA línea: [url](url))
        assert block.count("(https://etsiit.ugr.es/sec)") == 1

    def test_malformed_markers_removed(self):
        smap = get_source_map(_make_retrieved(2))
        text = "Dato [1], ref [5.1-3] y [1-3] final."
        report = validate_citations(text, smap)
        assert "[5.1-3]" not in report.clean_text
        assert "[1-3]" not in report.clean_text
        assert "[1]" in report.clean_text
        assert report.valid_markers == [1]
