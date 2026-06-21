"""structured — indexación a nivel de registro para documentos estructurados.

Motivación (análisis de errores + best practices de RAG sobre datos tabulares):
  El documento de horarios contiene cientos de registros casi idénticos. Si se
  trocea en bloques grandes (chunking genérico), una consulta por una asignatura
  concreta ("¿cuándo es Inteligencia Artificial?") recupera un bloque donde ese
  registro queda enterrado entre otros 20, y los *embeddings* no distinguen el
  fragmento correcto. La práctica recomendada para datos estructurados es
  **un registro = un chunk**, con **metadatos** por campo (curso, cuatrimestre,
  especialidad, asignatura, día, hora, aula) que permiten recuperación de alta
  precisión y trazabilidad.

Este módulo:
  - ``parse_horario_record(line)``: extrae los campos estructurados de una línea
    de registro producida por ``HorarioParser`` (registro de clase o resumen).
  - ``chunk_structured_records(doc)``: convierte un ``ParsedDocument`` de horario
    en una lista de ``Chunk``, uno por registro, con sus metadatos. Cada registro
    es autocontenido, por lo que se marca ``self_contained=True`` y la fase de
    *Contextual Retrieval* lo deja intacto (no necesita contexto adicional).
"""

from __future__ import annotations

import re

from graia.ingesta.models import Chunk, ParsedDocument

# Curso: primer "Nº" de la línea ("4º Grado…", "Asignaturas de 3º…")
_CURSO_RE = re.compile(r"(\d)\s*º")
# Curso en palabras (formato verificado: "Primer curso", "Cuarto curso")
_CURSO_WORD_RE = re.compile(r"\b(primer|segundo|tercer|cuarto)\s+curso", re.IGNORECASE)
_CURSO_WORD2NUM = {"primer": 1, "segundo": 2, "tercer": 3, "cuarto": 4}
# Grupo: letra A–H tras "Nº" sin ser el inicio de "Grado" (lookahead no-minúscula)
_GRUPO_RE = re.compile(r"\b\d\s*º\s*([A-H])(?![a-zñáéíóú])")
# Especialidad: texto entre paréntesis tras "Informática ("
_ESP_RE = re.compile(r"Inform[aá]tica\s*\(([^)]+)\)")
# Cuatrimestre: 1er/primer/1º … 2º/segundo
_CUATRI_RE = re.compile(r"(1er|primer|1\s*º|2\s*º|2[o]|segundo)\.?\s*cuatrimestre", re.IGNORECASE)
_DIA_RE = re.compile(r"\b(Lunes|Martes|Mi[ée]rcoles|Jueves|Viernes)\b")
_HORA_RE = re.compile(r"(\d{1,2}:\d{2}-\d{1,2}:\d{2})")
_AULA_RE = re.compile(r"aula\s+(.+?)\.?\s*$", re.IGNORECASE)
# Siglas entre paréntesis: "(DI)", "(ALEM)" (no captura grupos "(A1)")
_SIGLAS_RE = re.compile(r"\(([A-ZÁÉÍÓÚÑ]{2,6})\)")
# Registro-resumen en formato verificado: "… | Asignaturas: ALEM, CA, …"
_SUMMARY_VERIFICADO_RE = re.compile(r"\|\s*Asignaturas\s*:", re.IGNORECASE)


def parse_horario_record(line: str) -> dict:
    """Extrae los campos estructurados de una línea de registro de horario."""
    line = line.strip()
    meta: dict = {"tipo": "horario", "category": "horarios", "self_contained": True}
    # Registro-resumen: enumera las asignaturas de un curso/grupo. Se reconocen
    # los dos formatos vigentes: (a) el del HorarioParser ("Asignaturas de 3º…")
    # y (b) el del fichero verificado ("… | Asignaturas: ALEM, CA, …"). Sin (b) el
    # flag nunca se activaba con el corpus verificado actual y el boost a resúmenes
    # quedaba inerte.
    meta["is_summary"] = (
        line.startswith("Asignaturas de")
        or bool(_SUMMARY_VERIFICADO_RE.search(line))
    )

    m = _CURSO_RE.search(line)
    if m:
        meta["curso"] = int(m.group(1))
    else:
        cw = _CURSO_WORD_RE.search(line)
        if cw:
            meta["curso"] = _CURSO_WORD2NUM[cw.group(1).lower()]

    g = _GRUPO_RE.search(line)
    if g:
        meta["grupo"] = g.group(1)

    e = _ESP_RE.search(line)
    if e:
        meta["especialidad"] = e.group(1).strip()

    c = _CUATRI_RE.search(line)
    if c:
        tok = c.group(1).lower().replace(" ", "")
        meta["cuatrimestre"] = 1 if ("1" in tok or "primer" in tok) else 2

    if not meta["is_summary"]:
        d = _DIA_RE.search(line)
        if d:
            meta["dia"] = d.group(1)
        h = _HORA_RE.search(line)
        if h:
            meta["hora"] = h.group(1)
        a = _AULA_RE.search(line)
        if a:
            meta["aula"] = a.group(1).strip()

    siglas = _SIGLAS_RE.findall(line)
    if siglas:
        meta["siglas"] = siglas

    return meta


def parse_calendario_record(line: str) -> dict:
    """Extrae campos de una línea de calendario (verificado, 1 evento por línea)."""
    meta: dict = {"tipo": "calendario", "category": "calendario", "self_contained": True}
    # Etiqueta del evento: texto antes del primer ':' tras el último '|'
    body = line.split("|")[-1].strip()
    if ":" in body:
        meta["evento"] = body.split(":", 1)[0].strip()
    return meta


# Especialidad en registros de plan de estudios: "especialidad <Nombre> (<SIG>)"
_ESP_PLAN_RE = re.compile(r"especialidad\s+[^()|]+\(([A-Z]{2,4})\)", re.IGNORECASE)


def parse_plan_estudios_record(line: str) -> dict:
    """Extrae campos de una línea agregada del plan de estudios (verificado).

    Cada línea es un registro-resumen del catálogo (asignaturas por curso,
    por curso×especialidad, o la lista de especialidades). Se marca
    ``is_summary=True`` para que las consultas de listado lo prioricen
    (inyección de resúmenes, Sección 5.8).
    """
    meta: dict = {"tipo": "plan_estudios", "category": "plan_estudios",
                  "self_contained": True, "is_summary": True}
    cw = _CURSO_WORD_RE.search(line)
    if cw:
        meta["curso"] = _CURSO_WORD2NUM[cw.group(1).lower()]
    e = _ESP_PLAN_RE.search(line)
    if e:
        meta["especialidad"] = e.group(1).upper()
    siglas = _SIGLAS_RE.findall(line)
    if siglas:
        meta["siglas"] = siglas
    return meta


def _parse_record(line: str, tipo: str) -> dict:
    if tipo == "calendario":
        return parse_calendario_record(line)
    if tipo == "plan_estudios":
        return parse_plan_estudios_record(line)
    return parse_horario_record(line)


def chunk_structured_records(
    doc: ParsedDocument,
    *,
    min_chars: int = 15,
) -> list[Chunk]:
    """Convierte un ``ParsedDocument`` estructurado en chunks (uno por registro).

    Cada línea no vacía del texto se convierte en un ``Chunk`` autocontenido con
    sus metadatos estructurados, lo que habilita recuperación de alta precisión
    (por asignatura/curso/cuatrimestre o por evento) y trazabilidad campo a campo.
    Soporta horarios y calendarios (``metadata['tipo']``).
    """
    chunks: list[Chunk] = []
    position = 0
    base_meta = dict(doc.metadata or {})
    tipo = base_meta.get("tipo", "horario")
    for raw_line in doc.text.split("\n"):
        line = raw_line.strip()
        # Ignorar líneas de comentario (plantilla verificada) y líneas demasiado cortas
        if line.startswith("#") or len(line) < min_chars:
            continue
        meta = dict(base_meta)
        meta.update(_parse_record(line, tipo))
        chunks.append(
            Chunk(
                text=line,
                source_url=doc.url,
                source_type=doc.source_type,
                title=doc.title,
                position=position,
                char_start=0,
                char_end=len(line),
                fetched_at=doc.fetched_at,
                metadata=meta,
            )
        )
        position += 1
    return chunks


# Fusión de franjas horarias contiguas del mismo día (y misma aula en prácticas)
# en un único intervalo: "Lunes 12:30-13:30, Lunes 13:30-14:30" → "Lunes
# 12:30-14:30". Con franjas de 1 hora separadas, el LLM tiende a quedarse con la
# primera y omitir las contiguas; fusionarlas elimina ese fallo y mejora la
# legibilidad. Transformación SIN pérdida (conserva cobertura horaria y aulas).
_MERGE_DAY = r"(Lunes|Martes|Miércoles|Jueves|Viernes)"
_MERGE_PRAC_RE = re.compile(
    _MERGE_DAY + r" (\d{1,2}:\d{2})-(\d{1,2}:\d{2}) \((aula [^)]+)\), \1 \3-(\d{1,2}:\d{2}) \(\4\)"
)
_MERGE_TEO_RE = re.compile(
    _MERGE_DAY + r" (\d{1,2}:\d{2})-(\d{1,2}:\d{2}), \1 \3-(\d{1,2}:\d{2})"
)


def merge_contiguous_slots(line: str) -> str:
    """Fusiona franjas horarias contiguas del mismo día en un único intervalo.

    Aplica primero la fusión de prácticas (exige misma aula) y luego la de
    teoría, iterando hasta punto fijo para encadenar 3+ franjas consecutivas.
    """
    prev = None
    while prev != line:
        prev = line
        line = _MERGE_PRAC_RE.sub(r"\1 \2-\5 (\4)", line)
        line = _MERGE_TEO_RE.sub(r"\1 \2-\4", line)
    return line


def is_structured(doc: ParsedDocument) -> bool:
    """True si el documento debe indexarse a nivel de registro (horarios/calendario)."""
    tipo = (doc.metadata or {}).get("tipo")
    if tipo in ("horario", "calendario", "plan_estudios"):
        return True
    return "horario" in (doc.url or "").lower()
