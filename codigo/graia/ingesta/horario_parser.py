"""HorarioParser — reconstrucción geométrica de rejillas de horarios en PDF.

Motivación (análisis de errores del corpus):
  El PDF de horarios del GII (``Horarios GII (25-26).pdf``) es una **rejilla
  visual**: columnas = días (Lunes–Viernes), filas = franjas horarias, y cada
  celda contiene la(s) asignatura(s) y, en la línea inferior, el **aula**.
  La extracción lineal (``get_text('text')``) destruye la alineación
  columna→día y la asociación asignatura↔aula↔hora, produciendo un muro de
  ~160 000 caracteres del que el sistema RAG no puede recuperar una respuesta
  a «¿a qué hora es DI y en qué aula?».

Solución:
  Reconstrucción por **coordenadas** con ``page.get_text('words')`` (cada
  palabra con su caja x,y). Se detectan las rejillas por su título, se fijan
  las columnas a partir de los encabezados de día y las filas a partir de las
  franjas ``HH:MM-HH:MM``, y cada token (asignatura / aula) se asigna a su
  celda por posición. La leyenda del propio PDF (``DI. Derecho Informático.``)
  se usa para expandir las siglas.

Salida:
  Un texto compuesto por **registros autocontenidos**, uno por (rejilla, día,
  franja), p. ej.:

      4º Grado en Ingeniería Informática (Ingeniería del Software),
      2º cuatrimestre — Lunes 11:30-12:30: Derecho Informático (DI), aula 1.4.

  Estos registros, cortos y con todo el contexto, son ideales para los
  *embeddings*, BM25 y la posterior contextualización.

Limitaciones (honestidad de tribunal):
  - Es una heurística dependiente del *layout*; si la maqueta cambia
    radicalmente, ``parse()`` cae al ``PdfParser`` genérico (sin pérdida de
    cobertura).
  - Celdas con varios grupos (``A1/A2/A3``) y aulas dobles (``3.4/3.11``) se
    serializan concatenadas en el registro de esa celda.
"""

from __future__ import annotations

import logging
import re

from graia.ingesta.models import ParsedDocument, RawDocument

logger = logging.getLogger(__name__)

_DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
_DAYS_NORM = {d.lower(): d for d in _DAYS}

_TITLE_RE = re.compile(r"\d\s*º.*Grado\s+en\s+Ingenier[ií]a\s+Inform[aá]tica",
                       re.IGNORECASE)
_CUATRI_RE = re.compile(r"(?:1er|2[ºo]|primer|segundo|1|2)\.?\s*cuatrimestre",
                        re.IGNORECASE)
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$")
# Aula: 0.3, 3.4, -1.1, 3.4/3.11, A.1, B.2, A.2
_ROOM_RE = re.compile(r"^-?\d{1,2}\.\d{1,2}(?:[/-]\d{1,2}(?:\.\d{1,2})?)?$"
                      r"|^[A-Z]\.\d{1,2}$")
# Sigla de asignatura: 2-6 mayúsculas, opcional dígito final (MDA1, MAC)
_ABBR_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ]{2,6}\d?$")
# Token de grupo: (A1), (B3), (A4)
_GROUP_RE = re.compile(r"^\([A-Z]?\d\)$")
# Sigla de leyenda (admite punto/dos puntos al final): "CA", "DI.", "FFT."
_SIGLA_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ]{2,6}[.:]?$")
# Palabra "real": capitalizada con cola en minúsculas ("Cálculo", "Derecho").
# Distingue el inicio de una entrada de leyenda de una celda de clase (siglas).
_REAL_WORD_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}")


# ── Estructuras internas ──────────────────────────────────────────────────

class _Line:
    """Una línea visual: palabras (x0, texto) ordenadas por x y su y media."""

    __slots__ = ("y", "words")

    def __init__(self, y: float, words: list[tuple[float, str]]):
        self.y = y
        self.words = words  # [(x0, texto), ...] ordenadas por x0

    @property
    def text(self) -> str:
        return " ".join(w for _, w in self.words)


def _group_words_into_lines(words: list, y_tol: float = 3.0) -> list[_Line]:
    """Agrupa las palabras de ``get_text('words')`` en líneas por su y."""
    items = sorted(((w[1], w[0], w[4]) for w in words), key=lambda t: (t[0], t[1]))
    lines: list[_Line] = []
    cur_y: float | None = None
    cur: list[tuple[float, str]] = []
    for y0, x0, text in items:
        if not text.strip():
            continue
        if cur_y is None or abs(y0 - cur_y) <= y_tol:
            cur.append((x0, text))
            cur_y = y0 if cur_y is None else cur_y
        else:
            lines.append(_Line(cur_y, sorted(cur)))
            cur = [(x0, text)]
            cur_y = y0
    if cur:
        lines.append(_Line(cur_y, sorted(cur)))
    return lines


def _nearest_day(x: float, anchors: dict[str, float]) -> str | None:
    """Asigna una coordenada x al día cuyo ancla esté más cerca."""
    if not anchors:
        return None
    return min(anchors.items(), key=lambda kv: abs(kv[1] - x))[0]


def _detect_day_anchors(line: _Line) -> dict[str, float] | None:
    """Si la línea es un encabezado de días, devuelve {día: x_centro}."""
    found: dict[str, float] = {}
    for x0, w in line.words:
        key = w.strip().lower()
        if key in _DAYS_NORM:
            found[_DAYS_NORM[key]] = x0
    return found if len(found) >= 3 else None


def _is_room_line(line: _Line) -> bool:
    toks = [w for _, w in line.words]
    if not toks:
        return False
    room_like = sum(1 for t in toks if _ROOM_RE.match(t))
    has_time = any(_TIME_RE.match(t) for t in toks)
    return (not has_time) and room_like >= max(1, len(toks) // 2)


def _merge_groups(tokens: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Une un token de grupo ``(A1)`` con la sigla que lo precede."""
    out: list[tuple[float, str]] = []
    for x, t in tokens:
        if _GROUP_RE.match(t) and out:
            px, pt = out[-1]
            out[-1] = (px, f"{pt} {t}")
        else:
            out.append((x, t))
    return out


class HorarioParser:
    """Reconstruye un PDF de horarios en registros consultables por celda."""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=raw.content, filetype="pdf")
        # Recolectar todas las líneas (todas las páginas) una sola vez
        pages_lines: list[list[_Line]] = [
            _group_words_into_lines(page.get_text("words")) for page in doc
        ]
        num_pages = len(doc)
        doc.close()

        # ── Pasada 1: leyenda completa (sigla → nombre) ──────────────────
        # La leyenda puede aparecer DESPUÉS de las filas que la usan, por eso
        # se recopila por completo antes de reconstruir los registros.
        legend: dict[str, str] = {}
        for lines in pages_lines:
            for line in lines:
                entry = _parse_legend_entry(line)
                if entry:
                    legend[entry[0]] = entry[1]

        # ── Pasada 2: reconstrucción de registros por celda ──────────────
        records: list[str] = []
        # Acumulador para registros-resumen: (título, cuatrimestre) -> {siglas}.
        # Permite responder preguntas de AGREGACIÓN ("¿qué asignaturas tiene 3º
        # en el 2º cuatrimestre?") que el top-k no puede resolver listando
        # registros de clase individuales.
        summaries: dict[tuple[str, str], set[str]] = {}
        title = ""
        cuatri = ""
        anchors: dict[str, float] = {}

        for lines in pages_lines:
            i = 0
            while i < len(lines):
                line = lines[i]
                txt = line.text.strip()

                # 1) Título de rejilla
                if _TITLE_RE.search(txt):
                    title = txt
                    anchors = {}
                    i += 1
                    continue

                # 2) Cuatrimestre
                if _CUATRI_RE.search(txt) and len(txt) < 40:
                    cuatri = txt
                    i += 1
                    continue

                # 3) Encabezado de días → fijar columnas
                day_anchors = _detect_day_anchors(line)
                if day_anchors:
                    anchors = day_anchors
                    i += 1
                    continue

                # 5) Fila de franja horaria (subject row)
                time_tok = None
                if line.words and _TIME_RE.match(line.words[0][1]):
                    time_tok = line.words[0][1]

                # 4) Línea SOLO de leyenda (sin franja horaria): ya recogida en
                #    la pasada 1, se omite. OJO: las filas de horario que llevan
                #    una entrada de leyenda a la derecha NO se omiten — su leyenda
                #    se recorta con _legend_split_index al procesar la fila; de lo
                #    contrario se perderían las clases de esa franja.
                if time_tok is None and _parse_legend_entry(line):
                    i += 1
                    continue

                if time_tok and anchors:
                    # Recortar la zona de leyenda del margen derecho antes de
                    # asignar celdas (evita que siglas de la leyenda se cuelen
                    # como clases del último día).
                    leg_idx = _legend_split_index(line.words)
                    sched_words = line.words[1:leg_idx] if leg_idx is not None else line.words[1:]
                    subj_tokens = _merge_groups(sched_words)
                    # ¿La línea siguiente es de aulas?
                    rooms_by_day: dict[str, list[str]] = {}
                    if i + 1 < len(lines) and _is_room_line(lines[i + 1]):
                        room_line = lines[i + 1]
                        r_leg = _legend_split_index(room_line.words)
                        room_words = room_line.words[:r_leg] if r_leg is not None else room_line.words
                        for x, t in room_words:
                            d = _nearest_day(x, anchors)
                            if d:
                                rooms_by_day.setdefault(d, []).append(t)
                        i += 1  # consumir la línea de aulas

                    subj_by_day: dict[str, list[str]] = {}
                    for x, t in subj_tokens:
                        d = _nearest_day(x, anchors)
                        if d and (_ABBR_RE.match(t.split()[0]) or "(" in t):
                            subj_by_day.setdefault(d, []).append(t)

                    for day in _DAYS:
                        subs = subj_by_day.get(day)
                        if not subs:
                            continue
                        rooms = rooms_by_day.get(day, [])
                        records.append(
                            _format_record(title, cuatri, day, time_tok, subs, rooms, legend)
                        )
                        # Acumular siglas para el registro-resumen del grupo
                        bucket = summaries.setdefault((title, cuatri), set())
                        for s in subs:
                            sig = re.split(r"[ (]", s)[0].strip(".:")
                            if sig:
                                bucket.add(sig)

                i += 1

        # ── Registros-resumen por (curso/especialidad, cuatrimestre) ──────
        # Listan las asignaturas distintas de cada grupo, para responder
        # preguntas de agregación que el listado de clases individuales no cubre.
        n_summaries = 0
        for (s_title, s_cuatri), sigset in summaries.items():
            if not sigset:
                continue
            names = sorted(_expand(s, legend) for s in sigset)
            ctx = s_title + (f", {s_cuatri}" if s_cuatri else "")
            records.append(f"Asignaturas de {ctx}: {'; '.join(names)}.")
            n_summaries += 1

        text = "\n".join(records)
        logger.info(
            "Horario parseado: %s (%d registros, %d resúmenes, %d siglas leyenda, %d págs)",
            raw.url, len(records) - n_summaries, n_summaries, len(legend), num_pages,
        )
        return ParsedDocument(
            url=raw.url,
            source_type=raw.source_type,
            title="Horarios Grado en Ingeniería Informática",
            text=text,
            fetched_at=raw.fetched_at,
            metadata={"num_pages": num_pages, "n_registros": len(records),
                      "tipo": "horario"},
        )


def _legend_split_index(words: list[tuple[float, str]]) -> int | None:
    """Índice donde empieza la leyenda en una línea, o ``None``.

    Las entradas de leyenda ('CA Cálculo', 'DI. Derecho Informático.') aparecen
    al margen derecho de las filas del horario, no en líneas propias. Se detectan
    por patrón —una **sigla** seguida de una **palabra real** (capitalizada con
    minúsculas)— lo que permite separarlas de las celdas de clase (que solo
    contienen siglas, grupos ``(A1)`` y aulas) sin depender de coordenadas.
    """
    for i in range(len(words) - 1):
        if _SIGLA_RE.match(words[i][1]) and _REAL_WORD_RE.match(words[i + 1][1]):
            return i
    return None


def _parse_legend_entry(line: _Line) -> tuple[str, str] | None:
    """Extrae (sigla, nombre) de la porción de leyenda de la línea, si existe."""
    idx = _legend_split_index(line.words)
    if idx is None:
        return None
    sigla = line.words[idx][1].strip(".:")
    name = " ".join(w for _, w in line.words[idx + 1:]).strip(" .")
    if len(sigla) < 2 or not name:
        return None
    return sigla, name


def _expand(sigla_token: str, legend: dict[str, str]) -> str:
    """Devuelve 'Nombre Completo (SIGLA)[ grupo]' usando la leyenda si existe."""
    parts = sigla_token.split(" ", 1)
    sigla = parts[0]
    group = f" {parts[1]}" if len(parts) > 1 else ""
    name = legend.get(sigla)
    return f"{name} ({sigla}){group}" if name else f"{sigla}{group}"


def _format_record(title: str, cuatri: str, day: str, time: str,
                   subjects: list[str], rooms: list[str],
                   legend: dict[str, str]) -> str:
    subj_str = "; ".join(_expand(s, legend) for s in subjects)
    aula_str = f", aula {' / '.join(rooms)}" if rooms else ""
    ctx = title + (f", {cuatri}" if cuatri else "")
    return f"{ctx} — {day} {time}: {subj_str}{aula_str}."
