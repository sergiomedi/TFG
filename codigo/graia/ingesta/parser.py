"""Parser — extracción de texto estructurado desde documentos crudos.

Implementa el patrón Strategy descrito en la Sección 5.6 del diseño: una
estrategia concreta por cada ``SourceType`` (HTML, PDF, TXT), seleccionada
dinámicamente por la factoría :func:`parse`.

Objetivos de calidad (motivados por el análisis de errores del corpus v1):

  1. **Sin duplicación de texto.** El parser HTML v1 recorría ``descendants``
     emitiendo texto a varios niveles del DOM, produciendo líneas repetidas
     (``Asignatura\\nAsignatura``) que degradaban tanto los *embeddings* como
     la generación. El nuevo recorrido extrae el texto una sola vez y colapsa
     líneas consecutivas idénticas.
  2. **Tablas legibles.** Tanto en HTML como en PDF las tablas se convierten a
     **Markdown** (filas ``| celda | celda |``). Esto preserva la
     correspondencia fila–columna (planes de estudio, créditos, horarios) que
     el texto plano destruye, evitando que el LLM "se vuelva loco" al ver
     columnas mezcladas.
  3. **PDF escaneados / imágenes.** Cuando una página PDF carece de capa de
     texto (documento escaneado) se aplica **OCR** con Tesseract sobre la
     página rasterizada, de modo que su contenido entre igualmente al corpus.

Decisiones tecnológicas (justificadas en Cap. 5):
  - **HTML**: BeautifulSoup + ``lxml`` (parser más rápido de PyPI).
  - **PDF**: PyMuPDF (``fitz``) por velocidad y orden de lectura, con
    ``page.find_tables()`` (≥1.23) para extracción tabular nativa y
    rasterización para OCR.
  - **OCR**: ``pytesseract`` sobre el binario Tesseract (gratuito, local,
    reproducible; coherente con la restricción de ejecución 100 % local).
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

from bs4 import BeautifulSoup, NavigableString

from graia.ingesta.models import ParsedDocument, RawDocument, SourceType

logger = logging.getLogger(__name__)


# ============================================================================
# Utilidades compartidas
# ============================================================================

def _collapse_repeated_lines(text: str) -> str:
    """Colapsa líneas consecutivas idénticas (red de seguridad anti-duplicación).

    El renderizado de plantillas Drupal de la UGR genera con frecuencia el
    mismo texto dos veces seguidas (enlace + título accesible, encabezado +
    celda). Se elimina la repetición *consecutiva* preservando el orden y el
    contenido legítimo no adyacente.
    """
    out: list[str] = []
    prev: str | None = None
    for raw_line in text.split("\n"):
        line = raw_line.rstrip()
        key = line.strip()
        if key and key == prev:
            continue
        out.append(line)
        prev = key
    return "\n".join(out)


def _matrix_to_markdown(rows: list[list[str]]) -> str:
    """Convierte una matriz de celdas (lista de filas) a una tabla Markdown.

    Normaliza el número de columnas y usa la primera fila como cabecera. Las
    celdas vacías se conservan para no desalinear la correspondencia
    fila–columna.
    """
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    norm = [[(c or "").replace("\n", " ").strip() for c in r] + [""] * (ncol - len(r))
            for r in rows]
    header = norm[0]
    md = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * ncol) + " |",
    ]
    for r in norm[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


# ---------- Protocolo común (Strategy) ----------

class DocumentParser(Protocol):
    """Interfaz que todo parser concreto debe satisfacer."""

    def parse(self, raw: RawDocument) -> ParsedDocument: ...


# ============================================================================
# Estrategia HTML
# ============================================================================

_DISCARD_TAGS = {"nav", "footer", "header", "aside", "script", "style",
                 "noscript", "form", "button", "svg"}
_HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class HtmlParser:
    """Extrae texto limpio de un documento HTML con BeautifulSoup + lxml.

    Estrategia de extracción (corrige el bug de duplicación de v1):

      1. Se eliminan las etiquetas sin valor semántico (navegación, scripts…).
      2. Cada ``<table>`` se sustituye *in situ* por su versión Markdown, de
         modo que el texto tabular conserva su estructura.
      3. Cada encabezado ``<h1>``–``<h6>`` se sustituye por un marcador
         ``§H<n>§ <texto>`` que el *chunker* usará para propagar la ruta de
         cabeceras a cada fragmento (*contextual chunk headers*).
      4. Se extrae el texto **una sola vez** con ``get_text(separator='\\n')``
         y se colapsan líneas consecutivas duplicadas.
    """

    def parse(self, raw: RawDocument) -> ParsedDocument:
        html = raw.content.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")

        for tag in soup.find_all(_DISCARD_TAGS):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else None
        container = soup.find("main") or soup.find("article") or soup.body or soup

        # (2) Tablas → Markdown, sustituidas en el árbol
        n_tables = 0
        for table in container.find_all("table"):
            md = self._table_to_markdown(table)
            table.replace_with(NavigableString("\n\n" + md + "\n\n" if md else "\n"))
            if md:
                n_tables += 1

        # (3) Encabezados → marcadores §Hn§
        for h in container.find_all(_HEADER_TAGS):
            level = h.name[1]
            htext = h.get_text(" ", strip=True)
            h.replace_with(NavigableString(f"\n§H{level}§ {htext}\n" if htext else "\n"))

        # (4) Extracción única + colapso de duplicados
        text = container.get_text(separator="\n")
        text = _collapse_repeated_lines(text)

        logger.debug(
            "HTML parseado: %s (%d chars, %d tablas→Markdown)",
            raw.url, len(text), n_tables,
        )
        return ParsedDocument(
            url=raw.url,
            source_type=raw.source_type,
            title=title,
            text=text,
            fetched_at=raw.fetched_at,
            metadata={"n_tables": n_tables},
        )

    @staticmethod
    def _table_to_markdown(table) -> str:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if not cells:
                continue
            rows.append([c.get_text(" ", strip=True) for c in cells])
        return _matrix_to_markdown(rows)


# ============================================================================
# Estrategia PDF
# ============================================================================

class PdfParser:
    """Extrae texto, tablas y (si procede) OCR de un PDF con PyMuPDF.

    A diferencia de v1 —que solo hacía ``get_text('text')`` y reservaba
    pdfplumber para URLs con la palabra "horario"— este parser procesa **todos**
    los PDF de forma homogénea:

      - **Tablas**: ``page.find_tables()`` detecta tablas y ``to_markdown()``
        las serializa preservando filas y columnas. El texto corrido se extrae
        *excluyendo* las regiones tabulares para no duplicar su contenido.
      - **OCR**: si una página tiene una capa de texto pobre (umbral
        ``ocr_min_chars``) se asume escaneada y se rasteriza a ``ocr_dpi`` para
        aplicar Tesseract en español. Así los PDF escaneados (frecuentes en
        normativa y actas) entran al corpus en lugar de quedar vacíos.

    Parameters
    ----------
    enable_ocr : bool
        Activa el OCR de páginas escaneadas (requiere el binario ``tesseract``).
    ocr_lang : str
        Idioma(s) Tesseract, p. ej. ``"spa"`` o ``"spa+eng"``.
    ocr_min_chars : int
        Si el texto nativo de la página tiene menos caracteres que este umbral,
        se intenta OCR.
    ocr_dpi : int
        Resolución de rasterización para OCR (150–300 recomendado).
    extract_tables : bool
        Activa la detección de tablas con ``find_tables()``.
    """

    def __init__(
        self,
        *,
        enable_ocr: bool = True,
        ocr_lang: str = "spa",
        ocr_min_chars: int = 60,
        ocr_dpi: int = 200,
        extract_tables: bool = True,
    ) -> None:
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        self.ocr_min_chars = ocr_min_chars
        self.ocr_dpi = ocr_dpi
        self.extract_tables = extract_tables

    def parse(self, raw: RawDocument) -> ParsedDocument:
        import fitz  # PyMuPDF (import diferido: dependencia pesada)

        doc = fitz.open(stream=raw.content, filetype="pdf")
        pages_out: list[str] = []
        n_tables = 0
        n_ocr_pages = 0

        for page in doc:
            parts: list[str] = []

            # --- Tablas de la página ---
            table_rects: list = []
            if self.extract_tables:
                try:
                    finder = page.find_tables()
                    for tab in finder.tables:
                        md = self._table_to_markdown(tab)
                        if md:
                            parts.append(md)
                            n_tables += 1
                            table_rects.append(fitz.Rect(tab.bbox))
                except Exception as exc:  # find_tables puede fallar en páginas atípicas
                    logger.debug("find_tables falló en %s: %s", raw.url, exc)

            # --- Texto corrido excluyendo regiones tabulares ---
            body_text = self._text_excluding(page, table_rects)
            if body_text.strip():
                parts.insert(0, body_text)

            # --- OCR si la página parece escaneada ---
            page_chars = sum(len(p) for p in parts)
            if self.enable_ocr and page_chars < self.ocr_min_chars:
                ocr_text = self._ocr_page(page, fitz)
                if ocr_text.strip():
                    parts.append(ocr_text)
                    n_ocr_pages += 1

            if parts:
                pages_out.append("\n\n".join(parts))

        text = _collapse_repeated_lines("\n\n".join(pages_out))
        metadata = dict(doc.metadata or {})
        title = (metadata.get("title") or "").strip() or None
        num_pages = len(doc)
        doc.close()

        logger.info(
            "PDF parseado: %s (%d págs, %d tablas, %d págs OCR, %d chars)",
            raw.url, num_pages, n_tables, n_ocr_pages, len(text),
        )
        return ParsedDocument(
            url=raw.url,
            source_type=raw.source_type,
            title=title,
            text=text,
            fetched_at=raw.fetched_at,
            metadata={"num_pages": num_pages, "n_tables": n_tables,
                      "n_ocr_pages": n_ocr_pages},
        )

    @staticmethod
    def _table_to_markdown(tab) -> str:
        """Serializa una tabla de PyMuPDF a Markdown de forma robusta."""
        try:
            md = tab.to_markdown()
            if md and md.strip():
                return md.strip()
        except Exception:
            pass
        # Fallback: extraer la matriz manualmente
        try:
            return _matrix_to_markdown([[c or "" for c in row] for row in tab.extract()])
        except Exception:
            return ""

    @staticmethod
    def _text_excluding(page, rects: list) -> str:
        """Texto de la página excluyendo bloques que solapan con tablas."""
        if not rects:
            return page.get_text("text")
        import fitz
        kept: list[str] = []
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, btext = block[0], block[1], block[2], block[3], block[4]
            brect = fitz.Rect(x0, y0, x1, y1)
            # Descartar el bloque si su centro cae dentro de alguna tabla
            center = ((x0 + x1) / 2, (y0 + y1) / 2)
            in_table = any(r.contains(center) for r in rects)
            if not in_table and btext.strip():
                kept.append(btext.strip())
        return "\n".join(kept)

    def _ocr_page(self, page, fitz) -> str:
        """Rasteriza la página y aplica OCR con Tesseract."""
        try:
            import pytesseract
            from PIL import Image
            import io

            zoom = self.ocr_dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img, lang=self.ocr_lang)
        except Exception as exc:
            logger.warning("OCR falló en página: %s", exc)
            return ""


# ============================================================================
# Estrategia TXT
# ============================================================================

class TxtParser:
    """Devuelve el contenido de un fichero de texto plano sin transformaciones."""

    def parse(self, raw: RawDocument) -> ParsedDocument:
        text = raw.content.decode("utf-8", errors="replace")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        title = lines[0] if lines else None
        logger.debug("TXT parseado: %s (%d chars)", raw.url, len(text))
        return ParsedDocument(
            url=raw.url,
            source_type=raw.source_type,
            title=title,
            text=text,
            fetched_at=raw.fetched_at,
        )


# ============================================================================
# Factoría
# ============================================================================

def parse(raw: RawDocument, *, pdf_options: dict | None = None) -> ParsedDocument:
    """Punto de entrada: selecciona el parser adecuado según ``source_type``.

    Parameters
    ----------
    raw : RawDocument
        Documento crudo descargado por el *fetcher*.
    pdf_options : dict | None
        Opciones para :class:`PdfParser` (OCR, tablas). Si es ``None`` se usan
        los valores por defecto.
    """
    if raw.source_type is SourceType.HTML:
        return HtmlParser().parse(raw)
    if raw.source_type is SourceType.PDF:
        # PDFs de horarios: estrategia dedicada de reconstrucción de la rejilla.
        # Si no logra reconstruir ningún registro, se cae al PdfParser genérico
        # para no perder cobertura.
        if "horario" in raw.url.lower():
            from graia.ingesta.horario_parser import HorarioParser
            parsed = HorarioParser().parse(raw)
            if parsed.text.strip():
                return parsed
            logger.info(
                "HorarioParser no reconstruyó registros para %s; "
                "se usa el PdfParser genérico.", raw.url,
            )
        return PdfParser(**(pdf_options or {})).parse(raw)
    if raw.source_type is SourceType.TXT:
        return TxtParser().parse(raw)
    raise ValueError(f"No hay parser registrado para {raw.source_type}")
