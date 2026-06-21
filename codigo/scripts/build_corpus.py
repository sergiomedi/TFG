#!/usr/bin/env python3
"""build_corpus.py — Crawler dirigido por categoría para GRAIA (v2).

Reemplaza el crawl genérico (v1) por un enfoque de **lista blanca por
categoría**: cada perfil definido en ``sources.yaml`` declara sus propias
seeds y ``include_patterns``.  Un enlace descubierto que no case con ningún
``include_pattern`` de la categoría activa se ignora, lo que garantiza que el
corpus resultante contiene únicamente documentos relevantes.

Mejoras respecto a v1:
  - Crawl independiente por categoría con include_patterns (whitelist)
  - Filtrado temporal automático configurable por categoría
  - Deduplicación por hash de contenido (elimina copias Drupal _N)
  - Umbral mínimo de caracteres configurable
  - Mejor cobertura de formatos de año en URLs

Ejecución:
    cd codigo/
    python scripts/build_corpus.py                     # crawl completo
    python scripts/build_corpus.py --categories calendario normativa
    python scripts/build_corpus.py --max-pages 50
    python scripts/build_corpus.py --dry-run            # solo descubre, no descarga

Salida:
    data/raw/crawl_manifest.json    — registro de todo lo descargado
    data/processed/corpus.jsonl     — documentos limpios listos para revisar
    data/processed/crawl_report.txt — informe resumen del crawl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urldefrag, unquote as url_unquote

import yaml

# Añadir el directorio raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graia.ingesta.fetcher import fetch
from graia.ingesta.parser import parse
from graia.ingesta.cleaner import clean
from graia.ingesta.models import RawDocument, SourceType

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_corpus")


# ── Constantes ──────────────────────────────────────────────────────────────

_MESES_ES = (
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
)

_MESES_ABREV = (
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
)


# ── Filtrado temporal automático ────────────────────────────────────────────

def build_year_exclusion_patterns(current_year: str) -> list[re.Pattern]:
    """Genera regexes que excluyen URLs de cursos académicos anteriores.

    Dada ``current_year = "25-26"``, se generan patrones para cursos 15-16
    a 24-25 en todas las variantes de formato observadas en la ETSIIT:

      - ``(XX-YY)`` / ``XX-YY``   → Horarios GIIM (20-21).pdf
      - ``20XX-YY`` / ``20XX_YY`` → Calendario TFG 2024-2025.pdf
      - ``20XX.YY``               → formatos con punto
      - ``YYYY_YYYY``             → ETSIIT_Calendario_2023_2024_v2.pdf
      - ``MesYYYY`` / ``Mes YYYY``→ Asignación Aulas Febrero2024.pdf
      - ``MonYY`` abreviado       → AsignaciónAulasGIINov2020.pdf, nov24.pdf

    Para documentos temporales (calendarios, horarios, asignaciones de aulas),
    ``Mes 2025`` se considera del curso 24-25 (segundo semestre), no del 25-26.
    """
    parts = current_year.split("-")
    if len(parts) != 2:
        logger.error("current_academic_year debe ser 'YY-YY+1', recibido: %s", current_year)
        return []

    current_start = int(parts[0])  # e.g., 25
    current_full_start = 2000 + current_start      # 2025
    current_full_end = 2000 + current_start + 1     # 2026

    # Cursos anteriores: desde 15-16 hasta (current-1)-(current)
    old_courses = [(s, s + 1) for s in range(15, current_start)]
    if not old_courses:
        return []

    patterns: list[re.Pattern] = []

    # 1. Formato corto (XX-YY) con/sin paréntesis
    short_alts = "|".join(f"{s:02d}-{e:02d}" for s, e in old_courses)
    patterns.append(re.compile(
        rf"(?:\(|[^0-9])(?:{short_alts})(?:\)|[^0-9]|$)", re.IGNORECASE,
    ))

    # 2. Formato largo 20XX-YY, 20XX_YY
    long_alts = "|".join(
        f"20{s:02d}[-_]{e:02d}" for s, e in old_courses
    )
    patterns.append(re.compile(rf"(?:{long_alts})", re.IGNORECASE))

    # 3. Formato YYYY-YYYY / YYYY_YYYY con años completos
    full_alts = "|".join(
        f"20{s:02d}[-_]20{e:02d}" for s, e in old_courses
    )
    patterns.append(re.compile(rf"(?:{full_alts})", re.IGNORECASE))

    # 4. Formato con punto 20XX.YY
    dot_alts = "|".join(f"20{s:02d}\\.{e:02d}" for s, e in old_courses)
    patterns.append(re.compile(rf"(?:{dot_alts})", re.IGNORECASE))

    # 5. Mes completo + año antiguo: "Febrero 2024", "Noviembre2023", etc.
    #    Incluye año == current_full_start (2025) porque Mes+2025 en documentos
    #    temporales pertenece al curso 24-25 (segundo semestre), no al 25-26.
    old_years_full = "|".join(str(y) for y in range(2015, current_full_start + 1))
    meses_full = "|".join(_MESES_ES)
    patterns.append(re.compile(
        rf"(?:{meses_full})\s?(?:{old_years_full})(?!\d)", re.IGNORECASE,
    ))

    # 6. Mes abreviado + año: "Nov2020", "Nov21", "nov24", etc.
    old_short_years = "|".join(f"{y:02d}" for y in range(15, current_start + 1))
    old_long_years = "|".join(str(y) for y in range(2015, current_full_start + 1))
    meses_abrev = "|".join(_MESES_ABREV)
    patterns.append(re.compile(
        rf"(?:{meses_abrev})(?:{old_long_years}|{old_short_years})(?!\d)", re.IGNORECASE,
    ))

    return patterns


# ── Filtrado por extensión ──────────────────────────────────────────────────

def has_allowed_extension(url: str, allowed_extensions: list[str]) -> bool:
    """True si la URL tiene extensión permitida o no tiene extensión."""
    path = url_unquote(urlparse(url).path).lower()
    _, ext = os.path.splitext(path)
    if not ext:
        return True
    return ext.lower() in [e.lower() for e in allowed_extensions]


# ── Filtrado combinado ──────────────────────────────────────────────────────

def passes_global_filters(
    url: str,
    allowed_domains: list[str],
    exclude_patterns: list[re.Pattern],
    allowed_extensions: list[str],
    year_patterns: list[re.Pattern] | None = None,
    *,
    apply_temporal: bool = True,
) -> bool:
    """Comprueba si una URL pasa los filtros globales.

    Filtros (en orden):
      1. Dominio
      2. Extensión de fichero
      3. Exclusiones estáticas (regex sobre URL codificada)
      4. Exclusión temporal (regex sobre URL decodificada) — solo si apply_temporal
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # 1. Dominio
    if not any(domain == d or domain.endswith(f".{d}") for d in allowed_domains):
        return False

    # 2. Extensión
    if not has_allowed_extension(url, allowed_extensions):
        return False

    # 3. Exclusiones estáticas
    for pat in exclude_patterns:
        if pat.search(url):
            return False

    # 4. Exclusión temporal
    if apply_temporal and year_patterns:
        decoded = url_unquote(url)
        for pat in year_patterns:
            if pat.search(decoded):
                return False

    return True


def matches_category(url: str, include_patterns: list[re.Pattern]) -> bool:
    """True si la URL decodificada casa con al menos un include_pattern."""
    decoded = url_unquote(url)
    return any(pat.search(decoded) for pat in include_patterns)


# ── Extracción de enlaces ────────────────────────────────────────────────────

def _normalize_url(url: str) -> str:
    """Normaliza una URL: http→https para dominios UGR, elimina trailing slash."""
    # Muchos enlaces del CMS Drupal de la ETSIIT usan http:// aunque el
    # servidor sirve HTTPS.  Normalizar evita duplicados http/https y
    # garantiza que los exclude_patterns funcionen con ambos esquemas.
    if url.startswith("http://") and "ugr.es" in url:
        url = "https://" + url[7:]
    return url.rstrip("/") if not url.endswith(".pdf") else url


def extract_links(html_bytes: bytes, base_url: str) -> set[str]:
    """Extrae hipervínculos de un documento HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_bytes, "lxml")
    links: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        defragged, _ = urldefrag(absolute)
        normalized = _normalize_url(defragged)
        links.add(normalized)

    return links


# ── Estado del crawl ────────────────────────────────────────────────────────

class CrawlState:
    """Estado compartido del crawl: URLs visitadas, manifiesto, estadísticas."""

    def __init__(self):
        self.visited: set[str] = set()
        self.manifest: list[dict] = []
        self.stats = defaultdict(int)
        self.content_hashes: dict[str, str] = {}  # hash → primera URL

    def save_manifest(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2, default=str)


# ── Crawl dirigido por categoría ────────────────────────────────────────────

def crawl_category(
    cat_name: str,
    cat_cfg: dict,
    global_cfg: dict,
    state: CrawlState,
    year_patterns: list[re.Pattern],
) -> list[tuple[RawDocument, str]]:
    """Ejecuta el crawl para una categoría individual.

    Solo descarga URLs que casen con los ``include_patterns`` de la categoría
    (o que sean seeds directas).  Respeta los filtros globales.

    Cada categoría tiene su propio límite de páginas (``max_pages`` en el
    perfil de categoría).  Si no se especifica, se usa un valor por defecto
    razonable.  El límite global ``max_pages_total`` actúa como techo de
    seguridad sobre el total acumulado de todas las categorías.
    """
    allowed_domains = global_cfg["allowed_domains"]
    exclude_pats = [re.compile(p) for p in global_cfg.get("exclude_patterns", [])]
    allowed_ext = global_cfg.get("allowed_extensions", [".html", ".htm", ".pdf"])
    delay = global_cfg.get("politeness_delay_s", 1.0)
    max_pages_global = global_cfg.get("max_pages_total", 2000)

    seeds = cat_cfg.get("seeds", [])
    include_pats = [re.compile(p) for p in cat_cfg.get("include_patterns", [])]
    cat_exclude_pats = [re.compile(p) for p in cat_cfg.get("exclude_patterns", [])]
    follow_links = cat_cfg.get("follow_links", True)
    max_depth = cat_cfg.get("max_depth", 3)
    is_temporal = cat_cfg.get("temporal", False)
    download_seeds = cat_cfg.get("download_seeds", True)
    max_pages_cat = cat_cfg.get("max_pages", 200)  # Límite por categoría

    # Cola BFS: (url, depth, is_seed)
    # Las seeds SIEMPRE se encolan aunque estén en state.visited.
    # Motivo: el BFS de categorías anteriores descubre URLs del menú de
    # navegación y las marca como visitadas sin descargarlas (no casan con
    # sus include_patterns).  Si las seeds de categorías posteriores
    # respetasen state.visited, se perderían categorías enteras.
    queue: list[tuple[str, int, bool]] = []
    for seed in seeds:
        seed = _normalize_url(seed)
        queue.append((seed, 0, True))
        state.visited.add(seed)

    # Visited local: controla el BFS dentro de esta categoría para que
    # cada categoría descubra enlaces de forma independiente.  El
    # state.visited global solo evita re-descargar; cat_visited evita
    # re-encolar dentro del mismo BFS.
    cat_visited: set[str] = set(s for s, _, _ in queue)

    documents: list[tuple[RawDocument, str]] = []
    cat_fetched = 0

    logger.info("── Categoría: %s (%d seeds, temporal=%s, max_depth=%d, max_pages=%d) ──",
                cat_name, len(seeds), is_temporal, max_depth, max_pages_cat)

    while queue and cat_fetched < max_pages_cat and state.stats["pages_fetched"] < max_pages_global:
        url, depth, is_seed = queue.pop(0)

        # Filtros globales (dominio, extensión, exclusiones)
        if not is_seed and not passes_global_filters(
            url, allowed_domains, exclude_pats, allowed_ext,
            year_patterns=year_patterns, apply_temporal=is_temporal,
        ):
            state.stats["filtered_global"] += 1
            continue

        # Para no-seeds, verificar include_patterns y exclude_patterns de categoría
        if not is_seed and not matches_category(url, include_pats):
            state.stats["filtered_category"] += 1
            continue

        # Exclusiones específicas de la categoría (ej: otros grados, experiencias)
        if cat_exclude_pats:
            decoded_url = url_unquote(url)
            if any(pat.search(decoded_url) for pat in cat_exclude_pats):
                state.stats["filtered_category"] += 1
                continue

        # Evitar re-descarga: si otra categoría ya descargó esta URL,
        # no repetir la petición HTTP.  Las seeds están exentas porque
        # están configuradas explícitamente para esta categoría.
        if not is_seed and url in state.visited:
            state.stats["filtered_global"] += 1
            continue

        # Descargar
        logger.info("[%s] (d=%d) %s",
                    cat_name, depth, url_unquote(url)[-80:])
        raw = fetch(url, user_agent="GRAIA-academic-crawler/0.1 (+sergiomedinam98@gmail.com)")

        if raw is None:
            state.stats["failed"] += 1
            state.manifest.append({
                "url": url, "status": "failed", "category": cat_name,
                "depth": depth, "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            time.sleep(delay)
            continue

        state.stats["pages_fetched"] += 1
        cat_fetched += 1
        state.stats[raw.source_type.value] += 1
        state.visited.add(url)  # Marcar como descargada globalmente

        # Registrar en manifiesto
        state.manifest.append({
            "url": url, "status": "ok", "category": cat_name,
            "type": raw.source_type.value, "bytes": len(raw.content),
            "depth": depth, "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        documents.append((raw, cat_name))

        # Descubrir enlaces (solo HTML)
        if follow_links and raw.source_type == SourceType.HTML and depth < max_depth:
            new_links = extract_links(raw.content, url)
            discovered = 0
            for link in new_links:
                # Usar cat_visited (local) para el BFS, NO state.visited
                # (global).  Así cada categoría descubre enlaces de forma
                # independiente y no "roba" URLs a categorías posteriores.
                if link in cat_visited:
                    continue

                # Pre-filtrar antes de encolar
                if not passes_global_filters(
                    link, allowed_domains, exclude_pats, allowed_ext,
                    year_patterns=year_patterns, apply_temporal=is_temporal,
                ):
                    continue

                # Solo encolar si casa con la categoría (o es un link general
                # que podría llevar a contenido relevante — HTML sin extensión)
                decoded_link = url_unquote(link)
                _, link_ext = os.path.splitext(urlparse(decoded_link).path)

                if link_ext.lower() in (".pdf",):
                    # PDFs: solo si casan con include_patterns
                    if not matches_category(link, include_pats):
                        continue
                # HTML: seguimos para descubrir más enlaces (se filtrará después)

                cat_visited.add(link)
                queue.append((link, depth + 1, False))
                discovered += 1

            if discovered > 0:
                logger.debug("   └── %d nuevos enlaces encolados", discovered)

        # Guardar manifiesto periódicamente
        if state.stats["pages_fetched"] % 25 == 0:
            manifest_path = Path("data/raw/crawl_manifest.json")
            state.save_manifest(manifest_path)

        time.sleep(delay)

    logger.info("   → %s: %d documentos descargados", cat_name, cat_fetched)
    state.stats[f"cat_{cat_name}"] = cat_fetched

    return documents


# ── Post-procesado: títulos y detección de formularios ─────────────────────

def _infer_title_from_url(url: str, label: str | None = None) -> str:
    """Infiere un título legible a partir del nombre del fichero en la URL.

    Transforma ``Horarios%20GII%20(25-26).pdf`` → ``Horarios GII (25-26)``.
    Se usa cuando el parser no pudo extraer un título del contenido.

    Si se proporciona ``label`` (p.ej., desde extra_urls), se usa directamente.
    """
    if label:
        return label

    path = url_unquote(urlparse(url).path)
    filename = Path(path).stem  # sin extensión
    # Reemplazar guiones bajos y separadores por espacios
    title = filename.replace("_", " ")
    # Colapsar espacios múltiples
    title = re.sub(r"\s+", " ", title).strip()
    return title


_FORM_INDICATORS = re.compile(
    r"D\.N\.I\.|Apellido\s*1|Firma\s+del|Firma$|"
    r"Impreso\s+de\s+solicitud|Modelo\s+de\s+solicitud",
    re.IGNORECASE | re.MULTILINE,
)


def _is_fillable_form(text: str) -> bool:
    """Detecta si el texto corresponde a un formulario/impreso para rellenar.

    Heurística: si en los primeros 500 caracteres aparecen ≥2 indicadores
    de formulario (D.N.I., Apellido 1, Firma…) es un impreso, no contenido
    informativo útil para el RAG.
    """
    matches = _FORM_INDICATORS.findall(text[:500])
    return len(matches) >= 2


def _is_index_page(text: str, source_type: SourceType, char_count: int) -> bool:
    """Detecta si un documento HTML es una página índice / menú de navegación.

    Heurística combinada:
      1. Solo aplica a HTML (los PDFs siempre tienen contenido propio).
      2. Páginas muy cortas (<600 chars) son casi siempre índices.
      3. Páginas cortas (<1500 chars) con alta densidad de líneas cortas
         (típico de listas de enlaces) se descartan.

    Esto elimina sitemaps, menús de sección y páginas que solo listan
    enlaces a subpáginas, sin aportar contenido informativo al RAG.
    """
    if source_type != SourceType.HTML:
        return False

    # Páginas extremadamente cortas: casi siempre son índices
    if char_count < 600:
        return True

    # Páginas cortas con muchas líneas breves (listas de enlaces)
    if char_count < 1500:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return True
        short_lines = sum(1 for l in lines if len(l) < 60)
        ratio = short_lines / len(lines)
        # Si >70% de líneas son cortas, es probablemente un índice
        if ratio > 0.7 and len(lines) > 3:
            return True

    return False


# ── Procesado (parse + clean → JSONL) ───────────────────────────────────────

def process_documents(
    documents: list[tuple[RawDocument, str]],
    output_path: Path,
    *,
    min_chars: int = 200,
    deduplicate: bool = True,
    url_labels: dict[str, str] | None = None,
    pdf_options: dict | None = None,
) -> dict:
    """Procesa documentos descargados → JSONL con deduplicación y filtrado.

    Args:
        documents: Lista de (RawDocument, category).
        output_path: Ruta del fichero JSONL de salida.
        min_chars: Umbral mínimo de caracteres (por debajo se descarta).
        deduplicate: Si True, descarta documentos con hash de contenido repetido.
        url_labels: Diccionario URL → label para títulos de extra_urls.

    Returns:
        Diccionario con estadísticas del procesado.
    """
    url_labels = url_labels or {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(int)
    seen_hashes: dict[str, str] = {}  # hash → URL (para log)

    with open(output_path, "w", encoding="utf-8") as f:
        for raw, category in documents:
            try:
                parsed = parse(raw, pdf_options=pdf_options)
            except Exception as e:
                logger.warning("Error parseando %s: %s", raw.url, e)
                stats["parse_errors"] += 1
                continue

            cleaned = clean(parsed)
            if cleaned is None:
                stats["too_short_cleaner"] += 1
                continue

            text = cleaned.text
            char_count = len(text)

            # Filtro por min_chars
            if char_count < min_chars:
                logger.debug("   Descartado (<min_chars): %d chars — %s",
                             char_count, raw.url[-60:])
                stats["too_short"] += 1
                continue

            # Detectar formularios para rellenar (D.N.I., Firma, Apellido…)
            if _is_fillable_form(text):
                logger.debug("   Formulario descartado: %s", raw.url[-60:])
                stats["forms_discarded"] += 1
                continue

            # Detectar páginas índice / menú de navegación
            if _is_index_page(text, cleaned.source_type, char_count):
                logger.debug("   Página índice descartada: %d chars — %s",
                             char_count, raw.url[-60:])
                stats["index_pages_discarded"] += 1
                continue

            # Deduplicación por hash
            if deduplicate:
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                if content_hash in seen_hashes:
                    logger.debug("   Duplicado de %s — %s",
                                 seen_hashes[content_hash][-50:], raw.url[-50:])
                    stats["duplicates"] += 1
                    continue
                seen_hashes[content_hash] = raw.url

            # Título: prioridad label configurado > título extraído > URL
            # El label de extra_urls tiene prioridad absoluta porque es
            # configuración humana explícita.  El título extraído del PDF
            # puede ser incorrecto (ej: "Calendario en Blanco" por un
            # placeholder en la plantilla del documento).
            configured_label = url_labels.get(_normalize_url(raw.url))
            title = cleaned.title or ""
            if configured_label:
                title = configured_label
                logger.debug("   Título por label: '%s' ← %s", title, raw.url[-50:])
            elif len(title.strip()) <= 1:
                title = _infer_title_from_url(raw.url)
                if title:
                    logger.debug("   Título inferido: '%s' ← %s", title, raw.url[-50:])

            record = {
                "url": cleaned.url,
                "title": title,
                "category": category,
                "source_type": cleaned.source_type.value,
                "text": text,
                "char_count": char_count,
                "fetched_at": cleaned.fetched_at.isoformat(),
                "metadata": cleaned.metadata,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # Forzar escritura a disco tras cada registro
            stats["processed"] += 1
            stats[f"cat_{category}"] += 1

    logger.info("Procesados: %d documentos → %s", stats["processed"], output_path)
    logger.info("  Parse errors: %d | Cortos (cleaner): %d | Cortos (min_chars): %d | Formularios: %d | Índices: %d | Duplicados: %d",
                stats["parse_errors"], stats["too_short_cleaner"],
                stats["too_short"], stats["forms_discarded"],
                stats["index_pages_discarded"], stats["duplicates"])
    return dict(stats)


# ── Exportación CSV ──────────────────────────────────────────────────────────

def export_csv(jsonl_path: Path, csv_path: Path | None = None) -> Path:
    """Genera un CSV resumen a partir del corpus JSONL.

    Columnas: url, title, category, source_type, char_count, fetched_at.
    Se omite el texto completo para facilitar la revisión rápida.
    """
    if csv_path is None:
        csv_path = jsonl_path.with_suffix(".csv")

    fieldnames = ["url", "title", "category", "source_type", "char_count", "fetched_at"]

    with open(jsonl_path, encoding="utf-8") as fin, \
         open(csv_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for line in fin:
            if line.strip():
                record = json.loads(line)
                writer.writerow(record)

    logger.info("CSV exportado: %s", csv_path)
    return csv_path


# ── Informe de resumen ───────────────────────────────────────────────────────

def write_report(state: CrawlState, process_stats: dict, report_path: Path) -> None:
    """Genera un informe legible del crawl y procesado."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 60,
        "INFORME DE CRAWL Y PROCESADO — GRAIA v2",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "── CRAWL ──",
        f"  Páginas descargadas:      {state.stats['pages_fetched']}",
        f"  HTML:                     {state.stats.get('html', 0)}",
        f"  PDF:                      {state.stats.get('pdf', 0)}",
        f"  Filtradas (global):       {state.stats.get('filtered_global', 0)}",
        f"  Filtradas (categoría):    {state.stats.get('filtered_category', 0)}",
        f"  Fallidas:                 {state.stats.get('failed', 0)}",
        "",
        "── CRAWL POR CATEGORÍA ──",
    ]

    for key, val in sorted(state.stats.items()):
        if key.startswith("cat_"):
            lines.append(f"  {key[4:]:<25s} {val}")

    lines.extend([
        "",
        "── PROCESADO ──",
        f"  Documentos válidos:       {process_stats.get('processed', 0)}",
        f"  Errores de parsing:       {process_stats.get('parse_errors', 0)}",
        f"  Descartados (cleaner):    {process_stats.get('too_short_cleaner', 0)}",
        f"  Descartados (min_chars):  {process_stats.get('too_short', 0)}",
        f"  Formularios descartados:  {process_stats.get('forms_discarded', 0)}",
        f"  Índices descartados:      {process_stats.get('index_pages_discarded', 0)}",
        f"  Duplicados eliminados:    {process_stats.get('duplicates', 0)}",
        "",
        "── CORPUS POR CATEGORÍA ──",
    ])

    for key, val in sorted(process_stats.items()):
        if key.startswith("cat_"):
            lines.append(f"  {key[4:]:<25s} {val}")

    lines.extend(["", "── URLs DESCARGADAS ──"])
    for entry in state.manifest:
        status = "✓" if entry.get("status") == "ok" else "✗"
        etype = entry.get("type", "?")
        cat = entry.get("category", "?")
        lines.append(f"  {status} [{etype:4s}] [{cat:16s}] (d={entry.get('depth', '?')}) {entry['url']}")

    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("Informe guardado en: %s", report_path)


# -- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crawler dirigido por categoria para GRAIA (v2)"
    )
    parser.add_argument(
        "--config", default="config/sources.yaml",
        help="Ruta al fichero de fuentes (default: config/sources.yaml)"
    )
    parser.add_argument(
        "--categories", nargs="*", default=None,
        help="Categorias a procesar (default: todas). Ej: --categories calendario normativa"
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Sobrescribir el limite maximo de paginas total"
    )
    parser.add_argument(
        "--output", default="data/processed/corpus.jsonl",
        help="Ruta de salida del corpus procesado"
    )
    args = parser.parse_args()

    # Cargar configuracion
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Fichero de configuracion no encontrado: %s", config_path)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    global_cfg = config["global"]
    categorias = config.get("categorias", {})
    extra_urls = config.get("extra_urls", [])

    if args.max_pages:
        global_cfg["max_pages_total"] = args.max_pages

    # Filtrar categorias si se especifican
    if args.categories:
        invalid = set(args.categories) - set(categorias.keys())
        if invalid:
            logger.error("Categorias no encontradas: %s", invalid)
            logger.info("Disponibles: %s", list(categorias.keys()))
            sys.exit(1)
        categorias = {k: v for k, v in categorias.items() if k in args.categories}

    # Generar patrones de exclusion temporal
    current_year = global_cfg.get("current_academic_year", None)
    year_patterns: list[re.Pattern] = []
    if current_year:
        year_patterns = build_year_exclusion_patterns(current_year)
        logger.info("Curso vigente: %s -> %d patrones temporales generados",
                     current_year, len(year_patterns))
    else:
        logger.warning("current_academic_year no definido -- sin filtrado temporal")

    # -- Fase 1: Crawl por categorias --

    state = CrawlState()
    all_documents: list[tuple[RawDocument, str]] = []

    logger.info("=" * 60)
    logger.info("CRAWL DIRIGIDO POR CATEGORIA -- %d categorias + %d extras",
                len(categorias), len(extra_urls))
    logger.info("Dominios: %s | Extensiones: %s",
                global_cfg["allowed_domains"],
                global_cfg.get("allowed_extensions", ["todas"]))
    logger.info("min_chars: %d | deduplicate: %s",
                global_cfg.get("min_chars", 200),
                global_cfg.get("deduplicate", True))
    logger.info("=" * 60)

    for cat_name, cat_cfg in categorias.items():
        cat_docs = crawl_category(cat_name, cat_cfg, global_cfg, state, year_patterns)
        all_documents.extend(cat_docs)

    # -- Extra URLs (descarga directa, sin crawl) --

    if extra_urls:
        logger.info("-- Extra URLs: %d documentos directos --", len(extra_urls))
        delay = global_cfg.get("politeness_delay_s", 1.0)
        exclude_pats = [re.compile(p) for p in global_cfg.get("exclude_patterns", [])]

        for entry in extra_urls:
            url = _normalize_url(entry["url"])
            category = entry.get("category", "extra")

            # Las extra_urls SIEMPRE se descargan, sin comprobar
            # state.visited.  Son URLs configuradas explícitamente y
            # pueden haber sido marcadas como "visitadas" por el BFS
            # de categorías anteriores sin haber sido descargadas.
            # Solo se saltan si el exclude_pattern global las bloquea.

            # Aplicar filtros globales (exclusiones estaticas, extensiones)
            decoded_url = url_unquote(url)
            if any(pat.search(decoded_url) for pat in exclude_pats):
                logger.debug("   Extra excluida (patron global): %s", url[-60:])
                state.stats["filtered_global"] += 1
                continue

            state.visited.add(url)
            logger.info("[extra] %s", url[-80:])
            raw = fetch(url, user_agent="GRAIA-academic-crawler/0.1 (+sergiomedinam98@gmail.com)")

            if raw is None:
                state.stats["failed"] += 1
                state.manifest.append({
                    "url": url, "status": "failed", "category": category,
                    "depth": 0, "timestamp": datetime.now(timezone.utc).isoformat(),
                })
            else:
                state.stats["pages_fetched"] += 1
                state.stats[raw.source_type.value] += 1
                state.manifest.append({
                    "url": url, "status": "ok", "category": category,
                    "type": raw.source_type.value, "bytes": len(raw.content),
                    "depth": 0, "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                all_documents.append((raw, category))

            time.sleep(delay)

    # Guardar manifiesto final
    manifest_path = Path("data/raw/crawl_manifest.json")
    state.save_manifest(manifest_path)

    logger.info("=" * 60)
    logger.info("CRAWL COMPLETADO -- %d documentos descargados",
                state.stats["pages_fetched"])
    logger.info("=" * 60)

    if not all_documents:
        logger.warning("No se descargo ningun documento. Abortando.")
        sys.exit(1)

    # -- Fase 2: Procesar --

    # Construir diccionario URL -> label para titulos de extra_urls
    url_labels: dict[str, str] = {}
    for entry in extra_urls:
        label = entry.get("label", "")
        if label:
            url_labels[_normalize_url(entry["url"])] = label

    # Opciones de procesado de PDF (tablas + OCR) desde config/default.yaml.
    # Si no existe el fichero o la sección, el parser usa sus valores por defecto
    # (tablas ON, OCR ON en español).
    pdf_options: dict | None = None
    default_cfg_path = Path("config/default.yaml")
    if default_cfg_path.exists():
        try:
            with open(default_cfg_path, encoding="utf-8") as f:
                _dcfg = yaml.safe_load(f) or {}
            pdf_options = (_dcfg.get("ingesta", {}) or {}).get("pdf")
            if pdf_options:
                logger.info("Opciones PDF cargadas de default.yaml: %s", pdf_options)
        except Exception as exc:
            logger.warning("No se pudo leer config/default.yaml (PDF): %s", exc)

    output_path = Path(args.output)
    process_stats = process_documents(
        all_documents,
        output_path,
        min_chars=global_cfg.get("min_chars", 200),
        deduplicate=global_cfg.get("deduplicate", True),
        url_labels=url_labels,
        pdf_options=pdf_options,
    )

    # -- Fase 3: CSV resumen --

    csv_path = export_csv(output_path)

    # -- Fase 4: Informe --

    report_path = Path("data/processed/crawl_report.txt")
    write_report(state, process_stats, report_path)

    sep = "=" * 60
    print(f"\n{sep}")
    print("COMPLETADO!")
    print(f"  Corpus:   {output_path}  ({process_stats.get('processed', 0)} docs)")
    print(f"  CSV:      {csv_path}")
    print(f"  Informe:  {report_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"\nSiguiente paso: python scripts/review_corpus.py")
    print(sep)


if __name__ == "__main__":
    main()
