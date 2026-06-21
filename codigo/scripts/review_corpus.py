#!/usr/bin/env python3
"""review_corpus.py — Herramienta de revisión y limpieza del corpus de GRAIA.

Lee el fichero data/processed/corpus.jsonl generado por build_corpus.py
y presenta un informe interactivo para que el usuario valide el contenido
antes de proceder a la indexación.

Modos de consulta:
    python scripts/review_corpus.py                  # informe resumen
    python scripts/review_corpus.py --full            # muestra preview de cada doc
    python scripts/review_corpus.py --category normativa  # filtra por categoría
    python scripts/review_corpus.py --search "TFG"   # busca en títulos y texto
    python scripts/review_corpus.py --export-csv      # exporta a CSV para revisar en Excel

Modos de limpieza:
    python scripts/review_corpus.py --remove-url "https://example.com/page"
    python scripts/review_corpus.py --remove-min-chars 100
    python scripts/review_corpus.py --remove-search "2023"         # busca en título + URL decodificada + texto
    python scripts/review_corpus.py --remove-search "Febrero 2022" --dry-run

Modo de inserción manual:
    python scripts/review_corpus.py --add-url "https://example.com/doc.pdf" --add-category normativa
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

# Añadir el directorio raíz del proyecto al path (para imports de graia)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_corpus(path: Path) -> list[dict]:
    """Carga el corpus JSONL en memoria."""
    if not path.exists():
        print(f"ERROR: No se encontró {path}")
        print("       Ejecuta primero: python scripts/build_corpus.py")
        sys.exit(1)

    docs = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARN: Error JSON en línea {line_num}: {e}")
    return docs


def print_summary(docs: list[dict]) -> None:
    """Imprime resumen estadístico del corpus."""
    total_chars = sum(d.get("char_count", 0) for d in docs)
    cats = defaultdict(list)
    types = defaultdict(int)

    for d in docs:
        cats[d.get("category", "sin_categoria")].append(d)
        types[d.get("source_type", "?")] += 1

    print("=" * 65)
    print("  RESUMEN DEL CORPUS DE GRAIA")
    print("=" * 65)
    print(f"  Documentos totales:     {len(docs)}")
    print(f"  Caracteres totales:     {total_chars:,}")
    print(f"  Media por documento:    {total_chars // max(len(docs), 1):,} chars")
    print(f"  Tipos:                  {dict(types)}")
    print()
    print("  ── POR CATEGORÍA ──")
    print(f"  {'Categoría':<25s} {'Docs':>5s} {'Chars':>10s} {'Media':>8s}")
    print(f"  {'-'*25} {'-'*5} {'-'*10} {'-'*8}")

    for cat in sorted(cats.keys()):
        cat_docs = cats[cat]
        cat_chars = sum(d.get("char_count", 0) for d in cat_docs)
        avg = cat_chars // max(len(cat_docs), 1)
        print(f"  {cat:<25s} {len(cat_docs):>5d} {cat_chars:>10,} {avg:>8,}")

    print()

    # Documentos más largos y más cortos
    sorted_by_size = sorted(docs, key=lambda d: d.get("char_count", 0))
    print("  ── TOP 5 MÁS LARGOS ──")
    for d in sorted_by_size[-5:]:
        title = (d.get("title") or d.get("url", "?"))[:60]
        print(f"  {d.get('char_count', 0):>8,} chars | {title}")

    print()
    print("  ── TOP 5 MÁS CORTOS ──")
    for d in sorted_by_size[:5]:
        title = (d.get("title") or d.get("url", "?"))[:60]
        print(f"  {d.get('char_count', 0):>8,} chars | {title}")

    print()


def print_full(docs: list[dict], max_preview: int = 300) -> None:
    """Muestra un preview del contenido de cada documento."""
    for i, d in enumerate(docs, 1):
        title = d.get("title") or "(sin título)"
        url = d.get("url", "?")
        cat = d.get("category", "?")
        chars = d.get("char_count", 0)
        text = d.get("text", "")
        preview = text[:max_preview].replace("\n", " ") + ("..." if len(text) > max_preview else "")

        print(f"─── [{i}/{len(docs)}] {cat.upper()} ───")
        print(f"  Título:  {title}")
        print(f"  URL:     {url}")
        print(f"  Chars:   {chars:,}")
        print(f"  Preview: {preview}")
        print()


def _doc_matches(d: dict, query: str) -> bool:
    """Comprueba si un documento contiene la query en título, URL decodificada o texto."""
    q = query.lower()
    title = (d.get("title") or "").lower()
    url_decoded = unquote(d.get("url", "")).lower()
    text = (d.get("text") or "").lower()
    return q in title or q in url_decoded or q in text


def search_docs(docs: list[dict], query: str) -> list[dict]:
    """Filtra documentos que contengan la query en título, URL decodificada o texto."""
    return [d for d in docs if _doc_matches(d, query)]


def remove_docs(
    corpus_path: Path,
    url: str | None = None,
    min_chars: int | None = None,
    search: str | None = None,
    dry_run: bool = False,
) -> None:
    """Elimina documentos del corpus por URL exacta, tamaño mínimo o búsqueda.

    La búsqueda (--remove-search) opera sobre título, URL decodificada y texto,
    lo que permite encontrar PDFs sin título cuya URL contiene el término.

    Crea una copia de seguridad (.bak) antes de modificar el fichero original.
    Con --dry-run solo muestra lo que se borraría sin tocar el fichero.
    """
    docs = load_corpus(corpus_path)
    original_count = len(docs)

    # Identificar documentos a eliminar
    to_remove: list[dict] = []
    for d in docs:
        if url and d.get("url", "") == url:
            to_remove.append(d)
        if min_chars is not None and d.get("char_count", 0) <= min_chars:
            to_remove.append(d)
        if search and _doc_matches(d, search):
            to_remove.append(d)

    # Deduplicar (un doc podría cumplir ambos criterios)
    ids_to_remove = {id(d) for d in to_remove}
    to_remove = [d for d in docs if id(d) in ids_to_remove]

    if not to_remove:
        print("No se encontraron documentos que coincidan con los criterios.")
        return

    # Mostrar lo que se va a eliminar
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Documentos a eliminar: {len(to_remove)}")
    print("-" * 65)
    for d in to_remove:
        title = (d.get("title") or "(sin título)")[:55]
        chars = d.get("char_count", 0)
        doc_url = d.get("url", "?")
        print(f"  {chars:>6,} chars | {title}")
        print(f"              | {doc_url}")
    print("-" * 65)

    if dry_run:
        print(f"[DRY RUN] Se eliminarían {len(to_remove)} de {original_count} documentos.")
        print("          Ejecuta sin --dry-run para aplicar los cambios.")
        return

    # Copia de seguridad
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = corpus_path.with_suffix(f".bak_{timestamp}.jsonl")
    shutil.copy2(corpus_path, backup_path)
    print(f"\n  Backup creado: {backup_path.name}")

    # Escribir corpus filtrado
    kept = [d for d in docs if id(d) not in ids_to_remove]
    with open(corpus_path, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"  Eliminados: {len(to_remove)} documentos")
    print(f"  Corpus actualizado: {original_count} → {len(kept)} documentos\n")


def find_old_exam_docs(docs: list[dict]) -> list[dict]:
    """Identifica calendarios de exámenes y asignaciones de aulas de cursos anteriores.

    Detecta documentos cuya URL decodificada contiene patrones de calendarios
    y asignaciones de aulas, y excluye los del curso actual (25-26 / 2025-2026).
    """
    exam_patterns = ["Asignación Aulas", "CalendarioExamenes",
                     "Calendario Academico", "Calendario TFG"]
    current_patterns = ["25-26", "2025-2026"]

    old = []
    for d in docs:
        url_dec = unquote(d.get("url", ""))
        if any(p in url_dec for p in exam_patterns):
            if not any(p in url_dec for p in current_patterns):
                old.append(d)
    return old


def export_csv(docs: list[dict], output: Path) -> None:
    """Exporta el corpus a CSV para revisión en Excel/Sheets."""
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Categoría", "Tipo", "Título", "URL", "Chars", "Preview (200 chars)"])
        for i, d in enumerate(docs, 1):
            preview = (d.get("text") or "")[:200].replace("\n", " ")
            writer.writerow([
                i,
                d.get("category", "?"),
                d.get("source_type", "?"),
                d.get("title") or "(sin título)",
                d.get("url", "?"),
                d.get("char_count", 0),
                preview,
            ])
    print(f"Exportado a: {output}")


def add_url_to_corpus(corpus_path: Path, url: str, category: str) -> None:
    """Descarga una URL, la procesa y la añade al corpus existente.

    Útil para insertar documentos individuales sin relanzar el crawl completo.
    """
    from graia.ingesta.fetcher import fetch
    from graia.ingesta.parser import parse
    from graia.ingesta.cleaner import clean

    print(f"\n  Descargando: {url}")
    raw = fetch(url, user_agent="GRAIA-academic-crawler/0.1 (+sergiomedinam98@gmail.com)")
    if raw is None:
        print(f"  ERROR: No se pudo descargar {url}")
        return

    try:
        parsed = parse(raw)
    except Exception as e:
        print(f"  ERROR: No se pudo parsear: {e}")
        return

    cleaned = clean(parsed)
    if cleaned is None:
        print(f"  ERROR: Contenido vacío o demasiado corto tras limpieza")
        return

    text = cleaned.text
    char_count = len(text)

    # Verificar que no existe ya
    existing = load_corpus(corpus_path)
    for d in existing:
        if d.get("url") == url:
            print(f"  AVISO: URL ya existe en el corpus ({d.get('char_count', 0)} chars)")
            return

    record = {
        "url": cleaned.url,
        "title": cleaned.title or "",
        "category": category,
        "source_type": cleaned.source_type.value,
        "text": text,
        "char_count": char_count,
        "fetched_at": cleaned.fetched_at.isoformat(),
        "metadata": cleaned.metadata,
    }

    with open(corpus_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  ✓ Añadido: {cleaned.title or '(sin título)'} ({char_count:,} chars, cat={category})")
    print(f"  Corpus actualizado: {len(existing) + 1} documentos\n")


def main():
    parser = argparse.ArgumentParser(description="Revisión del corpus de GRAIA")
    parser.add_argument("--input", default="data/processed/corpus.jsonl",
                        help="Ruta al corpus JSONL")
    parser.add_argument("--full", action="store_true",
                        help="Mostrar preview de cada documento")
    parser.add_argument("--category", type=str, default=None,
                        help="Filtrar por categoría (normativa, guias_docentes, etc.)")
    parser.add_argument("--search", type=str, default=None,
                        help="Buscar en títulos y texto")
    parser.add_argument("--export-csv", action="store_true",
                        help="Exportar a CSV para revisar en Excel")
    parser.add_argument("--remove-url", type=str, default=None,
                        help="Eliminar el documento con esta URL exacta")
    parser.add_argument("--remove-min-chars", type=int, default=None,
                        help="Eliminar documentos con chars <= este valor")
    parser.add_argument("--remove-search", type=str, default=None,
                        help="Eliminar documentos que contengan este texto en título, URL decodificada o texto")
    parser.add_argument("--remove-old-exams", action="store_true",
                        help="Eliminar calendarios de exámenes y asignaciones de aulas de cursos anteriores al 25-26")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar qué se eliminaría sin modificar el fichero")
    parser.add_argument("--add-url", type=str, default=None,
                        help="Descargar una URL y añadirla al corpus manualmente")
    parser.add_argument("--add-category", type=str, default="extra",
                        help="Categoría para --add-url (default: extra)")
    args = parser.parse_args()

    corpus_path = Path(args.input)

    # Modo inserción manual
    if args.add_url:
        add_url_to_corpus(corpus_path, args.add_url, args.add_category)
        return

    # Modo limpieza especial: calendarios antiguos
    if args.remove_old_exams:
        all_docs = load_corpus(corpus_path)
        old_exams = find_old_exam_docs(all_docs)
        if not old_exams:
            print("No se encontraron calendarios/exámenes de cursos anteriores.")
            return

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Calendarios/exámenes antiguos: {len(old_exams)}")
        print("-" * 65)
        for d in old_exams:
            url_short = unquote(d.get("url", ""))[-75:]
            chars = d.get("char_count", 0)
            print(f"  {chars:>6,} chars | ...{url_short}")
        print("-" * 65)

        if args.dry_run:
            print(f"[DRY RUN] Se eliminarían {len(old_exams)} de {len(all_docs)} documentos.")
            return

        # Backup + escritura
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = corpus_path.with_suffix(f".bak_{timestamp}.jsonl")
        shutil.copy2(corpus_path, backup_path)
        print(f"\n  Backup creado: {backup_path.name}")

        ids_to_remove = {id(d) for d in old_exams}
        kept = [d for d in all_docs if id(d) not in ids_to_remove]
        with open(corpus_path, "w", encoding="utf-8") as f:
            for d in kept:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"  Eliminados: {len(old_exams)} calendarios/exámenes antiguos")
        print(f"  Corpus actualizado: {len(all_docs)} → {len(kept)} documentos\n")
        return

    # Modo limpieza genérica: ejecutar y salir
    if args.remove_url or args.remove_min_chars is not None or args.remove_search:
        remove_docs(
            corpus_path,
            url=args.remove_url,
            min_chars=args.remove_min_chars,
            search=args.remove_search,
            dry_run=args.dry_run,
        )
        return

    # Modo consulta
    docs = load_corpus(corpus_path)

    if args.category:
        docs = [d for d in docs if d.get("category") == args.category]
        print(f"Filtrado por categoría '{args.category}': {len(docs)} docs\n")

    if args.search:
        docs = search_docs(docs, args.search)
        print(f"Búsqueda '{args.search}': {len(docs)} resultados\n")

    print_summary(docs)

    if args.full:
        print_full(docs)

    if args.export_csv:
        csv_path = Path(args.input).with_suffix(".csv")
        export_csv(docs, csv_path)

    print("Siguiente paso (si el corpus es correcto):")
    print("  python scripts/index_corpus.py")


if __name__ == "__main__":
    main()
