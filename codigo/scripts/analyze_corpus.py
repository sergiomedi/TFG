#!/usr/bin/env python3
"""Análisis de calidad del corpus generado por build_corpus.py.

Ejecutar desde la raíz del proyecto:
    python codigo/scripts/analyze_corpus.py
"""


# Forzar codificación UTF-8 en la salida estándar. En Windows la consola usa
# por defecto cp1252, que no puede representar caracteres Unicode como los de
# dibujo de cajas (U+2500 «─») ni algunos acentos, provocando UnicodeEncodeError.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
JSONL = DATA_DIR / "corpus.jsonl"
CSV_FILE = DATA_DIR / "corpus.csv"

# ── 1. Cargar JSONL ──────────────────────────────────────────────────
print("=" * 70)
print("ANÁLISIS DEL CORPUS — GRAIA")
print("=" * 70)

docs = []
with open(JSONL, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            docs.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  ⚠ Línea {i} JSON inválido: {e}")

print(f"\nTotal documentos en JSONL: {len(docs)}")

# ── 2. Distribución por categoría ────────────────────────────────────
cat_counts = Counter(d.get("category", "SIN_CATEGORÍA") for d in docs)
print(f"\nCategorías encontradas: {len(cat_counts)}")
print("-" * 40)
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat:20s} → {count:4d} docs")

expected = {
    "calendario", "estudiantes", "guias_docentes", "movilidad",
    "normativa", "profesorado", "tfg", "tramites"
}
missing = expected - set(cat_counts.keys())
if missing:
    print(f"\n  ❌ CATEGORÍAS FALTANTES: {missing}")
else:
    print(f"\n  ✅ Todas las 8 categorías presentes")

# ── 3. Títulos de calendario ─────────────────────────────────────────
print("\n" + "=" * 70)
print("TÍTULOS DE CALENDARIO")
print("-" * 70)
for d in docs:
    if d.get("category") == "calendario":
        title = d.get("title", "(sin título)")
        url = d.get("url", "")[-60:]
        print(f"  • {title[:70]}")
        if "blanco" in title.lower():
            print(f"    ❌ TÍTULO SOSPECHOSO — debería usar label configurado")

# ── 4. Verificar extra_urls esperadas ────────────────────────────────
print("\n" + "=" * 70)
print("EXTRA_URLS ESPERADAS")
print("-" * 70)
extra_urls_esperadas = [
    # Normativa UGR (secretariageneral)
    ("BOUGR/197/CABECERAS", "Normativa TFG UGR — BOUGR 197"),
    ("260_8.pdf", "Reglamento permanencia UGR — BOUGR 260"),
    # Normativa UGR (www.ugr.es)
    ("texto-consolidado-normativa-evaluacion", "Normativa evaluación UGR"),
    ("ncg1421-normativa-programas-conjuntos", "Normativa dobles grados UGR"),
    ("acg1436-procedimiento-para-el-reconocimiento", "Normativa acreditación lingüística"),
    # Normativa ETSIIT
    ("Directrices%20TFG", "Directrices TFG ETSIIT"),
    # Plan de estudios
    ("informatica/docencia/plan-estudios", "Plan de estudios GII"),
    ("informatica/informacion/presentacion", "Presentación del Grado"),
    # Calendarios
    ("Calendario%20Academico%202025-2026", "Calendario académico ETSIIT"),
    ("Calendario%20TFG%202025-2026", "Calendario TFG ETSIIT"),
    # Horarios
    ("Horarios%20GII", "Horarios GII 25-26"),
    # Plazos y trámites
    ("PLAZOS%20DE%20INTERES", "Plazos de interés curso 25-26"),
    ("ADMISI", "Impreso admisión a estudios de grado"),
]
corpus_urls = {d.get("url", "") for d in docs}
for fragment, label in extra_urls_esperadas:
    matching = [u for u in corpus_urls if fragment in u]
    if matching:
        print(f"  ✅ — {label}")
    else:
        print(f"  ❌ NO ENCONTRADA — {label}")
        print(f"       fragmento buscado: {fragment}")

# ── 5. Comprobar ruido de secretaría general ─────────────────────────
print("\n" + "=" * 70)
print("RUIDO SECRETARÍA GENERAL")
print("-" * 70)
noise_keywords = [
    "protección de datos", "proteccion de datos", "protecciondedatos",
    "cláusulas informativas", "clausulas informativas",
    "información adicional sobre", "informacion adicional sobre",
    "comunicación de brechas", "comunicacion de brechas",
    "ejercicio de derechos", "infografía", "infografias",
    "recomendaciones e infografías",
]
noise_docs = []
for d in docs:
    title_lower = d.get("title", "").lower()
    url_lower = d.get("url", "").lower()
    text_start = d.get("text", "")[:300].lower()
    for kw in noise_keywords:
        if kw in title_lower or kw in url_lower or kw in text_start:
            noise_docs.append(d)
            break

if noise_docs:
    print(f"  ❌ {len(noise_docs)} documentos de ruido detectados:")
    for d in noise_docs[:10]:
        print(f"     • {d.get('title', '?')[:60]}")
        print(f"       URL: {d.get('url', '?')}")
        print(f"       cat: {d.get('category', '?')}")
else:
    print(f"  ✅ Sin ruido de secretaría general")

# ── 6. Contenido de categorías nuevas ────────────────────────────────
for cat in ["tfg", "estudiantes", "profesorado"]:
    print(f"\n{'=' * 70}")
    print(f"CATEGORÍA: {cat.upper()}")
    print("-" * 70)
    cat_docs = [d for d in docs if d.get("category") == cat]
    if not cat_docs:
        print(f"  ❌ SIN DOCUMENTOS")
        continue
    for d in cat_docs:
        title = d.get("title", "(sin título)")[:65]
        text_len = len(d.get("text", ""))
        url = d.get("url", "")[-55:]
        print(f"  • [{text_len:6d} chars] {title}")
        print(f"    {url}")

# ── 7. Calidad general ──────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("MÉTRICAS DE CALIDAD")
print("-" * 70)

text_lengths = [len(d.get("text", "")) for d in docs]
avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
short_docs = [d for d in docs if len(d.get("text", "")) < 100]
empty_titles = [d for d in docs if not d.get("title", "").strip()]

print(f"  Longitud media de texto:  {avg_len:,.0f} chars")
print(f"  Documento más corto:      {min(text_lengths):,d} chars")
print(f"  Documento más largo:      {max(text_lengths):,d} chars")
print(f"  Docs con < 100 chars:     {len(short_docs)}")
print(f"  Docs sin título:          {len(empty_titles)}")

if short_docs:
    print(f"\n  Documentos muy cortos:")
    for d in short_docs[:5]:
        print(f"    • [{len(d.get('text','')):4d}] {d.get('title','?')[:50]} ({d.get('category','')})")

# ── 8. Duplicados por contenido ──────────────────────────────────────
print(f"\n{'=' * 70}")
print("DUPLICADOS")
print("-" * 70)
seen_hashes = defaultdict(list)
for d in docs:
    h = d.get("content_hash", "")
    if h:
        seen_hashes[h].append(d.get("title", "?")[:40])

dupes = {h: titles for h, titles in seen_hashes.items() if len(titles) > 1}
if dupes:
    print(f"  ⚠ {len(dupes)} grupos de duplicados por hash:")
    for h, titles in list(dupes.items())[:5]:
        print(f"    hash={h[:12]}… → {titles}")
else:
    print(f"  ✅ Sin duplicados por content_hash")

# ── 9. Verificar CSV consistencia ────────────────────────────────────
print(f"\n{'=' * 70}")
print("CONSISTENCIA CSV vs JSONL")
print("-" * 70)
try:
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    csv_cats = Counter(r.get("category", "") for r in csv_rows)
    print(f"  CSV total:  {len(csv_rows)} filas")
    print(f"  JSONL total: {len(docs)} docs")
    if len(csv_rows) == len(docs):
        print(f"  ✅ Coinciden")
    else:
        print(f"  ❌ DIFERENCIA de {abs(len(csv_rows) - len(docs))} registros")
except Exception as e:
    print(f"  ⚠ No se pudo leer CSV: {e}")

print(f"\n{'=' * 70}")
print("FIN DEL ANÁLISIS")
print("=" * 70)
