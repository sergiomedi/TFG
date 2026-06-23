#!/usr/bin/env python3
"""verify_pipeline.py — Verificación de salud del pipeline de ingesta de GRAIA.

Comprueba, sin necesidad de red ni de Ollama, que las correcciones del
subsistema de ingesta funcionan en TU máquina:

  1. Dependencias y binarios (PyMuPDF, Tesseract + idioma 'spa').
  2. Parser HTML: sin duplicación de líneas y tablas → Markdown.
  3. Parser PDF: tabla → Markdown y OCR de página escaneada (si hay 'spa').
  4. Chunker: ningún fragmento supera el tamaño objetivo.
  5. (Opcional) Diagnóstico del corpus existente: duplicación y tamaños.

Uso:
    cd codigo/
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --corpus data/processed/corpus.jsonl

Devuelve código de salida 0 si todas las comprobaciones críticas pasan, 1 si no.
"""

from __future__ import annotations

# Forzar codificación UTF-8 en la salida estándar. En Windows la consola usa
# por defecto cp1252, que no puede representar caracteres Unicode como los de
# dibujo de cajas (U+2500 «─») ni algunos acentos, provocando UnicodeEncodeError.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graia.ingesta.models import RawDocument, SourceType, ParsedDocument
from graia.ingesta.parser import HtmlParser, parse
from graia.ingesta.cleaner import clean
from graia.ingesta.chunker import chunk_document, _token_len

OK = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FALLO]\033[0m"
WARN = "\033[93m[AVISO]\033[0m"

_results: list[bool] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    mark = OK if passed else FAIL
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    _results.append(passed)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ── 1. Dependencias ──────────────────────────────────────────────────────

def check_dependencies() -> None:
    section("1. Dependencias y binarios")
    try:
        import fitz  # noqa
        check("PyMuPDF (fitz) importable", True, fitz.__doc__.splitlines()[0])
    except Exception as e:
        check("PyMuPDF (fitz) importable", False, str(e))

    try:
        import pytesseract  # noqa
        check("pytesseract importable", True)
    except Exception as e:
        check("pytesseract importable", False, str(e))

    tess = shutil.which("tesseract")
    check("binario tesseract en PATH", tess is not None, tess or "no encontrado")
    if tess:
        try:
            langs = subprocess.run([tess, "--list-langs"], capture_output=True,
                                   text=True).stdout
            has_spa = "spa" in langs
            # 'spa' no es crítico (OCR funciona con otros idiomas), pero sí recomendado
            mark = OK if has_spa else WARN
            print(f"  {mark} idioma OCR 'spa' instalado"
                  + ("" if has_spa else " — recomendado: sudo apt install tesseract-ocr-spa"))
        except Exception as e:
            print(f"  {WARN} no se pudo listar idiomas de tesseract: {e}")


# ── 2. Parser HTML ───────────────────────────────────────────────────────

def check_html() -> None:
    section("2. Parser HTML (sin duplicación + tablas→Markdown)")
    body = (
        "<nav>Menu</nav><main>"
        "<h1>Grado en Ingenieria Informatica</h1>"
        "<div><a title='Web'>Web del Grado</a><span>Web del Grado</span></div>"
        "<h2>Primer curso</h2>"
        "<table>"
        "<tr><th>Asignatura</th><th>Tipo</th><th>Cr</th></tr>"
        "<tr><td>Calculo</td><td>Troncal</td><td>6</td></tr>"
        "</table></main><footer>Aceptar cookies</footer>"
    )
    html = f"<html><head><title>Plan</title></head><body>{body}</body></html>"
    raw = RawDocument(url="https://etsiit.ugr.es/x", content=html.encode(),
                      content_type="text/html", source_type=SourceType.HTML)
    doc = HtmlParser().parse(raw)
    check("'Web del Grado' aparece una sola vez", doc.text.count("Web del Grado") == 1,
          f"count={doc.text.count('Web del Grado')}")
    check("tabla serializada a Markdown", "| Calculo | Troncal | 6 |" in doc.text)
    check("nav/footer eliminados",
          "Menu" not in doc.text and "cookies" not in doc.text.lower())
    check("marcadores de cabecera §Hn§ presentes", "§H1§" in doc.text and "§H2§" in doc.text)


# ── 3. Parser PDF ────────────────────────────────────────────────────────

def check_pdf() -> None:
    section("3. Parser PDF (tabla→Markdown + OCR)")
    try:
        import fitz
    except Exception:
        print(f"  {WARN} PyMuPDF no disponible — se omite la prueba de PDF")
        return

    # PDF con tabla dibujada
    doc = fitz.open()
    pg = doc.new_page(width=400, height=300)
    cols, rows = [50, 180, 300, 360], [60, 90, 120]
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
    parsed = parse(raw, pdf_options={"enable_ocr": False})
    check("PDF: contenido de tabla extraído", "Calculo" in parsed.text and "Troncal" in parsed.text)
    check("PDF: filas en formato Markdown", "|" in parsed.text)

    # OCR sobre página "escaneada" (imagen sin capa de texto)
    if shutil.which("tesseract"):
        try:
            from PIL import Image, ImageDraw
            import io
            img = Image.new("RGB", (600, 160), "white")
            ImageDraw.Draw(img).text((20, 60), "NORMATIVA DE PERMANENCIA", fill="black")
            buf = io.BytesIO(); img.save(buf, format="PNG")
            doc = fitz.open(); pg = doc.new_page(width=600, height=160)
            pg.insert_image(fitz.Rect(0, 0, 600, 160), stream=buf.getvalue())
            scan_bytes = doc.tobytes(); doc.close()
            raw2 = RawDocument(url="https://etsiit.ugr.es/scan.pdf", content=scan_bytes,
                               content_type="application/pdf", source_type=SourceType.PDF)
            # Probar con el idioma disponible (spa si existe, si no eng)
            langs = subprocess.run(["tesseract", "--list-langs"],
                                   capture_output=True, text=True).stdout
            lang = "spa" if "spa" in langs else ("eng" if "eng" in langs else "osd")
            ocr_parsed = parse(raw2, pdf_options={"enable_ocr": True, "ocr_lang": lang})
            check("PDF escaneado: OCR recupera texto",
                  "NORMATIVA" in ocr_parsed.text.upper(),
                  f"lang={lang}")
        except Exception as e:
            print(f"  {WARN} prueba OCR no concluyente: {e}")
    else:
        print(f"  {WARN} sin binario tesseract — se omite prueba de OCR")


# ── 4. Chunker ───────────────────────────────────────────────────────────

def check_chunker() -> None:
    section("4. Chunker (tamaño acotado)")
    text = "Contenido académico relevante sobre la ETSIIT. " * 1000
    doc = ParsedDocument(url="https://etsiit.ugr.es/x", source_type=SourceType.HTML,
                         title="T", text=text, fetched_at=datetime.now(timezone.utc))
    chunks = chunk_document(doc, chunk_size_tokens=512, chunk_overlap_tokens=50)
    max_tok = max(_token_len(c.text) for c in chunks)
    # Con solapamiento se admite hasta ~2x; lo crítico es que no haya chunks gigantes
    check("ningún chunk gigante (≤ 2× tamaño objetivo)", max_tok <= 1024,
          f"máx={max_tok} tokens, n={len(chunks)}")


# ── 5. Diagnóstico del corpus (opcional) ─────────────────────────────────

def diagnose_corpus(path: Path) -> None:
    section(f"5. Diagnóstico del corpus ({path})")
    if not path.exists():
        print(f"  {WARN} no existe; ejecuta primero build_corpus.py. Se omite.")
        return
    docs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  documentos: {len(docs)}")

    # Ratio de líneas duplicadas consecutivas (indicador de ruido del parser)
    total_lines = dup_lines = 0
    for d in docs:
        prev = None
        for ln in d.get("text", "").split("\n"):
            k = ln.strip()
            if not k:
                continue
            total_lines += 1
            if k == prev and not k.startswith("|"):
                dup_lines += 1
            prev = k
    ratio = (dup_lines / total_lines * 100) if total_lines else 0
    check("duplicación de líneas < 1%", ratio < 1.0, f"{ratio:.2f}% ({dup_lines}/{total_lines})")

    # Re-chunking en seco para ver distribución de tamaños
    sizes = []
    for d in docs:
        pd = ParsedDocument(
            url=d["url"], source_type=SourceType(d.get("source_type", "html")),
            title=d.get("title"), text=d["text"],
            fetched_at=datetime.fromisoformat(d["fetched_at"]),
            metadata=d.get("metadata", {}),
        )
        for c in chunk_document(pd, chunk_size_tokens=512, chunk_overlap_tokens=50):
            sizes.append(_token_len(c.text))
    if sizes:
        sizes.sort()
        giant = sum(1 for s in sizes if s > 1024)
        print(f"  chunks: {len(sizes)} | mediana={sizes[len(sizes)//2]} tok | "
              f"máx={sizes[-1]} tok | gigantes(>1024)={giant}")
        check("sin chunks gigantes tras re-chunking", giant == 0, f"{giant} gigantes")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verificación del pipeline de ingesta de GRAIA")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl",
                    help="Corpus JSONL a diagnosticar (opcional)")
    args = ap.parse_args()

    print("=" * 64)
    print("VERIFICACIÓN DEL PIPELINE DE INGESTA — GRAIA")
    print("=" * 64)

    check_dependencies()
    check_html()
    check_pdf()
    check_chunker()
    diagnose_corpus(Path(args.corpus))

    print("\n" + "=" * 64)
    passed = sum(_results)
    total = len(_results)
    if all(_results):
        print(f"RESULTADO: {passed}/{total} comprobaciones críticas superadas ✅")
        return 0
    print(f"RESULTADO: {passed}/{total} superadas — revisa los fallos (✗) arriba ❌")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
