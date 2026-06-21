#!/usr/bin/env python3
"""inject_doc.py — Inyecta/sustituye un documento verificado en el corpus.

Permite reemplazar en ``corpus.jsonl`` el documento asociado a una URL por el
contenido de un fichero de texto **verificado a mano** (un registro autocontenido
por línea), conservando la URL original para las **citas**. Pensado para los
documentos más difíciles de extraer de forma fiable (horarios, calendario):
en vez de depender de un parser frágil, se usa una fuente verificada como única
fuente de verdad, y el sistema sigue citando el PDF oficial.

El documento queda marcado con ``metadata.tipo`` (horario/calendario) y
``self_contained=True``, de modo que la indexación lo trocea a nivel de
registro (1 línea = 1 chunk) y la contextualización LLM lo deja intacto.

Uso:
    cd codigo/
    python scripts/inject_doc.py --text horario_verificado.txt --tipo horario \
        --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/Horarios%20GII%20%2825-26%29.pdf" \
        --title "Horarios Grado en Ingeniería Informática 2025-2026"

    python scripts/inject_doc.py --text calendario_verificado.txt --tipo calendario \
        --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/Calendario%20Academico.pdf" \
        --title "Calendario académico y de TFG 2025-2026"

Tras inyectar, reindexar:  python scripts/index_corpus.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Inyecta/sustituye un documento verificado en el corpus")
    ap.add_argument("--text", required=True, help="Fichero de texto verificado (1 registro por línea)")
    ap.add_argument("--url", required=True, help="URL original (se conserva para las citas)")
    ap.add_argument("--tipo", default="horario",
                    choices=["horario", "calendario", "plan_estudios"],
                    help="Tipo de documento estructurado")
    ap.add_argument("--title", default=None, help="Título mostrado en las citas")
    ap.add_argument("--category", default=None, help="Categoría (por defecto, igual que tipo)")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl", help="Ruta del corpus JSONL")
    args = ap.parse_args()

    text_path = Path(args.text)
    if not text_path.exists():
        print(f"ERROR: no existe el fichero de texto {text_path}")
        return 1
    text = text_path.read_text(encoding="utf-8").strip()
    n_records = sum(1 for ln in text.split("\n") if ln.strip())
    if not text:
        print("ERROR: el fichero de texto está vacío")
        return 1

    corpus_path = Path(args.corpus)
    category = args.category or args.tipo
    title = args.title or f"Documento verificado ({args.tipo})"
    source_type = "pdf" if args.url.lower().endswith(".pdf") else "html"

    # Cargar corpus existente (si lo hay)
    records: list[dict] = []
    if corpus_path.exists():
        for ln in corpus_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                records.append(json.loads(ln))

    # Copia de seguridad antes de modificar
    if corpus_path.exists():
        backup = corpus_path.with_suffix(".jsonl.bak")
        shutil.copy2(corpus_path, backup)
        print(f"Copia de seguridad: {backup}")

    # Eliminar cualquier documento existente con la misma URL
    before = len(records)
    records = [r for r in records if r.get("url") != args.url]
    replaced = before - len(records)

    # Añadir el documento verificado
    new_record = {
        "url": args.url,
        "title": title,
        "category": category,
        "source_type": source_type,
        "text": text,
        "char_count": len(text),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {"tipo": args.tipo, "verificado": True, "self_contained": True},
    }
    records.append(new_record)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 64)
    print(f"Documento verificado inyectado en {corpus_path}")
    print(f"  URL (cita):   {args.url}")
    print(f"  tipo:         {args.tipo}   categoría: {category}")
    print(f"  registros:    {n_records} líneas")
    print(f"  sustituido:   {'sí (' + str(replaced) + ' doc previo eliminado)' if replaced else 'no (documento nuevo)'}")
    print(f"  total corpus: {len(records)} documentos")
    print("=" * 64)
    print("Siguiente paso:  python scripts/index_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
