#!/usr/bin/env python3
"""inject_doc.py — Inyecta/sustituye un documento verificado en el corpus.

Dos modos de incorporacion:

- Estructurado (--tipo horario|calendario|plan_estudios): datos tabulares
  dificiles de extraer. Un registro por linea; la indexacion lo trocea a nivel
  de registro (1 linea = 1 chunk, self_contained) y la contextualizacion lo deja
  intacto. Estos tres tipos participan en el filtro de coherencia horario<->calendario.

- Normal (sin --tipo): cualquier otro documento verificado (movilidad, normativa,
  etc.). Se incorpora con su categoria y se trocea de forma generica al indexar.
  La categoria es la pieza obligatoria: gobierna el enrutamiento de la recuperacion.

Uso:
    cd codigo/
    python scripts/inject_doc.py --text horario_verificado.txt --tipo horario \
        --url "https://etsiit.ugr.es/.../Horarios.pdf" --title "Horarios GII 2025-2026"

    python scripts/inject_doc.py --text tutores_verificado.txt --category movilidad \
        --url "https://etsiit.ugr.es/.../Tutores.pdf" \
        --title "Tutores de movilidad internacional de la ETSIIT (Erasmus+ y SICUE)"

Tras inyectar, reindexar:  python scripts/index_corpus.py
"""

from __future__ import annotations

import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Inyecta/sustituye un documento verificado en el corpus")
    ap.add_argument("--text", required=True, help="Fichero de texto verificado")
    ap.add_argument("--url", required=True, help="URL original (se conserva para las citas)")
    ap.add_argument("--tipo", default=None,
                    help="Tipo estructurado: horario | calendario | plan_estudios. "
                         "Omitir (o dejar vacio) para un documento verificado NORMAL, "
                         "troceado de forma generica y sin filtro de coherencia.")
    ap.add_argument("--title", default=None, help="Titulo mostrado en las citas")
    ap.add_argument("--category", default=None,
                    help="Categoria tematica del documento (gobierna el enrutamiento). "
                         "Obligatoria, salvo que se indique --tipo (en cuyo caso se usa el tipo).")
    ap.add_argument("--corpus", default="data/processed/corpus.jsonl", help="Ruta del corpus JSONL")
    args = ap.parse_args()

    # El tipo solo aplica a los tres documentos estructurados (troceo por registro
    # + filtro de coherencia). Es OPCIONAL. La categoria es la pieza obligatoria.
    tipo = (args.tipo or "").strip().lower() or None
    valid_tipos = {"horario", "calendario", "plan_estudios"}
    if tipo and tipo not in valid_tipos:
        print(f"ERROR: --tipo invalido '{tipo}'. Usa uno de {sorted(valid_tipos)} o omitelo.")
        return 1
    category = (args.category or "").strip() or tipo
    if not category:
        print("ERROR: indica --category (la categoria tematica, p.ej. 'movilidad').")
        return 1

    text_path = Path(args.text)
    if not text_path.exists():
        print(f"ERROR: no existe el fichero de texto {text_path}")
        return 1
    text = text_path.read_text(encoding="utf-8").strip()
    n_records = sum(1 for ln in text.split("\n") if ln.strip())
    if not text:
        print("ERROR: el fichero de texto esta vacio")
        return 1

    corpus_path = Path(args.corpus)
    title = args.title or f"Documento verificado ({tipo or category})"
    source_type = "pdf" if args.url.lower().endswith(".pdf") else "html"

    records: list[dict] = []
    if corpus_path.exists():
        for ln in corpus_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                records.append(json.loads(ln))

    if corpus_path.exists():
        backup = corpus_path.with_suffix(".jsonl.bak")
        shutil.copy2(corpus_path, backup)
        print(f"Copia de seguridad: {backup}")

    before = len(records)
    records = [r for r in records if r.get("url") != args.url]
    replaced = before - len(records)

    # Con tipo estructurado: troceo por registro (self_contained). Sin tipo:
    # documento verificado normal, troceado de forma generica al indexar.
    metadata = (
        {"tipo": tipo, "verificado": True, "self_contained": True}
        if tipo else {"verificado": True}
    )

    new_record = {
        "url": args.url,
        "title": title,
        "category": category,
        "source_type": source_type,
        "text": text,
        "char_count": len(text),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }
    records.append(new_record)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=" * 64)
    print(f"Documento verificado inyectado en {corpus_path}")
    print(f"  URL (cita):   {args.url}")
    print(f"  tipo:         {tipo or '(normal, troceo generico)'}   categoria: {category}")
    print(f"  registros:    {n_records} lineas")
    print(f"  sustituido:   {'si (' + str(replaced) + ' doc previo eliminado)' if replaced else 'no (documento nuevo)'}")
    print(f"  total corpus: {len(records)} documentos")
    print("=" * 64)
    print("Siguiente paso:  python scripts/index_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
