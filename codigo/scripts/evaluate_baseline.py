#!/usr/bin/env python3
"""evaluate_baseline.py — Baseline SIN recuperación (LLM solo) para la hipótesis H2.

Ejecuta el MISMO LLM sobre el MISMO conjunto de preguntas que ``evaluate.py``,
pero **sin** recuperar contexto del corpus: el modelo responde únicamente con su
conocimiento paramétrico. Permite contrastar el sistema RAG frente al LLM aislado
y cuantificar la reducción de alucinaciones (H2) y la mejora en abstención.

Para que la comparación sea justa, se reutiliza el mismo dataset, las mismas
palabras clave y la misma función de puntuación que en ``evaluate.py``; lo único
que cambia es que el prompt NO contiene fragmentos del corpus y NO se fuerza la
abstención (el modelo es libre de responder o no).

Uso:
    cd codigo/
    python scripts/evaluate_baseline.py --model llama3.1:8b-instruct-q4_K_M
    python scripts/evaluate_baseline.py --model qwen2.5:14b-instruct
    python scripts/evaluate_baseline.py --model llama3.1:8b-instruct-q4_K_M --show

Genera eval/results_baseline_<modelo>.json con el mismo formato de resumen que
evaluate.py, de modo que sus métricas son directamente comparables.

Requiere Ollama en ejecución (no necesita el índice).
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
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re

import yaml

from graia.generacion.ollama_client import OllamaClient
from graia.generacion.postprocess import clean_answer

# Reutilizamos la lógica de puntuación EXACTA de evaluate.py para que las
# métricas del baseline y del sistema RAG sean comparables sin sesgos.
import importlib.util

_eval_path = Path(__file__).resolve().parent / "evaluate.py"
_spec = importlib.util.spec_from_file_location("graia_evaluate", _eval_path)
_evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate)
score_item = _evaluate.score_item
load_dataset = _evaluate.load_dataset

# Prompt de sistema NEUTRO: el modelo sabe que es un asistente de la ETSIIT,
# pero NO recibe contexto del corpus ni instrucción de abstenerse. Así medimos
# su comportamiento "tal cual", incluida su tendencia a inventar.
_BASELINE_SYSTEM = (
    "Eres un asistente académico de la ETSIIT (Escuela Técnica Superior de "
    "Ingenierías Informática y de Telecomunicación) de la Universidad de Granada. "
    "Responde a la pregunta del estudiante de la forma más precisa y útil que "
    "puedas, en español."
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Baseline sin RAG (H2) para GRAIA")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default="eval/eval_dataset.jsonl")
    ap.add_argument("--model", default=None, help="Sobrescribe generacion.model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    gen = cfg["generacion"]
    model = args.model or gen["model"]

    print(f"[BASELINE sin RAG] Cargando modelo de generación: {model} ...")
    client = OllamaClient(model=model, temperature=gen["temperature"], top_p=gen["top_p"],
                          max_tokens=gen["max_tokens"], stop=gen.get("stop", []))

    dataset = load_dataset(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]

    results = []
    gen_ms_all = []
    for item in dataset:
        t1 = time.perf_counter()
        answer = clean_answer(client.generate(_BASELINE_SYSTEM, item["query"]).text)
        gen_ms = (time.perf_counter() - t1) * 1000
        gen_ms_all.append(gen_ms)

        # Sin recuperación no hay fuentes: retrieval_hit no aplica.
        r = score_item(item, answer, sources=[])
        r["generation_ms"] = round(gen_ms)
        r["question"] = item["query"]
        r["answer"] = answer
        results.append(r)

        mark = "OK " if r["answer_correct"] else "XX "
        print(f"  {mark} [{r['id']}] correcto={r['answer_correct']} "
              f"rehusó={r['refused']} fuga={r['leaked']} kw={r['kw_coverage']} "
              f"gen={r['generation_ms']}ms")
        if args.show:
            print("      Q:", item["query"])
            print("      A:", answer.replace("\n", " ")[:300])

    n = len(results)
    factual = [r for r in results if r["type"] != "no_info"]
    noinfo = [r for r in results if r["type"] == "no_info"]

    def pct(x, total):
        return f"{100*x/total:.0f}%" if total else "-"

    summary = {
        "model": model,
        "mode": "baseline_sin_rag",
        "n": n,
        "answer_accuracy": pct(sum(r["answer_correct"] for r in results), n),
        "factual_accuracy": pct(sum(r["answer_correct"] for r in factual), len(factual)),
        "noinfo_correct": pct(sum(r["answer_correct"] for r in noinfo), len(noinfo)),
        "false_refusals": sum(1 for r in factual if r["refused"]),
        "hallucination_noinfo": pct(sum(1 for r in noinfo if not r["refused"]), len(noinfo)),
        "leaks": sum(1 for r in results if r["leaked"]),
        "avg_generation_ms": round(sum(gen_ms_all) / n) if n else 0,
    }

    print("\n" + "=" * 64)
    print(f"RESUMEN BASELINE (sin RAG) — modelo: {model}")
    print("=" * 64)
    for k, v in summary.items():
        if k != "model":
            print(f"  {k:22s}: {v}")

    out = Path(args.out) if args.out else Path("eval") / f"results_baseline_{re.sub(r'[^a-zA-Z0-9]+','_',model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInforme guardado en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
