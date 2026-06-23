#!/usr/bin/env python3
"""evaluate.py — Evaluación automática del sistema RAG GRAIA.

Reproduce el pipeline de :mod:`graia.interfaz.app` (recuperación → generación)
sin Streamlit, sobre un conjunto de preguntas con respuesta/fuente esperada
(``eval/eval_dataset.jsonl``), y calcula métricas objetivas para poder comparar
configuraciones (modelo de generación, prompt, k) sin evaluar "a ojo".

Métricas:
  - Recall de recuperación: ¿aparece la fuente esperada entre los fragmentos
    recuperados? (solo en preguntas con fuente esperada)
  - Acierto de respuesta: en factual, ¿contiene la respuesta TODAS las palabras
    clave esperadas sin haber rehusado? En no_info, ¿ha rehusado correctamente?
  - Tasa de rechazo correcto/incorrecto (rehusar cuando había información = mal).
  - Fuga de razonamiento (preámbulos tipo "la pregunta del usuario es…").
  - Latencia de recuperación y generación.

Uso:
    cd codigo/
    python scripts/evaluate.py
    python scripts/evaluate.py --model qwen2.5:14b-instruct     # comparar modelo
    python scripts/evaluate.py --model llama3.1:8b-instruct-q4_K_M --out eval/res_llama.json
    python scripts/evaluate.py --limit 5 --show                 # depuración rápida

Requiere índice construido (index_corpus.py) y Ollama en ejecución.
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
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from graia.generacion.citation_validator import recover_sources, validate_citations
from graia.generacion.ollama_client import OllamaClient
from graia.generacion.postprocess import clean_answer, is_abstention
from graia.generacion.prompt_builder import build_messages, get_source_map
from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.query_router import expand_abbreviations
from graia.recuperacion.retriever import retrieve
from graia.recuperacion.scope_classifier import OUT_OF_SCOPE_ANSWER, is_in_scope

# Frases que delatan que el modelo expone su razonamiento o se va por las ramas.
_LEAK_PATTERNS = [
    r"la pregunta del usuario",
    r"tras revisar",
    r"después de revisar",
    r"necesito buscar",
    r"voy a buscar",
    r"en los fragmentos proporcionados",
    r"puedo sugerir",
    r"algunas opciones",
    r"pasos? para",
]
# Detección de abstención: reconoce la frase canónica Y las variantes semánticas
# equivalentes que el modelo produce con otras palabras ("no tengo información",
# "no hay información disponible", "no se menciona", etc.), para que la métrica no
# penalice abstenciones correctas redactadas de forma distinta.
_REFUSAL_RE = re.compile(
    r"no\s+(?:dispongo\s+de|tengo(?:\s+acceso\s+a)?|hay|cuento\s+con|dispone\s+de)"
    r"\s+(?:suficiente\s+)?(?:informaci[oó]n|datos)"
    r"|no\s+se\s+menciona"
    r"|sin\s+informaci[oó]n\s+(?:suficiente|disponible)",
    re.IGNORECASE,
)


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def score_item(item: dict, answer: str, sources: list[str]) -> dict:
    ans = answer.lower()
    # Rechazo = la respuesta ES (empieza siendo) una abstención de "no hay dato".
    # Se usa el detector compartido con el postprocesado (fuente única de verdad),
    # que exige que la abstención sea el INICIO de la respuesta: así no se marca
    # como rechazo una respuesta válida que de pasada diga "no se menciona X".
    refused = is_abstention(answer)
    leaked = any(re.search(p, ans) for p in _LEAK_PATTERNS)

    # Recuperación: ¿alguna fuente esperada aparece en las URLs recuperadas?
    exp_sources = item.get("expected_sources", [])
    if exp_sources:
        retrieval_hit = any(
            any(es.lower() in (s or "").lower() for s in sources) for es in exp_sources
        )
    else:
        retrieval_hit = None  # no aplica

    # Respuesta
    kws = [k.lower() for k in item.get("expected_keywords", [])]
    kw_hits = sum(1 for k in kws if k in ans)
    if item.get("should_refuse"):
        answer_correct = refused
    elif kws:
        answer_correct = (not refused) and kw_hits == len(kws)
    else:
        # Sin keywords: se considera correcto si responde algo (no rehúsa)
        answer_correct = not refused

    return {
        "id": item["id"],
        "type": item.get("type", "factual"),
        "refused": refused,
        "leaked": leaked,
        "retrieval_hit": retrieval_hit,
        "kw_coverage": f"{kw_hits}/{len(kws)}" if kws else "-",
        "answer_correct": answer_correct,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluación automática de GRAIA")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--dataset", default="eval/eval_dataset.jsonl")
    ap.add_argument("--model", default=None, help="Sobrescribe generacion.model")
    ap.add_argument("--limit", type=int, default=0, help="Evaluar solo N preguntas")
    ap.add_argument("--show", action="store_true", help="Muestra cada respuesta")
    ap.add_argument("--out", default=None, help="Ruta del informe JSON de salida")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rec = cfg["recuperacion"]
    gen = cfg["generacion"]
    model = args.model or gen["model"]

    print(f"Cargando índice y modelos (generación: {model}) ...")
    emb = cfg["embeddings"]
    embedder = Embedder(model_name=emb["model_name"], query_prefix=emb["query_prefix"],
                        passage_prefix=emb["passage_prefix"], batch_size=emb["batch_size"],
                        normalize=emb["normalize"])
    store = VectorStore.load(cfg["paths"]["index"])
    bm25_path = Path(cfg["paths"]["index"]) / "bm25_index.pkl"
    bm25 = BM25Index.load(cfg["paths"]["index"]) if bm25_path.exists() else None
    client = OllamaClient(model=model, temperature=gen["temperature"], top_p=gen["top_p"],
                          max_tokens=gen["max_tokens"], stop=gen.get("stop", []))

    dataset = load_dataset(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]

    results = []
    ret_ms_all, gen_ms_all = [], []
    for item in dataset:
        q = expand_abbreviations(item["query"])
        t0 = time.perf_counter()
        chunks = retrieve(
            q, embedder, store, bm25_index=bm25,
            use_hybrid=rec.get("use_hybrid", False) and bm25 is not None,
            rrf_k=rec.get("rrf_k", 60), k_candidates=rec["k_candidates"],
            k_final=rec["k_final"], k_final_listing=rec.get("k_final_listing"),
            similarity_threshold=rec["similarity_threshold"],
            mmr_lambda=rec["mmr_lambda"], use_reranker=rec.get("use_reranker", False),
            off_category_penalty=rec.get("off_category_penalty", 0.0),
        )
        ret_ms = (time.perf_counter() - t0) * 1000

        # Gate de ámbito (out-of-domain): si la consulta es ajena a la ETSIIT/UGR
        # se emite la abstención canónica SIN generar (mismo comportamiento que la
        # interfaz), para que la métrica de no_info refleje el sistema real.
        sg = cfg.get("scope_gate", {})
        t1 = time.perf_counter()
        if not is_in_scope(
            item["query"], chunks, client,
            enabled=sg.get("enabled", True),
            reranker_used=rec.get("use_reranker", False),
            high_margin=sg.get("high_margin", 2.0),
            low_margin=sg.get("low_margin", -1.0),
        ):
            answer = OUT_OF_SCOPE_ANSWER
        else:
            sys_p, usr_p = build_messages(item["query"], chunks)
            answer = clean_answer(client.generate(sys_p, usr_p).text)
        gen_ms = (time.perf_counter() - t1) * 1000
        ret_ms_all.append(ret_ms); gen_ms_all.append(gen_ms)

        sources = [c.source_url for c in chunks]
        r = score_item(item, answer, sources)

        # Citación verificable (H1): ¿la respuesta lleva al menos una cita a una
        # fuente real recuperada? Se replica la lógica de la interfaz —marcadores
        # [n] válidos y, si no hay, atribución por solapamiento léxico—, salvo en
        # las abstenciones (que correctamente no citan).
        cited = False
        if chunks and not is_abstention(answer):
            source_map = get_source_map(chunks)
            rep = validate_citations(answer, source_map)
            cited = bool(rep.valid_markers) or bool(
                recover_sources(rep.clean_text, chunks, source_map,
                                threshold=gen.get("recover_threshold", 0.3))
            )
        r["cited"] = cited
        r["retrieval_ms"] = round(ret_ms)
        r["generation_ms"] = round(gen_ms)
        # Evidencia: pregunta, respuesta literal y fuentes recuperadas (para
        # inspección cualitativa y diagnóstico).
        r["question"] = item["query"]
        r["answer"] = answer
        r["retrieved_sources"] = sources
        results.append(r)

        mark = "OK " if r["answer_correct"] else "XX "
        print(f"  {mark} [{r['id']}] correcto={r['answer_correct']} "
              f"recup={r['retrieval_hit']} rehusó={r['refused']} fuga={r['leaked']} "
              f"kw={r['kw_coverage']} gen={r['generation_ms']}ms")
        if args.show:
            print("      Q:", item["query"])
            print("      A:", answer.replace("\n", " ")[:300])

    # ── Agregados ──
    n = len(results)
    factual = [r for r in results if r["type"] != "no_info"]
    noinfo = [r for r in results if r["type"] == "no_info"]
    ret_scored = [r for r in results if r["retrieval_hit"] is not None]

    def pct(x, total):
        return f"{100*x/total:.0f}%" if total else "-"

    # Citación verificable (H1 ≥ 90 %): sobre las consultas factuales realmente
    # RESPONDIDAS (no abstenidas), porcentaje que incluye ≥1 cita verificable.
    answered_factual = [r for r in factual if not r["refused"]]
    # Latencia total por consulta (recuperación + generación) y su P95 (H3 < 5 s).
    total_ms = sorted(rm + gm for rm, gm in zip(ret_ms_all, gen_ms_all))
    p95 = total_ms[min(len(total_ms) - 1, int(0.95 * len(total_ms)))] if total_ms else 0

    summary = {
        "model": model,
        "n": n,
        "answer_accuracy": pct(sum(r["answer_correct"] for r in results), n),
        "retrieval_recall": pct(sum(1 for r in ret_scored if r["retrieval_hit"]), len(ret_scored)),
        "factual_accuracy": pct(sum(r["answer_correct"] for r in factual), len(factual)),
        "noinfo_correct": pct(sum(r["answer_correct"] for r in noinfo), len(noinfo)),
        "citation_rate": pct(sum(1 for r in answered_factual if r.get("cited")), len(answered_factual)),
        "false_refusals": sum(1 for r in factual if r["refused"]),
        "leaks": sum(1 for r in results if r["leaked"]),
        "avg_retrieval_ms": round(sum(ret_ms_all) / n) if n else 0,
        "avg_generation_ms": round(sum(gen_ms_all) / n) if n else 0,
        "p95_total_ms": round(p95),
    }

    print("\n" + "=" * 64)
    print(f"RESUMEN — modelo: {model}")
    print("=" * 64)
    for k, v in summary.items():
        if k != "model":
            print(f"  {k:20s}: {v}")

    out = Path(args.out) if args.out else Path("eval") / f"results_{re.sub(r'[^a-zA-Z0-9]+','_',model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInforme guardado en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
