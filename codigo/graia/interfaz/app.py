"""Interfaz conversacional de GRAIA — aplicación Streamlit.

Implementa la capa de interfaz descrita en la Sección 5.11 del diseño:
  - Chat con ``st.chat_input`` + ``st.chat_message``
  - Streaming de tokens con ``st.write_stream``
  - Historial de conversación en ``st.session_state``
  - Carga de configuración YAML al arranque
  - Bloque de fuentes verificadas tras cada respuesta
  - Mensaje de bienvenida de la persona GRAIA

Ejecución: ``streamlit run graia/interfaz/app.py``
"""

from __future__ import annotations

import multiprocessing
multiprocessing.freeze_support()          # Windows + PyTorch spawn fix

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # evita fork del tokenizer HF
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # Windows + PyTorch fix
# Evita el cierre nativo silencioso por OpenMP duplicado (torch y faiss traen su
# propia libiomp5md.dll en Windows). Debe fijarse ANTES de importar torch/faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
torch.set_num_threads(1)                  # evita conflicto de threads con Streamlit
# Workaround conocido de Streamlit + PyTorch: impide que la introspección de
# Streamlit acceda a ``torch.classes.__path__`` (causa de RuntimeError/segfault
# al re-ejecutar el script en cada interacción).
try:
    torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st
import yaml

from graia.generacion.citation_validator import (
    CitationReport,
    format_sources_block,
    recover_sources,
    validate_citations,
)
from graia.generacion.dedup import deduplicate_sentences
from graia.generacion.ollama_client import OllamaClient
from graia.generacion.postprocess import clean_answer, normalize_markdown_lists
from graia.generacion.prompt_builder import build_messages, get_source_map
from graia.interfaz.history import (
    CLOSINGS,
    is_meta_query as _is_meta_query,
    is_no_info_answer as _is_no_info_answer,
    is_subjective_decline as _is_subjective_decline,
)
from graia.indexacion.embedder import Embedder
from graia.indexacion.vector_store import VectorStore
from graia.recuperacion.bm25 import BM25Index
from graia.recuperacion.query_router import route_query
from graia.recuperacion.retriever import retrieve
from graia.registro.logger import get_logger

logger = get_logger("graia.interfaz")

# Raíz del proyecto (codigo/) para localizar los scripts de operación del admin.
ROOT = Path(__file__).resolve().parents[2]


# ---- Identidad visual: paleta blanco/rojo de Granada ----

def inject_granada_css() -> None:
    """Inyecta los estilos de la identidad visual (rojo y blanco de Granada)."""
    st.markdown(
        """
        <style>
        :root { --granada-red:#C8102E; --granada-dark:#8E0E20; }
        .stApp { background-color:#FFFFFF; }
        h1, h2, h3 { color: var(--granada-red) !important; }
        section[data-testid="stSidebar"] {
            background-color:#F7E8E8;
            border-right:3px solid var(--granada-red);
        }
        .stButton>button {
            border:1px solid var(--granada-red);
            color:var(--granada-red);
            border-radius:8px;
            font-weight:600;
        }
        .stButton>button:hover { background-color:var(--granada-red); color:#FFFFFF; }
        .stButton>button[kind="primary"] { background-color:var(--granada-red); color:#FFFFFF; }
        .granada-banner {
            background:linear-gradient(90deg,var(--granada-red),var(--granada-dark));
            color:#FFFFFF; padding:12px 18px; border-radius:12px; margin-bottom:14px;
            font-size:1.25rem; font-weight:700; letter-spacing:.3px;
        }
        .granada-banner small { font-weight:400; opacity:.92; }
        [data-testid="stChatMessage"] { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_banner(subtitle: str) -> None:
    """Cabecera de marca común a las dos vistas."""
    st.markdown(
        f'<div class="granada-banner">🎓 GRAIA &middot; {subtitle} '
        f'<small>— ETSIIT · Universidad de Granada</small></div>',
        unsafe_allow_html=True,
    )


# ---- Ejecución de los scripts de operación (panel de administrador) ----

def run_script(args: list[str], timeout: int = 1800) -> tuple[int, str]:
    """Ejecuta un script del proyecto y devuelve (código de salida, salida combinada).

    Reproduce, desde la interfaz, la invocación de los procedimientos de
    mantenimiento del administrador (RF-04 a RF-10), de modo que pueda operar
    el sistema mediante botones sin recurrir a la línea de comandos.
    """
    cmd = [sys.executable, *args]
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        out = proc.stdout or ""
        if proc.stderr:
            out += "\n[stderr]\n" + proc.stderr
        return proc.returncode, out.strip() or "(sin salida)"
    except subprocess.TimeoutExpired:
        return -1, f"Tiempo de ejecución agotado ({timeout}s)."
    except Exception as exc:  # pragma: no cover - feedback al usuario
        return -1, f"Error al ejecutar {' '.join(args)}: {exc}"


def _run_and_report(label: str, args: list[str], timeout: int = 1800) -> None:
    """Ejecuta un script mostrando spinner y volcando la salida en la interfaz."""
    with st.spinner(f"Ejecutando: {label} …"):
        code, out = run_script(args, timeout=timeout)
    (st.success if code == 0 else st.error)(
        f"{label} — código de salida {code}"
    )
    st.code(out or "(sin salida)", language="text")


# ---- Configuración de la página ----

st.set_page_config(
    page_title="GRAIA — Asistente Académico ETSIIT",
    page_icon="🎓",
    layout="centered",
)

WELCOME_MSG = (
    "Buenos días, soy GRAIA, el asistente académico de la ETSIIT. "
    "¿En qué puedo ayudarle hoy con sus consultas académicas?"
)

def _pick_closing() -> str:
    """Devuelve un cierre cortés evitando repetir el del turno anterior."""
    last = st.session_state.get("last_closing")
    options = [c for c in CLOSINGS if c != last] or list(CLOSINGS)
    choice = random.choice(options)
    st.session_state.last_closing = choice
    return choice


# ---- Carga de configuración y componentes (cacheados) ----

@st.cache_resource(show_spinner="Cargando configuración...")
def load_config(config_path: str = "config/default.yaml") -> dict[str, Any]:
    """Carga la configuración YAML una sola vez."""
    path = Path(config_path)
    if not path.exists():
        st.error(f"Fichero de configuración no encontrado: {config_path}")
        st.stop()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embedder(cfg: dict) -> Embedder:
    """Inicializa el Embedder (descarga pesos si es necesario)."""
    emb_cfg = cfg["embeddings"]
    return Embedder(
        model_name=emb_cfg["model_name"],
        query_prefix=emb_cfg["query_prefix"],
        passage_prefix=emb_cfg["passage_prefix"],
        batch_size=emb_cfg["batch_size"],
        normalize=emb_cfg["normalize"],
    )


@st.cache_resource(show_spinner="Cargando índice vectorial...")
def load_store(cfg: dict) -> VectorStore:
    """Carga el VectorStore desde disco."""
    index_path = cfg["paths"]["index"]
    return VectorStore.load(index_path)


@st.cache_resource(show_spinner="Cargando índice BM25...")
def load_bm25(cfg: dict) -> BM25Index | None:
    """Carga el índice BM25 si existe (para recuperación híbrida)."""
    index_path = Path(cfg["paths"]["index"])
    bm25_path = index_path / "bm25_index.pkl"
    if bm25_path.exists():
        return BM25Index.load(index_path)
    logger.warning("Índice BM25 no encontrado en %s; recuperación puramente densa.", index_path)
    return None


@st.cache_resource(show_spinner="Conectando con Ollama...")
def load_client(cfg: dict) -> OllamaClient:
    """Inicializa el cliente Ollama."""
    gen_cfg = cfg["generacion"]
    return OllamaClient(
        model=gen_cfg["model"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        max_tokens=gen_cfg["max_tokens"],
        stop=gen_cfg.get("stop", []),
    )


# ---- Estado de la sesión ----

def init_session_state() -> None:
    """Inicializa las variables de sesión si no existen."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MSG}
        ]
    if "conversations" not in st.session_state:
        # Historial de conversaciones previas: lista de dicts
        # {id, title, messages}
        st.session_state.conversations = []
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None
    if "last_closing" not in st.session_state:
        st.session_state.last_closing = None


# ---- Panel lateral: historial de conversaciones ----

def _extract_title(messages: list[dict]) -> str:
    """Extrae un título breve de la primera consulta del usuario."""
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"][:40]
            if len(msg["content"]) > 40:
                text += "..."
            return text
    return "Nueva conversación"


def render_sidebar() -> None:
    """Renderiza el panel lateral con el historial de conversaciones,
    tal como se muestra en la Figura 5.2 del diseño."""
    with st.sidebar:
        st.markdown("**Historial**")
        st.divider()

        # Botón para nueva conversación
        if st.button("➕ Nueva conversación", use_container_width=True):
            _save_current_conversation()
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MSG}
            ]
            st.session_state.current_conv_id = None
            st.rerun()

        # Mostrar conversaciones guardadas (más reciente primero)
        for conv in reversed(st.session_state.conversations):
            is_active = conv["id"] == st.session_state.current_conv_id
            label = conv["title"]
            if st.button(
                label,
                key=f"conv_{conv['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _save_current_conversation()
                st.session_state.messages = conv["messages"].copy()
                st.session_state.current_conv_id = conv["id"]
                st.rerun()


def _save_current_conversation() -> None:
    """Guarda la conversación actual en el historial si tiene mensajes del usuario."""
    has_user_msg = any(
        m["role"] == "user" for m in st.session_state.messages
    )
    if not has_user_msg:
        return

    conv_id = st.session_state.current_conv_id
    title = _extract_title(st.session_state.messages)

    if conv_id is not None:
        # Actualizar conversación existente
        for conv in st.session_state.conversations:
            if conv["id"] == conv_id:
                conv["messages"] = st.session_state.messages.copy()
                conv["title"] = title
                return
    else:
        # Nueva conversación
        new_id = len(st.session_state.conversations)
        st.session_state.conversations.append({
            "id": new_id,
            "title": title,
            "messages": st.session_state.messages.copy(),
        })
        st.session_state.current_conv_id = new_id


# ---- Renderizado del historial del chat ----

def render_history() -> None:
    """Muestra todos los mensajes anteriores del historial."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---- Pipeline de respuesta ----

def handle_query(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    client: OllamaClient,
    cfg: dict,
    bm25_index: BM25Index | None = None,
) -> None:
    """Ejecuta el pipeline completo para una consulta del usuario."""
    rec_cfg = cfg["recuperacion"]

    # 0. Recuperación consciente del historial: en seguimientos elípticos
    # ("y las de CA?", "me refiero al subgrupo A2") se arrastran las entidades de
    # horario (asignatura/grupo/subgrupo/tipo) de los turnos previos para que el
    # recuperador no traiga otra asignatura o grupo. Solo afecta a la búsqueda;
    # la generación recibe la consulta original + historial.
    from graia.recuperacion.contextual_query import enrich_query_with_history
    from graia.recuperacion.query_router import expand_abbreviations
    retrieval_query = enrich_query_with_history(query, st.session_state.messages)
    # Expansión de siglas de asignaturas (DI → Derecho Informático, etc.)
    expanded_query = expand_abbreviations(retrieval_query)

    # 1. Recuperación (híbrida si está configurada y disponible)
    use_hybrid = rec_cfg.get("use_hybrid", False) and bm25_index is not None
    t0 = time.perf_counter()
    chunks = retrieve(
        expanded_query,
        embedder,
        store,
        bm25_index=bm25_index,
        use_hybrid=use_hybrid,
        rrf_k=rec_cfg.get("rrf_k", 60),
        k_candidates=rec_cfg["k_candidates"],
        k_final=rec_cfg["k_final"],
        k_final_listing=rec_cfg.get("k_final_listing"),
        similarity_threshold=rec_cfg["similarity_threshold"],
        mmr_lambda=rec_cfg["mmr_lambda"],
        use_reranker=rec_cfg.get("use_reranker", False),
        off_category_penalty=rec_cfg.get("off_category_penalty", 0.0),
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # 1b. Gate de ámbito (out-of-domain): si la consulta no trata sobre asuntos
    # académicos de la ETSIIT/UGR, se responde con la abstención canónica SIN
    # generar, evitando respuestas paramétricas con citas espurias (p.ej.
    # "París"). Híbrido: margen del reranker (coste cero) + LLM solo si hay duda.
    from graia.recuperacion.scope_classifier import OUT_OF_SCOPE_ANSWER, is_in_scope
    sg_cfg = cfg.get("scope_gate", {})
    if not is_in_scope(
        query, chunks, client,
        enabled=sg_cfg.get("enabled", True),
        reranker_used=rec_cfg.get("use_reranker", False),
        high_margin=sg_cfg.get("high_margin", 2.0),
        low_margin=sg_cfg.get("low_margin", -1.0),
    ):
        with st.chat_message("assistant"):
            st.markdown(OUT_OF_SCOPE_ANSWER)
        st.session_state.messages.append(
            {"role": "assistant", "content": OUT_OF_SCOPE_ANSWER}
        )
        logger.info(
            "query_processed",
            extra={
                "query": query[:200],
                "num_chunks": len(chunks),
                "retrieval_ms": round(retrieval_ms, 1),
                "scope": "out_of_domain",
            },
        )
        return

    # 2. Construcción del prompt. Se usa la consulta con la ANÁFORA YA RESUELTA de
    # forma determinista (expanded_query = entidades arrastradas del historial por
    # enrich_query_with_history + siglas expandidas), de modo que la generación sea
    # AUTOCONTENIDA: "y los examenes?" llega al modelo como "y los examenes?
    # (Derecho Informático)" y el LLM no necesita el historial crudo para conocer el
    # sujeto.
    system_prompt, user_message = build_messages(expanded_query, chunks)
    source_map = get_source_map(chunks)

    # 2b. NO se inyecta el historial conversacional en la generación. La anáfora ya
    # queda resuelta en la consulta (paso anterior); inyectar además los turnos
    # previos hacía que un turno de OTRO tipo de dato (horario↔calendario) indujera
    # al modelo de 8B a abstenerse erróneamente. La memoria conversacional se
    # mantiene por la vía determinista (arrastre de entidades), no por historial
    # crudo —misma filosofía que la recuperación (enrich_query_with_history)—.
    gen_cfg = cfg.get("generacion", {})
    history: list[dict[str, str]] = []

    # 3. Generación con streaming
    with st.chat_message("assistant"):
        placeholder = st.empty()
        t1 = time.perf_counter()
        acc = ""
        for token in client.generate_stream(system_prompt, user_message, history=history):
            acc += token
            placeholder.markdown(acc)
        generation_ms = (time.perf_counter() - t1) * 1000

        # 3b. Post-filtro determinista: recorta preámbulos de razonamiento y
        # coletillas del modelo (red de seguridad con modelos pequeños).
        answer = clean_answer(acc)

        # 3c. Deduplicación de contenido: elimina frases redundantes que el
        # modelo a veces repite citando fuentes distintas (el sistema de citas
        # deduplica las fuentes por URL, pero no el contenido). Se aplica ANTES
        # de validar citas para que las citas sobrantes de las frases eliminadas
        # no aparezcan en el bloque de fuentes.
        answer = deduplicate_sentences(
            answer, threshold=gen_cfg.get("dedup_threshold", 0.5),
        )

        # 3d. Normalizar viñetas: el modelo a veces escribe listas con '*'/'+'
        # en línea (sin saltos), que Markdown no renderiza. Se convierten en
        # items de lista para que se muestren correctamente.
        answer = normalize_markdown_lists(answer)

        # 4. Validación de citas: el cuerpo se toma YA limpio de citas inválidas
        # o malformadas ([5.1-3], etc.); las fuentes se deduplican por URL.
        report = validate_citations(answer, source_map)
        sources_block = format_sources_block(report)

        # 4a-bis. Recuperación de citas: si el modelo respondió con datos del
        # contexto pero SIN marcadores [n] (no_sources), atribuir las fuentes por
        # solapamiento léxico, deduplicadas por URL. Evita dejar al usuario sin
        # fuente en respuestas sustantivas.
        if (
            not report.sources
            and chunks
            and not _is_no_info_answer(report.clean_text)
            and not _is_meta_query(query)
            and not _is_subjective_decline(report.clean_text)
        ):
            recovered = recover_sources(
                report.clean_text, chunks, source_map,
                threshold=gen_cfg.get("recover_threshold", 0.25),
            )
            if recovered:
                sources_block = format_sources_block(
                    CitationReport(sources=recovered)
                )

        # 4b. Cierre cortés variado (tono de la persona GRAIA), antes del pie de
        # fuentes para que estas queden como referencia final. Se omite en las
        # respuestas de "no dispongo de información" para no sonar redundante
        # (ya remiten a la instancia adecuada).
        full_response = report.clean_text
        if not _is_no_info_answer(report.clean_text):
            full_response += f"\n\n{_pick_closing()}"
        if sources_block:
            full_response += f"\n{sources_block}"

        # Re-render con la respuesta limpia + cierre + bloque de fuentes
        placeholder.markdown(full_response)

    # 5. Guardar en historial
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    # 6. Logging (incluye info de query routing para trazabilidad)
    route = route_query(query)  # re-calcular es O(1), solo regex matching
    logger.info(
        "query_processed",
        extra={
            "query": query[:200],
            "num_chunks": len(chunks),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "citations_valid": len(report.valid_markers),
            "citations_invalid": len(report.invalid_markers),
            "routed_categories": list(route.categories.keys()) if route.is_routed else [],
            "routing_rules": route.matched_rules,
        },
    )


# ---- Panel de administrador (RF-04 a RF-10 / CU-02 a CU-05) ----

def render_admin_panel(cfg: dict) -> None:
    """Vista de operación del administrador.

    Expone, mediante botones, las acciones de mantenimiento del sistema descritas
    en los requisitos funcionales del administrador (RF-04 a RF-10) y sus casos de
    uso (CU-02 a CU-05): reconstrucción y gestión del corpus, indexación y
    evaluación del rendimiento. Cada botón invoca el procedimiento (\\emph{script})
    correspondiente y muestra su salida.
    """
    render_banner("Panel de administración")
    st.caption("Operación y mantenimiento del corpus, la indexación y la evaluación del sistema.")

    # Categorías disponibles para la reconstrucción del corpus (leídas de sources.yaml).
    try:
        _src = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
        all_cats = list((_src.get("categorias") or {}).keys())
    except Exception:
        all_cats = []
    if not all_cats:
        all_cats = ["guias_docentes", "calendario", "normativa", "tramites",
                    "movilidad", "tfg", "estudiantes", "profesorado"]

    st.info(
        "Algunas operaciones (reconstruir el corpus, reindexar, evaluar) pueden "
        "tardar varios minutos; la interfaz queda a la espera hasta que terminan.",
        icon="⏳",
    )

    # ── CU-02 · RF-04: Reconstruir el corpus ────────────────────────────────
    with st.expander("🌐 Reconstrucción del corpus", expanded=True):
        selected_cats = st.multiselect(
            "Categorías a procesar",
            options=["Todas"] + all_cats,
            default=["Todas"],
            key="adm_build_cats",
            help="«Todas» procesa el corpus completo; o selecciona categorías concretas.",
        )
        max_pages = st.number_input(
            "Máximo de páginas (opcional, 0 = sin límite)",
            min_value=0, value=0, step=10, key="adm_build_maxp",
        )
        if st.button("Reconstruir corpus", key="adm_build_btn"):
            args = ["scripts/build_corpus.py"]
            chosen = [c for c in selected_cats if c != "Todas"]
            if chosen and "Todas" not in selected_cats:
                args += ["--categories", *chosen]
            if max_pages:
                args += ["--max-pages", str(int(max_pages))]
            _run_and_report("Reconstrucción del corpus", args)

    # ── Incorporar un documento verificado (CU-03 / RF-05) ──────────────────
    with st.expander("📄 Incorporar un documento verificado"):
        inj_url = st.text_input("URL oficial (debe estar ya en el corpus, se conserva para las citas)", key="adm_inj_url")
        inj_title = st.text_input("Título mostrado en las citas", key="adm_inj_title")
        c1, c2 = st.columns(2)
        with c1:
            inj_cat = st.text_input(
                "Categoría (obligatoria, p.ej. movilidad, normativa, tramites)",
                key="adm_inj_cat",
            )
        with c2:
            inj_tipo = st.selectbox(
                "Tipo estructurado (opcional)",
                options=["(ninguno)", "horario", "calendario", "plan_estudios"],
                index=0,
                help="Solo para horario/calendario/plan de estudios (troceo por registro). "
                     "Déjalo en «(ninguno)» para un documento verificado normal.",
                key="adm_inj_tipo",
            )
        inj_file = st.file_uploader(
            "Documento verificado (.txt). Para los tipos estructurados, un registro por línea.",
            type=["txt"], key="adm_inj_file",
        )
        if st.button("Incorporar documento", key="adm_inj_btn"):
            if not (inj_url.strip() and inj_file is not None and inj_cat.strip()):
                st.warning("Indica la URL, la categoría y selecciona el fichero .txt verificado.")
            else:
                tmp = ROOT / "data" / "verificado" / "_admin_inject.txt"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(inj_file.getvalue())
                args = ["scripts/inject_doc.py", "--url", inj_url,
                        "--text", "data/verificado/_admin_inject.txt",
                        "--category", inj_cat.strip()]
                if inj_tipo != "(ninguno)":
                    args += ["--tipo", inj_tipo]
                if inj_title.strip():
                    args += ["--title", inj_title]
                _run_and_report("Inserción de documento verificado", args, timeout=300)

    # ── Revisar el corpus (CU-04 / RF-06) ───────────────────────────────────
    with st.expander("🔍 Revisar el corpus"):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Informe resumen", key="adm_review_btn"):
                _run_and_report("Revisión del corpus",
                                ["scripts/review_corpus.py"], timeout=300)
        with col_b:
            rev_search = st.text_input("Buscar en el corpus", key="adm_review_search",
                                       placeholder="término…")
            if st.button("Buscar", key="adm_review_search_btn") and rev_search.strip():
                _run_and_report("Búsqueda en el corpus",
                                ["scripts/review_corpus.py", "--search", rev_search],
                                timeout=300)

    # ── Limpiar el corpus (CU-05 / RF-07) ───────────────────────────────────
    with st.expander("🧹 Limpiar el corpus"):
        rm_url = st.text_input(
            "Eliminar un documento por su URL exacta", key="adm_rm_url",
            help="Borra el documento cuya URL coincide exactamente con la indicada.",
        )
        rm_search = st.text_input(
            "Eliminar por término de búsqueda", key="adm_rm_search",
            help="Borra los documentos cuyo título o texto contienen el término.",
        )
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            rm_minchars = st.number_input("Eliminar con menos de N caracteres",
                                          min_value=0, value=0, step=50, key="adm_rm_minchars")
        with cc2:
            rm_oldexams = st.checkbox(
                "Eliminar calendarios/exámenes de cursos anteriores", key="adm_rm_oldexams",
                help="Heurística que detecta y borra calendarios de exámenes de años pasados.",
            )
        with cc3:
            rm_dry = st.checkbox(
                "Simulación (no borra)", value=True, key="adm_rm_dry",
                help="Muestra qué se eliminaría sin aplicar cambios al corpus.",
            )
        if st.button("Aplicar limpieza", key="adm_rm_btn"):
            args = ["scripts/review_corpus.py"]
            if rm_url.strip():
                args += ["--remove-url", rm_url]
            if rm_search.strip():
                args += ["--remove-search", rm_search]
            if rm_minchars:
                args += ["--remove-min-chars", str(int(rm_minchars))]
            if rm_oldexams:
                args += ["--remove-old-exams"]
            if rm_dry:
                args += ["--dry-run"]
            if len(args) == 1:
                st.warning("Selecciona al menos un criterio de limpieza.")
            else:
                _run_and_report("Limpieza del corpus", args, timeout=300)

    # ── Añadir una fuente específica (CU-06 / RF-08) ────────────────────────
    with st.expander("➕ Añadir una fuente específica"):
        add_url = st.text_input("URL del documento a añadir", key="adm_add_url")
        add_cat = st.text_input("Categoría", value="extra", key="adm_add_cat")
        if st.button("Añadir fuente", key="adm_add_btn"):
            if not add_url.strip():
                st.warning("Indica la URL de la fuente a añadir.")
            else:
                _run_and_report(
                    "Añadir fuente específica",
                    ["scripts/review_corpus.py", "--add-url", add_url,
                     "--add-category", add_cat],
                    timeout=600,
                )

    # ── Indexación del corpus ───────────────────────────────────────────────
    with st.expander("🧱 Indexación del corpus"):
        idx_dry = st.checkbox(
            "Simulación (opcional): solo calcular estadísticas de fragmentación",
            key="adm_idx_dry",
            help="Si se marca, no construye el índice: muestra cuántos fragmentos "
                 "se generarían y su tamaño. El resultado aparece en el recuadro "
                 "de salida bajo el botón.",
        )
        if st.button("Reindexar corpus", key="adm_idx_btn"):
            args = ["scripts/index_corpus.py"]
            if idx_dry:
                args += ["--dry-run"]
            _run_and_report("Indexación del corpus", args)

    # ── Evaluación del rendimiento ──────────────────────────────────────────
    with st.expander("📊 Evaluación del rendimiento del sistema"):
        default_model = cfg.get("generacion", {}).get("model", "llama3.1:8b-instruct-q4_K_M")
        eval_model = st.text_input("Modelo a evaluar", value=default_model, key="adm_eval_model")
        eval_limit = st.number_input("Limitar a N preguntas (0 = todas)",
                                     min_value=0, value=0, step=1, key="adm_eval_limit")
        if st.button("Evaluar rendimiento", key="adm_eval_btn"):
            args = ["scripts/evaluate.py", "--model", eval_model]
            if eval_limit:
                args += ["--limit", str(int(eval_limit))]
            _run_and_report("Evaluación del rendimiento", args)


# ---- Main ----

def main() -> None:
    """Punto de entrada de la aplicación Streamlit."""
    inject_granada_css()

    # Inicialización
    init_session_state()
    cfg = load_config()

    # Cargar (desde caché) los recursos pesados SIEMPRE, en cada \emph{rerun} y
    # vista. Tras la primera carga es instantáneo, pero garantiza que Streamlit no
    # los considere huérfanos al cambiar de vista y los libere: liberar los
    # modelos de torch/FAISS en mitad de la sesión puede provocar un cierre
    # nativo del proceso en Windows.
    embedder = load_embedder(cfg)
    store = load_store(cfg)
    bm25_index = load_bm25(cfg)
    client = load_client(cfg)

    # Selector de vista: asistente (estudiante) o panel de administración.
    with st.sidebar:
        vista = st.radio(
            "Vista",
            ["💬 Asistente", "🛠️ Administración"],
            key="vista",
        )
        st.divider()

    if vista.endswith("Administración"):
        render_admin_panel(cfg)
        return

    # ── Vista del estudiante: asistente conversacional ──
    render_banner("Asistente Académico Inteligente")

    # Panel lateral con historial de conversaciones (Figura 5.2)
    render_sidebar()

    render_history()

    # Entrada del usuario
    if query := st.chat_input("Pregunta sobre la ETSIIT..."):
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Procesar y responder
        handle_query(query, embedder, store, client, cfg, bm25_index=bm25_index)

        # Guardar conversación en el historial del sidebar
        _save_current_conversation()


if __name__ == "__main__":
    main()
