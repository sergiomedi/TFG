"""Contextualizer — *Contextual Retrieval* para GRAIA.

Implementa la técnica de **Contextual Retrieval** (Anthropic, 2024) como paso
opcional entre el *chunking* y la indexación.
Fuente: https://www.anthropic.com/news/contextual-retrieval (consultado 2026). La intuición: un fragmento
aislado pierde el contexto del documento del que procede ("¿de qué asignatura
es este horario?", "¿a qué grado pertenece este plan?"). Esto degrada tanto la
recuperación densa (el *embedding* no sabe situar el fragmento) como la léxica.

Solución: para cada *chunk* se pide a un LLM una breve descripción (1–2 frases)
que lo **sitúe** dentro del documento completo, y se antepone al texto del
fragmento *antes* de generar su *embedding* y de indexarlo en BM25. El fragmento
deja de ser "huérfano".

Diseño orientado a los principios del proyecto:

  - **Reproducibilidad / coste**: las contextualizaciones se **cachean en disco**
    (clave = hash del texto del fragmento). Re-ejecutar la indexación no vuelve a
    invocar al LLM; el corpus contextualizado es determinista.
  - **Trazabilidad**: el contexto generado se guarda también en
    ``chunk.metadata['context']``, de modo que cada respuesta del sistema puede
    trazarse hasta el fragmento y el contexto exactos que lo situaron.
  - **Robustez**: si el LLM falla o no está disponible, el fragmento se mantiene
    sin contexto (degradación elegante, nunca rompe la indexación).

Justificación de alternativas (mentalidad de tribunal):
  - *Title prepending* (anteponer el título): más barato, pero empeoró el ranking
    en pruebas previas (ver nota en ``index_corpus.py``) al homogeneizar las guías.
  - *Contextual Retrieval con LLM*: más costoso en indexación, pero el contexto es
    específico de cada fragmento y mejora la separación semántica. El coste se
    paga una sola vez (offline) y se amortiza con la caché.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Protocol, Sequence

from graia.ingesta.models import Chunk

logger = logging.getLogger(__name__)


# ── Prompt de contextualización ──
# Adaptado al español del prompt de Contextual Retrieval de Anthropic (2024):
# https://www.anthropic.com/news/contextual-retrieval
# (estructura original con etiquetas <document>/<chunk>; reescrito y traducido).

_SYSTEM_PROMPT = (
    "Eres un asistente experto que sitúa fragmentos de un documento académico "
    "en su contexto global para mejorar su recuperación en un buscador. "
    "Respondes siempre en español, de forma breve y factual."
)

_USER_TEMPLATE = """<documento>
{document}
</documento>

A continuación, el fragmento concreto que queremos situar dentro del documento anterior:
<fragmento>
{chunk}
</fragmento>

Redacta de 1 a 2 frases que SITÚEN este fragmento en el contexto del documento \
completo (a qué documento/título y a qué sección o tema pertenece, y a qué se \
refiere), pensadas para mejorar su recuperación en una búsqueda. No repitas el \
fragmento. Responde ÚNICAMENTE con esas frases de contexto, sin preámbulos ni \
comillas."""


class _Generator(Protocol):
    """Interfaz mínima esperada del cliente LLM (compatible con OllamaClient)."""

    def generate(self, system_prompt: str, user_message: str): ...  # -> obj con .text


class ContextCache:
    """Caché en disco (JSON) de contextos generados, indexada por hash de texto."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict[str, str] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                logger.info("Caché de contexto cargada: %d entradas (%s)",
                            len(self._data), self.path)
            except Exception as exc:
                logger.warning("No se pudo leer la caché de contexto: %s", exc)

    @staticmethod
    def key(chunk_text: str) -> str:
        return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:24]

    def get(self, chunk_text: str) -> str | None:
        return self._data.get(self.key(chunk_text))

    def put(self, chunk_text: str, context: str) -> None:
        self._data[self.key(chunk_text)] = context

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=0),
            encoding="utf-8",
        )
        logger.info("Caché de contexto guardada: %d entradas (%s)",
                    len(self._data), self.path)


def _generate_context(
    client: _Generator,
    document_text: str,
    chunk_text: str,
    *,
    max_doc_chars: int,
) -> str:
    """Pide al LLM una frase de contexto para *chunk_text* dentro del documento."""
    # El documento puede exceder la ventana del LLM local: se recorta de forma
    # conservadora a max_doc_chars (cabecera del documento, que en las páginas
    # UGR concentra título, grado y sección).
    doc = document_text[:max_doc_chars]
    user = _USER_TEMPLATE.format(document=doc, chunk=chunk_text)
    result = client.generate(_SYSTEM_PROMPT, user)
    context = (getattr(result, "text", "") or "").strip()
    # Saneado: una sola línea, sin comillas envolventes
    context = " ".join(context.split())
    if context.startswith(('"', "“", "'")) and context.endswith(('"', "”", "'")):
        context = context[1:-1].strip()
    return context


def contextualize_chunks(
    chunks: Sequence[Chunk],
    document_texts: dict[str, str],
    client: _Generator,
    *,
    cache: ContextCache | None = None,
    max_doc_chars: int = 6000,
    progress: Callable[[int, int], None] | None = None,
) -> list[Chunk]:
    """Antepone a cada *chunk* un breve contexto generado por el LLM.

    Parameters
    ----------
    chunks : Sequence[Chunk]
        Fragmentos producidos por el *chunker*.
    document_texts : dict[str, str]
        Mapa ``source_url -> texto completo del documento`` (para dar contexto).
    client : _Generator
        Cliente LLM con método ``generate(system, user)`` (e.g. ``OllamaClient``).
    cache : ContextCache | None
        Caché en disco. Si se proporciona, evita regenerar contextos ya vistos.
    max_doc_chars : int
        Límite de caracteres del documento que se pasa al LLM como contexto.
    progress : callable | None
        Callback ``(hechos, total)`` para mostrar progreso.

    Returns
    -------
    list[Chunk]
        Nuevos ``Chunk`` con ``text = "<contexto>\\n\\n<texto original>"`` y
        ``metadata['context']`` poblado. Los fragmentos cuyo contexto no se pudo
        generar se devuelven sin modificar.
    """
    out: list[Chunk] = []
    total = len(chunks)
    n_cached = n_generated = n_failed = n_skipped = 0

    for i, chunk in enumerate(chunks, 1):
        # Chunks ya autocontenidos (p.ej. registros de horario, que llevan
        # curso/especialidad/cuatrimestre en el propio texto): no necesitan
        # contexto adicional, así que se dejan intactos y se ahorra la llamada.
        if chunk.metadata.get("self_contained"):
            out.append(chunk)
            n_skipped += 1
            if progress and (i % 25 == 0 or i == total):
                progress(i, total)
            continue

        context = cache.get(chunk.text) if cache else None
        if context is None:
            doc_text = document_texts.get(chunk.source_url, chunk.title or "")
            try:
                context = _generate_context(
                    client, doc_text, chunk.text, max_doc_chars=max_doc_chars,
                )
                if cache is not None and context:
                    cache.put(chunk.text, context)
                n_generated += 1
            except Exception as exc:
                logger.warning("Contextualización falló (%s): %s",
                               chunk.source_url, exc)
                context = ""
                n_failed += 1
        else:
            n_cached += 1

        if context:
            new_meta = dict(chunk.metadata)
            new_meta["context"] = context
            out.append(chunk.model_copy(update={
                "text": f"{context}\n\n{chunk.text}",
                "metadata": new_meta,
            }))
        else:
            out.append(chunk)

        if progress and (i % 25 == 0 or i == total):
            progress(i, total)

    if cache is not None:
        cache.save()
    logger.info(
        "Contextualización completada: %d generados, %d de caché, %d fallidos, "
        "%d autocontenidos (saltados)",
        n_generated, n_cached, n_failed, n_skipped,
    )
    return out
