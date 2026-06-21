"""contextual_query — recuperación consciente del historial (anáfora).

Problema (análisis de errores, Cap. 7): el historial de conversación se inyecta
en el modelo de GENERACIÓN, pero el RECUPERADOR solo recibe la consulta actual.
En seguimientos elípticos ("y las de CA?", "me refiero al subgrupo A2") la
consulta es ambigua y el recuperador trae fragmentos de otra asignatura o grupo,
por lo que el modelo —aun teniendo el historial— no dispone del fragmento
correcto y falla.

Solución: enriquecer la consulta de recuperación arrastrando las **entidades del
dominio** (asignatura/sigla, grupo, subgrupo, tipo de clase) mencionadas en los
turnos de usuario recientes cuando la consulta actual es un seguimiento y le
faltan. Es determinista, de latencia cero e interpretable (misma filosofía que el
:mod:`query_router`), frente a una reescritura con LLM que añadiría latencia y un
punto de fallo. La consulta enriquecida se emplea tanto en la RECUPERACIÓN como en
la GENERACIÓN; la generación NO recibe el historial conversacional en bruto, pues la
anáfora ya queda resuelta de forma determinista en la propia consulta. Inyectar el
historial completo en un LLM pequeño resultaba frágil: un turno previo de otro tipo
de dato (horario↔calendario) podía inducir una abstención errónea.

Ámbito acotado al dominio de horarios de clase, que es donde aparece la anáfora
sobre el producto (asignatura × grupo × subgrupo × tipo).
"""

from __future__ import annotations

import logging
import re

from graia.recuperacion.query_router import detect_subject_siglas

logger = logging.getLogger(__name__)

# Disparadores de seguimiento elíptico: la consulta empieza por un conector
# anafórico ("y…", "me refiero…", "también…", "además…").
_ANAPHORIC_RE = re.compile(
    r"^\s*(?:¿\s*)?(?:y\b|me\s+(?:refier[oa]|refer[ií]a)|quer[ií]a\s+decir"
    r"|tambi[eé]n|adem[aá]s|igualmente|otra\s+vez)",
    re.IGNORECASE,
)

# Entidades del dominio de horarios.
_GRUPO_RE = re.compile(r"\bgrupo\s+([1-4]\s*º?\s*[A-H])\b", re.IGNORECASE)
_GRUPO_BARE_RE = re.compile(r"\b([1-4]\s*º\s*[A-H])\b")
_SUBGRUPO_RE = re.compile(r"\bsubgrupo\s+([A-H]?\s*[1-9])\b", re.IGNORECASE)
# "grupo D" / "grupo A" (letra sola, sin curso): se usa como respaldo cuando no
# aparece el formato "1ºD".
_GRUPO_LETTER_RE = re.compile(r"\bgrupo\s+([A-H])\b", re.IGNORECASE)
_PRACTICA_RE = re.compile(r"\bpr[aá]cticas?\b", re.IGNORECASE)
_TEORIA_RE = re.compile(r"\bteor[ií]a\b", re.IGNORECASE)
_SIGLA_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,6})\b")

# Orden estable al reconstruir la consulta enriquecida.
_ENTITY_ORDER = ("subject", "tipo", "grupo", "subgrupo")


def _extract_entities(text: str) -> dict[str, str]:
    """Extrae las entidades de horario presentes en *text* (forma superficial)."""
    ent: dict[str, str] = {}

    # Asignatura: por sigla (DI) o por nombre completo (Cálculo, Derecho
    # Informático), insensible a tildes. Reconocer el nombre completo evita
    # arrastrar la asignatura anterior cuando el usuario introduce una nueva
    # por su nombre ("Y Cálculo?").
    subjects = detect_subject_siglas(text)
    if subjects:
        ent["subject"] = subjects[0]

    g = _GRUPO_RE.search(text) or _GRUPO_BARE_RE.search(text)
    if g:
        ent["grupo"] = "grupo " + g.group(1).replace(" ", "")
    else:
        gl = _GRUPO_LETTER_RE.search(text)
        if gl:
            ent["grupo"] = "grupo " + gl.group(1).upper()

    sg = _SUBGRUPO_RE.search(text)
    if sg:
        ent["subgrupo"] = "subgrupo " + sg.group(1).replace(" ", "").upper()

    if _PRACTICA_RE.search(text):
        ent["tipo"] = "prácticas"
    elif _TEORIA_RE.search(text):
        ent["tipo"] = "teoría"

    return ent


def _is_followup(query: str, current_entities: dict[str, str]) -> bool:
    """True si *query* parece un seguimiento elíptico que conviene enriquecer."""
    if _ANAPHORIC_RE.match(query):
        return True
    # Consulta corta sin asignatura pero con grupo/subgrupo/tipo: probable
    # continuación del tema anterior ("¿y el subgrupo A2?").
    if (
        len(query.split()) <= 6
        and "subject" not in current_entities
        and any(k in current_entities for k in ("grupo", "subgrupo", "tipo"))
    ):
        return True
    return False


def enrich_query_with_history(
    query: str,
    messages: list[dict],
    *,
    max_lookback: int = 6,
) -> str:
    """Enriquece *query* con entidades de horario de los turnos de usuario previos.

    Solo actúa si *query* es un seguimiento elíptico (ver :func:`_is_followup`).
    Arrastra las entidades (asignatura, tipo, grupo, subgrupo) que faltan en la
    consulta actual, tomando el valor más reciente del historial de usuario.

    Parameters
    ----------
    query : str
        Consulta actual del usuario.
    messages : list[dict]
        Historial de la sesión (``{"role", "content"}``); puede incluir la
        consulta actual al final (se ignora).
    max_lookback : int
        Número máximo de turnos de usuario previos a inspeccionar.

    Returns
    -------
    str
        Consulta enriquecida para la RECUPERACIÓN (la generación usa la original).
    """
    current = _extract_entities(query)
    if not _is_followup(query, current):
        return query

    # Turnos de usuario previos (excluye la consulta actual si está al final).
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]
    if user_turns and user_turns[-1].strip() == query.strip():
        user_turns = user_turns[:-1]

    carried: dict[str, str] = {}
    for turn in reversed(user_turns[-max_lookback:]):
        for k, v in _extract_entities(turn).items():
            if k not in current and k not in carried:
                carried[k] = v

    if not carried:
        return query

    additions = [carried[k] for k in _ENTITY_ORDER if k in carried]
    enriched = f"{query.strip()} {' '.join(additions)}"
    logger.info("Consulta enriquecida por historial: %r → %r", query, enriched)
    return enriched
