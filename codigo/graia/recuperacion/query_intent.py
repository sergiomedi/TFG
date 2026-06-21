"""query_intent — detección de la intención de la consulta.

Complementa al :mod:`query_router` (que decide *qué categorías* del corpus son
relevantes) con la detección de *qué forma* tiene la respuesta esperada. En
concreto, distingue las consultas de **listado/agregación** ("¿qué asignaturas
hay en 3º?", "lista de optativas del segundo cuatrimestre") de las consultas
puntuales ("¿a qué hora es Cálculo?").

Motivación (análisis de errores, Cap. 7):
  El contexto pasado al LLM (``k_final``) está dimensionado para consultas
  puntuales (5 fragmentos). Una consulta de listado necesita recuperar y agregar
  muchos más registros para ser completa; con sólo 5 fragmentos la respuesta
  queda truncada y se vuelve sensible a la formulación. Detectar la intención de
  listado permite ampliar ``k_final`` y reforzar los registros-resumen sólo
  cuando procede, sin penalizar la latencia de las consultas puntuales.

Decisión de diseño:
  Clasificador basado en reglas (igual que :mod:`query_router`): determinista,
  de latencia cero, interpretable y auditable. Se prefiere a un clasificador
  neuronal por las mismas razones documentadas en el router.
"""

from __future__ import annotations

import re

# Patrones que delatan una intención de listar/enumerar/agregar. Se exige bien
# un verbo/sustantivo de enumeración, bien un sustantivo en plural del dominio
# (asignaturas, materias, optativas) acompañado de un ámbito (curso,
# cuatrimestre, especialidad, grado).
_LISTING_PATTERNS: list[re.Pattern] = [
    # Verbos/expresiones explícitas de enumeración
    re.compile(r"\b(?:lista|listado|enumera|enum[eé]rame|dame\s+(?:la\s+)?lista)\b", re.IGNORECASE),
    re.compile(r"\b(?:cu[aá]les\s+son|qu[eé]\s+(?:asignaturas?|materias?|optativas?))\b", re.IGNORECASE),
    re.compile(r"\b(?:todas?\s+las?|todos?\s+los?)\s+(?:asignaturas?|materias?|optativas?|cr[eé]ditos?)\b", re.IGNORECASE),
    # Sustantivo plural del dominio + ámbito (curso/cuatrimestre/especialidad/grado)
    re.compile(
        r"\b(?:asignaturas?|materias?|optativas?)\b.{0,40}"
        r"\b(?:curso|cuatrimestre|semestre|especialidad|menci[oó]n|grado|primer|segund|tercer|cuart|\d\s*º)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:primer|segund|tercer|cuart|\d\s*º)\b.{0,40}"
        r"\b(?:asignaturas?|materias?|optativas?)\b",
        re.IGNORECASE,
    ),
    # "qué se imparte/cursa en …"
    re.compile(r"\bqu[eé]\s+se\s+(?:imparte|cursa|estudia|da)\b", re.IGNORECASE),
    # Enumeración por grupos ("…de todos los grupos", "cada grupo", "todos los
    # grupos"): requiere ampliar k para cubrir los 6 grupos de primer curso.
    re.compile(r"\b(?:todos?\s+los|cada|de\s+todos?\s+los)\s+grupos?\b", re.IGNORECASE),
    re.compile(r"\btodos?\s+los\s+grupos?\b", re.IGNORECASE),
]

# Curso en cifra ("3º", "3o", "3") o en palabra ("tercer", "tercero").
_CURSO_NUM_RE = re.compile(r"\b([1-4])\s*º?\b")
_CURSO_WORD_RE = re.compile(r"\b(primer|segund|tercer|cuart)", re.IGNORECASE)
_CURSO_WORD2NUM = {"primer": 1, "segund": 2, "tercer": 3, "cuart": 4}


def is_listing_query(query: str) -> bool:
    """True si *query* pide un listado/agregación (no un dato puntual)."""
    return any(p.search(query) for p in _LISTING_PATTERNS)


# Intención de TIPO de dato estructurado. El corpus indexa por registro dos tipos
# que comparten casi todo el vocabulario y se contaminan entre sí: el HORARIO de
# clase (teoría/prácticas) y el CALENDARIO de exámenes. Cuando la consulta pide
# claramente uno, conviene descartar el otro (ver retriever, filtro de tipo).
_HORARIO_INTENT_RE = re.compile(
    r"\b(?:horario|clase|clases|a\s+qu[eé]\s+hora|aula|"
    r"teor[ií]a|pr[aá]cticas?)\b",
    re.IGNORECASE,
)
_EXAMEN_INTENT_RE = re.compile(
    r"\b(?:examen|ex[aá]menes|examina|examinar|convocatoria|ordinaria|"
    r"extraordinaria)\b",
    re.IGNORECASE,
)


def desired_structured_tipo(query: str) -> str | None:
    """Tipo de registro estructurado que la consulta pide de forma inequívoca.

    Devuelve ``"horario"`` si la consulta es claramente sobre horario de clase y
    NO menciona exámenes; ``"calendario"`` si es sobre exámenes/convocatorias y NO
    sobre horario de clase; y ``None`` si es ambigua o no aplica (en cuyo caso no
    se filtra por tipo, para no arriesgar falsos rechazos). Es la base del filtro
    de coherencia de tipo, simétrico al de asignatura.
    """
    hor = bool(_HORARIO_INTENT_RE.search(query))
    exa = bool(_EXAMEN_INTENT_RE.search(query))
    if hor and not exa:
        return "horario"
    if exa and not hor:
        return "calendario"
    return None


def extract_curso(query: str) -> int | None:
    """Extrae el número de curso (1–4) mencionado en la consulta, si lo hay.

    Reconoce tanto cifras ("3º", "3er", "3") como palabras ("tercero",
    "cuarto"). Devuelve ``None`` si no se menciona un curso de forma explícita,
    en cuyo caso el filtrado por curso no debe aplicarse.
    """
    w = _CURSO_WORD_RE.search(query)
    if w:
        return _CURSO_WORD2NUM[w.group(1).lower()]
    m = _CURSO_NUM_RE.search(query)
    if m:
        return int(m.group(1))
    return None
