"""QueryRouter — enrutamiento de consultas a categorías del corpus.

Implementa el componente de query routing descrito en la Sección 5.8.4:
  - Clasifica consultas en una o más categorías del corpus
  - Permite focalizar la búsqueda en el subconjunto relevante
  - Reduce el ruido de categorías irrelevantes (e.g., 841 guías docentes
    cuando se pregunta por el plan de estudios)

Decisión de diseño:
  Se usa un clasificador basado en reglas (keywords + patrones) en lugar
  de un clasificador neuronal por tres razones:
    1. Determinismo: misma query → misma categoría, facilitando depuración
    2. Latencia cero: no requiere inferencia adicional
    3. Transparencia: las reglas son interpretables y auditables

  El router devuelve TODAS las categorías que coinciden con la query,
  junto con un peso relativo. Si ninguna regla coincide, se devuelve
  None (sin filtrado), cayendo al retrieval estándar como fallback.

Categorías soportadas (11):
  plan_estudios, presentacion_grado, guias_docentes,
  tfg, calendario, horarios, movilidad, normativa, tramites,
  estudiantes, profesorado
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RouteResult:
    """Resultado del routing: categorías predichas con su peso."""
    categories: dict[str, float]   # {categoría: peso_boost}
    matched_rules: list[str]       # reglas que dispararon (para logging/debug)

    @property
    def is_routed(self) -> bool:
        """True si el router asignó al menos una categoría."""
        return len(self.categories) > 0


# ── Reglas de routing ────────────────────────────────────────────────────
# Cada regla es: (nombre, patrón_regex, {categoría: boost})
# El boost indica cuánto se multiplica el score de los chunks en esa categoría.
# Un boost de 1.0 = sin cambio; > 1.0 = favorece esa categoría.
#
# Las reglas se evalúan en orden. TODAS las que coinciden se acumulan.
# Esto permite que una query active múltiples categorías (e.g., "plazo TFG"
# activa tanto 'tfg' como 'calendario').

_ROUTING_RULES: list[tuple[str, re.Pattern, dict[str, float]]] = [
    # ── Plan de estudios / estructura del grado ──
    (
        "plan_estudios",
        re.compile(
            r"(?:plan\s+de\s+estudios|asignaturas?\s+(?:del?\s+)?(?:primer|segund|tercer|cuart|1|2|3|4)"
            r"|(?:primer|segund|tercer|cuart)\s+(?:curso|año)"
            r"|materias?\s+(?:del?\s+)?(?:grado|carrera)"
            r"|créditos?\s+(?:del?\s+)?grado"
            r"|estructura\s+del\s+grado"
            r"|optativas?\s+(?:del?\s+)?grado"
            r"|especialidades?\s+(?:del?\s+)?(?:grado|informática)"
            # Asignaturas/optativas asociadas a una especialidad (p.ej. «qué
            # asignaturas se imparten en cada especialidad»).
            r"|(?:asignaturas?|materias?|optativas?)\b[^|]{0,40}\bespecialidad"
            # Especialidad(es) de un curso (p.ej. «…en todas las especialidades
            # de cuarto»).
            r"|\bespecialidad(?:es)?\b[^|]{0,25}\b(?:primer|segund|tercer|cuart|\d\s*º|grado|inform[aá]tica)"
            # «cada/todas las especialidad(es)».
            r"|(?:cada|todas?\s+las?)\s+especialidad"
            # «qué/cuáles/cuántas especialidades (hay/tiene/existen)».
            r"|(?:qu[eé]|cu[aá]les|cu[aá]ntas)\s+(?:son\s+las\s+)?especialidades)",
            re.IGNORECASE,
        ),
        # Se añade un boost a 'horarios' porque el desglose por curso/cuatrimestre/
        # especialidad de 3º y 4º NO está en la página del plan de estudios, sino
        # en los registros de horarios (que sí distinguen 1er/2º cuatrimestre).
        {"plan_estudios": 2.5, "presentacion_grado": 1.5, "horarios": 1.3},
    ),

    # ── Cuatrimestre / semestre (qué asignaturas se imparten en cada uno) ──
    # La página del plan agrupa 3º/4º por especialidad SIN cuatrimestre; esa
    # información vive en los horarios. Por eso estas consultas se dirigen a
    # 'horarios' (fuente con el desglose) y a 'plan_estudios' (visión general).
    (
        "cuatrimestre",
        re.compile(r"(?:cuatrimestre|semestre)", re.IGNORECASE),
        {"horarios": 2.0, "plan_estudios": 1.5},
    ),

    # ── TFG ──
    (
        "tfg",
        re.compile(
            r"(?:tfg|trabajo\s+fin\s+de\s+grado|proyecto\s+fin\s+de\s+grado"
            r"|defensa\s+(?:del?\s+)?tfg|tribunal\s+tfg"
            r"|memoria\s+(?:del?\s+)?tfg|tutor\s+(?:del?\s+)?tfg"
            r"|convocatoria\s+(?:del?\s+)?tfg)",
            re.IGNORECASE,
        ),
        # Se añade boost a 'calendario' porque las FECHAS del TFG (plazos de
        # solicitud, asignación, entrega y DEFENSA) viven en el documento de
        # calendario (líneas "Calendario TFG 2025-2026 | …"), no en las guías ni
        # en la normativa. Sin este boost, una consulta de TFG sin palabra
        # temporal (p.ej. "¿qué días es la defensa del TFG?") no llegaba a
        # 'calendario' y la fecha quedaba fuera del contexto.
        {"tfg": 2.5, "normativa": 1.3, "calendario": 1.5},
    ),

    # ── Calendario / plazos / fechas ──
    (
        "calendario",
        re.compile(
            r"(?:cuándo|cuando|fecha|plazo|calendario|entrega|convocatoria"
            r"|examen|exámenes|periodo|matrícula|def(?:en|ien)\w*"
            r"|(?:junio|septiembre|noviembre|febrero)\s+\d{4})",
            re.IGNORECASE,
        ),
        {"calendario": 2.0, "tramites": 1.2},
    ),

    # ── Movilidad / Erasmus ──
    (
        "movilidad",
        re.compile(
            r"(?:erasmus|movilidad|intercambio|sicue|destino"
            r"|universidad\s+de\s+destino|beca\s+erasmus"
            r"|contrato\s+de\s+estudios|learning\s+agreement)",
            re.IGNORECASE,
        ),
        {"movilidad": 2.5},
    ),

    # ── Trámites / secretaría ──
    (
        "tramites",
        re.compile(
            r"(?:secretaría|secretaria|impreso|formulario|solicitud|solicitar"
            r"|certificado|título|automatrícula|anulación|traslado"
            r"|reconocimiento\s+de\s+créditos"
            r"|horario\s+(?:de\s+)?(?:la\s+)?(?:secretaría|secretaria|atención))",
            re.IGNORECASE,
        ),
        {"tramites": 2.0, "normativa": 1.2},
    ),

    # ── Normativa / reglamentos ──
    (
        "normativa",
        re.compile(
            r"(?:normativa|reglamento|directrices|resolución|regulación"
            r"|normas?\s+(?:de|del)\s+(?:evaluación|permanencia|tfg))",
            re.IGNORECASE,
        ),
        {"normativa": 2.5},
    ),

    # ── Prácticas en empresa ──
    (
        "practicas",
        re.compile(
            r"(?:prácticas?\s+(?:en\s+)?empresa|prácticas?\s+externas?"
            r"|curriculares|extracurriculares|convenio\s+de\s+prácticas)",
            re.IGNORECASE,
        ),
        {"tramites": 2.0, "estudiantes": 1.5},
    ),

    # ── Horarios de clase, aulas y franjas de asignaturas ──
    # Existe un documento dedicado de horarios (reconstruido por HorarioParser)
    # en la categoría 'horarios'. La regla NO debe capturar "horario de la
    # secretaría" (→ tramites) ni "horario/calendario de exámenes" (→ calendario),
    # por eso los lookaheads negativos excluyen 'secretar' y 'examen'.
    (
        "horarios",
        re.compile(
            r"(?:a\s+qué\s+hora"
            r"|(?:qué|en\s+qué)\s+aula"
            r"|clases?\s+de\s+(?!la\s+secretar)\w+"
            r"|horarios?\s+de\s+(?:las?\s+)?(?:clase|asignatura)"
            r"|horarios?\s+de\s+(?!.*(?:secretar|ex[áa]men))\w+)",
            re.IGNORECASE,
        ),
        {"horarios": 2.5, "calendario": 1.2},
    ),

    # ── Guías docentes (pregunta sobre una asignatura concreta) ──
    (
        "guia_docente",
        re.compile(
            r"(?:guía\s+docente|programa\s+de\s+(?:la\s+)?asignatura"
            r"|competencias?\s+de\s+(?:la\s+)?asignatura"
            r"|temario\s+de|bibliografía\s+de"
            r"|(?:profesor|profesora)\s+de\s+\w+)",
            re.IGNORECASE,
        ),
        {"guias_docentes": 2.0},
    ),

    # ── Profesorado ──
    (
        "profesorado",
        re.compile(
            r"(?:profesor|profesora|profesorado|docente|departamento"
            r"|despacho|tutorías?\s+(?:del?\s+)?profesor"
            r"|dirección\s+(?:del?\s+)?centro|decano|subdirector)",
            re.IGNORECASE,
        ),
        {"profesorado": 2.0},
    ),
]


# ── Diccionario de siglas de asignaturas del GII ────────────────────────
# Se usa para expandir queries como "clases de DI" → "Derecho Informático".
# Verificado manualmente a partir de los PDFs de horarios del GII 2025-26.

_SUBJECT_ABBREVIATIONS: dict[str, str] = {
    # Primer curso
    "CA": "Cálculo",
    "FFT": "Fundamentos Físicos y Tecnológicos",
    "FP": "Fundamentos de Programación",
    "FS": "Fundamentos del Software",
    "ALEM": "Álgebra Lineal y Estructuras Matemáticas",
    "EST": "Estadística",
    "IES": "Ingeniería, Empresa y Sociedad",
    "LMD": "Lógica y Métodos Discretos",
    "MP": "Metodología de la Programación",
    "TOC": "Tecnología y Organización de Computadores",
    # Segundo curso
    "EC": "Estructura de Computadores",
    "ED": "Estructura de Datos",
    "PDOO": "Programación y Diseño Orientado a Objetos",
    "SCD": "Sistemas Concurrentes y Distribuidos",
    "SO": "Sistemas Operativos",
    "ALG": "Algorítmica",
    "AC": "Arquitectura de Computadores",
    "FBD": "Fundamentos de Bases de Datos",
    "FIS": "Fundamentos de Ingeniería del Software",
    "IA": "Inteligencia Artificial",
    # Tercer curso
    "DDSI": "Diseño y Desarrollo de Sistemas de Información",
    "FR": "Fundamentos de Redes",
    "IG": "Informática Gráfica",
    "IS": "Ingeniería de Servidores",
    "MC": "Modelos de Computación",
    "SG": "Sistemas Gráficos",
    "SMM": "Sistemas Multimedia",
    # Cuarto curso (optativas comunes)
    "DI": "Derecho Informático",
    "EISI": "Ética, Informática y Sociedad de la Información",
    "CEGE": "Creación de Empresas y Gestión Emprendedora",
    "DIU": "Diseño de Interfaces de Usuario",
    "DGP": "Dirección y Gestión de Proyectos",
    "IV": "Infraestructura Virtual",
    "MDA": "Metodologías de Desarrollo Ágil",
    "MH": "Metaheurísticas",
    "NPI": "Nuevos Paradigmas de Interacción",
    "VC": "Visión por Computador",
    "RI": "Recuperación de Información",
    "TSI": "Técnicas de los Sistemas Inteligentes",
    "AA": "Aprendizaje Automático",
    "IC": "Ingeniería del Conocimiento",
    "SWAP": "Servidores Web de Altas Prestaciones",
    "DAI": "Desarrollo de Aplicaciones para Internet",
    "DS": "Desarrollo de Software",
    "SSO": "Seguridad en Sistemas Operativos",
    "PPR": "Programación Paralela",
    "PW": "Programación Web",
    "TW": "Tecnologías Web",
    "SIBW": "Sistemas de Información Basados en Web",
    "IN": "Inteligencia de Negocio",
    "TFG": "Trabajo Fin de Grado",
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


# Mapa inverso: nombre completo normalizado (sin tildes, en minúsculas) → sigla.
_NAME_TO_SIGLA: dict[str, str] = {
    _strip_accents(name.lower()): sigla
    for sigla, name in _SUBJECT_ABBREVIATIONS.items()
}
# Patrón de nombres completos (los más largos primero, para evitar que un nombre
# que es prefijo de otro capture de forma incorrecta).
_FULLNAME_RE = re.compile(
    r"\b(" + "|".join(
        re.escape(n) for n in sorted(_NAME_TO_SIGLA, key=len, reverse=True)
    ) + r")\b"
)
_SIGLA_TOKEN_RE = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,6})\b")


def detect_subject_siglas(text: str) -> list[str]:
    """Siglas de asignatura presentes en *text* (por sigla o por nombre completo).

    Reconoce tanto la sigla explícita en mayúsculas (``DI``, ``CA``) como el
    NOMBRE COMPLETO (``Derecho Informático``, ``Cálculo``), insensible a tildes.
    El soporte de nombres completos es clave: los usuarios escriben ``Cálculo``,
    no ``CA``; sin él la asignatura no se reconocía, ni para el boost por
    sigla exacta ni para el arrastre anafórico del historial.
    """
    found: list[str] = []
    for tok in _SIGLA_TOKEN_RE.findall(text):
        if tok in _SUBJECT_ABBREVIATIONS and tok not in found:
            found.append(tok)
    norm = _strip_accents(text.lower())
    for name in _FULLNAME_RE.findall(norm):
        sig = _NAME_TO_SIGLA[name]
        if sig not in found:
            found.append(sig)
    return found


def expand_abbreviations(query: str) -> str:
    """Enriquece la consulta con las siglas/nombres de las asignaturas.

    Dos pasos: (1) sustituye cada sigla conocida en mayúsculas por
    ``SIGLA (Nombre Completo)``; (2) si la consulta menciona una asignatura por
    su NOMBRE COMPLETO (``Cálculo``), añade su sigla ``(CA)``. Así, en ambos
    casos, la consulta contiene la sigla (para el matching estructurado y el
    boost exacto) y el nombre (para BM25 y embeddings).

    Ejemplos:
        "clases de DI"            → "clases de DI (Derecho Informático)"
        "en qué aula es Cálculo"  → "en qué aula es Cálculo (CA)"
    """
    words = query.split()
    expanded = False
    present: set[str] = set()
    for i, word in enumerate(words):
        clean = word.rstrip("?.,;:!")
        suffix = word[len(clean):]
        if clean.upper() in _SUBJECT_ABBREVIATIONS and clean == clean.upper() and len(clean) >= 2:
            full_name = _SUBJECT_ABBREVIATIONS[clean.upper()]
            words[i] = f"{clean} ({full_name}){suffix}"
            present.add(clean.upper())
            expanded = True
    result = " ".join(words)

    # Nombres completos → añadir la sigla si no figura ya.
    for sig in detect_subject_siglas(result):
        if sig not in present:
            result += f" ({sig})"
            present.add(sig)
            expanded = True

    if expanded:
        logger.info("Query expandida: '%s' → '%s'", query, result)
    return result


def route_query(query: str) -> RouteResult:
    """Clasifica *query* en categorías del corpus según reglas de keywords.

    Parameters
    ----------
    query : str
        Consulta del usuario en lenguaje natural.

    Returns
    -------
    RouteResult
        Categorías predichas con sus pesos de boost, o resultado vacío
        si ninguna regla coincide (fallback a retrieval sin filtro).
    """
    merged_categories: dict[str, float] = {}
    matched_rules: list[str] = []

    for rule_name, pattern, category_boosts in _ROUTING_RULES:
        if pattern.search(query):
            matched_rules.append(rule_name)
            for cat, boost in category_boosts.items():
                # Si múltiples reglas activan la misma categoría,
                # se toma el boost máximo (no se acumulan multiplicativamente)
                merged_categories[cat] = max(
                    merged_categories.get(cat, 0.0), boost,
                )

    result = RouteResult(
        categories=merged_categories,
        matched_rules=matched_rules,
    )

    if result.is_routed:
        logger.info(
            "Query routing: '%s' → categorías=%s (reglas: %s)",
            query[:60], dict(result.categories), result.matched_rules,
        )
    else:
        logger.debug(
            "Query routing: sin coincidencia para '%s' → fallback global",
            query[:60],
        )

    return result
