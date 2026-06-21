"""scope_classifier — gate de ámbito (out-of-domain) previo a la recuperación.

Motivación (análisis de errores, Cap. 7): el umbral de similitud τ del retriever
NO basta para separar las consultas dentro de ámbito de las ajenas. Sobre un
corpus tan homogéneo (documentación académica de una sola Escuela), el modelo de
embeddings E5 produce similitudes comprimidas y relativamente altas incluso para
preguntas de otro dominio ("¿cuál es la capital de Francia?"). Esos chunks
espurios superan el umbral, el contexto no queda vacío y el LLM —pese a la
instrucción de abstención— responde con su conocimiento paramétrico y adjunta una
cita sin sentido. El umbral, por sí solo, no puede gobernar el ámbito.

Solución: un clasificador de ámbito que decide ANTES de recuperar si la consulta
trata sobre asuntos académicos de la ETSIIT/UGR. Si no, el sistema responde con
la abstención canónica SIN recuperar ni generar, evitando de raíz la respuesta
paramétrica y la cita espuria.

Decisión de diseño (vs. el resto de clasificadores por reglas del sistema):
  El ámbito "fuera de dominio" es ABIERTO e inagotable (geografía, deportes,
  política, cocina…); no es enumerable con reglas como sí lo son las categorías
  del corpus (:mod:`query_router`) o la intención de listado
  (:mod:`query_intent`). Por eso aquí se emplea el propio LLM local como
  clasificador binario de coste acotado (una sola palabra, decodificación
  greedy), que generaliza a temas no vistos. La identidad del asistente
  (preguntas "meta") y los seguimientos anafóricos quedan EXENTOS del gate por
  reglas deterministas, para no introducir falsos rechazos en conversación.
"""

from __future__ import annotations

import logging

from graia.generacion.ollama_client import OllamaClient
from graia.recuperacion.contextual_query import _ANAPHORIC_RE
from graia.recuperacion.query_router import route_query
from graia.interfaz.history import is_meta_query

logger = logging.getLogger(__name__)

# Frase canónica de abstención (idéntica a la del prompt de generación). Se
# centraliza aquí para que el gate la emita sin pasar por el LLM y para que los
# detectores de "no_info" (history._NO_INFO_RE, evaluate._REFUSAL_RE) la
# reconozcan como abstención correcta.
OUT_OF_SCOPE_ANSWER = (
    "No dispongo de información suficiente sobre este tema. "
    "Le recomiendo consultar con la Secretaría de la ETSIIT."
)

_SCOPE_SYSTEM = """\
Eres un clasificador de ámbito para GRAIA, el asistente académico de la ETSIIT \
(Escuela Técnica Superior de Ingenierías Informática y de Telecomunicación) de \
la Universidad de Granada. Tu ÚNICA tarea es decidir si la pregunta trata sobre \
asuntos académicos de la ETSIIT o la UGR.

Responde ACADEMICO si la pregunta versa sobre: horarios y aulas de clase, \
asignaturas, planes de estudio, créditos, especialidades o menciones, exámenes y \
convocatorias, calendario académico, profesorado y tutorías, Trabajo Fin de Grado \
(TFG), prácticas en empresa, movilidad o Erasmus, matrícula, becas, secretaría, \
normativa académica, o cualquier asunto similar de la Escuela o la Universidad.

Responde FUERA si la pregunta versa sobre cualquier otra cosa: geografía, \
política, deportes, clima, historia, cultura general, cálculos matemáticos \
sueltos, cocina, salud, ocio, o programación genérica no ligada a una asignatura \
concreta.

Responde EXCLUSIVAMENTE con una sola palabra en mayúsculas: ACADEMICO o FUERA. \
Sin explicaciones ni puntuación.

Ejemplos:
Pregunta: ¿A qué hora es Cálculo? -> ACADEMICO
Pregunta: ¿Cuándo es el examen de Inteligencia Artificial? -> ACADEMICO
Pregunta: ¿Qué asignaturas optativas hay en cuarto? -> ACADEMICO
Pregunta: ¿Cuál es la capital de Francia? -> FUERA
Pregunta: ¿Quién ganó la última Champions? -> FUERA
Pregunta: ¿Qué tiempo hará mañana en Granada? -> FUERA"""


def _classify_with_llm(query: str, client: OllamaClient) -> bool:
    """Arbitraje LLM del ámbito: ``True`` si es académico (ETSIIT/UGR).

    Red de seguridad para la banda dudosa del gate (ver :func:`is_in_scope`). En
    caso de error del clasificador se hace *fail-open* (``True``): es preferible
    un escape ocasional de algo fuera de ámbito a rechazar por error una consulta
    académica legítima.
    """
    try:
        result = client.generate(_SCOPE_SYSTEM, query.strip(), num_predict=8)
        verdict = result.text.strip().upper()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gate de ámbito omitido por error del clasificador: %s", exc)
        return True  # fail-open

    # "FUERA" tiene prioridad: solo se rechaza si el modelo lo afirma de forma
    # explícita; cualquier otra salida (incluida una ambigua) se trata como
    # ámbito académico (fail-open).
    in_scope = "FUERA" not in verdict
    logger.info("Gate de ámbito (LLM): '%s' -> %s (veredicto=%r)",
                query[:60], "ACADEMICO" if in_scope else "FUERA", verdict[:20])
    return in_scope


def is_in_scope(
    query: str,
    chunks: list,
    client: OllamaClient,
    *,
    enabled: bool = True,
    reranker_used: bool = True,
    high_margin: float = 2.0,
    low_margin: float = -1.0,
) -> bool:
    """Decide si *query* está dentro del ámbito académico de la ETSIIT/UGR.

    Gate HÍBRIDO en dos señales, por orden de coste:

    1. **Margen del cross-encoder (coste cero).** Cuando el reranking está
       activo, el score del mejor fragmento recuperado es una señal de dominio
       sorprendentemente limpia: se dispara a valores altos cuando algún
       fragmento responde de verdad a la pregunta y colapsa a negativo cuando
       ninguno lo hace (calibración empírica, Cap. 7: ~+5.7 en ámbito frente a
       ~−2.7 fuera). Por encima de ``high_margin`` → claramente en ámbito; por
       debajo de ``low_margin`` → claramente fuera. Ambos extremos se deciden
       sin coste adicional.
    2. **Clasificador LLM (red de seguridad).** Solo en la banda intermedia
       ``(low_margin, high_margin)``, donde el margen no es concluyente, se
       consulta al LLM (ver :func:`_classify_with_llm`).

    Exenciones deterministas (siempre en ámbito, sin evaluar señales): preguntas
    "meta" sobre el propio asistente y seguimientos anafóricos ("¿y en
    septiembre?"), para no romper la conversación ni la presentación de GRAIA.

    Parameters
    ----------
    query : str
        Consulta del usuario (forma original).
    chunks : list[RetrievedChunk]
        Fragmentos ya recuperados y reordenados; su ``similarity`` es el score
        del cross-encoder cuando ``reranker_used`` es ``True``.
    client : OllamaClient
        Cliente del LLM local (se reutiliza el modelo de generación).
    enabled : bool
        Si ``False``, el gate no actúa (devuelve ``True``).
    reranker_used : bool
        Si ``False``, el ``similarity`` no es comparable a los márgenes del
        cross-encoder; se omite la señal 1 y se arbitra directamente con el LLM.
    high_margin, low_margin : float
        Umbrales de la banda de decisión sobre el score del reranker.

    Returns
    -------
    bool
        ``True`` si procede recuperar/generar; ``False`` si procede la
        abstención canónica directa (:data:`OUT_OF_SCOPE_ANSWER`).
    """
    if not enabled:
        return True

    # Exenciones deterministas: identidad del asistente y seguimientos anafóricos.
    if is_meta_query(query) or _ANAPHORIC_RE.match(query):
        return True

    # Sin fragmentos recuperados no hay cobertura posible: fuera de ámbito.
    if not chunks:
        logger.info("Gate de ámbito: sin fragmentos recuperados para '%s' -> FUERA",
                    query[:60])
        return False

    # Señal 1: margen del cross-encoder (coste cero) en sus extremos.
    if reranker_used:
        top = max(c.similarity for c in chunks)
        if top >= high_margin:
            return True
        if top <= low_margin:
            # El auto-rechazo por margen bajo solo es seguro cuando el router NO
            # ha clasificado la consulta en ninguna categoría académica. El
            # enrutador (reglas deterministas) es una señal de dominio fuerte:
            # algunas consultas legítimas tienen margen bajo porque el
            # cross-encoder infravalora su fragmento (p.ej. "¿cuándo abre la
            # secretaría?", cuyo registro de horario puntúa negativo). Si la
            # consulta está enrutada, se difiere al clasificador LLM en lugar de
            # rechazarla de plano, evitando falsos rechazos sin coste de latencia
            # en el caso común (consultas ajenas, que no enrutan, se siguen
            # rechazando aquí).
            if not route_query(query).is_routed:
                logger.info("Gate de ámbito (margen=%.3f<=%.2f, sin enrutar): '%s' -> FUERA",
                            top, low_margin, query[:60])
                return False
            logger.info("Gate de ámbito: margen bajo (%.3f) pero consulta enrutada; "
                        "arbitra LLM para '%s'", top, query[:60])
        else:
            logger.info("Gate de ámbito: margen dudoso (%.3f) para '%s' -> arbitra LLM",
                        top, query[:60])

    # Señal 2: arbitraje LLM (banda dudosa, consulta enrutada con margen bajo, o
    # reranker desactivado).
    return _classify_with_llm(query, client)
