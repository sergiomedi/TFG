"""Lógica pura de memoria conversacional de la interfaz.

Se separa de ``app.py`` (acoplado a Streamlit) para poder testear de forma
aislada el saneado y la construcción del historial que se inyecta en el LLM.
Ningún símbolo de este módulo depende de ``streamlit``.

Funciones:
  - :func:`sanitize_for_history` — limpia una respuesta del asistente para
    reinyectarla como contexto de un turno posterior.
  - :func:`build_history` — selecciona y sanea los turnos previos a partir del
    historial de la sesión.

Constantes:
  - :data:`CLOSINGS` — cierres corteses de la persona GRAIA (también usados al
    sanear, para no arrastrarlos al historial).
"""

from __future__ import annotations

import re

# Cierres corteses de la persona GRAIA. Se añaden de forma determinista al final
# de cada respuesta (rotando para no ser monótonos) en ``app.py``, en lugar de
# delegarlos al modelo: así garantizamos el tono "de usted" sin arriesgar
# duplicaciones ni coletillas no informativas (que ``postprocess.clean_answer``
# sigue recortando del texto del modelo).
CLOSINGS: tuple[str, ...] = (
    "¿Puedo ayudarle en algo más?",
    "¿Hay algo más en lo que pueda echarle una mano?",
    "Si necesita cualquier otra cosa, aquí estoy.",
    "Cualquier otra duda, dígamelo sin problema.",
    "Encantado de ayudarle con lo que necesite.",
    "¿Quiere que mire alguna otra cosa por usted?",
)

# Patrón de marcadores de cita [n], usado al sanear el historial.
_MARKER_RE = re.compile(r"\[\d+\]")

# Marca de las respuestas de "sin información". Cubre la frase canónica del
# prompt ("no dispongo de información…") y las variantes que el modelo produce
# espontáneamente ("no hay información sobre…", "no tengo información/acceso…",
# "no se menciona…"). Detectarlas evita atribuir una fuente falsa (recuperación
# de citas) en respuestas que en realidad declaran no tener el dato.
_NO_INFO_RE = re.compile(
    r"no\s+(?:dispongo\s+de|tengo(?:\s+acceso\s+a)?|hay)\s+"
    r"(?:suficiente\s+)?(?:informaci[oó]n|datos)"
    r"|no\s+se\s+menciona"
    r"|no\s+(?:dispongo|cuento)\s+con\s+(?:informaci[oó]n|datos)",
    re.IGNORECASE,
)


# Preguntas "meta" sobre el propio asistente (identidad, capacidades). Sus
# respuestas describen a GRAIA, no proceden del corpus, por lo que NO deben
# recibir atribución de fuentes (evita citas alucinadas como la observada).
_META_RE = re.compile(
    r"\b(?:qui[eé]n\s+eres|qu[eé]\s+eres|c[oó]mo\s+te\s+llamas|"
    r"qu[eé]\s+(?:puedes|sabes)\s+hacer|para\s+qu[eé]\s+sirves|"
    r"qui[eé]n\s+te\s+(?:cre[oó]|hizo|program[oó]|desarroll[oó]))\b",
    re.IGNORECASE,
)


def is_meta_query(query: str) -> bool:
    """True si *query* pregunta por la identidad/capacidades del asistente."""
    return bool(_META_RE.search(query))


# Declinación de opinión/recomendación personal. Estas respuestas NO se basan en
# un documento del corpus, por lo que tampoco deben recibir atribución de fuentes
# (evita citas alucinadas como «296 OPTATIVAS» al preguntar «¿cuál es la mejor?»).
_DECLINE_RE = re.compile(
    r"no\s+puedo\s+(?:emitir|dar|ofrecer|opinar|recomendar|aconsejar|valorar|"
    r"posicionarme|decantarme)"
    r"|no\s+(?:doy|emito|hago)\s+(?:opiniones|recomendaciones|valoraciones)",
    re.IGNORECASE,
)


def is_subjective_decline(text: str) -> bool:
    """True si la respuesta declina dar una opinión/recomendación personal."""
    return bool(_DECLINE_RE.search(text))


def is_no_info_answer(text: str) -> bool:
    """True si *text* es una respuesta de "no dispongo de información".

    Permite a la interfaz omitir el cierre cortés en esas respuestas, que ya
    remiten a la instancia adecuada (Secretaría, etc.).
    """
    return bool(_NO_INFO_RE.search(text))


def sanitize_for_history(content: str) -> str:
    """Limpia una respuesta del asistente para reinyectarla como historial.

    Elimina el bloque de fuentes (pie ``---``), los marcadores de cita ``[n]``
    (que solo tienen sentido frente al contexto de SU turno) y los cierres
    corteses, dejando únicamente el contenido factual que da utilidad al modelo
    para resolver referencias en turnos posteriores.
    """
    # Cortar el bloque de fuentes verificadas (todo lo que sigue a "---").
    content = content.split("\n---")[0]
    # Quitar marcadores de cita.
    content = _MARKER_RE.sub("", content)
    # Quitar cierres corteses conocidos.
    for closing in CLOSINGS:
        content = content.replace(closing, "")
    # Normalizar espacios (incluido antes de signos de puntuación).
    content = re.sub(r"\s+([.,;:])", r"\1", content)
    content = re.sub(r"[ \t]{2,}", " ", content)
    return content.strip()


def build_history(
    messages: list[dict], max_turns: int
) -> list[dict[str, str]]:
    """Construye el historial a inyectar en el LLM desde la sesión.

    Excluye el saludo de bienvenida (primer mensaje) y el mensaje actual del
    usuario (último, ya añadido antes de invocar el pipeline). Sanea las
    respuestas del asistente y limita a los ``max_turns`` últimos turnos (pares
    usuario+asistente) para acotar el contexto y la latencia.

    Parameters
    ----------
    messages : list[dict]
        Historial completo de la sesión (``{"role", "content"}``), incluyendo
        la bienvenida inicial y la consulta actual al final.
    max_turns : int
        Número máximo de turnos previos a conservar. ``0`` → sin memoria.

    Returns
    -------
    list[dict[str, str]]
        Lista de mensajes saneados lista para pasar a ``OllamaClient``.
    """
    if max_turns <= 0 or len(messages) <= 2:
        return []

    history: list[dict[str, str]] = []
    for msg in messages[1:-1]:  # sin bienvenida ni consulta actual
        content = msg["content"]
        if msg["role"] == "assistant":
            content = sanitize_for_history(content)
        if content.strip():
            history.append({"role": msg["role"], "content": content})

    return history[-2 * max_turns:]
