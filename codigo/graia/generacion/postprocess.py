"""postprocess — limpieza determinista de la respuesta del LLM.

Red de seguridad frente a modelos pequeños (p.ej. llama3.1:8b) que, pese al
prompt, tienden a (a) exponer su razonamiento con preámbulos del tipo
«La pregunta del usuario es…», «Tras revisar los fragmentos…», y (b) cerrar con
coletillas no informativas («Espero que esto le ayude», «Si tiene otra
pregunta…»). Esta función recorta esas líneas de forma conservadora, sin tocar
el contenido factual ni los marcadores de cita ``[n]``.

Es determinista y barata: se aplica tras la generación, antes de mostrar y
guardar la respuesta. No sustituye al prompt; lo complementa.
"""

from __future__ import annotations

import re

# Preámbulos de "pensamiento en voz alta" al inicio de la respuesta.
_PREAMBLE = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*la pregunta del usuario\b.*$",
    r"^\s*seg[uú]n (?:los|el|la)\s+fragmento.*$",
    r"^\s*(?:tras|despu[eé]s de)\s+revisar.*$",
    r"^\s*para responder( a)? (?:a )?(?:esta|la) pregunta.*$",
    r"^\s*(?:voy a|necesito)\s+(?:buscar|revisar|analizar).*$",
    r"^\s*bas[aá]ndome en (?:el|los) (?:contexto|fragmento).*$",
    r"^\s*(?:analizando|revisando) (?:el|los) (?:contexto|fragmento).*$",
    # Meta-comentario: el modelo habla de "la respuesta" o de "los fragmentos"
    # en vez de darla directamente (fuga de proceso). Solo se recortan estas
    # líneas cuando son íntegramente meta (van a su propia línea).
    r"^\s*la respuesta (?:a (?:tu|la|su|esta) pregunta|final)\b[^:]*:?\s*$",
    r"^\s*(?:la informaci[oó]n|esto) se (?:puede |podr[ií]a )?(?:encontrar|encuentr[ae]).*fragmento.*$",
    r"^\s*(?:seg[uú]n|en|de acuerdo con) (?:el|los) fragmento[s]?\s*\[?\d*\]?\b.*$",
]]

# Muletilla de "hedging" al inicio: el modelo encabeza la respuesta con un
# preámbulo vacío de contenido ("Según la información proporcionada, …", "De
# acuerdo con la información…", "Basándome en…"). Se recorta SOLO la cláusula
# inicial hasta la primera coma/dos puntos, conservando el resto de la frase.
# NO afecta a "Según el Calendario…/la Normativa…", que sí aportan la fuente.
_LEADING_HEDGE_RE = re.compile(
    r"^\s*(?:"
    r"seg[uú]n\s+(?:la\s+|lo\s+|los\s+|el\s+)?(?:informaci[oó]n|datos?|contexto|"
    r"indicado|mencionado|proporcionad[oa]s?|disponible)[^,.:]*"
    r"|de\s+acuerdo\s+con\s+(?:la\s+)?informaci[oó]n[^,.:]*"
    r"|bas[aá]ndome\s+en\s+(?:la\s+)?informaci[oó]n[^,.:]*"
    r")[,:.]?\s+",
    re.IGNORECASE,
)

# Marcadores de viñeta que el modelo escribe en línea ('*', '+', '•') sin saltos,
# por lo que Markdown no los renderiza como lista (se ven los asteriscos).
_INLINE_BULLET_RE = re.compile(r"[ \t]*[*+•]\s+")

# Coletillas finales no informativas.
_TRAILING = [re.compile(p, re.IGNORECASE) for p in [
    r"^\s*espero que (?:esto|esta|le|te).*$",
    r"^\s*si (?:tiene|tienes|necesita|necesitas|desea|deseas|quiere|quieres)\b.*$",
    r"^\s*(?:no dude|no dudes) en (?:preguntar|consultar).*$",
    r"^\s*qued[oó] a su disposici[oó]n.*$",
    r"^\s*puedo (?:ayudarte|ayudarle|sugerir).*$",
    r"^\s*(?:tambi[eé]n )?puedes? (?:consultar|revisar|buscar).*$",
    # Meta-comentario al final (cierres que hablan de los fragmentos/la respuesta).
    r"^\s*la respuesta final es\b.*$",
    r"^\s*(?:esto |esta informaci[oó]n )?se (?:puede |podr[ií]a )?(?:encontrar|encuentr[ae]).*fragmento.*$",
    r"^\s*ambos fragmentos\b.*$",
]]

# Frase canónica de abstención (única y fija). Debe coincidir con la del prompt
# de generación y con scope_classifier.OUT_OF_SCOPE_ANSWER, para que los
# detectores de "no_info" la reconozcan y la interfaz omita cierre y fuentes.
CANONICAL_ABSTENTION = (
    "No dispongo de información suficiente sobre este tema. "
    "Le recomiendo consultar con la Secretaría de la ETSIIT."
)

# Detector de abstención AL INICIO de la respuesta. Reconoce las múltiples formas
# en que el modelo declara no tener el dato ("no dispongo de…", "no hay
# información…", "no puedo proporcionar información…", "no se encuentra
# información…"), opcionalmente precedidas de una breve disculpa. Exige la palabra
# "información"/"datos" cerca para NO confundir respuestas parciales legítimas
# (p.ej. "No hay clase el viernes…"), que SÍ aportan dato y no deben colapsarse.
_ABSTENTION_START_RE = re.compile(
    r"^\s*(?:(?:lo\s+siento|lamentablemente|disculpe|por\s+desgracia)[,.\s]*(?:pero)?[,.\s]*)?"
    r"(?:"
    r"no\s+(?:dispongo\s+de|tengo(?:\s+acceso\s+a)?|hay|cuento\s+con|"
    r"puedo\s+(?:proporcionar|ofrecer|dar|facilitar|responder(?:le)?\s+a))"
    r"\s+(?:[\wáéíóúñ]+\s+){0,4}?(?:informaci[oó]n|datos)"
    r"|no\s+se\s+(?:encuentra|dispone\s+de|proporciona|especifica|menciona|indica|facilita)"
    r"\s+(?:suficiente\s+)?(?:informaci[oó]n|datos)"
    r"|sin\s+informaci[oó]n\s+(?:suficiente|disponible)"
    r")",
    re.IGNORECASE,
)


def is_abstention(text: str) -> bool:
    """True si *text* es (empieza siendo) una abstención de "no hay dato".

    Fuente única de verdad para detectar abstención, compartida por la interfaz
    (omitir cierre/fuentes) y por la evaluación (métrica de rechazo). Evita los
    falsos positivos de buscar "no se menciona" suelto dentro de una respuesta
    válida (p.ej. un listado que indica "[no se menciona]" para un grupo).
    """
    return bool(_ABSTENTION_START_RE.match(text or ""))


# Meta-comentario en PREFIJO de una frase: "La respuesta (a la pregunta) es: X".
# Se recorta solo la cláusula meta inicial, conservando el dato X que la sigue.
_LEADING_META_RE = re.compile(
    r"^\s*la\s+respuesta\s+(?:a\s+(?:la|esta|tu|su)\s+pregunta\s+)?(?:es|ser[ií]a)\s*:?\s+",
    re.IGNORECASE,
)


def clean_answer(text: str) -> str:
    """Recorta preámbulos de razonamiento y coletillas finales de *text*."""
    # Normalización de abstención: si la respuesta DECLARA desde el inicio no
    # disponer del dato (en cualquiera de sus variantes), se colapsa a la frase
    # canónica única. Elimina el parloteo y las fugas de razonamiento típicas de
    # las abstenciones del modelo pequeño ("...la pregunta parece relacionada
    # con el Reglamento...", "...¿hay algo más en lo que pueda ayudarte?").
    if _ABSTENTION_START_RE.match(text):
        return CANONICAL_ABSTENTION

    lines = text.split("\n")

    # Quitar preámbulos al principio (líneas iniciales que casen)
    start = 0
    while start < len(lines):
        ln = lines[start].strip()
        if ln == "" or any(p.match(ln) for p in _PREAMBLE):
            start += 1
        else:
            break

    # Quitar coletillas al final
    end = len(lines)
    while end > start:
        ln = lines[end - 1].strip()
        if ln == "" or any(p.match(ln) for p in _TRAILING):
            end -= 1
        else:
            break

    cleaned = "\n".join(lines[start:end]).strip()
    if not cleaned:
        # Si por lo que sea se vacía (todo eran coletillas), devolver el original
        return text.strip()

    # Recortar la muletilla de hedging inicial (cláusula), recapitalizando.
    stripped = _LEADING_HEDGE_RE.sub("", cleaned, count=1)
    if stripped and stripped != cleaned:
        cleaned = stripped[0].upper() + stripped[1:]

    # Recortar el meta-prefijo "La respuesta (a la pregunta) es:" conservando el dato.
    stripped = _LEADING_META_RE.sub("", cleaned, count=1)
    if stripped and stripped != cleaned:
        # Recapitalizar solo si el resto empieza por letra y NO es una URL/scheme
        # (evita convertir "http://…" en "Http://…").
        first_word = stripped.split(maxsplit=1)[0] if stripped.split() else ""
        if stripped[0].isalpha() and "://" not in first_word:
            cleaned = stripped[0].upper() + stripped[1:]
        else:
            cleaned = stripped

    return cleaned


def normalize_markdown_lists(text: str) -> str:
    """Convierte viñetas en línea ('*', '+', '•', y separadores ' - ') en items
    de lista con salto de línea, para que Markdown los renderice como lista en
    lugar de mostrar los asteriscos literales.

    Conserva intactos los rangos horarios (``12:30-13:30``) y los marcadores de
    cita ``[n]``: solo trata guiones rodeados de espacios tras fin de cláusula.
    """
    if not text:
        return text
    # Viñetas '*', '+', '•' (con espacio detrás) → item de lista en su línea.
    out = _INLINE_BULLET_RE.sub("\n- ", text)
    # Separador ' - ' tras fin de cláusula (". - ", ": - ", ") - ") → viñeta.
    out = re.sub(r"([.:)])\s+-\s+(?=[0-9A-Za-zÁÉÍÓÚÑáéíóú])", r"\1\n- ", out)
    # Eliminar viñetas VACÍAS (líneas con solo el marcador '-', '*', '+', '•' y
    # nada de contenido): el modelo a veces deja una viñeta colgando al final o
    # tras recortar una frase, que se renderizaría como un '*' suelto.
    out = re.sub(r"(?m)^[ \t]*[-*+•][ \t]*$\n?", "", out)
    # Normalizar espacios/saltos sobrantes.
    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()
