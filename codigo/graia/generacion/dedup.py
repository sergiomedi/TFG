"""dedup — eliminación determinista de frases redundantes en la respuesta.

Red de seguridad frente a modelos pequeños (p.ej. llama3.1:8b) que, pese a la
instrucción explícita del prompt de no repetir información, a veces enuncian el
mismo dato dos veces citando fuentes distintas (p.ej. "atiende de 9 a 14 [1]"
y "el horario de atención es de 9 a 14 [2]"). El sistema de citas ya deduplica
las *fuentes* por URL, pero no el *contenido*: este módulo cierra ese hueco.

Estrategia (determinista, barata, interpretable):
  1. Segmentar la respuesta en frases.
  2. Para cada frase, calcular su conjunto de *tokens salientes* (sin siglas de
     cita, sin tildes, sin palabras vacías, con las horas ``HH:MM`` normalizadas
     a la hora ``HH`` para que "9:00" y "9" coincidan).
  3. Considerar una frase redundante si su solapamiento (índice de Jaccard) con
     una frase ya conservada supera un umbral, exigiendo además un mínimo de
     tokens compartidos para no fusionar frases cortas heterogéneas.
  4. Conservar la PRIMERA aparición (con su marcador de cita) y descartar las
     redundantes (y, con ellas, su cita sobrante).

Limitación asumida (Cap. 7): es un filtro léxico, no semántico; con un umbral
conservador prioriza la precisión (no fusionar datos distintos) sobre el
recall (puede dejar pasar paráfrasis muy divergentes). El umbral es configurable.
"""

from __future__ import annotations

import re
import unicodedata

# Palabras vacías frecuentes en español (lista mínima; no se busca exhaustividad,
# solo reducir el ruido de conectores en el cálculo de solapamiento).
_STOPWORDS: frozenset[str] = frozenset({
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "en", "y", "o", "a", "al", "es", "son", "para", "por", "con", "su", "sus",
    "lo", "se", "que", "como", "más", "mas", "este", "esta", "estos", "estas",
    "the", "of", "horas", "hora",
})

# Segmentación de frases: tras . ! ? seguidos de espacio. Las horas "9:00" no
# llevan punto, por lo que no rompen la segmentación.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[\d+\]")
_TIME_RE = re.compile(r"(\d{1,2}):\d{2}")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Mínimo de tokens salientes compartidos para arriesgar una fusión. Evita que
# dos frases cortas ("Sí." / "Correcto.") se consideren redundantes.
_MIN_SHARED = 3


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def salient_tokens(text: str) -> set[str]:
    """Conjunto de tokens salientes de un texto (sin tildes, sin palabras vacías,
    horas ``HH:MM`` normalizadas a ``HH`` y marcadores ``[n]`` eliminados).

    Función pública reutilizada por la deduplicación de frases y por la
    recuperación de citas (:mod:`citation_validator`).
    """
    s = _strip_accents(text.lower())
    s = _CITATION_RE.sub(" ", s)          # quitar marcadores [n]
    s = _TIME_RE.sub(r"\1", s)            # 9:00 -> 9 (normalizar horas)
    tokens = _TOKEN_RE.findall(s)
    return {
        t for t in tokens
        if t not in _STOPWORDS and (t.isdigit() or len(t) >= 3)
    }


# Alias interno conservado para compatibilidad con el resto del módulo.
_salient_tokens = salient_tokens


def _is_redundant(
    tokens: set[str], kept_token_sets: list[set[str]], threshold: float
) -> bool:
    """True si *tokens* solapa por encima del umbral con alguna frase conservada.

    Discriminador clave: dos frases solo son redundantes si comparten los MISMOS
    tokens numéricos (horas, aulas, fechas). Esto distingue "el mismo dato
    repetido" (mismos números → redundante) de "datos paralelos con valores
    distintos" (p.ej. el horario de cada grupo, con horas y aulas diferentes →
    NO redundante), evitando que el deduplicador colapse listas legítimas.
    """
    nums = {t for t in tokens if t.isdigit()}
    for kept in kept_token_sets:
        if not kept or not tokens:
            continue
        # Si los conjuntos de números difieren, son datos distintos: no fusionar.
        if nums != {t for t in kept if t.isdigit()}:
            continue
        shared = tokens & kept
        if len(shared) < _MIN_SHARED:
            continue
        union = tokens | kept
        jaccard = len(shared) / len(union)
        if jaccard >= threshold:
            return True
    return False


def deduplicate_sentences(text: str, threshold: float = 0.5) -> str:
    """Elimina frases redundantes de *text*, conservando la primera aparición.

    Parameters
    ----------
    text : str
        Respuesta generada (con marcadores de cita ``[n]`` aún presentes).
    threshold : float
        Umbral de Jaccard (0–1) sobre tokens salientes a partir del cual dos
        frases se consideran redundantes. Valores altos → más conservador
        (menos fusiones, mayor precisión).

    Returns
    -------
    str
        Respuesta sin las frases redundantes. Si ``text`` no tiene frases
        repetidas, se devuelve intacto.
    """
    if not text or not text.strip():
        return text

    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    kept_sentences: list[str] = []
    kept_token_sets: list[set[str]] = []

    for sentence in sentences:
        if not sentence.strip():
            continue
        tokens = _salient_tokens(sentence)
        if _is_redundant(tokens, kept_token_sets, threshold):
            continue  # frase redundante: se descarta (con su cita sobrante)
        kept_sentences.append(sentence.strip())
        kept_token_sets.append(tokens)

    return " ".join(kept_sentences)
