"""OllamaClient — interfaz con el LLM local vía Ollama.

Implementa el componente de generación descrito en la Sección 5.9.2 del diseño:
  - Chat completion con streaming de tokens (para la interfaz Streamlit)
  - Modo no-streaming (para evaluación batch)
  - Parámetros configurables: model, temperature, top_p, max_tokens, stop
  - Memoria conversacional opcional vía el parámetro ``history``
  - Manejo de errores de conexión y timeout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Sequence

import ollama

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Resultado de una generación del LLM."""
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OllamaClient:
    """Cliente para Ollama que expone chat completion con y sin streaming.

    Parameters
    ----------
    model : str
        Identificador del modelo (e.g. ``llama3.1:8b-instruct-q4_K_M``).
    temperature : float
        Temperatura de muestreo (0.0–2.0). Valores bajos → más determinista.
    top_p : float
        Nucleus sampling.
    max_tokens : int
        Máximo de tokens generados por respuesta.
    stop : Sequence[str]
        Secuencias de parada adicionales.
    host : str
        Endpoint de Ollama (por defecto ``http://localhost:11434``).
    """

    def __init__(
        self,
        model: str = "llama3.1:8b-instruct-q4_K_M",
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stop: Sequence[str] = ("</respuesta>",),
        host: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = list(stop)
        self.client = ollama.Client(host=host)
        logger.info("OllamaClient inicializado: model=%s, T=%.2f", model, temperature)

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Compone la lista de mensajes: sistema + historial + consulta actual.

        El *history* opcional (turnos previos ``{"role", "content"}``) se inserta
        entre el prompt de sistema y el mensaje actual, dotando al modelo de
        memoria conversacional para resolver referencias anafóricas. En
        evaluación batch se deja a ``None`` (cada consulta independiente).
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[dict[str, str]] | None = None,
        num_predict: int | None = None,
    ) -> GenerationResult:
        """Genera una respuesta completa (sin streaming).

        Adecuado para evaluación batch donde no se necesita mostrar tokens
        incrementalmente. Acepta *history* opcional (ver :meth:`_build_messages`).
        El parámetro *num_predict* permite acotar la longitud de ESTA generación
        (p.ej. tareas de clasificación que solo necesitan una palabra), sin
        alterar el ``max_tokens`` por defecto del cliente.
        """
        messages = self._build_messages(system_prompt, user_message, history)
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": num_predict if num_predict is not None else self.max_tokens,
                    "stop": self.stop,
                },
            )
            text = response["message"]["content"]
            # Ollama devuelve contadores de tokens en el campo opcional
            pt = response.get("prompt_eval_count", 0)
            ct = response.get("eval_count", 0)

            logger.debug(
                "Generación completada: %d tokens prompt, %d tokens respuesta",
                pt, ct,
            )
            return GenerationResult(
                text=text,
                model=self.model,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
            )
        except Exception as exc:
            logger.error("Error en generación Ollama: %s", exc)
            raise

    def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Genera tokens de forma incremental (streaming).

        Acepta *history* opcional (ver :meth:`_build_messages`) para dotar de
        memoria conversacional a la interfaz Streamlit.

        Yields
        ------
        str
            Fragmentos de texto conforme el LLM los produce.
        """
        messages = self._build_messages(system_prompt, user_message, history)
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens,
                    "stop": self.stop,
                },
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except Exception as exc:
            logger.error("Error en streaming Ollama: %s", exc)
            raise
