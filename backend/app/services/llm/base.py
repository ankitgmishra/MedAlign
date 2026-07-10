"""Abstract base for external LLM API clients.

Used by the Medical Investigator, LLM Judge, and Preference Generator
to call cloud-hosted or locally-served LLMs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from pydantic import BaseModel


class BaseLLMClient(ABC):
    """Interface for LLM API wrappers (Groq, Ollama, OpenAI, …)."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        response_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> str:
        """Send a chat completion request and return the raw text response.

        Parameters
        ----------
        messages:
            OpenAI-style message list.
        temperature:
            Sampling temperature.
        response_schema:
            If provided, request structured JSON output conforming to
            this Pydantic model's JSON schema.
        """
        ...
