"""Abstract base for inference services.

An inference service wraps model loading and text generation so that
evaluation modules don't need to know whether inference runs locally
(transformers + PEFT) or via an API (Groq, OpenAI, Ollama).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseInferenceService(ABC):
    """Interface for all inference backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[dict[str, str]],
        **kwargs: Any,
    ) -> tuple[str, float]:
        """Generate a completion and return ``(text, latency_seconds)``.

        Parameters
        ----------
        messages:
            Chat-style message list ``[{"role": ..., "content": ...}, ...]``.
        **kwargs:
            Backend-specific options (``max_new_tokens``, ``temperature``, …).
        """
        ...

    @abstractmethod
    def load_model(self, **kwargs: Any) -> None:
        """Load or initialise the underlying model / client."""
        ...
