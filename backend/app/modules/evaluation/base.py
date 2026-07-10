"""Abstract base for evaluators.

An evaluator takes a prediction and ground truth, and returns a structured
evaluation result.  Concrete implementations include the LLM Judge
(``MedicalLLMJudge``) and simple accuracy scoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEvaluator(ABC):
    """Interface for all evaluation strategies."""

    @abstractmethod
    def evaluate(
        self,
        question: str,
        ground_truth: str,
        prediction: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate a single prediction against its ground truth.

        Returns a dict with evaluation results whose structure depends
        on the concrete evaluator.
        """
        ...
