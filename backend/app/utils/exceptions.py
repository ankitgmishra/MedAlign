"""Structured exception hierarchy for MedAlign.

Every domain module raises a specific subclass of ``ApplicationError``.
The API layer catches these and maps them to consistent JSON responses.
"""

from __future__ import annotations

from typing import Any, Optional


class ApplicationError(Exception):
    """Base class for all MedAlign application errors."""

    def __init__(
        self,
        message: str = "An application error occurred.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(ApplicationError):
    """Raised when input validation fails (schemas, parameters, etc.)."""

    def __init__(
        self,
        message: str = "Validation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class DatasetError(ApplicationError):
    """Raised when dataset loading, parsing, or adaptation fails."""

    def __init__(
        self,
        message: str = "Dataset operation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class InferenceError(ApplicationError):
    """Raised when model inference or external LLM calls fail."""

    def __init__(
        self,
        message: str = "Inference failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class EvaluationError(ApplicationError):
    """Raised when an evaluation pipeline step fails."""

    def __init__(
        self,
        message: str = "Evaluation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class InvestigationError(ApplicationError):
    """Raised when the Medical Investigator pipeline fails."""

    def __init__(
        self,
        message: str = "Investigation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class BenchmarkError(ApplicationError):
    """Raised during benchmark aggregation or report generation."""

    def __init__(
        self,
        message: str = "Benchmark operation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class PreferenceError(ApplicationError):
    """Raised when preference-dataset generation fails."""

    def __init__(
        self,
        message: str = "Preference generation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)


class DPOError(ApplicationError):
    """Raised when DPO training pipeline fails."""

    def __init__(
        self,
        message: str = "DPO training failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details=details)
