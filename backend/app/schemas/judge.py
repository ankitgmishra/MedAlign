"""Judge schemas — Rich per-sample evaluation with clinical rubric scores."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class MedicalJudgeResponse(BaseModel):
    """Rich structured response from the LLM Judge — one record per sample."""

    # Core correctness
    correct: bool = False
    reasoning_score: float = Field(default=0.0, ge=0.0, le=1.0)
    medical_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    guideline_adherence: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)

    # Safety & hallucination flags
    unsafe: bool = False
    hallucination: bool = False

    # Qualitative explanation
    explanation: str = ""


class MedicalRubric(BaseModel):
    """Rubric for evaluation."""
    pass

