"""Common domain types shared across all MedAlign modules.

Every dataset must be converted into ``InternalSample`` before it touches
any evaluation, investigation, or benchmarking logic.  This is the single
canonical internal representation described in the architecture spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported evaluation task types."""

    CLINICAL_QA = "clinical_qa"
    FREE_TEXT_QA = "free_text_qa"
    REPORT_GENERATION = "report_generation"
    SUMMARIZATION = "summarization"
    DIAGNOSIS = "diagnosis"
    TREATMENT_PLANNING = "treatment_planning"


class InternalSample(BaseModel):
    """Dataset-agnostic internal representation.

    Every dataset adapter produces a list of ``InternalSample`` instances.
    Evaluation modules **only** consume this type — never raw dataset rows.
    """

    task_type: TaskType = Field(
        description="The kind of evaluation task this sample represents.",
    )
    input: dict[str, Any] = Field(
        description=(
            "Task input.  For clinical_qa this includes 'question' and 'options'."
        ),
    )
    reference: dict[str, Any] = Field(
        description="Ground-truth reference.  Keys vary by task_type.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (dataset name, specialty, …).",
    )
    sample_id: Optional[str] = Field(
        default=None,
        description="Optional unique identifier for the sample.",
    )
