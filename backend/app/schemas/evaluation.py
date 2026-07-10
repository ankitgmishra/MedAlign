"""Evaluation report schemas — ported from ``03_benchmark_engine.py``.

These models define the aggregated statistics produced by the
``StatisticsEngine`` from a list of ``FailureRecord`` instances.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from app.schemas.training import RootCauseInsight


# ── Component metrics ────────────────────────────────────────────────────────


class OverallMetrics(BaseModel):
    """High-level aggregate metrics across all evaluated samples."""

    total_samples: int
    accuracy: Optional[float] = None
    exact_match_rate: Optional[float] = None
    unsafe_rate: float
    hallucination_rate: float
    guideline_adherence_rate: float
    average_claim_support_rate: float
    average_failure_confidence: float


class FailureProfile(BaseModel):
    """Distribution of failure types, capabilities, and severities."""

    primary_failure_distribution: Dict[str, int]
    capability_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]


class EvidenceMetrics(BaseModel):
    """Claim-level evidence statistics."""

    total_claims: int
    supported_claims: int
    unsupported_claims: int
    supported_claim_rate: float
    unsupported_claim_rate: float
    average_claims_per_prediction: float


class SafetyMetrics(BaseModel):
    """Safety-related aggregate metrics."""

    unsafe_predictions: int
    safe_predictions: int
    critical_failures: int
    high_severity_failures: int
    medium_severity_failures: int
    low_severity_failures: int
    top_harm_types: Dict[str, int]


class MissingInformationAnalysis(BaseModel):
    """Frequency of commonly missing clinical information."""

    most_common_missing_information: Dict[str, int]


class PositiveFindingsAnalysis(BaseModel):
    """Frequency of commonly noted positive findings."""

    most_common_positive_findings: Dict[str, int]


class FailureSignature(BaseModel):
    """Normalised distribution of dominant failure capabilities."""

    dominant_capabilities: Dict[str, float]


class TrainingRecommendation(BaseModel):
    """Actionable training recommendation derived from failure analysis."""

    capability: str
    frequency: int
    priority: str
    recommendation: str
    suggested_training: str
    expected_impact: str


# ── Top-level report ─────────────────────────────────────────────────────────


class EvaluationReport(BaseModel):
    """Full evaluation report aggregating all analytical dimensions."""

    overall_metrics: OverallMetrics
    failure_profile: FailureProfile
    evidence_metrics: EvidenceMetrics
    safety_metrics: SafetyMetrics
    missing_information_analysis: MissingInformationAnalysis
    positive_findings_analysis: PositiveFindingsAnalysis
    failure_signature: FailureSignature
    training_recommendations: List[TrainingRecommendation]
    training_insights: List[RootCauseInsight]
