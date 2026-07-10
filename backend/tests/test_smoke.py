"""Smoke tests — verify imports and basic functionality.

These run without any ML dependencies and confirm the architecture compiles.
"""

from __future__ import annotations


def test_settings_load() -> None:
    """Settings should load without error."""
    from app.config.settings import get_settings

    settings = get_settings()
    assert settings.app_name == "MedAlign"
    assert settings.app_version == "0.1.0"
    assert settings.environment in ("development", "staging", "production")


def test_internal_sample_creation() -> None:
    """InternalSample should be constructible with minimal fields."""
    from app.schemas.common import InternalSample, TaskType

    sample = InternalSample(
        task_type=TaskType.CLINICAL_QA,
        input={"question": "What is the diagnosis?", "options": {"A": "X", "B": "Y"}},
        reference={"correct_option": "A", "correct_answer": "X"},
    )
    assert sample.task_type == TaskType.CLINICAL_QA
    assert sample.sample_id is None


def test_investigation_report_schema() -> None:
    """InvestigationReport should validate a well-formed dict."""
    from app.schemas.investigation import InvestigationReport

    data = {
        "prediction_summary": {"concise_prediction": "Model predicts A."},
        "observations": [{"source": "Prediction", "text": "Said A."}],
        "failure_tags": ["Clinical Reasoning", "Diagnosis"],
        "inferences": [
            {
                "type": "Reasoning",
                "reasoning": "A is wrong.",
                "supporting_observations": ["Said A."],
            }
        ],
        "claims": [
            {
                "claim": "Answer is A",
                "supported": False,
                "support_source": "None",
                "explanation": "Ground truth is B.",
            }
        ],
        "positive_findings": [],
        "missing_information": ["Differential diagnosis"],
        "safety": {
            "unsafe": False,
            "severity": "None",
            "explanation": "No safety concern.",
        },
        "guideline": {
            "follows_guideline": True,
            "explanation": "OK.",
        },
        "hallucination": False,
        "failure_hypotheses": [
            {
                "category": "Knowledge Gap",
                "subcategory": "Pharmacology",
                "severity": "Medium",
                "confidence": 0.8,
                "explanation": "Lacks knowledge.",
            }
        ],
        "primary_failure": {
            "category": "Knowledge Gap",
            "confidence": 0.8,
            "rationale": "Dominant failure.",
        },
        "clinical_summary": "The model answered A instead of B.",
    }
    report = InvestigationReport.model_validate(data)
    assert report.hallucination is False
    assert len(report.claims) == 1


def test_failure_record_schema() -> None:
    """FailureRecord should validate correctly."""
    from app.schemas.investigation import FailureRecord

    record = FailureRecord(
        primary_failure="Knowledge Gap",
        failure_confidence=0.8,
        unsafe=False,
        severity="None",
        hallucination=False,
        guideline_followed=True,
        total_claims=3,
        supported_claims=2,
        unsupported_claims=1,
        claim_support_rate=0.67,
        summary="Test summary.",
    )
    assert record.claim_support_rate == 0.67


def test_preference_example_schema() -> None:
    """PreferenceExample should validate correctly."""
    from app.schemas.preference import PreferenceExample

    example = PreferenceExample(
        prompt="What is the diagnosis?",
        chosen="B is correct because...",
        rejected="A is correct.",
        failure_capabilities=["Clinical Reasoning"],
        primary_failure="Knowledge Gap",
        severity="Medium",
    )
    assert example.chosen.startswith("B")


def test_api_response_format() -> None:
    """api_response should return the standard envelope."""
    from app.utils.response import api_response

    resp = api_response(message="OK", data={"key": "value"})
    assert resp["success"] is True
    assert resp["errors"] is None
    assert resp["data"]["key"] == "value"


def test_exception_hierarchy() -> None:
    """All domain errors should inherit from ApplicationError."""
    from app.utils.exceptions import (
        ApplicationError,
        BenchmarkError,
        DatasetError,
        DPOError,
        EvaluationError,
        InferenceError,
        InvestigationError,
        PreferenceError,
        ValidationError,
    )

    for cls in (
        ValidationError,
        DatasetError,
        InferenceError,
        EvaluationError,
        InvestigationError,
        BenchmarkError,
        PreferenceError,
        DPOError,
    ):
        err = cls("test")
        assert isinstance(err, ApplicationError)
        d = err.to_dict()
        assert "error" in d
        assert "message" in d


def test_extract_option_letter() -> None:
    """extract_option_letter should find A-D from standard format."""
    from app.utils.text import extract_option_letter

    assert extract_option_letter("The correct answer is B.\nExplanation...") == "B"
    assert extract_option_letter("Some random text") is None
    assert extract_option_letter(None) is None


def test_failure_attribution_engine() -> None:
    """FailureAttributionEngine should convert report to record."""
    from app.modules.investigator.failure_attribution import FailureAttributionEngine
    from app.schemas.investigation import InvestigationReport

    report = InvestigationReport.model_validate({
        "prediction_summary": {"concise_prediction": "Test."},
        "observations": [],
        "failure_tags": ["Diagnosis"],
        "inferences": [],
        "claims": [
            {
                "claim": "X",
                "supported": True,
                "support_source": "Question",
                "explanation": "OK.",
            }
        ],
        "positive_findings": ["Good structure"],
        "missing_information": [],
        "safety": {"unsafe": False, "severity": "None", "explanation": "Safe."},
        "guideline": {"follows_guideline": True, "explanation": "Compliant."},
        "hallucination": False,
        "failure_hypotheses": [
            {
                "category": "A",
                "subcategory": "B",
                "severity": "Low",
                "confidence": 0.5,
                "explanation": "Minor.",
            }
        ],
        "primary_failure": {
            "category": "A",
            "confidence": 0.5,
            "rationale": "Primary.",
        },
        "clinical_summary": "Summary.",
    })

    engine = FailureAttributionEngine()
    record = engine.analyze(report, sample_id="s1")
    assert record.sample_id == "s1"
    assert record.claim_support_rate == 1.0
