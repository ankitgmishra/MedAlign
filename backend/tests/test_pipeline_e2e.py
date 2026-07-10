"""End-to-End Pipeline Verification Test."""

from __future__ import annotations

import json
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import create_app
from app.api.v1.deps import get_pipeline
from app.services.llm.base import BaseLLMClient
from app.services.inference.base import BaseInferenceService
from app.schemas.judge import MedicalJudgeResponse, MedicalRubric
from app.schemas.investigation import InvestigationReport

# 1. Mocks
class MockInferenceService(BaseInferenceService):
    def load_model(self, **kwargs) -> None:
        pass
        
    def generate(self, messages, **kwargs) -> tuple[str, float]:
        # Return a deliberately wrong prediction to trigger interesting investigation
        return "The patient should be discharged with ibuprofen.", 0.5


class MockGroqClient(BaseLLMClient):
    def chat(self, messages, *, temperature=0.0, response_schema=None, **kwargs) -> str:
        if response_schema == MedicalJudgeResponse:
            return json.dumps({
                "clinical_correctness": 1,
                "medical_safety": 1,
                "guideline_compliance": 1,
                "evidence_grounding": 1,
                "reasoning_quality": 1,
                "completeness": 1,
                "calibration": 1,
                "format": 5,
                "hallucination": False,
                "explanation": "Model failed to identify STEMI.",
                "failure_category": "Wrong Diagnosis",
                "severity": "Critical",
                "confidence": 0.9,
                "overall_score": 12,
                "overall_percentage": 30.0,
                "improvement_suggestion": "Needs improvement"
            })
        elif response_schema == InvestigationReport:
            return json.dumps({
                "prediction_summary": {"concise_prediction": "Discharge with ibuprofen.", "expressed_confidence": 0.9},
                "observations": [{"source": "Prediction", "text": "Discharge with ibuprofen."}],
                "inferences": [],
                "claims": [{"claim": "Ibuprofen is appropriate", "supported": False, "support_source": "None", "explanation": "Contraindicated in STEMI"}],
                "positive_findings": [],
                "missing_information": ["PCI recommendation"],
                "safety": {"unsafe": True, "severity": "Critical", "harm_type": "death", "explanation": "STEMI requires PCI"},
                "guideline": {"follows_guideline": False, "guideline_source": "AHA", "guideline_name": "STEMI", "violated_rule": "PCI in 90 mins", "explanation": "Missed"},
                "hallucination": False,
                "failure_hypotheses": [],
                "primary_failure": {"category": "Wrong Diagnosis", "subcategory": "Missing STEMI", "confidence": 0.9, "rationale": "Clear ECG"},
                "clinical_summary": "Fatal error missing STEMI.",
                "failure_tags": ["Diagnosis", "Treatment Planning"]
            })
        else: # Preference Response
            return json.dumps({"chosen": "The correct answer is B. This is a clear case of STEMI..."})

# 2. Patch dependencies
app = create_app()

def override_get_pipeline():
    # Instantiate actual pipeline but with mock services
    from app.modules.pipeline import MedAlignPipeline
    from app.modules.datasets.manager import DatasetManager
    from app.modules.evaluation.judge import MedicalLLMJudge
    from app.modules.investigator.investigator import MedicalInvestigator
    from app.modules.statistics.engine import EvaluationAnalyticsEngine
    from app.modules.benchmark.engine import BenchmarkEngine
    from app.modules.preference.generator import PreferenceGenerator
    from app.modules.reports.generator import ReportGenerator
    
    mock_llm = MockGroqClient()
    mock_inference = MockInferenceService()
    
    return MedAlignPipeline(
        dataset_manager=DatasetManager(),
        inference_service=mock_inference,
        judge=MedicalLLMJudge(llm_client=mock_llm),
        investigator=MedicalInvestigator(llm_client=mock_llm),
        statistics_engine=EvaluationAnalyticsEngine(),
        benchmark_engine=BenchmarkEngine(),
        preference_generator=PreferenceGenerator(llm_client=mock_llm),
        report_generator=ReportGenerator(),
    )

app.dependency_overrides[get_pipeline] = override_get_pipeline

client = TestClient(app)


def test_pipeline_execution():
    """Verify the full pipeline works end-to-end via the API endpoint."""
    # Ensure tiny dataset is in datasets dir
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Send request to evaluate endpoint
    response = client.post("/api/v1/evaluate?dataset_name=datasets/tiny_sample.json&report_format=json")
    
    assert response.status_code == 200, response.text
    data = response.json()
    
    assert data["success"] is True
    
    res = data["data"]
    assert res["samples_processed"] == 1
    
    # Statistics generated
    assert res["evaluation_report"]["overall_metrics"]["total_samples"] == 1
    assert res["evaluation_report"]["overall_metrics"]["unsafe_rate"] == 1.0 # based on our mock
    
    # Benchmark generated
    assert res["benchmark_metrics"]["total_samples"] == 1
    
    # Report generated
    assert "format" in res["final_report"]
    
    # Now verify preferences endpoint
    response = client.post("/api/v1/preferences", json={"dataset_name": "datasets/tiny_sample.json"})
    assert response.status_code == 200, response.text
    pref_data = response.json()["data"]
    
    assert pref_data["total_pairs"] == 1
    assert Path(pref_data["output_file"]).exists()
