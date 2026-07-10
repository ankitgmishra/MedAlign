"""Evaluation module — LLM-as-a-Judge and accuracy scoring."""

from app.modules.evaluation.base import BaseEvaluator
from app.modules.evaluation.judge import MedicalLLMJudge

__all__ = ["BaseEvaluator", "MedicalLLMJudge"]
