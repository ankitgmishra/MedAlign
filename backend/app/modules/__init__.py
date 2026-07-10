"""Evaluation module — LLM Judge and accuracy scoring."""

from app.modules.evaluation.base import BaseEvaluator
from app.modules.pipeline import MedAlignPipeline

__all__ = ["BaseEvaluator", "MedAlignPipeline"]
