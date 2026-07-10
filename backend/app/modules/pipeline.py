"""MedAlign Pipeline — thin orchestrator using eval_engine for all evaluation stages."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from app.modules.datasets.manager import DatasetManager
from app.modules.eval_engine import run_evaluation, compare_evaluations
from app.schemas.common import InternalSample
from app.services.inference.base import BaseInferenceService
from app.services.llm.base import BaseLLMClient
from app.utils.logging import get_logger, log_execution

logger = get_logger("pipeline")


class MedAlignPipeline:
    """
    Orchestrates: Inference → Evaluation.
    All heavy lifting is delegated to eval_engine.run_evaluation().
    """

    def __init__(
        self,
        dataset_manager: DatasetManager,
        inference_service: BaseInferenceService,
        llm_client: BaseLLMClient,
    ) -> None:
        self.dataset_manager = dataset_manager
        self.inference_service = inference_service
        self.llm_client = llm_client

    def run(self, dataset_name: str, label: str = "base", lora_path: str = None, model_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Full pipeline run for a given label (base / sft / dpo).
        Returns evaluation results.
        """
        with log_execution(logger, "medalign_pipeline_run"):
            # 1. Load dataset
            logger.info(f"Loading dataset: {dataset_name}")
            samples = self.dataset_manager.load(dataset_name)

            if lora_path:
                logger.info(f"Loading inference model with LoRA adapter: {lora_path}")
                self.inference_service.load_model(lora_path=lora_path, model_name=model_name)
            elif label == "base":
                # Ensure base model is loaded cleanly
                logger.info("Loading base inference model (no adapters)")
                self.inference_service.load_model(lora_path=None, model_name=model_name)

            # 2. Inference
            logger.info(f"Running inference ({len(samples)} samples)...")
            predictions = self._run_inference(samples)

            # 3. LLM Judge Evaluation (parallel, via eval_engine)
            eval_result = run_evaluation(
                label=label,
                samples=samples,
                predictions=predictions,
                llm_client=self.llm_client,
                max_workers=kwargs.get("max_workers", 8),
            )
            
            return {
                **eval_result,
                "samples_processed": len(samples),
            }

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _run_inference(self, samples: List[InternalSample]) -> List[str]:
        predictions = []
        for sample in samples:
            prompt = self._format_prompt(sample)
            pred, _ = self.inference_service.generate([{"role": "user", "content": prompt}])
            predictions.append(pred)
        return predictions

    def _format_prompt(self, sample: InternalSample) -> str:
        q = sample.input.get("question", "")
        opts = sample.input.get("options", {})
        prompt = f"{q}\n\nOptions:\n"
        if isinstance(opts, dict):
            for k, v in opts.items():
                prompt += f"{k}: {v}\n"
        elif isinstance(opts, list):
            for i, o in enumerate(opts):
                prompt += f"{chr(65+i)}: {o}\n"
        return prompt
