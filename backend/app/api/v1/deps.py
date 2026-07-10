"""Dependency injection — simplified to only what's needed."""

from __future__ import annotations

from fastapi import Depends
from app.config.settings import get_settings
from app.modules.pipeline import MedAlignPipeline
from app.modules.datasets.manager import DatasetManager
from app.services.inference.transformers_service import TransformersInferenceService
from app.services.llm.ollama_client import OllamaClient
from app.modules.workflow import MedAlignWorkflow

# Global singletons — avoid reloading GPU models on every request
_workflow: MedAlignWorkflow | None = None


def get_llm_client() -> OllamaClient:
    settings = get_settings()
    return OllamaClient(
        base_url=settings.inference.ollama_base_url,
        fallback_models=settings.inference.fallback_models_list
    )


def get_inference_service() -> TransformersInferenceService:
    settings = get_settings()
    return TransformersInferenceService(
        model_id=settings.model.default_base_model
    )


def get_dataset_manager() -> DatasetManager:
    return DatasetManager()


def get_pipeline(
    dataset_manager: DatasetManager = Depends(get_dataset_manager),
    inference: TransformersInferenceService = Depends(get_inference_service),
    ollama: OllamaClient = Depends(get_llm_client),
) -> MedAlignPipeline:
    return MedAlignPipeline(
        dataset_manager=dataset_manager,
        inference_service=inference,
        llm_client=ollama,
    )


def get_workflow() -> MedAlignWorkflow:
    global _workflow
    if _workflow is not None:
        return _workflow

    settings = get_settings()
    # Manual instantiation for the global workflow object
    pipeline = MedAlignPipeline(
        dataset_manager=DatasetManager(),
        inference_service=TransformersInferenceService(
            model_id=settings.model.default_base_model
        ),
        llm_client=OllamaClient(
            base_url=settings.inference.ollama_base_url,
            fallback_models=settings.inference.fallback_models_list
        ),
    )
    _workflow = MedAlignWorkflow(
        dataset_manager=DatasetManager(),
        pipeline=pipeline,
        storage_dir=settings.storage.outputs_dir,
    )
    return _workflow
