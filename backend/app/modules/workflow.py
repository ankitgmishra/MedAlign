"""MedAlign Master Workflow Orchestrator (Minimal State Container)."""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

from app.modules.datasets.manager import DatasetManager
from app.modules.pipeline import MedAlignPipeline
from app.utils.logging import get_logger
from app.utils.io import save_json, load_json

logger = get_logger("workflow")


class MedAlignWorkflow:
    """State orchestrator for the MedAlign process."""

    def __init__(
        self,
        dataset_manager: DatasetManager,
        pipeline: MedAlignPipeline,
        storage_dir: Path,
    ) -> None:
        self.dataset_manager = dataset_manager
        self.pipeline = pipeline
        self.storage_dir = storage_dir
        
        self.state_file = self.storage_dir / "workflow_state.json"
        self._state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_file.exists():
            return load_json(self.state_file)
        return {
            "current_dataset": None,
        }

    def _save_state(self) -> None:
        save_json(self._state, self.state_file)

    def upload_dataset(self, file_path: str | Path) -> str:
        """Register the uploaded dataset in the workflow."""
        self._state["current_dataset"] = str(file_path)
        self._save_state()
        return str(file_path)
