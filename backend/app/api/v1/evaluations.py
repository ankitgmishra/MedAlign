"""Evaluation API — base, sft, dpo endpoints + cross-run comparison."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from app.api.v1.deps import get_workflow
from app.modules.eval_engine import compare_evaluations
from app.modules.workflow import MedAlignWorkflow
from app.utils.response import api_response
from app.utils.exceptions import ApplicationError

router = APIRouter()


def _serialize(results: dict) -> dict:
    """Strip non-serializable objects from pipeline output."""
    pref = results.get("preference_dataset", [])
    serialized_pref = [p.model_dump() if hasattr(p, "model_dump") else p for p in pref]
    return {
        "label": results.get("label", ""),
        "samples_processed": results.get("samples_processed", 0),
        "aggregate": results.get("aggregate", {}),
        "judge_results": results.get("rows", []),
        "csv_path": results.get("csv_path", ""),
        "agent_summary": results.get("agent_summary", ""),
        "failure_records": results.get("failure_records", []),
        "preference_dataset": serialized_pref,
    }


# ── Base Evaluation ───────────────────────────────────────────────────────────

@router.post("/evaluate")
async def run_base_eval(
    dataset_name: str,
    model_name: str = None,
    workflow: MedAlignWorkflow = Depends(get_workflow),
) -> dict:
    """Run base model evaluation. Saves base_eval.csv + base_eval.json."""
    try:
        workflow.upload_dataset(dataset_name)
        results = workflow.pipeline.run(dataset_name, label="base", model_name=model_name)
        return api_response(
            message=f"Base evaluation complete — {results.get('samples_processed', 0)} samples",
            data=_serialize(results),
        )
    except ApplicationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ── Post-SFT Evaluation ───────────────────────────────────────────────────────

@router.post("/evaluate/sft")
async def run_sft_eval(
    dataset_name: str = None,
    lora_path: str = None,
    workflow: MedAlignWorkflow = Depends(get_workflow),
) -> dict:
    """Evaluate a trained SFT checkpoint. `lora_path` selects which checkpoint to use (defaults to /tmp/sft_output)."""
    try:
        if dataset_name:
            workflow.upload_dataset(dataset_name)

        dataset = workflow._state.get("current_dataset")
        if not dataset:
            raise ApplicationError("No dataset uploaded. Run base evaluation first.")

        resolved_path = lora_path or "/tmp/sft_output"
        if not os.path.exists(resolved_path):
            raise ApplicationError(f"SFT model weights not found at '{resolved_path}'. Please run SFT training first.")

        results = workflow.pipeline.run(dataset, label="sft", lora_path=resolved_path)
        return api_response(
            message=f"Post-SFT evaluation complete — {results.get('samples_processed', 0)} samples",
            data=_serialize(results),
        )
    except ApplicationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ── Post-DPO Evaluation ───────────────────────────────────────────────────────

@router.post("/evaluate/dpo")
async def run_dpo_eval(
    dataset_name: str = None,
    lora_path: str = None,
    workflow: MedAlignWorkflow = Depends(get_workflow),
) -> dict:
    """Evaluate a trained DPO checkpoint. `lora_path` selects which checkpoint to use (defaults to /tmp/dpo_output)."""
    try:
        if dataset_name:
            workflow.upload_dataset(dataset_name)

        dataset = workflow._state.get("current_dataset")
        if not dataset:
            raise ApplicationError("No dataset uploaded. Run base evaluation first.")

        resolved_path = lora_path or "/tmp/dpo_output"
        if not os.path.exists(resolved_path):
            raise ApplicationError(f"DPO model weights not found at '{resolved_path}'. Please run DPO training first.")

        results = workflow.pipeline.run(dataset, label="dpo", lora_path=resolved_path)
        return api_response(
            message=f"Post-DPO evaluation complete — {results.get('samples_processed', 0)} samples",
            data=_serialize(results),
        )
    except ApplicationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ── Cross-Run Comparison ──────────────────────────────────────────────────────

@router.get("/evaluate/compare")
async def compare_all_evals() -> dict:
    """
    Compare base vs SFT vs DPO evaluation results.
    Returns aggregate metrics + deltas for each stage.
    """
    try:
        comparison = compare_evaluations()
        return api_response(message="Evaluation comparison generated.", data=comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ── CSV Download ────────────────────────────────────────────────────

@router.get("/evaluate/download/{label}")
async def download_eval_csv(label: str):
    """Return the raw CSV file for a given eval label (base | sft | dpo)."""
    path = Path("outputs") / f"{label}_eval.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No CSV file found for label '{label}'")

    return FileResponse(
        path=path,
        media_type="text/csv",
        filename=f"{label}_eval.csv"
    )
