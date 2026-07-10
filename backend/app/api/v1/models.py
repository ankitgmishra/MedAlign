"""Model checkpoint management API.

Provides endpoints to list all saved SFT/DPO checkpoints
(sorted by creation time, newest first) and delete them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException
from app.utils.response import api_response

router = APIRouter()

# Checkpoints are stored under /tmp — each named sft_output, sft_output_1, dpo_output, etc.
CHECKPOINT_BASE = Path("/tmp")


def _list_checkpoints(prefix: str) -> list[dict]:
    """Return all checkpoint dirs for a given prefix, newest first."""
    dirs = []
    for p in CHECKPOINT_BASE.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            stat = p.stat()
            dirs.append({
                "name": p.name,
                "path": str(p),
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "size_mb": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1_048_576, 2),
            })
    # Sort newest first
    dirs.sort(key=lambda d: d["created_at"], reverse=True)
    return dirs


@router.get("/models/checkpoints")
async def list_all_checkpoints():
    """List all SFT and DPO model checkpoints sorted by creation time (newest first)."""
    try:
        sft = _list_checkpoints("sft_output")
        dpo = _list_checkpoints("dpo_output")
        return api_response(
            message="Checkpoints loaded.",
            data={"sft": sft, "dpo": dpo},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/models/checkpoints/{kind}")
async def list_checkpoints_by_kind(kind: str):
    """List SFT or DPO checkpoints specifically."""
    if kind not in ("sft", "dpo"):
        raise HTTPException(status_code=400, detail="kind must be 'sft' or 'dpo'")
    prefix = f"{kind}_output"
    try:
        checkpoints = _list_checkpoints(prefix)
        return api_response(message=f"{kind.upper()} checkpoints loaded.", data=checkpoints)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.delete("/models/checkpoints/{kind}/{name}")
async def delete_checkpoint(kind: str, name: str):
    """Delete a specific model checkpoint directory."""
    if kind not in ("sft", "dpo"):
        raise HTTPException(status_code=400, detail="kind must be 'sft' or 'dpo'")

    # Safety: ensure the requested name actually starts with the correct prefix
    prefix = f"{kind}_output"
    if not name.startswith(prefix):
        raise HTTPException(status_code=400, detail=f"Invalid checkpoint name for kind='{kind}'")

    target = CHECKPOINT_BASE / name
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint '{name}' not found")

    try:
        shutil.rmtree(target)
        return api_response(message=f"Checkpoint '{name}' deleted successfully.", data={"deleted": name})
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
