"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from app import __app_name__, __version__
from app.config.settings import get_settings
from app.utils.response import api_response

router = APIRouter()


@router.get("/health")
async def health_check() -> dict:
    """Return application health status."""
    settings = get_settings()
    return api_response(
        message="healthy",
        data={
            "app": __app_name__,
            "version": __version__,
            "environment": settings.environment,
            "status": "operational",
        },
    )
