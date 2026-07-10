"""Central API v1 router — aggregates all endpoint modules."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.datasets import router as datasets_router
from app.api.v1.evaluations import router as evaluations_router
from app.api.v1.health import router as health_router
from app.api.v1.upload import router as upload_router
from app.api.v1.train import router as train_router
from app.api.v1.models import router as models_router

api_router = APIRouter()

api_router.include_router(health_router, tags=["Health"])
api_router.include_router(train_router, prefix="/train", tags=["Train"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(evaluations_router, tags=["Evaluations"])
api_router.include_router(upload_router, tags=["Upload"])
api_router.include_router(models_router, tags=["Models"])
