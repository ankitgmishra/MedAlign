"""FastAPI application factory.

This is the ONLY file that depends on FastAPI.  All domain logic lives
in ``app.modules`` and can be imported directly from Python.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app import __app_name__, __version__
from app.api.v1.router import api_router
from app.config.settings import get_settings
from app.utils.exceptions import ApplicationError
from app.utils.logging import setup_logging
from app.utils.response import api_response


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=__app_name__,
        version=__version__,
        description=(
            "Open-Source Medical AI Evaluation & Post-Training Platform. "
            "Evaluate Medical LLMs · Analyze Failures · Generate Preference "
            "Datasets · Perform DPO Post-Training · Benchmark Models."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ─────────────────────────────────────────
    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        _request: Request,
        exc: ApplicationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=api_response(
                success=False,
                message=exc.message,
                errors=exc.to_dict(),
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=api_response(
                success=False,
                message="An unexpected error occurred.",
                errors={"error": str(exc)},
            ),
        )

    # ── Routers ──────────────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    return app

app = create_app()
