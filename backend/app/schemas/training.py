"""Training insight schemas — ported from ``03_benchmark_engine.py``."""

from __future__ import annotations

from pydantic import BaseModel


class RootCauseInsight(BaseModel):
    """Deep-dive insight connecting a failure capability to training actions."""

    capability: str
    frequency: int
    unsafe_cases: int
    critical_cases: int
    average_claim_support: float
    recommendation: str
    suggested_training: str
    priority: str
