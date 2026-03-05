"""
Pydantic request / response schemas for the FastAPI churn scoring API.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction."""

    ooc_days: float = Field(..., description="Days out of contract (negative = in contract)")
    tenure_days: float = Field(..., description="Days with UK Telecoms")
    dd_cancel_60_day: float = Field(0, description="DD cancellations in last 60 days")
    contract_status_risk: int = Field(1, description="Ordinal risk: 0=early, 1=in, 2=near_expiry, 3=OOC")
    speed_gap_pct: float = Field(0.0, description="(speed - line_speed) / speed")
    call_count_30d: int = Field(0, description="Calls in last 30 days")
    loyalty_call_flag_30d: int = Field(0, description="1 if Loyalty call in last 30 days")
    avg_download_30d: float = Field(0.0, description="Avg daily download MB (last 30d)")
    download_trend_7_30: float = Field(1.0, description="7d/30d download ratio")
    avg_talk_time_30d: float = Field(0.0, description="Avg talk time (seconds)")

    model_config = {"json_schema_extra": {
        "examples": [{
            "ooc_days": 15,
            "tenure_days": 730,
            "dd_cancel_60_day": 1,
            "contract_status_risk": 3,
            "speed_gap_pct": 0.15,
            "call_count_30d": 4,
            "loyalty_call_flag_30d": 1,
            "avg_download_30d": 850.0,
            "download_trend_7_30": 0.6,
            "avg_talk_time_30d": 420.0,
        }]
    }}


class PredictionResponse(BaseModel):
    """Output from /predict endpoint."""
    churn_probability: float = Field(..., ge=0, le=1)
    risk_tier: str = Field(..., description="High / Medium / Low")
    top_drivers: list[str] = Field(default_factory=list, description="Top SHAP drivers")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    n_customers: int
