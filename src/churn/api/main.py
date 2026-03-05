"""
FastAPI application for real-time churn scoring.
Endpoints:
    GET  /health       → model status check
    POST /predict      → single customer churn prediction
    POST /batch-score  → batch prediction for multiple customers

Start with:  uvicorn churn.api.main:app --host 0.0.0.0 --port 8000
         or: make serve
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn.api.model_loader import registry
from churn.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn.config import cfg

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model into memory on startup."""
    try:
        registry.load_from_disk()
        logger.info("Model loaded successfully at startup.")
    except FileNotFoundError:
        logger.warning("No model found at startup — /predict will return 503.")
    yield
    logger.info("Shutting down — cleaning up.")


app = FastAPI(
    title="UK Telecoms Churn Prediction API",
    description="Real-time customer churn scoring for proactive retention.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if registry.is_loaded else "model_not_loaded",
        model_loaded=registry.is_loaded,
        model_version=registry.version,
    )


def _features_to_df(features: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to a single-row DataFrame matching model input."""
    return pd.DataFrame([features.model_dump()])


def _score_df(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Score a DataFrame and return (probabilities, top_drivers_per_row)."""
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    probs = registry.calibrator.predict_proba(df)

    # Feature importance ranking from model (global — per-request SHAP too expensive)
    feature_names = registry.model.feature_names
    if registry.model._booster is not None:
        importance = registry.model._booster.feature_importance(importance_type="gain")
        top_idx = np.argsort(importance)[::-1][: cfg.scoring.top_n_drivers]
        top_drivers = [feature_names[i] for i in top_idx]
    else:
        top_drivers = []

    return probs, top_drivers


def _risk_tier(p: float) -> str:
    if p >= cfg.scoring.high_risk_threshold:
        return "High"
    elif p >= cfg.scoring.medium_risk_threshold:
        return "Medium"
    return "Low"


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """Score a single customer."""
    df = _features_to_df(features)
    probs, drivers = _score_df(df)
    p = float(probs[0])
    return PredictionResponse(
        churn_probability=round(p, 4),
        risk_tier=_risk_tier(p),
        top_drivers=drivers,
    )


@app.post("/batch-score", response_model=BatchPredictionResponse)
async def batch_score(request: BatchPredictionRequest):
    """Score a batch of customers."""
    rows = [c.model_dump() for c in request.customers]
    df = pd.DataFrame(rows)
    probs, drivers = _score_df(df)
    predictions = [
        PredictionResponse(
            churn_probability=round(float(p), 4),
            risk_tier=_risk_tier(float(p)),
            top_drivers=drivers,
        )
        for p in probs
    ]
    return BatchPredictionResponse(predictions=predictions, n_customers=len(predictions))
