"""FastAPI service: send item snapshots, get 7-day stockout risk back."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Request

from app import __version__
from app.logging_setup import setup_logging
from app.middleware import RequestIDMiddleware, RequestLoggingMiddleware
from app.model import StockoutModel
from app.schemas import PredictionRequest, PredictionResponse
from app.security import require_api_key
from app.settings import DEV_API_KEY, get_settings

logger = logging.getLogger("pharma_sight")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    if settings.app_env == "production" and DEV_API_KEY in settings.keys:
        raise RuntimeError("Set API_KEYS to a real secret before running in production.")
    app.state.model = StockoutModel(settings.model_dir)  # fail fast if the artifacts are missing
    logger.info("Model %s loaded", app.state.model.meta["model_version"])
    yield


app = FastAPI(
    title="PharmaSight stockout API",
    description="Probability that an in-stock pharmacy item runs out within the next 7 days. "
    "Trained on synthetic data; see the repository README.",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


@app.get("/api/v1/health", tags=["Health"])
def health(request: Request) -> dict:
    return {"status": "ok", "version": __version__, "model_version": request.app.state.model.meta["model_version"]}


@app.get("/api/v1/model", tags=["Health"])
def model_info(request: Request) -> dict:
    """Model metadata: features, threshold, training data and headline test metrics."""
    return request.app.state.model.meta


@app.post("/api/v1/predictions", response_model=PredictionResponse, tags=["Predictions"],
          dependencies=[Depends(require_api_key)])
def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
    model: StockoutModel = request.app.state.model
    snapshots = pd.DataFrame([item.model_dump() for item in body.items]).rename(columns={"as_of": "date"})
    scored = model.predict(snapshots).sort_values("stockout_probability", ascending=False)
    return PredictionResponse(
        model_version=model.meta["model_version"],
        horizon_days=model.meta["horizon_days"],
        review_threshold=model.threshold,
        predictions=scored.rename(columns={"date": "as_of"}).to_dict("records"),
    )
