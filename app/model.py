"""Loads the trained model and scores item snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from pipeline.features import FEATURES, SNAPSHOT, add_features, days_of_stock


class StockoutModel:
    def __init__(self, model_dir: Path) -> None:
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        if self.meta["features"] != FEATURES:
            raise RuntimeError("Model was trained on different features than this code builds; retrain it.")
        self.estimator = joblib.load(model_dir / "stockout_model.joblib")  # our own artifact, not user input
        self.threshold = self.meta["review_threshold"]

    def predict(self, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Snapshots need a ``date`` column plus the SNAPSHOT fields."""
        snapshots = snapshots.astype({c: float for c in SNAPSHOT})
        p = self.estimator.predict_proba(add_features(snapshots))[:, 1]
        return snapshots.assign(
            stockout_probability=p.round(4),
            review=p >= self.threshold,
            days_of_stock=days_of_stock(snapshots).round(1),
        )
