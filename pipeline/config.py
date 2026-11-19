"""Paths and study settings shared by every pipeline step."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

SEED = 8401
N_ITEMS = 80
START = pd.Timestamp("2024-05-01")  # mirrors the planned hospital extract window
END = pd.Timestamp("2026-02-28")

HORIZON = 7  # label: stockout on any of the next 7 days, counting today
VAL_START = pd.Timestamp("2025-07-01")
TEST_START = pd.Timestamp("2025-11-01")
ALERT_CAPACITY = 5  # items a pharmacist can review per day
