"""Model features from an operational snapshot.

Training and the API both call ``add_features``, so a request is turned into
features by the same code that built the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# What a pharmacy system knows about one item at the start of a day.
SNAPSHOT = [
    "stock_on_hand",
    "dispensed_last_7d",
    "dispensed_last_28d",
    "dispensing_std_28d",
    "outstanding_order_qty",
    "oldest_open_order_age_days",  # empty when no order is open
    "days_since_last_order",  # empty when the item was never ordered
    "recent_lead_time_days",  # mean of the last 3 deliveries; empty before the first
    "stockout_days_last_90d",
    "days_since_last_stockout",  # empty when no stockout on record
]

FEATURES = [
    "log_stock",
    "log_avg_daily_28d",
    "recent_trend",
    "cv_28d",
    "log_days_of_stock",
    "log_cover_with_orders",
    "has_open_order",
    "oldest_open_order_age_days",
    "days_since_last_order",
    "recent_lead_time_days",
    "cover_minus_lead_time",
    "stockout_days_last_90d",
    "log_days_since_last_stockout",
    "day_of_week",
    "month_sin",
    "month_cos",
]


def days_of_stock(df: pd.DataFrame) -> pd.Series:
    """Stock divided by average daily dispensing over 28 days, capped at a year."""
    rate = np.maximum(df["dispensed_last_28d"] / 28, 0.1)
    return (df["stock_on_hand"] / rate).clip(upper=365)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature matrix for snapshot rows with a ``date`` column."""
    avg28 = df["dispensed_last_28d"] / 28
    avg7 = df["dispensed_last_7d"] / 7
    dos = days_of_stock(df)
    cover = ((df["stock_on_hand"] + df["outstanding_order_qty"]) / np.maximum(avg28, 0.1)).clip(upper=365)
    month = pd.to_datetime(df["date"]).dt.month
    out = pd.DataFrame(
        {
            "log_stock": np.log1p(df["stock_on_hand"]),
            "log_avg_daily_28d": np.log1p(avg28),
            "recent_trend": np.log((avg7 + 0.1) / (avg28 + 0.1)),
            "cv_28d": df["dispensing_std_28d"] / (avg28 + 0.1),
            "log_days_of_stock": np.log1p(dos),
            "log_cover_with_orders": np.log1p(cover),
            "has_open_order": (df["outstanding_order_qty"] > 0).astype(float),
            "oldest_open_order_age_days": df["oldest_open_order_age_days"],
            "days_since_last_order": df["days_since_last_order"],
            "recent_lead_time_days": df["recent_lead_time_days"],
            "cover_minus_lead_time": dos - df["recent_lead_time_days"],
            "stockout_days_last_90d": df["stockout_days_last_90d"],
            "log_days_since_last_stockout": np.log1p(df["days_since_last_stockout"]),
            "day_of_week": pd.to_datetime(df["date"]).dt.dayofweek,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=df.index,
    )
    return out[FEATURES].astype(float)
