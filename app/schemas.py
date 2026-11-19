"""Request and response bodies."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, model_validator


class ItemSnapshot(BaseModel):
    """What the pharmacy system knows about one item at the start of a day. Quantities are in base units."""

    item_id: str = Field(..., max_length=64, examples=["MED-012"])
    as_of: date = Field(..., description="The day being predicted for; history covers days before it.")
    stock_on_hand: float = Field(..., gt=0, description="Usable stock at the start of the day. Must be above zero.")
    dispensed_last_7d: float = Field(..., ge=0)
    dispensed_last_28d: float = Field(..., ge=0)
    dispensing_std_28d: float = Field(0, ge=0, description="Standard deviation of daily dispensing over 28 days.")
    outstanding_order_qty: float = Field(0, ge=0, description="Ordered and not yet received.")
    oldest_open_order_age_days: float | None = Field(None, ge=0, description="Leave empty when no order is open.")
    days_since_last_order: float | None = Field(None, ge=0)
    recent_lead_time_days: float | None = Field(None, ge=0, description="Mean lead time of the last 3 deliveries.")
    stockout_days_last_90d: float = Field(0, ge=0, le=90)
    days_since_last_stockout: float | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _week_within_month(self) -> "ItemSnapshot":
        if self.dispensed_last_7d > self.dispensed_last_28d:
            raise ValueError("dispensed_last_7d cannot exceed dispensed_last_28d")
        return self


class PredictionRequest(BaseModel):
    items: list[ItemSnapshot] = Field(..., min_length=1, max_length=1000)


class ItemPrediction(BaseModel):
    item_id: str
    as_of: date
    stockout_probability: float = Field(..., description="Calibrated probability of a stockout within the horizon.")
    review: bool = Field(..., description="True when the probability is at or above the review threshold.")
    days_of_stock: float


class PredictionResponse(BaseModel):
    model_version: str
    horizon_days: int
    review_threshold: float
    predictions: list[ItemPrediction] = Field(..., description="Highest risk first.")
