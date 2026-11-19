import json
import os
from pathlib import Path

os.environ["API_KEYS"] = "test-key"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

EXAMPLE = json.loads((Path(__file__).parent.parent / "examples" / "request.json").read_text())
KEY = {"X-Api-Key": "test-key"}


def client() -> TestClient:
    return TestClient(app)


def test_health_reports_loaded_model():
    with client() as c:
        r = c.get("/api/v1/health")
    assert r.status_code == 200 and r.json()["model_version"]


def test_prediction_needs_a_valid_key():
    with client() as c:
        assert c.post("/api/v1/predictions", json=EXAMPLE).status_code == 401
        assert c.post("/api/v1/predictions", json=EXAMPLE, headers={"X-Api-Key": "wrong"}).status_code == 403


def test_predictions_are_probabilities_sorted_by_risk():
    with client() as c:
        r = c.post("/api/v1/predictions", json=EXAMPLE, headers=KEY)
    assert r.status_code == 200
    p = [x["stockout_probability"] for x in r.json()["predictions"]]
    assert len(p) == 3 and all(0 <= v <= 1 for v in p) and p == sorted(p, reverse=True)


def test_less_stock_means_more_risk():
    low, high = dict(EXAMPLE["items"][0]), dict(EXAMPLE["items"][0])
    low["item_id"], high["item_id"], high["stock_on_hand"] = "low", "high", 4000
    with client() as c:
        r = c.post("/api/v1/predictions", json={"items": [low, high]}, headers=KEY)
    risk = {x["item_id"]: x["stockout_probability"] for x in r.json()["predictions"]}
    assert risk["low"] > risk["high"]


def test_rejects_items_already_out_of_stock_and_inconsistent_history():
    bad_stock = dict(EXAMPLE["items"][0], stock_on_hand=0)
    bad_history = dict(EXAMPLE["items"][0], dispensed_last_7d=500)
    with client() as c:
        for item in (bad_stock, bad_history):
            assert c.post("/api/v1/predictions", json={"items": [item]}, headers=KEY).status_code == 422
