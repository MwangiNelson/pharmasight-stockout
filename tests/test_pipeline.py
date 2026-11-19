import numpy as np
import pandas as pd

from pipeline import build_dataset, simulate
from pipeline.config import HORIZON, TEST_START, VAL_START
from pipeline.train import split

START, END = pd.Timestamp("2024-05-01"), pd.Timestamp("2024-10-31")


def small_extract(tmp_path):
    for name, df in simulate.simulate(seed=1, n_items=6, start=START, end=END).items():
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    return build_dataset.load_raw(tmp_path)


def test_features_do_not_change_when_the_future_changes(tmp_path):
    raw = small_extract(tmp_path)
    cut = pd.Timestamp("2024-08-15")
    table, *_ = build_dataset.build(raw, END)

    # Rewrite everything that happens from the cut onwards.
    future = dict(raw)
    d = raw["dispensing"]
    future["dispensing"] = d.assign(qty=np.where(d["dispensed_at"] >= cut, d["qty"] * 3, d["qty"]))
    future["moves"] = raw["moves"][raw["moves"]["moved_at"] < cut]
    p = raw["pos"]
    future["pos"] = p[p["ordered_at"] < cut].assign(
        closed_at=lambda x: x["closed_at"].where(x["closed_at"] < cut), status=lambda x: x["status"].where(x["closed_at"] < cut, "OPEN")
    )
    changed, *_ = build_dataset.build(future, END)

    before = lambda t: t[t["date"] <= cut].drop(columns="y").set_index(["item_id", "date"])
    a, b = before(table), before(changed)
    common = a.index.intersection(b.index)
    assert len(common) > 50
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_label_is_one_only_in_the_week_before_stock_runs_out():
    days = pd.date_range(START, periods=90)
    lines = [
        {"line_id": n, "item_id": item, "dispensed_at": day + pd.Timedelta(hours=10), "qty": 5, "reversal_of": None}
        for n, (item, day) in enumerate([(i, d) for d in days for i in ("A", "B")])
        if not (item == "A" and day >= days[60])  # A has 300 units at 5 a day: empty at close of day 59
    ]
    raw = {
        "items": pd.DataFrame({"item_id": ["A", "B"], "pack_size": [1, 1]}),
        "opening": pd.DataFrame({"item_id": ["A", "B"], "as_of": [START, START], "qty": [300, 10**6], "unit": "UNIT"}),
        "dispensing": pd.DataFrame(lines),
        "moves": pd.DataFrame({"item_id": ["B"], "moved_at": [days[5]], "movement_type": ["EXPIRY"], "qty": [1], "unit": ["UNIT"]}),
        "pos": pd.DataFrame({"item_id": ["B"], "ordered_at": [days[1]], "closed_at": [days[10]], "qty_ordered": [10],
                             "status": ["RECEIVED"]}),
    }
    table, *_ = build_dataset.build(raw, days[-1])
    a = table[table["item_id"] == "A"].set_index("date")
    t = (a.index - START).days
    assert t.min() == 28 and t.max() == 59  # warm-up first; no rows once out of stock
    assert (a["y"] == (t >= 60 - HORIZON)).all()
    assert (a["stock_on_hand"] == 300 - 5 * t).all()


def test_splits_share_no_label_window():
    df = pd.DataFrame({"date": pd.date_range("2024-06-01", "2026-02-28"), "y": 0})
    train, val, test = split(df)
    window = pd.Timedelta(days=HORIZON - 1)
    assert train["date"].max() + window < VAL_START <= val["date"].min()
    assert val["date"].max() + window < TEST_START <= test["date"].min()
