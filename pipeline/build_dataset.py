"""Turn the raw extract into the model table.

One row per item per day for items in stock at the start of that day.
Features use only records up to the end of the previous day. The label is 1
when the recorded balance reaches zero on any of the next HORIZON days.

Steps:
1. Clean dispensing: drop duplicate exports, erroneous lines and their reversals.
2. Convert pack quantities to base units.
3. Rebuild each item's daily closing balance from its opening balance and
   dated movements.
4. Flag days whose balance cannot be trusted: store-wide recording outages
   (until the item's next stock-take) and negative balances.
5. Build features and labels; drop rows whose label window is incomplete or
   touches an untrusted day.

Writes data/processed/model_table.csv.gz, data/processed/episodes.csv and
reports/data_quality.json.

Run: python -m pipeline.build_dataset
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pipeline.config import END, HORIZON, PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from pipeline.features import SNAPSHOT

MOVES = ["RECEIPT", "TRANSFER_OUT", "EXPIRY", "ADJUSTMENT"]


def load_raw(raw_dir=RAW_DIR) -> dict:
    read = lambda name, dates=(): pd.read_csv(raw_dir / f"{name}.csv", parse_dates=list(dates))
    return {
        "items": read("items"),
        "opening": read("opening_balances", ["as_of"]),
        "dispensing": read("dispensing", ["dispensed_at"]),
        "moves": read("stock_movements", ["moved_at"]),
        "pos": read("purchase_orders", ["ordered_at", "closed_at"]),
    }


def clean_dispensing(disp: pd.DataFrame) -> pd.DataFrame:
    disp = disp.drop_duplicates("line_id")
    reversed_ids = disp["reversal_of"].dropna().astype(int)
    keep = disp["reversal_of"].isna() & ~disp["line_id"].isin(reversed_ids)
    return disp[keep].assign(date=lambda d: d["dispensed_at"].dt.normalize())


def to_base_units(df: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    pack = df["item_id"].map(items.set_index("item_id")["pack_size"])
    return df["qty"] * np.where(df["unit"] == "PACK", pack, 1)


def last_true_date(g: pd.DataFrame, flag: pd.Series) -> pd.Series:
    """Most recent date, up to and including each row, on which flag was true."""
    return g["date"].where(flag).groupby(g["item_id"]).ffill()


def per_item(g: pd.DataFrame, s: pd.Series, fn) -> pd.Series:
    return s.groupby(g["item_id"]).transform(fn)


def daily_ledger(raw: dict, end=END) -> tuple[pd.DataFrame, pd.Series]:
    """Daily balance and trust flag per item; also returns detected outage days."""
    items = raw["items"]
    disp = clean_dispensing(raw["dispensing"])

    # Store-wide outage: far fewer lines than the trailing four-week median.
    lines = disp.groupby("date").size().reindex(pd.date_range(disp["date"].min(), end), fill_value=0)
    outage_days = lines.index[lines < 0.3 * lines.rolling(28, min_periods=7).median().shift(1)]

    opening = raw["opening"].assign(qty=lambda o: to_base_units(o, items))
    grid = pd.concat(
        pd.DataFrame({"item_id": o.item_id, "date": pd.date_range(o.as_of, end)}) for o in opening.itertuples()
    )

    moves = raw["moves"].assign(date=lambda m: m["moved_at"].dt.normalize())
    moves["qty"] = to_base_units(moves, items)
    moves = moves.pivot_table(index=["item_id", "date"], columns="movement_type", values="qty", aggfunc="sum")
    moves = moves.reindex(columns=MOVES)
    dispensed = disp.groupby(["item_id", "date"])["qty"].sum().rename("dispensed")

    g = grid.merge(dispensed, on=["item_id", "date"], how="left").merge(moves, on=["item_id", "date"], how="left")
    g[["dispensed", *MOVES]] = g[["dispensed", *MOVES]].fillna(0)
    net = g["RECEIPT"] - g["dispensed"] - g["TRANSFER_OUT"] - g["EXPIRY"] + g["ADJUSTMENT"]
    g["balance"] = g["item_id"].map(opening.set_index("item_id")["qty"]) + net.groupby(g["item_id"]).cumsum()

    # An outage leaves the balance too high until a stock-take corrects it.
    last_outage = last_true_date(g, g["date"].isin(outage_days))
    last_count = last_true_date(g, g["ADJUSTMENT"] != 0)
    g["untrusted"] = (last_outage.notna() & ~(last_count >= last_outage)) | (g["balance"] < 0)
    g["stockout"] = (g["balance"] <= 0) & ~g["untrusted"]
    return g, pd.Series(outage_days)


def add_order_history(g: pd.DataFrame, pos: pd.DataFrame, end=END) -> pd.DataFrame:
    """Order features as known at the start of each day."""
    # Outstanding at the start of day t: ordered before t, not closed before t.
    p = pos.reset_index(drop=True).assign(closed_at=lambda p: p["closed_at"].fillna(end + pd.Timedelta(days=1)))
    n_days = (p["closed_at"] - p["ordered_at"]).dt.days
    e = p.loc[p.index.repeat(n_days)]
    e = e.assign(date=e["ordered_at"] + pd.to_timedelta(e.groupby(level=0).cumcount() + 1, unit="D"))
    open_orders = e.groupby(["item_id", "date"]).agg(
        outstanding_order_qty=("qty_ordered", "sum"), oldest_order=("ordered_at", "min")
    )
    g = g.merge(open_orders, on=["item_id", "date"], how="left")
    g["outstanding_order_qty"] = g["outstanding_order_qty"].fillna(0)
    g["oldest_open_order_age_days"] = (g["date"] - g["oldest_order"]).dt.days

    ordered = pos.groupby(["item_id", "ordered_at"]).size().rename("n_orders").rename_axis(["item_id", "date"])
    g = g.merge(ordered, on=["item_id", "date"], how="left")
    last_order = per_item(g, last_true_date(g, g["n_orders"].notna()), lambda s: s.shift(1))
    g["days_since_last_order"] = (g["date"] - last_order).dt.days

    # Lead time of the last three deliveries, known from the day after receipt.
    rec = pos[pos["status"] == "RECEIVED"].sort_values("closed_at")
    rec = rec.assign(lead=(rec["closed_at"] - rec["ordered_at"]).dt.days)
    rec["recent_lead"] = rec.groupby("item_id")["lead"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    rec = rec.drop_duplicates(["item_id", "closed_at"], keep="last").rename(columns={"closed_at": "date"})
    g = g.merge(rec[["item_id", "date", "recent_lead"]], on=["item_id", "date"], how="left")
    g["recent_lead_time_days"] = per_item(g, g["recent_lead"], lambda s: s.ffill().shift(1))
    return g.drop(columns=["oldest_order", "n_orders", "recent_lead"])


def build(raw: dict, end=END) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    g, outage_days = daily_ledger(raw, end)
    g = add_order_history(g, raw["pos"], end)

    prev = lambda s: per_item(g, s, lambda x: x.shift(1))
    rolling = lambda s, w, stat, min_p=None: per_item(
        g, s, lambda x: getattr(x.rolling(w, min_periods=min_p or w), stat)().shift(1)
    )
    g["stock_on_hand"] = prev(g["balance"])
    g["dispensed_last_7d"] = rolling(g["dispensed"], 7, "sum")
    g["dispensed_last_28d"] = rolling(g["dispensed"], 28, "sum")
    g["dispensing_std_28d"] = rolling(g["dispensed"], 28, "std")
    g["stockout_days_last_90d"] = rolling(g["stockout"].astype(float), 90, "sum", 1)
    g["days_since_last_stockout"] = (g["date"] - prev(last_true_date(g, g["stockout"]))).dt.days

    # Label window: today and the next HORIZON - 1 days, all trusted and observed.
    ahead = lambda s: per_item(g, s.astype(float), lambda x: x[::-1].rolling(HORIZON, min_periods=HORIZON).max()[::-1])
    g["y"] = ahead(g["stockout"])
    window_untrusted = ahead(g["untrusted"])

    steps = {"item_days": len(g)}
    keep = g["dispensed_last_28d"].notna()
    steps["after_28_day_warm_up"] = int(keep.sum())
    keep &= ~prev(g["untrusted"]).fillna(True).astype(bool)
    steps["after_untrusted_opening_balance"] = int(keep.sum())
    keep &= g["stock_on_hand"] > 0
    steps["in_stock_at_start_of_day"] = int(keep.sum())
    keep &= g["y"].notna() & (window_untrusted == 0)
    steps["label_window_complete_and_trusted"] = int(keep.sum())

    table = g.loc[keep, ["item_id", "date", *SNAPSHOT, "y"]].astype({"y": int})
    onset = g["stockout"] & ~prev(g["stockout"]).fillna(True).astype(bool) & ~prev(g["untrusted"]).fillna(True).astype(bool)
    episodes = g.loc[onset, ["item_id", "date"]].rename(columns={"date": "onset"})

    quality = {
        "items_in_extract": int(raw["items"]["item_id"].nunique()),
        "items_without_opening_balance": int(raw["items"]["item_id"].nunique() - raw["opening"]["item_id"].nunique()),
        "dispensing_lines_raw": len(raw["dispensing"]),
        "duplicate_lines_dropped": int(raw["dispensing"]["line_id"].duplicated().sum()),
        "reversal_pairs_dropped": int(raw["dispensing"]["reversal_of"].notna().sum()),
        "pack_unit_records_converted": int((raw["moves"]["unit"] == "PACK").sum()),
        "outage_days_detected": [d.date().isoformat() for d in outage_days],
        "untrusted_item_day_share": round(float(g["untrusted"].mean()), 4),
        "rows": steps,
        "positive_rate": round(float(table["y"].mean()), 4),
        "stockout_episodes": len(episodes),
    }
    return table, episodes, g, quality


def audit_against_truth(g: pd.DataFrame, outage_days: list[str], raw_dir=RAW_DIR) -> dict:
    """Synthetic data only: how well do record-based labels match true stock?"""
    truth = pd.read_csv(raw_dir / "_truth.csv", parse_dates=["date"])
    m = g.merge(truth, on=["item_id", "date"])
    true_out, recorded_out = m["true_stock"] <= 0, m["balance"] <= 0

    def agree(mask: pd.Series) -> dict:
        hit = (recorded_out & true_out & mask).sum()
        return {
            "precision": round(float(hit / max((recorded_out & mask).sum(), 1)), 3),
            "recall": round(float(hit / max((true_out & mask).sum(), 1)), 3),
        }

    injected = {d.date().isoformat() for d in truth.loc[truth["outage"], "date"]}
    return {
        "stockout_days_on_trusted_days": agree(~m["untrusted"]),
        "stockout_days_on_untrusted_days": agree(m["untrusted"]),
        "outage_days": {
            "injected": len(injected),
            "detected": len(outage_days),
            "detected_and_injected": len(injected & set(outage_days)),
        },
    }


def main() -> None:
    table, episodes, g, quality = build(load_raw())
    quality["label_audit_against_simulator_truth"] = audit_against_truth(g, quality["outage_days_detected"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED_DIR / "model_table.csv.gz", index=False)
    episodes.to_csv(PROCESSED_DIR / "episodes.csv", index=False)
    (REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    print(json.dumps({k: v for k, v in quality.items() if k != "outage_days_detected"}, indent=2))


if __name__ == "__main__":
    main()
