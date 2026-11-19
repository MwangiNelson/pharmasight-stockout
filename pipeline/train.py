"""Train, select, calibrate and evaluate the 7-day stockout classifier.

Splits are chronological. A row is kept in an earlier split only when its
whole label window ends before the next split starts, so no outcome is shared
across splits. Hyper-parameters, class weighting, calibration and the alert
threshold are chosen on the validation period. The test period is scored once,
at the end.

Writes models/stockout_model.joblib, models/metadata.json,
reports/metrics.json, reports/model_selection.csv and reports/figures/*.png.

Run: python -m pipeline.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import (
    ALERT_CAPACITY, HORIZON, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SEED, TEST_START, VAL_START,
)
from pipeline.features import FEATURES, SNAPSHOT, add_features, days_of_stock

# Reference palette, light mode (categorical slots 1-3 pass all-pairs CVD checks).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_ends_before = lambda cut: df["date"] + pd.Timedelta(days=HORIZON - 1) < cut
    train = df[window_ends_before(VAL_START)]
    val = df[(df["date"] >= VAL_START) & window_ends_before(TEST_START)]
    test = df[df["date"] >= TEST_START]
    return train, val, test


def candidates():
    for C in (0.1, 1.0):
        for cw in (None, "balanced"):
            yield "logistic_regression", {"C": C, "class_weight": cw}, make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(C=C, class_weight=cw, max_iter=2000),
            )
    for lr in (0.05, 0.1):
        for leaves in (15, 31):
            for cw in (None, "balanced"):
                params = {"learning_rate": lr, "max_leaf_nodes": leaves, "class_weight": cw}
                yield "gradient_boosting", params, HistGradientBoostingClassifier(
                    **params, max_iter=300, l2_regularization=1.0, early_stopping=False, random_state=SEED
                )


def ap_interval(y: np.ndarray, score: np.ndarray, items: pd.Series, reps: int = 200) -> list[float]:
    """95% interval for average precision, resampling whole items (rows within an item are correlated)."""
    rng = np.random.default_rng(SEED)
    rows_of = {i: np.flatnonzero(items.values == i) for i in items.unique()}
    ids = list(rows_of)
    stats = []
    for _ in range(reps):
        rows = np.concatenate([rows_of[i] for i in rng.choice(ids, len(ids))])
        if y[rows].any():
            stats.append(average_precision_score(y[rows], score[rows]))
    return [round(float(v), 3) for v in np.percentile(stats, [2.5, 97.5])]


def alert_metrics(df: pd.DataFrame, score: np.ndarray, episodes: pd.DataFrame, k: int = ALERT_CAPACITY) -> dict:
    """Flag the k highest-scoring items each day; how many stockout episodes get a warning first?"""
    t = df[["item_id", "date", "y"]].assign(score=score)
    t["flag"] = t.groupby("date")["score"].rank(ascending=False, method="first") <= k
    m = episodes[episodes["onset"] >= TEST_START].merge(t, on="item_id")
    m["lead"] = (m["onset"] - m["date"]).dt.days
    m = m[(m["lead"] >= 0) & (m["lead"] < HORIZON)]
    eligible = m.groupby(["item_id", "onset"]).ngroups
    warned = m[m["flag"]].groupby(["item_id", "onset"])["lead"].max()  # earliest warning
    return {
        "alerts_per_day": k,
        "precision_of_alerts": round(float(t.loc[t["flag"], "y"].mean()), 3),
        "episodes": int(eligible),
        "episodes_warned": int(len(warned)),
        "episode_recall": round(len(warned) / max(eligible, 1), 3),
        "median_warning_days": float(warned.median()) if len(warned) else None,
    }


def evaluate(name: str, df: pd.DataFrame, score: np.ndarray, episodes: pd.DataFrame, prob=None) -> dict:
    y = df["y"].to_numpy()
    out = {
        "model": name,
        "average_precision": round(float(average_precision_score(y, score)), 3),
        "average_precision_95ci": ap_interval(y, score, df["item_id"]),
        "roc_auc": round(float(roc_auc_score(y, score)), 3),
        "brier": round(float(brier_score_loss(y, prob)), 4) if prob is not None else None,
        **alert_metrics(df, score, episodes),
    }
    return out


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name: str) -> None:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / name, dpi=160)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "model_table.csv.gz", parse_dates=["date"])
    episodes = pd.read_csv(PROCESSED_DIR / "episodes.csv", parse_dates=["onset"])
    train, val, test = split(df)
    X = {name: add_features(part) for name, part in (("train", train), ("val", val), ("test", test))}

    # 1. Select hyper-parameters and class weighting on validation.
    rows, best = [], {}
    for family, params, model in candidates():
        model.fit(X["train"], train["y"])
        ap = average_precision_score(val["y"], model.predict_proba(X["val"])[:, 1])
        rows.append({"family": family, **{k: str(v) for k, v in params.items()}, "val_average_precision": round(ap, 4)})
        if ap > best.get(family, (-1,))[0]:
            best[family] = (ap, params, model)
        print(f"{family:20s} {params} val AP {ap:.3f}")
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "model_selection.csv", index=False)

    rule_val_ap = average_precision_score(val["y"], -days_of_stock(val))
    family = max(best, key=lambda f: best[f][0])
    _, params, chosen = best[family]

    # 2. Calibrate the chosen model on validation; Platt scaling keeps the ranking.
    final = CalibratedClassifierCV(FrozenEstimator(chosen), method="sigmoid").fit(X["val"], val["y"])
    p_val = final.predict_proba(X["val"])[:, 1]
    threshold = float(np.quantile(p_val, 1 - ALERT_CAPACITY * val["date"].nunique() / len(val)))

    # 3. Score the test period once.
    score = {f: best[f][2].predict_proba(X["test"])[:, 1] for f in best}
    p_test = final.predict_proba(X["test"])[:, 1]
    dos = days_of_stock(test).to_numpy()
    results = [
        evaluate("days_of_stock_rule", test, -dos, episodes),
        evaluate("logistic_regression", test, score["logistic_regression"], episodes, score["logistic_regression"]),
        evaluate("gradient_boosting", test, score["gradient_boosting"], episodes, score["gradient_boosting"]),
        evaluate(f"final ({family}, calibrated)", test, p_test, episodes, p_test),
    ]
    flag_rule = dos < HORIZON  # the common rule: review anything with under a week of stock
    flag_model = p_test >= threshold
    days = test["date"].nunique()
    threshold_view = {
        name: {
            "alerts_per_day": round(float(flags.sum() / days), 1),
            "precision": round(float(test["y"][flags].mean()), 3),
            "recall": round(float(test["y"][flags].sum() / test["y"].sum()), 3),
        }
        for name, flags in (("days_of_stock_below_7", flag_rule), ("model_threshold", flag_model))
    }

    capacity = [
        {"alerts_per_day": k, **{name: alert_metrics(test, sc, episodes, k)["episode_recall"]
                                 for name, sc in (("rule_episode_recall", -dos), ("model_episode_recall", p_test))}}
        for k in (3, 5, 10)
    ]

    # 4. Subgroups: slow, medium and fast movers, by average daily use in training.
    use = train.groupby("item_id")["dispensed_last_28d"].mean()
    tier = pd.qcut(use, 3, labels=["slow", "medium", "fast"])
    subgroups = []
    for name in tier.cat.categories:
        mask = test["item_id"].map(tier).eq(name).to_numpy()
        y = test["y"].to_numpy()[mask]
        subgroups.append({
            "demand_tier": name,
            "rows": int(mask.sum()),
            "positive_rate": round(float(y.mean()), 3),
            "average_precision_model": round(float(average_precision_score(y, p_test[mask])), 3),
            "average_precision_rule": round(float(average_precision_score(y, -dos[mask])), 3),
        })

    imp = permutation_importance(final, X["test"], test["y"], scoring="average_precision",
                                 n_repeats=5, random_state=SEED)
    importance = pd.DataFrame({"feature": FEATURES, "mean": imp.importances_mean, "std": imp.importances_std})
    importance = importance.sort_values("mean", ascending=False)

    metrics = {
        "split": {
            name: {"rows": len(part), "from": str(part["date"].min().date()), "to": str(part["date"].max().date()),
                   "positive_rate": round(float(part["y"].mean()), 3)}
            for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "validation": {
            "days_of_stock_rule_average_precision": round(float(rule_val_ap), 3),
            **{f"best_{f}": {"average_precision": round(float(b[0]), 3), "params": {k: str(v) for k, v in b[1].items()}}
               for f, b in best.items()},
            "chosen_family": family,
        },
        "test": results,
        "test_threshold_view": threshold_view,
        "test_alert_capacity_sensitivity": capacity,
        "test_by_demand_tier": subgroups,
        "permutation_importance_test": importance.round(4).to_dict("records"),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / "stockout_model.joblib")
    (MODELS_DIR / "metadata.json").write_text(json.dumps({
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "synthetic extract from pipeline/simulate.py (seed %d)" % SEED,
        "family": family,
        "params": {k: str(v) for k, v in params.items()},
        "calibration": "sigmoid, fitted on validation",
        "horizon_days": HORIZON,
        "alert_capacity_per_day": ALERT_CAPACITY,
        "review_threshold": round(threshold, 4),
        "snapshot_fields": SNAPSHOT,
        "features": FEATURES,
        "scikit_learn": sklearn.__version__,
        "test_average_precision": results[-1]["average_precision"],
        "test_episode_recall_at_capacity": results[-1]["episode_recall"],
    }, indent=2))

    plot(test, dos, score, p_test, importance, family)
    print(json.dumps({"capacity": capacity, "chosen": family}, indent=2))


def plot(test, dos, score, p_test, importance, family) -> None:
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    y = test["y"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label, s, colour in (("Days-of-stock rule", -dos, BLUE),
                             ("Logistic regression", score["logistic_regression"], ORANGE),
                             ("Gradient boosting", score["gradient_boosting"], AQUA)):
        precision, recall, _ = precision_recall_curve(y, s)
        ax.plot(recall, precision, color=colour, linewidth=2,
                label=f"{label} (AP {average_precision_score(y, s):.2f})")
    ax.axhline(y.mean(), color=INK_2, linewidth=1, linestyle="--")
    ax.text(0.01, y.mean() + 0.015, f"Base rate {y.mean():.2f}", color=INK_2, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    style(ax, "Precision and recall on the test period", "Recall", "Precision")
    save(fig, "pr_curves.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    name = family.replace("_", " ").capitalize()
    for label, p, colour in ((f"{name}, before calibration", score[family], ORANGE),
                             (f"{name}, after calibration", p_test, BLUE)):
        observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, color=colour, linewidth=2, marker="o", markersize=5, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    style(ax, "Calibration on the test period", "Predicted probability", "Observed stockout rate")
    save(fig, "calibration.png")

    top = importance.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(top["feature"], top["mean"], xerr=top["std"], color=BLUE, height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1})
    style(ax, "Permutation importance, final model", "Drop in test average precision", "")
    ax.grid(axis="y", visible=False)
    save(fig, "importance.png")


if __name__ == "__main__":
    main()
