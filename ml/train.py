#!/usr/bin/env python3
"""Train and honestly evaluate Medrano Mercantile sale price and time-to-sell models.

A model is registered only if it beats the best simple baseline with a 95% bootstrap
interval that excludes zero. Fields known only after a sale, such as platform, are not used.
"""
import json
import os
import warnings
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
MIN_CATEGORY_SIZE = 4
FOLDS = 5
REPEATS = 20
BOOTSTRAPS = 4000
PERMUTATIONS = 100
HOLDOUT_SHARE = 0.25
SELL_WINDOW_DAYS = 30
TIME_ZERO_OFFSET = 0.5  # same-day sales still need a positive duration
OUTPUT_DIR = os.environ.get("ML_OUTPUT_DIR", "/opt/mercantile-ml/output")

QUERY = """
select sku, item_name, category, is_sold, listed_date, sold_date, asking_price, sale_price
from marts.dim_items
where listed_date is not null and asking_price > 0
"""


def load_items():
    with psycopg.connect(
        host=os.environ["ML_PG_HOST"], dbname=os.environ["ML_PG_DB"],
        user=os.environ["ML_PG_USER"], password=os.environ["ML_PG_PASSWORD"],
        options="-c default_transaction_read_only=on",
    ) as conn, conn.cursor() as cur:
        cur.execute(QUERY)
        columns = [c.name for c in cur.description]
        items = pd.DataFrame(cur.fetchall(), columns=columns)
    for column in ("asking_price", "sale_price"):
        items[column] = pd.to_numeric(items[column])
    for column in ("listed_date", "sold_date"):
        items[column] = pd.to_datetime(items[column])
    counts = items["category"].value_counts()
    keep = counts[counts >= MIN_CATEGORY_SIZE].index
    items["category_group"] = items["category"].where(items["category"].isin(keep), "Other")
    today = pd.Timestamp(date.today())
    end = items["sold_date"].fillna(today)
    items["duration"] = (end - items["listed_date"]).dt.days.clip(lower=0) + TIME_ZERO_OFFSET
    items["event"] = items["is_sold"].astype(int)
    return items


class GlobalMedian(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class CategoryMedian(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.fallback_ = float(y.median())
        self.values_ = y.groupby(X["category_group"]).median().to_dict()
        return self

    def predict(self, X):
        return X["category_group"].map(self.values_).fillna(self.fallback_).to_numpy()


class AskingPrice(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["asking_price"].to_numpy()


class AskingTimesRatio(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.ratio_ = float(np.median(np.asarray(y) / X["asking_price"].to_numpy()))
        return self

    def predict(self, X):
        return X["asking_price"].to_numpy() * self.ratio_


def log_scale():
    return TransformedTargetRegressor(
        regressor=RidgeCV(alphas=np.logspace(-2, 3, 20)), func=np.log1p, inverse_func=np.expm1
    )


def text_features():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def category_features():
    return OneHotEncoder(handle_unknown="ignore")


def asking_features():
    return Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())])


def ridge(parts):
    return Pipeline([("features", ColumnTransformer(parts)), ("model", log_scale())])


def forest(parts):
    return Pipeline([
        ("features", ColumnTransformer(parts, sparse_threshold=0)),
        ("model", TransformedTargetRegressor(
            regressor=RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED),
            func=np.log1p, inverse_func=np.expm1)),
    ])


CATEGORY = ("category", category_features(), ["category_group"])
TEXT = ("text", text_features(), "item_name")
ASKING = ("asking", asking_features(), ["asking_price"])


def out_of_fold(model, X, y):
    """Average out-of-fold prediction for each row across repeated K-fold splits."""
    totals = np.zeros(len(X))
    for repeat in range(REPEATS):
        folds = KFold(FOLDS, shuffle=True, random_state=SEED + repeat)
        for train, test in folds.split(X):
            fitted = _clone(model).fit(X.iloc[train], y[train])
            totals[test] += fitted.predict(X.iloc[test])
    return totals / REPEATS


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def bootstrap_gain(base_errors, model_errors):
    rng = np.random.default_rng(SEED)
    gaps = base_errors - model_errors
    samples = rng.integers(0, len(gaps), (BOOTSTRAPS, len(gaps)))
    means = gaps[samples].mean(axis=1)
    return float(gaps.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holdout_mae(model, X, y, order):
    cut = int(len(X) * (1 - HOLDOUT_SHARE))
    train, test = order[:cut], order[cut:]
    fitted = _clone(model).fit(X.iloc[train], y[train])
    return float(np.mean(np.abs(fitted.predict(X.iloc[test]) - y[test])))


def evaluate_regression(experiment, baselines, candidates, X, y, order, register_name, sold):
    mlflow.set_experiment(experiment)
    results = {}
    errors = {}
    predictions = {}
    for name, model in {**baselines, **candidates}.items():
        oof = out_of_fold(model, X, y)
        err = np.abs(oof - y)
        errors[name], predictions[name] = err, oof
        results[name] = {
            "mae": float(err.mean()), "median_ae": float(np.median(err)),
            "median_pct_error": float(np.median(err / y)),
            "holdout_mae": holdout_mae(model, X, y, order),
            "kind": "baseline" if name in baselines else "model",
        }
    best_base = min(baselines, key=lambda n: results[n]["mae"])
    best_model = min(candidates, key=lambda n: results[n]["mae"])
    gain, low, high = bootstrap_gain(errors[best_base], errors[best_model])
    passes = low > 0
    for name, row in results.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"kind": row["kind"], "rows": len(X), "best_baseline": best_base})
            mlflow.log_params({"folds": FOLDS, "repeats": REPEATS, "holdout_share": HOLDOUT_SHARE})
            mlflow.log_metrics({k: v for k, v in row.items() if k != "kind"})

    with mlflow.start_run(run_name=f"verdict: {best_model} vs {best_base}"):
        mlflow.set_tags({"passes_gate": str(passes), "best_model": best_model, "best_baseline": best_base})
        mlflow.log_metrics({"gain_mae": gain, "gain_low": low, "gain_high": high})
        figure, axis = plt.subplots(figsize=(5, 5))
        axis.scatter(y, predictions[best_model], s=18, color="#e8590c")
        limits = [max(1, y.min()), y.max()]
        axis.plot(limits, limits, color="#16181d", linewidth=1)
        axis.set(xscale="log", yscale="log", xlabel="actual sale price ($)", ylabel="predicted, out of fold ($)", title=best_model)
        figure.tight_layout()
        mlflow.log_figure(figure, "predicted_vs_actual.png")
        plt.close(figure)
        final = _clone(candidates[best_model]).fit(X, y)
        if passes:
            mlflow.sklearn.log_model(final, name="model", registered_model_name=register_name)
    fallback = _clone(baselines[best_base]).fit(X, y)
    return {
        "rows": len(X), "sold_rows": sold, "results": results, "best_baseline": best_base, "best_model": best_model,
        "gain_mae": gain, "gain_ci": [low, high], "passes_gate": passes,
    }, final if passes else fallback


def cox_frame(items):
    frame = pd.get_dummies(items[["category_group"]], drop_first=True, dtype=float)
    frame["log_asking"] = np.log1p(items["asking_price"])
    frame["duration"], frame["event"] = items["duration"], items["event"]
    return frame


def cox_cv_cindex(frame, repeats=REPEATS, seed=SEED):
    scores = []
    for repeat in range(repeats):
        for train, test in KFold(FOLDS, shuffle=True, random_state=seed + repeat).split(frame):
            # a category can be missing or uniform in a small training split, which breaks the fit
            part = frame.iloc[train]
            part = part.loc[:, part.nunique() > 1]
            fitted = CoxPHFitter(penalizer=0.2).fit(part, "duration", "event")
            risk = fitted.predict_partial_hazard(frame.iloc[test])
            scores.append(concordance_index(frame.iloc[test]["duration"], -risk, frame.iloc[test]["event"]))
    return float(np.mean(scores))


def evaluate_survival(items):
    mlflow.set_experiment("mercantile-time-to-sell")
    frame = cox_frame(items)
    observed = cox_cv_cindex(frame)
    rng = np.random.default_rng(SEED)
    null = []
    for i in range(PERMUTATIONS):
        shuffled = frame.copy()
        shuffled[["duration", "event"]] = frame[["duration", "event"]].to_numpy()[rng.permutation(len(frame))]
        null.append(cox_cv_cindex(shuffled, repeats=1, seed=SEED + i))
    p_value = float((1 + sum(n >= observed for n in null)) / (1 + PERMUTATIONS))
    passes = p_value < 0.05

    fitter = KaplanMeierFitter().fit(items["duration"], items["event"])
    sold_by = {f"sold_by_{d}d": float(1 - fitter.predict(d)) for d in (14, 30, 60, 90)}
    categories = {}
    for name, part in items.groupby("category_group"):
        by_group = KaplanMeierFitter().fit(part["duration"], part["event"])
        median = by_group.median_survival_time_
        categories[name] = {"items": len(part), "sold": int(part["event"].sum()), "median_days": None if np.isinf(median) else float(median), "sold_by_30d": float(1 - by_group.predict(30))}

    final = CoxPHFitter(penalizer=0.2).fit(frame, "duration", "event")
    with mlflow.start_run(run_name="cox: category + asking price"):
        mlflow.set_tags({"passes_gate": str(passes), "rows": len(frame), "events": int(frame["event"].sum())})
        mlflow.log_params({"penalizer": 0.2, "folds": FOLDS, "repeats": REPEATS, "permutations": PERMUTATIONS})
        mlflow.log_metrics({"cv_c_index": observed, "permutation_p": p_value, "null_c_index_mean": float(np.mean(null)), **sold_by})
        figure, axis = plt.subplots(figsize=(6, 4))
        fitter.plot_survival_function(ax=axis, color="#e8590c", legend=False)
        axis.set(xlabel="days since listed", ylabel="share still unsold", title="All items, Kaplan-Meier")
        figure.tight_layout()
        mlflow.log_figure(figure, "time_to_sell.png")
        plt.close(figure)
        mlflow.log_dict(categories, "by_category.json")
    curve = fitter.survival_function_.reset_index()
    curve_points = [[float(day), float(1 - unsold)] for day, unsold in curve.to_numpy()]
    report = {"rows": len(frame), "events": int(frame["event"].sum()), "median_days": float(fitter.median_survival_time_), **sold_by, "curve": curve_points,
              "cv_c_index": observed, "null_c_index": float(np.mean(null)), "permutation_p": p_value, "passes_gate": passes, "by_category": categories}
    return report, final, frame, fitter


def sell_probability(items, final, frame, passes, overall):
    """Chance each unsold item sells in the next window, given how long it has already waited."""
    unsold = items[items["event"] == 0]
    rows = frame.loc[unsold.index]
    # Past the longest observed sale the curve goes flat, which would read as a false 0% chance.
    longest_sale = items.loc[items["event"] == 1, "duration"].max()
    if passes:
        now = final.predict_survival_function(rows, times=sorted(set(unsold["duration"])))
        later = final.predict_survival_function(rows, times=sorted(set(unsold["duration"] + SELL_WINDOW_DAYS)))
        chance = pd.Series({i: 1 - later[i].loc[unsold.loc[i, "duration"] + SELL_WINDOW_DAYS] / max(now[i].loc[unsold.loc[i, "duration"]], 1e-6) for i in unsold.index})
    else:
        now = overall.predict(unsold["duration"])
        later = overall.predict(unsold["duration"] + SELL_WINDOW_DAYS)
        chance = pd.Series(1 - later.to_numpy() / np.clip(now.to_numpy(), 1e-6, None), index=unsold.index)
    return chance.where(unsold["duration"] + SELL_WINDOW_DAYS <= longest_sale)


def scorecard_row(question, result):
    best = result["results"][result["best_model"]]
    base = result["results"][result["best_baseline"]]
    return (question, result["best_baseline"], base["mae"], result["best_model"], best["mae"], "average error, dollars", result["passes_gate"])


def publish_to_warehouse(report, predictions):
    """Replace the dashboard tables in one transaction, using a role that can only write to the ml schema."""
    speed = report["time_to_sell"]
    scorecard = [
        scorecard_row("Sale price from category and name", report["list_price"]),
        scorecard_row("Sale price given asking price", report["sale_given_asking"]),
        ("Time to sell", "shuffled labels", speed["null_c_index"], "Cox model", speed["cv_c_index"], "C-index, higher is better", speed["passes_gate"]),
    ]
    chance = f"chance_sells_next_{SELL_WINDOW_DAYS}d"
    with psycopg.connect(
        host=os.environ["ML_PG_HOST"], dbname=os.environ["ML_PG_DB"],
        user=os.environ["ML_PG_WRITE_USER"], password=os.environ["ML_PG_WRITE_PASSWORD"],
    ) as conn, conn.cursor() as cur:
        cur.execute("""
            create table if not exists ml.model_scorecard (question text, best_baseline text, baseline_score numeric, best_model text, model_score numeric, metric text, passes_gate boolean, trained_at timestamptz default now());
            create table if not exists ml.listed_item_predictions (sku text, item_name text, category text, asking_price numeric, days_listed int, chance_sells_next_30d numeric, trained_at timestamptz default now());
            create table if not exists ml.time_to_sell_by_category (category text, items int, sold int, median_days numeric, sold_by_30d numeric, trained_at timestamptz default now());
            create table if not exists ml.time_to_sell_summary (items int, sold int, median_days numeric, sold_by_14d numeric, sold_by_30d numeric, sold_by_60d numeric, sold_by_90d numeric, c_index numeric, permutation_p numeric, trained_at timestamptz default now());
            truncate ml.model_scorecard, ml.listed_item_predictions, ml.time_to_sell_by_category, ml.time_to_sell_summary;
        """)
        cur.executemany("insert into ml.model_scorecard (question,best_baseline,baseline_score,best_model,model_score,metric,passes_gate) values (%s,%s,%s,%s,%s,%s,%s)", scorecard)
        cur.executemany(
            "insert into ml.listed_item_predictions (sku,item_name,category,asking_price,days_listed,chance_sells_next_30d) values (%s,%s,%s,%s,%s,%s)",
            [(r.sku, r.item_name, r.category, float(r.asking_price), int(r.days_listed), None if pd.isna(getattr(r, chance)) else float(getattr(r, chance))) for r in predictions.itertuples()],
        )
        cur.executemany(
            "insert into ml.time_to_sell_by_category (category,items,sold,median_days,sold_by_30d) values (%s,%s,%s,%s,%s)",
            [(name, c["items"], c["sold"], c["median_days"], c["sold_by_30d"]) for name, c in speed["by_category"].items()],
        )
        cur.execute(
            "insert into ml.time_to_sell_summary (items,sold,median_days,sold_by_14d,sold_by_30d,sold_by_60d,sold_by_90d,c_index,permutation_p) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (speed["rows"], speed["events"], speed["median_days"], speed["sold_by_14d"], speed["sold_by_30d"], speed["sold_by_60d"], speed["sold_by_90d"], speed["cv_c_index"], speed["permutation_p"]),
        )


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = load_items()
    sold = items[items["event"] == 1].reset_index(drop=True)
    y = sold["sale_price"].to_numpy(dtype=float)
    order = np.argsort(sold["sold_date"].to_numpy())

    pricing, price_model = evaluate_regression(
        "mercantile-list-price",
        {"global median": GlobalMedian(), "category median": CategoryMedian()},
        {"ridge: category": ridge([CATEGORY]), "ridge: category + name": ridge([CATEGORY, TEXT]), "forest: category + name": forest([CATEGORY, TEXT])},
        sold, y, order, "mercantile-list-price", len(sold),
    )
    realized, realized_model = evaluate_regression(
        "mercantile-sale-given-asking",
        {"asking price": AskingPrice(), "asking x median ratio": AskingTimesRatio()},
        {"ridge: asking": ridge([ASKING]), "ridge: asking + category": ridge([ASKING, CATEGORY]), "ridge: asking + category + name": ridge([ASKING, CATEGORY, TEXT])},
        sold, y, order, "mercantile-sale-given-asking", len(sold),
    )
    speed, cox, frame, overall = evaluate_survival(items)

    unsold = items[items["event"] == 0].copy()
    unsold["days_listed"] = (unsold["duration"] - TIME_ZERO_OFFSET).astype(int)
    columns = ["sku", "item_name", "category", "asking_price", "days_listed"]
    # A price column is published only when its model passed the gate. A baseline's output isn't a prediction.
    if pricing["passes_gate"]:
        unsold["price_estimate_without_asking"] = price_model.predict(unsold).round(0)
        columns.append("price_estimate_without_asking")
    if realized["passes_gate"]:
        unsold["expected_sale_at_current_ask"] = realized_model.predict(unsold).round(0)
        columns.append("expected_sale_at_current_ask")
    unsold[f"chance_sells_next_{SELL_WINDOW_DAYS}d"] = sell_probability(items, cox, frame, speed["passes_gate"], overall).round(2)
    columns.append(f"chance_sells_next_{SELL_WINDOW_DAYS}d")
    unsold = unsold[columns].sort_values("days_listed", ascending=False)
    path = os.path.join(OUTPUT_DIR, "listed_item_predictions.csv")
    unsold.to_csv(path, index=False)
    report = {"run_date": str(date.today()), "items": len(items), "list_price": pricing, "sale_given_asking": realized, "time_to_sell": speed}
    with open(os.path.join(OUTPUT_DIR, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    mlflow.set_experiment("mercantile-time-to-sell")
    with mlflow.start_run(run_name="listed item predictions"):
        mlflow.log_artifact(path)
    publish_to_warehouse(report, unsold)
    print(json.dumps(report, indent=1))
    print(unsold.drop(columns=["sku"]).to_string(index=False))


if __name__ == "__main__":
    main()
