"""
Fair per-city evaluation of ARIMA vs persistence vs stacked ensemble.

Drop-in usage inside main(), after the stacked model is fitted:

    from fair_arima_eval import evaluate_all_per_city
    table2, per_city = evaluate_all_per_city(df_full, X_test, y_pred_stack)
    table2.to_csv(out_path("table2_per_city_median.csv"))

Design rules (the point of this file):
  1. SAME rows for every model: the test rows are exactly X_test.index
     (same per-city 75/25 split used by prepare_data), not a re-split of df_raw.
  2. SAME aggregation for every model: R2/MAE/RMSE computed per city, then
     the MEDIAN across cities (the doc's convention).
  3. SAME units: original deg C and %.
  4. Persistence uses the lag-1 column inside each city, so it never reads
     across a city boundary (a pooled y_test.shift(1) does).
  5. Baselines/ARIMA variants that respect the ~6 h cadence (diurnal cycle,
     period = 4 steps) are added; plain non-seasonal ARIMA cannot see it.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ORDERS = ((1, 0, 0), (1, 0, 1), (2, 0, 0), (2, 0, 1))


def _harmonics(datetimes, k=2):
    h = pd.to_datetime(datetimes).dt.hour.values + pd.to_datetime(datetimes).dt.minute.values / 60
    cols = []
    for j in range(1, k + 1):
        cols += [np.sin(2 * np.pi * j * h / 24), np.cos(2 * np.pi * j * h / 24)]
    return np.column_stack(cols)


def arimax_city(y, X, n_tr, orders=ORDERS):
    """AIC-select on TRAIN only, then one-step-ahead rolling forecast over the
    test rows with parameters fixed and the state updated by observed values."""
    best = None
    for o in orders:
        try:
            r = SARIMAX(y[:n_tr], exog=X[:n_tr], order=o, trend="c").fit(disp=False)
        except Exception:
            continue
        if best is None or r.aic < best[0].aic:
            best = (r, o)
    if best is None:
        return np.full(len(y) - n_tr, np.nan), None
    res, order = best
    preds = []
    for t in range(n_tr, len(y)):
        preds.append(float(np.asarray(res.forecast(1, exog=X[t:t + 1]))[0]))
        res = res.append(y[t:t + 1], exog=X[t:t + 1], refit=False)
    return np.array(preds), order


def _safe_r2(y, p, min_var=1e-3):
    return np.nan if (len(y) < 2 or np.var(y) < min_var) else r2_score(y, p)


def evaluate_all_per_city(df_full, X_test, y_pred_stack, train_ratio=0.75, k_harm=2):
    d = df_full.copy()
    for tgt in ("temp", "hum"):
        d[f"{tgt}_lag4"] = d.groupby("city")[tgt].shift(4)  # seasonal naive (24 h)
    test = d.loc[X_test.index].copy()
    test["stack_temp"], test["stack_hum"] = y_pred_stack[:, 0], y_pred_stack[:, 1]
    test["pers_temp"], test["pers_hum"] = test["temp_lag1"], test["hum_lag1"]
    test["snaive_temp"], test["snaive_hum"] = test["temp_lag4"], test["hum_lag4"]

    # ARIMAX with diurnal harmonics, fitted per city on that city's TRAIN rows
    for tgt in ("temp", "hum"):
        test[f"arimax_{tgt}"] = np.nan
    for city, g in d.groupby("city"):
        g = g.sort_values("datetime")
        n_tr = int(len(g) * train_ratio)
        test_idx = g.index[n_tr:]
        if len(test_idx) == 0 or n_tr < 20:
            continue
        X = _harmonics(g["datetime"], k_harm)
        for tgt in ("temp", "hum"):
            p, _ = arimax_city(g[tgt].values.astype(float), X, n_tr)
            test.loc[test_idx.intersection(test.index), f"arimax_{tgt}"] = p[: len(test_idx)]

    models = {"Persistence (lag-1)": "pers", "Seasonal naive (lag-4, 24 h)": "snaive",
              "ARIMAX (diurnal harmonics)": "arimax", "Stacked Ensemble": "stack"}
    rows, per_city = [], []
    for name, key in models.items():
        rec = {"Model": name}
        for tgt in ("temp", "hum"):
            sub = test.dropna(subset=[f"{key}_{tgt}", tgt])
            m = sub.groupby("city").apply(lambda g: pd.Series({
                "r2": _safe_r2(g[tgt].values, g[f"{key}_{tgt}"].values),
                "mae": mean_absolute_error(g[tgt], g[f"{key}_{tgt}"]),
                "rmse": float(np.sqrt(mean_squared_error(g[tgt], g[f"{key}_{tgt}"]))),
                "n": len(g)}))
            m["model"], m["target"] = name, tgt
            per_city.append(m.reset_index())
            rec.update({f"{tgt}_r2": m["r2"].median(), f"{tgt}_mae": m["mae"].median(),
                        f"{tgt}_rmse": m["rmse"].median(), f"{tgt}_n_rows": int(m["n"].sum())})
        rows.append(rec)
    return pd.DataFrame(rows).set_index("Model"), pd.concat(per_city, ignore_index=True)