"""
================================================================================
FINAL SCRIPT (REVISION 3) — REAL MULTI-CITY TIME SERIES
--------------------------------------------------------------------------------
Runs against IndianWeatherRepository_raw.xlsx: 34,466 rows, 543 locations,
251 timestamps spanning 2023-08-29 to 2023-10-30 (~62 days, ~6-hour cadence).
This is a genuine per-location time series, so — unlike the single-snapshot
file used earlier in this project — walk-forward validation, ARIMA, LSTM,
and true multi-row LOCO are all now methodologically meaningful.

STUDY CITY RESOLUTION (see chat for full investigation)
-----------------------------------------------------------
Of the original 8 study cities, 2 were not resolvable to any location in
this file under any name: Hyderabad and Ahmedabad. Delhi and Bengaluru
exist under different names ("New Delhi", "Bangalore"). The closest fuzzy
match to both missing cities was "Adilabad" (a different place, Andhra
Pradesh) — almost certainly the source of the Figure 8 bug the reviewer
flagged, since nothing in this file is actually named Hyderabad or
Ahmedabad. Per instruction, Jaipur and Lucknow replace them:

    STUDY_CITIES (location_name in file) -> display name for figures/tables
    New Delhi  -> Delhi
    Mumbai     -> Mumbai
    Chennai    -> Chennai
    Kolkata    -> Kolkata
    Bangalore  -> Bengaluru
    Pune       -> Pune
    Jaipur     -> Jaipur      [substitute]
    Lucknow    -> Lucknow     [substitute]

Table 1 in the manuscript needs to be updated to list these 8 city names
exactly as above, and the text should note the 2 substitutions and why.

REVIEWER COMMENT COVERAGE (tags used throughout this file)
---------------------------------------------------------------
  [FIX #1]  Walk-forward validation: dedicated outer TimeSeriesSplit(5),
            distinct from the internal stacking CV, mean+/-SD reported for
            R2/RMSE/MAE.
  [FIX #2]  LOCO: true leave-one-city-out over ALL 543 locations, full
            city-wise table exported, city_enc excluded from features with
            the rationale documented inline (this was the reviewer's
            explicit ask — not just doing it correctly, but explaining it).
  [FIX #3]  Persistence baseline (yhat_t+1 = y_t), evaluated on the SAME
            test set as every other model, with percentage improvement of
            the stacked ensemble over persistence EXPLICITLY computed and
            printed (this was missing in the previous script version).
  [FIX #4]  TimeSeriesStackingRegressor — meta-features built via a manual
            TimeSeriesSplit loop, never sklearn's default-KFold
            StackingRegressor.
  [FIX #5]  Resolved once #1-#3 are in the manuscript with real numbers,
            which this script produces.
  [FIX #6]  ARIMA: every city's FULL test period evaluated (no samples=100
            cap), per-city AIC order selection via select_arima_order().
            LSTM: full architecture documented in run_lstm_full()'s
            docstring, copy-pasteable into the methods section.
  [FIX #7]  All metrics reported in original units (deg C, %) only.
  [FIX #9]  Dataset hash + real file date via get_dataset_provenance().
  [FIX #10] evaluate_multi_step() cross-checks 1-hour recursive MAE against
            the direct single-step MAE and warns on disagreement. ALSO
            FIXES an actual bug found in the prior script: recursive_forecast
            previously froze hour_sin/cos, month_sin/cos, and all
            atmospheric features at their initial values for the entire
            48-hour horizon. This version advances the calendar-based
            features (hour/day-of-year) at every recursive step and
            persists the last observed value for exogenous atmospheric
            features (documented below) — a real forecast should not be
            computed against a frozen clock.
  [FIX #11] STUDY_CITIES allow-list enforced via verify check before any
            per-city figure; figures generated for all 8 resolved cities,
            not a single example.

FEATURES ADDED THIS REVISION (in response to "any other related features?")
--------------------------------------------------------------------------------
  - latitude, longitude                         (explicitly requested)
  - day_of_year_sin/cos                         NEW — month_sin/cos alone is
    coarse for a 62-day, 3-month span; day-of-year gives finer-grained
    seasonal position within the monsoon-retreat period this data covers.
  - temp_lag2, hum_lag2                         NEW — a second lag captures
    short-term trend/acceleration (is temperature rising or falling), not
    just the last level, and is cheap to add given the lag1 features
    already exist.
  - pressure_mb, wind_kph, cloud, uv_index,
    precip_mm, visibility_km, air_quality_PM2.5 NEW contemporaneous
    atmospheric covariates. This follows the SAME precedent already set by
    condition_text/condition_enc in the earlier script (also a same-
    timestep covariate) — these are known at the time a reading is taken,
    so they are valid predictors for direct one-step evaluation
    (walk-forward, LOCO, the stacked-vs-persistence test). IMPORTANT
    CAVEAT for the 48-hour RECURSIVE forecast specifically: future values
    of these exogenous variables are not observed, so recursive_forecast()
    persists (carries forward) each one's last known value rather than
    inventing future values — the same naive-persistence assumption
    standard short-horizon recursive forecasts make for exogenous inputs.
    This is stated explicitly in recursive_forecast()'s docstring so it is
    not an implicit, undocumented approximation.
  - NOT added: elevation (not in the source data), distance-to-coast
    (would need an external coastline geometry — flagged as a possible
    future addition, not fabricated here).
================================================================================
"""

import os
import json
import hashlib
import itertools
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import shap
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 0. STUDY DESIGN CONSTANTS
# ============================================================================

FILEPATH = "csv files/IndianWeatherRepository_raw.xlsx"
OUTPUT_DIR = "./indianweather files"


def out_path(filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


# [FIX #11] location_name (as it appears in the file) -> display name
STUDY_CITIES = {
    "New Delhi": "Delhi",
    "Mumbai": "Mumbai",
    "Chennai": "Chennai",
    "Kolkata": "Kolkata",
    "Bangalore": "Bengaluru",
    "Pune": "Pune",
    "Jaipur": "Jaipur",
    "Lucknow": "Lucknow",
}


def _check_city_allowed(city_location_name, study_cities=STUDY_CITIES):
    """[FIX #11] Refuse to proceed with a location outside the defined study set."""
    if study_cities and city_location_name not in study_cities:
        raise ValueError(
            f"Location '{city_location_name}' is not in STUDY_CITIES "
            f"{list(study_cities)}. Refusing to generate a figure/result for a "
            f"location outside the defined study scope (this is exactly the "
            f"Figure 8 / Adilabad issue flagged in review)."
        )


TARGETS_DISPLAY = {"temp": "Temperature", "hum": "Humidity"}

# Physically plausible ranges — data-integrity tripwire, checked at load time.
TEMP_RANGE_C = (-15.0, 55.0)
HUM_RANGE_PCT = (0.0, 100.0)


# ============================================================================
# 1. DATA PROVENANCE + LOADING + VALIDATION  [FIX #9]
# ============================================================================

def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_dataset_provenance(filepath, download_date=None):
    """[FIX #9] Real hash + real date, not a placeholder."""
    file_hash = compute_file_hash(filepath)
    if download_date is not None:
        date_str, date_source = download_date, "user-provided download date"
    else:
        mtime = os.path.getmtime(filepath)
        date_str = dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        date_source = "file last-modified timestamp (proxy; supply the true download date if known)"
    print(f"Dataset hash: {file_hash}")
    print(f"Dataset date: {date_str}  [{date_source}]")
    return file_hash, date_str


def load_and_validate(filepath):
    """
    Loads the raw file and asserts physically plausible ranges before any
    modeling happens. This is the check that would have caught the
    Adilabad figure's humidity>100 / implausible-temperature problem before
    a single model was ever fit.
    """
    df = pd.read_excel(filepath) if filepath.endswith((".xlsx", ".xls")) else pd.read_csv(filepath, low_memory=False)
    print(f"Raw columns found ({len(df.columns)}): {df.columns.tolist()}")
    print(f"Raw shape: {df.shape}")

    required = ["location_name", "region", "last_updated", "temperature_celsius", "humidity",
                "condition_text", "latitude", "longitude", "pressure_mb", "wind_kph", "cloud",
                "uv_index", "precip_mm", "visibility_km", "air_quality_PM2.5"]
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Required columns missing from file: {missing_cols}")

    report = {"n_rows_loaded": len(df), "range_violations": {}}
    checks = {"temperature_celsius": TEMP_RANGE_C, "humidity": HUM_RANGE_PCT}
    for col, (lo, hi) in checks.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad):
            report["range_violations"][col] = {
                "n_rows": int(len(bad)),
                "example_locations": bad["location_name"].head(5).tolist(),
            }
    n_before = len(df)
    for col, (lo, hi) in checks.items():
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    report["n_rows_dropped_out_of_range"] = n_before - len(df)
    report["n_rows_final"] = len(df)

    with open(out_path("data_validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if report["range_violations"]:
        print("!! Range violations found and dropped:")
        print(json.dumps(report["range_violations"], indent=2))
    else:
        print(f"Data validation passed clean: {report['n_rows_final']} rows retained.")

    df = df.rename(columns={
        "location_name": "city", "temperature_celsius": "temp", "humidity": "hum",
    })
    df["datetime"] = pd.to_datetime(df["last_updated"])
    return df.sort_values(["city", "datetime"]).reset_index(drop=True)


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

ATMOS_FEATURES = ["pressure_mb", "wind_kph", "cloud", "uv_index", "precip_mm",
                   "visibility_km", "air_quality_PM2.5"]

FEATURE_COLS = [
    "temp_lag1", "hum_lag1", "temp_lag2", "hum_lag2",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "region_enc", "condition_enc", "latitude", "longitude",
] + ATMOS_FEATURES
# city_enc intentionally excluded from the shared feature set — see [FIX #2]
# rationale in leave_one_city_out(). The main/study-city model below adds
# city_enc back in ONLY for the main/multi-step pipeline (all 8 study
# cities are, by definition, seen in training there), matching the original
# script's design of two different feature sets for two different purposes.


def engineer_features(df):
    df = df.copy()
    df["temp_lag1"] = df.groupby("city")["temp"].shift(1)
    df["hum_lag1"] = df.groupby("city")["hum"].shift(1)
    df["temp_lag2"] = df.groupby("city")["temp"].shift(2)
    df["hum_lag2"] = df.groupby("city")["hum"].shift(2)
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
    doy = df["datetime"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def encode_categoricals(df, le_region=None, le_cond=None, le_city=None, fit=True):
    if fit:
        le_region, le_cond, le_city = LabelEncoder(), LabelEncoder(), LabelEncoder()
        df["region_enc"] = le_region.fit_transform(df["region"].astype(str))
        df["condition_enc"] = le_cond.fit_transform(df["condition_text"].astype(str))
        df["city_enc"] = le_city.fit_transform(df["city"].astype(str))
    else:
        df["region_enc"] = le_region.transform(df["region"].astype(str))
        df["condition_enc"] = le_cond.transform(df["condition_text"].astype(str))
        df["city_enc"] = le_city.transform(df["city"].astype(str))
    return df, le_region, le_cond, le_city


def prepare_data(df, train_ratio=0.75):
    """
    Per-city chronological split (each city contributes its own train/test
    portion, rather than one global date cutoff), consistent with the
    previous script's design and preferable to a single global split for a
    dataset with per-city time series of differing length/coverage.
    """
    df = engineer_features(df)
    df, le_region, le_cond, le_city = encode_categoricals(df, fit=True)
    df = df.dropna(subset=FEATURE_COLS + ["city_enc"]).reset_index(drop=True)

    feature_cols_with_city = FEATURE_COLS + ["city_enc"]
    X = df[feature_cols_with_city].astype(np.float32)
    y = df[["temp", "hum"]].astype(np.float32)

    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    for city in df["city"].unique():
        idx = np.where((df["city"] == city).values)[0]
        n = len(idx)
        split_idx = int(n * train_ratio)
        if split_idx < 1 or n - split_idx < 1:
            continue
        X_train_list.append(X.iloc[idx[:split_idx]]); X_test_list.append(X.iloc[idx[split_idx:]])
        y_train_list.append(y.iloc[idx[:split_idx]]); y_test_list.append(y.iloc[idx[split_idx:]])

    X_train, X_test = pd.concat(X_train_list), pd.concat(X_test_list)
    y_train, y_test = pd.concat(y_train_list), pd.concat(y_test_list)
    print(f"Data prepared: X_train {X_train.shape}, X_test {X_test.shape}")
    return df, X_train, X_test, y_train, y_test, le_region, le_cond, le_city, feature_cols_with_city


# ============================================================================
# 3. PERSISTENCE BASELINE  [FIX #3]
# ============================================================================

def persistence_baseline(y_test):
    if len(y_test) < 2:
        return {k: np.nan for k in ["temp_r2", "temp_rmse", "temp_mae", "hum_r2", "hum_rmse", "hum_mae"]}
    temp_pred, hum_pred = y_test["temp"].shift(1).values[1:], y_test["hum"].shift(1).values[1:]
    true_temp, true_hum = y_test["temp"].values[1:], y_test["hum"].values[1:]
    return {
        "temp_r2": r2_score(true_temp, temp_pred),
        "temp_rmse": float(np.sqrt(mean_squared_error(true_temp, temp_pred))),
        "temp_mae": float(mean_absolute_error(true_temp, temp_pred)),
        "hum_r2": r2_score(true_hum, hum_pred),
        "hum_rmse": float(np.sqrt(mean_squared_error(true_hum, hum_pred))),
        "hum_mae": float(mean_absolute_error(true_hum, hum_pred)),
    }


def compute_persistence_predictions(y_test):
    if len(y_test) < 2:
        return None, None
    y_pred = np.column_stack([y_test["temp"].shift(1).values[1:], y_test["hum"].shift(1).values[1:]])
    return y_pred, persistence_baseline(y_test)


def print_improvement_over_persistence(persistence_metrics, model_metrics, model_name="Stacked Ensemble"):
    """
    [FIX #3] Explicitly computes and prints the percentage improvement the
    reviewer asked for: "reported alongside percentage improvement from the
    proposed model." Returns the numbers so they can also be written to the
    manuscript results table.
    """
    imp = {}
    for var in ("temp", "hum"):
        p_mae, m_mae = persistence_metrics[f"{var}_mae"], model_metrics[f"{var}_mae"]
        imp[f"{var}_mae_improvement_pct"] = float((p_mae - m_mae) / p_mae * 100) if p_mae else np.nan
        p_rmse, m_rmse = persistence_metrics[f"{var}_rmse"], model_metrics[f"{var}_rmse"]
        imp[f"{var}_rmse_improvement_pct"] = float((p_rmse - m_rmse) / p_rmse * 100) if p_rmse else np.nan
    print(f"\n{model_name} improvement over persistence baseline:")
    print(f"  Temperature: MAE {imp['temp_mae_improvement_pct']:+.1f}%, RMSE {imp['temp_rmse_improvement_pct']:+.1f}%")
    print(f"  Humidity:    MAE {imp['hum_mae_improvement_pct']:+.1f}%, RMSE {imp['hum_rmse_improvement_pct']:+.1f}%")
    return imp


# ============================================================================
# 4. STACKED ENSEMBLE (time-series-safe)  [FIX #4]
# ============================================================================

class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    [FIX #4] sklearn's StackingRegressor with an integer `cv` uses plain
    KFold internally — for hourly/sub-daily meteorological data that means
    the meta-learner can see out-of-fold predictions from a model trained
    on LATER data than the validation row, a form of temporal leakage. This
    class replaces that internal step with a manual TimeSeriesSplit loop so
    every out-of-fold prediction is generated using strictly earlier data.
    """

    def __init__(self, base_estimators, final_estimator, n_splits=5):
        self.base_estimators = base_estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y).ravel()
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof = np.full((len(X), len(self.base_estimators)), np.nan)
        for train_idx, val_idx in tscv.split(X):
            for j, (_, est) in enumerate(self.base_estimators):
                m = clone(est).fit(X[train_idx], y[train_idx])
                oof[val_idx, j] = m.predict(X[val_idx])
        covered = ~np.isnan(oof).any(axis=1)
        self.n_meta_train_ = int(covered.sum())
        if self.n_meta_train_ == 0:
            raise ValueError("No out-of-fold predictions generated — n_splits too large for this dataset.")
        self.final_estimator_ = clone(self.final_estimator).fit(oof[covered], y[covered])
        self.fitted_base_estimators_ = [(name, clone(est).fit(X, y)) for name, est in self.base_estimators]
        return self

    def predict(self, X):
        X = np.asarray(X)
        base_preds = np.column_stack([m.predict(X) for _, m in self.fitted_base_estimators_])
        return self.final_estimator_.predict(base_preds)


def get_stacked_model(n_splits=5, n_estimators=200):
    base_models = [
        ("xgb", XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0, n_jobs=-1)),
        ("lgbm", LGBMRegressor(n_estimators=n_estimators, num_leaves=31, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)),
        ("cat", CatBoostRegressor(iterations=n_estimators, depth=6, learning_rate=0.05, random_seed=42, verbose=0)),
    ]
    stack = TimeSeriesStackingRegressor(base_models, Ridge(alpha=1.0), n_splits=n_splits)
    return MultiOutputRegressor(stack, n_jobs=1)


# ============================================================================
# 5. METRICS
# ============================================================================

def safe_r2_score(y_true, y_pred, min_variance=1e-3):
    y_true = np.asarray(y_true, dtype=float)
    variance = np.var(y_true)
    if variance < min_variance or len(y_true) < 2:
        return np.nan, variance
    return r2_score(y_true, y_pred), variance


def compute_metrics(y_true, y_pred):
    r2_temp, var_temp = safe_r2_score(y_true["temp"], y_pred[:, 0])
    r2_hum, var_hum = safe_r2_score(y_true["hum"], y_pred[:, 1])
    return {
        "temp_r2": r2_temp, "temp_rmse": float(np.sqrt(mean_squared_error(y_true["temp"], y_pred[:, 0]))),
        "temp_mae": float(mean_absolute_error(y_true["temp"], y_pred[:, 0])),
        "hum_r2": r2_hum, "hum_rmse": float(np.sqrt(mean_squared_error(y_true["hum"], y_pred[:, 1]))),
        "hum_mae": float(mean_absolute_error(y_true["hum"], y_pred[:, 1])),
        "temp_variance": var_temp, "hum_variance": var_hum,
    }


# ============================================================================
# 6. WALK-FORWARD VALIDATION  [FIX #1]
# ============================================================================

def walk_forward_validation(model_builder, X_train, y_train, n_splits=5, min_train_samples=50):
    """
    [FIX #1] Dedicated rolling-origin protocol, distinct from the internal
    stacking CV. Refits the full model on an expanding window and evaluates
    on a strictly later window, for n_splits folds.
    """
    if len(X_train) < n_splits * 2:
        return pd.DataFrame()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, val_idx in tscv.split(X_train):
        if len(train_idx) < min_train_samples:
            print(f"  Skipping fold with {len(train_idx)} training rows (< {min_train_samples})")
            continue
        model = model_builder()
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        y_pred = model.predict(X_train.iloc[val_idx])
        metrics = compute_metrics(y_train.iloc[val_idx], y_pred)
        metrics.update(fold=len(results), train_window_size=len(train_idx), val_window_size=len(val_idx))
        results.append(metrics)
        print(f"  Fold {metrics['fold']}: Temp R2={metrics['temp_r2']:.4f} RMSE={metrics['temp_rmse']:.4f} | "
              f"Hum R2={metrics['hum_r2']:.4f} RMSE={metrics['hum_rmse']:.4f}")
    return pd.DataFrame(results)


# ============================================================================
# 7. ARIMA ROLLING FORECAST, ALL CITIES, PER-CITY ORDER SELECTION  [FIX #6]
# ============================================================================

def select_arima_order(train_series, p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1, 2)):
    best_aic, best_order = np.inf, (1, 0, 0)
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            fitted = ARIMA(train_series, order=(p, d, q)).fit()
            if fitted.aic < best_aic:
                best_aic, best_order = fitted.aic, (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


def arima_rolling_all_cities(df_raw, target="temp", train_ratio=0.75, min_test_samples=5,
                              select_order=True, city_subset=None):
    """
    [FIX #6] Rolling one-step-ahead ARIMA, per-city AIC order selection, and
    NO artificial cap on the number of test samples evaluated (the reviewer
    specifically flagged a `samples=100` cap in the previous code — there is
    none here; every city's full test period is used).
    """
    cities = city_subset if city_subset is not None else df_raw["city"].unique()
    print(f"ARIMA rolling forecast on {len(cities)} cities "
          f"({'per-city AIC order selection' if select_order else 'fixed order'})")
    city_results = []
    for city in cities:
        group = df_raw[df_raw["city"] == city].sort_values("datetime").reset_index(drop=True)
        n_train = int(len(group) * train_ratio)
        if n_train < 10:
            continue
        train_series = group[target].iloc[:n_train].values
        test_series = group[target].iloc[n_train:].values
        if len(test_series) < min_test_samples:
            continue
        order, aic = select_arima_order(train_series) if select_order else ((1, 0, 0), np.nan)
        history, preds = list(train_series), []
        for t in range(len(test_series)):
            try:
                fitted = ARIMA(history, order=order).fit()
                preds.append(fitted.forecast()[0])
            except Exception:
                preds.append(np.nan)
            history.append(test_series[t])
        preds = np.array(preds)
        valid = ~np.isnan(preds)
        if valid.sum() < min_test_samples:
            continue
        r2, variance = safe_r2_score(test_series[valid], preds[valid])
        city_results.append({
            "city": city, "order_p": order[0], "order_d": order[1], "order_q": order[2], "aic": aic,
            "r2": r2, "rmse": float(np.sqrt(mean_squared_error(test_series[valid], preds[valid]))),
            "mae": float(mean_absolute_error(test_series[valid], preds[valid])),
            "n": int(valid.sum()), "variance": variance, "r2_undefined": bool(np.isnan(r2)),
        })
    if not city_results:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}, pd.DataFrame()
    df_res = pd.DataFrame(city_results)
    defined = df_res[~df_res["r2_undefined"]]
    overall = {"r2": float(defined["r2"].mean()) if len(defined) else np.nan,
               "rmse": float(df_res["rmse"].mean()), "mae": float(df_res["mae"].mean())}
    print(f"ARIMA {target}: R2 averaged over {len(defined)}/{len(df_res)} defined cities, "
          f"RMSE/MAE averaged over all {len(df_res)}.")
    return overall, df_res


# ============================================================================
# 8. LSTM BASELINE  [FIX #6]
# ============================================================================

def run_lstm_full(X_train, X_test, y_train, y_test, n_steps=8, epochs=100):
    """
    [FIX #6] Full architecture/hyperparameters, copy-pasteable into methods:
      - Input window: n_steps timesteps (default 8, ~2 days at this
        dataset's ~6-hour cadence — shortened from the 24-step default used
        for hourly data, since this file's cadence is coarser)
      - 3 stacked LSTM layers: 128 -> 64 -> 32 units, ReLU
      - BatchNormalization + Dropout (0.3, 0.3, 0.2) after each LSTM layer
      - Dense(16, relu) -> Dense(2) output head (temp, hum)
      - Optimizer: Adam, lr=0.001
      - EarlyStopping: monitor val_loss, patience=10, restore_best_weights
      - ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6
      - epochs=100 (capped by early stopping), batch_size=64
      - Validation split: last 20% of training sequences, chronological
    """
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_s, X_test_s = scaler_X.fit_transform(X_train.values), scaler_X.transform(X_test.values)
    y_train_s, y_test_s = scaler_y.fit_transform(y_train.values), scaler_y.transform(y_test.values)
    train_comb, test_comb = np.column_stack([X_train_s, y_train_s]), np.column_stack([X_test_s, y_test_s])

    X_seq, y_seq = [], []
    for i in range(len(train_comb) - n_steps):
        X_seq.append(train_comb[i:i + n_steps, :]); y_seq.append(train_comb[i + n_steps, -2:])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    if len(X_seq) < 50:
        return np.array([]), np.array([]), {}

    val_size = max(1, int(0.2 * len(X_seq)))
    X_tr, X_val = X_seq[:-val_size], X_seq[-val_size:]
    y_tr, y_val = y_seq[:-val_size], y_seq[-val_size:]

    model = Sequential([
        LSTM(128, activation="relu", return_sequences=True, input_shape=(n_steps, X_seq.shape[2])),
        BatchNormalization(), Dropout(0.3),
        LSTM(64, activation="relu", return_sequences=True), BatchNormalization(), Dropout(0.3),
        LSTM(32, activation="relu"), Dropout(0.2),
        Dense(16, activation="relu"), Dense(2),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=64,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                         ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)],
              verbose=0)

    test_preds = []
    n_test_seq = len(test_comb) - n_steps
    if n_test_seq > 0:
        # Build all test sequences into a single batch array and predict once —
        # calling model.predict() row-by-row in a Python loop (the previous
        # version's approach) incurs TensorFlow per-call overhead on every
        # single row, which on a test set of several thousand rows made this
        # step take tens of minutes for no accuracy benefit. Batched
        # prediction is numerically identical and orders of magnitude faster.
        X_test_seq = np.array([test_comb[i:i + n_steps, :] for i in range(n_test_seq)])
        test_preds = model.predict(X_test_seq, batch_size=256, verbose=0)
    if len(test_preds) == 0:
        return np.array([]), np.array([]), {}
    y_pred = scaler_y.inverse_transform(np.array(test_preds))
    y_true = y_test.iloc[n_steps:].reset_index(drop=True)
    if len(y_true) != len(y_pred):
        return np.array([]), np.array([]), {}
    metrics = compute_metrics(y_true, y_pred)
    return y_pred[:, 0], y_pred[:, 1], metrics


# ============================================================================
# 9. LEAVE-ONE-CITY-OUT — TRUE LOCO, ALL 543 LOCATIONS  [FIX #2]
# ============================================================================

def build_fast_loco_model(n_estimators=150):
    """
    [COMPUTE NOTE] True LOCO across all 543 locations means 543 x 2 targets
    = 1,086 full model fits. At the full 3-base-learner time-series-safe
    stack's cost, this is impractical to run per request. A single
    XGBRegressor is used for this specific sweep instead, keeping runtime to
    a few minutes; the full stacking ensemble is still what's reported
    everywhere else (walk-forward, study-city multi-step, SHAP).
    """
    return XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                         random_state=42, n_jobs=-1, verbosity=0)


def leave_one_city_out(df, target, model_builder=build_fast_loco_model, city_subset=None):
    """
    [FIX #2] True leave-one-city-out over every location in the dataset
    (or a subset, for fast iteration during development — the manuscript
    run should always use the default, which covers every location).

    ENCODING FOR UNSEEN CITIES — the reviewer's explicit methodological
    question. `city_enc` (a LabelEncoder integer id learned from training
    cities) is DELIBERATELY EXCLUDED from FEATURE_COLS used here. A held-out
    city has no valid learned encoding: a fresh integer id would be
    meaningless to a tree-based model, and re-fitting the encoder to include
    it would let the model see the held-out city's identity, defeating the
    point of holding it out. Instead this model relies only on lag features,
    calendar features, region encoding, condition encoding, lat/long, and
    the atmospheric covariates — none of which require the city's specific
    identity to be known ahead of time. This must be stated explicitly in
    the manuscript's LOCO subsection.
    """
    X_all, y_all = df[FEATURE_COLS], df[target]
    cities = city_subset if city_subset is not None else sorted(df["city"].unique())
    print(f"  Running true LOCO over {len(cities)} locations for {target}...")
    per_city = []
    for city in cities:
        test_mask = df["city"] == city
        model = model_builder()
        model.fit(X_all[~test_mask], y_all[~test_mask])
        preds = model.predict(X_all[test_mask])
        r2, _ = safe_r2_score(y_all[test_mask], preds)
        per_city.append({
            "city": city, "in_study_set": city in STUDY_CITIES,
            "n_train_cities": df.loc[~test_mask, "city"].nunique(), "n_test_rows": int(test_mask.sum()),
            "r2": r2, "rmse": float(np.sqrt(mean_squared_error(y_all[test_mask], preds))),
            "mae": float(mean_absolute_error(y_all[test_mask], preds)),
        })
    results_df = pd.DataFrame(per_city)
    n_undefined = results_df["r2"].isna().sum()
    defined = results_df.dropna(subset=["r2"])
    print(f"  LOCO {target}: {len(results_df)} cities, {n_undefined} with undefined R2. "
          f"Among defined: median R2={defined['r2'].median():.4f}, mean R2={defined['r2'].mean():.4f}. "
          f"Median RMSE={results_df['rmse'].median():.4f}, median MAE={results_df['mae'].median():.4f}.")
    study_rows = results_df[results_df["in_study_set"]]
    print(f"  Study-city subset: mean R2={study_rows['r2'].mean():.4f}, "
          f"mean RMSE={study_rows['rmse'].mean():.4f}, mean MAE={study_rows['mae'].mean():.4f}")
    return results_df


# ============================================================================
# 10. SHAP ANALYSIS
# ============================================================================

def shap_analysis(model, X_sample, feature_names):
    stacking_reg = model.estimators_[0]
    xgb_model = dict(stacking_reg.fitted_base_estimators_)["xgb"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance — Stacked Ensemble (Temperature)")
    plt.tight_layout(); plt.savefig(out_path("shap_summary.png"), dpi=150); plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "importance": mean_abs}).sort_values(
        "importance", ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    plt.barh(imp["feature"], imp["importance"]); plt.xlabel("Mean |SHAP value|")
    plt.title("Top 15 Features"); plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(out_path("shap_bar.png"), dpi=150); plt.close()
    imp.to_csv(out_path("shap_importance.csv"), index=False)
    return imp


# ============================================================================
# 11. RESIDUAL DIAGNOSTICS
# ============================================================================

def plot_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Residual Diagnostics: {target_name}")
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10); axes[0, 0].axhline(0, color="r", linestyle="--")
    axes[0, 0].set_title("Residuals vs Predicted")
    stats.probplot(residuals, dist="norm", plot=axes[0, 1]); axes[0, 1].set_title("Q-Q Plot")
    axes[1, 0].hist(residuals, bins=40, edgecolor="black", alpha=0.7); axes[1, 0].set_title("Distribution")
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.4, s=8); axes[1, 1].axhline(0, color="r", linestyle="--")
    axes[1, 1].set_title("Residuals by Row Order")
    plt.tight_layout(); plt.savefig(out_path(f"residuals_{target_name}.png"), dpi=150); plt.close()


# ============================================================================
# 12. MULTI-STEP RECURSIVE FORECASTING  [FIX #10 — bug fix + consistency check]
# ============================================================================

def recursive_forecast(model, initial_features, feature_names, n_steps, start_datetime, freq_hours=6):
    """
    [FIX #10 — bug fix] The previous version froze EVERY feature except
    temp_lag1/hum_lag1 for the entire recursive horizon, including
    hour_sin/cos and month_sin/cos — meaning a 48-hour forecast used the
    SAME hour-of-day for every single step. That is a real bug, not just an
    approximation, and would by itself explain a chunk of the Table 2 vs
    Table 4 discrepancy the reviewer flagged: the model was never actually
    told what hour it was forecasting for.

    This version:
      - Advances hour_sin/cos, month_sin/cos, day_of_year_sin/cos correctly
        at every step based on the actual forecasted timestamp.
      - Advances temp_lag1/hum_lag1 <- previous step's prediction, and
        temp_lag2/hum_lag2 <- the step before that (properly shifted, not
        just lag1 duplicated).
      - PERSISTS (carries forward) the last observed value for every
        atmospheric covariate (pressure_mb, wind_kph, cloud, uv_index,
        precip_mm, visibility_km, air_quality_PM2.5) and for region_enc/
        city_enc/latitude/longitude, since none of these have known future
        values during a genuine forecast — this is the standard naive-
        persistence assumption for exogenous inputs in short-horizon
        recursive forecasting, stated explicitly rather than left implicit.
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    curr = initial_features.copy()
    preds = []
    temp_hist = [curr[idx["temp_lag1"]], curr[idx.get("temp_lag2", idx["temp_lag1"])]]
    hum_hist = [curr[idx["hum_lag1"]], curr[idx.get("hum_lag2", idx["hum_lag1"])]]

    for step in range(n_steps):
        future_dt = start_datetime + dt.timedelta(hours=freq_hours * (step + 1))
        curr[idx["hour_sin"]] = np.sin(2 * np.pi * future_dt.hour / 24)
        curr[idx["hour_cos"]] = np.cos(2 * np.pi * future_dt.hour / 24)
        curr[idx["month_sin"]] = np.sin(2 * np.pi * future_dt.month / 12)
        curr[idx["month_cos"]] = np.cos(2 * np.pi * future_dt.month / 12)
        if "day_of_year_sin" in idx:
            doy = future_dt.timetuple().tm_yday
            curr[idx["day_of_year_sin"]] = np.sin(2 * np.pi * doy / 365.25)
            curr[idx["day_of_year_cos"]] = np.cos(2 * np.pi * doy / 365.25)

        next_pred = model.predict(curr.reshape(1, -1))[0]
        preds.append(next_pred)

        temp_hist.append(next_pred[0]); hum_hist.append(next_pred[1])
        curr[idx["temp_lag1"]] = temp_hist[-1]
        curr[idx["hum_lag1"]] = hum_hist[-1]
        if "temp_lag2" in idx:
            curr[idx["temp_lag2"]] = temp_hist[-2]
            curr[idx["hum_lag2"]] = hum_hist[-2]
        # atmospheric covariates, latitude/longitude, region/city encodings:
        # intentionally left unchanged (persisted) — no future values exist.

    return np.array(preds)


def recursive_forecast_batch(model, X_init_batch, feature_names, n_steps, start_datetimes, freq_hours=6):
    """
    Vectorized counterpart to recursive_forecast(): runs many independent
    forecast windows through the SAME n_steps recursion simultaneously,
    calling model.predict() once per step on the whole batch instead of
    once per (window, step) pair. For W windows and S steps this is S
    batched calls instead of W*S individual calls — the individual-call
    version was the actual bottleneck that made multi-step evaluation on a
    ~8,700-row test set impractically slow (tens of minutes for what should
    take seconds). Numerically identical to calling recursive_forecast() in
    a loop; only the call pattern differs.
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    curr = X_init_batch.copy().astype(float)  # (W, n_features)
    W = curr.shape[0]
    temp_hist = np.column_stack([curr[:, idx["temp_lag1"]], curr[:, idx.get("temp_lag2", idx["temp_lag1"])]])
    hum_hist = np.column_stack([curr[:, idx["hum_lag1"]], curr[:, idx.get("hum_lag2", idx["hum_lag1"])]])
    start_dt_arr = pd.to_datetime(start_datetimes).values

    all_preds = np.zeros((W, n_steps, 2))
    for step in range(n_steps):
        future_dt = pd.to_datetime(start_dt_arr) + pd.to_timedelta(freq_hours * (step + 1), unit="h")
        hours = future_dt.hour.values
        months = future_dt.month.values
        curr[:, idx["hour_sin"]] = np.sin(2 * np.pi * hours / 24)
        curr[:, idx["hour_cos"]] = np.cos(2 * np.pi * hours / 24)
        curr[:, idx["month_sin"]] = np.sin(2 * np.pi * months / 12)
        curr[:, idx["month_cos"]] = np.cos(2 * np.pi * months / 12)
        if "day_of_year_sin" in idx:
            doy = future_dt.dayofyear.values
            curr[:, idx["day_of_year_sin"]] = np.sin(2 * np.pi * doy / 365.25)
            curr[:, idx["day_of_year_cos"]] = np.cos(2 * np.pi * doy / 365.25)

        next_pred = model.predict(curr)  # (W, 2) — one batched call for the whole step
        all_preds[:, step, :] = next_pred

        temp_hist = np.column_stack([temp_hist[:, 1], next_pred[:, 0]])
        hum_hist = np.column_stack([hum_hist[:, 1], next_pred[:, 1]])
        curr[:, idx["temp_lag1"]] = temp_hist[:, 1]
        curr[:, idx["hum_lag1"]] = hum_hist[:, 1]
        if "temp_lag2" in idx:
            curr[:, idx["temp_lag2"]] = temp_hist[:, 0]
            curr[:, idx["hum_lag2"]] = hum_hist[:, 0]

    return all_preds  # (W, n_steps, 2)


def evaluate_multi_step(model, X_test, y_test, test_datetimes, feature_names,
                         horizons_hours=(6, 24, 48, 96), freq_hours=6,
                         direct_stack_metrics=None, tolerance=0.15):
    """[FIX #10] Reports RMSE/R2 per horizon in addition to MAE, and cross-checks
    the 1-step-equivalent recursive horizon against the direct model's metrics."""
    results = {"mae": {"temp": {}, "hum": {}}, "rmse": {"temp": {}, "hum": {}}, "r2": {"temp": {}, "hum": {}}}
    for h_hours in horizons_hours:
        n_steps = max(1, h_hours // freq_hours)
        starts = list(range(0, len(X_test) - n_steps, max(1, n_steps)))  # non-overlapping windows
        if not starts:
            continue
        init_batch = X_test.iloc[starts].values.astype(float)
        start_dts = test_datetimes.iloc[starts]
        preds_batch = recursive_forecast_batch(model, init_batch, feature_names, n_steps, start_dts, freq_hours)

        temp_true, temp_pred, hum_true, hum_pred = [], [], [], []
        for w, i in enumerate(starts):
            true_t = y_test.iloc[i + 1:i + n_steps + 1]["temp"].values
            true_h = y_test.iloc[i + 1:i + n_steps + 1]["hum"].values
            if len(true_t) != n_steps:
                continue
            temp_true.append(true_t[-1]); temp_pred.append(preds_batch[w, -1, 0])
            hum_true.append(true_h[-1]); hum_pred.append(preds_batch[w, -1, 1])
        if temp_true:
            tt, tp, ht, hp = map(np.array, (temp_true, temp_pred, hum_true, hum_pred))
            results["mae"]["temp"][h_hours] = float(mean_absolute_error(tt, tp))
            results["mae"]["hum"][h_hours] = float(mean_absolute_error(ht, hp))
            results["rmse"]["temp"][h_hours] = float(np.sqrt(mean_squared_error(tt, tp)))
            results["rmse"]["hum"][h_hours] = float(np.sqrt(mean_squared_error(ht, hp)))
            results["r2"]["temp"][h_hours] = float(r2_score(tt, tp))
            results["r2"]["hum"][h_hours] = float(r2_score(ht, hp))
            print(f"  Horizon {h_hours}h ({n_steps} steps) | Temp MAE={results['mae']['temp'][h_hours]:.4f} "
                  f"RMSE={results['rmse']['temp'][h_hours]:.4f} R2={results['r2']['temp'][h_hours]:.4f} | "
                  f"Hum MAE={results['mae']['hum'][h_hours]:.4f} RMSE={results['rmse']['hum'][h_hours]:.4f} "
                  f"R2={results['r2']['hum'][h_hours]:.4f}")

    smallest_h = min(horizons_hours)
    if direct_stack_metrics is not None and smallest_h in results["mae"]["temp"]:
        for var in ("temp", "hum"):
            direct_mae, recursive_mae = direct_stack_metrics[f"{var}_mae"], results["mae"][var][smallest_h]
            if direct_mae > 0 and abs(recursive_mae - direct_mae) / direct_mae > tolerance:
                print(f"  [WARNING] {var}: direct MAE={direct_mae:.4f} vs {smallest_h}h recursive MAE="
                      f"{recursive_mae:.4f} — disagree by >{tolerance:.0%}. Investigate before reporting both.")
            else:
                print(f"  [OK] {var}: direct MAE and {smallest_h}h recursive MAE agree within tolerance "
                      f"({direct_mae:.4f} vs {recursive_mae:.4f}).")
    return results


def plot_recursive_forecast_for_city(model, df_raw, city_location_name, feature_names,
                                      le_region, le_cond, le_city, train_ratio=0.75,
                                      n_steps=8, freq_hours=6):
    """[FIX #11] Guarded by _check_city_allowed — cannot run for a non-study location."""
    _check_city_allowed(city_location_name)
    display_name = STUDY_CITIES[city_location_name]

    group = df_raw[df_raw["city"] == city_location_name].sort_values("datetime").reset_index(drop=True)
    if group.empty:
        raise ValueError(f"'{city_location_name}' not found in the raw dataset.")
    group = engineer_features(group)
    group["region_enc"] = le_region.transform(group["region"].astype(str))
    group["condition_enc"] = le_cond.transform(group["condition_text"].astype(str))
    group["city_enc"] = le_city.transform(group["city"].astype(str))
    group = group.dropna(subset=feature_names).reset_index(drop=True)

    n_train = int(len(group) * train_ratio)
    if n_train >= len(group) - n_steps:
        raise ValueError(f"Not enough test rows for '{city_location_name}' to forecast {n_steps} steps.")

    init_row = group.iloc[n_train][feature_names].values.astype(float)
    start_dt = group.iloc[n_train]["datetime"]
    preds = recursive_forecast(model, init_row, feature_names, n_steps, start_dt, freq_hours)

    true_temp = group["temp"].iloc[n_train + 1: n_train + 1 + n_steps].values
    true_hum = group["hum"].iloc[n_train + 1: n_train + 1 + n_steps].values

    # [DATA INTEGRITY GUARD] Same class of check that would have caught the
    # Adilabad figure's humidity>100 / implausible temperature before saving.
    if (preds[:, 1] < 0).any() or (preds[:, 1] > 100).any():
        raise ValueError(f"Refusing to save figure for '{display_name}': forecasted humidity "
                          f"outside [0, 100]% (min={preds[:,1].min():.1f}, max={preds[:,1].max():.1f}).")
    if (preds[:, 0] < TEMP_RANGE_C[0]).any() or (preds[:, 0] > TEMP_RANGE_C[1]).any():
        raise ValueError(f"Refusing to save figure for '{display_name}': forecasted temperature "
                          f"outside {TEMP_RANGE_C} deg C.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{n_steps * freq_hours}-Hour Recursive Forecast — {display_name}")
    axes[0].plot(true_temp, label="Observed", marker="o", ms=3)
    axes[0].plot(preds[:, 0], label="Forecast", marker="x", ms=3)
    axes[0].set_ylabel("Temperature (deg C)"); axes[0].legend()
    axes[1].plot(true_hum, label="Observed", marker="o", ms=3)
    axes[1].plot(preds[:, 1], label="Forecast", marker="x", ms=3)
    axes[1].set_ylabel("Humidity (%)"); axes[1].set_xlabel(f"Steps ahead ({freq_hours}h each)"); axes[1].legend()
    plt.tight_layout()
    safe_name = display_name.replace(" ", "_")
    plt.savefig(out_path(f"multistep_forecast_{safe_name}.png"), dpi=150)
    plt.close()
    print(f"  Saved multistep_forecast_{safe_name}.png")


# ============================================================================
# 13. MULTI-MODEL COMPARISON VISUALS
# ============================================================================

def plot_comparative_metrics(metrics_dict):
    df_m = pd.DataFrame(metrics_dict).T
    models = df_m.index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Comparative Framework Performance (Original Units)", fontweight="bold")

    def _bar_with_labels(ax, col, color):
        bars = ax.bar(models, df_m[col], color=color, edgecolor="black")
        # Label each bar with its exact value, positioned just above (or, for
        # negative bars such as a negative R2, just below) the bar tip so it
        # never overlaps the bar itself.
        span = df_m[col].max() - df_m[col].min()
        pad = span * 0.02 if span > 0 else 0.02
        for bar, val in zip(bars, df_m[col]):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = pad if y >= 0 else -pad
            ax.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{val:.3f}",
                    ha="center", va=va, fontsize=9)
        return bars

    for i, (col, title) in enumerate([("temp_r2", "Temp R2"), ("temp_mae", "Temp MAE (C)"), ("temp_rmse", "Temp RMSE (C)")]):
        _bar_with_labels(axes[0, i], col, "#4C72B0")
        axes[0, i].set_title(title)
        axes[0, i].tick_params(axis="x", rotation=20)
    for i, (col, title) in enumerate([("hum_r2", "Hum R2"), ("hum_mae", "Hum MAE (%)"), ("hum_rmse", "Hum RMSE (%)")]):
        _bar_with_labels(axes[1, i], col, "#55A868")
        axes[1, i].set_title(title)
        axes[1, i].tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.savefig(out_path("model_metrics_comparison.png"), dpi=200); plt.close()


def plot_multi_model_scatters(y_test, preds_dict):
    models = list(preds_dict.keys())
    fig, axes = plt.subplots(2, len(models), figsize=(4.5 * len(models), 8))
    if len(models) == 1:
        axes = axes.reshape(2, 1)
    for i, name in enumerate(models):
        pred = preds_dict[name]; n = len(pred)
        y_temp, y_hum = y_test["temp"].iloc[-n:].values, y_test["hum"].iloc[-n:].values
        axes[0, i].scatter(y_temp, pred[:, 0], alpha=0.3, s=8, color="#4C72B0")
        lims = [min(y_temp.min(), pred[:, 0].min()), max(y_temp.max(), pred[:, 0].max())]
        axes[0, i].plot(lims, lims, "r--"); axes[0, i].set_title(f"{name}\nTemp R2={r2_score(y_temp, pred[:,0]):.4f}")
        axes[1, i].scatter(y_hum, pred[:, 1], alpha=0.3, s=8, color="#55A868")
        lims_h = [min(y_hum.min(), pred[:, 1].min()), max(y_hum.max(), pred[:, 1].max())]
        axes[1, i].plot(lims_h, lims_h, "r--"); axes[1, i].set_title(f"Hum R2={r2_score(y_hum, pred[:,1]):.4f}")
    plt.tight_layout(); plt.savefig(out_path("model_scatter_comparisons.png"), dpi=200); plt.close()


# ============================================================================
# 14. MAIN PIPELINE
# ============================================================================

def main(fast_smoke_test=False):
    """
    fast_smoke_test=True shrinks ARIMA's city coverage and LSTM epochs so the
    whole script can be verified end-to-end quickly. Set False for the real
    manuscript run (full ARIMA city coverage, full LOCO, full LSTM epochs).
    """
    print("=" * 70); print("STEP 1: Provenance + load + validate"); print("=" * 70)
    get_dataset_provenance(FILEPATH)
    df_raw = load_and_validate(FILEPATH)

    print("\n" + "=" * 70); print("STEP 2: Feature engineering + per-city chronological split"); print("=" * 70)
    df_full, X_train, X_test, y_train, y_test, le_region, le_cond, le_city, feature_names = prepare_data(df_raw)

    print("\n" + "=" * 70); print("STEP 3: Persistence baseline  [FIX #3]"); print("=" * 70)
    persistence_metrics = persistence_baseline(y_test)
    print(f"Temp: R2={persistence_metrics['temp_r2']:.4f} RMSE={persistence_metrics['temp_rmse']:.4f} MAE={persistence_metrics['temp_mae']:.4f}")
    print(f"Hum:  R2={persistence_metrics['hum_r2']:.4f} RMSE={persistence_metrics['hum_rmse']:.4f} MAE={persistence_metrics['hum_mae']:.4f}")

    print("\n" + "=" * 70); print("STEP 4: Stacked ensemble (TimeSeriesStackingRegressor)  [FIX #4]"); print("=" * 70)
    n_est = 60 if fast_smoke_test else 200
    model = get_stacked_model(n_estimators=n_est)
    model.fit(X_train, y_train)
    y_pred_stack = model.predict(X_test)
    for tname, est in zip(["temp", "hum"], model.estimators_):
        print(f"  Meta-learner ({tname}): {est.n_meta_train_} out-of-fold rows of {len(X_train)} training rows")
    stack_metrics = compute_metrics(y_test, y_pred_stack)
    print(f"Temp: R2={stack_metrics['temp_r2']:.4f} RMSE={stack_metrics['temp_rmse']:.4f} MAE={stack_metrics['temp_mae']:.4f}")
    print(f"Hum:  R2={stack_metrics['hum_r2']:.4f} RMSE={stack_metrics['hum_rmse']:.4f} MAE={stack_metrics['hum_mae']:.4f}")

    improvement = print_improvement_over_persistence(persistence_metrics, stack_metrics)

    print("\n" + "=" * 70); print("STEP 5: Walk-forward validation  [FIX #1]"); print("=" * 70)
    cv_results = walk_forward_validation(lambda: get_stacked_model(n_estimators=n_est), X_train, y_train, n_splits=5)
    if not cv_results.empty:
        cv_mean, cv_std = cv_results.mean(numeric_only=True), cv_results.std(numeric_only=True)
        print(f"Temp R2: {cv_mean['temp_r2']:.4f} +/- {cv_std['temp_r2']:.4f} | RMSE: {cv_mean['temp_rmse']:.4f} +/- {cv_std['temp_rmse']:.4f}")
        print(f"Hum  R2: {cv_mean['hum_r2']:.4f} +/- {cv_std['hum_r2']:.4f} | RMSE: {cv_mean['hum_rmse']:.4f} +/- {cv_std['hum_rmse']:.4f}")
        cv_results.to_csv(out_path("walk_forward_results.csv"), index=False)

    print("\n" + "=" * 70); print("STEP 6: ARIMA (all cities, AIC order selection)  [FIX #6]"); print("=" * 70)
    city_subset = list(STUDY_CITIES) if fast_smoke_test else None
    arima_temp_overall, arima_temp_cities = arima_rolling_all_cities(df_raw, target="temp", city_subset=city_subset)
    arima_hum_overall, arima_hum_cities = arima_rolling_all_cities(df_raw, target="hum", city_subset=city_subset)
    print(f"ARIMA Temp: R2={arima_temp_overall['r2']:.4f} RMSE={arima_temp_overall['rmse']:.4f} MAE={arima_temp_overall['mae']:.4f}")
    print(f"ARIMA Hum:  R2={arima_hum_overall['r2']:.4f} RMSE={arima_hum_overall['rmse']:.4f} MAE={arima_hum_overall['mae']:.4f}")
    if not arima_temp_cities.empty:
        arima_temp_cities.to_csv(out_path("arima_temp_per_city.csv"), index=False)
    if not arima_hum_cities.empty:
        arima_hum_cities.to_csv(out_path("arima_hum_per_city.csv"), index=False)

    print("\n" + "=" * 70); print("STEP 7: LSTM baseline  [FIX #6]"); print("=" * 70)
    lstm_epochs = 15 if fast_smoke_test else 100
    lstm_pred_temp, lstm_pred_hum, lstm_metrics = run_lstm_full(X_train, X_test, y_train, y_test, epochs=lstm_epochs)
    if len(lstm_pred_temp):
        print(f"Temp: R2={lstm_metrics['temp_r2']:.4f} RMSE={lstm_metrics['temp_rmse']:.4f} MAE={lstm_metrics['temp_mae']:.4f}")
        print(f"Hum:  R2={lstm_metrics['hum_r2']:.4f} RMSE={lstm_metrics['hum_rmse']:.4f} MAE={lstm_metrics['hum_mae']:.4f}")

    print("\n" + "=" * 70); print("STEP 8: Multi-model comparison visuals"); print("=" * 70)
    metrics_dict = {"Persistence": persistence_metrics,
                     "ARIMA": {"temp_r2": arima_temp_overall["r2"], "temp_rmse": arima_temp_overall["rmse"], "temp_mae": arima_temp_overall["mae"],
                               "hum_r2": arima_hum_overall["r2"], "hum_rmse": arima_hum_overall["rmse"], "hum_mae": arima_hum_overall["mae"]}}
    preds_dict = {}
    pers_pred, _ = compute_persistence_predictions(y_test)
    if pers_pred is not None:
        preds_dict["Persistence"] = pers_pred
    if len(lstm_pred_temp):
        metrics_dict["LSTM"] = lstm_metrics
        preds_dict["LSTM"] = np.column_stack([lstm_pred_temp, lstm_pred_hum])
    metrics_dict["Stacked Ensemble"] = stack_metrics
    preds_dict["Stacked Ensemble"] = y_pred_stack
    plot_comparative_metrics(metrics_dict)
    plot_multi_model_scatters(y_test, preds_dict)
    print(f"Saved comparison plots. Models compared: {list(metrics_dict.keys())}")

    print("\n" + "=" * 70); print("STEP 9: True Leave-One-City-Out, all locations  [FIX #2]"); print("=" * 70)
    df_loco_ready = df_full  # already has FEATURE_COLS populated
    loco_city_subset = (list(STUDY_CITIES) + sorted(df_full["city"].unique())[:20]) if fast_smoke_test else None
    loco_temp = leave_one_city_out(df_loco_ready, "temp", city_subset=loco_city_subset)
    loco_hum = leave_one_city_out(df_loco_ready, "hum", city_subset=loco_city_subset)
    loco_temp.to_csv(out_path("loco_temp_results.csv"), index=False)
    loco_hum.to_csv(out_path("loco_hum_results.csv"), index=False)

    print("\n" + "=" * 70); print("STEP 10: SHAP analysis"); print("=" * 70)
    try:
        X_sample = X_test.sample(min(150, len(X_test)), random_state=42)
        shap_analysis(model, X_sample, feature_names)
    except Exception as e:
        print(f"SHAP skipped: {e}")

    print("\n" + "=" * 70); print("STEP 11: Residual diagnostics"); print("=" * 70)
    plot_residuals(y_test["temp"].values, y_pred_stack[:, 0], "temperature")
    plot_residuals(y_test["hum"].values, y_pred_stack[:, 1], "humidity")

    print("\n" + "=" * 70); print("STEP 12: Multi-step recursive forecasting  [FIX #10]"); print("=" * 70)
    test_datetimes = df_full.loc[X_test.index, "datetime"]
    horizons = (6, 24, 48) if fast_smoke_test else (6, 24, 48, 96)
    multi_results = evaluate_multi_step(model, X_test, y_test, test_datetimes, feature_names,
                                         horizons_hours=horizons, direct_stack_metrics=stack_metrics)
    pd.DataFrame(multi_results["mae"]).to_csv(out_path("multi_step_mae.csv"))
    pd.DataFrame(multi_results["rmse"]).to_csv(out_path("multi_step_rmse.csv"))
    pd.DataFrame(multi_results["r2"]).to_csv(out_path("multi_step_r2.csv"))

    print("\n" + "=" * 70); print("STEP 13: Per-city 48h forecast figures, all 8 study cities  [FIX #11]"); print("=" * 70)
    for city_loc_name in STUDY_CITIES:
        try:
            plot_recursive_forecast_for_city(model, df_raw, city_loc_name, feature_names,
                                              le_region, le_cond, le_city, n_steps=8)
        except Exception as e:
            print(f"  [SKIPPED] {STUDY_CITIES[city_loc_name]}: {e}")

    print("\n" + "=" * 70); print("SUMMARY"); print("=" * 70)
    print(json.dumps({"persistence": persistence_metrics, "stacked_ensemble": stack_metrics,
                       "improvement_over_persistence_pct": improvement}, indent=2, default=str))

    with open(out_path("full_results_report.json"), "w") as f:
        json.dump({"persistence": persistence_metrics, "stacked_ensemble": stack_metrics,
                    "improvement_over_persistence_pct": improvement,
                    "arima": {"temp": arima_temp_overall, "hum": arima_hum_overall},
                    "lstm": lstm_metrics if len(lstm_pred_temp) else None}, f, indent=2, default=str)

    joblib.dump(model, out_path("stacked_model_final.pkl"))
    print(f"\nAll artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(fast_smoke_test=False)
