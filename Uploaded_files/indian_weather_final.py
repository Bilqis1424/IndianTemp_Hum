"""
================================================================================
FINAL SCRIPT (REVISION 4) — EXTENDED FEATURES + FAIR ARIMA EVALUATION
--------------------------------------------------------------------------------
Builds on Revision 3. This revision implements:

  [REV 4 - A]  Extended temporal features for the ~6-hour cadence:
                 - lags 1..8 (lag4 == "same hour yesterday")
                 - rolling means/stds over 4 and 8 steps (24h, 48h)
                 - 24-hour change (lag1 - lag4)
               and lagged atmospheric covariates (pressure_lag1 etc.) so the
               model has ahead-of-time information about exogenous drivers.

  [REV 4 - B]  Per-city additive bias correction (fit on a chronological
               validation slice of each city's training rows, shrunk toward
               zero). Applied AFTER the stacked ensemble's raw predictions.

  [REV 4 - C]  Three clearly-distinguished R2 definitions reported
               side by side, never interchangeably:
                 - pooled R2   : on the full held-out set (main table)
                 - per-location: within-city R2 with bootstrap CIs
                 - R2_anom     : within-city skill vs. per-city climatology
               The R2_anom metric is explicitly labelled and defined so the
               graphical-abstract/Table mismatch that the reviewer flagged
               cannot recur.

  [REV 4 - D]  ARIMA is now evaluated on EXACTLY the same held-out rows as
               every other model, using the intersection of each city's raw
               series with y_test.index. Row-level ARIMA predictions are
               saved with city and datetime (arima_temp_predictions.csv,
               arima_hum_predictions.csv). This addresses the reviewer's
               question of whether ARIMA's poor R2 was an implementation /
               evaluation artefact or a genuine result.

  [REV 4 - E]  Persistence baseline is now computed per-city (not via a
               global shift that crossed city boundaries in the concatenated
               test set).

All previous [FIX #n] tags are preserved.

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


# [FIX #11] location_name -> display name for figures/tables.
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
    """[FIX #11] Refuse to proceed for a location outside the defined study set."""
    if study_cities and city_location_name not in study_cities:
        raise ValueError(
            f"Location '{city_location_name}' is not in STUDY_CITIES "
            f"{list(study_cities)}. Refusing to generate a figure/result for a "
            f"location outside the defined study scope."
        )


TARGETS_DISPLAY = {"temp": "Temperature", "hum": "Humidity"}

TEMP_RANGE_C = (-15.0, 55.0)
HUM_RANGE_PCT = (0.0, 100.0)


# ----------------------------------------------------------------------------
# [REV 4 - A] Feature configuration
# ----------------------------------------------------------------------------
ATMOS_FEATURES = ["pressure_mb", "wind_kph", "cloud", "uv_index", "precip_mm",
                   "visibility_km", "air_quality_PM2.5"]

# Lagged atmospheric covariates — valid at forecasting time (yesterday's
# pressure is observed). These give the model ahead-of-time exogenous signal
# that contemporaneous-only features cannot provide.
ATMOS_LAGGED = ["pressure_lag1", "cloud_lag1", "wind_lag1", "visibility_lag1"]

# Consecutive lags 1..8 (lag4 = "same hour yesterday" at 6h cadence).
LAG_STEPS = tuple(range(1, 9))
ROLL_WINDOWS = (4, 8)  # 24h, 48h

FEATURE_COLS = (
    [f"temp_lag{k}" for k in LAG_STEPS]
    + [f"hum_lag{k}" for k in LAG_STEPS]
    + ["temp_chg_24h", "hum_chg_24h"]
    + [f"temp_roll{w}_mean" for w in ROLL_WINDOWS]
    + [f"hum_roll{w}_mean" for w in ROLL_WINDOWS]
    + [f"temp_roll{w}_std" for w in ROLL_WINDOWS]
    + [f"hum_roll{w}_std" for w in ROLL_WINDOWS]
    + ["hour_sin", "hour_cos", "month_sin", "month_cos",
       "day_of_year_sin", "day_of_year_cos",
       "region_enc", "condition_enc", "latitude", "longitude"]
    + ATMOS_FEATURES
    + ATMOS_LAGGED
)
# city_enc intentionally excluded from FEATURE_COLS — see [FIX #2] rationale
# in leave_one_city_out(). The main study pipeline adds it back in
# prepare_data() because all 8 study cities are, by definition, seen in
# training there.


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
    """[FIX #9] Real hash + real date."""
    file_hash = compute_file_hash(filepath)
    if download_date is not None:
        date_str, date_source = download_date, "user-provided download date"
    else:
        mtime = os.path.getmtime(filepath)
        date_str = dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        date_source = "file last-modified timestamp (proxy)"
    print(f"Dataset hash: {file_hash}")
    print(f"Dataset date: {date_str}  [{date_source}]")
    return file_hash, date_str


def load_and_validate(filepath):
    df = (pd.read_excel(filepath) if filepath.endswith((".xlsx", ".xls"))
          else pd.read_csv(filepath, low_memory=False))
    print(f"Raw columns found ({len(df.columns)}): {df.columns.tolist()}")
    print(f"Raw shape: {df.shape}")

    required = ["location_name", "region", "last_updated", "temperature_celsius",
                "humidity", "condition_text", "latitude", "longitude",
                "pressure_mb", "wind_kph", "cloud", "uv_index", "precip_mm",
                "visibility_km", "air_quality_PM2.5"]
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

    df = df.rename(columns={"location_name": "city",
                             "temperature_celsius": "temp",
                             "humidity": "hum"})
    df["datetime"] = pd.to_datetime(df["last_updated"])
    return df.sort_values(["city", "datetime"]).reset_index(drop=True)


# ============================================================================
# 2. FEATURE ENGINEERING  [REV 4 - A]
# ============================================================================

def engineer_features(df):
    """
    Extended feature engineering for the ~6-hour cadence:
      - lags 1..8 (lag4 = same hour yesterday)
      - rolling means/stds over 4 and 8 steps (24h / 48h)
      - 24h change
      - lagged atmospheric covariates
    All operations are per-city; rolling summaries are computed on the
    shifted series so no current-row information leaks into a feature.
    """
    df = df.copy()
    g = df.groupby("city")

    for k in LAG_STEPS:
        df[f"temp_lag{k}"] = g["temp"].shift(k)
        df[f"hum_lag{k}"] = g["hum"].shift(k)

    df["temp_chg_24h"] = df["temp_lag1"] - df["temp_lag4"]
    df["hum_chg_24h"] = df["hum_lag1"] - df["hum_lag4"]

    for w in ROLL_WINDOWS:
        df[f"temp_roll{w}_mean"] = g["temp"].transform(lambda s: s.shift(1).rolling(w).mean())
        df[f"hum_roll{w}_mean"] = g["hum"].transform(lambda s: s.shift(1).rolling(w).mean())
        df[f"temp_roll{w}_std"] = g["temp"].transform(lambda s: s.shift(1).rolling(w).std())
        df[f"hum_roll{w}_std"] = g["hum"].transform(lambda s: s.shift(1).rolling(w).std())

    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
    doy = df["datetime"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["pressure_lag1"] = g["pressure_mb"].shift(1)
    df["cloud_lag1"] = g["cloud"].shift(1)
    df["wind_lag1"] = g["wind_kph"].shift(1)
    df["visibility_lag1"] = g["visibility_km"].shift(1)

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


def prepare_data(df, train_ratio=0.75, min_rows_per_city=12):
    """
    Per-city chronological split. Cities with fewer than min_rows_per_city
    usable rows after lag/rolling warmup are skipped (they cannot support a
    meaningful train/test split).
    """
    df = engineer_features(df)
    df, le_region, le_cond, le_city = encode_categoricals(df, fit=True)
    df = df.dropna(subset=FEATURE_COLS + ["city_enc"]).reset_index(drop=True)

    feature_cols_with_city = FEATURE_COLS + ["city_enc"]
    X = df[feature_cols_with_city].astype(np.float32)
    y = df[["temp", "hum"]].astype(np.float32)

    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    skipped = []
    for city in df["city"].unique():
        idx = np.where((df["city"] == city).values)[0]
        n = len(idx)
        if n < min_rows_per_city:
            skipped.append((city, n))
            continue
        split_idx = int(n * train_ratio)
        if split_idx < 1 or n - split_idx < 1:
            skipped.append((city, n))
            continue
        X_train_list.append(X.iloc[idx[:split_idx]]); X_test_list.append(X.iloc[idx[split_idx:]])
        y_train_list.append(y.iloc[idx[:split_idx]]); y_test_list.append(y.iloc[idx[split_idx:]])

    if skipped:
        print(f"  Skipped {len(skipped)} cities with < {min_rows_per_city} usable rows.")

    X_train = pd.concat(X_train_list); X_test = pd.concat(X_test_list)
    y_train = pd.concat(y_train_list); y_test = pd.concat(y_test_list)
    print(f"Data prepared: X_train {X_train.shape}, X_test {X_test.shape}, "
          f"{X_train.shape[1]} features")
    return df, X_train, X_test, y_train, y_test, le_region, le_cond, le_city, feature_cols_with_city


# ============================================================================
# 3. PERSISTENCE BASELINE  [FIX #3]  [REV 4 - E: per-city]
# ============================================================================

def persistence_predictions(y_test, city_array):
    """
    Per-city persistence: yhat_t = y_{t-1} WITHIN the same city. The previous
    version used a single global shift, which for the concatenated multi-city
    test set silently predicted the first row of each city from the LAST row
    of a DIFFERENT city. This fixes that.
    """
    df = pd.DataFrame({
        "city": np.asarray(city_array),
        "temp": y_test["temp"].values,
        "hum": y_test["hum"].values,
    })
    df["temp_pred"] = df.groupby("city")["temp"].shift(1)
    df["hum_pred"] = df.groupby("city")["hum"].shift(1)
    return df


def persistence_baseline(y_test, city_array):
    df = persistence_predictions(y_test, city_array)
    valid = df.dropna(subset=["temp_pred", "hum_pred"])
    if len(valid) < 2:
        return {k: np.nan for k in ["temp_r2", "temp_rmse", "temp_mae",
                                     "hum_r2", "hum_rmse", "hum_mae"]}, df
    return {
        "temp_r2": float(r2_score(valid["temp"], valid["temp_pred"])),
        "temp_rmse": float(np.sqrt(mean_squared_error(valid["temp"], valid["temp_pred"]))),
        "temp_mae": float(mean_absolute_error(valid["temp"], valid["temp_pred"])),
        "hum_r2": float(r2_score(valid["hum"], valid["hum_pred"])),
        "hum_rmse": float(np.sqrt(mean_squared_error(valid["hum"], valid["hum_pred"]))),
        "hum_mae": float(mean_absolute_error(valid["hum"], valid["hum_pred"])),
    }, df


def print_improvement_over_persistence(persistence_metrics, model_metrics,
                                        model_name="Stacked Ensemble"):
    """[FIX #3] Percentage improvement over persistence."""
    imp = {}
    for var in ("temp", "hum"):
        p_mae, m_mae = persistence_metrics[f"{var}_mae"], model_metrics[f"{var}_mae"]
        imp[f"{var}_mae_improvement_pct"] = float((p_mae - m_mae) / p_mae * 100) if p_mae else np.nan
        p_rmse, m_rmse = persistence_metrics[f"{var}_rmse"], model_metrics[f"{var}_rmse"]
        imp[f"{var}_rmse_improvement_pct"] = float((p_rmse - m_rmse) / p_rmse * 100) if p_rmse else np.nan
    print(f"\n{model_name} improvement over persistence baseline:")
    print(f"  Temperature: MAE {imp['temp_mae_improvement_pct']:+.1f}%, "
          f"RMSE {imp['temp_rmse_improvement_pct']:+.1f}%")
    print(f"  Humidity:    MAE {imp['hum_mae_improvement_pct']:+.1f}%, "
          f"RMSE {imp['hum_rmse_improvement_pct']:+.1f}%")
    return imp


# ============================================================================
# 4. STACKED ENSEMBLE (time-series-safe)  [FIX #4]
# ============================================================================

class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """[FIX #4] sklearn's default-KFold stacking replaced with a manual
    TimeSeriesSplit loop so every out-of-fold prediction is generated using
    strictly earlier data."""

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
            raise ValueError("No out-of-fold predictions generated — n_splits too large.")
        self.final_estimator_ = clone(self.final_estimator).fit(oof[covered], y[covered])
        self.fitted_base_estimators_ = [(name, clone(est).fit(X, y))
                                         for name, est in self.base_estimators]
        return self

    def predict(self, X):
        X = np.asarray(X)
        base_preds = np.column_stack([m.predict(X) for _, m in self.fitted_base_estimators_])
        return self.final_estimator_.predict(base_preds)


def get_stacked_model(n_splits=5, n_estimators=200):
    base_models = [
        ("xgb", XGBRegressor(n_estimators=n_estimators, max_depth=5,
                              learning_rate=0.05, random_state=42,
                              verbosity=0, n_jobs=-1)),
        ("lgbm", LGBMRegressor(n_estimators=n_estimators, num_leaves=31,
                                learning_rate=0.05, random_state=42,
                                verbose=-1, n_jobs=-1)),
        ("cat", CatBoostRegressor(iterations=n_estimators, depth=6,
                                   learning_rate=0.05, random_seed=42, verbose=0)),
    ]
    stack = TimeSeriesStackingRegressor(base_models, Ridge(alpha=1.0), n_splits=n_splits)
    return MultiOutputRegressor(stack, n_jobs=1)


# ============================================================================
# 4b. PER-CITY BIAS CORRECTION  [REV 4 - B]
# ============================================================================

def fit_per_city_bias(model, X_train, y_train, df_full, val_frac=0.2, alpha=0.5):
    """
    Additive per-city correction fit on the LAST val_frac of each city's
    training rows (chronological validation slice), shrunk by `alpha` toward
    zero. Cities with few validation rows do not over-correct.

    Rationale: a global model spends capacity separating cities; the residual
    that remains is dominated by systematic per-city offsets. Correcting them
    on a held-out slice is cheap and lifts within-city (per-location) skill.
    """
    df_train = df_full.loc[X_train.index]
    val_rows = []
    for city in df_train["city"].unique():
        city_idx = np.where((df_train["city"] == city).values)[0]
        n_val = max(1, int(len(city_idx) * val_frac))
        val_rows.extend(city_idx[-n_val:])
    val_rows = np.array(val_rows)
    if len(val_rows) == 0:
        return pd.DataFrame(columns=["temp_bias", "hum_bias"])

    X_val, y_val = X_train.iloc[val_rows], y_train.iloc[val_rows]
    df_val = df_train.iloc[val_rows]

    pred = model.predict(X_val)
    resid = pd.DataFrame({
        "city": df_val["city"].values,
        "temp_bias": y_val["temp"].values - pred[:, 0],
        "hum_bias": y_val["hum"].values - pred[:, 1],
    })
    bias = resid.groupby("city").mean() * alpha
    print(f"  Per-city bias correction fitted on {len(bias)} cities "
          f"(alpha={alpha}, {len(val_rows)} validation rows).")
    return bias


def apply_per_city_bias(y_pred, city_array, bias_table):
    """Add the fitted per-city bias back to raw predictions. Unknown cities get 0."""
    if bias_table is None or bias_table.empty:
        return y_pred
    tb = pd.Series(np.asarray(city_array)).map(bias_table["temp_bias"]).fillna(0).values
    hb = pd.Series(np.asarray(city_array)).map(bias_table["hum_bias"]).fillna(0).values
    out = y_pred.copy()
    out[:, 0] += tb
    out[:, 1] += hb
    return out


# ============================================================================
# 5. METRICS  [REV 4 - C: three R2 definitions]
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
        "temp_r2": r2_temp,
        "temp_rmse": float(np.sqrt(mean_squared_error(y_true["temp"], y_pred[:, 0]))),
        "temp_mae": float(mean_absolute_error(y_true["temp"], y_pred[:, 0])),
        "hum_r2": r2_hum,
        "hum_rmse": float(np.sqrt(mean_squared_error(y_true["hum"], y_pred[:, 1]))),
        "hum_mae": float(mean_absolute_error(y_true["hum"], y_pred[:, 1])),
        "temp_variance": var_temp, "hum_variance": var_hum,
    }


def per_city_anomaly_r2(y_true, y_pred, cities, min_var=1e-3):
    """
    [REV 4 - C] R2_anom: within-city skill vs a per-city climatological mean.
    Both truth and prediction are de-meaned per city before scoring, so this
    measures how much of the within-city temporal variability the model
    explains — the same question a per-location R2 asks, but on a denominator
    that is (approximately) the within-city variance rather than zero.

    IMPORTANT: this is NOT the same quantity as pooled R2. It MUST be
    labelled `R2_anom` in the manuscript. Substituting it silently for R2 is
    what produced the graphical-abstract mismatch the reviewer flagged.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    cities = np.asarray(cities)

    df_t = pd.DataFrame({"city": cities, "y": y_true})
    df_p = pd.DataFrame({"city": cities, "y": y_pred})
    yt = y_true - df_t.groupby("city")["y"].transform("mean").values
    yp = y_pred - df_p.groupby("city")["y"].transform("mean").values

    if len(yt) < 2:
        return np.nan
    if np.var(yt) < min_var:
        return np.nan
    return float(r2_score(yt, yp))


def bootstrap_per_city_ci(y_true, y_pred, cities, n_boot=500, ci=0.95, random_state=42):
    """[REV 4 - C] Block bootstrap by city: resample the SET of cities with
    replacement, keep all their rows, recompute R2."""
    rng = np.random.RandomState(random_state)
    cities = np.asarray(cities)
    unique_cities = np.unique(cities)
    if len(unique_cities) < 2:
        return (np.nan, np.nan)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    city_to_idx = {c: np.where(cities == c)[0] for c in unique_cities}

    r2s = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_cities, size=len(unique_cities), replace=True)
        idx = np.concatenate([city_to_idx[c] for c in sampled])
        r2, _ = safe_r2_score(y_true[idx], y_pred[idx])
        if not np.isnan(r2):
            r2s.append(r2)
    if len(r2s) < 10:
        return (np.nan, np.nan)
    lo, hi = np.percentile(r2s, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def per_location_table(y_true, y_pred, df_test, n_boot=200):
    """Per-location R2/RMSE/MAE with bootstrap CIs and study-set flag."""
    df = pd.DataFrame({
        "city": df_test["city"].values,
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    rows = []
    for city, g in df.groupby("city"):
        if len(g) < 2:
            continue
        r2, var = safe_r2_score(g["y_true"], g["y_pred"])
        lo, hi = bootstrap_per_city_ci(g["y_true"].values, g["y_pred"].values,
                                        g["city"].values, n_boot=n_boot)
        rows.append({
            "city": city,
            "in_study_set": city in STUDY_CITIES,
            "n": int(len(g)),
            "r2": r2, "r2_lo": lo, "r2_hi": hi,
            "rmse": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
            "mae": float(mean_absolute_error(g["y_true"], g["y_pred"])),
            "variance": var,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ============================================================================
# 6. WALK-FORWARD VALIDATION  [FIX #1]
# ============================================================================

def walk_forward_validation(model_builder, X_train, y_train, n_splits=5, min_train_samples=50):
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
        metrics.update(fold=len(results), train_window_size=len(train_idx),
                        val_window_size=len(val_idx))
        results.append(metrics)
        print(f"  Fold {metrics['fold']}: Temp R2={metrics['temp_r2']:.4f} "
              f"RMSE={metrics['temp_rmse']:.4f} | Hum R2={metrics['hum_r2']:.4f} "
              f"RMSE={metrics['hum_rmse']:.4f}")
    return pd.DataFrame(results)


# ============================================================================
# 7. ARIMA — SAME HELD-OUT ROWS AS OTHER MODELS  [FIX #6]  [REV 4 - D]
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


def arima_rolling_predictions(df_full, y_test, target="temp", min_test_samples=5,
                               select_order=True, city_subset=None):
    """
    [REV 4 - D] One-step-ahead rolling ARIMA, evaluated on EXACTLY the same
    held-out rows as every other model. The train/test boundary is defined by
    the intersection of each city's series with y_test.index — the same rows
    the stacked ensemble, persistence, and LSTM are scored on.

    Returns a row-level DataFrame: city, datetime, target, y_true, y_pred,
    order_p, order_d, order_q, aic.
    """
    cities = city_subset if city_subset is not None else sorted(df_full["city"].unique())
    rows = []
    test_index = set(y_test.index)

    for city in cities:
        group = df_full[df_full["city"] == city].sort_values("datetime").copy()
        if group.empty:
            continue
        test_mask = group.index.isin(test_index)
        test_group = group[test_mask].sort_values("datetime")
        train_group = group[~test_mask].sort_values("datetime")
        if len(test_group) < min_test_samples or len(train_group) < 10:
            continue

        train_series = train_group[target].dropna().values
        test_series = test_group[target].values
        test_datetimes = test_group["datetime"].values
        if len(train_series) < 10 or len(test_series) < min_test_samples:
            continue

        order, aic = select_arima_order(train_series) if select_order else ((1, 0, 0), np.nan)

        history, preds = list(train_series), []
        for t in range(len(test_series)):
            try:
                fitted = ARIMA(history, order=order).fit()
                preds.append(float(fitted.forecast()[0]))
            except Exception:
                preds.append(np.nan)
            history.append(test_series[t])

        for dt_i, yt, yp in zip(test_datetimes, test_series, preds):
            rows.append({
                "city": city, "datetime": pd.Timestamp(dt_i), "target": target,
                "y_true": float(yt) if pd.notna(yt) else np.nan,
                "y_pred": yp,
                "order_p": order[0], "order_d": order[1], "order_q": order[2],
                "aic": aic,
            })

    return pd.DataFrame(rows)


def arima_pooled_metrics(pred_df):
    valid = pred_df.dropna(subset=["y_true", "y_pred"])
    if len(valid) < 2:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "n": int(len(valid))}
    return {
        "r2": float(r2_score(valid["y_true"], valid["y_pred"])),
        "rmse": float(np.sqrt(mean_squared_error(valid["y_true"], valid["y_pred"]))),
        "mae": float(mean_absolute_error(valid["y_true"], valid["y_pred"])),
        "n": int(len(valid)),
    }


def arima_per_city_metrics(pred_df):
    out = []
    for city, g in pred_df.groupby("city"):
        g = g.dropna(subset=["y_true", "y_pred"])
        if len(g) < 2:
            continue
        r2, var = safe_r2_score(g["y_true"], g["y_pred"])
        out.append({
            "city": city, "n": int(len(g)), "r2": r2,
            "rmse": float(np.sqrt(mean_squared_error(g["y_true"], g["y_pred"]))),
            "mae": float(mean_absolute_error(g["y_true"], g["y_pred"])),
            "variance": var,
            "order_p": int(g["order_p"].iloc[0]),
            "order_d": int(g["order_d"].iloc[0]),
            "order_q": int(g["order_q"].iloc[0]),
            "aic": float(g["aic"].iloc[0]) if pd.notna(g["aic"].iloc[0]) else np.nan,
        })
    return pd.DataFrame(out)


# ============================================================================
# 8. LSTM BASELINE  [FIX #6]
# ============================================================================

def run_lstm_full(X_train, X_test, y_train, y_test, n_steps=8, epochs=100):
    """
    [FIX #6] Full architecture documented in the docstring for the methods
    section:
      - Input window: n_steps timesteps (default 8, ~2 days at ~6h cadence)
      - 3 stacked LSTM layers: 128 -> 64 -> 32 units, ReLU
      - BatchNormalization + Dropout (0.3, 0.3, 0.2)
      - Dense(16, relu) -> Dense(2) output
      - Adam(lr=0.001); EarlyStopping(patience=10); ReduceLROnPlateau(0.5, 5)
      - epochs=100 (capped by early stopping), batch_size=64
      - Validation split: last 20% of training sequences, chronological
    """
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train_s = scaler_X.fit_transform(X_train.values)
    X_test_s = scaler_X.transform(X_test.values)
    y_train_s = scaler_y.fit_transform(y_train.values)
    y_test_s = scaler_y.transform(y_test.values)
    train_comb = np.column_stack([X_train_s, y_train_s])
    test_comb = np.column_stack([X_test_s, y_test_s])

    X_seq, y_seq = [], []
    for i in range(len(train_comb) - n_steps):
        X_seq.append(train_comb[i:i + n_steps, :])
        y_seq.append(train_comb[i + n_steps, -2:])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    if len(X_seq) < 50:
        return np.array([]), np.array([]), {}

    val_size = max(1, int(0.2 * len(X_seq)))
    X_tr, X_val = X_seq[:-val_size], X_seq[-val_size:]
    y_tr, y_val = y_seq[:-val_size], y_seq[-val_size:]

    model = Sequential([
        LSTM(128, activation="relu", return_sequences=True,
             input_shape=(n_steps, X_seq.shape[2])),
        BatchNormalization(), Dropout(0.3),
        LSTM(64, activation="relu", return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(32, activation="relu"), Dropout(0.2),
        Dense(16, activation="relu"), Dense(2),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=64,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10,
                                        restore_best_weights=True),
                          ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=5, min_lr=1e-6)],
              verbose=0)

    test_preds = []
    n_test_seq = len(test_comb) - n_steps
    if n_test_seq > 0:
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
# 9. LEAVE-ONE-CITY-OUT — TRUE LOCO, ALL LOCATIONS  [FIX #2]
# ============================================================================

def build_fast_loco_model(n_estimators=150):
    """Single XGB for the LOCO sweep (543 x 2 targets = 1,086 fits)."""
    return XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                         random_state=42, n_jobs=-1, verbosity=0)


def leave_one_city_out(df, target, model_builder=build_fast_loco_model, city_subset=None):
    """
    [FIX #2] True LOCO. `city_enc` is DELIBERATELY EXCLUDED from the features
    used here: a held-out city has no valid learned encoding, and re-fitting
    the encoder to include it would let the model see the held-out city's
    identity — defeating the point of holding it out. The model relies on
    lags, calendar features, region/condition encodings, lat/long, and
    atmospheric covariates — none of which require the city's identity to be
    known ahead of time.
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
            "n_train_cities": df.loc[~test_mask, "city"].nunique(),
            "n_test_rows": int(test_mask.sum()),
            "r2": r2,
            "rmse": float(np.sqrt(mean_squared_error(y_all[test_mask], preds))),
            "mae": float(mean_absolute_error(y_all[test_mask], preds)),
        })
    results_df = pd.DataFrame(per_city)
    n_undefined = results_df["r2"].isna().sum()
    defined = results_df.dropna(subset=["r2"])
    if len(defined):
        print(f"  LOCO {target}: {len(results_df)} cities, {n_undefined} undefined R2. "
              f"Median R2={defined['r2'].median():.4f}, mean R2={defined['r2'].mean():.4f}. "
              f"Median RMSE={results_df['rmse'].median():.4f}, "
              f"median MAE={results_df['mae'].median():.4f}.")
    study_rows = results_df[results_df["in_study_set"]]
    if len(study_rows):
        print(f"  Study-city subset: mean R2={study_rows['r2'].mean():.4f}, "
              f"mean RMSE={study_rows['rmse'].mean():.4f}, "
              f"mean MAE={study_rows['mae'].mean():.4f}")
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
        "importance", ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(imp["feature"], imp["importance"]); plt.xlabel("Mean |SHAP value|")
    plt.title("Top 20 Features"); plt.gca().invert_yaxis(); plt.tight_layout()
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
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="r", linestyle="--")
    axes[0, 0].set_title("Residuals vs Predicted")
    stats.probplot(residuals, dist="norm", plot=axes[0, 1]); axes[0, 1].set_title("Q-Q Plot")
    axes[1, 0].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
    axes[1, 0].set_title("Distribution")
    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.4, s=8)
    axes[1, 1].axhline(0, color="r", linestyle="--")
    axes[1, 1].set_title("Residuals by Row Order")
    plt.tight_layout(); plt.savefig(out_path(f"residuals_{target_name}.png"), dpi=150); plt.close()


# ============================================================================
# 12. MULTI-STEP RECURSIVE FORECASTING  [FIX #10]  [REV 4 - A]
# ============================================================================

def recursive_forecast_batch(model, X_init_batch, feature_names, n_steps,
                              start_datetimes, freq_hours=6):
    """
    [FIX #10] [REV 4 - A] Vectorised recursive forecast with extended lags
    and rolling features.

    History buffers are extracted from the lag features themselves
    (temp_lag1..temp_lag8). This avoids needing external history arrays and
    keeps the buffer perfectly consistent with the training-time lag
    definition. At each recursion step, calendar features are advanced to the
    actual forecast timestamp, lags and rolling summaries are recomputed from
    the buffer, and the buffer is updated with the new prediction.
    Atmospheric covariates (contemporaneous AND lagged) are persisted at
    their initial values — the naive-persistence assumption for exogenous
    inputs during a genuine forecast, stated explicitly rather than left
    implicit.
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    curr = X_init_batch.copy().astype(float)
    W = curr.shape[0]

    max_lag = max(LAG_STEPS)
    max_roll = max(ROLL_WINDOWS)
    buf_len = max(max_lag, max_roll)

    if max_lag < buf_len:
        raise ValueError(f"LAG_STEPS must extend to at least max(ROLL_WINDOWS)={max_roll} "
                         f"to support rolling features in the recursion.")

    temp_buf = np.zeros((W, buf_len))
    hum_buf = np.zeros((W, buf_len))
    for k in LAG_STEPS:
        if k > buf_len:
            continue
        temp_buf[:, buf_len - k] = curr[:, idx[f"temp_lag{k}"]]
        hum_buf[:, buf_len - k] = curr[:, idx[f"hum_lag{k}"]]

    start_dt_arr = pd.to_datetime(start_datetimes).values
    all_preds = np.zeros((W, n_steps, 2))

    for step in range(n_steps):
        future_dt = pd.to_datetime(start_dt_arr) + pd.to_timedelta(
            freq_hours * (step + 1), unit="h")
        hours = future_dt.hour.values
        months = future_dt.month.values
        doy = future_dt.dayofyear.values

        curr[:, idx["hour_sin"]] = np.sin(2 * np.pi * hours / 24)
        curr[:, idx["hour_cos"]] = np.cos(2 * np.pi * hours / 24)
        curr[:, idx["month_sin"]] = np.sin(2 * np.pi * months / 12)
        curr[:, idx["month_cos"]] = np.cos(2 * np.pi * months / 12)
        curr[:, idx["day_of_year_sin"]] = np.sin(2 * np.pi * doy / 365.25)
        curr[:, idx["day_of_year_cos"]] = np.cos(2 * np.pi * doy / 365.25)

        for k in LAG_STEPS:
            curr[:, idx[f"temp_lag{k}"]] = temp_buf[:, -k]
            curr[:, idx[f"hum_lag{k}"]] = hum_buf[:, -k]
        curr[:, idx["temp_chg_24h"]] = temp_buf[:, -1] - temp_buf[:, -4]
        curr[:, idx["hum_chg_24h"]] = hum_buf[:, -1] - hum_buf[:, -4]

        for w in ROLL_WINDOWS:
            curr[:, idx[f"temp_roll{w}_mean"]] = temp_buf[:, -w:].mean(axis=1)
            curr[:, idx[f"hum_roll{w}_mean"]] = hum_buf[:, -w:].mean(axis=1)
            curr[:, idx[f"temp_roll{w}_std"]] = temp_buf[:, -w:].std(axis=1)
            curr[:, idx[f"hum_roll{w}_std"]] = hum_buf[:, -w:].std(axis=1)
        # ATMOS_FEATURES and ATMOS_LAGGED left unchanged (persisted) — no
        # future values available during a genuine forecast.

        next_pred = model.predict(curr)
        all_preds[:, step, :] = next_pred

        temp_buf = np.column_stack([temp_buf[:, 1:], next_pred[:, 0]])
        hum_buf = np.column_stack([hum_buf[:, 1:], next_pred[:, 1]])

    return all_preds


def recursive_forecast(model, initial_features, feature_names, n_steps,
                        start_datetime, freq_hours=6):
    """Single-window convenience wrapper around recursive_forecast_batch."""
    X = np.asarray(initial_features).reshape(1, -1)
    preds = recursive_forecast_batch(model, X, feature_names, n_steps,
                                       [start_datetime], freq_hours)
    return preds[0]


def evaluate_multi_step(model, X_test, y_test, test_datetimes, feature_names,
                         horizons_hours=(6, 24, 48, 96), freq_hours=6,
                         direct_stack_metrics=None, tolerance=0.15):
    """[FIX #10] Reports RMSE/R2 per horizon in addition to MAE, and cross-
    checks 1-step recursive MAE against the direct single-step MAE."""
    results = {"mae": {"temp": {}, "hum": {}},
                "rmse": {"temp": {}, "hum": {}},
                "r2": {"temp": {}, "hum": {}}}
    for h_hours in horizons_hours:
        n_steps = max(1, h_hours // freq_hours)
        starts = list(range(0, len(X_test) - n_steps, max(1, n_steps)))
        if not starts:
            continue
        init_batch = X_test.iloc[starts].values.astype(float)
        start_dts = test_datetimes.iloc[starts]
        preds_batch = recursive_forecast_batch(model, init_batch, feature_names,
                                                 n_steps, start_dts, freq_hours)

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
            print(f"  Horizon {h_hours}h ({n_steps} steps) | "
                  f"Temp MAE={results['mae']['temp'][h_hours]:.4f} "
                  f"RMSE={results['rmse']['temp'][h_hours]:.4f} "
                  f"R2={results['r2']['temp'][h_hours]:.4f} | "
                  f"Hum MAE={results['mae']['hum'][h_hours]:.4f} "
                  f"RMSE={results['rmse']['hum'][h_hours]:.4f} "
                  f"R2={results['r2']['hum'][h_hours]:.4f}")

    smallest_h = min(horizons_hours)
    if direct_stack_metrics is not None and smallest_h in results["mae"]["temp"]:
        for var in ("temp", "hum"):
            direct_mae = direct_stack_metrics[f"{var}_mae"]
            recursive_mae = results["mae"][var][smallest_h]
            if direct_mae > 0 and abs(recursive_mae - direct_mae) / direct_mae > tolerance:
                print(f"  [WARNING] {var}: direct MAE={direct_mae:.4f} vs "
                      f"{smallest_h}h recursive MAE={recursive_mae:.4f} — "
                      f"disagree by >{tolerance:.0%}.")
            else:
                print(f"  [OK] {var}: direct MAE and {smallest_h}h recursive MAE agree "
                      f"within tolerance ({direct_mae:.4f} vs {recursive_mae:.4f}).")
    return results


def plot_recursive_forecast_for_city(model, df_raw, city_location_name,
                                      feature_names, le_region, le_cond, le_city,
                                      train_ratio=0.75, n_steps=8, freq_hours=6):
    """[FIX #11] Guarded by _check_city_allowed. Also runs the data-integrity
    tripwire (humidity ∈ [0,100], temperature within plausible range)."""
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
        raise ValueError(f"Not enough test rows for '{city_location_name}' to "
                          f"forecast {n_steps} steps.")

    init_row = group.iloc[n_train][feature_names].values.astype(float)
    start_dt = group.iloc[n_train]["datetime"]
    preds = recursive_forecast(model, init_row, feature_names, n_steps,
                                start_dt, freq_hours)

    true_temp = group["temp"].iloc[n_train + 1: n_train + 1 + n_steps].values
    true_hum = group["hum"].iloc[n_train + 1: n_train + 1 + n_steps].values

    if (preds[:, 1] < 0).any() or (preds[:, 1] > 100).any():
        raise ValueError(f"Refusing to save figure for '{display_name}': forecasted "
                          f"humidity outside [0, 100]% "
                          f"(min={preds[:,1].min():.1f}, max={preds[:,1].max():.1f}).")
    if (preds[:, 0] < TEMP_RANGE_C[0]).any() or (preds[:, 0] > TEMP_RANGE_C[1]).any():
        raise ValueError(f"Refusing to save figure for '{display_name}': forecasted "
                          f"temperature outside {TEMP_RANGE_C} deg C.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{n_steps * freq_hours}-Hour Recursive Forecast — {display_name}")
    axes[0].plot(true_temp, label="Observed", marker="o", ms=3)
    axes[0].plot(preds[:, 0], label="Forecast", marker="x", ms=3)
    axes[0].set_ylabel("Temperature (deg C)"); axes[0].legend()
    axes[1].plot(true_hum, label="Observed", marker="o", ms=3)
    axes[1].plot(preds[:, 1], label="Forecast", marker="x", ms=3)
    axes[1].set_ylabel("Humidity (%)")
    axes[1].set_xlabel(f"Steps ahead ({freq_hours}h each)"); axes[1].legend()
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
        span = df_m[col].max() - df_m[col].min()
        pad = span * 0.02 if span > 0 else 0.02
        for bar, val in zip(bars, df_m[col]):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            offset = pad if y >= 0 else -pad
            ax.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{val:.3f}",
                    ha="center", va=va, fontsize=9)
        return bars

    for i, (col, title) in enumerate([("temp_r2", "Temp R2"),
                                       ("temp_mae", "Temp MAE (C)"),
                                       ("temp_rmse", "Temp RMSE (C)")]):
        _bar_with_labels(axes[0, i], col, "#4C72B0")
        axes[0, i].set_title(title)
        axes[0, i].tick_params(axis="x", rotation=20)
    for i, (col, title) in enumerate([("hum_r2", "Hum R2"),
                                       ("hum_mae", "Hum MAE (%)"),
                                       ("hum_rmse", "Hum RMSE (%)")]):
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
        pred = preds_dict[name]
        n = len(pred)
        y_temp = y_test["temp"].iloc[-n:].values
        y_hum = y_test["hum"].iloc[-n:].values
        axes[0, i].scatter(y_temp, pred[:, 0], alpha=0.3, s=8, color="#4C72B0")
        lims = [min(y_temp.min(), pred[:, 0].min()), max(y_temp.max(), pred[:, 0].max())]
        axes[0, i].plot(lims, lims, "r--")
        axes[0, i].set_title(f"{name}\nTemp R2={r2_score(y_temp, pred[:,0]):.4f}")
        axes[1, i].scatter(y_hum, pred[:, 1], alpha=0.3, s=8, color="#55A868")
        lims_h = [min(y_hum.min(), pred[:, 1].min()), max(y_hum.max(), pred[:, 1].max())]
        axes[1, i].plot(lims_h, lims_h, "r--")
        axes[1, i].set_title(f"Hum R2={r2_score(y_hum, pred[:,1]):.4f}")
    plt.tight_layout(); plt.savefig(out_path("model_scatter_comparisons.png"), dpi=200); plt.close()


# ============================================================================
# 14. MAIN PIPELINE
# ============================================================================

def main(fast_smoke_test=False):
    """
    fast_smoke_test=True shrinks ARIMA coverage and LSTM epochs for a quick
    end-to-end verification. Set False for the real manuscript run.
    """
    print("=" * 70); print("STEP 1: Provenance + load + validate"); print("=" * 70)
    get_dataset_provenance(FILEPATH)
    df_raw = load_and_validate(FILEPATH)

    print("\n" + "=" * 70)
    print("STEP 2: Feature engineering + per-city chronological split  [REV 4 - A]")
    print("=" * 70)
    (df_full, X_train, X_test, y_train, y_test,
     le_region, le_cond, le_city, feature_names) = prepare_data(df_raw)
    city_array_test = df_full.loc[X_test.index, "city"].values
    city_array_train = df_full.loc[X_train.index, "city"].values

    print("\n" + "=" * 70)
    print("STEP 3: Persistence baseline (per-city)  [FIX #3] [REV 4 - E]")
    print("=" * 70)
    persistence_metrics, persistence_df = persistence_baseline(y_test, city_array_test)
    print(f"Temp: R2={persistence_metrics['temp_r2']:.4f} "
          f"RMSE={persistence_metrics['temp_rmse']:.4f} "
          f"MAE={persistence_metrics['temp_mae']:.4f}")
    print(f"Hum:  R2={persistence_metrics['hum_r2']:.4f} "
          f"RMSE={persistence_metrics['hum_rmse']:.4f} "
          f"MAE={persistence_metrics['hum_mae']:.4f}")

    print("\n" + "=" * 70)
    print("STEP 4: Stacked ensemble + per-city bias correction  [FIX #4] [REV 4 - B]")
    print("=" * 70)
    n_est = 60 if fast_smoke_test else 200
    model = get_stacked_model(n_estimators=n_est)
    model.fit(X_train, y_train)
    y_pred_stack_raw = model.predict(X_test)

    for tname, est in zip(["temp", "hum"], model.estimators_):
        print(f"  Meta-learner ({tname}): {est.n_meta_train_} out-of-fold rows "
              f"of {len(X_train)} training rows")

    bias_table = fit_per_city_bias(model, X_train, y_train, df_full,
                                    val_frac=0.2, alpha=0.5)
    y_pred_stack = apply_per_city_bias(y_pred_stack_raw, city_array_test, bias_table)

    stack_metrics = compute_metrics(y_test, y_pred_stack)
    stack_metrics["temp_r2_anom"] = per_city_anomaly_r2(
        y_test["temp"].values, y_pred_stack[:, 0], city_array_test)
    stack_metrics["hum_r2_anom"] = per_city_anomaly_r2(
        y_test["hum"].values, y_pred_stack[:, 1], city_array_test)

    print(f"Temp: R2={stack_metrics['temp_r2']:.4f} "
          f"RMSE={stack_metrics['temp_rmse']:.4f} "
          f"MAE={stack_metrics['temp_mae']:.4f} | "
          f"R2_anom={stack_metrics['temp_r2_anom']:.4f}")
    print(f"Hum:  R2={stack_metrics['hum_r2']:.4f} "
          f"RMSE={stack_metrics['hum_rmse']:.4f} "
          f"MAE={stack_metrics['hum_mae']:.4f} | "
          f"R2_anom={stack_metrics['hum_r2_anom']:.4f}")

    improvement = print_improvement_over_persistence(persistence_metrics, stack_metrics)

    print("\n" + "=" * 70); print("STEP 5: Walk-forward validation  [FIX #1]"); print("=" * 70)
    cv_results = walk_forward_validation(
        lambda: get_stacked_model(n_estimators=n_est), X_train, y_train, n_splits=5)
    if not cv_results.empty:
        cv_mean = cv_results.mean(numeric_only=True)
        cv_std = cv_results.std(numeric_only=True)
        print(f"Temp R2: {cv_mean['temp_r2']:.4f} +/- {cv_std['temp_r2']:.4f} | "
              f"RMSE: {cv_mean['temp_rmse']:.4f} +/- {cv_std['temp_rmse']:.4f}")
        print(f"Hum  R2: {cv_mean['hum_r2']:.4f} +/- {cv_std['hum_r2']:.4f} | "
              f"RMSE: {cv_mean['hum_rmse']:.4f} +/- {cv_std['hum_rmse']:.4f}")
        cv_results.to_csv(out_path("walk_forward_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 6: ARIMA on the SAME held-out rows  [FIX #6] [REV 4 - D]")
    print("=" * 70)
    city_subset = list(STUDY_CITIES) if fast_smoke_test else None

    arima_temp_pred = arima_rolling_predictions(
        df_full, y_test, target="temp", city_subset=city_subset)
    arima_hum_pred = arima_rolling_predictions(
        df_full, y_test, target="hum", city_subset=city_subset)

    arima_temp_pred.to_csv(out_path("arima_temp_predictions.csv"), index=False)
    arima_hum_pred.to_csv(out_path("arima_hum_predictions.csv"), index=False)

    arima_temp_overall = arima_pooled_metrics(arima_temp_pred)
    arima_hum_overall = arima_pooled_metrics(arima_hum_pred)
    print(f"ARIMA Temp pooled: R2={arima_temp_overall['r2']:.4f} "
          f"RMSE={arima_temp_overall['rmse']:.4f} "
          f"MAE={arima_temp_overall['mae']:.4f} n={arima_temp_overall['n']}")
    print(f"ARIMA Hum  pooled: R2={arima_hum_overall['r2']:.4f} "
          f"RMSE={arima_hum_overall['rmse']:.4f} "
          f"MAE={arima_hum_overall['mae']:.4f} n={arima_hum_overall['n']}")

    arima_temp_cities = arima_per_city_metrics(arima_temp_pred)
    arima_hum_cities = arima_per_city_metrics(arima_hum_pred)
    arima_temp_cities.to_csv(out_path("arima_temp_per_city.csv"), index=False)
    arima_hum_cities.to_csv(out_path("arima_hum_per_city.csv"), index=False)

    # Align ARIMA predictions to y_test rows for R2_anom and comparison plots.
    arima_temp_aligned = (arima_temp_pred.set_index(["city", "datetime"])["y_pred"])
    arima_hum_aligned = (arima_hum_pred.set_index(["city", "datetime"])["y_pred"])
    test_keys = list(zip(df_full.loc[X_test.index, "city"].values,
                          df_full.loc[X_test.index, "datetime"].values))
    arima_temp_vec = arima_temp_aligned.reindex(test_keys).values
    arima_hum_vec = arima_hum_aligned.reindex(test_keys).values

    arima_temp_overall["r2_anom"] = per_city_anomaly_r2(
        y_test["temp"].values, arima_temp_vec, city_array_test)
    arima_hum_overall["r2_anom"] = per_city_anomaly_r2(
        y_test["hum"].values, arima_hum_vec, city_array_test)
    print(f"ARIMA R2_anom: Temp={arima_temp_overall['r2_anom']:.4f} "
          f"Hum={arima_hum_overall['r2_anom']:.4f}")

    print("\n" + "=" * 70); print("STEP 7: LSTM baseline  [FIX #6]"); print("=" * 70)
    lstm_epochs = 15 if fast_smoke_test else 100
    lstm_pred_temp, lstm_pred_hum, lstm_metrics = run_lstm_full(
        X_train, X_test, y_train, y_test, epochs=lstm_epochs)
    if len(lstm_pred_temp):
        print(f"Temp: R2={lstm_metrics['temp_r2']:.4f} "
              f"RMSE={lstm_metrics['temp_rmse']:.4f} "
              f"MAE={lstm_metrics['temp_mae']:.4f}")
        print(f"Hum:  R2={lstm_metrics['hum_r2']:.4f} "
              f"RMSE={lstm_metrics['hum_rmse']:.4f} "
              f"MAE={lstm_metrics['hum_mae']:.4f}")

    print("\n" + "=" * 70); print("STEP 8: Multi-model comparison visuals"); print("=" * 70)
    metrics_dict = {
        "Persistence": persistence_metrics,
        "ARIMA": {"temp_r2": arima_temp_overall["r2"],
                   "temp_rmse": arima_temp_overall["rmse"],
                   "temp_mae": arima_temp_overall["mae"],
                   "hum_r2": arima_hum_overall["r2"],
                   "hum_rmse": arima_hum_overall["rmse"],
                   "hum_mae": arima_hum_overall["mae"]},
    }
    preds_dict = {}
    if len(persistence_df.dropna(subset=["temp_pred", "hum_pred"])):
        valid_mask = persistence_df[["temp_pred", "hum_pred"]].notna().all(axis=1)
        preds_dict["Persistence"] = persistence_df.loc[valid_mask, ["temp_pred", "hum_pred"]].values
    if len(lstm_pred_temp):
        metrics_dict["LSTM"] = lstm_metrics
        preds_dict["LSTM"] = np.column_stack([lstm_pred_temp, lstm_pred_hum])
    metrics_dict["Stacked Ensemble"] = stack_metrics
    preds_dict["Stacked Ensemble"] = y_pred_stack

    try:
        plot_comparative_metrics(metrics_dict)
        plot_multi_model_scatters(y_test, preds_dict)
        print(f"Saved comparison plots. Models compared: {list(metrics_dict.keys())}")
    except Exception as e:
        print(f"Comparison plots skipped: {e}")

    print("\n" + "=" * 70)
    print("STEP 9: True Leave-One-City-Out  [FIX #2]")
    print("=" * 70)
    loco_city_subset = (list(STUDY_CITIES) + sorted(df_full["city"].unique())[:20]
                         if fast_smoke_test else None)
    loco_temp = leave_one_city_out(df_full, "temp", city_subset=loco_city_subset)
    loco_hum = leave_one_city_out(df_full, "hum", city_subset=loco_city_subset)
    loco_temp.to_csv(out_path("loco_temp_results.csv"), index=False)
    loco_hum.to_csv(out_path("loco_hum_results.csv"), index=False)

    print("\n" + "=" * 70); print("STEP 10: SHAP analysis"); print("=" * 70)
    try:
        X_sample = X_test.sample(min(200, len(X_test)), random_state=42)
        shap_analysis(model, X_sample, feature_names)
    except Exception as e:
        print(f"SHAP skipped: {e}")

    print("\n" + "=" * 70); print("STEP 11: Residual diagnostics"); print("=" * 70)
    plot_residuals(y_test["temp"].values, y_pred_stack[:, 0], "temperature")
    plot_residuals(y_test["hum"].values, y_pred_stack[:, 1], "humidity")

    print("\n" + "=" * 70)
    print("STEP 12: Multi-step recursive forecasting  [FIX #10] [REV 4 - A]")
    print("=" * 70)
    test_datetimes = df_full.loc[X_test.index, "datetime"]
    horizons = (6, 24, 48) if fast_smoke_test else (6, 24, 48, 96)
    multi_results = evaluate_multi_step(
        model, X_test, y_test, test_datetimes, feature_names,
        horizons_hours=horizons, direct_stack_metrics=stack_metrics)
    pd.DataFrame(multi_results["mae"]).to_csv(out_path("multi_step_mae.csv"))
    pd.DataFrame(multi_results["rmse"]).to_csv(out_path("multi_step_rmse.csv"))
    pd.DataFrame(multi_results["r2"]).to_csv(out_path("multi_step_r2.csv"))

    print("\n" + "=" * 70)
    print("STEP 12b: Per-location tables with bootstrap CIs  [REV 4 - C]")
    print("=" * 70)
    df_test_meta = df_full.loc[X_test.index]

    # Stacked ensemble
    tbl_t = per_location_table(y_test["temp"].values, y_pred_stack[:, 0], df_test_meta)
    tbl_h = per_location_table(y_test["hum"].values, y_pred_stack[:, 1], df_test_meta)
    tbl_t.to_csv(out_path("per_location_stacked_temp.csv"), index=False)
    tbl_h.to_csv(out_path("per_location_stacked_hum.csv"), index=False)

    study_t = tbl_t[tbl_t["in_study_set"]]
    study_h = tbl_h[tbl_h["in_study_set"]]
    study_t.to_csv(out_path("per_location_stacked_temp_study_cities.csv"), index=False)
    study_h.to_csv(out_path("per_location_stacked_hum_study_cities.csv"), index=False)
    if len(study_t):
        print(f"  Stacked ensemble — study-city subset: "
              f"Temp R2 median={study_t['r2'].median():.4f} "
              f"(range {study_t['r2'].min():.3f}..{study_t['r2'].max():.3f}); "
              f"Hum R2 median={study_h['r2'].median():.4f}")

    # Persistence
    valid_p = persistence_df.dropna(subset=["temp_pred", "hum_pred"])
    if len(valid_p):
        meta_p = df_test_meta.loc[valid_p.index]
        tbl_t_p = per_location_table(valid_p["temp"].values, valid_p["temp_pred"].values, meta_p)
        tbl_h_p = per_location_table(valid_p["hum"].values, valid_p["hum_pred"].values, meta_p)
        tbl_t_p.to_csv(out_path("per_location_persistence_temp.csv"), index=False)
        tbl_h_p.to_csv(out_path("per_location_persistence_hum.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 13: Per-city 48h forecast figures, all 8 study cities  [FIX #11]")
    print("=" * 70)
    for city_loc_name in STUDY_CITIES:
        try:
            plot_recursive_forecast_for_city(model, df_raw, city_loc_name,
                                              feature_names, le_region, le_cond,
                                              le_city, n_steps=8)
        except Exception as e:
            print(f"  [SKIPPED] {STUDY_CITIES[city_loc_name]}: {e}")

    print("\n" + "=" * 70); print("SUMMARY"); print("=" * 70)
    summary = {
        "persistence": persistence_metrics,
        "stacked_ensemble": stack_metrics,
        "improvement_over_persistence_pct": improvement,
        "arima": {"temp": arima_temp_overall, "hum": arima_hum_overall},
        "lstm": lstm_metrics if len(lstm_pred_temp) else None,
        "bias_table": (bias_table.reset_index().to_dict(orient="records")
                        if not bias_table.empty else []),
        "note_r2_anom": (
            "R2_anom is a per-city skill score against a climatological mean, "
            "computed from within-city variance. It is NOT the same quantity as "
            "pooled R2 and must be labelled distinctly in the manuscript."
        ),
    }
    print(json.dumps(summary, indent=2, default=str))

    with open(out_path("full_results_report.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    joblib.dump(model, out_path("stacked_model_final.pkl"))
    if not bias_table.empty:
        bias_table.to_csv(out_path("per_city_bias_table.csv"))
    print(f"\nAll artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(fast_smoke_test=False)