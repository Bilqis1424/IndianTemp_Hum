"""
================================================================================
STANDALONE RESIDUAL DIAGNOSTICS – FULL 22‑FEATURE VERSION
--------------------------------------------------------------------------------
Matches the feature set of the model saved from indian_weather_pipeline_v3_final.
================================================================================
"""

import os
import sys
import types
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# 0. CREATE FAKE MODULE FOR THE ORIGINAL SCRIPT (to avoid ModuleNotFoundError)
# ============================================================================
fake_module = types.ModuleType("indian_weather_pipeline_v3_final")
sys.modules["indian_weather_pipeline_v3_final"] = fake_module

class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
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
            raise ValueError("No out-of-fold predictions generated.")
        self.final_estimator_ = clone(self.final_estimator).fit(oof[covered], y[covered])
        self.fitted_base_estimators_ = [(name, clone(est).fit(X, y)) for name, est in self.base_estimators]
        return self

    def predict(self, X):
        X = np.asarray(X)
        base_preds = np.column_stack([m.predict(X) for _, m in self.fitted_base_estimators_])
        return self.final_estimator_.predict(base_preds)

fake_module.TimeSeriesStackingRegressor = TimeSeriesStackingRegressor

# ============================================================================
# 1. CONFIGURATION – UPDATE PATHS
# ============================================================================
DATA_PATH = "csv files/IndianWeatherRepository_raw.xlsx"          # or .csv
MODEL_PATH = "indianweather files/stacked_model_final.pkl"
OUTPUT_DIR = "./residual_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 2. FULL FEATURE SET (22 features – matches the model)
# ============================================================================
# These are the exact features the model was trained on, in the same order.
# Copied from the main script's feature list.

FULL_FEATURES = [
    "temp_lag1", "hum_lag1", "temp_lag2", "hum_lag2",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "region_enc", "condition_enc", "latitude", "longitude",
    "pressure_mb", "wind_kph", "cloud", "uv_index", "precip_mm",
    "visibility_km", "air_quality_PM2.5",
    "city_enc"
]

# ============================================================================
# 3. DATA PREPARATION (replicates the main pipeline)
# ============================================================================

def load_and_prepare_test_data(filepath):
    # Load raw file (Excel or CSV)
    if filepath.endswith((".xlsx", ".xls")):
        df_raw = pd.read_excel(filepath)
    else:
        df_raw = pd.read_csv(filepath, low_memory=False)

    # Rename columns to standard names
    df_raw = df_raw.rename(columns={
        "location_name": "city",
        "temperature_celsius": "temp",
        "humidity": "hum",
    })
    df_raw["datetime"] = pd.to_datetime(df_raw["last_updated"])
    df_raw = df_raw.sort_values(["city", "datetime"]).reset_index(drop=True)

    # Feature engineering (same as in the main script)
    df = df_raw.copy()
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

    # Encode categoricals (region, condition, city)
    le_region = LabelEncoder()
    le_cond = LabelEncoder()
    le_city = LabelEncoder()
    df["region_enc"] = le_region.fit_transform(df["region"].astype(str))
    df["condition_enc"] = le_cond.fit_transform(df["condition_text"].astype(str))
    df["city_enc"] = le_city.fit_transform(df["city"].astype(str))

    # Add atmospheric features (already present in raw data; just ensure numeric)
    atmos_cols = ["pressure_mb", "wind_kph", "cloud", "uv_index", "precip_mm",
                  "visibility_km", "air_quality_PM2.5"]
    for col in atmos_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with all features present
    df = df.dropna(subset=FULL_FEATURES).reset_index(drop=True)

    # Per‑city chronological split (75/25) – same as training
    X = df[FULL_FEATURES].astype(np.float32)
    y = df[["temp", "hum"]].astype(np.float32)

    X_test_list, y_test_list = [], []
    for city in df["city"].unique():
        idx = np.where((df["city"] == city).values)[0]
        n = len(idx)
        split_idx = int(n * 0.75)
        if split_idx < 1 or n - split_idx < 1:
            continue
        X_test_list.append(X.iloc[idx[split_idx:]])
        y_test_list.append(y.iloc[idx[split_idx:]])
    X_test = pd.concat(X_test_list)
    y_test = pd.concat(y_test_list)

    print(f"Test set features: {X_test.shape[1]} (expected {len(FULL_FEATURES)})")
    print(f"Test set rows: {X_test.shape[0]}")
    return X_test, y_test

# ============================================================================
# 4. RESIDUAL PLOTTING (unchanged)
# ============================================================================

def plot_residuals(y_true, y_pred, target_name, output_dir):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Residual Diagnostics: {target_name}", fontsize=14)

    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("Predicted"); axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Predicted")

    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot")

    axes[1, 0].hist(residuals, bins=40, edgecolor="black", alpha=0.7)
    axes[1, 0].set_xlabel("Residuals"); axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Residuals")

    axes[1, 1].scatter(range(len(residuals)), residuals, alpha=0.4, s=8)
    axes[1, 1].axhline(0, color="r", linestyle="--")
    axes[1, 1].set_xlabel("Row Order"); axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].set_title("Residuals by Row Order")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"residuals_{target_name}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("Loading dataset and preparing test set...")
    X_test, y_test = load_and_prepare_test_data(DATA_PATH)

    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please update MODEL_PATH.")

    model = joblib.load(MODEL_PATH)

    print("Generating predictions on test set...")
    y_pred = model.predict(X_test.values)

    print("Plotting residuals...")
    plot_residuals(y_test["temp"].values, y_pred[:, 0], "temperature", OUTPUT_DIR)
    plot_residuals(y_test["hum"].values, y_pred[:, 1], "humidity", OUTPUT_DIR)

    print(f"\nAll residual plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()