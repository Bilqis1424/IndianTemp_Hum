"""
================================================================================
STANDALONE SCRIPT – INTERACTIVE INDIA MAP (RAW VS PREDICTED)
--------------------------------------------------------------------------------
Generates an interactive Folium map of all 543 locations. Each marker shows
observed and predicted temperature and humidity when clicked.
Uses predicted values from the stacked ensemble model (loaded from checkpoint).
================================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import folium
from folium.plugins import MarkerCluster
import webbrowser

# Ensure the main script's custom class is available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from indian_weather_pipeline_v3_final import (
        TimeSeriesStackingRegressor,
        load_and_validate,
        engineer_features,
        encode_categoricals,
        FULL_FEATURES,
    )
except ImportError:
    # Fallback definitions (same as previous scripts)
    from sklearn.base import BaseEstimator, RegressorMixin, clone
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.preprocessing import LabelEncoder

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

    def load_and_validate(filepath):
        if filepath.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath, low_memory=False)
        df = df.rename(columns={"location_name": "city", "temperature_celsius": "temp", "humidity": "hum"})
        df["datetime"] = pd.to_datetime(df["last_updated"])
        return df.sort_values(["city", "datetime"]).reset_index(drop=True)

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

    FULL_FEATURES = [
        "temp_lag1", "hum_lag1", "temp_lag2", "hum_lag2",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "region_enc", "condition_enc", "latitude", "longitude",
        "pressure_mb", "wind_kph", "cloud", "uv_index", "precip_mm",
        "visibility_km", "air_quality_PM2.5", "city_enc"
    ]

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = "csv files/IndianWeatherRepository_raw.xlsx"
MODEL_PATH = "checkpoints/stacked_model_final.pkl"
CHECKPOINT_PREDICTIONS = "checkpoints/predictions_all_locations.pkl"  # optional
OUTPUT_HTML = "india_weather_map.html"

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")

def get_all_locations_data():
    """Load raw data, engineer features, and return DataFrame with features and targets."""
    df = load_and_validate(DATA_PATH)
    df = engineer_features(df)
    df, _, _, _ = encode_categoricals(df, fit=True)
    df = df.dropna(subset=FULL_FEATURES).reset_index(drop=True)
    # Keep only one row per city (the most recent, or any – we want a single prediction per location)
    # For the map, we'll use the last observation for each city.
    df_latest = df.sort_values(["city", "datetime"]).groupby("city").last().reset_index()
    return df_latest

# ============================================================================
# GENERATE MAP
# ============================================================================
def create_map(df, model):
    """Create a Folium map with markers for each location."""
    # Prepare features for prediction
    X = df[FULL_FEATURES].values.astype(np.float32)
    y_pred = model.predict(X)
    y_true_temp = df["temp"].values
    y_true_hum = df["hum"].values

    # Create base map centered on India
    center_lat, center_lon = 20.5937, 78.9629  # India center
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="OpenStreetMap")

    # Use MarkerCluster for cleaner display
    marker_cluster = MarkerCluster().add_to(m)

    for i, row in df.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        city = row["city"]
        obs_temp, obs_hum = y_true_temp[i], y_true_hum[i]
        pred_temp, pred_hum = y_pred[i, 0], y_pred[i, 1]

        # Popup content
        popup_text = f"""
        <b>{city}</b><br>
        <b>Observed:</b><br>
        Temperature: {obs_temp:.1f} °C<br>
        Humidity: {obs_hum:.1f} %<br>
        <b>Predicted (stacked ensemble):</b><br>
        Temperature: {pred_temp:.1f} °C<br>
        Humidity: {pred_hum:.1f} %<br>
        <b>Error (temp):</b> {obs_temp - pred_temp:.1f} °C
        """
        popup = folium.Popup(popup_text, max_width=300)

        # Colour marker based on temperature error (optional)
        error = obs_temp - pred_temp
        if abs(error) < 1.0:
            color = "green"
        elif abs(error) < 2.0:
            color = "orange"
        else:
            color = "red"

        folium.Marker(
            location=[lat, lon],
            popup=popup,
            icon=folium.Icon(color=color, icon="cloud"),
        ).add_to(marker_cluster)

    return m

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading dataset and model...")
    df_latest = get_all_locations_data()
    model = load_model()

    print("Generating predictions for all locations...")
    map_obj = create_map(df_latest, model)

    # Save map
    map_obj.save(OUTPUT_HTML)
    print(f"Map saved to {OUTPUT_HTML}")

    # Optionally open in browser
    try:
        webbrowser.open(OUTPUT_HTML)
    except:
        print("Open the HTML file manually in your browser.")

if __name__ == "__main__":
    main()