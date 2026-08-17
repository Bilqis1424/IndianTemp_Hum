"""
================================================================================
FINAL SCRIPT (REVISION 2) - ADDRESSES SECOND-ROUND REVIEWER COMMENTS
--------------------------------------------------------------------------------
Fixes applied relative to the previous version, tagged [REVIEWER #n] at each
change so they can be traced directly to the review letter:

  [REVIEWER #1]  Walk-forward validation kept and clarified as a DISTINCT
                 outer TimeSeriesSplit protocol (separate from the internal
                 stacking CV). Results reported as mean +/- SD.
  [REVIEWER #2]  LOCO now defaults to ALL cities ,
                 city_enc is explicitly dropped with a documented rationale
                 for how unseen-city encoding is handled, and full city-wise
                 results are exported to CSV for a manuscript table.
  [REVIEWER #3]  Persistence baseline retained (was already correct).
  [REVIEWER #4]  Replaced sklearn's StackingRegressor with a custom
                 TimeSeriesStackingRegressor. sklearn's StackingRegressor
                 builds its meta-learner's training data via
                 cross_val_predict, which REQUIRES the cv splitter's test
                 folds to partition the full dataset -- TimeSeriesSplit
                 does not (the first block is never a test fold), so
                 cv=TimeSeriesSplit(...) raises "cross_val_predict only
                 works for partitions" and cannot be used with
                 StackingRegressor at all. The custom class below generates
                 out-of-fold predictions manually via TimeSeriesSplit,
                 trains the meta-learner only on the chronologically valid
                 (covered) rows, and fits the base learners on the full
                 training set for inference -- the same idea as
                 StackingRegressor, without the partition restriction.
  [REVIEWER #6]  ARIMA now performs a small AIC-based grid search PER CITY to
                 select (p,d,q) instead of using an unexplained fixed order,
                 and the chosen order is logged per city. LSTM hyperparameters
                 remain fully specified in code (layers, dropout, batch size,
                 epochs, early stopping, validation split) so they can be
                 copied into the manuscript's methods section.
  [REVIEWER #7]  All reported metrics (ARIMA, LSTM, stacked ensemble,
                 persistence) are computed in ORIGINAL units (deg C, %) only.
                 No scaled-metric variant is produced by this script, to avoid
                 the Table 2 / Figure 4 unit-mismatch problem.
  [REVIEWER #9]  Dataset hash retained; the previously hard-coded download
                 date is replaced with the file's actual modification
                 timestamp (or an explicit override), so the reported date is
                 no longer a placeholder.
  [REVIEWER #10] evaluate_multi_step() now cross-checks its own 1-hour-horizon
                 MAE against the direct single-step stacked-model MAE and
                 prints a warning if they disagree by more than a small
                 tolerance, so a Table 2 vs Table 4 mismatch would be caught
                 automatically before submission rather than after review.
  [REVIEWER #11] A STUDY_CITIES allow-list is enforced before any per-city
                 multi-step forecast figure is generated, so a figure can no
                 longer be produced for a city (e.g. "Adilabad") that is not
                 among the defined study cities.
  [MINOR]        evaluate_multi_step() now reports RMSE and R^2 per horizon,
                 not only MAE.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import hashlib
import itertools
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
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
from scipy import stats
from pandas.plotting import autocorrelation_plot

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 0. STUDY DESIGN CONSTANTS
# ============================================================================

# All generated files (CSVs, PNGs, the saved model) are written here so
# everything lands in the Kaggle working directory rather than wherever the
# script happens to be invoked from.
OUTPUT_DIR = '/kaggle/working/'


def out_path(filename):
    """Build a path under OUTPUT_DIR, creating the directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


# [REVIEWER #11] Single source of truth for which cities belong to the study.
# Replace this with the exact eight cities listed in your Table 1. Any
# per-city figure function below will refuse to plot a city not in this list,
# which is what would have caught the "Adilabad" mislabeling before
# submission.
# [MERGE] Populated from the STUDY_CITIES constant used consistently across
# the uploaded EDA/visualization scripts (eda_lstm_arima.py, eda_visuals_rest.py) --
# update this list if your manuscript's Table 1 uses a different eight cities.
STUDY_CITIES = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Pune']



def _check_city_allowed(city, study_cities=STUDY_CITIES):
    """[REVIEWER #11] Refuse to proceed with a city outside the defined study set."""
    if study_cities and city not in study_cities:
        raise ValueError(
            f"City '{city}' is not in STUDY_CITIES {study_cities}. "
            f"Refusing to generate a figure/result for a city outside the "
            f"defined study scope (this is the exact issue flagged for "
            f"Figure 8 / 'Adilabad')."
        )

# ============================================================================
# 1. PROVEN DATA PREPARATION (unchanged from script #1, plus fixed hash/date)
# ============================================================================

def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_dataset_provenance(filepath, download_date=None):
    """
    [REVIEWER #9] Returns a real (hash, date) pair for reproducibility
    reporting, instead of a hard-coded placeholder date.

    - If `download_date` is supplied explicitly (e.g. the date you actually
      pulled the Kaggle snapshot), that value is used and reported verbatim.
    - Otherwise the file's last-modified timestamp on disk is used as the
      best available proxy and is clearly labeled as such.

    For a fully citable snapshot, archive this exact file to Zenodo (or
    similar) and report the resulting DOI in the manuscript instead of relying
    on hash+date alone, since the Kaggle source is described as "Live".
    """
    file_hash = compute_file_hash(filepath)
    if download_date is not None:
        date_str = download_date
        date_source = "user-provided download date"
    else:
        mtime = os.path.getmtime(filepath)
        date_str = dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        date_source = "file last-modified timestamp (proxy; supply the true " \
                      "download date explicitly if known)"
    print(f"Dataset hash: {file_hash}")
    print(f"Dataset date: {date_str}  [{date_source}]")
    return file_hash, date_str


def _resolve_columns(df_columns):
    """
    [FIX] Auto-detect the real column name for each required semantic field
    instead of assuming a fixed mapping that may not match this file. The
    previous version hardcoded `'wind_mph': 'datetime'` -- a numeric wind
    speed column parsed as if it were a date string -- which silently
    produced an all-NaT column and a 0-row dataset after dropna. Candidates
    are tried in priority order; the first one present in the dataframe is
    used.
    """
    candidates = {
        'city': ['location_name', 'city', 'timezone'],
        'region': ['region', 'state', 'country', 'timezone'],
        'datetime_col': ['last_updated', 'localtime', 'date', 'datetime', 'last_updated_epoch'],
        'temp': ['temperature_celsius', 'temp_celsius', 'temperature', 'temp'],
        'hum': ['humidity', 'hum'],
        'condition': ['condition_text', 'condition', 'weather_condition'],
    }
    resolved = {}
    missing = []
    for field, opts in candidates.items():
        match = next((c for c in opts if c in df_columns), None)
        if match is None:
            missing.append((field, opts))
        else:
            resolved[field] = match
    if missing:
        raise KeyError(
            "Could not resolve the following required fields to any column "
            f"in the dataset: {missing}. Available columns: {list(df_columns)}. "
            "Add the correct column name to the relevant list inside "
            "_resolve_columns()."
        )
    return resolved


def _parse_datetime_flexibly(series):
    """
    [FIX] Correctly handles BOTH a Unix-epoch integer column (e.g.
    last_updated_epoch) and a human-readable date string column (e.g.
    last_updated), instead of assuming one fixed string format. Reports the
    parse success rate so a near-total parse failure -- the exact bug that
    silently produced a 0-row dataset before -- is caught immediately
    instead of propagating downstream as empty arrays.
    """
    if pd.api.types.is_numeric_dtype(series):
        parsed = pd.to_datetime(series, unit='s', errors='coerce')
    else:
        parsed = pd.to_datetime(series, format='%m/%d/%Y %H:%M', errors='coerce')
        if parsed.notna().mean() < 0.5:
            parsed = pd.to_datetime(series, errors='coerce')  # fall back to general parser
    success_rate = parsed.notna().mean()
    print(f"  Datetime parse success rate: {success_rate:.1%}")
    if success_rate < 0.5:
        sample = series.dropna().astype(str).head(3).tolist()
        raise ValueError(
            f"Fewer than 50% of rows produced a valid datetime (sample raw "
            f"values: {sample}). Refusing to continue with a dataset that "
            f"would be silently emptied out by dropna()."
        )
    return parsed


def load_and_standardize(filepath):
    """
    [FIX] Shared, defensive raw-data loader used by both prepare_data() and
    load_raw_for_arima_loco(), replacing the two near-duplicate hardcoded
    rename blocks that both had the wind_mph->datetime bug. Prints the
    detected columns and cleaning row counts, and raises a clear error
    instead of silently returning an empty dataframe.
    """
    df_raw = pd.read_csv(filepath, low_memory=False)
    df_raw = df_raw.loc[:, ~df_raw.columns.str.contains('^Unnamed', na=False)]
    print(f"Raw columns found ({len(df_raw.columns)}): {df_raw.columns.tolist()}")

    resolved = _resolve_columns(df_raw.columns)
    print(f"Resolved column mapping: {resolved}")

    df = df_raw[[resolved['city'], resolved['region'], resolved['datetime_col'],
                 resolved['temp'], resolved['hum'], resolved['condition']]].copy()
    df.columns = ['city', 'region', 'datetime', 'temp', 'hum', 'condition_text']

    df['datetime'] = _parse_datetime_flexibly(df['datetime'])
    df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
    df['hum'] = pd.to_numeric(df['hum'], errors='coerce')

    n_before = len(df)
    df = df.dropna(subset=['temp', 'hum', 'datetime', 'city', 'region'])
    n_after = len(df)
    print(f"Rows before/after cleaning: {n_before} -> {n_after}")
    if n_after == 0:
        raise ValueError(
            "All rows were dropped during cleaning even though the datetime "
            "parse succeeded -- check for NaNs in the resolved temp/hum/"
            "city/region columns printed above."
        )

    return df.sort_values(['city', 'datetime']).reset_index(drop=True)


def prepare_data(filepath, train_ratio=0.75, download_date=None):
    get_dataset_provenance(filepath, download_date=download_date)

    df = load_and_standardize(filepath)

    df['temp_lag1'] = df.groupby('city')['temp'].shift(1)
    df['hum_lag1'] = df.groupby('city')['hum'].shift(1)
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)

    le_city = LabelEncoder()
    le_region = LabelEncoder()
    le_cond = LabelEncoder()
    df['city_enc'] = le_city.fit_transform(df['city'].astype(str))
    df['region_enc'] = le_region.fit_transform(df['region'].astype(str))
    df['condition_enc'] = le_cond.fit_transform(df['condition_text'].astype(str))

    df = df.dropna().reset_index(drop=True)

    feature_cols = ['temp_lag1', 'hum_lag1', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'city_enc', 'region_enc', 'condition_enc']
    X = df[feature_cols].astype(np.float32)
    y = df[['temp', 'hum']].astype(np.float32)

    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"Data prepared: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, y_train, y_test, le_city, le_region, le_cond


# ============================================================================
# 2. PERSISTENCE BASELINE  [REVIEWER #3 - retained]
# ============================================================================

def persistence_baseline(y_test):
    """yhat_{t+1} = y_t, evaluated on the same test set as every other model."""
    if len(y_test) < 2:
        return {'temp_r2': np.nan, 'temp_rmse': np.nan, 'temp_mae': np.nan,
                'hum_r2': np.nan, 'hum_rmse': np.nan, 'hum_mae': np.nan}
    temp_pred = y_test['temp'].shift(1).values[1:]
    hum_pred = y_test['hum'].shift(1).values[1:]
    true_temp = y_test['temp'].values[1:]
    true_hum = y_test['hum'].values[1:]
    r2_temp = r2_score(true_temp, temp_pred)
    rmse_temp = np.sqrt(mean_squared_error(true_temp, temp_pred))
    mae_temp = mean_absolute_error(true_temp, temp_pred)
    r2_hum = r2_score(true_hum, hum_pred)
    rmse_hum = np.sqrt(mean_squared_error(true_hum, hum_pred))
    mae_hum = mean_absolute_error(true_hum, hum_pred)
    return {'temp_r2': r2_temp, 'temp_rmse': rmse_temp, 'temp_mae': mae_temp,
            'hum_r2': r2_hum, 'hum_rmse': rmse_hum, 'hum_mae': mae_hum}


# ============================================================================
# 3. STACKED ENSEMBLE  [REVIEWER #4 - FIXED: custom time-series-safe stacker]
# ============================================================================

class TimeSeriesStackingRegressor(BaseEstimator, RegressorMixin):
    """
    [REVIEWER #4] Drop-in replacement for sklearn's StackingRegressor that
    is safe to use with TimeSeriesSplit.

    Why this exists: sklearn's StackingRegressor builds the meta-learner's
    training data with `cross_val_predict`, which raises
    "cross_val_predict only works for partitions" if the cv splitter's test
    folds don't cover every row exactly once. TimeSeriesSplit's test folds
    never include the first block (there's nothing earlier to train on for
    it), so it is NOT a partition and StackingRegressor(cv=TimeSeriesSplit(...))
    fails immediately on .fit(). This class reimplements the same stacking
    idea by hand so it can use TimeSeriesSplit correctly:

      1. For each of `n_splits` TimeSeriesSplit folds, fit a fresh clone of
         every base model on that fold's (strictly earlier) training rows
         and predict on that fold's (strictly later) validation rows. This
         guarantees the meta-learner only ever sees predictions made from
         past data, eliminating the temporal leakage the reviewer flagged.
      2. Stack those out-of-fold predictions into a meta-feature matrix.
         Rows from the initial block (never in any validation fold) are
         necessarily excluded -- there is no leakage-free way to produce an
         out-of-fold prediction for the very first rows in a time series,
         so the meta-learner is trained on a slightly smaller, but strictly
         valid, subset. The number of rows used is stored in
         `self.n_meta_train_` for transparency in the manuscript.
      3. Fit the final (meta) estimator on that meta-feature matrix.
      4. Separately, fit each base model on the FULL training set (this is
         what actually gets used to generate predictions on new/test data --
         identical to how StackingRegressor's base estimators are refit on
         the whole training set after CV).
    """

    def __init__(self, base_estimators, final_estimator, n_splits=5):
        self.base_estimators = base_estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n_samples = len(X)

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        oof_preds = np.full((n_samples, len(self.base_estimators)), np.nan)

        for train_idx, val_idx in tscv.split(X):
            for j, (name, est) in enumerate(self.base_estimators):
                model = clone(est)
                model.fit(X[train_idx], y[train_idx])
                oof_preds[val_idx, j] = model.predict(X[val_idx])

        covered = ~np.isnan(oof_preds).any(axis=1)
        self.n_meta_train_ = int(covered.sum())
        if self.n_meta_train_ == 0:
            raise ValueError(
                "No out-of-fold predictions were generated -- n_splits is too "
                "large relative to the training set size."
            )

        self.final_estimator_ = clone(self.final_estimator)
        self.final_estimator_.fit(oof_preds[covered], y[covered])

        # Refit base models on the FULL training set for use at inference time
        self.fitted_base_estimators_ = []
        for name, est in self.base_estimators:
            model = clone(est)
            model.fit(X, y)
            self.fitted_base_estimators_.append((name, model))

        return self

    def predict(self, X):
        X = np.asarray(X)
        base_preds = np.column_stack(
            [model.predict(X) for _, model in self.fitted_base_estimators_]
        )
        return self.final_estimator_.predict(base_preds)


def get_stacked_model(n_splits=5):
    """
    [REVIEWER #4] Builds the time-series-safe stacked ensemble described
    above. Report `n_splits` and the resulting `n_meta_train_` (printed
    after fitting) in the manuscript's methods section so the meta-learner's
    effective training size is transparent.
    """
    base_models = [
        ('xgb', XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             random_state=42, verbosity=0)),
        ('lgbm', LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               random_state=42, verbose=-1)),
        ('cat', CatBoostRegressor(iterations=200, depth=6, learning_rate=0.05,
                                  random_seed=42, verbose=0))
    ]
    stack = TimeSeriesStackingRegressor(
        base_estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        n_splits=n_splits
    )
    return MultiOutputRegressor(stack, n_jobs=1)


# ============================================================================
# 4. METRICS  [FIX: guarded R^2 that reports "undefined" instead of exploding
#    when the true target has near-zero variance in the evaluation window]
# ============================================================================

def safe_r2_score(y_true, y_pred, min_variance=1e-3):
    """
    Standard R^2 divides by Var(y_true). When y_true is (near-)constant --
    e.g. a 62-row per-city window where temperature barely changes -- that
    denominator collapses toward zero and R^2 can blow up to any magnitude
    (e.g. -1e24) even when RMSE/MAE show the model is doing something
    reasonable. That is a metric-definition problem, not evidence of a bad
    or good model.

    Returns (r2_or_nan, variance). If variance(y_true) < min_variance, r2 is
    reported as NaN ("undefined") instead of a runaway number, and the
    caller can decide how to handle/flag it (e.g. exclude from an average,
    or report RMSE/MAE only for that row).
    """
    y_true = np.asarray(y_true, dtype=float)
    variance = np.var(y_true)
    if variance < min_variance or len(y_true) < 2:
        return np.nan, variance
    return r2_score(y_true, y_pred), variance


def diagnose_target_variance(df, group_col='city', value_cols=('temp', 'hum')):
    """
    [DIAGNOSTIC] For each city (or other grouping), reports:
      - n_rows, n_unique values, unique_ratio (1.0 = every reading distinct)
      - max_run: the longest streak of consecutive identical values
      - variance of the raw series

    Run this BEFORE trusting any per-city R^2. A low unique_ratio / long
    max_run for temp but not hum is the fingerprint of a data staleness or
    duplication artifact (cause A) rather than genuinely low variability
    (cause B) -- in the latter case unique_ratio would typically still be
    high (values differ slightly) even though the range is small.
    """
    def max_consecutive_run(s):
        # Longest run of consecutive equal values in a pandas Series
        return (s != s.shift()).cumsum().pipe(lambda g: s.groupby(g).transform('size')).max()

    rows = []
    for city, g in df.groupby(group_col):
        row = {'city': city, 'n_rows': len(g)}
        for col in value_cols:
            s = g[col].reset_index(drop=True)
            row[f'{col}_nunique'] = s.nunique()
            row[f'{col}_unique_ratio'] = s.nunique() / max(len(s), 1)
            row[f'{col}_max_run'] = int(max_consecutive_run(s)) if len(s) else 0
            row[f'{col}_variance'] = float(np.var(s.values)) if len(s) else np.nan
        rows.append(row)

    report = pd.DataFrame(rows)
    print("\n=== Target variance / staleness diagnostic ===")
    for col in value_cols:
        low_var = (report[f'{col}_variance'] < 1e-3).sum()
        print(f"{col}: {low_var}/{len(report)} cities have variance < 1e-3 "
              f"(median unique_ratio={report[f'{col}_unique_ratio'].median():.2f}, "
              f"median max_run={report[f'{col}_max_run'].median():.0f})")
    return report


def compute_metrics(y_true, y_pred):
    r2_temp, var_temp = safe_r2_score(y_true['temp'], y_pred[:, 0])
    r2_hum, var_hum = safe_r2_score(y_true['hum'], y_pred[:, 1])
    rmse_temp = np.sqrt(mean_squared_error(y_true['temp'], y_pred[:, 0]))
    rmse_hum = np.sqrt(mean_squared_error(y_true['hum'], y_pred[:, 1]))
    mae_temp = mean_absolute_error(y_true['temp'], y_pred[:, 0])
    mae_hum = mean_absolute_error(y_true['hum'], y_pred[:, 1])
    return {'temp_r2': r2_temp, 'temp_rmse': rmse_temp, 'temp_mae': mae_temp,
            'hum_r2': r2_hum, 'hum_rmse': rmse_hum, 'hum_mae': mae_hum,
            'temp_variance': var_temp, 'hum_variance': var_hum}


# ============================================================================
# 5. WALK-FORWARD VALIDATION  [REVIEWER #1 - retained, documented]
# ============================================================================

def walk_forward_validation(model, X_train, y_train, n_splits=5):
    """
    [REVIEWER #1] This is a DEDICATED rolling-origin protocol, distinct from
    the internal StackingRegressor CV in get_stacked_model(). It re-fits the
    ENTIRE stacked pipeline on an expanding training window and evaluates on
    a strictly later validation window, for n_splits folds. Report the
    returned per-fold table plus mean +/- SD (R2, RMSE, MAE) directly in the
    manuscript's validation subsection -- this is what was missing before.
    """
    if len(X_train) < n_splits * 2:
        return pd.DataFrame()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        metrics['fold'] = len(results)
        metrics['train_window_size'] = len(train_idx)
        metrics['val_window_size'] = len(val_idx)
        results.append(metrics)
    return pd.DataFrame(results)


# ============================================================================
# 6. LOAD RAW DATA FOR ARIMA/LOCO (uses the same shared, robust loader)
# ============================================================================

def load_raw_for_arima_loco(filepath):
    """[FIX] Now just delegates to load_and_standardize() -- see section 1 --
    instead of duplicating the same (previously buggy) column mapping."""
    return load_and_standardize(filepath)



# ============================================================================
# 7. ARIMA ORDER SELECTION  [REVIEWER #6 - NEW: explains the (p,d,q) choice]
# ============================================================================

def select_arima_order(train_series, p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1, 2)):
    """
    [REVIEWER #6] Reviewer asked whether ARIMA order selection was performed
    (ACF/PACF inspection, AIC/BIC comparison) or whether an unexplained fixed
    order was used. This performs a small AIC-based grid search over
    (p,d,q) in {0,1,2} x {0,1} x {0,1,2} (18 candidates) and returns the
    order with the lowest AIC on the training series. This is done PER CITY
    (not pooled), since each city's series has its own dynamics -- report
    this explicitly in the manuscript methods section, along with the
    candidate grid used here.
    """
    best_aic = np.inf
    best_order = (1, 0, 0)  # sane fallback if every candidate fails
    for p, d, q in itertools.product(p_range, d_range, q_range):
        try:
            fitted = ARIMA(train_series, order=(p, d, q)).fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = (p, d, q)
        except Exception:
            continue
    return best_order, best_aic


# ============================================================================
# 8. ROLLING ARIMA - ALL CITIES, WITH PER-CITY ORDER SELECTION
# ============================================================================

def arima_rolling_all_cities(filepath, target='temp', train_ratio=0.75,
                              min_test_samples=5, verbose=True,
                              select_order=True, fixed_order=(1, 0, 0)):
    """
    Rolling one-step-ahead ARIMA on ALL cities with >= min_test_samples test
    points, matching the stacked model's evaluation scope
    ([REVIEWER #6]: every baseline uses the SAME test period/cities/horizon
    as the proposed model -- there is no more "first 100 samples" shortcut).

    If select_order=True (default), each city gets its own AIC-selected
    (p,d,q) via select_arima_order(); the chosen order is logged and returned
    in the per-city results table so it can go straight into a supplementary
    table. Set select_order=False to force `fixed_order` for all cities
    (faster, but then that choice must be justified separately).
    """
    df = load_raw_for_arima_loco(filepath)
    all_cities = df['city'].unique()
    print(f"ARIMA rolling forecast on all {len(all_cities)} cities "
          f"(minimum {min_test_samples} test samples, "
          f"{'per-city AIC order selection' if select_order else 'fixed order'})")

    city_results = []
    for city in all_cities:
        group = df[df['city'] == city].sort_values('datetime').reset_index(drop=True)
        n_train = int(len(group) * train_ratio)
        if n_train < 10:
            continue
        train_series = group[target].iloc[:n_train].values
        test_series = group[target].iloc[n_train:].values
        if len(test_series) < min_test_samples:
            continue

        if select_order:
            order, aic = select_arima_order(train_series)
        else:
            order, aic = fixed_order, np.nan

        history = list(train_series)
        preds = []
        for t in range(len(test_series)):
            try:
                fitted = ARIMA(history, order=order).fit()
                pred = fitted.forecast()[0]
                preds.append(pred)
                history.append(test_series[t])
            except Exception as e:
                if verbose:
                    print(f"  ARIMA failed for {city} at step {t}: {e}")
                preds.append(np.nan)
                history.append(test_series[t])

        preds = np.array(preds)
        test_vals = np.array(test_series)
        valid = ~np.isnan(preds)
        if valid.sum() < min_test_samples:
            continue
        r2, variance = safe_r2_score(test_vals[valid], preds[valid])
        rmse = np.sqrt(mean_squared_error(test_vals[valid], preds[valid]))
        mae = mean_absolute_error(test_vals[valid], preds[valid])
        city_results.append({
            'city': city, 'order_p': order[0], 'order_d': order[1], 'order_q': order[2],
            'aic': aic, 'r2': r2, 'rmse': rmse, 'mae': mae, 'n': valid.sum(),
            'variance': variance, 'r2_undefined': np.isnan(r2)
        })

    if not city_results:
        return {'r2': np.nan, 'rmse': np.nan, 'mae': np.nan}, pd.DataFrame()

    df_res = pd.DataFrame(city_results)
    n_undefined = int(df_res['r2_undefined'].sum())
    if n_undefined > 0:
        print(f"  [NOTE] R^2 was undefined (near-zero target variance) for "
              f"{n_undefined}/{len(df_res)} cities -- excluded from the R^2 "
              f"average below; RMSE/MAE remain valid and are still included.")
    # Among cities with a defined R^2, still exclude extreme outliers as before
    df_res_clean = df_res[~df_res['r2_undefined'] & (df_res['r2'] > -5) & (df_res['r2'] < 1.2)]
    if len(df_res_clean) == 0:
        df_res_clean = df_res[~df_res['r2_undefined']]  # fallback if all are extreme
    if len(df_res_clean) == 0:
        df_res_clean = df_res  # last-resort fallback if everything is undefined/extreme
    overall = {
        'r2': df_res_clean['r2'].mean(),
        'rmse': df_res['rmse'].mean(),   # RMSE/MAE stay valid regardless of R^2 status
        'mae': df_res['mae'].mean()
    }
    print(f"ARIMA R^2 averaged over {len(df_res_clean)} valid cities (from {len(df_res)} attempted); "
          f"RMSE/MAE averaged over all {len(df_res)} cities.")
    return overall, df_res


# ============================================================================
# 9. LSTM BASELINE (unchanged; hyperparameters fully specified per [REVIEWER #6])
# ============================================================================

def run_lstm_full(X_train, X_test, y_train, y_test, n_steps=24):
    """
    Architecture / hyperparameters for the manuscript's methods section:
      - Input window: n_steps=24 timesteps
      - 3 stacked LSTM layers: 128 -> 64 -> 32 units, ReLU activation
      - BatchNormalization + Dropout (0.3, 0.3, 0.2) after each LSTM layer
      - Dense(16, relu) -> Dense(2) output head (temp, hum)
      - Optimizer: Adam, lr=0.001
      - EarlyStopping: monitor val_loss, patience=10, restore_best_weights
      - ReduceLROnPlateau: factor=0.5, patience=5, min_lr=1e-6
      - epochs=100 (capped by early stopping), batch_size=64
      - Validation split: last 20% of the training sequences (chronological,
        not random) -- i.e. a held-out-in-time validation slice, consistent
        with the time-series-appropriate approach used elsewhere.
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train.values)
    X_test_scaled = scaler_X.transform(X_test.values)
    y_train_scaled = scaler_y.fit_transform(y_train.values)
    y_test_scaled = scaler_y.transform(y_test.values)

    train_comb = np.column_stack([X_train_scaled, y_train_scaled])
    test_comb = np.column_stack([X_test_scaled, y_test_scaled])

    X_seq, y_seq = [], []
    for i in range(len(train_comb) - n_steps):
        X_seq.append(train_comb[i:i + n_steps, :])
        y_seq.append(train_comb[i + n_steps, -2:])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if len(X_seq) < 100:
        return np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    val_size = int(0.2 * len(X_seq))
    X_tr, X_val = X_seq[:-val_size], X_seq[-val_size:]
    y_tr, y_val = y_seq[:-val_size], y_seq[-val_size:]

    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(n_steps, X_seq.shape[2])),
        BatchNormalization(), Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=64, callbacks=[early_stop, reduce_lr], verbose=0)

    test_preds = []
    for i in range(len(test_comb) - n_steps):
        seq = test_comb[i:i + n_steps, :].reshape(1, n_steps, -1)
        pred = model.predict(seq, verbose=0)[0]
        test_preds.append(pred)
    if not test_preds:
        return np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    y_pred_scaled = np.array(test_preds)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)   # [REVIEWER #7] original units
    y_true_aligned = y_test.iloc[n_steps:].reset_index(drop=True)
    if len(y_true_aligned) != len(y_pred):
        return np.array([]), np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    r2_temp = r2_score(y_true_aligned['temp'], y_pred[:, 0])
    r2_hum = r2_score(y_true_aligned['hum'], y_pred[:, 1])
    rmse_temp = np.sqrt(mean_squared_error(y_true_aligned['temp'], y_pred[:, 0]))
    rmse_hum = np.sqrt(mean_squared_error(y_true_aligned['hum'], y_pred[:, 1]))
    mae_temp = mean_absolute_error(y_true_aligned['temp'], y_pred[:, 0])
    mae_hum = mean_absolute_error(y_true_aligned['hum'], y_pred[:, 1])
    return y_pred[:, 0], y_pred[:, 1], r2_temp, r2_hum, rmse_temp, rmse_hum, mae_temp, mae_hum


# ============================================================================
# 10. LEAVE-ONE-CITY-OUT  [REVIEWER #2 - FIXED: all cities by default,
#     documented unseen-city encoding strategy]
# ============================================================================

def leave_one_city_out(filepath, model_builder, n_cities=None, random_state=42):
    """
    [REVIEWER #2] Two changes from the prior version:

    1. `n_cities=None` by default now means ALL cities are evaluated (the
       reviewer wants the full city-wise breakdown for a paper centered on
       cross-location generalisation). Pass an int only to subsample for a
       quick local test run -- state clearly in the manuscript if you ever
       report a subsampled number instead of the full set.

    2. Unseen-city encoding: `city_enc` (a LabelEncoder-derived integer id
       learned from training cities) is DELIBERATELY EXCLUDED from the
       feature set below. A held-out city has no valid learned encoding --
       a fresh integer id would be meaningless to the trained trees, and a
       one-hot vector would be all zeros, either of which silently corrupts
       the prediction. Instead the model relies only on lag features,
       cyclical time features, region encoding, and condition encoding,
       none of which require the specific city identity to be known ahead
       of time. This must be stated explicitly in the manuscript's LOCO
       subsection, since it directly answers the reviewer's methodological
       question about how unseen cities are encoded at test time.
    """
    np.random.seed(random_state)
    df = load_raw_for_arima_loco(filepath)

    df['temp_lag1'] = df.groupby('city')['temp'].shift(1)
    df['hum_lag1'] = df.groupby('city')['hum'].shift(1)
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)

    le_region = LabelEncoder()
    le_cond = LabelEncoder()
    df['region_enc'] = le_region.fit_transform(df['region'].astype(str))
    df['condition_enc'] = le_cond.fit_transform(df['condition_text'].astype(str))
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['temp_lag1', 'hum_lag1', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'region_enc', 'condition_enc']  # city_enc intentionally excluded -- see docstring

    all_cities = df['city'].unique()
    if n_cities is not None and len(all_cities) > n_cities:
        sampled_cities = np.random.choice(all_cities, size=n_cities, replace=False)
        print(f"LOCO on a SUBSAMPLE of {len(sampled_cities)} cities (from {len(all_cities)} total). "
              f"State this explicitly in the manuscript if used for the reported results.")
    else:
        sampled_cities = all_cities
        print(f"LOCO on ALL {len(sampled_cities)} cities.")

    results = []
    for test_city in sampled_cities:
        print(f"  Testing on {test_city}")
        train_df = df[df['city'] != test_city].copy()
        test_df = df[df['city'] == test_city].copy()

        X_train = train_df[feature_cols].values
        y_train = train_df[['temp', 'hum']].values
        X_test = test_df[feature_cols].values
        y_test = test_df[['temp', 'hum']].values

        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2_temp, var_temp = safe_r2_score(y_test[:, 0], y_pred[:, 0])
        rmse_temp = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
        mae_temp = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
        r2_hum, var_hum = safe_r2_score(y_test[:, 1], y_pred[:, 1])
        rmse_hum = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
        mae_hum = mean_absolute_error(y_test[:, 1], y_pred[:, 1])

        # [DIAGNOSTIC] Staleness check computed on THIS EXACT 62-row test
        # window (not the whole city), to directly explain a per-city
        # R^2 blow-up: a low unique_ratio / long max_run here means the
        # temperature values in this specific held-out window barely
        # change, which collapses Var(y_true) and destabilizes R^2 even
        # though RMSE/MAE remain informative.
        temp_series_test = pd.Series(y_test[:, 0])
        temp_unique_ratio = temp_series_test.nunique() / max(len(temp_series_test), 1)
        temp_max_run = int((temp_series_test != temp_series_test.shift())
                            .cumsum().pipe(lambda g: temp_series_test.groupby(g).transform('size')).max())

        results.append({
            'city': test_city,
            'n_train_cities': train_df['city'].nunique(),
            'n_test_rows': len(test_df),
            'temp_r2': r2_temp, 'temp_rmse': rmse_temp, 'temp_mae': mae_temp,
            'hum_r2': r2_hum, 'hum_rmse': rmse_hum, 'hum_mae': mae_hum,
            'temp_variance': var_temp, 'hum_variance': var_hum,
            'temp_r2_undefined': np.isnan(r2_temp), 'hum_r2_undefined': np.isnan(r2_hum),
            'temp_test_unique_ratio': temp_unique_ratio, 'temp_test_max_run': temp_max_run
        })

    results_df = pd.DataFrame(results)

    # [SUMMARY] Report honestly rather than let a mean() get wrecked by
    # undefined/extreme values -- report how many are undefined, and the
    # median (robust to outliers) for the rest.
    n_undefined = int(results_df['temp_r2_undefined'].sum())
    defined = results_df[~results_df['temp_r2_undefined']]
    print(f"\nLOCO temp_r2: {n_undefined}/{len(results_df)} cities undefined "
          f"(near-zero target variance in that city's test window).")
    if len(defined) > 0:
        print(f"  Among the {len(defined)} defined cities: median temp_r2="
              f"{defined['temp_r2'].median():.4f}, mean temp_r2={defined['temp_r2'].mean():.4f}")
    low_variety = (results_df['temp_test_unique_ratio'] < 0.3).sum()
    print(f"  {low_variety}/{len(results_df)} cities have <30% unique temperature "
          f"values within their own LOCO test window (staleness fingerprint) -- "
          f"see temp_test_unique_ratio / temp_test_max_run columns in the saved CSV.")
    print(f"  Median temp_rmse={results_df['temp_rmse'].median():.4f}, "
          f"median temp_mae={results_df['temp_mae'].median():.4f} (unaffected by the R^2 issue).")

    return results_df


# ============================================================================
# 11. SHAP ANALYSIS (unchanged)
# ============================================================================

def shap_analysis(model, X_sample, feature_names):
    # model is MultiOutputRegressor -> .estimators_[0] is the fitted
    # TimeSeriesStackingRegressor for the "temp" target.
    stacking_reg = model.estimators_[0]
    xgb_model = dict(stacking_reg.fitted_base_estimators_)['xgb']
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (Stacked Ensemble - Temperature)")
    plt.tight_layout()
    plt.savefig(out_path('shap_summary.png'), dpi=150)
    plt.close()
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.DataFrame({'feature': feature_names, 'importance': mean_abs})
    imp = imp.sort_values('importance', ascending=False).head(15)
    plt.figure(figsize=(10, 5))
    plt.barh(imp['feature'], imp['importance'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 15 Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path('shap_bar.png'), dpi=150)
    plt.close()
    imp.to_csv(out_path('shap_importance.csv'), index=False)
    return imp


# ============================================================================
# 12. RESIDUAL DIAGNOSTICS (unchanged)
# ============================================================================

def plot_residuals(y_true, y_pred, target_name):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Residual Diagnostics: {target_name}')
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted'); axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution')
    autocorrelation_plot(pd.Series(residuals), ax=axes[1, 1])
    axes[1, 1].set_title('Autocorrelation')
    axes[1, 1].set_xlim([0, 40])
    plt.tight_layout()
    plt.savefig(out_path(f'residuals_{target_name}.png'), dpi=150)
    plt.close()


# ============================================================================
# 13. MULTI-STEP FORECASTING  [REVIEWER #10 - consistency check added;
#     MINOR - RMSE/R^2 per horizon added]
# ============================================================================

def recursive_forecast(model, initial_features, n_steps=48, feature_names=None):
    preds = []
    curr = initial_features.copy()
    if feature_names and 'temp_lag1' in feature_names:
        temp_lag_idx = feature_names.index('temp_lag1')
        hum_lag_idx = feature_names.index('hum_lag1')
    else:
        temp_lag_idx = hum_lag_idx = None
    for _ in range(n_steps):
        next_pred = model.predict(curr.reshape(1, -1))[0]
        preds.append(next_pred)
        if temp_lag_idx is not None:
            curr[temp_lag_idx] = next_pred[0]
            curr[hum_lag_idx] = next_pred[1]
        else:
            break
    return np.array(preds)


def evaluate_multi_step(model, X_test, y_test, horizons=(1, 6, 12, 24, 48),
                         direct_stack_metrics=None, tolerance=0.02):
    """
    [MINOR] Now reports RMSE and R^2 alongside MAE for every horizon, not
    just MAE, so the degradation pattern is fully interpretable.

    [REVIEWER #10] If `direct_stack_metrics` (the Table-2-style metrics from
    compute_metrics() on the direct single-step model) is supplied, the
    1-hour-horizon recursive MAE computed here is compared against it. At
    h=1 the recursive forecast's first step uses the same input features as
    the direct model, so these two numbers SHOULD match closely -- if they
    don't, that is exactly the Table 2 vs Table 4 discrepancy the reviewer
    flagged, and this function will print an explicit warning rather than
    let the mismatch reach the manuscript silently.
    """
    feat_names = X_test.columns.tolist()
    results = {'temp': {}, 'hum': {}}
    results_rmse = {'temp': {}, 'hum': {}}
    results_r2 = {'temp': {}, 'hum': {}}

    for h in horizons:
        temp_true_all, temp_pred_all = [], []
        hum_true_all, hum_pred_all = [], []
        max_start = len(X_test) - h
        if max_start <= 0:
            continue
        for i in range(max_start):
            init = X_test.iloc[i].values
            true_temp = y_test.iloc[i + 1:i + h + 1]['temp'].values
            true_hum = y_test.iloc[i + 1:i + h + 1]['hum'].values
            pred = recursive_forecast(model, init, h, feat_names)
            if len(pred) != h:
                continue
            temp_true_all.append(true_temp[-1]); temp_pred_all.append(pred[-1, 0])
            hum_true_all.append(true_hum[-1]); hum_pred_all.append(pred[-1, 1])

        if temp_true_all:
            temp_true_all = np.array(temp_true_all); temp_pred_all = np.array(temp_pred_all)
            hum_true_all = np.array(hum_true_all); hum_pred_all = np.array(hum_pred_all)

            results['temp'][h] = mean_absolute_error(temp_true_all, temp_pred_all)
            results['hum'][h] = mean_absolute_error(hum_true_all, hum_pred_all)
            results_rmse['temp'][h] = np.sqrt(mean_squared_error(temp_true_all, temp_pred_all))
            results_rmse['hum'][h] = np.sqrt(mean_squared_error(hum_true_all, hum_pred_all))
            results_r2['temp'][h] = r2_score(temp_true_all, temp_pred_all)
            results_r2['hum'][h] = r2_score(hum_true_all, hum_pred_all)

    print("\nMulti-step metrics (Stacked Ensemble, recursive forecast):")
    for h in sorted(results['temp'].keys()):
        print(f"  Horizon {h}h | Temp: MAE={results['temp'][h]:.4f} "
              f"RMSE={results_rmse['temp'][h]:.4f} R2={results_r2['temp'][h]:.4f} | "
              f"Hum: MAE={results['hum'][h]:.4f} RMSE={results_rmse['hum'][h]:.4f} "
              f"R2={results_r2['hum'][h]:.4f}")

    # [REVIEWER #10] Cross-check horizon-1 recursive MAE against the direct model
    if direct_stack_metrics is not None and 1 in results['temp']:
        for var in ('temp', 'hum'):
            direct_mae = direct_stack_metrics[f'{var}_mae']
            recursive_mae = results[var][1]
            if direct_mae > 0 and abs(recursive_mae - direct_mae) / direct_mae > tolerance:
                print(f"  [WARNING] Table-2-vs-Table-4 style mismatch detected for {var}: "
                      f"direct 1-step MAE={direct_mae:.4f} vs recursive 1-step MAE={recursive_mae:.4f}. "
                      f"Investigate before reporting both numbers in the manuscript.")
            else:
                print(f"  [OK] {var}: direct 1-step MAE and recursive 1-step MAE agree within tolerance "
                      f"({direct_mae:.4f} vs {recursive_mae:.4f}).")

    return {'mae': results, 'rmse': results_rmse, 'r2': results_r2}


# ============================================================================
# 14. PER-CITY MULTI-STEP FIGURE  [REVIEWER #11 - NEW: enforces STUDY_CITIES]
# ============================================================================

def plot_recursive_forecast_for_city(model, df_raw, city, feature_cols,
                                      le_region, le_cond, train_ratio=0.75,
                                      n_steps=48, study_cities=STUDY_CITIES):
    """
    [REVIEWER #11] Generates the "Figure 8 style" recursive multi-step
    forecast plot for a single named city, RAISING AN ERROR if that city is
    not in STUDY_CITIES. This makes it structurally impossible to
    accidentally ship a figure for a city outside the defined study scope.
    """
    _check_city_allowed(city, study_cities)

    group = df_raw[df_raw['city'] == city].sort_values('datetime').reset_index(drop=True)
    if group.empty:
        raise ValueError(f"City '{city}' not found in the raw dataset.")

    group = group.copy()
    group['temp_lag1'] = group['temp'].shift(1)
    group['hum_lag1'] = group['hum'].shift(1)
    group['hour_sin'] = np.sin(2 * np.pi * group['datetime'].dt.hour / 24)
    group['hour_cos'] = np.cos(2 * np.pi * group['datetime'].dt.hour / 24)
    group['month_sin'] = np.sin(2 * np.pi * group['datetime'].dt.month / 12)
    group['month_cos'] = np.cos(2 * np.pi * group['datetime'].dt.month / 12)
    group['region_enc'] = le_region.transform(group['region'].astype(str))
    group['condition_enc'] = le_cond.transform(group['condition_text'].astype(str))
    group = group.dropna().reset_index(drop=True)

    n_train = int(len(group) * train_ratio)
    if n_train >= len(group) - n_steps:
        raise ValueError(f"Not enough test rows for city '{city}' to forecast {n_steps} steps.")

    init_row = group.iloc[n_train][feature_cols].values.astype(float)
    preds = recursive_forecast(model, init_row, n_steps=n_steps, feature_names=feature_cols)

    true_temp = group['temp'].iloc[n_train + 1: n_train + 1 + n_steps].values
    true_hum = group['hum'].iloc[n_train + 1: n_train + 1 + n_steps].values

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{n_steps}-Hour Recursive Forecast - {city}")
    axes[0].plot(true_temp, label='Observed', marker='o', ms=3)
    axes[0].plot(preds[:, 0], label='Forecast', marker='x', ms=3)
    axes[0].set_ylabel('Temperature (deg C)'); axes[0].legend()
    axes[1].plot(true_hum, label='Observed', marker='o', ms=3)
    axes[1].plot(preds[:, 1], label='Forecast', marker='x', ms=3)
    axes[1].set_ylabel('Humidity (%)'); axes[1].set_xlabel('Hours ahead'); axes[1].legend()
    plt.tight_layout()
    safe_name = city.replace(' ', '_').replace('/', '_')
    plt.savefig(out_path(f'multistep_forecast_{safe_name}.png'), dpi=150)
    plt.close()
    print(f"Saved multistep_forecast_{safe_name}.png")
# ============================================================================
# 16. EXPLORATORY DATA ANALYSIS VISUALS
#     [MERGE] Adapted from the uploaded eda_lstm_arima.py / eda_visuals*.py
#     scripts. Unlike those scripts, everything here operates on the
#     dataframe already loaded within THIS run (no external file re-loading,
#     no risk of the synthetic-data fallback flagged during methodology
#     review). Figures are written via the existing out_path() helper so
#     they land in the same OUTPUT_DIR as every other artifact.
# ============================================================================

EDA_COLOR_PRIMARY = '#1F77B4'
EDA_COLOR_SECONDARY = '#FF7F0E'
EDA_COLOR_SUCCESS = '#2CA02C'
EDA_COLOR_ACCENT = '#D62728'
EDA_COLOR_PALETTE = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02', '#A6761D', '#666666']


def plot_eda_target_distributions(df):
    """Density distributions of temperature and humidity with mean/median markers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df['temp'], kde=True, ax=axes[0], color=EDA_COLOR_PRIMARY, bins=45, stat="density", alpha=0.6)
    axes[0].axvline(df['temp'].mean(), color=EDA_COLOR_ACCENT, linestyle='--', linewidth=2,
                     label=f"Mean: {df['temp'].mean():.2f}\u00b0C")
    axes[0].axvline(df['temp'].median(), color=EDA_COLOR_SUCCESS, linestyle=':', linewidth=2,
                     label=f"Median: {df['temp'].median():.2f}\u00b0C")
    axes[0].set_title("Near-Surface Temperature Density Distribution", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Temperature (\u00b0C)"); axes[0].set_ylabel("Density"); axes[0].legend()

    sns.histplot(df['hum'], kde=True, ax=axes[1], color=EDA_COLOR_SUCCESS, bins=45, stat="density", alpha=0.6)
    axes[1].axvline(df['hum'].mean(), color=EDA_COLOR_ACCENT, linestyle='--', linewidth=2,
                     label=f"Mean: {df['hum'].mean():.2f}%")
    axes[1].axvline(df['hum'].median(), color=EDA_COLOR_PRIMARY, linestyle=':', linewidth=2,
                     label=f"Median: {df['hum'].median():.2f}%")
    axes[1].set_title("Relative Humidity Density Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Relative Humidity (%)"); axes[1].set_ylabel("Density"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path('eda_target_distributions_enhanced.png'), dpi=300)
    plt.close()


def plot_eda_citywise_distributions(df, study_cities=STUDY_CITIES):
    """Boxplots of temperature/humidity across the defined study cities."""
    df_study = df[df['city'].isin(study_cities)]
    if df_study.empty:
        print("  [SKIP] eda_citywise_distributions: none of STUDY_CITIES found in this dataset.")
        return
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    sns.boxplot(data=df_study, x='city', y='temp', ax=axes[0], hue='city', palette='YlOrRd', legend=False)
    axes[0].set_title("Temperature Variations Across Target Study Locations", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Temperature (\u00b0C)"); axes[0].set_xlabel("")

    sns.boxplot(data=df_study, x='city', y='hum', ax=axes[1], hue='city', palette='GnBu', legend=False)
    axes[1].set_title("Relative Humidity Variations Across Target Study Locations", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Relative Humidity (%)"); axes[1].set_xlabel("Location")

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_path('eda_citywise_distributions.png'), dpi=300)
    plt.close()


def plot_eda_diurnal_profiles(df, study_cities=STUDY_CITIES):
    """24-hour mean diurnal cycles of temperature/humidity per study city."""
    df_study = df[df['city'].isin(study_cities)].copy()
    if df_study.empty:
        print("  [SKIP] eda_diurnal_profiles: none of STUDY_CITIES found in this dataset.")
        return
    df_study['hour'] = df_study['datetime'].dt.hour
    diurnal = df_study.groupby(['city', 'hour'])[['temp', 'hum']].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.lineplot(data=diurnal, x='hour', y='temp', hue='city', ax=axes[0], marker='o',
                 palette=EDA_COLOR_PALETTE, linewidth=2)
    axes[0].set_title("Diurnal Temperature Progression (Hourly Means)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Hour of Day"); axes[0].set_ylabel("Mean Temperature (\u00b0C)")
    axes[0].set_xticks(range(0, 24, 2)); axes[0].legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')

    sns.lineplot(data=diurnal, x='hour', y='hum', hue='city', ax=axes[1], marker='s',
                 palette=EDA_COLOR_PALETTE, linewidth=2)
    axes[1].set_title("Diurnal Humidity Progression (Hourly Means)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Hour of Day"); axes[1].set_ylabel("Mean Relative Humidity (%)")
    axes[1].set_xticks(range(0, 24, 2)); axes[1].legend(title="City", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path('eda_diurnal_profiles.png'), dpi=300)
    plt.close()


def plot_eda_timeseries_sample(df, n_cities=6):
    """Raw time-series traces for a sample of cities (orientation/QC plot)."""
    cities = [c for c in STUDY_CITIES if c in df['city'].unique()][:n_cities]
    if len(cities) < n_cities:
        remaining = [c for c in df['city'].unique() if c not in cities]
        cities += remaining[:max(0, n_cities - len(cities))]
    if not cities:
        print("  [SKIP] eda_timeseries: no cities available to sample.")
        return

    fig, axes = plt.subplots(len(cities), 1, figsize=(14, 2.5 * len(cities)), sharex=False)
    axes = np.atleast_1d(axes)
    fig.suptitle('Time Series for Selected Cities', fontsize=14)
    for i, city in enumerate(cities):
        g = df[df['city'] == city].sort_values('datetime')
        axes[i].plot(g['datetime'], g['temp'], color=EDA_COLOR_PRIMARY, linewidth=0.8, label='Temp (\u00b0C)')
        ax2 = axes[i].twinx()
        ax2.plot(g['datetime'], g['hum'], color=EDA_COLOR_SUCCESS, linewidth=0.8, alpha=0.6, label='Humidity (%)')
        axes[i].set_title(city)
        axes[i].set_ylabel('Temp (\u00b0C)', color=EDA_COLOR_PRIMARY)
        ax2.set_ylabel('Humidity (%)', color=EDA_COLOR_SUCCESS)

    plt.tight_layout()
    plt.savefig(out_path('eda_timeseries.png'), dpi=300)
    plt.close()


def plot_eda_feature_correlation(df):
    """Correlation heatmap among targets and engineered features."""
    df_corr = df.copy()
    df_corr['hour'] = df_corr['datetime'].dt.hour
    df_corr['temp_lag1'] = df_corr.groupby('city')['temp'].shift(1)
    df_corr['hum_lag1'] = df_corr.groupby('city')['hum'].shift(1)
    df_corr['hour_sin'] = np.sin(2 * np.pi * df_corr['hour'] / 24)
    df_corr['month_sin'] = np.sin(2 * np.pi * df_corr['datetime'].dt.month / 12)

    corr_cols = ['temp', 'hum', 'temp_lag1', 'hum_lag1', 'hour_sin', 'month_sin']
    corr_matrix = df_corr[corr_cols].dropna().corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={'shrink': .8})
    plt.title("Correlation Heatmap: Meteorological Target & Engineered Features", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path('eda_feature_correlation.png'), dpi=300)
    plt.close()


def run_all_eda_visuals(df):
    """Generates all five EDA figures from the already-loaded raw dataframe."""
    print("\n=== Generating EDA visualizations ===")
    plot_eda_target_distributions(df)
    plot_eda_citywise_distributions(df)
    plot_eda_diurnal_profiles(df)
    plot_eda_timeseries_sample(df)
    plot_eda_feature_correlation(df)
    print("EDA visualizations saved: eda_target_distributions_enhanced.png, "
          "eda_citywise_distributions.png, eda_diurnal_profiles.png, "
          "eda_timeseries.png, eda_feature_correlation.png")


# ============================================================================
# 17. MULTI-MODEL COMPARISON VISUALS
#     [MERGE] Adapted from the uploaded eda_lstm_arima.py / eda_visuals*.py
#     scripts. Every prediction/metric consumed here comes from a model
#     already trained earlier in THIS run (Sections 2-9 above) -- there is
#     no file-loading fallback and therefore no risk of the synthetic-data
#     or in-sample-refit shortcuts flagged during methodology review.
# ============================================================================

def compute_persistence_predictions(y_test):
    """
    Same definition as persistence_baseline() (Section 2) but additionally
    returns the prediction array itself, needed for the scatter comparison
    plot. Kept as a separate function so persistence_baseline() itself is
    untouched.
    """
    if len(y_test) < 2:
        return None, None
    temp_pred = y_test['temp'].shift(1).values[1:]
    hum_pred = y_test['hum'].shift(1).values[1:]
    y_pred = np.column_stack([temp_pred, hum_pred])
    metrics = persistence_baseline(y_test)
    return y_pred, metrics


def plot_comparative_metrics(metrics_dict):
    """Grouped bar chart of R^2/MAE/RMSE (original units) across all models."""
    df_m = pd.DataFrame(metrics_dict).T
    models = df_m.index.tolist()
    if not models:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Comparative Framework Performance (Unscaled Physical Units)', fontsize=14, fontweight='bold')

    palette_temp = sns.color_palette("Blues_r", len(models))
    palette_hum = sns.color_palette("Greens_r", len(models))

    axes[0, 0].bar(models, df_m['temp_r2'], color=palette_temp, edgecolor='black', alpha=0.85)
    axes[0, 0].set_title('Temperature R\u00b2 Score (Higher is Better)', fontweight='bold')
    axes[0, 1].bar(models, df_m['temp_mae'], color=palette_temp, edgecolor='black', alpha=0.85)
    axes[0, 1].set_title('Temperature MAE (\u00b0C) (Lower is Better)', fontweight='bold')
    axes[0, 2].bar(models, df_m['temp_rmse'], color=palette_temp, edgecolor='black', alpha=0.85)
    axes[0, 2].set_title('Temperature RMSE (\u00b0C) (Lower is Better)', fontweight='bold')

    axes[1, 0].bar(models, df_m['hum_r2'], color=palette_hum, edgecolor='black', alpha=0.85)
    axes[1, 0].set_title('Humidity R\u00b2 Score (Higher is Better)', fontweight='bold')
    axes[1, 1].bar(models, df_m['hum_mae'], color=palette_hum, edgecolor='black', alpha=0.85)
    axes[1, 1].set_title('Humidity MAE (%) (Lower is Better)', fontweight='bold')
    axes[1, 2].bar(models, df_m['hum_rmse'], color=palette_hum, edgecolor='black', alpha=0.85)
    axes[1, 2].set_title('Humidity RMSE (%) (Lower is Better)', fontweight='bold')

    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path('model_metrics_comparison_unscaled.png'), dpi=300)
    plt.close()


def plot_multi_model_scatters(y_test, preds_dict):
    """
    Observed-vs-predicted scatter per model/target. Alignment rule: every
    prediction array here is either the same length as y_test (Stacked
    Ensemble) or shorter because it drops leading rows (Persistence drops 1;
    LSTM drops the first n_steps for its input window) -- so aligning to the
    TRAILING len(pred) rows of y_test is correct for every model uniformly,
    with no per-model special-casing required.
    """
    models = list(preds_dict.keys())
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, n_models, figsize=(4.5 * n_models, 8))
    fig.suptitle("Observed vs. Predicted Regression Diagnostics Across Frameworks", fontsize=14, fontweight='bold')
    if n_models == 1:
        axes = axes.reshape(2, 1)

    for i, m_name in enumerate(models):
        pred = preds_dict[m_name]
        n = len(pred)
        y_temp = y_test['temp'].iloc[-n:].values
        y_hum = y_test['hum'].iloc[-n:].values
        p_temp, p_hum = pred[:, 0], pred[:, 1]

        axes[0, i].scatter(y_temp, p_temp, alpha=0.3, s=8, color=EDA_COLOR_PRIMARY)
        lims = [min(y_temp.min(), p_temp.min()), max(y_temp.max(), p_temp.max())]
        axes[0, i].plot(lims, lims, 'r--', linewidth=1.5)
        axes[0, i].set_title(f"{m_name}\nTemp R\u00b2 = {r2_score(y_temp, p_temp):.4f}", fontweight='bold')
        axes[0, i].set_xlabel("Observed Temp (\u00b0C)"); axes[0, i].set_ylabel("Predicted Temp (\u00b0C)")

        axes[1, i].scatter(y_hum, p_hum, alpha=0.3, s=8, color=EDA_COLOR_SUCCESS)
        lims_h = [min(y_hum.min(), p_hum.min()), max(y_hum.max(), p_hum.max())]
        axes[1, i].plot(lims_h, lims_h, 'r--', linewidth=1.5)
        axes[1, i].set_title(f"Hum R\u00b2 = {r2_score(y_hum, p_hum):.4f}", fontweight='bold')
        axes[1, i].set_xlabel("Observed Humidity (%)"); axes[1, i].set_ylabel("Predicted Humidity (%)")

    plt.tight_layout()
    plt.savefig(out_path('model_scatter_comparisons.png'), dpi=300)
    plt.close()


# ============================================================================
# 18. MAIN PIPELINE
# ============================================================================

def main():
    filepath = '/kaggle/input/datasets/atasidas/indianweatherrepository/IndianWeatherRepository.csv'

    print("=== Loading data ===")
    # [REVIEWER #9] Pass the TRUE download date here if you have it, e.g.
    # download_date="2025-03-15". Left as None to use the file's mtime.
    X_train, X_test, y_train, y_test, le_city, le_region, le_cond = prepare_data(
        filepath, download_date=None
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ---- Diagnose temperature vs humidity variance/staleness BEFORE modeling ----
    # This directly investigates the "R2 = -1e24 for temp but not hum" pattern:
    # if temp has a much lower unique_ratio / much longer max_run than hum,
    # that points to a data staleness/duplication artifact rather than a
    # modeling problem.
    diag_df = load_and_standardize(filepath)
    variance_report = diagnose_target_variance(diag_df)
    variance_report.to_csv(out_path('target_variance_diagnostic.csv'), index=False)
    print("Saved target_variance_diagnostic.csv -- inspect this before trusting any per-city temp R^2.")

    # ---- EDA visuals (Section 16) -- reuses diag_df, no extra data loading ----
    run_all_eda_visuals(diag_df)

    # ---- Persistence [REVIEWER #3] ----
    persistence_metrics = persistence_baseline(y_test)
    print("\nPersistence baseline (yhat_t+1 = y_t):")
    print(f"Temp: R2={persistence_metrics['temp_r2']:.4f}, RMSE={persistence_metrics['temp_rmse']:.4f}, MAE={persistence_metrics['temp_mae']:.4f}")
    print(f"Hum:  R2={persistence_metrics['hum_r2']:.4f}, RMSE={persistence_metrics['hum_rmse']:.4f}, MAE={persistence_metrics['hum_mae']:.4f}")

    # ---- Stacked ensemble [REVIEWER #4: now TimeSeriesSplit internally] ----
    print("\nTraining stacked ensemble (internal CV = TimeSeriesSplit(5))...")
    model = get_stacked_model()
    model.fit(X_train, y_train)
    y_pred_stack = model.predict(X_test)
    for target_name, est in zip(['temp', 'hum'], model.estimators_):
        print(f"  Meta-learner ({target_name}) trained on {est.n_meta_train_} "
              f"out-of-fold rows (out of {len(X_train)} training rows).")
    stack_metrics = compute_metrics(y_test, y_pred_stack)
    print("Stacked ensemble test performance:")
    print(f"Temp: R2={stack_metrics['temp_r2']:.4f}, RMSE={stack_metrics['temp_rmse']:.4f}, MAE={stack_metrics['temp_mae']:.4f}")
    print(f"Hum:  R2={stack_metrics['hum_r2']:.4f}, RMSE={stack_metrics['hum_rmse']:.4f}, MAE={stack_metrics['hum_mae']:.4f}")

    # ---- Walk-forward [REVIEWER #1] ----
    print("\nWalk-forward (rolling-origin) validation, 5 folds:")
    cv_results = walk_forward_validation(model, X_train, y_train, n_splits=5)
    if not cv_results.empty:
        cv_mean = cv_results.mean(numeric_only=True)
        cv_std = cv_results.std(numeric_only=True)
        print(f"Temp R2: {cv_mean['temp_r2']:.4f} +/- {cv_std['temp_r2']:.4f}")
        print(f"Hum  R2: {cv_mean['hum_r2']:.4f} +/- {cv_std['hum_r2']:.4f}")
        print(f"Temp RMSE: {cv_mean['temp_rmse']:.4f} +/- {cv_std['temp_rmse']:.4f}")
        print(f"Hum  RMSE: {cv_mean['hum_rmse']:.4f} +/- {cv_std['hum_rmse']:.4f}")
        cv_results.to_csv(out_path('walk_forward_results.csv'), index=False)
        print("Saved walk_forward_results.csv -- put this table in the manuscript.")

    # ---- ARIMA: ALL CITIES, per-city order selection [REVIEWER #6] ----
    print("\nARIMA rolling one-step-ahead on ALL cities (per-city AIC order selection):")
    arima_temp_overall, arima_temp_cities = arima_rolling_all_cities(
        filepath, target='temp', min_test_samples=5, select_order=True
    )
    arima_hum_overall, arima_hum_cities = arima_rolling_all_cities(
        filepath, target='hum', min_test_samples=5, select_order=True
    )
    print(f"ARIMA Temp: R2={arima_temp_overall['r2']:.4f}, RMSE={arima_temp_overall['rmse']:.4f}, MAE={arima_temp_overall['mae']:.4f}")
    print(f"ARIMA Hum:  R2={arima_hum_overall['r2']:.4f}, RMSE={arima_hum_overall['rmse']:.4f}, MAE={arima_hum_overall['mae']:.4f}")
    if not arima_temp_cities.empty:
        arima_temp_cities.to_csv(out_path('arima_temp_per_city.csv'), index=False)
    if not arima_hum_cities.empty:
        arima_hum_cities.to_csv(out_path('arima_hum_per_city.csv'), index=False)

    # ---- LSTM ----
    print("\nLSTM baseline (full test set, same scope as stacked model):")
    (lstm_pred_temp, lstm_pred_hum, lstm_r2_temp, lstm_r2_hum,
     lstm_rmse_temp, lstm_rmse_hum, lstm_mae_temp, lstm_mae_hum) = run_lstm_full(X_train, X_test, y_train, y_test)
    if len(lstm_pred_temp) > 0:
        print(f"Temp: R2={lstm_r2_temp:.4f}, RMSE={lstm_rmse_temp:.4f}, MAE={lstm_mae_temp:.4f}")
        print(f"Hum:  R2={lstm_r2_hum:.4f}, RMSE={lstm_rmse_hum:.4f}, MAE={lstm_mae_hum:.4f}")

    # ---- Multi-model comparison visuals (Section 17) ----
    # Every input here was computed earlier in THIS run (persistence_metrics,
    # arima_temp_overall/arima_hum_overall, the LSTM results just above, and
    # stack_metrics/y_pred_stack from the stacked-ensemble section) -- no
    # external file loading, so none of the fallback risks flagged during
    # methodology review apply here.
    print("\nGenerating multi-model comparison visuals...")
    metrics_dict = {'Persistence': persistence_metrics}
    pers_pred, _ = compute_persistence_predictions(y_test)
    preds_dict = {}
    if pers_pred is not None:
        preds_dict['Persistence'] = pers_pred

    # ARIMA is evaluated per-city (a separate model per held-out city), so it
    # has no single row-aligned prediction series comparable to the pooled
    # y_test used by the other three models -- it is included in the bar
    # chart (aggregate metrics) but intentionally excluded from the scatter
    # comparison, which requires row-aligned predictions.
    metrics_dict['ARIMA'] = {
        'temp_r2': arima_temp_overall['r2'], 'temp_rmse': arima_temp_overall['rmse'], 'temp_mae': arima_temp_overall['mae'],
        'hum_r2': arima_hum_overall['r2'], 'hum_rmse': arima_hum_overall['rmse'], 'hum_mae': arima_hum_overall['mae'],
    }

    if len(lstm_pred_temp) > 0:
        metrics_dict['LSTM'] = {
            'temp_r2': lstm_r2_temp, 'temp_rmse': lstm_rmse_temp, 'temp_mae': lstm_mae_temp,
            'hum_r2': lstm_r2_hum, 'hum_rmse': lstm_rmse_hum, 'hum_mae': lstm_mae_hum,
        }
        preds_dict['LSTM'] = np.column_stack([lstm_pred_temp, lstm_pred_hum])

    metrics_dict['Stacked Ensemble'] = stack_metrics
    preds_dict['Stacked Ensemble'] = y_pred_stack

    plot_comparative_metrics(metrics_dict)
    plot_multi_model_scatters(y_test, preds_dict)
    print("Saved model_metrics_comparison_unscaled.png and model_scatter_comparisons.png "
          f"(models compared: {list(metrics_dict.keys())}; ARIMA shown in bar chart only -- see note above).")

    # ---- LOCO: ALL cities by default [REVIEWER #2] ----
    print("\nLeave-one-city-out validation (ALL cities by default):")
    loco_results = leave_one_city_out(filepath, get_stacked_model, n_cities=None, random_state=42)
    loco_results.to_csv(out_path('loco_results.csv'), index=False)
    print("Saved loco_results.csv (full city-wise breakdown) -- use this for the LOCO table.")
    print("LOCO summary:")
    print(loco_results[['temp_r2', 'hum_r2']].describe())

    # ---- SHAP ----
    try:
        print("\nSHAP analysis:")
        X_sample = X_test.sample(min(100, len(X_test)), random_state=42)
        shap_imp = shap_analysis(model, X_sample, X_test.columns.tolist())
        print("SHAP plots saved.")
    except Exception as e:
        print(f"SHAP skipped due to: {e}")

    # ---- Residuals ----
    print("\nResidual diagnostics:")
    plot_residuals(y_test['temp'], y_pred_stack[:, 0], 'temperature')
    plot_residuals(y_test['hum'], y_pred_stack[:, 1], 'humidity')
    print("Residual plots saved.")

    # ---- Multi-step [REVIEWER #10, MINOR] ----
    print("\nMulti-step recursive forecasting:")
    multi_results = evaluate_multi_step(
        model, X_test, y_test, horizons=(1, 6, 12, 24, 48),
        direct_stack_metrics=stack_metrics
    )
    pd.DataFrame(multi_results['mae']).to_csv(out_path('multi_step_mae.csv'))
    pd.DataFrame(multi_results['rmse']).to_csv(out_path('multi_step_rmse.csv'))
    pd.DataFrame(multi_results['r2']).to_csv(out_path('multi_step_r2.csv'))

    # ---- Per-city multi-step figure [REVIEWER #11] ----
    if STUDY_CITIES:
        try:
            df_raw = load_raw_for_arima_loco(filepath)
            example_city = STUDY_CITIES[0]
            plot_recursive_forecast_for_city(
                model, df_raw, example_city,
                feature_cols=['temp_lag1', 'hum_lag1', 'hour_sin', 'hour_cos',
                              'month_sin', 'month_cos', 'city_enc', 'region_enc', 'condition_enc'],
                le_region=le_region, le_cond=le_cond, n_steps=48
            )
        except Exception as e:
            print(f"Per-city multistep figure skipped: {e}")
    else:
        print("\n[NOTE] STUDY_CITIES is empty -- fill it in with your 8 Table 1 cities "
              "before generating any per-city multistep figure (this is the guardrail "
              "for the Figure 8 / Adilabad issue).")

    # ---- Persistence improvement ----
    imp_temp = (persistence_metrics['temp_mae'] - stack_metrics['temp_mae']) / persistence_metrics['temp_mae'] * 100
    imp_hum = (persistence_metrics['hum_mae'] - stack_metrics['hum_mae']) / persistence_metrics['hum_mae'] * 100
    print(f"\nImprovement over persistence: {imp_temp:.1f}% (temp), {imp_hum:.1f}% (hum)")

    # ---- Save model ----
    joblib.dump(model, out_path('stacked_model_final.pkl'))
    print("\nAll done. Model, results tables, and plots saved.")


if __name__ == "__main__":
    main()