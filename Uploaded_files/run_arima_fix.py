"""Runs the patched script's own functions for the steps that matter to the reviewer's ARIMA comment:
STEP 1-2 (load/prepare), STEP 3 (persistence, per-city), STEP 4 (stacked ensemble), STEP 6 (ARIMA on the same rows).
Everything else in main() (walk-forward, LSTM, LOCO, SHAP, multi-step) is untouched and not needed for this comment."""
import sys, json, importlib.util, warnings
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")
P, XLSX, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("v3fix", P); o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o); o.OUTPUT_DIR = OUT
df_raw = o.load_and_validate(XLSX)
df_full, X_train, X_test, y_train, y_test, *_ = o.prepare_data(df_raw)

pers_df = o.persistence_predictions_by_city(df_full, y_test); pers_df.to_csv(f"{OUT}/persistence_predictions.csv", index=False)
pers_m = o.pooled_persistence_metrics(pers_df); print("PERSISTENCE (per-city, pooled):", json.dumps(pers_m), flush=True)

model = o.get_stacked_model(n_estimators=200).fit(X_train, y_train); yp = model.predict(X_test)
stack_m = o.compute_metrics(y_test, yp); print("STACK (pooled):", json.dumps(stack_m, default=float), flush=True)
sp = df_full.loc[X_test.index, ["city", "datetime"]].copy()
sp["temp_true"], sp["hum_true"] = y_test["temp"].values, y_test["hum"].values; sp["temp_pred"], sp["hum_pred"] = yp[:, 0], yp[:, 1]
sp.to_csv(f"{OUT}/stacked_ensemble_predictions.csv", index=False)

at = o.arima_rolling_predictions(df_full, y_test, target="temp"); ah = o.arima_rolling_predictions(df_full, y_test, target="hum")
at.to_csv(f"{OUT}/arima_temp_predictions.csv", index=False); ah.to_csv(f"{OUT}/arima_hum_predictions.csv", index=False)
am_t, am_h = o.arima_pooled_metrics(at), o.arima_pooled_metrics(ah); print("ARIMA temp pooled:", am_t, "\nARIMA hum pooled:", am_h, flush=True)
o.arima_per_city_metrics(at).to_csv(f"{OUT}/arima_temp_per_city.csv", index=False); o.arima_per_city_metrics(ah).to_csv(f"{OUT}/arima_hum_per_city.csv", index=False)
json.dump({"temp": am_t, "hum": am_h}, open(f"{OUT}/arima_pooled_metrics.json", "w"), indent=2)

# ---- verification of the reviewer's four points ----
rep = {}
test_keys = set(zip(df_full.loc[y_test.index, "city"], df_full.loc[y_test.index, "datetime"]))
for name, a in (("temp", at), ("hum", ah)):
    ak = list(zip(a["city"], a["datetime"]))
    m = a.merge(df_full.loc[y_test.index, ["city", "datetime", name]].rename(columns={name: "df_full_value"}), on=["city", "datetime"], how="left")
    rep[name] = {"arima_rows": len(a), "y_test_rows": len(y_test), "same_row_set_as_y_test": set(ak) == test_keys,
                 "duplicate_city_datetime": int(a.duplicated(["city", "datetime"]).sum()),
                 "n_nan_predictions": int(a["y_pred"].isna().sum()),
                 "y_true_equals_df_full_value_at_same_city_datetime": bool(np.allclose(m["y_true"], m["df_full_value"], equal_nan=True)),
                 "datetime_monotonic_within_city": bool(a.groupby("city")["datetime"].apply(lambda s: s.is_monotonic_increasing).all()),
                 "y_true_range": [float(a.y_true.min()), float(a.y_true.max())], "y_pred_range": [float(a.y_pred.min()), float(a.y_pred.max())]}
    # all three metrics from the same saved rows (reload from CSV)
    rl = pd.read_csv(f"{OUT}/arima_{name}_predictions.csv").dropna(subset=["y_true", "y_pred"])
    rep[name]["metrics_recomputed_from_saved_csv"] = {"r2": float(r2_score(rl.y_true, rl.y_pred)), "rmse": float(np.sqrt(mean_squared_error(rl.y_true, rl.y_pred))), "mae": float(mean_absolute_error(rl.y_true, rl.y_pred)), "n": len(rl)}
    # one-step alignment: prediction should track y_true at lag 0, not at lag +/-1 (within city, demeaned)
    lc = {}
    for lag in (-1, 0, 1):
        aa, bb = [], []
        for _, g in a.groupby("city"):
            p, y = g.y_pred.values, g.y_true.values
            pp, yy = (p, y) if lag == 0 else ((p[lag:], y[:-lag]) if lag > 0 else (p[:lag], y[-lag:]))
            if len(pp) > 2: aa.append(pp - pp.mean()); bb.append(yy - yy.mean())
        lc[f"corr(pred_t, true_t{-lag:+d})"] = float(np.corrcoef(np.concatenate(aa), np.concatenate(bb))[0, 1])
    rep[name]["within_city_lag_correlation"] = lc

# ---- final comparison: pooled (as asked) + per-location mean (as in the original ARIMA convention), same rows ----
def per_city_mean(df, t, p):
    out = []
    for c, g in df.groupby("city"):
        r2, _ = o.safe_r2_score(g[f"{t}_true"], g[p]); out.append((r2, np.sqrt(mean_squared_error(g[f"{t}_true"], g[p])), mean_absolute_error(g[f"{t}_true"], g[p])))
    x = np.array(out, dtype=float); return {"r2": float(np.nanmean(x[:, 0])), "rmse": float(x[:, 1].mean()), "mae": float(x[:, 2].mean())}
ar = at.rename(columns={"y_true": "temp_true", "y_pred": "temp_pred"})[["city", "datetime", "temp_true", "temp_pred"]].merge(
     ah.rename(columns={"y_true": "hum_true", "y_pred": "hum_pred"})[["city", "datetime", "hum_true", "hum_pred"]], on=["city", "datetime"])
tab = {}
for name, d in (("Persistence (per-city shift)", pers_df), ("ARIMA (per-city, AIC-selected)", ar), ("Stacked Ensemble", sp)):
    r = {}
    for t in ("temp", "hum"):
        dd = d.dropna(subset=[f"{t}_true", f"{t}_pred"])
        r[t] = {"pooled": {"r2": float(r2_score(dd[f"{t}_true"], dd[f"{t}_pred"])), "mae": float(mean_absolute_error(dd[f"{t}_true"], dd[f"{t}_pred"])), "rmse": float(np.sqrt(mean_squared_error(dd[f"{t}_true"], dd[f"{t}_pred"])))},
                "per_location_mean": per_city_mean(dd, t, f"{t}_pred"), "n": len(dd)}
    tab[name] = r
json.dump({"verification": rep, "comparison_same_rows": tab, "original_v3_reported": {"persistence": [0.817, 0.662], "arima": [0.069, -0.061], "stack": [0.933, 0.786]}}, open(f"{OUT}/arima_fix_report.json", "w"), indent=2, default=str)
print("FIX DONE"); print(json.dumps({"verification": rep, "comparison_same_rows": tab}, indent=1, default=str))
