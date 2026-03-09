#!/usr/bin/env python3
"""
AHPI Mid-Market · Facebook Prophet Training
============================================
Trains a Prophet model on the Accra Home Price Index (mid-market composite,
Jan 2010 – Dec 2024) with six macroeconomic regressors, evaluates on a
held-out test set (2023-2024), then forecasts 24 months (2025-2026) under
three economic scenarios.

Outputs
-------
  models/ahpi_prophet_model.json    — serialised production Prophet model
  models/ahpi_scaler.pkl            — fitted StandardScaler (for inference)
  forecasts/ahpi_test_eval.csv      — test-set yhat vs actuals + residuals
  forecasts/ahpi_forecast_bear.csv  — Bear scenario (continued depreciation)
  forecasts/ahpi_forecast_base.csv  — Base scenario (gradual stabilisation)
  forecasts/ahpi_forecast_bull.csv  — Bull scenario (cedi recovery)

Usage
-----
  python ahpi_prophet.py
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "accra_home_price_index.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
FORECASTS_DIR = os.path.join(BASE_DIR, "forecasts")

# ── regressors (ranked by Pearson ρ with AHPI) ────────────────────────────────
REGRESSORS = [
    "exchange_rate_ghs_usd",   # ρ = +0.991 — primary cedi-depreciation driver
    "cpi_index",               # ρ = +0.986 — accumulated inflation
    "urban_pop_pct",           # ρ = +0.871 — structural urbanisation trend
    "broad_money_pct_gdp",     # ρ = +0.813 — monetary expansion
    "gold_price_usd",          # ρ = +0.786 — fiscal / remittance proxy
    "cocoa_price_usd",         # ρ = +0.764 — export revenue proxy
]

# ── model configuration ───────────────────────────────────────────────────────
CHANGEPOINTS = ["2014-01-01", "2022-01-01"]   # 1st cedi crisis; debt crisis
TRAIN_END    = "2022-12-01"                   # last training month (inclusive)
TEST_START   = "2023-01-01"                   # first test month
FORECAST_MONTHS = 24                          # Jan 2025 → Dec 2026

# ── forward economic scenarios for 2025-2026 ──────────────────────────────────
# Keys map to REGRESSORS that have known scenario assumptions.
# All other regressors are extrapolated from the last 36 months' linear trend.
SCENARIOS: dict[str, dict] = {
    "bear": {
        # Continued cedi depreciation; higher inflation; lower commodity prices
        "exchange_rate_ghs_usd": 20.0,
        "inflation_cpi_pct":     31.0,
        "gold_price_usd":      1_900.0,
        "cocoa_price_usd":     4_000.0,
    },
    "base": {
        # Gradual stabilisation; mid cocoa revenues; gold near current levels
        "exchange_rate_ghs_usd": 15.0,
        "inflation_cpi_pct":     20.0,
        "gold_price_usd":      2_250.0,
        "cocoa_price_usd":     5_500.0,
    },
    "bull": {
        # Cedi recovery on cocoa windfall; gold rally; falling inflation
        "exchange_rate_ghs_usd": 12.0,
        "inflation_cpi_pct":     14.0,
        "gold_price_usd":      2_500.0,
        "cocoa_price_usd":     6_500.0,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["ds"])


def fit_scaler(df: pd.DataFrame, regressors: list[str] = REGRESSORS) -> StandardScaler:
    """Fit a StandardScaler on the provided rows."""
    scaler = StandardScaler()
    scaler.fit(df[regressors])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler,
                 regressors: list[str] = REGRESSORS) -> pd.DataFrame:
    """Return a copy of df with regressors standardised."""
    out = df.copy()
    out[regressors] = scaler.transform(out[regressors])
    return out


def build_model() -> Prophet:
    m = Prophet(
        changepoints=CHANGEPOINTS,
        changepoint_prior_scale=0.5,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,          # 90 % credible interval on forecasts
    )
    for reg in REGRESSORS:
        m.add_regressor(reg)
    return m


def _linear_extrap(series: np.ndarray, n: int) -> np.ndarray:
    """Project a 1-D array n steps forward using a fitted OLS line."""
    t = np.arange(len(series))
    slope, intercept = np.polyfit(t, series, 1)
    t_future = np.arange(len(series), len(series) + n)
    return slope * t_future + intercept


def make_future_df(df_orig: pd.DataFrame, df_scaled: pd.DataFrame,
                   scaler: StandardScaler, scenario: dict,
                   n_months: int = FORECAST_MONTHS) -> pd.DataFrame:
    """
    Build a full future DataFrame for m.predict():
      rows 0 … 179  : historical period with scaled regressors
      rows 180 … +n : forecast period with scenario-derived scaled regressors
    """
    last_date   = df_orig["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months,
        freq="MS",
    )

    # ── generate raw future regressor values ──────────────────────────────────
    recent = df_orig.tail(36)   # 3-year window for trend extrapolation
    raw_future = pd.DataFrame({"ds": future_dates})

    for col in REGRESSORS:
        if col in scenario:
            # Constant scenario assumption for the whole forecast window
            raw_future[col] = float(scenario[col])

        elif col == "cpi_index":
            # Compound monthly from the last known CPI using the scenario rate
            annual_inf   = scenario.get("inflation_cpi_pct",
                                        df_orig["inflation_cpi_pct"].iloc[-12:].mean())
            monthly_rate = (1 + annual_inf / 100) ** (1 / 12) - 1
            cpi0         = df_orig["cpi_index"].iloc[-1]
            raw_future[col] = [cpi0 * (1 + monthly_rate) ** (i + 1)
                                for i in range(n_months)]

        else:
            # Linear trend extrapolation from the recent window
            raw_future[col] = _linear_extrap(recent[col].values, n_months)

    # ── scale future regressors using the fitted scaler ───────────────────────
    raw_future[REGRESSORS] = scaler.transform(raw_future[REGRESSORS])

    # ── concatenate historical + future ───────────────────────────────────────
    hist = df_scaled[["ds"] + REGRESSORS]
    full = pd.concat(
        [hist, raw_future[["ds"] + REGRESSORS]],
        ignore_index=True,
    )
    return full


def evaluate(model: Prophet, df_scaled: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate a trained model on the held-out test period.
    Returns a metrics dict and a DataFrame with predictions vs actuals.
    """
    test = df_scaled[df_scaled["ds"] >= TEST_START][["ds", "y"] + REGRESSORS].copy()
    fc   = model.predict(test)

    y_true = test["y"].values
    y_pred = fc["yhat"].values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    result_df = test[["ds", "y"]].copy().reset_index(drop=True)
    result_df["yhat"]       = fc["yhat"].values
    result_df["yhat_lower"] = fc["yhat_lower"].values
    result_df["yhat_upper"] = fc["yhat_upper"].values
    result_df["residual"]   = result_df["y"] - result_df["yhat"]

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}, result_df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(MODELS_DIR,    exist_ok=True)
    os.makedirs(FORECASTS_DIR, exist_ok=True)

    sep = "─" * 50
    print(f"\n  AHPI Mid-Market · Prophet Training\n  {sep}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_data()
    print(f"\n  Data  : {len(df)} rows  "
          f"({df['ds'].min().strftime('%Y-%m')} → {df['ds'].max().strftime('%Y-%m')})")
    print(f"  y     : min={df['y'].min():.1f}  mean={df['y'].mean():.1f}  "
          f"max={df['y'].max():.1f}")
    print(f"  Regressors ({len(REGRESSORS)}): {', '.join(REGRESSORS)}")

    # ── 2. Scaler — fit on TRAINING rows only to prevent leakage ─────────────
    df_train_raw = df[df["ds"] <= TRAIN_END]
    scaler       = fit_scaler(df_train_raw)

    # Scale the full dataset for evaluation model
    df_scaled     = apply_scaler(df, scaler)
    df_train_sc   = df_scaled[df_scaled["ds"] <= TRAIN_END]

    # ── 3. Train evaluation model (2010-2022) ─────────────────────────────────
    print(f"\n  {sep}")
    print(f"  [1/4] Training evaluation model  "
          f"(2010-01 → {TRAIN_END[:7]}, n={len(df_train_sc)})")

    m_eval = build_model()
    m_eval.fit(df_train_sc[["ds", "y"] + REGRESSORS])

    # ── 4. Test-set evaluation (2023-2024) ────────────────────────────────────
    print(f"  [2/4] Evaluating on test set  ({TEST_START[:7]} → 2024-12, n={len(df) - len(df_train_sc)})")
    metrics, eval_df = evaluate(m_eval, df_scaled)

    print(f"\n        MAE  = {metrics['MAE']:.2f}  index points")
    print(f"        RMSE = {metrics['RMSE']:.2f}  index points")
    print(f"        MAPE = {metrics['MAPE']:.1f}%")

    eval_path = os.path.join(FORECASTS_DIR, "ahpi_test_eval.csv")
    eval_df.to_csv(eval_path, index=False)
    print(f"\n        Saved → forecasts/ahpi_test_eval.csv")

    # ── 5. Cross-validation (optional — uses ~2-3 min) ────────────────────────
    print(f"\n  {sep}")
    print(f"  [3/4] Prophet cross-validation  (initial=3y, period=6m, horizon=12m)")
    try:
        df_cv = cross_validation(
            m_eval,
            initial="1095 days",   # 3 years initial training window
            period="182 days",     # expand every 6 months
            horizon="365 days",    # evaluate 12 months ahead
            parallel="processes",
        )
        pm = performance_metrics(df_cv)
        print(f"\n        Horizon-averaged metrics (cross-validation):")
        print(f"        MAE  = {pm['mae'].mean():.2f}")
        print(f"        RMSE = {pm['rmse'].mean():.2f}")
        print(f"        MAPE = {pm['mape'].mean() * 100:.1f}%")
        cv_path = os.path.join(FORECASTS_DIR, "ahpi_cv_metrics.csv")
        pm.to_csv(cv_path, index=False)
        print(f"\n        Saved → forecasts/ahpi_cv_metrics.csv")
    except Exception as e:
        print(f"        Cross-validation skipped: {e}")

    # ── 6. Production model — refit on full dataset (2010-2024) ───────────────
    print(f"\n  {sep}")
    print(f"  [4/4] Fitting production model on full dataset (n={len(df)})")

    # Re-fit scaler on full data for maximum representativeness in forecasting
    scaler_full = fit_scaler(df)
    df_scaled_full = apply_scaler(df, scaler_full)

    m_prod = build_model()
    m_prod.fit(df_scaled_full[["ds", "y"] + REGRESSORS])

    # ── 7. Scenario forecasts ─────────────────────────────────────────────────
    print(f"\n        Forecasting {FORECAST_MONTHS} months (2025-01 → 2026-12):\n")
    for name, scenario in SCENARIOS.items():
        future = make_future_df(df, df_scaled_full, scaler_full, scenario)
        fc     = m_prod.predict(future)

        # Retain only the forecast period (after the last observed date)
        fc_out = fc[fc["ds"] > df["ds"].max()][
            ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        ].copy().reset_index(drop=True)

        # Round for readability
        for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
            fc_out[col] = fc_out[col].round(2)

        out_path = os.path.join(FORECASTS_DIR, f"ahpi_forecast_{name}.csv")
        fc_out.to_csv(out_path, index=False)

        dec26 = fc_out.iloc[-1]
        print(f"        {name.upper():5s}  Dec 2026 AHPI: {dec26['yhat']:>7.1f}"
              f"  [{dec26['yhat_lower']:.1f} – {dec26['yhat_upper']:.1f}]"
              f"  → forecasts/ahpi_forecast_{name}.csv")

    # ── 8. Persist model + scaler ─────────────────────────────────────────────
    model_path  = os.path.join(MODELS_DIR, "ahpi_prophet_model.json")
    scaler_path = os.path.join(MODELS_DIR, "ahpi_scaler.pkl")

    with open(model_path, "w") as fh:
        fh.write(model_to_json(m_prod))

    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler_full, fh)

    print(f"\n  {sep}")
    print(f"  Model  → models/ahpi_prophet_model.json")
    print(f"  Scaler → models/ahpi_scaler.pkl")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
