#!/usr/bin/env python3
"""
AHPI Prime Areas · Facebook Prophet Training (per-area models)
==============================================================
Trains one Prophet model per prime area (6 areas) using the same
macroeconomic regressors as the mid-market model.  Each model is
evaluated on a held-out test set (2023-2024) and used to generate
24-month scenario forecasts (2025-2026).

Prime-area context
------------------
Prime locations (East Legon, Cantonments, Airport Residential,
Labone/Roman Ridge, Dzorwulu/Abelenkpe, Trasacco Valley) are USD-indexed:
sale prices are quoted in dollars, so their AHPI captures both USD capital
appreciation and GHS depreciation.  As a result, exchange_rate_ghs_usd is
the single most powerful regressor.

Outputs (per area, slug = snake_case area name)
-----------------------------------------------
  models/prime_prophet_{slug}.json         — serialised production model
  models/prime_scaler.pkl                  — shared StandardScaler (regressors
                                             are identical national macro data)
  forecasts/prime_test_eval_{slug}.csv     — test-set yhat vs actuals
  forecasts/prime_forecast_{scen}_{slug}.csv — Bear/Base/Bull forecasts
  forecasts/prime_test_summary.csv         — cross-area accuracy comparison

Usage
-----
  python ahpi_prime_prophet.py
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA     = os.path.join(BASE_DIR, "data", "accra_home_price_index.csv")
PRIME_DATA    = os.path.join(BASE_DIR, "data", "accra_prime_prices.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
FORECASTS_DIR = os.path.join(BASE_DIR, "forecasts")

# ── prime area definitions ─────────────────────────────────────────────────────
AREAS = [
    "East Legon",
    "Cantonments",
    "Airport Residential",
    "Labone / Roman Ridge",
    "Dzorwulu / Abelenkpe",
    "Trasacco Valley",
]

AREA_SLUGS: dict[str, str] = {
    "East Legon":           "east_legon",
    "Cantonments":          "cantonments",
    "Airport Residential":  "airport_residential",
    "Labone / Roman Ridge": "labone_roman_ridge",
    "Dzorwulu / Abelenkpe": "dzorwulu_abelenkpe",
    "Trasacco Valley":      "trasacco_valley",
}

# ── regressors — same national macro set as mid-market ────────────────────────
REGRESSORS = [
    "exchange_rate_ghs_usd",   # dominant driver for USD-priced markets
    "cpi_index",               # accumulated inflation
    "urban_pop_pct",           # structural demand
    "broad_money_pct_gdp",     # monetary conditions
    "gold_price_usd",          # fiscal / remittance proxy
    "cocoa_price_usd",         # export revenue proxy
]

# ── model configuration ───────────────────────────────────────────────────────
CHANGEPOINTS    = ["2014-01-01", "2022-01-01"]   # 1st cedi crisis; debt crisis
TRAIN_END       = "2022-12-01"
TEST_START      = "2023-01-01"
FORECAST_MONTHS = 24                              # Jan 2025 → Dec 2026

# ── forward economic scenarios ─────────────────────────────────────────────────
SCENARIOS: dict[str, dict] = {
    "bear": {
        "exchange_rate_ghs_usd": 20.0,
        "inflation_cpi_pct":     31.0,
        "gold_price_usd":      1_900.0,
        "cocoa_price_usd":     4_000.0,
    },
    "base": {
        "exchange_rate_ghs_usd": 15.0,
        "inflation_cpi_pct":     20.0,
        "gold_price_usd":      2_250.0,
        "cocoa_price_usd":     5_500.0,
    },
    "bull": {
        "exchange_rate_ghs_usd": 12.0,
        "inflation_cpi_pct":     14.0,
        "gold_price_usd":      2_500.0,
        "cocoa_price_usd":     6_500.0,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (macro_df, prime_df).
    prime_df has macro columns joined in so each area row has all regressors.
    """
    macro = pd.read_csv(MAIN_DATA, parse_dates=["ds"])
    prime = pd.read_csv(PRIME_DATA, parse_dates=["ds"])

    macro_cols = ["ds"] + REGRESSORS + ["inflation_cpi_pct"]
    prime = prime.merge(macro[macro_cols], on="ds", how="left")
    return macro, prime


def area_df(prime: pd.DataFrame, area: str) -> pd.DataFrame:
    """Return the single-area slice with the columns Prophet needs."""
    d = prime[prime["district"] == area][["ds", "y"] + REGRESSORS].copy()
    return d.reset_index(drop=True)


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the regressor columns of df."""
    scaler = StandardScaler()
    scaler.fit(df[REGRESSORS])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[REGRESSORS] = scaler.transform(out[REGRESSORS])
    return out


def build_model() -> Prophet:
    m = Prophet(
        changepoints=CHANGEPOINTS,
        changepoint_prior_scale=0.5,
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.90,
    )
    for reg in REGRESSORS:
        m.add_regressor(reg)
    return m


def _linear_extrap(series: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(len(series))
    slope, intercept = np.polyfit(t, series, 1)
    t_future = np.arange(len(series), len(series) + n)
    return slope * t_future + intercept


def make_future_df(df_orig: pd.DataFrame, df_scaled: pd.DataFrame,
                   scaler: StandardScaler, scenario: dict,
                   n_months: int = FORECAST_MONTHS) -> pd.DataFrame:
    """Build historical + forecast DataFrame for m.predict()."""
    last_date    = df_orig["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months,
        freq="MS",
    )
    recent = df_orig.tail(36)
    raw_future = pd.DataFrame({"ds": future_dates})

    for col in REGRESSORS:
        if col in scenario:
            raw_future[col] = float(scenario[col])
        elif col == "cpi_index":
            annual_inf   = scenario.get("inflation_cpi_pct",
                                        df_orig["inflation_cpi_pct"].iloc[-12:].mean()
                                        if "inflation_cpi_pct" in df_orig.columns
                                        else 20.0)
            monthly_rate = (1 + annual_inf / 100) ** (1 / 12) - 1
            cpi0         = df_orig["cpi_index"].iloc[-1]
            raw_future[col] = [cpi0 * (1 + monthly_rate) ** (i + 1)
                                for i in range(n_months)]
        else:
            raw_future[col] = _linear_extrap(recent[col].values, n_months)

    raw_future[REGRESSORS] = scaler.transform(raw_future[REGRESSORS])

    hist = df_scaled[["ds"] + REGRESSORS]
    full = pd.concat([hist, raw_future[["ds"] + REGRESSORS]], ignore_index=True)
    return full


def evaluate_area(model: Prophet, df_scaled: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Test-set evaluation for a single area model."""
    test = df_scaled[df_scaled["ds"] >= TEST_START][["ds", "y"] + REGRESSORS].copy()
    fc   = model.predict(test)

    y_true = test["y"].values
    y_pred = fc["yhat"].values

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    result = test[["ds", "y"]].copy().reset_index(drop=True)
    result["yhat"]       = fc["yhat"].values
    result["yhat_lower"] = fc["yhat_lower"].values
    result["yhat_upper"] = fc["yhat_upper"].values
    result["residual"]   = result["y"] - result["yhat"]

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}, result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(MODELS_DIR,    exist_ok=True)
    os.makedirs(FORECASTS_DIR, exist_ok=True)

    sep = "─" * 55
    print(f"\n  AHPI Prime Areas · Prophet Training (per-area)\n  {sep}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    macro, prime = load_data()
    print(f"\n  Main dataset : {len(macro)} rows  "
          f"({macro['ds'].min().strftime('%Y-%m')} → {macro['ds'].max().strftime('%Y-%m')})")
    print(f"  Prime dataset: {len(prime)} rows  "
          f"({len(AREAS)} areas × {len(prime) // len(AREAS)} months)")
    print(f"  Regressors ({len(REGRESSORS)}): {', '.join(REGRESSORS)}")

    # ── 2. Shared scaler (regressors are national macro — same across all areas)
    # Fit evaluation scaler on training rows (from any area, macro is identical)
    sample_area   = area_df(prime, AREAS[0])
    train_raw     = sample_area[sample_area["ds"] <= TRAIN_END]
    scaler_eval   = fit_scaler(train_raw)

    # Production scaler: fit on full date range
    scaler_full   = fit_scaler(sample_area)

    # Persist shared scaler
    scaler_path = os.path.join(MODELS_DIR, "prime_scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler_full, fh)

    summary_rows: list[dict] = []

    # ── 3. Per-area training loop ─────────────────────────────────────────────
    for area in AREAS:
        slug = AREA_SLUGS[area]
        print(f"\n  {sep}")
        print(f"  Area: {area}  (slug: {slug})")

        df_area = area_df(prime, area)
        # Also keep inflation_cpi_pct for cpi compound logic in make_future_df
        df_area_with_infl = prime[prime["district"] == area][
            ["ds", "y"] + REGRESSORS + ["inflation_cpi_pct"]
        ].copy().reset_index(drop=True)

        train_n = (df_area["ds"] <= TRAIN_END).sum()
        test_n  = (df_area["ds"] >= TEST_START).sum()
        print(f"    y: min={df_area['y'].min():.1f}  "
              f"mean={df_area['y'].mean():.1f}  max={df_area['y'].max():.1f}")

        # ── 3a. Evaluation model (2010-2022) ──────────────────────────────────
        df_scaled_eval = apply_scaler(df_area, scaler_eval)
        df_train_sc    = df_scaled_eval[df_scaled_eval["ds"] <= TRAIN_END]

        print(f"    [1/3] Training evaluation model  "
              f"(n_train={train_n}, n_test={test_n})")
        m_eval = build_model()
        m_eval.fit(df_train_sc[["ds", "y"] + REGRESSORS])

        # ── 3b. Test-set evaluation ────────────────────────────────────────────
        metrics, eval_df = evaluate_area(m_eval, df_scaled_eval)
        print(f"          MAE={metrics['MAE']:.2f}  "
              f"RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.1f}%")

        eval_path = os.path.join(FORECASTS_DIR, f"prime_test_eval_{slug}.csv")
        eval_df.to_csv(eval_path, index=False)

        summary_rows.append({
            "area": area, "slug": slug,
            "mae": round(metrics["MAE"], 2),
            "rmse": round(metrics["RMSE"], 2),
            "mape_pct": round(metrics["MAPE"], 1),
        })

        # ── 3c. Production model (full 2010-2024) ─────────────────────────────
        print(f"    [2/3] Fitting production model (n={len(df_area)})")
        df_scaled_full = apply_scaler(df_area, scaler_full)
        m_prod = build_model()
        m_prod.fit(df_scaled_full[["ds", "y"] + REGRESSORS])

        # ── 3d. Scenario forecasts ─────────────────────────────────────────────
        print(f"    [3/3] Forecasting {FORECAST_MONTHS} months → 2026-12:")
        for sc_name, scenario in SCENARIOS.items():
            future = make_future_df(df_area_with_infl, df_scaled_full,
                                    scaler_full, scenario)
            fc = m_prod.predict(future)

            fc_out = fc[fc["ds"] > df_area["ds"].max()][
                ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
            ].copy().reset_index(drop=True)
            for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
                fc_out[col] = fc_out[col].round(2)

            out_path = os.path.join(FORECASTS_DIR,
                                    f"prime_forecast_{sc_name}_{slug}.csv")
            fc_out.to_csv(out_path, index=False)

            dec26 = fc_out.iloc[-1]
            print(f"          {sc_name.upper():5s}  Dec 2026: {dec26['yhat']:>7.1f}"
                  f"  [{dec26['yhat_lower']:.1f} – {dec26['yhat_upper']:.1f}]")

        # ── 3e. Persist production model ──────────────────────────────────────
        model_path = os.path.join(MODELS_DIR, f"prime_prophet_{slug}.json")
        with open(model_path, "w") as fh:
            fh.write(model_to_json(m_prod))

    # ── 4. Cross-area summary ─────────────────────────────────────────────────
    print(f"\n  {sep}")
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(FORECASTS_DIR, "prime_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n  Test-set accuracy summary (2023–2024, n=24 per area):\n")
    print(f"  {'Area':<26}  {'MAE':>7}  {'RMSE':>7}  {'MAPE':>7}")
    print(f"  {'─'*26}  {'─'*7}  {'─'*7}  {'─'*7}")
    for r in summary_rows:
        print(f"  {r['area']:<26}  {r['mae']:>7.2f}  {r['rmse']:>7.2f}  {r['mape_pct']:>6.1f}%")

    avg = summary_df[["mae", "rmse", "mape_pct"]].mean()
    print(f"  {'─'*26}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'Average':<26}  {avg['mae']:>7.2f}  {avg['rmse']:>7.2f}  {avg['mape_pct']:>6.1f}%")

    print(f"\n  Saved → models/prime_prophet_{{slug}}.json  (×{len(AREAS)})")
    print(f"  Saved → models/prime_scaler.pkl")
    print(f"  Saved → forecasts/prime_test_eval_{{slug}}.csv  (×{len(AREAS)})")
    print(f"  Saved → forecasts/prime_forecast_{{scen}}_{{slug}}.csv  "
          f"(×{len(AREAS) * len(SCENARIOS)})")
    print(f"  Saved → forecasts/prime_test_summary.csv\n")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
