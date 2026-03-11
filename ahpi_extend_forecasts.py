#!/usr/bin/env python3
"""
AHPI – Extend Forecast Horizon to 60 Months (Jan 2025 → Dec 2029)
==================================================================
Loads the already-trained Prophet models (no re-training needed) and
generates 60-month scenario forecast CSVs for all 12 models:
  - 1 composite mid-market
  - 6 prime areas
  - 5 mid-market districts

Outputs
-------
  forecasts/ahpi_forecast_{bear,base,bull}.csv          (60 rows)
  forecasts/prime_forecast_{sc}_{slug}.csv              (60 rows × 18)
  forecasts/district_forecast_{sc}_{slug}.csv           (60 rows × 15)

Usage
-----
  python ahpi_extend_forecasts.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from prophet.serialize import model_from_json

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data", "accra_home_price_index.csv")
DISTRICT_PATH = os.path.join(BASE_DIR, "data", "accra_district_prices.csv")
PRIME_PATH    = os.path.join(BASE_DIR, "data", "accra_prime_prices.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
FORECASTS_DIR = os.path.join(BASE_DIR, "forecasts")

REGRESSORS = [
    "exchange_rate_ghs_usd",
    "cpi_index",
    "urban_pop_pct",
    "broad_money_pct_gdp",
    "gold_price_usd",
    "cocoa_price_usd",
]

FORECAST_MONTHS = 60   # Jan 2025 → Dec 2029

SCENARIOS = {
    "bear": {"exchange_rate_ghs_usd": 20.0, "inflation_cpi_pct": 31.0,
             "gold_price_usd": 1_900.0, "cocoa_price_usd": 4_000.0},
    "base": {"exchange_rate_ghs_usd": 15.0, "inflation_cpi_pct": 20.0,
             "gold_price_usd": 2_250.0, "cocoa_price_usd": 5_500.0},
    "bull": {"exchange_rate_ghs_usd": 12.0, "inflation_cpi_pct": 14.0,
             "gold_price_usd": 2_500.0, "cocoa_price_usd": 6_500.0},
}

PRIME_SLUGS = {
    "East Legon": "east_legon", "Cantonments": "cantonments",
    "Airport Residential": "airport_residential",
    "Labone / Roman Ridge": "labone_roman_ridge",
    "Dzorwulu / Abelenkpe": "dzorwulu_abelenkpe",
    "Trasacco Valley": "trasacco_valley",
}

DISTRICT_SLUGS = {
    "Spintex Road": "spintex_road", "Adenta": "adenta",
    "Tema": "tema", "Dome": "dome", "Kasoa": "kasoa",
}


def _linear_extrap(series: np.ndarray, n: int) -> np.ndarray:
    t = np.arange(len(series))
    slope, intercept = np.polyfit(t, series, 1)
    return slope * np.arange(len(series), len(series) + n) + intercept


def make_future_df(df_orig: pd.DataFrame, scaler, scenario: dict,
                   n_months: int = FORECAST_MONTHS) -> pd.DataFrame:
    """Build full future DataFrame for model.predict()."""
    from sklearn.preprocessing import StandardScaler  # noqa
    last_date    = df_orig["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=n_months, freq="MS",
    )
    recent     = df_orig.tail(36)
    raw_future = pd.DataFrame({"ds": future_dates})

    for col in REGRESSORS:
        if col in scenario:
            raw_future[col] = float(scenario[col])
        elif col == "cpi_index":
            annual_inf   = scenario.get("inflation_cpi_pct",
                                        df_orig["inflation_cpi_pct"].iloc[-12:].mean())
            monthly_rate = (1 + annual_inf / 100) ** (1 / 12) - 1
            cpi0         = df_orig["cpi_index"].iloc[-1]
            raw_future[col] = [cpi0 * (1 + monthly_rate) ** (i + 1)
                                for i in range(n_months)]
        else:
            raw_future[col] = _linear_extrap(recent[col].values, n_months)

    # Scale future regressors
    raw_future_scaled = raw_future.copy()
    raw_future_scaled[REGRESSORS] = scaler.transform(raw_future[REGRESSORS])

    # Historical period (already scaled at train time) — reconstruct
    df_hist = df_orig.copy()
    df_hist[REGRESSORS] = scaler.transform(df_orig[REGRESSORS])
    hist = df_hist[["ds"] + REGRESSORS]

    full = pd.concat([hist, raw_future_scaled[["ds"] + REGRESSORS]], ignore_index=True)
    return full


def predict_and_save(model, df_orig: pd.DataFrame, scaler,
                     scenario: dict, out_path: str) -> pd.Series:
    future  = make_future_df(df_orig, scaler, scenario)
    fc      = model.predict(future)
    fc_out  = fc[fc["ds"] > df_orig["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
    ].copy().reset_index(drop=True)
    for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
        fc_out[col] = fc_out[col].round(2)
    fc_out.to_csv(out_path, index=False)
    return fc_out.iloc[-1]


def load_model(path: str):
    with open(path) as f:
        return model_from_json(f.read())


def load_scaler(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    sep = "─" * 54
    print(f"\n  AHPI – Extending Forecasts to {FORECAST_MONTHS} months "
          f"(Jan 2025 → Dec 2029)\n  {sep}")

    df = pd.read_csv(DATA_PATH, parse_dates=["ds"])

    # ── 1. Composite mid-market ───────────────────────────────────────────────
    print(f"\n  Composite mid-market")
    m_comp    = load_model(os.path.join(MODELS_DIR, "ahpi_prophet_model.json"))
    sc_comp   = load_scaler(os.path.join(MODELS_DIR, "ahpi_scaler.pkl"))

    for sc_name, scenario in SCENARIOS.items():
        out = os.path.join(FORECASTS_DIR, f"ahpi_forecast_{sc_name}.csv")
        last = predict_and_save(m_comp, df, sc_comp, scenario, out)
        print(f"    {sc_name.upper():<5}  Dec 2029: {last['yhat']:>8.1f}"
              f"  [{last['yhat_lower']:.1f} – {last['yhat_upper']:.1f}]")

    # ── 2. Prime areas ────────────────────────────────────────────────────────
    print(f"\n  Prime areas  (shared scaler)")
    sc_prime  = load_scaler(os.path.join(MODELS_DIR, "prime_scaler.pkl"))
    df_prime  = pd.read_csv(PRIME_PATH, parse_dates=["ds"])
    # Join macro columns so make_future_df can use them for OLS extrapolation
    df_macro  = df[["ds", "inflation_cpi_pct"] + REGRESSORS]

    for area, slug in PRIME_SLUGS.items():
        m_area = load_model(os.path.join(MODELS_DIR, f"prime_prophet_{slug}.json"))
        # Use macro data from composite df for regressor extrapolation
        df_area = (
            df_prime[df_prime["district"] == area][["ds", "y"]]
            .merge(df_macro, on="ds", how="left")
        )
        print(f"    {area}")
        for sc_name, scenario in SCENARIOS.items():
            out  = os.path.join(FORECASTS_DIR, f"prime_forecast_{sc_name}_{slug}.csv")
            last = predict_and_save(m_area, df_area, sc_prime, scenario, out)
            print(f"      {sc_name.upper():<5}  Dec 2029: {last['yhat']:.1f}")

    # ── 3. Mid-market districts ───────────────────────────────────────────────
    print(f"\n  Mid-market districts  (shared scaler)")
    sc_dist   = load_scaler(os.path.join(MODELS_DIR, "district_scaler.pkl"))
    df_dist   = pd.read_csv(DISTRICT_PATH, parse_dates=["ds"])

    for district, slug in DISTRICT_SLUGS.items():
        m_dist = load_model(os.path.join(MODELS_DIR, f"district_prophet_{slug}.json"))
        df_d = (
            df_dist[df_dist["district"] == district][["ds", "y"]]
            .merge(df_macro, on="ds", how="left")
        )
        print(f"    {district}")
        for sc_name, scenario in SCENARIOS.items():
            out  = os.path.join(FORECASTS_DIR, f"district_forecast_{sc_name}_{slug}.csv")
            last = predict_and_save(m_dist, df_d, sc_dist, scenario, out)
            print(f"      {sc_name.upper():<5}  Dec 2029: {last['yhat']:.1f}")

    print(f"\n  {sep}")
    print(f"  Done. All forecasts extended to Dec 2029.\n")


if __name__ == "__main__":
    main()
