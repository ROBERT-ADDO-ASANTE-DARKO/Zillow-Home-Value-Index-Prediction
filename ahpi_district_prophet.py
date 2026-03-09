#!/usr/bin/env python3
"""
AHPI Mid-Market Districts · Facebook Prophet Training (per-district models)
===========================================================================
Trains one Prophet model per mid-market district (5 districts) using the
same macroeconomic regressors as the composite mid-market model.

Mid-market context
------------------
Districts (Spintex Road, Adenta, Tema, Dome, Kasoa) are GHS-denominated
markets.  Prices appreciate through a combination of cedi depreciation,
local inflation, urbanisation, and district-level supply/demand dynamics.
The multiplier applied to the composite AHPI means each district has a
distinct beta to macro drivers.

Outputs (per district, slug = snake_case district name)
-------------------------------------------------------
  models/district_prophet_{slug}.json        — serialised production model
  models/district_scaler.pkl                 — shared StandardScaler
  forecasts/district_test_eval_{slug}.csv    — test-set yhat vs actuals
  forecasts/district_forecast_{scen}_{slug}.csv — Bear/Base/Bull forecasts
  forecasts/district_test_summary.csv        — cross-district accuracy table

Usage
-----
  python ahpi_district_prophet.py
"""

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
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MAIN_DATA      = os.path.join(BASE_DIR, "data", "accra_home_price_index.csv")
DISTRICT_DATA  = os.path.join(BASE_DIR, "data", "accra_district_prices.csv")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
FORECASTS_DIR  = os.path.join(BASE_DIR, "forecasts")

# ── district definitions ───────────────────────────────────────────────────────
DISTRICTS = ["Spintex Road", "Adenta", "Tema", "Dome", "Kasoa"]

DISTRICT_SLUGS: dict[str, str] = {
    "Spintex Road": "spintex_road",
    "Adenta":       "adenta",
    "Tema":         "tema",
    "Dome":         "dome",
    "Kasoa":        "kasoa",
}

# ── regressors — same national macro set as composite mid-market model ─────────
REGRESSORS = [
    "exchange_rate_ghs_usd",
    "cpi_index",
    "urban_pop_pct",
    "broad_money_pct_gdp",
    "gold_price_usd",
    "cocoa_price_usd",
]

# ── model configuration ───────────────────────────────────────────────────────
CHANGEPOINTS    = ["2014-01-01", "2022-01-01"]
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
    macro    = pd.read_csv(MAIN_DATA, parse_dates=["ds"])
    district = pd.read_csv(DISTRICT_DATA, parse_dates=["ds"])
    macro_cols = ["ds"] + REGRESSORS + ["inflation_cpi_pct"]
    district   = district.merge(macro[macro_cols], on="ds", how="left")
    return macro, district


def district_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["district"] == name][["ds", "y"] + REGRESSORS].copy().reset_index(drop=True)


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    s = StandardScaler()
    s.fit(df[REGRESSORS])
    return s


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
    return slope * np.arange(len(series), len(series) + n) + intercept


def make_future_df(df_orig: pd.DataFrame, df_scaled: pd.DataFrame,
                   scaler: StandardScaler, scenario: dict,
                   n_months: int = FORECAST_MONTHS) -> pd.DataFrame:
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
            annual_inf   = scenario.get(
                "inflation_cpi_pct",
                df_orig["inflation_cpi_pct"].iloc[-12:].mean()
                if "inflation_cpi_pct" in df_orig.columns else 20.0,
            )
            monthly_rate = (1 + annual_inf / 100) ** (1 / 12) - 1
            cpi0         = df_orig["cpi_index"].iloc[-1]
            raw_future[col] = [cpi0 * (1 + monthly_rate) ** (i + 1)
                                for i in range(n_months)]
        else:
            raw_future[col] = _linear_extrap(recent[col].values, n_months)

    raw_future[REGRESSORS] = scaler.transform(raw_future[REGRESSORS])
    hist = df_scaled[["ds"] + REGRESSORS]
    return pd.concat([hist, raw_future[["ds"] + REGRESSORS]], ignore_index=True)


def evaluate_district(model: Prophet, df_scaled: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    test = df_scaled[df_scaled["ds"] >= TEST_START][["ds", "y"] + REGRESSORS].copy()
    fc   = model.predict(test)
    y_true, y_pred = test["y"].values, fc["yhat"].values
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    result              = test[["ds", "y"]].copy().reset_index(drop=True)
    result["yhat"]      = fc["yhat"].values
    result["yhat_lower"] = fc["yhat_lower"].values
    result["yhat_upper"] = fc["yhat_upper"].values
    result["residual"]  = result["y"] - result["yhat"]
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}, result


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(MODELS_DIR,    exist_ok=True)
    os.makedirs(FORECASTS_DIR, exist_ok=True)

    sep = "─" * 55
    print(f"\n  AHPI Mid-Market Districts · Prophet Training (per-district)\n  {sep}")

    macro, dist_all = load_data()
    print(f"\n  Main dataset    : {len(macro)} rows  "
          f"({macro['ds'].min().strftime('%Y-%m')} → {macro['ds'].max().strftime('%Y-%m')})")
    print(f"  District dataset: {len(dist_all)} rows  "
          f"({len(DISTRICTS)} districts × {len(dist_all) // len(DISTRICTS)} months)")
    print(f"  Regressors ({len(REGRESSORS)}): {', '.join(REGRESSORS)}")

    # Shared scalers (regressors are identical national macro data)
    sample      = district_df(dist_all, DISTRICTS[0])
    scaler_eval = fit_scaler(sample[sample["ds"] <= TRAIN_END])
    scaler_full = fit_scaler(sample)

    scaler_path = os.path.join(MODELS_DIR, "district_scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler_full, fh)

    summary_rows: list[dict] = []

    for district in DISTRICTS:
        slug = DISTRICT_SLUGS[district]
        print(f"\n  {sep}")
        print(f"  District: {district}  (slug: {slug})")

        df_d = district_df(dist_all, district)
        df_d_infl = dist_all[dist_all["district"] == district][
            ["ds", "y"] + REGRESSORS + ["inflation_cpi_pct"]
        ].copy().reset_index(drop=True)

        train_n = (df_d["ds"] <= TRAIN_END).sum()
        test_n  = (df_d["ds"] >= TEST_START).sum()
        print(f"    y: min={df_d['y'].min():.1f}  "
              f"mean={df_d['y'].mean():.1f}  max={df_d['y'].max():.1f}")

        # Evaluation model (2010-2022)
        df_sc_eval = apply_scaler(df_d, scaler_eval)
        df_train   = df_sc_eval[df_sc_eval["ds"] <= TRAIN_END]
        print(f"    [1/3] Training evaluation model  "
              f"(n_train={train_n}, n_test={test_n})")
        m_eval = build_model()
        m_eval.fit(df_train[["ds", "y"] + REGRESSORS])

        # Test-set evaluation
        metrics, eval_df = evaluate_district(m_eval, df_sc_eval)
        print(f"          MAE={metrics['MAE']:.2f}  "
              f"RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE']:.1f}%")
        eval_df.to_csv(
            os.path.join(FORECASTS_DIR, f"district_test_eval_{slug}.csv"), index=False)
        summary_rows.append({
            "district": district, "slug": slug,
            "mae": round(metrics["MAE"], 2),
            "rmse": round(metrics["RMSE"], 2),
            "mape_pct": round(metrics["MAPE"], 1),
        })

        # Production model (full 2010-2024)
        print(f"    [2/3] Fitting production model (n={len(df_d)})")
        df_sc_full = apply_scaler(df_d, scaler_full)
        m_prod     = build_model()
        m_prod.fit(df_sc_full[["ds", "y"] + REGRESSORS])

        # Scenario forecasts
        print(f"    [3/3] Forecasting {FORECAST_MONTHS} months → 2026-12:")
        for sc_name, scenario in SCENARIOS.items():
            future = make_future_df(df_d_infl, df_sc_full, scaler_full, scenario)
            fc     = m_prod.predict(future)
            fc_out = fc[fc["ds"] > df_d["ds"].max()][
                ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
            ].copy().reset_index(drop=True)
            for col in ["yhat", "yhat_lower", "yhat_upper", "trend"]:
                fc_out[col] = fc_out[col].round(2)
            fc_out.to_csv(
                os.path.join(FORECASTS_DIR, f"district_forecast_{sc_name}_{slug}.csv"),
                index=False,
            )
            dec26 = fc_out.iloc[-1]
            print(f"          {sc_name.upper():5s}  Dec 2026: {dec26['yhat']:>7.1f}"
                  f"  [{dec26['yhat_lower']:.1f} – {dec26['yhat_upper']:.1f}]")

        # Persist production model
        with open(os.path.join(MODELS_DIR, f"district_prophet_{slug}.json"), "w") as fh:
            fh.write(model_to_json(m_prod))

    # Summary
    print(f"\n  {sep}")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(FORECASTS_DIR, "district_test_summary.csv"), index=False)

    print(f"\n  Test-set accuracy summary (2023–2024, n=24 per district):\n")
    print(f"  {'District':<16}  {'MAE':>7}  {'RMSE':>7}  {'MAPE':>7}")
    print(f"  {'─'*16}  {'─'*7}  {'─'*7}  {'─'*7}")
    for r in summary_rows:
        print(f"  {r['district']:<16}  {r['mae']:>7.2f}  {r['rmse']:>7.2f}  {r['mape_pct']:>6.1f}%")
    avg = summary_df[["mae", "rmse", "mape_pct"]].mean()
    print(f"  {'─'*16}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'Average':<16}  {avg['mae']:>7.2f}  {avg['rmse']:>7.2f}  {avg['mape_pct']:>6.1f}%")

    print(f"\n  Saved → models/district_prophet_{{slug}}.json  (×{len(DISTRICTS)})")
    print(f"  Saved → models/district_scaler.pkl")
    print(f"  Saved → forecasts/district_test_eval_{{slug}}.csv  (×{len(DISTRICTS)})")
    print(f"  Saved → forecasts/district_forecast_{{scen}}_{{slug}}.csv  "
          f"(×{len(DISTRICTS) * len(SCENARIOS)})")
    print(f"  Saved → forecasts/district_test_summary.csv\n")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
