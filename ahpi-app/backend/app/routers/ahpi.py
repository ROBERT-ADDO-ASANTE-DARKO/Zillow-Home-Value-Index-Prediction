"""
AHPI overview endpoints.

Serves the aggregate Accra Home Price Index time series and summary metrics.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..auth import AuthUser
from ..config import Settings, get_settings
from ..rate_limiter import sliding_window_rate_limit

router = APIRouter(
    prefix="/ahpi",
    tags=["AHPI Overview"],
    dependencies=[Depends(sliding_window_rate_limit)],
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _data_path(settings: Settings) -> Path:
    base = Path(__file__).resolve().parents[3]  # repo root
    return base / "data" / "accra_home_price_index.csv"


@lru_cache(maxsize=1)
def _load_ahpi(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _get_df(settings: Annotated[Settings, Depends(get_settings)]) -> pd.DataFrame:
    path = _data_path(settings)
    if not path.exists():
        raise HTTPException(status_code=503, detail="AHPI data unavailable")
    return _load_ahpi(str(path))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/index")
async def get_ahpi_index(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
):
    """Return the full AHPI time series (date + index value)."""
    return {
        "data": df[["ds", "y"]].assign(ds=df["ds"].dt.strftime("%Y-%m")).to_dict(
            orient="records"
        )
    }


@router.get("/macro")
async def get_macro_data(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
    regressor: str | None = None,
):
    """
    Return macro-regressor time series.

    Pass ?regressor=exchange_rate_ghs_usd to retrieve a single column,
    or omit to receive all columns.
    """
    available = [
        "gdp_growth_pct", "cpi_index", "inflation_cpi_pct",
        "exchange_rate_ghs_usd", "lending_rate_pct", "unemployment_pct",
        "urban_pop_pct", "remittances_pct_gdp", "fdi_pct_gdp",
        "broad_money_pct_gdp", "gold_price_usd", "cocoa_price_usd",
        "oil_brent_usd",
    ]
    if regressor and regressor not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown regressor '{regressor}'. Available: {available}",
        )
    cols = ["ds"] + ([regressor] if regressor else available)
    return {
        "regressors": available,
        "data": df[cols].assign(ds=df["ds"].dt.strftime("%Y-%m")).to_dict(
            orient="records"
        ),
    }


@router.get("/summary")
async def get_summary(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
):
    """Return summary statistics and key metrics for the AHPI."""
    latest = df.iloc[-1]
    earliest = df.iloc[0]
    pct_change = ((latest["y"] - earliest["y"]) / earliest["y"]) * 100

    # Exchange-rate driven analysis
    fx_change = (
        (latest["exchange_rate_ghs_usd"] - earliest["exchange_rate_ghs_usd"])
        / earliest["exchange_rate_ghs_usd"]
    ) * 100

    # USD appreciation (price_usd_per_sqm)
    usd_change = (
        (latest["price_usd_per_sqm"] - earliest["price_usd_per_sqm"])
        / earliest["price_usd_per_sqm"]
    ) * 100

    return {
        "period": {
            "start": earliest["ds"].strftime("%Y-%m"),
            "end": latest["ds"].strftime("%Y-%m"),
        },
        "ahpi": {
            "start": round(float(earliest["y"]), 2),
            "end": round(float(latest["y"]), 2),
            "pct_change": round(pct_change, 1),
        },
        "exchange_rate": {
            "start": round(float(earliest["exchange_rate_ghs_usd"]), 2),
            "end": round(float(latest["exchange_rate_ghs_usd"]), 2),
            "pct_change": round(fx_change, 1),
        },
        "price_usd_per_sqm": {
            "start": round(float(earliest["price_usd_per_sqm"]), 0),
            "end": round(float(latest["price_usd_per_sqm"]), 0),
            "pct_change": round(usd_change, 1),
        },
        "observations": len(df),
    }
