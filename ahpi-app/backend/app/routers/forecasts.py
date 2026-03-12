"""
Forecast endpoints.

Serves pre-computed Prophet forecast outputs (bear / base / bull scenarios)
for the aggregate AHPI, each district, and each prime area.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..auth import AuthUser
from ..config import Settings, get_settings
from ..rate_limiter import sliding_window_rate_limit

router = APIRouter(
    prefix="/forecasts",
    tags=["Forecasts"],
    dependencies=[Depends(sliding_window_rate_limit)],
)

Scenario = Literal["bear", "base", "bull"]
SCENARIOS: list[Scenario] = ["bear", "base", "bull"]

DISTRICTS = ["Spintex Road", "Adenta", "Tema", "Dome", "Kasoa"]
PRIME_AREAS = [
    "East Legon",
    "Cantonments",
    "Airport Residential",
    "Labone/Roman Ridge",
    "Dzorwulu/Abelenkpe",
    "Trasacco Valley",
]

# Map area names to safe file-system slugs (matching existing forecast filenames)
_DISTRICT_SLUGS = {
    "Spintex Road": "spintex_road",
    "Adenta": "adenta",
    "Tema": "tema",
    "Dome": "dome",
    "Kasoa": "kasoa",
}
_PRIME_SLUGS = {
    "East Legon": "east_legon",
    "Cantonments": "cantonments",
    "Airport Residential": "airport_residential",
    "Labone/Roman Ridge": "labone_roman_ridge",
    "Dzorwulu/Abelenkpe": "dzorwulu_abelenkpe",
    "Trasacco Valley": "trasacco_valley",
}


def _forecasts_dir(settings: Settings) -> Path:
    base = Path(__file__).resolve().parents[3]
    return base / "forecasts"


def _forecast_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper"] if c in df.columns]


@lru_cache(maxsize=64)
def _load_forecast(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values("ds", inplace=True)
    return df


def _get_settings_dep(settings: Annotated[Settings, Depends(get_settings)]) -> Settings:
    return settings


# ---------------------------------------------------------------------------
# Aggregate AHPI forecasts
# ---------------------------------------------------------------------------

@router.get("/ahpi/{scenario}")
async def get_ahpi_forecast(
    scenario: Scenario,
    _user: AuthUser,
    settings: Annotated[Settings, Depends(_get_settings_dep)],
):
    """
    Return aggregate AHPI forecast for a given scenario (bear / base / bull).
    Includes yhat, yhat_lower, yhat_upper columns.
    """
    fdir = _forecasts_dir(settings)
    path = fdir / f"ahpi_forecast_{scenario}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast file not found: {path.name}")
    df = _load_forecast(str(path))
    cols = _forecast_cols(df)
    return {
        "scenario": scenario,
        "data": df[cols].assign(ds=df["ds"].dt.strftime("%Y-%m")).to_dict(orient="records"),
    }


@router.get("/ahpi/eval/metrics")
async def get_ahpi_eval(
    _user: AuthUser,
    settings: Annotated[Settings, Depends(_get_settings_dep)],
):
    """Return test-set evaluation metrics (MAE, RMSE, MAPE) for the aggregate AHPI model."""
    fdir = _forecasts_dir(settings)
    eval_path = fdir / "ahpi_test_eval.csv"
    cv_path = fdir / "ahpi_cv_metrics.csv"

    result: dict = {}
    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)
        result["test_eval"] = eval_df.to_dict(orient="records")
    if cv_path.exists():
        cv_df = pd.read_csv(cv_path)
        result["cv_metrics"] = cv_df.to_dict(orient="records")
    if not result:
        raise HTTPException(status_code=404, detail="Evaluation data not found")
    return result


# ---------------------------------------------------------------------------
# District forecasts
# ---------------------------------------------------------------------------

@router.get("/districts/{district}/{scenario}")
async def get_district_forecast(
    district: str,
    scenario: Scenario,
    _user: AuthUser,
    settings: Annotated[Settings, Depends(_get_settings_dep)],
):
    """Return district-level forecast for a given district and scenario."""
    if district not in DISTRICTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown district '{district}'. Available: {DISTRICTS}",
        )
    slug = _DISTRICT_SLUGS[district]
    fdir = _forecasts_dir(settings)
    path = fdir / f"district_forecast_{slug}_{scenario}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast file not found: {path.name}")
    df = _load_forecast(str(path))
    cols = _forecast_cols(df)
    return {
        "district": district,
        "scenario": scenario,
        "data": df[cols].assign(ds=df["ds"].dt.strftime("%Y-%m")).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Prime area forecasts
# ---------------------------------------------------------------------------

@router.get("/prime/{area}/{scenario}")
async def get_prime_forecast(
    area: str,
    scenario: Scenario,
    _user: AuthUser,
    settings: Annotated[Settings, Depends(_get_settings_dep)],
):
    """Return prime-area forecast for a given area and scenario."""
    if area not in PRIME_AREAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown area '{area}'. Available: {PRIME_AREAS}",
        )
    slug = _PRIME_SLUGS[area]
    fdir = _forecasts_dir(settings)
    path = fdir / f"prime_forecast_{slug}_{scenario}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Forecast file not found: {path.name}")
    df = _load_forecast(str(path))
    cols = _forecast_cols(df)
    return {
        "area": area,
        "scenario": scenario,
        "data": df[cols].assign(ds=df["ds"].dt.strftime("%Y-%m")).to_dict(orient="records"),
    }
