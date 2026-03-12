"""
Prime-area AHPI endpoints.

Serves per-location price indices for Accra's 6 premium areas:
East Legon, Cantonments, Airport Residential, Labone/Roman Ridge,
Dzorwulu/Abelenkpe, Trasacco Valley.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from ..auth import AuthUser
from ..config import Settings, get_settings
from ..rate_limiter import sliding_window_rate_limit

router = APIRouter(
    prefix="/prime",
    tags=["Prime Areas"],
    dependencies=[Depends(sliding_window_rate_limit)],
)

PRIME_AREAS = [
    "East Legon",
    "Cantonments",
    "Airport Residential",
    "Labone/Roman Ridge",
    "Dzorwulu/Abelenkpe",
    "Trasacco Valley",
]


def _data_path(settings: Settings) -> Path:
    base = Path(__file__).resolve().parents[3]
    return base / "data" / "accra_prime_prices.csv"


@lru_cache(maxsize=1)
def _load_prime(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ds"])
    df.sort_values(["area", "ds"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _get_df(settings: Annotated[Settings, Depends(get_settings)]) -> pd.DataFrame:
    path = _data_path(settings)
    if not path.exists():
        raise HTTPException(status_code=503, detail="Prime areas data unavailable")
    return _load_prime(str(path))


@router.get("")
async def list_prime_areas(_user: AuthUser):
    """Return the list of available prime areas."""
    return {"prime_areas": PRIME_AREAS}


@router.get("/index")
async def get_prime_index(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
    area: str | None = None,
):
    """
    Return prime-area AHPI time series.

    Pass ?area=East+Legon to filter by one area, or omit for all.
    """
    if area and area not in PRIME_AREAS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown area '{area}'. Available: {PRIME_AREAS}",
        )
    filtered = df[df["area"] == area] if area else df
    return {
        "data": filtered.assign(ds=filtered["ds"].dt.strftime("%Y-%m")).to_dict(
            orient="records"
        )
    }


@router.get("/summary")
async def get_prime_summary(
    _user: AuthUser,
    df: Annotated[pd.DataFrame, Depends(_get_df)],
):
    """Return latest AHPI value and % change for each prime area."""
    results = []
    for area, grp in df.groupby("area"):
        grp = grp.sort_values("ds")
        start_val = float(grp.iloc[0]["y"])
        end_val = float(grp.iloc[-1]["y"])
        pct = ((end_val - start_val) / start_val) * 100
        results.append(
            {
                "area": area,
                "latest_ahpi": round(end_val, 2),
                "start_ahpi": round(start_val, 2),
                "pct_change": round(pct, 1),
                "latest_date": grp.iloc[-1]["ds"].strftime("%Y-%m"),
            }
        )
    return {"summary": sorted(results, key=lambda x: x["latest_ahpi"], reverse=True)}
